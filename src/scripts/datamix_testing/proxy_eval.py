"""
Proxy Model Evaluation
========================

Lightweight evaluation functions for proxy-scale mixture experiments.
Computes perplexity across 3 domains:

    1. Code       — starcoderdata/python (streaming)
    2. General    — fineweb-edu (streaming)
    3. Reasoning  — finemath-4plus (streaming)

Each domain evaluates on a small number of batches (default 20)
to keep eval fast on single-GPU rented instances.

Reuses the streaming tokenization pattern from validate_domains.py.
"""

import math
import time
import torch
import torch.nn as nn
from torch.amp import autocast
from datasets import load_dataset
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass

from ..tokenizer import tokenizer


# ══════════════════════════════════════════════════════════════
#  Eval Domain Definitions
# ══════════════════════════════════════════════════════════════

@dataclass
class EvalDomain:
    """Configuration for a single evaluation domain."""
    name: str
    key: str
    repo_id: str
    format_fn_name: str
    config_name: Optional[str] = None
    data_dir: Optional[str] = None
    split: str = "train"


EVAL_DOMAINS: List[EvalDomain] = [
    EvalDomain(
        name="Code (Python)",
        key="code",
        repo_id="bigcode/starcoderdata",
        format_fn_name="starcoder",
        data_dir="python",
    ),
    EvalDomain(
        name="General Knowledge",
        key="general",
        repo_id="HuggingFaceTB/smollm-corpus",
        format_fn_name="default",
        config_name="fineweb-edu-dedup",
    ),
    EvalDomain(
        name="Math/Reasoning",
        key="reasoning",
        repo_id="HuggingFaceTB/finemath",
        format_fn_name="finemath",
        config_name="finemath-4plus",
    ),
]


# ══════════════════════════════════════════════════════════════
#  Format Functions (minimal subset for eval)
# ══════════════════════════════════════════════════════════════

def _fmt_default(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text", "")
    return text if text else None


def _fmt_starcoder(row: Dict[str, Any]) -> Optional[str]:
    content = row.get("content", "")
    if not content or len(content) < 100 or len(content) > 100_000:
        return None
    return content


def _fmt_finemath(row: Dict[str, Any]) -> Optional[str]:
    text = row.get("text", "")
    if not text or len(text) < 50:
        return None
    return text


EVAL_FORMAT_FNS: Dict[str, Callable] = {
    "default": _fmt_default,
    "starcoder": _fmt_starcoder,
    "finemath": _fmt_finemath,
}


# ══════════════════════════════════════════════════════════════
#  Streaming Tokenized Batch Generator
# ══════════════════════════════════════════════════════════════

_MAX_ROWS_SCANNED = 200_000
_DOMAIN_TIMEOUT_SECONDS = 90


def _stream_eval_batches(
    domain: EvalDomain,
    batch_size: int = 16,
    context_length: int = 2048,
    max_batches: int = 20,
    timeout_seconds: float = _DOMAIN_TIMEOUT_SECONDS,
):
    """Stream tokenized batches from an evaluation domain.

    Yields tensors of shape ``(batch_size, context_length + 1)``,
    matching the training dataloader format.

    Stops after ``max_batches`` complete batches, after scanning
    ``_MAX_ROWS_SCANNED`` rows, or after ``timeout_seconds``.

    Args:
        domain: EvalDomain configuration.
        batch_size: Samples per batch.
        context_length: Token sequence length.
        max_batches: Maximum batches to yield.
        timeout_seconds: Wall-clock timeout.

    Yields:
        torch.LongTensor of shape ``(batch_size, context_length + 1)``.
    """
    fmt_fn = EVAL_FORMAT_FNS.get(domain.format_fn_name)
    if fmt_fn is None:
        print(f"  [ProxyEval] WARNING: Unknown format '{domain.format_fn_name}' "
              f"for domain '{domain.name}'. Skipping.")
        return

    kwargs = {}
    if domain.data_dir is not None:
        kwargs["data_dir"] = domain.data_dir
    if domain.config_name is not None:
        kwargs["name"] = domain.config_name

    try:
        stream = load_dataset(
            domain.repo_id,
            split=domain.split,
            streaming=True,
            **kwargs,
        )
    except Exception as e:
        print(f"  [ProxyEval] WARNING: Could not load '{domain.name}': {e}. Skipping.")
        return

    chunk_size = context_length + 1
    buffer = []
    batch = []
    batches_yielded = 0
    rows_scanned = 0
    deadline = time.monotonic() + timeout_seconds
    eos_id = tokenizer.eos_token_id

    for row in stream:
        if time.monotonic() > deadline:
            print(f"  [ProxyEval] TIMEOUT on '{domain.name}' "
                  f"({batches_yielded} batches, {rows_scanned} rows).")
            return

        rows_scanned += 1
        if rows_scanned > _MAX_ROWS_SCANNED:
            print(f"  [ProxyEval] Row limit on '{domain.name}' "
                  f"({batches_yielded}/{max_batches} batches).")
            return

        text = fmt_fn(row)
        if text is None:
            continue

        tokens = tokenizer.encode(text)
        buffer.extend(tokens)
        buffer.append(eos_id)

        while len(buffer) >= chunk_size:
            chunk = torch.tensor(buffer[:chunk_size], dtype=torch.long)
            buffer = buffer[chunk_size:]
            batch.append(chunk)

            if len(batch) == batch_size:
                yield torch.stack(batch, dim=0)
                batch = []
                batches_yielded += 1
                if batches_yielded >= max_batches:
                    return


# ══════════════════════════════════════════════════════════════
#  Per-Domain Evaluation
# ══════════════════════════════════════════════════════════════

@dataclass
class DomainEvalResult:
    """Results from evaluating one domain."""
    name: str
    key: str
    avg_loss: float
    ppl: float
    num_batches: int
    num_tokens: int


@torch.inference_mode()
def eval_domain_perplexity(
    model: nn.Module,
    domain: EvalDomain,
    device: str = "cuda",
    batch_size: int = 16,
    context_length: int = 2048,
    max_batches: int = 20,
) -> Optional[DomainEvalResult]:
    """Compute perplexity on a single evaluation domain.

    Args:
        model: The model in eval mode.
        domain: Domain configuration.
        device: Device string.
        batch_size: Batch size.
        context_length: Context length.
        max_batches: Max batches to evaluate.

    Returns:
        DomainEvalResult or None if domain could not be loaded.
    """
    eos_id = tokenizer.eos_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)

    total_loss = 0.0
    num_batches = 0
    num_tokens = 0

    for batch in _stream_eval_batches(domain, batch_size, context_length, max_batches):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            batch = batch.to(device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            logits, _aux = model(inputs)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))

        total_loss += loss.item()
        num_batches += 1
        num_tokens += targets.numel()

    if num_batches == 0:
        print(f"  [ProxyEval] WARNING: No eval data for '{domain.name}'.")
        return None

    avg_loss = total_loss / num_batches
    ppl = math.exp(min(avg_loss, 20))

    return DomainEvalResult(
        name=domain.name,
        key=domain.key,
        avg_loss=avg_loss,
        ppl=ppl,
        num_batches=num_batches,
        num_tokens=num_tokens,
    )


# ══════════════════════════════════════════════════════════════
#  Full Proxy Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_proxy(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 16,
    context_length: int = 2048,
    max_batches: int = 20,
    domains: Optional[List[EvalDomain]] = None,
) -> Dict[str, float]:
    """Run lightweight evaluation across all domains.

    Returns a flat dict of metrics suitable for W&B logging and
    mixing law fitting.

    The combined_score is a weighted harmonic mean:
        60% code quality + 40% general quality (lower loss = better).
    We negate losses for the harmonic mean since lower is better,
    then use the reciprocal form.

    Args:
        model: The model (will be set to eval mode).
        device: Device string.
        batch_size: Batch size per domain.
        context_length: Context length.
        max_batches: Max batches per domain.
        domains: Custom domain list (defaults to EVAL_DOMAINS).

    Returns:
        Dict with per-domain and aggregate metrics.
    """
    model.eval()
    if domains is None:
        domains = EVAL_DOMAINS

    metrics: Dict[str, float] = {}
    results: List[DomainEvalResult] = []

    print(f"\n{'─'*60}")
    print(f"  Proxy Evaluation ({max_batches} batches/domain)")
    print(f"{'─'*60}")

    for domain in domains:
        print(f"  → {domain.name}...", end=" ", flush=True)
        try:
            result = eval_domain_perplexity(
                model, domain, device, batch_size, context_length, max_batches,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            result = None

        if result is not None:
            results.append(result)
            metrics[f"eval/{result.key}/loss"] = result.avg_loss
            metrics[f"eval/{result.key}/ppl"] = result.ppl
            metrics[f"eval/{result.key}/tokens"] = result.num_tokens
            print(f"loss={result.avg_loss:.4f}  ppl={result.ppl:.2f}  "
                  f"({result.num_batches} batches, {result.num_tokens:,} tokens)")
        else:
            print("SKIPPED")

    # ── Combined score ──
    # Weighted harmonic mean of (1/loss) values — lower loss → higher score
    # Weights: 60% code, 40% general — coding is the primary objective
    if results:
        domain_weights = {"code": 0.60, "general": 0.25, "reasoning": 0.15}
        weighted_inv_sum = 0.0
        total_w = 0.0
        for r in results:
            w = domain_weights.get(r.key, 0.1)
            if r.avg_loss > 0:
                weighted_inv_sum += w / r.avg_loss
                total_w += w

        combined = total_w / weighted_inv_sum if weighted_inv_sum > 0 else float('inf')
        metrics["eval/combined_loss"] = combined
        metrics["eval/combined_ppl"] = math.exp(min(combined, 20))

        # Simple average for reference
        avg_loss = sum(r.avg_loss for r in results) / len(results)
        metrics["eval/avg_loss"] = avg_loss
        metrics["eval/avg_ppl"] = math.exp(min(avg_loss, 20))

    print(f"{'─'*60}")
    if "eval/combined_loss" in metrics:
        print(f"  Combined: loss={metrics['eval/combined_loss']:.4f}  "
              f"ppl={metrics['eval/combined_ppl']:.2f}")
    print(f"{'─'*60}\n")

    return metrics
