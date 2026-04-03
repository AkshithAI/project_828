"""
Domain-Specific Validation Suite
=================================

Evaluates model performance across the 5 macro domain categories that
compose the training data mix, producing **per-domain loss/perplexity
scores** and logging them to W&B.

This supplements the existing single-number validation loss by showing
*where* the model is improving or degrading.

Domain Mapping (from model_config.py):
    ┌──────────────────┬────────┬──────────────────────────────────────┐
    │ Domain           │ Weight │ Validation Source                    │
    ├──────────────────┼────────┼──────────────────────────────────────┤
    │ Source Code      │  45%   │ bigcode/starcoderdata (python)       │
    │ General Knowledge│  19%   │ HuggingFaceFW/fineweb-edu            │
    │ Math/Reasoning   │  14%   │ nvidia/OpenMathInstruct-2            │
    │ Code-Adjacent    │  17%   │ HuggingFaceH4/stack-exchange-prefs   │
    │ Instruction      │   5%   │ teknium/OpenHermes-2.5               │
    └──────────────────┴────────┴──────────────────────────────────────┘

Usage in train.py:
    from .validate_domains import validate_domains
    domain_results = validate_domains(model, criterion, wandb_run, optim_step, phase_config)
"""

import math
import torch
import torch.nn as nn
from torch.amp import autocast
from datasets import load_dataset
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

from ..tokenizer import tokenizer
from ..configs.model_config import config


# ═══════════════════════════════════════════════════════════════
#  Domain definitions
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationDomain:
    """Configuration for a single validation domain."""
    name: str                   # Human-readable domain name
    key: str                    # W&B metric key (lowercase, no spaces)
    repo_id: str                # HuggingFace dataset repository
    weight_pct: int             # Training weight percentage (for context)
    format_fn: str              # Name of the format function to apply
    config_name: Optional[str] = None
    data_dir: Optional[str] = None
    split: str = "train"


# The 5 macro domain categories matching our data mix
VALIDATION_DOMAINS: List[ValidationDomain] = [
    ValidationDomain(
        name="Source Code",
        key="source_code",
        repo_id="bigcode/starcoderdata",
        weight_pct=45,
        format_fn="starcoder",
        data_dir="python",
    ),
    ValidationDomain(
        name="General Knowledge",
        key="general_knowledge",
        repo_id="HuggingFaceFW/fineweb-edu",
        weight_pct=19,
        format_fn="fineweb_edu",
        config_name="sample-100BT",
    ),
    ValidationDomain(
        name="Math/Reasoning",
        key="math_reasoning",
        repo_id="nvidia/OpenMathInstruct-2",
        weight_pct=14,
        format_fn="openmath",
    ),
    ValidationDomain(
        name="Code-Adjacent",
        key="code_adjacent",
        repo_id="HuggingFaceH4/stack-exchange-preferences",
        weight_pct=17,
        format_fn="stackexchange",
    ),
    ValidationDomain(
        name="Instruction",
        key="instruction",
        repo_id="teknium/OpenHermes-2.5",
        weight_pct=5,
        format_fn="openhermes",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  Format functions (copied from dataloader.py to keep self-contained)
# ═══════════════════════════════════════════════════════════════

def _fmt_starcoder(row: Dict[str, Any]) -> Optional[str]:
    content = row.get("content", "")
    if not content or len(content) < 100 or len(content) > 100_000:
        return None
    return content


def _fmt_fineweb_edu(row: Dict[str, Any]) -> Optional[str]:
    score = row.get("score", 0.0)
    if score is None or score < 3.5:
        return None
    return row.get("text", "") or None


def _fmt_openmath(row: Dict[str, Any]) -> Optional[str]:
    problem = row.get("problem", "")
    solution = row.get("generated_solution", "")
    if not problem and not solution:
        return None
    return f"{problem}\n\n{solution}"


def _fmt_stackexchange(row: Dict[str, Any]) -> Optional[str]:
    question = row.get("question", "")
    chosen = row.get("chosen", "")
    if not question or not chosen or len(chosen) < 50:
        return None
    return f"{question}\n\n{chosen}"


def _fmt_openhermes(row: Dict[str, Any]) -> Optional[str]:
    conversations = row.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        return None
    parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        value = turn.get("value", "")
        if not value:
            continue
        speaker = turn.get("from", "")
        parts.append(f"{speaker}: {value}" if speaker else value)
    return "\n\n".join(parts) if parts else None


FORMAT_FNS: Dict[str, Callable] = {
    "starcoder": _fmt_starcoder,
    "fineweb_edu": _fmt_fineweb_edu,
    "openmath": _fmt_openmath,
    "stackexchange": _fmt_stackexchange,
    "openhermes": _fmt_openhermes,
}


# ═══════════════════════════════════════════════════════════════
#  Domain data streaming + tokenization
# ═══════════════════════════════════════════════════════════════

def _stream_domain_batches(
    domain: ValidationDomain,
    batch_size: int = 16,
    context_length: int = 2048,
    max_batches: int = 100,
):
    """
    Stream tokenized batches from a single validation domain.

    Yields tensors of shape (batch_size, context_length + 1) — same format
    as the training dataloader.  Stops after ``max_batches`` complete batches.

    Args:
        domain:         ValidationDomain config.
        batch_size:     Samples per batch.
        context_length: Token sequence length per sample.
        max_batches:    Maximum number of batches to yield.

    Yields:
        torch.LongTensor of shape (batch_size, context_length + 1)
    """
    fmt_fn = FORMAT_FNS.get(domain.format_fn)
    if fmt_fn is None:
        print(f"  [DomainVal] WARNING: Unknown format_fn '{domain.format_fn}' "
              f"for domain '{domain.name}'. Skipping.")
        return

    # Load streaming dataset
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
        print(f"  [DomainVal] WARNING: Could not load dataset for domain "
              f"'{domain.name}': {e}. Skipping.")
        return

    chunk_size = context_length + 1
    buffer = []
    batch = []
    batches_yielded = 0

    for row in stream:
        text = fmt_fn(row)
        if text is None:
            continue

        tokens = tokenizer.encode(text)
        buffer.extend(tokens)
        buffer.append(tokenizer.eos_token_id)

        # Drain complete chunks from buffer
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


# ═══════════════════════════════════════════════════════════════
#  Domain validation core
# ═══════════════════════════════════════════════════════════════

@dataclass
class DomainResult:
    """Results from validating on a single domain."""
    name: str
    key: str
    weight_pct: int
    avg_loss: float
    ppl: float
    num_batches: int
    num_tokens: int


@torch.inference_mode()
def validate_single_domain(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    domain: ValidationDomain,
    device: str = "cuda",
    batch_size: int = 16,
    context_length: int = 2048,
    max_batches: int = 100,
) -> Optional[DomainResult]:
    """
    Run validation on a single domain.

    Args:
        model:          The model (in eval mode).
        criterion:      CrossEntropyLoss criterion.
        domain:         Domain configuration.
        device:         Device string.
        batch_size:     Batch size for validation.
        context_length: Context length.
        max_batches:    Max batches per domain.

    Returns:
        DomainResult or None if the domain could not be loaded.
    """
    total_loss = 0.0
    num_batches = 0
    num_tokens = 0

    for batch in _stream_domain_batches(domain, batch_size, context_length, max_batches):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            batch = batch.to(device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))

        total_loss += loss.item()
        num_batches += 1
        num_tokens += targets.numel()

    if num_batches == 0:
        print(f"  [DomainVal] WARNING: No validation data for domain '{domain.name}'.")
        return None

    avg_loss = total_loss / num_batches
    ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow

    return DomainResult(
        name=domain.name,
        key=domain.key,
        weight_pct=domain.weight_pct,
        avg_loss=avg_loss,
        ppl=ppl,
        num_batches=num_batches,
        num_tokens=num_tokens,
    )


def validate_domains(
    model: nn.Module,
    wandb_run,
    train_step: int,
    phase_config=None,
    device: str = "cuda",
    batch_size: int = 16,
    max_batches_per_domain: int = 100,
    domains: Optional[List[ValidationDomain]] = None,
) -> List[DomainResult]:
    """
    Run validation across all training domains and log per-domain metrics.

    Args:
        model:                   The model (will be set to eval mode).
        wandb_run:               Active W&B run for metric logging.
        train_step:              Current optimizer step (for W&B x-axis).
        phase_config:            Optional PhaseConfig (unused currently, for future extension).
        device:                  Device string.
        batch_size:              Batch size for domain validation.
        max_batches_per_domain:  Maximum validation batches per domain.
        domains:                 Optional custom domain list (defaults to VALIDATION_DOMAINS).

    Returns:
        List of DomainResult objects.
    """
    model.eval()
    eos_id = tokenizer.eos_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)
    context_length = config.max_context_len

    if domains is None:
        domains = VALIDATION_DOMAINS

    results: List[DomainResult] = []
    metrics: Dict[str, float] = {}

    print(f"\n{'='*70}")
    print(f"  Domain Validation — Step {train_step}")
    print(f"{'='*70}")

    for domain in domains:
        print(f"\n  → Validating: {domain.name} ({domain.weight_pct}% of training mix)...")
        result = validate_single_domain(
            model=model,
            criterion=criterion,
            domain=domain,
            device=device,
            batch_size=batch_size,
            context_length=context_length,
            max_batches=max_batches_per_domain,
        )

        if result is not None:
            results.append(result)
            metrics[f"val_domain/{result.key}/loss"] = result.avg_loss
            metrics[f"val_domain/{result.key}/ppl"] = result.ppl
            metrics[f"val_domain/{result.key}/num_tokens"] = result.num_tokens

    # ── Summary table ──
    _print_summary_table(results)

    # ── Aggregate metrics ──
    if results:
        # Weighted average loss (weighted by training mix proportions)
        total_weight = sum(r.weight_pct for r in results)
        weighted_loss = sum(r.avg_loss * r.weight_pct for r in results) / max(total_weight, 1)
        weighted_ppl = math.exp(min(weighted_loss, 20))
        metrics["val_domain/weighted_avg_loss"] = weighted_loss
        metrics["val_domain/weighted_avg_ppl"] = weighted_ppl

        # Simple average (unweighted)
        simple_avg_loss = sum(r.avg_loss for r in results) / len(results)
        metrics["val_domain/simple_avg_loss"] = simple_avg_loss

    # ── Log to W&B ──
    if wandb_run is not None and metrics:
        wandb_run.log(metrics, step=8 * train_step, commit=False)

    return results


def _print_summary_table(results: List[DomainResult]) -> None:
    """Print a formatted summary table of domain validation results."""
    if not results:
        print("  No domain validation results to display.\n")
        return

    print(f"\n  {'─'*68}")
    print(f"  {'Domain':<22} {'Weight':>6} {'Loss':>8} {'PPL':>10} {'Batches':>8} {'Tokens':>10}")
    print(f"  {'─'*68}")

    for r in sorted(results, key=lambda x: x.avg_loss):
        print(f"  {r.name:<22} {r.weight_pct:>5}% {r.avg_loss:>8.4f} {r.ppl:>10.2f} "
              f"{r.num_batches:>8} {r.num_tokens:>10,}")

    print(f"  {'─'*68}")

    # Weighted average
    total_weight = sum(r.weight_pct for r in results)
    if total_weight > 0:
        w_loss = sum(r.avg_loss * r.weight_pct for r in results) / total_weight
        w_ppl = math.exp(min(w_loss, 20))
        total_tokens = sum(r.num_tokens for r in results)
        print(f"  {'Weighted Average':<22} {'':>6} {w_loss:>8.4f} {w_ppl:>10.2f} "
              f"{'':>8} {total_tokens:>10,}")

    print(f"  {'─'*68}\n")
