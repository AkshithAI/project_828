"""
Position-Bucket Validation (long-context monitoring)
====================================================

Computes per-position-bucket cross-entropy loss so the YaRN extension run can
tell *where* in the sequence the model is (or isn't) learning long-range
structure. Averaging a single loss over an 8K sequence hides whether the tail
positions (4K–8K) are improving at all — bucketed loss makes that explicit.

Buckets (token index of the *target* position):
    (0, 2047)    — native window
    (2048, 4095) — first extension band
    (4096, 6143) — second extension band
    (6144, 8191) — far extension band

Logs ``val_position/bucket_0_2047`` (loss) and ``.../ppl`` per bucket to W&B.
Used by both the extension training loop and post-hoc evaluation.

Usage:
    from .validate_position_buckets import validate_position_buckets
    validate_position_buckets(
        model, val_stream_8k, wandb_run, train_step,
        device="cuda", context_length=8192,
    )
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ..tokenizer import tokenizer


POSITION_BUCKETS: List[Tuple[int, int]] = [
    (0, 2047),      # Native window
    (2048, 4095),   # First extension band
    (4096, 6143),   # Second extension band
    (6144, 8191),   # Far extension band
]


def _bucket_key(lo: int, hi: int) -> str:
    return f"val_position/bucket_{lo}_{hi}"


@torch.no_grad()
def validate_position_buckets(
    model: nn.Module,
    val_batches: Iterable[torch.Tensor],
    wandb_run,
    train_step: int,
    device: str = "cuda",
    context_length: int = 8192,
    max_batches: int = 50,
    buckets: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Compute per-position-bucket loss over a long-context validation stream.

    Args:
        model:          Model to evaluate (set to eval mode internally).
        val_batches:    Iterable of ``[B, context_length + 1]`` LongTensors, or
                        ``[B, S]`` batches at least ``context_length + 1`` wide.
        wandb_run:      Active W&B run (or None to skip logging).
        train_step:     Current optimizer step (W&B x-axis).
        device:         Device string.
        context_length: Sequence length used for evaluation (default 8192).
        max_batches:    Cap on number of validation batches.
        buckets:        Optional custom position buckets.

    Returns:
        Dict of ``{bucket_key: avg_loss}`` (also includes ``/ppl`` entries).
    """
    if buckets is None:
        buckets = POSITION_BUCKETS

    was_training = model.training
    model.eval()

    ignore_index = tokenizer.pad_token_id
    if ignore_index is None:
        ignore_index = tokenizer.eos_token_id
    if ignore_index is None:
        ignore_index = -100

    # Running sums per bucket: (sum_loss, n_tokens).
    bucket_loss_sum = {b: 0.0 for b in buckets}
    bucket_tok_count = {b: 0 for b in buckets}

    num_batches = 0
    for batch in val_batches:
        if num_batches >= max_batches:
            break

        batch = batch.to(device, non_blocking=True).long()
        # Trim/truncate to context_length + 1 (inputs + shifted targets).
        if batch.shape[1] < context_length + 1:
            continue
        batch = batch[:, : context_length + 1]
        inputs = batch[:, :-1].contiguous()          # [B, T]
        targets = batch[:, 1:].contiguous()          # [B, T]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            # Force logits (per-position) rather than the fused scalar loss.
            out = model(inputs, return_logits=True)
            logits = out[0] if isinstance(out, tuple) else out

        # Per-token CE, keep position dimension: [B, T].
        B, T, V = logits.shape
        per_tok = F.cross_entropy(
            logits.reshape(-1, V).float(),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(B, T)

        valid = (targets != ignore_index)

        # Accumulate into buckets by *target position index* (0..T-1).
        for (lo, hi) in buckets:
            hi_clamped = min(hi, T - 1)
            if lo > hi_clamped:
                continue
            seg_loss = per_tok[:, lo : hi_clamped + 1]
            seg_valid = valid[:, lo : hi_clamped + 1]
            n = int(seg_valid.sum().item())
            if n > 0:
                bucket_loss_sum[(lo, hi)] += float((seg_loss * seg_valid).sum().item())
                bucket_tok_count[(lo, hi)] += n

        num_batches += 1

    # ── Reduce + log ──
    metrics: Dict[str, float] = {}
    print(f"\n{'='*60}")
    print(f"  Position-Bucket Validation — Step {train_step}")
    print(f"{'='*60}")
    for (lo, hi) in buckets:
        n = bucket_tok_count[(lo, hi)]
        if n == 0:
            continue
        avg_loss = bucket_loss_sum[(lo, hi)] / n
        ppl = math.exp(min(avg_loss, 20))
        key = _bucket_key(lo, hi)
        metrics[key] = avg_loss
        metrics[f"{key}/ppl"] = ppl
        print(f"  {lo:>5d}–{hi:<5d} | loss {avg_loss:.4f} | ppl {ppl:8.2f} | tokens {n}")

    # Tail-vs-native delta: quick signal for whether long positions lag.
    native_key = _bucket_key(*buckets[0])
    far_key = _bucket_key(*buckets[-1])
    if native_key in metrics and far_key in metrics:
        metrics["val_position/tail_minus_native_loss"] = (
            metrics[far_key] - metrics[native_key]
        )

    if wandb_run is not None and metrics:
        wandb_run.log(metrics, step=8 * train_step, commit=False)

    if was_training:
        model.train()

    return metrics
