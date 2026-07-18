"""
Proxy Model Training Runner
==============================

Trains a small proxy MoE model on a specific data mixture, with full
checkpointing and resumption support.

Designed for single-GPU sequential execution on rented instances.
Each proxy run trains for 500M tokens (~1920 optimizer steps at the
default batch config), saves checkpoints every 500 steps, and runs
lightweight evaluation every 250 steps.

Resumption:
    - Within a run: loads latest checkpoint from the run's subdirectory.
    - Across runs: managed by the ProxyManifest in run_datamix_tests.py.
"""

import os
import math
import time
import json
import warnings
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast
from typing import Dict, Optional, Tuple, Any

import wandb

from ..tokenizer import tokenizer
from ..dataloader import (
    load_phase_datasets, ResumableDataLoader, PrefetchedDataLoader,
)
from ..configs.model_config import PhaseConfig, DatasetEntry
from ..helper_funcs import save_checkpoint, load_checkpoint, get_base_dir
from ...models.model_improv import GPT_FLASH
from ...models.weight_init import init_gpt_model, count_parameters
from ..training.schedulers import create_phase_scheduler

from .datamix_config import (
    ProxyModelConfig, ProxyExperimentConfig, MixturePoint,
    ProxyRunResult, build_datasets_for_mixture,
    WANDB_ENTITY, WANDB_PROJECT,
)
from .proxy_eval import evaluate_proxy


# ══════════════════════════════════════════════════════════════
#  Model Creation
# ══════════════════════════════════════════════════════════════

def create_proxy_model(
    config: ProxyModelConfig,
    device: str = "cuda",
) -> GPT_FLASH:
    """Instantiate and initialize a proxy-scale GPT_FLASH model.

    Args:
        config: Proxy model configuration.
        device: Device for model placement.

    Returns:
        Initialized GPT_FLASH model.
    """
    print(f"\n[ProxyRunner] Creating proxy model on {device}...")
    model = GPT_FLASH(config, device)
    init_gpt_model(model, config)
    total, trainable = count_parameters(model)
    print(f"[ProxyRunner] Proxy model: {total:,} total, {trainable:,} trainable")
    return model


# ══════════════════════════════════════════════════════════════
#  Dataloader Creation
# ══════════════════════════════════════════════════════════════

def create_proxy_dataloaders(
    mixture: MixturePoint,
    experiment_config: ProxyExperimentConfig,
    train_state: Optional[Dict[str, Any]] = None,
) -> PrefetchedDataLoader:
    """Build a weighted-mixer dataloader for a specific mixture point.

    Reuses the existing ``load_phase_datasets`` infrastructure by
    constructing a temporary ``PhaseConfig`` from the mixture weights.

    Args:
        mixture: The mixture point to train on.
        experiment_config: Experiment-level configuration.
        train_state: Saved mixer state for resumption (or None).

    Returns:
        A PrefetchedDataLoader wrapping the weighted mixer.
    """
    datasets = build_datasets_for_mixture(mixture)
    total_steps = experiment_config.total_steps_per_run()

    # Build a PhaseConfig compatible with load_phase_datasets
    phase_config = PhaseConfig(
        phase_name=f"proxy_{mixture.label}",
        phase_num=1,
        peak_lr=experiment_config.peak_lr,
        min_lr=experiment_config.min_lr,
        warmup_steps=experiment_config.warmup_steps,
        total_steps=total_steps,
        scheduler_type=experiment_config.scheduler_type,
        wsd_stable_frac=experiment_config.wsd_stable_frac,
        micro_batch_size=experiment_config.micro_batch_size,
        grad_accum_steps=experiment_config.grad_accum_steps,
        grad_clip=experiment_config.grad_clip,
        val_interval=experiment_config.eval_interval,
        val_steps=experiment_config.eval_batches_per_domain,
        eval_suite_interval=0,
        datasets=datasets,
    )

    mixer = load_phase_datasets(
        phase_config,
        mixer_state=train_state,
        context_length=experiment_config.context_length,
    )

    raw_loader = ResumableDataLoader(
        mixer,
        batch_size=experiment_config.micro_batch_size,
        pin_memory=True,
        num_workers=0,
    )

    return PrefetchedDataLoader(raw_loader, num_prefetch=2)


# ══════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════

def train_proxy(
    mixture: MixturePoint,
    experiment_config: ProxyExperimentConfig,
    device: str = "cuda",
    wandb_run=None,
) -> ProxyRunResult:
    """Train a single proxy model on the given mixture.

    Handles:
    - Model creation and initialization
    - Checkpoint resumption (within this run)
    - Gradient accumulation with clipping
    - Periodic evaluation (lightweight, 20 batches/domain)
    - W&B logging
    - Emergency checkpoint on interrupt

    Args:
        mixture: The mixture point configuration.
        experiment_config: Experiment-level settings.
        device: CUDA device string.
        wandb_run: Active W&B run (or None to create one).

    Returns:
        ProxyRunResult with final evaluation metrics.
    """
    ec = experiment_config
    mc = ec.model_config
    total_steps = ec.total_steps_per_run()
    ctx_len = ec.context_length
    eos_id = tokenizer.eos_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)

    # ── Run directory ──
    run_dir = Path(ec.checkpoint_dir) / mixture.label
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    own_wandb = False
    if wandb_run is None:
        own_wandb = True
        wandb_run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group="proxy_grid",
            name=f"proxy_{mixture.label}",
            config={
                "mixture": mixture.to_weights_dict(),
                "tokens_per_run": ec.tokens_per_run,
                "proxy_model": {
                    "hidden_dim": mc.hidden_dim,
                    "num_layers": mc.num_hidden_layers,
                    "num_experts": mc.num_experts,
                    "top_k": mc.num_experts_per_tok,
                },
                "lr": ec.peak_lr,
                "effective_batch": ec.effective_batch_size,
            },
            resume="allow",
        )

    print(f"\n{'='*70}")
    print(f"  Proxy Run: {mixture.label}")
    print(f"  Mix: {mixture.to_weights_dict()}")
    print(f"  Budget: {ec.tokens_per_run/1e6:.0f}M tokens, {total_steps} steps")
    print(f"{'='*70}")

    # ── Model ──
    model = create_proxy_model(mc, device)

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ec.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # Build a PhaseConfig for the scheduler
    phase_cfg = PhaseConfig(
        phase_name=f"proxy_{mixture.label}",
        phase_num=1,
        peak_lr=ec.peak_lr,
        min_lr=ec.min_lr,
        warmup_steps=ec.warmup_steps,
        total_steps=total_steps,
        scheduler_type=ec.scheduler_type,
        wsd_stable_frac=ec.wsd_stable_frac,
    )
    scheduler = create_phase_scheduler(optimizer, phase_cfg)

    # ── Resume from checkpoint ──
    start_step = 0
    dataloader_state = None
    if (run_dir / "checkpoints").exists():
        ckpt_dir = str(run_dir / "checkpoints")
        start_step, dataloader_state, _ = load_checkpoint(
            ckpt_dir, model, optimizer, scheduler, device=device,
        )
        if start_step > 0:
            # Rebuild scheduler at correct position
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", ec.peak_lr)
            scheduler = create_phase_scheduler(
                optimizer, phase_cfg, last_epoch=start_step - 1,
            )
            print(f"[ProxyRunner] Resumed at step {start_step}/{total_steps}")

    if start_step >= total_steps:
        print(f"[ProxyRunner] Run already complete ({start_step} >= {total_steps})")
        # Run final eval and return
        final_metrics = evaluate_proxy(
            model, device, ec.eval_batch_size, ctx_len, ec.eval_batches_per_domain,
        )
        model.train()
        return _build_result(mixture, total_steps, ec, final_metrics)

    # ── Dataloader ──
    train_data = create_proxy_dataloaders(mixture, ec, train_state=dataloader_state)

    # ── Compile model for speed ──
    try:
        model = torch.compile(model, mode="default")
        print("[ProxyRunner] Model compiled with torch.compile")
    except Exception:
        print("[ProxyRunner] torch.compile unavailable — running eager")

    # ── Training loop ──
    optim_step = start_step
    accum_loss = 0.0
    micro_count = 0
    tokens_per_step = ec.tokens_per_step
    ckpt_dir = str(run_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    n_params = sum(p.numel() for p in _unwrap(model).parameters())

    try:
        model.train()
        optimizer.zero_grad()
        step_start = time.perf_counter()

        for batch in tqdm(train_data, desc=f"Proxy {mixture.label}", total=None):
            batch = batch.to(device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux_loss = _unwrap(model)(inputs)
                ce_loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1),
                )
                loss = ce_loss + aux_loss

            (loss / ec.grad_accum_steps).backward()
            accum_loss += ce_loss.detach()
            micro_count += 1

            if micro_count == ec.grad_accum_steps:
                optim_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ec.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_loss = (accum_loss / ec.grad_accum_steps).item()
                elapsed = time.perf_counter() - step_start
                tps = tokens_per_step / elapsed if elapsed > 0 else 0

                # ── Log metrics ──
                metrics = {
                    "train/loss": avg_loss,
                    "train/ppl": math.exp(min(avg_loss, 10)),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/aux_loss": aux_loss.item(),
                    "perf/tokens_per_sec": tps,
                    "perf/step_time_sec": elapsed,
                }
                wandb_run.log(metrics, step=optim_step)

                if optim_step % 100 == 0:
                    print(f"  [{mixture.label}] step {optim_step}/{total_steps} "
                          f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                          f"tps={tps:.0f}")

                # ── Evaluation ──
                if optim_step % ec.eval_interval == 0:
                    eval_metrics = evaluate_proxy(
                        _unwrap(model), device, ec.eval_batch_size,
                        ctx_len, ec.eval_batches_per_domain,
                    )
                    wandb_run.log(eval_metrics, step=optim_step, commit=False)
                    model.train()

                # ── Checkpoint ──
                if optim_step % ec.checkpoint_interval == 0:
                    dl_state = train_data.get_state()
                    save_checkpoint(
                        ckpt_dir, optim_step,
                        model_data=_unwrap(model).state_dict(),
                        optimizer_data=optimizer.state_dict(),
                        scheduler_data=scheduler.state_dict(),
                        wandb_run=wandb_run,
                        dataloader_state=dl_state,
                        phase=1,
                    )

                accum_loss = 0.0
                micro_count = 0
                step_start = time.perf_counter()

                if optim_step >= total_steps:
                    print(f"[ProxyRunner] Reached {total_steps} steps. Run complete.")
                    break

    except KeyboardInterrupt:
        print(f"\n[ProxyRunner] Interrupted at step {optim_step}. Saving checkpoint...")
        dl_state = train_data.get_state()
        save_checkpoint(
            ckpt_dir, optim_step,
            model_data=_unwrap(model).state_dict(),
            optimizer_data=optimizer.state_dict(),
            scheduler_data=scheduler.state_dict(),
            wandb_run=wandb_run,
            dataloader_state=dl_state,
            phase=1,
        )
        print("[ProxyRunner] Checkpoint saved. Re-raising interrupt.")
        if own_wandb:
            wandb_run.finish(quiet=True)
        raise

    # ── Final evaluation ──
    print(f"\n[ProxyRunner] Final evaluation for {mixture.label}...")
    final_metrics = evaluate_proxy(
        _unwrap(model), device, ec.eval_batch_size,
        ctx_len, ec.eval_batches_per_domain,
    )
    wandb_run.log(final_metrics, step=optim_step)

    if own_wandb:
        wandb_run.finish()

    # ── Cleanup GPU memory ──
    del model, optimizer, scheduler, train_data
    torch.cuda.empty_cache()

    return _build_result(mixture, optim_step, ec, final_metrics)


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _unwrap(model):
    """Return the raw module behind torch.compile."""
    return getattr(model, "_orig_mod", model)


def _build_result(
    mixture: MixturePoint,
    final_step: int,
    ec: ProxyExperimentConfig,
    metrics: Dict[str, float],
) -> ProxyRunResult:
    """Construct a ProxyRunResult from evaluation metrics."""
    return ProxyRunResult(
        label=mixture.label,
        code_pct=mixture.code_pct,
        book_pct=mixture.book_pct,
        web_pct=mixture.web_pct,
        final_step=final_step,
        total_tokens_seen=final_step * ec.tokens_per_step,
        code_loss=metrics.get("eval/code/loss", float("nan")),
        general_loss=metrics.get("eval/general/loss", float("nan")),
        reasoning_loss=metrics.get("eval/reasoning/loss", float("nan")),
        code_ppl=metrics.get("eval/code/ppl", float("nan")),
        general_ppl=metrics.get("eval/general/ppl", float("nan")),
        reasoning_ppl=metrics.get("eval/reasoning/ppl", float("nan")),
        combined_score=metrics.get("eval/combined_loss", float("nan")),
    )
