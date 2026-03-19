import torch
import math
import warnings
import os
import time
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from ..configs.model_config import config, PHASE_1_CONFIG, PHASE_2_CONFIG
from ...models.model import GPT
from ..tokenizer import tokenizer
from ..dataloader import create_phase_dataloaders
from ...models.model_flash_attn import GPT_FLASH
from ..helper_funcs import get_base_dir, save_checkpoint, load_checkpoint
from .schedulers import create_phase_scheduler
from ...models.weight_init import init_gpt_model, count_parameters
from ..inference import generate

@torch.inference_mode()
def validation(model, criterion, val_data, train_step, wandb_run, phase_config):
  model.eval()
  total_val_loss = 0
  steps = 0
  for batch in val_data:
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        batch = batch.to(config.device, non_blocking=True).long()
        labels = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        logits = model(labels)
        val_loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
    steps += 1
    if (steps + 1) % 1000 == 0:
        print(f"Val Step: {steps+1}, Loss: {val_loss.item():.4f}")
    total_val_loss += val_loss.item()
    if steps == phase_config.val_steps:
      break
  avg_val_loss = total_val_loss / max(1, steps)
  wandb_run.log({
      "val/loss": avg_val_loss,
      "val/ppl": math.exp(min(avg_val_loss, 10)),
  }, step=8*train_step, commit=False)
  return avg_val_loss

def train_phase(
    model, optimizer, scheduler, criterion,
    train_data, val_data, wandb_run, phase_config,
    base_dir, start_step=0,
):
    """
    Train one phase.  Supports exact resumption via the ResumableDataLoader
    and MixerState checkpoint.

    Args:
        model, optimizer, scheduler, criterion: the usual.
        train_data:   ``ResumableDataLoader`` wrapping a ``WeightedMixerDataset``.
        val_data:     Plain ``DataLoader`` for validation.
        wandb_run:    Active W&B run.
        phase_config: ``PhaseConfig`` for this phase.
        base_dir:     Checkpoint directory path.
        start_step:   Optimizer step to resume from (0 = fresh).
    """
    optim_step = start_step
    meta_data = None
    grad_accumulation_steps = phase_config.grad_accum_steps
    val_interval = phase_config.val_interval
    phase_num = phase_config.phase_num
    seq_len = config.max_context_len
    micro_bs = phase_config.micro_batch_size
    tokens_per_step = micro_bs * seq_len * grad_accumulation_steps

    # Estimate model FLOPs per forward pass 
    raw_model = _unwrap(model)
    n_params = sum(p.numel() for p in raw_model.parameters())
    flops_per_token = 6 * n_params  
    gpu_peak_flops = 989.4e12  # H200 bf16 peak FLOPS

    try:
        model.train()
        best_val_loss = float('inf')
        optimizer.zero_grad()
        accum_loss = 0.0
        micro_count = 0
        step_start_time = time.perf_counter()

        for i, batch in enumerate(tqdm(train_data, desc=f"Phase {phase_num} Training")):
            batch = batch.to(config.device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
                loss_value = loss.item()
            loss = loss / grad_accumulation_steps
            loss.backward()
            accum_loss += loss_value
            micro_count += 1

            if micro_count == grad_accumulation_steps:
                optim_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), phase_config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                avg_accum_loss = accum_loss / grad_accumulation_steps
                # ── Throughput & hardware metrics ──
                step_elapsed = time.perf_counter() - step_start_time
                tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
                step_flops = flops_per_token * tokens_per_step
                mfu = step_flops / (step_elapsed * gpu_peak_flops) if step_elapsed > 0 else 0.0

                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

                metrics = {
                    "train/loss": avg_accum_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/ppl": math.exp(min(avg_accum_loss, 10)),
                    "train/phase": phase_num,
                    "train/grad_norm": grad_norm.item(),
                    "perf/tokens_per_sec": tps,
                    "perf/mfu": mfu,
                    "perf/vram_allocated_gb": allocated_gb,
                    "perf/vram_reserved_gb": reserved_gb,
                    "perf/step_time_sec": step_elapsed,
                }

                # ── Expert usage logging ──
                raw = _unwrap(model)
                for layer_idx, layer in enumerate(raw.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
                        moe = layer.mlp
                        if moe.total_tokens > 0:
                            moe_metrics = moe.get_wandb_metrics()
                            metrics.update({
                                f"moe/layer_{layer_idx}/{k}": v
                                for k, v in moe_metrics.items()
                            })
                            moe.reset_expert_counts()

                wandb_run.log(metrics, step=8*optim_step)
                accum_loss = 0.0
                micro_count = 0

                if optim_step % 100 == 0:
                    print(
                        f"Step : {optim_step} , Loss : {avg_accum_loss:.4f} , "
                        f"TPS : {tps:.0f} , MFU : {mfu:.2%} , "
                        f"VRAM : {allocated_gb:.2f}/{reserved_gb:.2f} GB"
                    )

                if optim_step % val_interval == 0:
                    val_loss = validation(
                        model, criterion, val_data, train_step=optim_step,
                        wandb_run=wandb_run, phase_config=phase_config,
                    )
                    raw = _unwrap(model)
                    print(generate(raw,
                            "Chapter 1. The dark forest was",
                            config.device, max_tokens=60, temp=0.8))
                    print(generate(raw,
                            "The following is a Python function that reverses a string:\n\ndef reverse_string(s):",
                            config.device, max_tokens=80, temp=0.3))
                    print(generate(raw,
                            "To solve the quadratic equation x^2 - 5x + 6 = 0, we first",
                            config.device, max_tokens=120, temp=0.2))
                    print(generate(raw,
                            "The theory of general relativity, published by Albert Einstein in 1915, states that",
                            config.device, max_tokens=80, temp=0.4))
                    print(generate(raw,
                            "In this essay, I will argue that renewable energy is essential for economic growth because",
                            config.device, max_tokens=250, temp=0.5))
                    torch.cuda.empty_cache()
                    model.train()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    meta_data = {
                        "step": optim_step,
                        "train_loss": avg_accum_loss,
                        "val_loss": val_loss,
                    }
                    dataloader_state = train_data.get_state()
                    save_checkpoint(
                        base_dir, optim_step,
                        model_data=_unwrap(model).state_dict(),
                        optimizer_data=optimizer.state_dict(),
                        scheduler_data=scheduler.state_dict(),
                        wandb_run=wandb_run,
                        dataloader_state=dataloader_state,
                        meta_data=meta_data,
                        phase=phase_num,
                    )

                step_start_time = time.perf_counter()
                if optim_step >= phase_config.total_steps:
                    print(f"Reached total_steps ({phase_config.total_steps}). Phase complete.")
                    break

        print(f"Phase {phase_num} training complete at optimizer step {optim_step}.")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving checkpoint at optimizer step {optim_step}...")
        dataloader_state = train_data.get_state()
        save_checkpoint(
            base_dir, optim_step,
            model_data=_unwrap(model).state_dict(),
            optimizer_data=optimizer.state_dict(),
            scheduler_data=scheduler.state_dict(),
            wandb_run=wandb_run,
            dataloader_state=dataloader_state,
            meta_data=meta_data,
            phase=phase_num,
        )
        print(f"[Interrupt] Checkpoint saved successfully.")
        raise  



def _unwrap(model):
    """Return the raw module behind torch.compile (or the model itself)."""
    return getattr(model, '_orig_mod', model)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.set_float32_matmul_precision('high')       
    torch.backends.cudnn.benchmark = True            
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    base_dir = get_base_dir("checkpoints")

    # ── Model ──────────────────────────────────────────────
    use_flash_attn = True
    model = GPT_FLASH(config, "cuda") if use_flash_attn else GPT(config, "cuda")

    # Initialize model weights
    init_gpt_model(model, config)
    count_parameters(model)

    # ── Phase selection ────────────────────────────────────
    phase_config = PHASE_1_CONFIG

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_config.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )
    scheduler = create_phase_scheduler(optimizer, phase_config)

    # ── W&B ────────────────────────────────────────────────
    wandb_run = wandb.init(
        entity="akshithmarepally-akai",
        project="828_pretraining_h200",
        config={
            "architecture": "GPT_FLASH_MoE",
            "phase": phase_config.phase_name,
            "datasets": [ds.name for ds in phase_config.datasets],
            "model_config": {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_hidden_layers,
                "num_experts": config.num_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "num_attn_heads": config.num_attn_heads,
                "context_len": config.max_context_len,
            },
            "phase_config": {
                "peak_lr": phase_config.peak_lr,
                "min_lr": phase_config.min_lr,
                "scheduler": phase_config.scheduler_type,
                "warmup_steps": phase_config.warmup_steps,
                "total_steps": phase_config.total_steps,
                "effective_batch": phase_config.effective_batch_size(),
                "grad_accum": phase_config.grad_accum_steps,
            },
        },
    )

    # ── Resume from checkpoint ─────────────────────────────
    start_step, dataloader_state, saved_phase = load_checkpoint(
        base_dir, model, optimizer, scheduler, device=config.device
    )

    if saved_phase == 2 and phase_config.phase_num != 2:
        print("[Train] Checkpoint is from Phase 2 — switching to PHASE_2_CONFIG")
        phase_config = PHASE_2_CONFIG
        for pg in optimizer.param_groups:
            pg["lr"] = phase_config.peak_lr

    if start_step > 0:
        scheduler = create_phase_scheduler(optimizer, phase_config)
        for _ in range(start_step):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Scheduler] Rebuilt from config and fast-forwarded to optimizer step {start_step}")
        print(f"[Scheduler] Current LR: {current_lr:.6e}")
        print(f"[Scheduler] Remaining steps: {phase_config.total_steps - start_step}")

    # ── Compile model ──────────────────────────────────────
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    # ── Dataloaders ────────────────────────────────────────
    train_data, val_data = create_phase_dataloaders(
        phase_config=phase_config,
        train_state=dataloader_state,
        val_repo_id="HuggingFaceFW/fineweb-edu",
        batch_size_val=16,
        context_length=config.max_context_len,
    )

    # ── Train ──────────────────────────────────────────────
    try:
        train_phase(
            model, optimizer, scheduler, criterion,
            train_data, val_data, wandb_run, phase_config,
            base_dir, start_step=start_step,
        )
    except KeyboardInterrupt:
        pass
    finally:
        wandb_run.finish()
