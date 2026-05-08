import torch
import math
import warnings
import os
import time
from pathlib import Path
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from ..configs.model_config import config, PHASE_1_CONFIG, PHASE_2_CONFIG
from ...models.model import GPT
from ..tokenizer import tokenizer
from ..dataloader import create_phase_dataloaders
from ...models.model_flash_attn import GPT_FLASH
from ..helper_funcs import (
    get_base_dir, save_checkpoint, save_checkpoint_async,
    load_checkpoint,
)
from .schedulers import create_phase_scheduler
from ...models.weight_init import init_gpt_model, count_parameters
from ..inference import generate
from .validate_domains import validate_domains


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
    model, optimizer, scheduler,
    train_data, val_data, wandb_run, phase_config,
    base_dir, start_step=0,
):
    """
    Train one phase.  Supports exact resumption via the ResumableDataLoader
    and MixerState checkpoint.

    Args:
        model, optimizer, scheduler: the usual.
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
    eos_id = tokenizer.eos_token_id

    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)

    # Estimate model FLOPs per forward pass 
    n_params = sum(p.numel() for p in _unwrap(model).parameters())
    flops_per_token = 6 * n_params  
    gpu_peak_flops = 989.4e12  # H200 bf16 peak FLOPS

    # ── Async checkpoint thread handle ──
    _save_thread = None

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
                loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1),
                )

            (loss / grad_accumulation_steps).backward()
            accum_loss = accum_loss + loss.detach()  
            micro_count += 1

            if micro_count == grad_accumulation_steps:
                optim_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), phase_config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


                avg_accum_loss = (accum_loss / grad_accumulation_steps).item()  
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

                    # ── Domain-specific validation ──
                    validate_domains(
                        model=model,
                        wandb_run=wandb_run,
                        train_step=optim_step,
                        phase_config=phase_config,
                        device=config.device,
                        batch_size=16,
                        max_batches_per_domain=100,
                    )

                    raw = _unwrap(model)
                    # 1. Python — code completion (Code Replay 35%)
                    print(generate(raw,
                            "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        current = min((d, n) for n, d in distances.items() if n not in visited)[1]\n        visited.add(current)\n        for neighbor, weight in graph[current]:",
                            config.device, max_tokens=120, temp=0.3))
                    # 2. Code Understanding — explain what code does (Educational Code 15%)
                    print(generate(raw,
                            "# What does this function compute?\ndef mystery(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Answer: This function computes the",
                            config.device, max_tokens=120, temp=0.3))
                    # 3. CS Knowledge — REST API concepts (CS Knowledge 18%)
                    print(generate(raw,
                            "Question: What is the difference between PUT and PATCH in RESTful APIs?\n\nAnswer:",
                            config.device, max_tokens=150, temp=0.4))
                    # 4. Rust — systems programming (Code Replay 35%)
                    print(generate(raw,
                            "use std::collections::HashMap;\n\nfn word_count(text: &str) -> HashMap<&str, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {",
                            config.device, max_tokens=100, temp=0.3))
                    # 5. TypeScript — typed web (Code Replay 35%)
                    print(generate(raw,
                            "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n\nasync function fetchUsers(apiUrl: string): Promise<User[]> {\n  const response = await fetch(apiUrl);\n  if (!response.ok) {",
                            config.device, max_tokens=100, temp=0.3))
                    model.train()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    meta_data = {
                        "step": optim_step,
                        "train_loss": avg_accum_loss,
                        "val_loss": val_loss,
                    }
                    dataloader_state = train_data.get_state()

                    # ── Async checkpoint save ──
                    # Wait for any previous async save to finish first
                    if _save_thread is not None:
                        _save_thread.join()
                    _save_thread = save_checkpoint_async(
                        base_dir, optim_step,
                        model_data=_unwrap(model).state_dict(),
                        optimizer_data=optimizer.state_dict(),
                        scheduler_data=scheduler.state_dict(),
                        wandb_run=wandb_run,
                        dataloader_state=dataloader_state,
                        meta_data=meta_data,
                        phase=phase_num,
                        prefetch_loader=train_data,
                    )

                step_start_time = time.perf_counter()
                if optim_step >= phase_config.total_steps:
                    print(f"Reached total_steps ({phase_config.total_steps}). Phase complete.")
                    raise KeyboardInterrupt

        # Wait for any in-flight async save before exiting
        if _save_thread is not None:
            _save_thread.join()
        print(f"Phase {phase_num} training complete at optimizer step {optim_step}.")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving checkpoint at optimizer step {optim_step}...")
        # Wait for any in-flight async save first
        if _save_thread is not None:
            _save_thread.join()
        # Emergency save is SYNCHRONOUS to guarantee completion before exit
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
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    _cache_dir = str(Path.cwd() / ".dynamo_cache")
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

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
    phase_config = PHASE_2_CONFIG

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
            "optimizations": {
                "async_checkpointing": True,
                "flash_attention": True,
            },
        },
    )

    # ── Resume from checkpoint ─────────────────────────────
    start_step, dataloader_state, saved_phase = load_checkpoint(
        base_dir, model, optimizer, scheduler, device=config.device
    )

    if saved_phase != phase_config.phase_num:
        print(f"[Train] Phase transition detected: checkpoint is phase {saved_phase}, "
              f"config is phase {phase_config.phase_num}")
        # Reset optimizer (clear stale Adam momentum from previous phase)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
        )
        # Reset step counter — Phase 2 starts from step 0
        start_step = 0
        dataloader_state = None
        print(f"[Train] Fresh optimizer + scheduler for Phase {phase_config.phase_num}")

    if start_step > 0 and saved_phase == phase_config.phase_num:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', phase_config.peak_lr)
        scheduler = create_phase_scheduler(optimizer, phase_config, last_epoch=start_step - 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Scheduler] Rebuilt from config and fast-forwarded to optimizer step {start_step}")
        print(f"[Scheduler] Current LR: {current_lr:.6e}")
        print(f"[Scheduler] Remaining steps: {phase_config.total_steps - start_step}")

    # ── Compile model ──────────────────────────────────────
    torch._dynamo.config.capture_scalar_outputs = True
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
            model, optimizer, scheduler,
            train_data, val_data, wandb_run, phase_config,
            base_dir, start_step=start_step,
        )
    except KeyboardInterrupt:
        pass
    finally:
        wandb_run.finish()
