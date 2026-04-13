import os
import argparse
import warnings
import time
import deepspeed
import wandb
import torch
import math
from torch.amp import autocast
import torch.distributed as dist
import torch.nn as nn
from ...models.model import GPT
from ...models.model_flash_attn import GPT_FLASH
from ...models.weight_init import init_gpt_model, count_parameters
from ..helper_funcs import get_base_dir
from ..tokenizer import tokenizer
from ..inference import generate
from ..dist_dataloader import create_phase_dataloaders
from .schedulers import create_phase_scheduler
from .validate_domains import validate_domains
from ..configs.model_config import config, PHASE_1_CONFIG, PHASE_2_CONFIG


@torch.inference_mode()
def validation(model_engine, criterion, val_loader, wandb_run, phase_config):
    """
    Run validation across all ranks, synchronize loss via all_reduce.

    Wraps forward pass in autocast to match training precision (bf16).
    Only logs the final averaged loss on rank 0 (no per-step W&B noise).
    """
    model_engine.eval()
    local_rank = model_engine.local_rank
    total_val_loss = 0
    steps = 0

    for batch in val_loader:
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            batch = batch.to(local_rank, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()

            logits = model_engine(inputs)
            val_loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))

        steps += 1
        if (steps + 1) % 1000 == 0 and dist.get_rank() == 0:
            print(f"Val Step: {steps+1}, Loss: {val_loss.item():.4f}")

        total_val_loss += val_loss.item()
        if steps == phase_config.val_steps:
            break

    # Synchronize validation loss across all ranks
    total_val_loss_tensor = torch.tensor(total_val_loss, device=local_rank)
    steps_tensor = torch.tensor(steps, dtype=torch.long, device=local_rank)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)
    avg_val_loss = total_val_loss_tensor.item() / max(1, steps_tensor.item())

    if dist.get_rank() == 0 and wandb_run is not None:
        wandb_run.log({
            "val/loss": avg_val_loss,
            "val/ppl": math.exp(min(avg_val_loss, 10)),
        }, commit=False)

    return avg_val_loss


def train_phase(
    model_engine, criterion, scheduler,
    train_data, val_data, wandb_run, phase_config,
    base_dir, start_step=0
):
    """
    Distributed training phase using DeepSpeed engine.

    Gradient accumulation and all-reduce are fully managed by DeepSpeed.
    We call ``backward()`` and ``step()`` on every micro-batch — DeepSpeed
    internally tracks the accumulation counter and only fires the optimizer
    + all-reduce at the accumulation boundary.  We use
    ``model_engine.is_gradient_accumulation_boundary()`` to detect when a
    real optimizer step occurred (for logging / validation / checkpointing).
    """
    model_engine.train()
    meta_data = None
    best_val_loss = float('inf')
    global_step = start_step
    phase_num = phase_config.phase_num
    grad_accumulation_steps = phase_config.grad_accum_steps
    val_interval = phase_config.val_interval
    seq_len = config.max_context_len
    micro_bs = phase_config.micro_batch_size
    local_rank = model_engine.local_rank

    # Effective tokens per full optimizer step (across all micro-batches, this rank)
    tokens_per_step = micro_bs * seq_len * grad_accumulation_steps

    # Estimate model FLOPs per forward pass
    base_model = model_engine.module
    n_params = sum(p.numel() for p in base_model.parameters())
    flops_per_token = 6 * n_params  # 6N per token (fwd + bwd)
    gpu_peak_flops = 989.4e12  # H200 bf16 peak FLOPS

    # ── Loop state ──
    accum_loss = 0.0
    step_start_time = time.perf_counter()

    try:
        for step, batch in enumerate(train_data):
            batch = batch.to(local_rank, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model_engine(inputs)
                loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1),
                )

            # DeepSpeed handles loss scaling, gradient accumulation,
            # all-reduce, and optimizer step internally.
            model_engine.backward(loss)
            model_engine.step()

            accum_loss = accum_loss + loss.detach()

            # ── Optimizer step boundary ──
            # DeepSpeed only fires the actual optimizer update at the
            # accumulation boundary.  All logging / validation / checkpoint
            # logic lives inside this gate.
            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                avg_accum_loss = (accum_loss / grad_accumulation_steps).item()

                if dist.get_rank() == 0 and wandb_run is not None:
                    # ── Throughput & hardware metrics ──
                    step_elapsed = time.perf_counter() - step_start_time
                    tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
                    step_flops = flops_per_token * tokens_per_step
                    mfu = step_flops / (step_elapsed * gpu_peak_flops) if step_elapsed > 0 else 0.0

                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

                    log_dict = {
                        "train/loss": avg_accum_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/ppl": math.exp(min(avg_accum_loss, 10)),
                        "train/phase": phase_num,
                        "train/grad_norm": model_engine.get_global_grad_norm() or 0.0,
                        "perf/tokens_per_sec": tps,
                        "perf/mfu": mfu,
                        "perf/vram_allocated_gb": allocated_gb,
                        "perf/vram_reserved_gb": reserved_gb,
                        "perf/step_time_sec": step_elapsed,
                    }

                    # Log MoE expert routing statistics
                    expert_logging(model_engine, log_dict)

                    wandb_run.log(log_dict, step=8 * global_step)

                # Reset accumulation state
                accum_loss = 0.0

                if global_step % 100 == 0 and dist.get_rank() == 0:
                    print(
                        f"Step: {global_step}, Loss: {avg_accum_loss:.4f}, "
                        f"TPS: {tps:.0f}, MFU: {mfu:.2%}, "
                        f"VRAM: {allocated_gb:.2f}/{reserved_gb:.2f} GB"
                    )

                if global_step % val_interval == 0:
                    dist.barrier()
                    val_loss = validation(
                        model_engine, criterion, val_data,
                        wandb_run=wandb_run, phase_config=phase_config,
                    )

                    # ── Domain-specific validation ──
                    if dist.get_rank() == 0:
                        validate_domains(
                            model=model_engine.module,
                            wandb_run=wandb_run,
                            train_step=global_step,
                            phase_config=phase_config,
                            device=torch.device(f"cuda:{local_rank}"),
                            batch_size=16,
                            max_batches_per_domain=100,
                        )

                        run_eval(model_engine)

                    model_engine.train()

                    # ── Checkpoint ──
                    if dist.get_rank() == 0:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                        meta_data = {
                            "step": global_step,
                            "train_loss": avg_accum_loss,
                            "val_loss": val_loss,
                        }

                    # DeepSpeed checkpoint: saves model + optimizer + scheduler
                    # sharded state across all ranks.
                    tag = f"step_{global_step:06d}"
                    dataloader_state = train_data.get_state() if dist.get_rank() == 0 else None
                    client_state = {
                        "global_step": global_step,
                        "phase": phase_num,
                    }
                    if dataloader_state is not None:
                        client_state["dataloader_state"] = dataloader_state
                    if meta_data is not None:
                        client_state["meta_data"] = meta_data

                    model_engine.save_checkpoint(
                        str(base_dir), tag=tag, client_state=client_state
                    )
                    if dist.get_rank() == 0:
                        print(f"[Checkpoint] DeepSpeed checkpoint saved: {tag}")

                step_start_time = time.perf_counter()
                if global_step >= phase_config.total_steps:
                    print(f"Reached total_steps ({phase_config.total_steps}). Phase complete.")
                    break

        print(f"Phase {phase_num} training complete at optimizer step {global_step}.")
    except KeyboardInterrupt:
        if dist.get_rank() == 0:
            print(f"\n[Interrupt] Saving emergency checkpoint at step {global_step}...")
        tag = f"interrupt_step_{global_step:06d}"
        client_state = {
            "global_step": global_step,
            "phase": phase_num,
        }
        if dist.get_rank() == 0:
            try:
                client_state["dataloader_state"] = train_data.get_state()
            except Exception:
                pass
        model_engine.save_checkpoint(
            str(base_dir), tag=tag, client_state=client_state
        )
        if dist.get_rank() == 0:
            print(f"[Interrupt] Checkpoint saved: {tag}")
        raise


def _unwrap(model):
    """Return the raw module behind DeepSpeed engine (or the model itself)."""
    return model.module if hasattr(model, 'module') else model


def expert_logging(model_engine, log_dict):
    raw = _unwrap(model_engine)
    for layer_idx, layer in enumerate(raw.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
            moe = layer.mlp
            if moe.total_tokens > 0:
                moe_metrics = moe.get_wandb_metrics()
                log_dict.update({
                    f"moe/layer_{layer_idx}/{k}": v
                    for k, v in moe_metrics.items()
                })
                moe.reset_expert_counts()


def run_eval(model_engine):
    """Run qualitative text generation samples (rank 0 only)."""
    raw = _unwrap(model_engine)
    device = torch.device(f"cuda:{model_engine.local_rank}")
    # 1. Python — graph algorithm (Source Code 50%)
    print(generate(raw,
            "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        current = min((d, n) for n, d in distances.items() if n not in visited)[1]\n        visited.add(current)\n        for neighbor, weight in graph[current]:",
            device, max_tokens=120, temp=0.3))
    # 2. C++ — systems programming (Source Code 50%)
    print(generate(raw,
            "#include <iostream>\n#include <thread>\n#include <mutex>\n\nstd::mutex mtx;\nint shared_counter = 0;\n\nvoid increment(int times) {\n    for (int i = 0; i < times; ++i) {\n        std::lock_guard<std::mutex> lock(mtx);\n        shared_counter++;\n    }\n}\n\nint main() {",
            device, max_tokens=100, temp=0.3))
    # 3. Math — clean step-by-step (Math/Reasoning 10% — finemath)
    print(generate(raw,
            "To find the area of a triangle with vertices at (1,2), (4,6), and (7,1), we can use the coordinate geometry formula.\n\nArea = (1/2) |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|\n\nSubstituting the values:",
            device, max_tokens=120, temp=0.2))
    # 4. CS Q&A — StackExchange style (CS/Engineering 22%)
    print(generate(raw,
            "Question: What is the difference between a process and a thread in operating systems?\n\nAnswer: A process is an independent execution unit with its own memory space,",
            device, max_tokens=120, temp=0.4))
    # 5. Code task — OpenCodeInstruct style (CS/Engineering 22%)
    print(generate(raw,
            "Write a Python function that takes a list of intervals and merges all overlapping intervals.\n\ndef merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]",
            device, max_tokens=120, temp=0.3))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description='DeepSpeed GPT Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    config.local_rank = local_rank
    config.global_rank = int(os.environ.get('RANK', 0))

    base_dir = get_base_dir("checkpoints")

    # ── Model ──────────────────────────────────────────────
    use_flash_attn = True
    model = GPT_FLASH(config, "cuda") if use_flash_attn else GPT(config, "cuda")

    # Initialize model weights
    init_gpt_model(model, config)
    count_parameters(model)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters()
    )

    # ── Phase selection ────────────────────────────────────
    phase_config = PHASE_1_CONFIG

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    scheduler = create_phase_scheduler(optimizer, phase_config)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # ── W&B ────────────────────────────────────────────────
    if rank == 0:
        wandb_run = wandb.init(
            entity="akshithmarepally-akai",
            project="828_distributed_pretraining",
            config={
                "architecture": "GPT_FLASH_MoE",
                "phase": phase_config.phase_name,
                "world_size": world_size,
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
                    "flash_attention": True,
                    "deepspeed": True,
                },
            },
        )
    else:
        wandb_run = None


    # ── Resume from DeepSpeed checkpoint ───────────────────
    start_step = 0
    dataloader_state = None
    saved_phase = 1

    # Try to load the latest DeepSpeed checkpoint
    _, client_state = model_engine.load_checkpoint(str(base_dir))
    if client_state is not None:
        start_step = client_state.get("global_step", 0)
        dataloader_state = client_state.get("dataloader_state", None)
        saved_phase = client_state.get("phase", 1)
        if rank == 0:
            print(f"[Resume] Loaded DeepSpeed checkpoint at step {start_step}, phase {saved_phase}")

    if saved_phase == 2 and phase_config.phase_num != 2:
        print("[Train] Checkpoint is from Phase 2 — switching to PHASE_2_CONFIG")
        phase_config = PHASE_2_CONFIG
        for pg in optimizer.param_groups:
            pg["lr"] = phase_config.peak_lr

    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', phase_config.peak_lr)
        scheduler = create_phase_scheduler(optimizer, phase_config, last_epoch=start_step - 1)
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(f"[Scheduler] Rebuilt from config and fast-forwarded to optimizer step {start_step}")
            print(f"[Scheduler] Current LR: {current_lr:.6e}")
            print(f"[Scheduler] Remaining steps: {phase_config.total_steps - start_step}")

    # ── Dataloaders ────────────────────────────────────────
    train_data, val_data = create_phase_dataloaders(
        phase_config=phase_config,
        train_state=dataloader_state,
        val_repo_id="HuggingFaceFW/fineweb-edu",
        rank=rank,
        world_size=world_size,
        batch_size_val=16,
        context_length=config.max_context_len,
    )

    # ── Train ──────────────────────────────────────────────
    try:
        train_phase(
            model_engine, criterion, scheduler,
            train_data, val_data, wandb_run, phase_config,
            base_dir, start_step
        )
    except KeyboardInterrupt:
        if rank == 0:
            print("\n[INFO] Training interrupted by user. Cleaning up...")
    finally:
        if rank == 0 and wandb_run is not None:
            wandb_run.finish()

        dist.barrier()
        dist.destroy_process_group()

        if rank == 0:
            print("[INFO] Process group destroyed. Cleanup complete.")
