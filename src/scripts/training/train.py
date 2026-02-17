import torch
import math
import warnings
import os
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
  }, step=train_step, commit=False)
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
        start_step:   Step to resume from (0 = fresh).
    """
    step = start_step
    meta_data = None
    grad_accumulation_step = phase_config.grad_accum_steps
    val_interval = phase_config.val_interval
    patience = phase_config.patience
    phase_num = phase_config.phase_num

    try:
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        optimizer.zero_grad()

        resume_offset = 1 if start_step > 0 else 0
        for i, batch in enumerate(tqdm(train_data, desc=f"Phase {phase_num} Training")):
            step = i + start_step + resume_offset
            batch = batch.to(config.device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
                loss_value = loss.item()
            loss = loss / grad_accumulation_step
            loss.backward()
            metrics = {
                "train/loss": loss_value,
                "train/lr": scheduler.get_last_lr()[0],
                "train/ppl": math.exp(min(loss_value, 10)),
                "train/phase": phase_num,
            }

            if (step + 1) % grad_accumulation_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), phase_config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                metrics["train/grad_norm"] = grad_norm.item()

                for layer_idx, layer in enumerate(model.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
                        moe = layer.mlp
                        if moe.total_tokens > 0:
                            moe_metrics = moe.get_wandb_metrics()
                            for k, v in moe_metrics.items():
                                metrics[f"moe/layer_{layer_idx}/{k}"] = v
                            moe.reset_expert_counts()

            wandb_run.log(metrics, step=step)
            if (step + 1) % 1000 == 0:
                print(f"Step : {step+1} , Loss : {loss_value:.4f}")

            if (step + 1) % val_interval == 0:
                val_loss = validation(
                    model, criterion, val_data, train_step=step,
                    wandb_run=wandb_run, phase_config=phase_config,
                )
                print(generate(model,
                        "The old clock in the hallway stopped at midnight, and when I touched it a hidden drawer slid open revealing...",
                        config.device, max_tokens=60, temp=0.8))
                print(generate(model,
                        "Explain like I'm five: how does a battery make electricity?",
                        config.device, max_tokens=80, temp=0.3))
                print(generate(model,
                        "Write a Python function that reverses a string and explain its time complexity in one paragraph.",
                        config.device, max_tokens=120, temp=0.2))
                print(generate(model,
                        "Customer: I received a damaged package yesterday and the item is broken. Agent:",
                        config.device, max_tokens=80, temp=0.4))
                print(generate(model,
                        "In 200-250 words, argue for investing in renewable energy for economic growth. Cite one realistic-sounding statistic and label it as an example (do not invent specific study names).",
                        config.device, max_tokens=250, temp=0.5))
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    meta_data = {
                        "step": step,
                        "train_loss": loss_value,
                        "val_loss": val_loss,
                    }
                    dataloader_state = train_data.get_state()
                    save_checkpoint(
                        base_dir, step,
                        model_data=model.state_dict(),
                        optimizer_data=optimizer.state_dict(),
                        scheduler_data=scheduler.state_dict(),
                        wandb_run=wandb_run,
                        dataloader_state=dataloader_state,
                        meta_data=meta_data,
                        phase=phase_num,
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early Stopping triggered at step : {step}")
                        break

        print(f"Phase {phase_num} training complete at step {step}.")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving checkpoint at step {step}...")
        dataloader_state = train_data.get_state()
        save_checkpoint(
            base_dir, step,
            model_data=model.state_dict(),
            optimizer_data=optimizer.state_dict(),
            scheduler_data=scheduler.state_dict(),
            wandb_run=wandb_run,
            dataloader_state=dataloader_state,
            meta_data=meta_data,
            phase=phase_num,
        )
        print(f"[Interrupt] Checkpoint saved successfully.")
        raise  



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        project="828_testing_h200",
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
        scheduler = create_phase_scheduler(optimizer, phase_config)
        if start_step > 0:
            # Fast-forward scheduler to the saved step
            for _ in range(start_step):
                scheduler.step()

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
