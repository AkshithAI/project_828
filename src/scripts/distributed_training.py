import argparse
import warnings
import deepspeed
import json
import wandb
import torch
import math
import torch.distributed as dist
import torch.nn as nn
from .configs import config
from ..models.model import GPT
from ..models.model_flash_attn import GPT_FLASH
from ..models.weight_init import init_gpt_model
from .dist_dataloader import val_data, train_loader_0, train_loader_1
from .tokenizer import tokenizer
import os
from .helper_funcs import get_base_dir


@torch.inference_mode()
def validation(model, criterion, val_loader, wandb_run=None):
    model.eval()
    total_val_loss = 0
    steps = 0
    
    for step, batch in enumerate(val_loader):
        batch = batch.to(model.local_rank).long()
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        
        logits = model(inputs)
        val_loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
        
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({
                "val/loss": val_loss.item(),
                "val/step": step
            })
        
        steps += 1
        if (steps + 1) % 1000 == 0 and dist.get_rank() == 0:
            print(f"Step: {steps+1}, Loss: {val_loss:.4f}")
        
        total_val_loss += val_loss.item()
        if steps == 5000:
            break
    
    # Synchronize validation loss across all ranks
    total_val_loss_tensor = torch.tensor(total_val_loss, device=model.local_rank)
    dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
    avg_val_loss = total_val_loss_tensor.item() / (dist.get_world_size() * max(1, steps))
    
    return avg_val_loss


def train(model_engine, train_loader, criterion, val_loader, wandb_run=None, checkpoint_dir=None):
    model_engine.train()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    global_step = 0
    
    import time
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        batch = batch.to(model_engine.local_rank).long()
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        
        logits = model_engine(inputs)
        loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss_value = loss.item()
        
        model_engine.backward(loss)
        model_engine.step()
        
        if dist.get_rank() == 0 and wandb_run is not None:
            # Calculate tokens per second
            elapsed = time.time() - start_time
            tokens_per_sec = (global_step + 1) * 128 * 2048 / elapsed if elapsed > 0 else 0
            
            log_dict = {
                "train/loss": loss_value,
                "train/step": global_step,
                "train/ppl": math.exp(min(loss_value, 10)),
                "train/lr": model_engine.get_lr()[0],
                "perf/tokens_per_sec": tokens_per_sec,
            }
            
            # Log GPU memory every 100 steps
            if global_step % 100 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                log_dict["perf/gpu_memory_allocated_gb"] = allocated
                log_dict["perf/gpu_memory_reserved_gb"] = reserved
            
            wandb_run.log(log_dict)
        
        if (global_step + 1) % 1000 == 0 and dist.get_rank() == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Step: {global_step+1}, Loss: {loss_value:.4f}, GPU Memory: {allocated:.2f}GB")
        
        if (global_step + 1) % 50000 == 0:
            dist.barrier()  
            val_loss = validation(model_engine, criterion, val_loader, wandb_run)
            model_engine.train()
            
            if dist.get_rank() == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_id = f"step_{global_step}"
                    
                    # Save checkpoint using DeepSpeed
                    if checkpoint_dir is not None:
                        ckpt_dir = os.path.join(checkpoint_dir, ckpt_id)
                        model_engine.save_checkpoint(checkpoint_dir, tag=ckpt_id)
                        
                        # Upload to WandB if available
                        if wandb_run is not None:
                            artifact = wandb.Artifact(
                                name=f"model-checkpoint-{ckpt_id}",
                                type="model",
                                description=f"Model checkpoint at step {global_step} with val_loss {val_loss:.4f}",
                                metadata={
                                    "step": global_step,
                                    "val_loss": val_loss,
                                    "train_loss": loss_value,
                                    "best_val_loss": best_val_loss,
                                }
                            )
                            artifact.add_dir(ckpt_dir)
                            wandb_run.log_artifact(artifact)
                            
                            wandb_run.log({
                                "val/best_loss": best_val_loss,
                                "val/checkpoint_step": global_step
                            })
                        
                        print(f"Checkpoint saved to {ckpt_dir}")
                    
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early Stopping triggered at step: {global_step}")
                        break
        
        global_step += 1
    
    if dist.get_rank() == 0 and wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description='DeepSpeed GPT Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    
    config.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    config.global_rank = int(os.environ.get('RANK', 0))
    config.seq_len = 2048  
    
    base_dir = get_base_dir()
    
    use_flash_attn = False
    if use_flash_attn:
        model = GPT_FLASH(config, "cuda")
    else:
        model = GPT(config, "cuda")
        
    init_gpt_model(model, config)
    ds_config_path = os.path.join(os.path.dirname(__file__), "ds-config.json")
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        train_loader = train_loader_0
    elif rank == 1:
        train_loader = train_loader_1
    else:
        train_loader = train_loader_0
        if rank == 0:
            print(f"Warning: More than 2 GPUs detected. Rank {rank} using default dataloader.")
    
    if rank == 0:
        wandb_run = wandb.init(
            entity="akshithmarepally-akai",
            project="828_Distributed_testing_5090",
            config={
                "architecture": "GPT",
                "dataset": "codeparrot/codeparrot-clean",
                "world_size": world_size,
                "use_flash_attn": use_flash_attn,
                "configs": vars(config),
            }
        )
    else:
        wandb_run = None
    
    if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description='DeepSpeed GPT Training')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    
    config.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    config.global_rank = int(os.environ.get('RANK', 0))
    config.seq_len = 2048  
    
    base_dir = get_base_dir()
    
    use_flash_attn = False
    if use_flash_attn:
        model = GPT_FLASH(config, "cuda")
    else:
        model = GPT(config, "cuda")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters()
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        train_loader = train_loader_0
    elif rank == 1:
        train_loader = train_loader_1
    else:
        train_loader = train_loader_0
        if rank == 0:
            print(f"Warning: More than 2 GPUs detected. Rank {rank} using default dataloader.")
    
    if rank == 0:
        wandb_run = wandb.init(
            entity="akshithmarepally-akai",
            project="828_Distributed_testing_5090",
            config={
                "architecture": "GPT",
                "dataset": "codeparrot/codeparrot-clean",
                "world_size": world_size,
                "use_flash_attn": use_flash_attn,
                "configs": vars(config),
            }
        )
    else:
        wandb_run = None
    
    try:
        train(model_engine, train_loader, criterion, val_data, wandb_run, base_dir)
    except KeyboardInterrupt:
        if rank == 0:
            print("\n[INFO] Training interrupted by user. Cleaning up...")
    except Exception as e:
        if rank == 0:
            print(f"\n[ERROR] Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        if rank == 0 and wandb_run is not None:
            wandb_run.finish()
            
        dist.barrier()
        dist.destroy_process_group()
        
        if rank == 0:
            print("[INFO] Process group destroyed. Cleanup complete.")
