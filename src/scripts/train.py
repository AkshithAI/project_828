import torch
import math
import warnings
import os
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from .configs import config 
from ..models.model import GPT
from .tokenizer import tokenizer
from .dataloader import train_data,val_data
from ..models.model_flash_attn import GPT_FLASH
from .helper_funcs import get_base_dir,save_checkpoint,load_checkpoint
from transformers import get_cosine_schedule_with_warmup
from ..models.weight_init import init_gpt_model, count_parameters
from .inference import generate

@torch.inference_mode()
def validation(model,criterion):
  model.eval()
  total_val_loss = 0
  steps = 0
  for step,batch in enumerate(val_data):
    with autocast(device_type = "cuda",dtype = torch.bfloat16):
        batch = batch.to(config.device,non_blocking=True).long()
        labels = batch[:,:-1].contiguous()
        targets = batch[:,1:].contiguous()
        logits = model(labels)
        val_loss = criterion(logits.view(-1,logits.shape[-1]),targets.view(-1))
    wandb_run.log({
        "val/loss" : val_loss.item(),
        "val/step": step
    })
    steps += 1
    if (steps + 1) % 1000 == 0:
        print(f"Step : {steps+1} , Loss : {val_loss}")
    total_val_loss += val_loss.item()
    if steps == 5000:
      break
  return total_val_loss / max(1,steps)

def train(config, start_step=0):
    model.train()
    grad_accumulation_step = 16
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    optimizer.zero_grad()

    for step,batch in enumerate(tqdm(train_data, initial=start_step, desc="Training")):
        if step < start_step:
            continue
            
        batch = batch.to(config.device,non_blocking=True).long()
        inputs = batch[:,:-1].contiguous()
        targets = batch[:,1:].contiguous()
        with autocast(device_type = "cuda",dtype = torch.bfloat16):
            logits = model(inputs)
            loss = criterion(logits.view(-1,logits.shape[-1]),targets.view(-1))
            loss_value = loss.item()
        loss = loss / grad_accumulation_step
        loss.backward()
        if (step+1) % grad_accumulation_step == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            wandb_run.log({"train/grad_norm": grad_norm.item()})
            
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
                    moe = layer.mlp
                    if moe.total_tokens > 0:
                        metrics = moe.get_wandb_metrics()
                        wandb_run.log({
                            f"moe/layer_{layer_idx}/{k}": v for k, v in metrics.items()
                        })
                        
                        if (step + 1) % 1000 == 0:
                            print(f"\nLayer {layer_idx} MoE:")
                            print(f"  Total routed tokens: {moe.total_tokens}")
                            print(f"  Expert counts: {dict(enumerate(moe.expert_counts.tolist()))}")
                            print(f"  Load balance: {metrics['load_balance_score']:.1f}%")
                            
                            with open("expert_routing.txt", "a") as f:
                                f.write(f"\nStep {step+1} - Layer {layer_idx} MoE:\n")
                                for i in range(moe.num_experts):
                                    f.write(f"    Expert {i}: {metrics[f'expert_{i}']:.2f}%\n")
                        
                        moe.reset_expert_counts()

        wandb_run.log({
          "train/loss" : loss_value,
          "train/lr": scheduler.get_last_lr()[0],
          "train/step": step,
          "train/ppl": math.exp(min(loss_value, 10)),  
        })
        if (step + 1) % 1000 == 0:
            print(f"Step : {step+1} , Loss : {loss_value:.4f}")
        if (step+1) % 25000 == 0:
            val_loss = validation(model,criterion)
            print(generate(model,
                     "The old clock in the hallway stopped at midnight, and when I touched it a hidden drawer slid open revealing...",
                     config.device,max_tokens=60,temp=0.8))
            print(generate(model,
                     "Explain like I'm five: how does a battery make electricity?",
                     config.device,max_tokens=80,temp=0.3))
            print(generate(model,
                     "Write a Python function that reverses a string and explain its time complexity in one paragraph.",
                     config.device,max_tokens=120,temp=0.2))
            print(generate(model,
                     "Customer: I received a damaged package yesterday and the item is broken. Agent:",
                     config.device,max_tokens=80,temp=0.4))
            print(generate(model,
                     "In 200–250 words, argue for investing in renewable energy for economic growth. Cite one realistic-sounding statistic and label it as an example (do not invent specific study names).", 
                     config.device,max_tokens=250,temp=0.5))
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                meta_data = {
                    "step" : step,
                    "train_loss" : loss.item(),
                    "val_loss" : val_loss
                }
                save_checkpoint(
                    base_dir,
                    step,
                    model_data=model.state_dict(),
                    optimizer_data=optimizer.state_dict(),
                    scheduler_data=scheduler.state_dict(),
                    wandb_run=wandb_run,
                    meta_data=meta_data
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping triggered at step : {step}")
                    break
          
    wandb_run.finish()


if __name__ == '__main__' : 
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    base_dir = get_base_dir()
    use_flash_attn = True
    if use_flash_attn:
        model = GPT_FLASH(config,"cuda")
    else:
        model = GPT(config,"cuda")
    
    # Initialize model weights
    init_gpt_model(model, config)
    count_parameters(model)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=config.learning_rate,
      betas=(0.9, 0.95),
      weight_decay=0.01,
      eps=1e-8 
    )
    scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=2000,
      num_training_steps=100000,
      num_cycles=0.5 
    )
    wandb_run = wandb.init(
        entity = "akshithmarepally-akai",
        project = "828_testing_5090",
        config = {
            "architecture" : "GPT",
            "dataset" : "allenai/c4",
            "configs" : config,
        }
    )
    
    start_step = load_checkpoint(base_dir, model, optimizer, scheduler, device=config.device)
    
    train(config, start_step=start_step)
