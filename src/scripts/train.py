from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.amp import autocast
import wandb
from .configs import config 
from ..models.model import GPT
from ..models.model_flash_attn import GPT_FLASH
from ..models.weight_init import init_gpt_model, count_parameters
import torch.nn as nn
from .dataloader import train_data,val_data
from .tokenizer import tokenizer
from .helper_funcs import get_base_dir,save_checkpoint
import os

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

def train(config):
    model.train()
    grad_accumulation_step = 16
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    for step,batch in enumerate(tqdm(train_data)):
        optimizer.zero_grad()
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
            wandb_run.log({"train/grad_norm": grad_norm.item()})
        wandb_run.log({
          "train/loss" : loss_value,
          "train/lr": scheduler.get_last_lr()[0],
          "train/step": step,
          "train/ppl": math.exp(min(loss_value, 10)),  
        })
        if (step + 1) % 1000 == 0:
            print(f"Step : {step+1} , Loss : {loss_value:.4f}")
        if (step+1) % 50000 == 0:
            val_loss = validation(model,criterion)
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    base_dir = get_base_dir()
    use_flash_attn = False
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
            "dataset" : "codeparrot/codeparrot-clean",
            "configs" : config,
        }
    )
    train(config)
