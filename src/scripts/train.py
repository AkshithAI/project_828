from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.amp import GradScaler, autocast
import wandb
from .configs import config 
from ..models.model import GPT
import torch.nn as nn
from .dataloader import train_data,val_data
from .tokenizer import tokenizer
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
    grad_accumulation_step = 16
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    save_path = os.path.join(config.model_dir,f"model_{step:05d}.pth") 
    torch.save({
                    "step" : step,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "scheduler_state_dict" : scheduler.state_dict(),
                    "scaler_state_dict" : scaler.state_dict(),
                    "train_loss" : loss.item(),
                    "val_loss" : val_loss,
                },save_path)        
    artifact.add_file(save_path)
    wandb_run.log_artifact(artifact)
    for step,batch in enumerate(tqdm(train_data)):
        batch = batch.to(config.device,non_blocking=True).long()
        inputs = batch[:,:-1].contiguous()
        targets = batch[:,1:].contiguous()
        with autocast(device_type = "cuda",dtype = torch.bfloat16):
            logits = model(inputs)
            loss = criterion(logits.view(-1,logits.shape[-1]),targets.view(-1))
            loss = loss / grad_accumulation_step
        scaler.scale(loss).backward()
        if (step+1) % grad_accumulation_step == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        wandb_run.log({
        "train/loss" : loss.item() * grad_accumulation_step,
        "train/step": step
        })
        if (step + 1) % 100 == 0:
            print(f"Step : {step+1} , Loss : {loss.item()}")
        if (step+1) % 10000 == 0:
            val_loss = validation(model,criterion)
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config.model_dir,f"model_{step:05d}.pth")
                torch.save({
                    "step" : step,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "scheduler_state_dict" : scheduler.state_dict(),
                    "scaler_state_dict" : scaler.state_dict(),
                    "train_loss" : loss.item(),
                    "val_loss" : val_loss,
                },save_path)
                artifact.add_file(save_path)
                wandb_run.log_artifact(artifact)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping triggered at step : {step}")
                    break
          
    wandb_run.finish()


def count_params(model):
  return sum([p.numel() for p in model.parameters() if p.requires_grad])

if __name__ == '__main__' : 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = GPT(config,"cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,betas=(0.9, 0.95),weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=1000000)
    scaler = GradScaler()
    wandb_run = wandb.init(
        entity = "akshithmarepally-akai",
        project = "828_testing_5090",
        config = {
            "architecture" : "GPT",
            "dataset" : "codeparrot/codeparrot-clean",
            "configs" : config,
        }
    )
    artifact = wandb.Artifact("model-checkpoint",type = 'model')
    print(f"Total parameters : {count_params(model)}")
    train(config)