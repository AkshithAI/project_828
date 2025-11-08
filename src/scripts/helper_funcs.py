import os
import torch
import json

def get_base_dir():
    if os.environ.get("project-828_BASE_DIR"):
        project_828_dir = os.environ.get("project-828_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir,".cache")
        project_828_dir = os.path.join(cache_dir,"project_828")
    os.makedirs(project_828_dir,exist_ok=True)
    return project_828_dir

def save_checkpoint(ckpt_dir,step,model_data,optimizer_data,scheduler_data,meta_data,artifact,wandb_run):
    os.makedirs(ckpt_dir,exist_ok=True)
    # Define paths to the same base directory
    model_path = os.path.join(ckpt_dir,f"model_{step:05d}.pt")
    optimizer_path = os.path.join(ckpt_dir,f"optim_{step:05d}.pt")
    scheduler_path = os.path.join(ckpt_dir,f"scheduler_{step:05d}.pt")
    meta_path = os.path.join(ckpt_dir,f"meta_{step:05d}.json")

    # Save state dict to assigned paths
    torch.save(model_data,model_path)
    torch.save(optimizer_data,optimizer_path)
    torch.save(scheduler_data,scheduler_path)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
        
    artifact.add_file(model_path)
    artifact.add_file(optimizer_path)
    artifact.add_file(scheduler_path)
    artifact.add_file(meta_path)
    wandb_run.log_artifact(artifact)
