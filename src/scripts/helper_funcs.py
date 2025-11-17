import os
import torch
import wandb

def get_base_dir():
    if os.environ.get("project-828_BASE_DIR"):
        project_828_dir = os.environ.get("project-828_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir,".cache")
        project_828_dir = os.path.join(cache_dir,"project_828_weights")
    os.makedirs(project_828_dir,exist_ok=True)
    return project_828_dir

def save_checkpoint(ckpt_dir,step,model_data,optimizer_data,scheduler_data,wandb_run,meta_data=None):
    os.makedirs(ckpt_dir,exist_ok=True)
    # Define paths to the same base directory
    model_path = os.path.join(ckpt_dir,f"model_{step:05d}.pt")
    optimizer_path = os.path.join(ckpt_dir,f"optim_{step:05d}.pt")
    scheduler_path = os.path.join(ckpt_dir,f"scheduler_{step:05d}.pt")
    
    # Prepare checkpoint data with metadata
    checkpoint_data = {
        "model": model_data,
        "optimizer": optimizer_data,
        "scheduler": scheduler_data,
        "step": step
    }
    if meta_data is not None:
        checkpoint_data.update(meta_data)

    # Save state dict to assigned paths
    torch.save(model_data,model_path)
    torch.save(optimizer_data,optimizer_path)
    torch.save(scheduler_data,scheduler_path)

    art_name = f"model-checkpoint-{step:06d}"
    artifact = wandb.Artifact(art_name,type = "model")    
    artifact.add_file(model_path)
    artifact.add_file(optimizer_path)
    artifact.add_file(scheduler_path)
    wandb_run.log_artifact(artifact)
