import os
import torch
import wandb
from pathlib import Path

def get_base_dir(sub_folder : str):
    """
    Get Base Directory of Project Folder

    Args:
        None
    
    Returns:
        PosixPath: Path to Project Folder
    """
    base_dir = Path.cwd()   
    out_dir = base_dir / sub_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory")

    return out_dir.resolve()

def get_latest_checkpoint_step(base_dir):
    """
    Find the latest checkpoint step number.
    
    Args:
        base_dir: Path to checkpoint directory
        
    Returns:
        int: Latest step number, or None if no checkpoints exist
    """
    base_dir = Path(base_dir)
    model_files = list(base_dir.glob("model_*.pt"))
    
    if not model_files:
        return None
    
    latest_step = max(
        int(p.stem.split("_")[1]) for p in model_files
    )
    return latest_step

def get_checkpoint_paths(base_dir, step):
    """
    Get paths for model, optimizer, scheduler, and dataloader checkpoints at a given step.
    
    Args:
        base_dir: Path to checkpoint directory
        step: Step number
        
    Returns:
        tuple: (model_path, optimizer_path, scheduler_path, dataloader_path)
    """
    base_dir = Path(base_dir)
    model_path = base_dir / f"model_{step:05d}.pt"
    optim_path = base_dir / f"optim_{step:05d}.pt"
    scheduler_path = base_dir / f"scheduler_{step:05d}.pt"
    dataloader_path = base_dir / f"dataloader_{step:05d}.pt"
    return model_path, optim_path, scheduler_path, dataloader_path

def load_checkpoint(base_dir, model, optimizer=None, scheduler=None, device="cuda"):
    """
    Load the latest checkpoint for model, optimizer, and scheduler.
    
    Args:
        base_dir: Path to checkpoint directory
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to
        
    Returns:
        tuple: (start_step, dataloader_state, phase) 
               - step number to resume from (0 if no checkpoint)
               - dataloader state dict (None if no checkpoint or no dataloader state)
               - phase number (1 if no metadata)
    """
    base_dir = Path(base_dir)
    latest_step = get_latest_checkpoint_step(base_dir)
    
    if latest_step is None:
        print("No checkpoint found. Starting from scratch.")
        return 0, None, 1
    
    model_path, optim_path, scheduler_path, dataloader_path = get_checkpoint_paths(base_dir, latest_step)
    
    # Load model
    if model_path.exists():
        print(f"Loading model checkpoint from {model_path}")
        model_state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(model_state)
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        return 0, None, 1
    
    # Load optimizer
    if optimizer is not None and optim_path.exists():
        print(f"Loading optimizer checkpoint from {optim_path}")
        optim_state = torch.load(optim_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(optim_state)
    elif optimizer is not None:
        print(f"Warning: Optimizer checkpoint not found at {optim_path}")
    
    # Load scheduler
    if scheduler is not None and scheduler_path.exists():
        print(f"Loading scheduler checkpoint from {scheduler_path}")
        scheduler_state = torch.load(scheduler_path, map_location=device, weights_only=True)
        scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None:
        print(f"Warning: Scheduler checkpoint not found at {scheduler_path}")
    
    # Load dataloader state
    dataloader_state = None
    phase = 1
    if dataloader_path.exists():
        print(f"Loading dataloader checkpoint from {dataloader_path}")
        dataloader_state = torch.load(dataloader_path, map_location="cpu", weights_only=False)
        phase = dataloader_state.get("phase", 1)
        ds_states = dataloader_state.get("dataset_states", {})
        total_docs = sum(s.get("documents_processed", 0) for s in ds_states.values())
        print(f"  -> Mixer state: phase={phase}, "
              f"{dataloader_state.get('samples_yielded', 0)} samples, "
              f"{total_docs} total documents across {len(ds_states)} datasets")
        for ds_name, ds_s in ds_states.items():
            print(f"     {ds_name}: {ds_s.get('documents_processed', 0)} docs, "
                  f"{len(ds_s.get('buffer_tokens', []))} buffered tokens")
    else:
        print(f"Warning: Dataloader checkpoint not found at {dataloader_path}")
    
    print(f"Resumed from step {latest_step} (phase {phase})")
    return latest_step, dataloader_state, phase


def save_checkpoint(ckpt_dir, step, model_data, optimizer_data, scheduler_data, wandb_run, 
                    dataloader_state=None, meta_data=None, phase=1):
    """
    Save model state dict with meta data
    
    Args:
        ckpt_dir: Path to Checkpoint Directory
        step: Global Step for checkpoint 
        model_data: model's state info
        optimizer_data: optimizer's state info
        scheduler_data: scheduler's state info
        wandb_run: wandb object to save the session details
        dataloader_state: dataloader's state info (dict from ResumableDataLoader.get_state())
        meta_data: meta data
        phase: current training phase number
    
    Returns:
        None
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"model_{step:05d}.pt")
    optimizer_path = os.path.join(ckpt_dir, f"optim_{step:05d}.pt")
    scheduler_path = os.path.join(ckpt_dir, f"scheduler_{step:05d}.pt")
    dataloader_path = os.path.join(ckpt_dir, f"dataloader_{step:05d}.pt")

    # For now not tracked
    checkpoint_data = {
        "model": model_data,
        "optimizer": optimizer_data,
        "scheduler": scheduler_data,
        "step": step,
        "phase": phase,
    }
    if meta_data is not None:
        checkpoint_data.update(meta_data)

    torch.save(model_data, model_path)
    torch.save(optimizer_data, optimizer_path)
    torch.save(scheduler_data, scheduler_path)
    
    # Save dataloader state (inject phase number for resume detection)
    if dataloader_state is not None:
        dataloader_state["phase"] = phase
        torch.save(dataloader_state, dataloader_path)
        ds_states = dataloader_state.get("dataset_states", {})
        total_docs = sum(s.get("documents_processed", 0) for s in ds_states.values())
        print(f"[Checkpoint] Saved mixer state (phase {phase}): "
              f"{dataloader_state.get('samples_yielded', 0)} samples, "
              f"{total_docs} docs across {len(ds_states)} datasets")

    art_name = f"model-checkpoint-test-{step:06d}" 
    artifact = wandb.Artifact(art_name, type="model")    
    artifact.add_file(model_path)
    artifact.add_file(optimizer_path)
    artifact.add_file(scheduler_path)
    if dataloader_state is not None and os.path.exists(dataloader_path):
        artifact.add_file(dataloader_path)
    wandb_run.log_artifact(artifact)

if __name__ == '__main__':
    get_base_dir("checkpoints")