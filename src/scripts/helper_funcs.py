import os
import copy
import threading
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
    Save model state dict with meta data (synchronous).
    
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


# ── Async Checkpoint Saving ──────────────────────────────────────────────────

def _deep_copy_to_cpu(state_dict):
    """Recursively deep-copy a state dict, moving all tensors to CPU.
    
    This performs GPU → CPU transfer + clone in one pass so the GPU
    tensors can be freed / overwritten immediately after this call returns.
    """
    if isinstance(state_dict, dict):
        return {k: _deep_copy_to_cpu(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [_deep_copy_to_cpu(v) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(_deep_copy_to_cpu(v) for v in state_dict)
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.detach().cpu().clone()
    else:
        return copy.deepcopy(state_dict)


def _available_cpu_memory_gb():
    """Best-effort check of available CPU RAM in GB.
    
    Returns None if detection fails (safe fallback: caller proceeds).
    """
    # Linux (H200 training environment)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 * 1024)  # kB → GB
    except Exception:
        pass
    # psutil fallback (if installed)
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    return None  # unknown — caller should proceed optimistically


# Minimum free CPU RAM (GB) required to start an async save.
# The model + optimizer state copies need ~2.5 GB; we add headroom
# for the prefetch thread, tokenizer, and OS.
_MIN_ASYNC_SAVE_MEMORY_GB = 5.0


def save_checkpoint_async(ckpt_dir, step, model_data, optimizer_data, scheduler_data,
                          wandb_run, dataloader_state=None, meta_data=None, phase=1,
                          prefetch_loader=None):
    """Save checkpoint asynchronously in a background thread.
    
    1. Pauses the data-prefetch thread (if provided) to reduce CPU contention.
    2. Checks available CPU memory — falls back to synchronous save if low.
    3. Deep-copies all state dicts to CPU (fast, ~1-2s D2H transfer).
    4. Resumes prefetch, then spawns a background thread for disk I/O + W&B upload.
    5. Returns the thread handle so the caller can ``.join()`` before the
       next save to prevent overlapping writes.
    
    Args:
        Same as ``save_checkpoint``, plus:
        prefetch_loader: Optional ``PrefetchedDataLoader`` — its prefetch
            thread will be paused during the CPU snapshot to avoid memory
            and CPU contention, then automatically resumed.
    
    Returns:
        threading.Thread | None: The background save thread, or ``None``
        if a synchronous fallback was used (nothing to join).
    """
    import time

    # ── Safety: check available CPU memory ────────────────────────────────
    avail_gb = _available_cpu_memory_gb()
    if avail_gb is not None and avail_gb < _MIN_ASYNC_SAVE_MEMORY_GB:
        print(f"[Checkpoint] WARNING: Low CPU memory ({avail_gb:.1f} GB available, "
              f"{_MIN_ASYNC_SAVE_MEMORY_GB:.0f} GB required). "
              f"Falling back to synchronous save at step {step}.")
        save_checkpoint(
            ckpt_dir, step, model_data, optimizer_data, scheduler_data,
            wandb_run, dataloader_state, meta_data, phase,
        )
        return None  # no background thread

    # ── Step 1: Pause prefetch to avoid CPU / memory contention ───────────
    if prefetch_loader is not None:
        prefetch_loader.pause_prefetch()

    try:
        t0 = time.perf_counter()

        model_cpu = _deep_copy_to_cpu(model_data)
        optim_cpu = _deep_copy_to_cpu(optimizer_data)
        sched_cpu = _deep_copy_to_cpu(scheduler_data)
        dl_cpu = copy.deepcopy(dataloader_state) if dataloader_state is not None else None

        elapsed = time.perf_counter() - t0
        print(f"[Checkpoint] CPU snapshot took {elapsed:.2f}s (step {step})")
    finally:
        # ── Always resume prefetch, even if the snapshot fails ────────────
        if prefetch_loader is not None:
            prefetch_loader.resume_prefetch()

    # ── Step 2: Save in background thread (I/O-bound, low CPU contention) ─
    def _save():
        try:
            save_checkpoint(
                ckpt_dir, step, model_cpu, optim_cpu, sched_cpu,
                wandb_run, dl_cpu, meta_data, phase,
            )
            print(f"[Checkpoint] Async save completed (step {step})")
        except Exception as e:
            print(f"[Checkpoint] ERROR in async save (step {step}): {e}")

    thread = threading.Thread(target=_save, daemon=True)
    thread.start()
    return thread


def get_gpu_peak_flops(device=None):
    """
    Get the peak FLOPS (BF16/FP16 dense) for the current GPU device.
    Supports common datacenter and consumer GPUs.
    If no GPU is found or the GPU is unrecognized, returns a sensible default (H200: 989.4e12).
    """
    default_flops = 989.4e12  # H200 bf16 peak FLOPS
    if not torch.cuda.is_available():
        return default_flops
        
    try:
        if device is None:
            device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device).upper()
        
        # Mapping common GPU models to their dense FP16/BF16 peak tensor FLOPS
        gpu_flops_map = {
            "H200": 989.4e12,
            "H100": 989.4e12,  # SXM is 989.4e12, PCIe is 756e12
            "A100": 312e12,
            "A10G": 125e12,
            "A30": 165e12,
            "L40": 362e12,
            "L4": 121e12,
            "RTX 4090": 330e12,
            "RTX 3090": 71e12,
            "RTX 4080": 197e12,
            "RTX 3080": 68e12,
            "V100": 125e12,
            "T4": 65e12,
        }
        
        # Check for matches
        for model, flops in gpu_flops_map.items():
            if model in device_name:
                # Handle specific H100 PCIe vs SXM distinction if possible
                if model == "H100" and "PCIE" in device_name:
                    return 756e12
                return flops
                
        # If not explicitly mapped, check for general architectures or compute capability
        prop = torch.cuda.get_device_properties(device)
        major = prop.major
        minor = prop.minor
        
        # Fallback based on compute capability (rough estimations of peak BF16/FP16 FLOPS)
        if major == 9:  # Hopper
            return 989.4e12
        elif major == 8:
            if minor == 9:  # Ada Lovelace (e.g. L40, RTX 4080/4090 class)
                return 330e12
            elif minor == 6:  # Ampere consumer (RTX 30-series)
                return 71e12
            elif minor == 0:  # Ampere datacenter (A100)
                return 312e12
        elif major == 7:  # Volta/Turing
            return 125e12
            
    except Exception as e:
        print(f"[get_gpu_peak_flops] Error determining GPU peak flops: {e}")
        
    return default_flops


if __name__ == '__main__':
    get_base_dir("checkpoints")