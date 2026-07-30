import torch
import math
import warnings
import os
import time
from pathlib import Path
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from ..tokenizer import tokenizer
from ..dataloader import create_phase_dataloaders
from ..helper_funcs import (
    get_base_dir, save_checkpoint, save_checkpoint_async,
    load_checkpoint, get_gpu_peak_flops, get_training_logger,
)
from .schedulers import create_phase_scheduler
from ...models.weight_init import initialize_gpt_model, count_parameters
from ..inference import generate
from .validate_domains import validate_domains
from .telemetry import (
    compute_routing_telemetry,
    compute_weight_update_ratios,
    compute_hidden_state_telemetry,
)


def nvtx_push(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def train_phase(
    model, optimizer, scheduler,
    train_data, wandb_run, phase_config,
    base_dir, start_step=0, eval_suite_interval=0,
    profile=False, profile_warmup_steps=5, profile_active_steps=10, profile_exit=True,
):
    """
    Train one phase.  Supports exact resumption via the ResumableDataLoader
    and MixerState checkpoint. Optionally profiles a window of steps using CUDA Profiler API + NVTX.

    Args:
        model, optimizer, scheduler: the usual.
        train_data:   ``ResumableDataLoader`` wrapping a ``WeightedMixerDataset``.
        wandb_run:    Active W&B run.
        phase_config: ``PhaseConfig`` for this phase.
        base_dir:     Checkpoint directory path.
        start_step:   Optimizer step to resume from (0 = fresh).
        eval_suite_interval: Run eval suite every N optimizer steps (0 = disabled).
    """
    optim_step = start_step
    meta_data = None
    grad_accumulation_steps = phase_config.grad_accum_steps
    val_interval = phase_config.val_interval
    phase_num = phase_config.phase_num
    seq_len = config.max_context_len
    micro_bs = phase_config.micro_batch_size
    tokens_per_step = micro_bs * seq_len * grad_accumulation_steps
    eos_id = tokenizer.eos_token_id

    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)

    # Estimate model FLOPs per forward pass 
    n_params = sum(p.numel() for p in _unwrap(model).parameters())
    flops_per_token = 6 * n_params  
    gpu_peak_flops = get_gpu_peak_flops(config.device)

    # ── Profiling State ──
    profiling_started = False
    profiling_stopped = False
    profile_target_start = start_step + profile_warmup_steps
    profile_target_stop = profile_target_start + profile_active_steps
    if profile:
        print(f"[NSYS Profile] Profiling enabled: Warmup until step {profile_target_start}, profile {profile_active_steps} steps until step {profile_target_stop}.")

    # ── Async checkpoint thread handle ──
    _save_thread = None

    try:
        model.train()
        best_domain_loss = float('inf')
        optimizer.zero_grad()
        accum_loss = 0.0
        micro_count = 0
        last_inputs = None  # Saved for hidden state telemetry at val_interval
        step_start_time = time.perf_counter()

        for i, batch in enumerate(tqdm(train_data, desc=f"Phase {phase_num} Training")):
            # Start CUDA Profiler at warmup step
            if profile and not profiling_started and optim_step >= profile_target_start:
                print(f"\n[NSYS Profile] >>> Starting CUDA Profiler at optimizer step {optim_step} <<<")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                profiling_started = True

            nvtx_push("data_to_device")
            batch = batch.to(config.device, non_blocking=True).long()
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            nvtx_pop()

            nvtx_push("forward")
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux_loss = model(inputs)
            nvtx_pop()

            nvtx_push("loss_calc")
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1),
                )
            nvtx_pop()

            nvtx_push("backward")
            total_loss = loss + aux_loss
            (total_loss / grad_accumulation_steps).backward()
            nvtx_pop()

            accum_loss = accum_loss + loss.detach()  
            micro_count += 1
            last_inputs = inputs.detach()  # Keep reference for telemetry

            if micro_count == grad_accumulation_steps:
                optim_step += 1

                nvtx_push("grad_clip")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), phase_config.grad_clip
                )
                nvtx_pop()

                nvtx_push("optimizer_step")
                optimizer.step()
                nvtx_pop()

                nvtx_push("scheduler_step")
                scheduler.step()
                optimizer.zero_grad()
                nvtx_pop()

                # Stop CUDA Profiler after profile active steps
                if profile and profiling_started and not profiling_stopped and optim_step >= profile_target_stop:
                    print(f"\n[NSYS Profile] >>> Stopping CUDA Profiler at optimizer step {optim_step} <<<")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                    profiling_stopped = True
                    if profile_exit:
                        print("[NSYS Profile] Targeted profiling window finished. Exiting training run.")
                        break

                raw_model = _unwrap(model)

                avg_accum_loss = (accum_loss / grad_accumulation_steps).item()  
                # ── Throughput & hardware metrics ──
                step_elapsed = time.perf_counter() - step_start_time
                tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
                step_flops = flops_per_token * tokens_per_step
                mfu = step_flops / (step_elapsed * gpu_peak_flops) if step_elapsed > 0 else 0.0

                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

                metrics = {
                    "train/loss": avg_accum_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/ppl": math.exp(min(avg_accum_loss, 10)),
                    "train/phase": phase_num,
                    "train/grad_norm": grad_norm.item(),
                    "perf/tokens_per_sec": tps,
                    "perf/mfu": mfu,
                    "perf/vram_allocated_gb": allocated_gb,
                    "perf/vram_reserved_gb": reserved_gb,
                    "perf/step_time_sec": step_elapsed,
                }

                # ── Expert usage logging ──
                raw = _unwrap(model)
                for layer_idx, layer in enumerate(raw.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
                        moe = layer.mlp
                        if moe.total_tokens > 0:
                            moe_metrics = moe.get_wandb_metrics()
                            metrics.update({
                                f"moe/layer_{layer_idx}/{k}": v
                                for k, v in moe_metrics.items()
                            })
                            moe.reset_expert_counts()


                # ── Telemetry: routing entropy + weight update ratios (every step) ──
                current_lr = scheduler.get_last_lr()[0]
                include_hidden = (optim_step % val_interval == 0)
                nvtx_push("telemetry_diagnostics")
                telemetry_metrics = raw_model.get_telemetry_diagnostics(
                    input_ids=last_inputs if include_hidden else None,
                    optimizer=optimizer,
                    lr=current_lr,
                    include_hidden_states=include_hidden,
                )
                metrics.update(telemetry_metrics)

                if optim_step % val_interval == 0:
                    attn_diag = raw.get_attention_diagnostics()
                    metrics.update(attn_diag)
                nvtx_pop()

                wandb_run.log(metrics, step=grad_accumulation_steps * optim_step)
                accum_loss = 0.0
                micro_count = 0

                if optim_step % 100 == 0:
                    print(
                        f"Step : {optim_step} , Loss : {avg_accum_loss:.4f} , "
                        f"TPS : {tps:.0f} , MFU : {mfu:.2%} , "
                        f"VRAM : {allocated_gb:.2f}/{reserved_gb:.2f} GB"
                    )

                if optim_step % val_interval == 0:
                    # ── Domain-specific validation ──
                    validate_domains(
                        model=model,
                        wandb_run=wandb_run,
                        train_step=optim_step,
                        phase_config=phase_config,
                        device=config.device,
                        batch_size=16,
                        max_batches_per_domain=100,
                    )
                if optim_step % 1000 == 0:
                    raw = _unwrap(model)
                    print(f"\n--- GENERATION TESTS AT STEP {optim_step} ---")
                    # 1. Python (Source Code 20%)
                    print("--- 1. Python (20%) ---")
                    print(generate(raw,
                            "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        current = min((d, n) for n, d in distances.items() if n not in visited)[1]\n        visited.add(current)\n        for neighbor, weight in graph[current]:",
                            config.device, max_tokens=120, temp=0.3))
                    # 2. Javascript (Source Code 8%)
                    print("--- 2. Javascript (8%) ---")
                    print(generate(raw,
                            "function debounce(func, wait) {\n    let timeout;\n    return function(...args) {\n        const context = this;\n        clearTimeout(timeout);\n        timeout = setTimeout(() => {",
                            config.device, max_tokens=100, temp=0.3))
                    # 3. Java (Source Code 7%, NEW in Phase 2)
                    print("--- 3. Java (7%) ---")
                    print(generate(raw,
                            "import java.util.*;\n\npublic class LRUCache<K, V> {\n    private final int capacity;\n    private final Map<K, V> cache;\n\n    public LRUCache(int capacity) {\n        this.capacity = capacity;\n        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {\n            @Override\n            protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {",
                            config.device, max_tokens=120, temp=0.3))
                    # 4. TypeScript (Source Code 5%)
                    print("--- 4. TypeScript (5%) ---")
                    print(generate(raw,
                            "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n\nasync function fetchUsers(apiUrl: string): Promise<User[]> {\n  const response = await fetch(apiUrl);\n  if (!response.ok) {",
                            config.device, max_tokens=100, temp=0.3))
                    # 5. C++ (Source Code 6%)
                    print("--- 5. C++ (6%) ---")
                    print(generate(raw,
                            "#include <vector>\n#include <algorithm>\n\ntemplate<typename T>\nclass MinHeap {\n    std::vector<T> data;\n    void sift_up(int idx) {\n        while (idx > 0) {\n            int parent = (idx - 1) / 2;\n            if (data[idx] < data[parent]) {",
                            config.device, max_tokens=100, temp=0.3))
                    # 6. Go (Source Code 5%)
                    print("--- 6. Go (5%) ---")
                    print(generate(raw,
                            "package main\n\nimport (\n\t\"fmt\"\n\t\"net/http\"\n)\n\nfunc helloHandler(w http.ResponseWriter, r *http.Request) {\n\tfmt.Fprintf(w, \"Hello, World!\")\n}\n\nfunc main() {\n\thttp.HandleFunc(\"/\", helloHandler)\n\terr := http.ListenAndServe(\":8080\", nil)\n\tif err != nil {",
                            config.device, max_tokens=100, temp=0.3))
                    # 7. Rust (Source Code 5%)
                    print("--- 7. Rust (5%) ---")
                    print(generate(raw,
                            "use std::sync::{Arc, Mutex};\nuse std::thread;\n\nfn main() {\n    let counter = Arc::new(Mutex::new(0));\n    let mut handles = vec![];\n    for _ in 0..10 {\n        let counter = Arc::clone(&counter);\n        let handle = thread::spawn(move || {\n            let mut num = counter.lock().unwrap();",
                            config.device, max_tokens=120, temp=0.3))
                    # 8. Tiny-Codes (Educational Code 9%)
                    print("--- 8. Tiny-Codes (9%) ---")
                    print(generate(raw,
                            "# What does this function compute?\ndef mystery(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Answer: This function computes the",
                            config.device, max_tokens=120, temp=0.3))
                    # 9. StackExchange (CS Knowledge 12%)
                    print("--- 9. StackExchange (12%) ---")
                    print(generate(raw,
                            "Question: What is the difference between a stack and a queue, and when would you use each?\n\nAnswer:",
                            config.device, max_tokens=150, temp=0.4))
                    # 10. DCLM-Edu (CS/Edu Web 5%)
                    print("--- 10. DCLM-Edu (5%) ---")
                    print(generate(raw,
                            "Tutorial: Understanding Big O Notation and Binary Search\n\nBinary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list that could contain the item, until you've narrowed down the possible locations to just one. The time complexity of binary search is",
                            config.device, max_tokens=120, temp=0.3))
                    # 11. FineWeb-Edu (General Knowledge / Edu Web 15%)
                    print("--- 11. FineWeb-Edu (15%) ---")
                    print(generate(raw,
                            "Explain the process of photosynthesis in plants and why it is important for the ecosystem.\n\nPhotosynthesis is a chemical process that occurs in plants, algae, and some bacteria. It converts light energy into chemical energy. The general equation is:",
                            config.device, max_tokens=150, temp=0.3))
                    # 12. Wikipedia (General Knowledge 3%)
                    print("--- 12. Wikipedia (3%) ---")
                    print(generate(raw,
                            "Alan Turing (23 June 1912 - 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine. During the Second World War, Turing worked for",
                            config.device, max_tokens=150, temp=0.3))
                    print("-----------------------------------------\n")
                    model.train()
                    meta_data = {
                        "step": optim_step,
                        "train_loss": avg_accum_loss,
                    }
                if optim_step % 500 == 0:
                    dataloader_state = train_data.get_state()

                    # ── Async checkpoint save ──
                    # Wait for any previous async save to finish first
                    if _save_thread is not None:
                        _save_thread.join()
                    _save_thread = save_checkpoint_async(
                        base_dir, optim_step,
                        model_data=_unwrap(model).state_dict(),
                        optimizer_data=optimizer.state_dict(),
                        scheduler_data=scheduler.state_dict(),
                        wandb_run=wandb_run,
                        dataloader_state=dataloader_state,
                        meta_data=meta_data,
                        phase=phase_num,
                        prefetch_loader=train_data,
                    )

                # ── Eval Suite (comprehensive benchmarks) ──
                if eval_suite_interval > 0 and optim_step % eval_suite_interval == 0:
                    try:
                        from ..data.eval_suite import run_training_eval
                        raw = _unwrap(model)
                        run_training_eval(
                            raw, config.device,
                            wandb_run=wandb_run,
                            train_step=optim_step,
                            grad_accum_steps=grad_accumulation_steps,
                        )
                        model.train()
                    except Exception as eval_exc:
                        print(f"[EvalSuite] Error during eval: {eval_exc}")
                        model.train()

                step_start_time = time.perf_counter()
                if optim_step >= phase_config.total_steps:
                    print(f"Reached total_steps ({phase_config.total_steps}). Phase complete.")
                    raise KeyboardInterrupt

        # Wait for any in-flight async save before exiting
        if _save_thread is not None:
            _save_thread.join()
        tlog = get_training_logger()
        tlog.logger.info(f"[TRAIN] Phase {phase_num} complete at step {optim_step}")
        tlog.flush()
        print(f"Phase {phase_num} training complete at optimizer step {optim_step}.")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving checkpoint at optimizer step {optim_step}...")
        # Wait for any in-flight async save first
        if _save_thread is not None:
            _save_thread.join()
        # Emergency save is SYNCHRONOUS to guarantee completion before exit
        dataloader_state = train_data.get_state()
        save_checkpoint(
            base_dir, optim_step,
            model_data=_unwrap(model).state_dict(),
            optimizer_data=optimizer.state_dict(),
            scheduler_data=scheduler.state_dict(),
            wandb_run=wandb_run,
            dataloader_state=dataloader_state,
            meta_data=meta_data,
            phase=phase_num,
        )
        print(f"[Interrupt] Checkpoint saved successfully.")
        tlog = get_training_logger()
        tlog.logger.info(f"[INTERRUPT] Checkpoint saved at step {optim_step}")
        tlog.flush()
        raise  
    except Exception as exc:
        print(f"\n[CRASH] {type(exc).__name__}: {exc}")
        print(f"[CRASH] Attempting emergency checkpoint save at optimizer step {optim_step}...")
        try:
            # Wait for any in-flight async save first
            if _save_thread is not None:
                _save_thread.join(timeout=30)
            dataloader_state = train_data.get_state()
            save_checkpoint(
                base_dir, optim_step,
                model_data=_unwrap(model).state_dict(),
                optimizer_data=optimizer.state_dict(),
                scheduler_data=scheduler.state_dict(),
                wandb_run=wandb_run,
                dataloader_state=dataloader_state,
                meta_data=meta_data,
                phase=phase_num,
            )
            print(f"[CRASH] Emergency checkpoint saved successfully at step {optim_step}.")
        except Exception as save_exc:
            print(f"[CRASH] Emergency save FAILED: {save_exc}")
        tlog = get_training_logger()
        tlog.logger.error(f"[CRASH] {type(exc).__name__}: {exc}")
        tlog.flush()
        raise



def _unwrap(model):
    """Return the raw module behind torch.compile (or the model itself)."""
    return getattr(model, '_orig_mod', model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Pretraining script with Nsight Systems profiling")
    parser.add_argument("--model", type=str, default="model_improv", choices=["model_adv", "model_improv"],
                        help="Which model architecture to use (default: model_improv)")
    parser.add_argument("--profile", action="store_true", help="Enable Nsight Systems profiling window via CUDA profiler API")
    parser.add_argument("--profile-warmup-steps", type=int, default=5, help="Optimizer steps to wait before starting CUDA profiler")
    parser.add_argument("--profile-active-steps", type=int, default=10, help="Optimizer steps to profile before stopping CUDA profiler")
    parser.add_argument("--profile-no-exit", action="store_true", help="Do not exit training after profiling range finishes")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile during profiling/execution")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, skip checkpoint loading")
    cli_args, _ = parser.parse_known_args()

    # ── Conditional model + config import ─────────────────
    if cli_args.model == "model_adv":
        from ..configs.new_model_config import config, PRETRAINING_PHASE_CONFIG
        from ...models.model_adv import GPT_FLASH, build_optimizer_param_groups
        PHASE_1_CONFIG = PRETRAINING_PHASE_CONFIG
        PHASE_2_CONFIG = PRETRAINING_PHASE_CONFIG
    else:
        from ..configs.model_config import config, PHASE_1_CONFIG, PHASE_2_CONFIG
        from ...models.model_improv import GPT_FLASH
        from ...models.model_adv import build_optimizer_param_groups

    profile_enabled = cli_args.profile or (os.environ.get("PROFILE_NSYS", "0").lower() in ("1", "true", "yes"))

    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    _cache_dir = str(Path.cwd() / ".dynamo_cache")
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir

    torch.set_float32_matmul_precision('high')       
    torch.backends.cudnn.benchmark = True            
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    base_dir = get_base_dir("checkpoints")

    # ── Initialize training logger (writes to checkpoints/training.log) ──
    tlog = get_training_logger(log_dir=str(base_dir))
    tlog.logger.info(f"[INIT] Training started | phase={PHASE_2_CONFIG.phase_name}")

    # ── Model ──────────────────────────────────────────────
    print(f"[Train] Using model: {cli_args.model}")
    model = GPT_FLASH(config, "cuda")

    # model_adv has its own reset_parameters() called in __init__;
    # model_improv uses the external weight_init module
    if cli_args.model != "model_adv":
        initialize_gpt_model(
            model=model,
            model_config=config,
        )
    count_parameters(model)

    # ── Phase selection ────────────────────────────────────
    phase_config = PHASE_2_CONFIG

    # ── Eval Suite interval ───────────────────────────────
    try:
        eval_suite_interval = int(getattr(phase_config, "eval_suite_interval", 0))
    except (ValueError, TypeError):
        eval_suite_interval = 0
    if eval_suite_interval > 0:
        print(f"[Train] Eval suite will run every {eval_suite_interval} steps")

    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(
            model,
            config.weight_decay,
        ),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = create_phase_scheduler(optimizer, phase_config)

    # ── W&B ────────────────────────────────────────────────
    wandb_run = wandb.init(
        entity="akshithmarepally-akai",
        project="828_pretraining_h200",
        group=phase_config.phase_name,
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
            "optimizations": {
                "async_checkpointing": True,
                "flash_attention": True,
            },
        },
    )

    # ── Resume from checkpoint ─────────────────────────────
    if getattr(cli_args, 'no_resume', False):
        print("[Train] --no-resume: skipping checkpoint, starting fresh")
        start_step, dataloader_state, saved_phase = 0, None, phase_config.phase_num
    else:
        start_step, dataloader_state, saved_phase = load_checkpoint(
            base_dir, model, optimizer, scheduler, device=config.device
        )

    if saved_phase != phase_config.phase_num:
        tlog.log_phase_transition(saved_phase, phase_config.phase_num, start_step)
        print(f"[Train] Phase transition detected: checkpoint is phase {saved_phase}, "
              f"config is phase {phase_config.phase_num}")
        # Reset optimizer (clear stale Adam momentum from previous phase)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
        )

        scheduler = create_phase_scheduler(optimizer, phase_config)
        # Reset step counter — Phase 2 starts from step 0
        start_step = 0
        dataloader_state = None
        print(f"[Train] Fresh optimizer + scheduler for Phase {phase_config.phase_num}")

    if start_step > 0 and saved_phase == phase_config.phase_num:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', phase_config.peak_lr)
        scheduler = create_phase_scheduler(optimizer, phase_config, last_epoch=start_step - 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Scheduler] Rebuilt from config and fast-forwarded to optimizer step {start_step}")
        print(f"[Scheduler] Current LR: {current_lr:.6e}")
        print(f"[Scheduler] Remaining steps: {phase_config.total_steps - start_step}")

    # ── Compile model ──────────────────────────────────────
    if not cli_args.no_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "default" if profile_enabled else "max-autotune-no-cudagraphs"
        print(f"[Train] Compiling model with mode='{compile_mode}'...")
        model = torch.compile(model, mode=compile_mode)
    else:
        print("[Train] torch.compile disabled via --no-compile")

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
            model, optimizer, scheduler,
            train_data, wandb_run, phase_config,
            base_dir, start_step=start_step,
            eval_suite_interval=eval_suite_interval,
            profile=profile_enabled,
            profile_warmup_steps=cli_args.profile_warmup_steps,
            profile_active_steps=cli_args.profile_active_steps,
            profile_exit=not cli_args.profile_no_exit,
        )
    except KeyboardInterrupt:
        pass
    finally:
        tlog = get_training_logger()
        tlog.logger.info("[SHUTDOWN] Training session ended")
        tlog.flush()
        wandb_run.finish()
