import torch
import math
import warnings
import os
import time
import threading
from pathlib import Path
import wandb
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from ..tokenizer import tokenizer
from ..dataloader import create_phase_dataloaders, compute_packed_attention_metadata
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
    AsyncTelemetryLogger,
)


def nvtx_push(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


class _SyncHunt:
    """Diagnostic: records every synchronizing CUDA op with its source location.

    Enabled for the first N microbatches via SYNC_HUNT_MICROBATCHES=N.
    Uses torch.cuda.set_sync_debug_mode("warn") and captures the resulting
    warnings so hidden .item()/.cpu()-style pipeline drains inside
    forward/backward can be attributed to exact file:line call sites.

    Scope note: step-boundary syncs (deferred metric reads, event waits) are
    known and intentional — this hunt only wraps forward_and_loss + backward,
    where ANY sync is a bug in a GPU-bound training loop.
    """

    def __init__(self):
        self.records = []
        self._ctx = None

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self.records = self._ctx.__enter__()
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        return self

    def __exit__(self, *exc):
        torch.cuda.set_sync_debug_mode("default")
        self._ctx.__exit__(*exc)
        return False

    def report(self):
        from collections import Counter

        locs = Counter()
        for w in self.records:
            if "synchronizing" in str(w.message).lower():
                locs[f"{w.filename}:{w.lineno}"] += 1
        total = sum(locs.values())
        print(f"\n{'=' * 70}")
        print(f"SYNC HUNT RESULTS — {total} hidden syncing ops caught in this microbatch")
        print(f"{'=' * 70}")
        for loc, count in locs.most_common(40):
            print(f"{count:5d}x  {loc}")
        print("=" * 70 + "\n")


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

    # If the scheduler is token-based, drive it by cumulative non-padding tokens
    # rather than optimizer steps 
    scheduler_is_token_based = getattr(scheduler, "token_based", False)

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

    # ── Sync-hunt diagnostic (SYNC_HUNT_MICROBATCHES=N) ──
    _sync_hunt_target = int(os.environ.get("SYNC_HUNT_MICROBATCHES", "0"))
    _sync_hunt_done = 0
    if _sync_hunt_target > 0:
        print(f"[SyncHunt] Scanning the first {_sync_hunt_target} microbatches for hidden CUDA syncs...")

    # ── Async telemetry logger (non-blocking background thread) ──
    async_logger = AsyncTelemetryLogger(wandb_run)

    # ── Async checkpoint thread handle ──
    _save_thread = None

    # ── Optimization: Secondary CUDA stream for host→device data transfers ──
    # Allows next-batch DMA copy to overlap with optimizer.step() on the default stream.
    transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    transfer_done = torch.cuda.Event() if torch.cuda.is_available() else None

    # ── Optimization: CUDA event for deferred .item() reads ──
    # Instead of calling .item() (full pipeline drain) on grad_norm/loss/tokens
    # in the current step, we record an event after the optimizer step and read
    # the values at the start of the NEXT step, when the GPU has long finished.
    _prev_step_event = None          # CUDA event from previous optimizer step
    _prev_grad_norm = None           # GPU tensor: grad norm from previous step
    _prev_accum_loss = None          # GPU tensor: accumulated loss from previous step
    _prev_accum_tokens = None        # GPU tensor: accumulated tokens from previous step
    _prev_optim_step = None          # int: optimizer step number for deferred metrics
    _prev_step_elapsed = None        # float: wall-clock time for previous step

    # ── Optimization: Async metrics collection thread ──
    # Expert usage + telemetry diagnostics run in a background thread so the
    # GPU can start the next forward pass immediately.
    _metrics_thread = None

    try:
        model.train()
        best_domain_loss = float('inf')
        optimizer.zero_grad()
        accum_loss = 0.0
        micro_count = 0
        last_inputs = None  # Saved for hidden state telemetry at val_interval
        step_start_time = time.perf_counter()

        for i, batch in enumerate(tqdm(train_data, desc=f"Phase {phase_num} Training")):
            nvtx_push("batch_fetch")
            nvtx_pop()  # batch already fetched by iterator; marks queue-wait time
            # Start CUDA Profiler at warmup step
            if profile and not profiling_started and optim_step >= profile_target_start:
                print(f"\n[NSYS Profile] >>> Starting CUDA Profiler at optimizer step {optim_step} <<<")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                profiling_started = True

            nvtx_push("data_to_device")
            # The mixer packs several documents in each fixed-length row.  Build
            # block-diagonal FlashAttention metadata before moving the batch to
            # the GPU, then reset RoPE positions at every document boundary.
            # Metadata must describe ``inputs`` (the shifted batch), not the
            # unshifted row, because the model never sees its final token.
            packed_inputs = batch[:, :-1].contiguous()
            cu_seqlens_cpu, position_ids_cpu, max_seqlen = compute_packed_attention_metadata(
                packed_inputs, eos_id,
            )
            # ── Optimization: H2D copy on a dedicated transfer stream ──
            # The copy engine (CE) executes the DMA concurrently with the
            # previous microbatch's backward kernels still draining on the
            # default stream. The compute stream waits on `transfer_done`
            # before consuming, so correctness is preserved by construction.
            #
            # NOTE: this ONLY overlaps if `batch` is page-locked (pinned).
            # A pageable source makes cudaMemcpyAsync behave synchronously
            # AND respect stream order — the host blocks here until every
            # previously queued kernel finishes (nsys shows this as a long
            # red cudaMemcpyAsync in the CUDA API row while the GPU-side
            # copy itself takes only ~8µs). Pin defensively as a fallback;
            # a no-op if the upstream pin_memory=True path already worked.
            if torch.cuda.is_available() and not batch.is_pinned():
                batch = batch.pin_memory()
            with torch.cuda.stream(transfer_stream):
                gpu_batch = batch.to(config.device, non_blocking=True).long()
                cu_seqlens = cu_seqlens_cpu.to(config.device, non_blocking=True)
                position_ids = position_ids_cpu.to(config.device, non_blocking=True)
                transfer_done.record(transfer_stream)
            torch.cuda.current_stream(config.device).wait_event(transfer_done)
            inputs = gpu_batch[:, :-1].contiguous()
            targets = gpu_batch[:, 1:].contiguous()
            nvtx_pop()

            # Retain full [T, E] routing probabilities only on the microbatch that
            # feeds validation-interval telemetry, so entropy can be computed
            # without pinning the large tensor on every step.
            collect_routing = ((optim_step + 1) % val_interval == 0) and (
                micro_count == grad_accumulation_steps - 1
            )

            _hunt = None
            if _sync_hunt_done < _sync_hunt_target:
                _hunt = _SyncHunt()
                _hunt.__enter__()

            nvtx_push("forward_and_loss")
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    res = model(
                        inputs,
                        labels=targets,
                        collect_routing_telemetry=collect_routing,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                except TypeError:
                    # model_improv is a logits-only compatibility model.  It
                    # still needs the packing metadata even though it does not
                    # accept the model_adv loss/telemetry arguments.
                    res = model(
                        inputs,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )

                first, aux_loss = res
                if first.dim() == 0:
                    loss = first
                else:
                    loss = criterion(first.view(-1, first.shape[-1]), targets.view(-1))
            nvtx_pop()

            nvtx_push("backward")
            total_loss = loss + aux_loss
            (total_loss / grad_accumulation_steps).backward()
            nvtx_pop()

            if _hunt is not None:
                _hunt.__exit__(None, None, None)
                _sync_hunt_done += 1
                _hunt.report()
                if _sync_hunt_done == _sync_hunt_target:
                    print("[SyncHunt] Scan complete — unset SYNC_HUNT_MICROBATCHES for normal training.")

            accum_loss = accum_loss + loss.detach()  
            micro_count += 1
            last_inputs = inputs.detach()  # Keep reference for telemetry

            # Accumulate non-padding tokens for the token-based LR schedule.
            # Keep the running count on-device (a GPU tensor) and defer the single
            # .item() sync to the optimizer-step boundary, so we don't force a
            # device-to-host sync on every microbatch.
            if scheduler_is_token_based:
                step_tokens_tensor = (targets != eos_id).sum()
                if micro_count == 1:
                    accum_tokens_gpu = step_tokens_tensor
                else:
                    accum_tokens_gpu = accum_tokens_gpu + step_tokens_tensor

            if micro_count == grad_accumulation_steps:
                optim_step += 1

                # ────────────────────────────────────────────────────────
                # GPU work FIRST: launch grad_clip + optimizer immediately
                # so the GPU stays busy while CPU does deferred reads.
                # ────────────────────────────────────────────────────────
                nvtx_push("grad_clip")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    phase_config.grad_clip,
                    foreach=True if torch.cuda.is_available() else None,
                )
                nvtx_pop()

                # Guard against non-finite gradients — check on GPU to
                # avoid a CPU sync.
                grad_finite = torch.isfinite(grad_norm)

                nvtx_push("optimizer_step")
                optimizer.step()
                nvtx_pop()

                # Commit load-balancing bias updates only after a successful
                # optimizer step, keeping routing bias aligned with the
                # parameters that actually changed.
                _unwrap(model).commit_moe_bias_updates()

                nvtx_push("scheduler_step")
                if scheduler_is_token_based:
                    scheduler.step()
                else:
                    scheduler.step()
                optimizer.zero_grad()
                nvtx_pop()

                # ────────────────────────────────────────────────────────
                # DEFERRED .item() reads — runs while GPU processes the
                # optimizer kernels we just launched above.
                # ────────────────────────────────────────────────────────
                if _prev_step_event is not None:
                    nvtx_push("deferred_metrics_read")
                    _prev_step_event.synchronize()  # instant — GPU passed this point long ago

                    # Read previous step's GPU scalars (~free, stream already synced)
                    prev_grad_norm_val = _prev_grad_norm.item()
                    prev_avg_loss = (_prev_accum_loss / grad_accumulation_steps).item()
                    if _prev_accum_tokens is not None:
                        prev_tokens_val = int(_prev_accum_tokens.item())
                    else:
                        prev_tokens_val = None

                    prev_step_elapsed = _prev_step_elapsed
                    prev_tps = tokens_per_step / prev_step_elapsed if prev_step_elapsed > 0 else 0.0
                    prev_step_flops = flops_per_token * tokens_per_step
                    prev_mfu = prev_step_flops / (prev_step_elapsed * gpu_peak_flops) if prev_step_elapsed > 0 else 0.0

                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

                    prev_metrics = {
                        "train/loss": prev_avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/ppl": math.exp(min(prev_avg_loss, 10)),
                        "train/phase": phase_num,
                        "train/grad_norm": prev_grad_norm_val,
                        "perf/tokens_per_sec": prev_tps,
                        "perf/mfu": prev_mfu,
                        "perf/vram_allocated_gb": allocated_gb,
                        "perf/vram_reserved_gb": reserved_gb,
                        "perf/step_time_sec": prev_step_elapsed,
                    }

                    # Wait for any previous async metrics thread to finish
                    # before we read its results and log.
                    if _metrics_thread is not None:
                        _metrics_thread.join()
                        _metrics_thread = None

                    # Merge any async metrics that the background thread computed
                    # (stored in _async_metrics_result by the thread).
                    if _async_metrics_result:
                        prev_metrics.update(_async_metrics_result)
                        _async_metrics_result = {}

                    nvtx_push("async_log_enqueue")
                    async_logger.log(prev_metrics, step=grad_accumulation_steps * _prev_optim_step)
                    nvtx_pop()

                    if _prev_optim_step % 100 == 0:
                        print(
                            f"Step : {_prev_optim_step} , Loss : {prev_avg_loss:.4f} , "
                            f"TPS : {prev_tps:.0f} , MFU : {prev_mfu:.2%} , "
                            f"VRAM : {allocated_gb:.2f}/{reserved_gb:.2f} GB"
                        )
                    nvtx_pop()  # deferred_metrics_read

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

                # ────────────────────────────────────────────────────────
                # OPTIMIZATION 2: Async telemetry in background thread
                # ────────────────────────────────────────────────────────
                # Snapshot expert counts (clone GPU tensors) so the
                # background thread can safely read them while the main
                # thread continues to the next iteration.
                nvtx_push("telemetry_snapshot")
                raw_model = _unwrap(model)
                expert_snapshots = []
                for layer_idx, layer in enumerate(raw_model.layers):
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'get_wandb_metrics'):
                        moe = layer.mlp
                        if moe.total_tokens > 0:
                            # Snapshot the counts before resetting
                            snapshot = {
                                'layer_idx': layer_idx,
                                'expert_counts': moe.expert_counts.clone(),
                                'total_tokens': moe.total_tokens,
                                'top_k': moe.top_k,
                                'num_experts': moe.num_experts,
                            }
                            expert_snapshots.append(snapshot)
                            moe.reset_expert_counts()

                # Capture telemetry inputs for background thread
                current_lr = scheduler.get_last_lr()[0]
                include_hidden = (optim_step % val_interval == 0)
                _telemetry_inputs = last_inputs.clone() if (include_hidden and last_inputs is not None) else None
                _do_attn_diag = (optim_step % val_interval == 0)
                _current_optim_step = optim_step
                nvtx_pop()  # telemetry_snapshot

                # Record a CUDA event after all GPU work for this step is
                # enqueued. The background thread will wait on this before
                # reading GPU tensors.
                _step_event = torch.cuda.Event()
                _step_event.record()

                # Store current step's GPU tensors for deferred reading
                # in the NEXT step.
                _prev_step_event = _step_event
                _prev_grad_norm = grad_norm.detach()
                _prev_accum_loss = accum_loss.detach().clone()
                _prev_accum_tokens = accum_tokens_gpu.detach().clone() if scheduler_is_token_based else None
                _prev_optim_step = optim_step
                _prev_step_elapsed = time.perf_counter() - step_start_time

                # Launch background thread for expert metrics + telemetry
                _async_metrics_result = {}

                def _collect_metrics_bg(
                    event, model_ref, expert_snaps, lr, inc_hidden,
                    telemetry_inputs, do_attn_diag, result_dict, step_num,
                    optimizer_ref, val_iv,
                ):
                    """Background thread: wait for GPU, compute metrics."""
                    try:
                        event.synchronize()  # wait for GPU to finish this step

                        metrics = {}

                        # Expert usage metrics from snapshots — use bulk
                        # .cpu().tolist() instead of per-expert .item() to
                        # avoid hundreds of individual GPU→CPU syncs.
                        for snap in expert_snaps:
                            total_assignments = max(snap['total_tokens'] * snap['top_k'], 1)
                            fractions = snap['expert_counts'].float() / total_assignments
                            lid = snap['layer_idx']
                            # Single bulk transfer: GPU tensor → CPU list
                            frac_list = fractions.cpu().tolist()
                            for eid, frac_val in enumerate(frac_list):
                                metrics[f"moe/layer_{lid}/expert_{eid}"] = frac_val
                            max_frac = max(frac_list)
                            uniform = 1.0 / snap['num_experts']
                            metrics[f"moe/layer_{lid}/load_balance_score"] = max(
                                0.0, 1.0 - (max_frac - uniform) / (1.0 - uniform)
                            )

                        # Telemetry diagnostics (routing entropy, weight update ratios)
                        raw = model_ref
                        telemetry_metrics = raw.get_telemetry_diagnostics(
                            input_ids=telemetry_inputs,
                            optimizer=optimizer_ref,
                            lr=lr,
                            include_hidden_states=inc_hidden,
                        )
                        metrics.update(telemetry_metrics)

                        if do_attn_diag:
                            attn_diag = raw.get_attention_diagnostics()
                            metrics.update(attn_diag)

                        result_dict.update(metrics)
                    except Exception as e:
                        print(f"[AsyncMetrics] Warning: background metrics failed: {e}")

                # Wait for any previous metrics thread before launching new one
                if _metrics_thread is not None:
                    _metrics_thread.join()

                _metrics_thread = threading.Thread(
                    target=_collect_metrics_bg,
                    args=(
                        _step_event, raw_model, expert_snapshots,
                        current_lr, include_hidden, _telemetry_inputs,
                        _do_attn_diag, _async_metrics_result, optim_step,
                        optimizer, val_interval,
                    ),
                    daemon=True,
                )
                _metrics_thread.start()

                accum_loss = 0.0
                micro_count = 0

                if optim_step % val_interval == 0:
                    # Wait for metrics thread to finish before validation
                    # (validation switches to eval mode and needs clean state)
                    if _metrics_thread is not None:
                        _metrics_thread.join()
                        _metrics_thread = None
                    # ── Domain-specific validation ──
                    nvtx_push("validation")
                    validate_domains(
                        model=model,
                        wandb_run=wandb_run,
                        train_step=optim_step,
                        phase_config=phase_config,
                        device=config.device,
                        batch_size=16,
                        max_batches_per_domain=100,
                    )
                    nvtx_pop()
                if optim_step % 1000 == 0:
                    # Wait for metrics thread before generation tests
                    if _metrics_thread is not None:
                        _metrics_thread.join()
                        _metrics_thread = None
                    nvtx_push("generation_tests")
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
                        "train_loss": 0.0,  # Will be updated by deferred read
                    }
                    nvtx_pop()  # generation_tests
                if optim_step % 500 == 0:
                    # Wait for metrics thread before checkpointing
                    if _metrics_thread is not None:
                        _metrics_thread.join()
                        _metrics_thread = None
                    nvtx_push("checkpoint_save")
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
                    nvtx_pop()  # checkpoint_save

                # ── Eval Suite (comprehensive benchmarks) ──
                if eval_suite_interval > 0 and optim_step % eval_suite_interval == 0:
                    # Wait for metrics thread before eval
                    if _metrics_thread is not None:
                        _metrics_thread.join()
                        _metrics_thread = None
                    nvtx_push("eval_suite")
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
                    nvtx_pop()  # eval_suite

                step_start_time = time.perf_counter()
                if optim_step >= phase_config.total_steps:
                    print(f"Reached total_steps ({phase_config.total_steps}). Phase complete.")
                    raise KeyboardInterrupt

        # ── Flush final deferred metrics ──
        # The last optimizer step's metrics are still buffered; flush them now.
        if _metrics_thread is not None:
            _metrics_thread.join()
        if _prev_step_event is not None:
            _prev_step_event.synchronize()
            final_grad_norm = _prev_grad_norm.item()
            final_avg_loss = (_prev_accum_loss / grad_accumulation_steps).item()
            final_elapsed = _prev_step_elapsed
            final_tps = tokens_per_step / final_elapsed if final_elapsed > 0 else 0.0
            final_flops = flops_per_token * tokens_per_step
            final_mfu = final_flops / (final_elapsed * gpu_peak_flops) if final_elapsed > 0 else 0.0
            final_metrics = {
                "train/loss": final_avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "train/ppl": math.exp(min(final_avg_loss, 10)),
                "train/phase": phase_num,
                "train/grad_norm": final_grad_norm,
                "perf/tokens_per_sec": final_tps,
                "perf/mfu": final_mfu,
                "perf/step_time_sec": final_elapsed,
            }
            if _async_metrics_result:
                final_metrics.update(_async_metrics_result)
            async_logger.log(final_metrics, step=grad_accumulation_steps * _prev_optim_step)

        # Wait for any in-flight async save before exiting
        if _save_thread is not None:
            _save_thread.join()
        tlog = get_training_logger()
        tlog.logger.info(f"[TRAIN] Phase {phase_num} complete at step {optim_step}")
        tlog.flush()
        print(f"Phase {phase_num} training complete at optimizer step {optim_step}.")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving checkpoint at optimizer step {optim_step}...")
        # Wait for any in-flight async metrics/save threads first
        if _metrics_thread is not None:
            _metrics_thread.join(timeout=5.0)
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
            # Wait for any in-flight async metrics/save threads first
            if _metrics_thread is not None:
                _metrics_thread.join(timeout=5.0)
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
    finally:
        async_logger.flush_and_shutdown()



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
        from ..configs.new_model_config import config, PRETRAINING_PHASE_CONFIG
        from ...models.model_improv import GPT_FLASH
        from ...models.model_adv import build_optimizer_param_groups
        PHASE_1_CONFIG = PRETRAINING_PHASE_CONFIG
        PHASE_2_CONFIG = PRETRAINING_PHASE_CONFIG

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
        fused=torch.cuda.is_available(),
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
            fused=torch.cuda.is_available(),
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
