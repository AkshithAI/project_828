#!/usr/bin/env python3
"""
Kernel Profiler — Nsight Compute + Correctness + Benchmarks → W&B
==================================================================

Profiles each Triton kernel with Nsight Compute (ncu), runs the existing
correctness and benchmarking tests, and uploads all results to W&B.

Kernels profiled:
    1. apply_rope      — Rotary Position Embeddings (fwd + bwd)
    2. swiglu          — SwiGLU activation with soft clamping (fwd + bwd)
    3. fused_add_rms_norm — Fused residual add + RMSNorm (fwd + bwd)
    4. fused_linear_cross_entropy — Chunked fused linear + CE loss (fwd + bwd)

Usage (on a GPU VM):
    # Run everything (ncu + correctness + benchmarks):
    python profile_kernels.py

    # Skip ncu profiling (just correctness + benchmarks):
    python profile_kernels.py --skip-ncu

    # Skip correctness tests:
    python profile_kernels.py --skip-correctness

    # Only benchmark:
    python profile_kernels.py --skip-ncu --skip-correctness

    # Only specific kernels:
    python profile_kernels.py --kernels apply_rope swiglu

Requirements:
    - NVIDIA GPU with CUDA
    - ncu (Nsight Compute CLI) installed (for kernel profiling)
    - wandb, torch, triton, etc. (same as training requirements)
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, asdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

# ── Ensure project root is importable ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # project_828/
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# Data structures for results
# ═══════════════════════════════════════════════════════════════

@dataclass
class CorrectnessResult:
    kernel: str
    dtype: str
    fwd_pass: bool
    bwd_pass: bool
    fwd_max_diff: float
    bwd_max_diff: float
    error: str = ""


@dataclass
class BenchmarkEntry:
    kernel: str
    shape: str
    pytorch_ms: float
    triton_ms: float
    speedup: float


@dataclass
class KernelResults:
    kernel_name: str
    correctness: List[CorrectnessResult] = field(default_factory=list)
    benchmarks: List[BenchmarkEntry] = field(default_factory=list)
    ncu_rep_path: Optional[str] = None
    raw_stdout: str = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════
# NCU (Nsight Compute) Profiling
# ═══════════════════════════════════════════════════════════════

def find_ncu_binary() -> Optional[str]:
    """Locate the ncu binary on the system."""
    ncu = shutil.which("ncu")
    if ncu:
        return ncu

    candidates = [
        "/usr/local/cuda/bin/ncu",
        "/usr/bin/ncu",
        "/opt/nvidia/nsight-compute/*/ncu",
        "/usr/local/cuda-*/bin/ncu",
    ]
    for candidate in candidates:
        matches = glob(candidate)
        if matches:
            return sorted(matches)[-1]

    # Search Nsight Compute install paths
    for search_dir in [Path("/opt/nvidia"), Path("/usr/local/cuda")]:
        if search_dir.exists():
            for p in search_dir.rglob("ncu"):
                if p.is_file() and os.access(str(p), os.X_OK):
                    return str(p)

    return None


def run_ncu_profile(
    kernel_name: str,
    output_dir: Path,
    ncu_bin: str,
) -> Optional[Path]:
    """
    Run ncu on a kernel's __main__ block to profile its CUDA kernels.

    Returns the path to the .ncu-rep file, or None if profiling failed.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ncu_{kernel_name}_{timestamp}.ncu-rep"

    # Map kernel names to their module paths
    kernel_modules = {
        "apply_rope": "src.kernels.apply_rope",
        "swiglu": "src.kernels.swiglu",
        "fused_add_rms_norm": "src.kernels.fused_add_rms_norm",
        "fused_linear_cross_entropy": "src.kernels.fused_linear_cross_entropy",
    }

    module = kernel_modules.get(kernel_name)
    if not module:
        print(f"[WARNING] Unknown kernel: {kernel_name}")
        return None

    kernel_file = PROJECT_ROOT / "src" / "kernels" / f"{kernel_name}.py"
    py_inline = (
        f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
        f"sys.path.insert(0, '{PROJECT_ROOT / 'src' / 'kernels'}'); "
        f"exec(open('{kernel_file}').read())"
    )

    cmd = [
        ncu_bin,
        "--set", "full",                   # Full metric collection
        "--target-processes", "all",       # Profile all child processes
        "--force-overwrite",
        "-o", str(output_file.with_suffix("")),  # ncu adds .ncu-rep
        "--kernel-name-base", "function",  # Use function names for readability
        "--launch-skip", "0",              # Profile from start
        "--launch-count", "20",            # Limit kernel launches to profile
        "python3", "-c", py_inline,
    ]

    # For fused_linear_cross_entropy, add CLI args to the inline script
    if kernel_name == "fused_linear_cross_entropy":
        py_inline = (
            f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); "
            f"sys.path.insert(0, '{PROJECT_ROOT / 'src' / 'kernels'}'); "
            f"sys.argv.append('--correctness-only'); "
            f"exec(open('{kernel_file}').read())"
        )
        cmd[-1] = py_inline

    print(f"\n  [NCU] Profiling {kernel_name}...")
    print(f"  [NCU] Command: {ncu_bin} ... python3 -c 'exec(open({kernel_file.name}))'")

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )

        if result.returncode != 0:
            print(f"  [NCU] Warning: ncu exited with code {result.returncode}")
            if result.stderr:
                print(f"  [NCU] stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"  [NCU] Timeout after 600s for {kernel_name}")
        return None
    except Exception as e:
        print(f"  [NCU] Error: {e}")
        return None

    # Check for output file (ncu may add .ncu-rep suffix)
    ncu_rep = output_file
    if not ncu_rep.exists():
        ncu_rep = output_file.with_suffix(".ncu-rep")
    if not ncu_rep.exists():
        # Search for any matching file
        matches = list(output_dir.glob(f"ncu_{kernel_name}*.ncu-rep"))
        if matches:
            ncu_rep = sorted(matches)[-1]

    if ncu_rep.exists():
        print(f"  [NCU] ✅ Report saved: {ncu_rep.name}")
        return ncu_rep
    else:
        print(f"  [NCU] ⚠ No .ncu-rep file generated")
        return None


# ═══════════════════════════════════════════════════════════════
# Correctness & Benchmark Runners
# ═══════════════════════════════════════════════════════════════

def run_rope_tests() -> KernelResults:
    """Run apply_rope correctness and benchmark tests."""
    import torch
    results = KernelResults(kernel_name="apply_rope")

    try:
        from src.kernels.apply_rope import (
            TritonRoPEFunction, pytorch_apply_rope,
            run_correctness_tests, run_benchmark,
        )

        # ── Correctness ──
        print("  [RoPE] Running correctness tests...")
        device = "cuda"
        B, S, H, D = 4, 2048, 32, 128
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        atols = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1.5e-1}
        rtols = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

        for dtype in dtypes:
            torch.manual_seed(42)
            x_ref = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)
            x_tri = x_ref.detach().clone().requires_grad_(True)
            cos = torch.randn((S, D // 2), device=device, dtype=dtype)
            sin = torch.randn((S, D // 2), device=device, dtype=dtype)
            dy = torch.randn((B, S, H, D), device=device, dtype=dtype)

            out_ref = pytorch_apply_rope(x_ref, cos, sin)
            out_tri = TritonRoPEFunction.apply(x_tri, cos, sin)
            out_ref.backward(dy)
            out_tri.backward(dy)

            fwd_diff = torch.max(torch.abs(out_ref - out_tri)).item()
            bwd_diff = torch.max(torch.abs(x_ref.grad - x_tri.grad)).item()
            fwd_pass = torch.allclose(out_ref, out_tri, atol=atols[dtype], rtol=rtols[dtype])
            bwd_pass = torch.allclose(x_ref.grad, x_tri.grad, atol=atols[dtype], rtol=rtols[dtype])

            results.correctness.append(CorrectnessResult(
                kernel="apply_rope", dtype=str(dtype),
                fwd_pass=fwd_pass, bwd_pass=bwd_pass,
                fwd_max_diff=fwd_diff, bwd_max_diff=bwd_diff,
            ))
            status = "✅" if (fwd_pass and bwd_pass) else "❌"
            print(f"    [{str(dtype):<14}] Fwd: {fwd_diff:.2e} Bwd: {bwd_diff:.2e} {status}")

        # ── Benchmark ──
        print("  [RoPE] Running benchmarks...")
        import triton
        dtype = torch.bfloat16
        B_b, H_b, D_b = 4, 32, 128
        for S in [1024, 2048, 4096, 8192]:
            x = torch.randn((B_b, S, H_b, D_b), device="cuda", dtype=dtype, requires_grad=True)
            cos = torch.randn((S, D_b // 2), device="cuda", dtype=dtype)
            sin = torch.randn((S, D_b // 2), device="cuda", dtype=dtype)
            dy = torch.randn_like(x)

            def bench_pt():
                xr = x.detach().clone().requires_grad_(True)
                out = pytorch_apply_rope(xr, cos, sin)
                out.backward(dy)

            def bench_tri():
                xt = x.detach().clone().requires_grad_(True)
                out = TritonRoPEFunction.apply(xt, cos, sin)
                out.backward(dy)

            ms_pt = triton.testing.do_bench(bench_pt)
            ms_tri = triton.testing.do_bench(bench_tri)
            speedup = ms_pt / ms_tri if ms_tri > 0 else 0
            results.benchmarks.append(BenchmarkEntry(
                kernel="apply_rope", shape=f"({B_b},{S},{H_b},{D_b})",
                pytorch_ms=ms_pt, triton_ms=ms_tri, speedup=speedup,
            ))
            print(f"    ({B_b},{S},{H_b},{D_b}): PyTorch={ms_pt:.3f}ms Triton={ms_tri:.3f}ms {speedup:.2f}x")

    except Exception as e:
        results.error = str(e)
        print(f"  [RoPE] ❌ Error: {e}")

    return results


def run_swiglu_tests() -> KernelResults:
    """Run swiglu correctness and benchmark tests."""
    import torch
    results = KernelResults(kernel_name="swiglu")

    try:
        from src.kernels.swiglu import (
            TritonSwigluFunction, triton_swiglu, naive_swiglu,
        )

        # ── Correctness ──
        print("  [SwiGLU] Running correctness tests...")
        device = "cuda"
        limit = 30.0
        shape = (16, 2048, 4096)
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        atols = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1.5e-1}
        rtols = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

        for dtype in dtypes:
            torch.manual_seed(42)
            x_naive = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            x_triton = x_naive.detach().clone().requires_grad_(True)
            dout = torch.randn((shape[0], shape[1], shape[2] // 2), device=device, dtype=dtype)

            out_naive = naive_swiglu(x_naive, limit=limit)
            out_triton = triton_swiglu(x_triton, limit=limit)
            out_naive.backward(dout)
            out_triton.backward(dout)

            fwd_diff = torch.max(torch.abs(out_naive - out_triton)).item()
            bwd_diff = torch.max(torch.abs(x_naive.grad - x_triton.grad)).item()
            fwd_pass = torch.allclose(out_naive, out_triton, atol=atols[dtype], rtol=rtols[dtype])
            bwd_pass = torch.allclose(x_naive.grad, x_triton.grad, atol=atols[dtype], rtol=rtols[dtype])

            results.correctness.append(CorrectnessResult(
                kernel="swiglu", dtype=str(dtype),
                fwd_pass=fwd_pass, bwd_pass=bwd_pass,
                fwd_max_diff=fwd_diff, bwd_max_diff=bwd_diff,
            ))
            status = "✅" if (fwd_pass and bwd_pass) else "❌"
            print(f"    [{str(dtype):<14}] Fwd: {fwd_diff:.2e} Bwd: {bwd_diff:.2e} {status}")

        # ── Benchmark ──
        print("  [SwiGLU] Running benchmarks...")
        import triton
        dtype = torch.bfloat16
        test_configs = [(1024, 4096), (4096, 4096), (8192, 8192), (16384, 11008)]

        for M, N_in in test_configs:
            x = torch.randn((M, N_in), device="cuda", dtype=dtype, requires_grad=True)
            ms_naive = triton.testing.do_bench(lambda: naive_swiglu(x, limit=limit))
            ms_triton = triton.testing.do_bench(lambda: triton_swiglu(x, limit=limit))
            speedup = ms_naive / ms_triton if ms_triton > 0 else 0
            results.benchmarks.append(BenchmarkEntry(
                kernel="swiglu", shape=f"({M},{N_in})",
                pytorch_ms=ms_naive, triton_ms=ms_triton, speedup=speedup,
            ))
            print(f"    ({M},{N_in}): Naive={ms_naive:.3f}ms Triton={ms_triton:.3f}ms {speedup:.2f}x")

    except Exception as e:
        results.error = str(e)
        print(f"  [SwiGLU] ❌ Error: {e}")

    return results


def run_fused_add_rms_norm_tests() -> KernelResults:
    """Run fused_add_rms_norm correctness tests."""
    import torch
    results = KernelResults(kernel_name="fused_add_rms_norm")

    try:
        from src.kernels.fused_add_rms_norm import (
            FusedAddRMSNormFunction, PyTorchFusedAddRMSNorm,
        )

        print("  [FusedAddRMSNorm] Running correctness tests...")
        B, T, H = 4, 128, 4096
        eps = 1e-6
        device = "cuda"

        for dtype in [torch.float32, torch.bfloat16]:
            torch.manual_seed(42)
            X_base = torch.randn((B, T, H), dtype=dtype, device=device)
            R_base = torch.randn((B, T, H), dtype=dtype, device=device)
            W_base = torch.randn((H,), dtype=dtype, device=device)
            dY_in = torch.randn((B, T, H), dtype=dtype, device=device)
            dS_in = torch.randn((B, T, H), dtype=dtype, device=device)

            # PyTorch reference
            X_ref = X_base.clone().requires_grad_(True)
            R_ref = R_base.clone().requires_grad_(True)
            W_ref = W_base.clone().requires_grad_(True)
            ref_mod = PyTorchFusedAddRMSNorm(H, eps=eps).to(device)
            Y_ref, S_ref = ref_mod(X_ref, R_ref, W_ref)
            loss_ref = (Y_ref * dY_in).sum() + (S_ref * dS_in).sum()
            loss_ref.backward()

            # Triton kernel
            X_tri = X_base.clone().requires_grad_(True)
            R_tri = R_base.clone().requires_grad_(True)
            W_tri = W_base.clone().requires_grad_(True)
            Y_tri, S_tri = FusedAddRMSNormFunction.apply(X_tri, R_tri, W_tri, eps)
            loss_tri = (Y_tri * dY_in).sum() + (S_tri * dS_in).sum()
            loss_tri.backward()

            # The kernel's own test uses atol=1e-2, rtol=1e-2 even for fp32
            # due to internal float arithmetic in the Triton kernel
            atol, rtol = 1e-2, 1e-2
            fwd_diff = max(
                torch.max(torch.abs(Y_ref - Y_tri)).item(),
                torch.max(torch.abs(S_ref - S_tri)).item(),
            )
            bwd_diff = max(
                torch.max(torch.abs(X_ref.grad - X_tri.grad)).item(),
                torch.max(torch.abs(R_ref.grad - R_tri.grad)).item(),
                torch.max(torch.abs(W_ref.grad - W_tri.grad)).item(),
            )
            fwd_pass = (
                torch.allclose(Y_ref, Y_tri, atol=atol, rtol=rtol) and
                torch.allclose(S_ref, S_tri, atol=atol, rtol=rtol)
            )
            bwd_pass = (
                torch.allclose(X_ref.grad, X_tri.grad, atol=atol, rtol=rtol) and
                torch.allclose(R_ref.grad, R_tri.grad, atol=atol, rtol=rtol) and
                torch.allclose(W_ref.grad, W_tri.grad, atol=atol, rtol=rtol)
            )

            results.correctness.append(CorrectnessResult(
                kernel="fused_add_rms_norm", dtype=str(dtype),
                fwd_pass=fwd_pass, bwd_pass=bwd_pass,
                fwd_max_diff=fwd_diff, bwd_max_diff=bwd_diff,
            ))
            status = "✅" if (fwd_pass and bwd_pass) else "❌"
            print(f"    [{str(dtype):<14}] Fwd: {fwd_diff:.2e} Bwd: {bwd_diff:.2e} {status}")

        # ── Benchmark ──
        print("  [FusedAddRMSNorm] Running benchmarks...")
        import triton
        dtype = torch.bfloat16
        for (B_b, T_b, H_b) in [(4, 128, 4096), (4, 512, 4096), (4, 2048, 4096), (8, 2048, 768)]:
            X = torch.randn((B_b, T_b, H_b), dtype=dtype, device="cuda")
            R = torch.randn((B_b, T_b, H_b), dtype=dtype, device="cuda")
            W = torch.randn((H_b,), dtype=dtype, device="cuda")

            ref_mod = PyTorchFusedAddRMSNorm(H_b, eps=1e-6).to("cuda")

            ms_pt = triton.testing.do_bench(lambda: ref_mod(X, R, W))
            ms_tri = triton.testing.do_bench(
                lambda: FusedAddRMSNormFunction.apply(X, R, W, 1e-6)
            )
            speedup = ms_pt / ms_tri if ms_tri > 0 else 0
            results.benchmarks.append(BenchmarkEntry(
                kernel="fused_add_rms_norm", shape=f"({B_b},{T_b},{H_b})",
                pytorch_ms=ms_pt, triton_ms=ms_tri, speedup=speedup,
            ))
            print(f"    ({B_b},{T_b},{H_b}): PyTorch={ms_pt:.3f}ms Triton={ms_tri:.3f}ms {speedup:.2f}x")

    except Exception as e:
        results.error = str(e)
        print(f"  [FusedAddRMSNorm] ❌ Error: {e}")

    return results


def run_fused_linear_ce_tests() -> KernelResults:
    """Run fused_linear_cross_entropy correctness and benchmark tests."""
    import torch
    results = KernelResults(kernel_name="fused_linear_cross_entropy")

    try:
        from src.kernels.fused_linear_cross_entropy import (
            fused_linear_cross_entropy, naive_linear_cross_entropy,
            tensor_error_metrics, create_inputs,
        )

        # ── Correctness ──
        print("  [FusedLinearCE] Running correctness tests...")
        device = torch.device("cuda")
        workspace_bytes = 512 * 1024 * 1024

        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            T, D, V = 2048, 4096, 32000
            hidden, weight, target = create_inputs(
                num_tokens=T, hidden_dim=D, vocab_size=V,
                dtype=dtype, device=device,
                ignore_fraction=0.05, seed=1234,
            )

            n_non_ignore = torch.sum(target != -100, dtype=torch.int32)

            # Naive
            h_naive = hidden.detach().clone().requires_grad_(True)
            w_naive = weight.detach().clone().requires_grad_(True)
            loss_naive = naive_linear_cross_entropy(h_naive, w_naive, target, -100)
            (loss_naive * 128.0).backward()

            # Custom
            h_custom = hidden.detach().clone().requires_grad_(True)
            w_custom = weight.detach().clone().requires_grad_(True)
            loss_custom = fused_linear_cross_entropy(
                h_custom, w_custom, target, -100, n_non_ignore, workspace_bytes,
            )
            (loss_custom * 128.0).backward()

            torch.cuda.synchronize()

            loss_diff = abs(loss_custom.float().item() - loss_naive.float().item())
            h_metrics = tensor_error_metrics(h_custom.grad, h_naive.grad)
            w_metrics = tensor_error_metrics(w_custom.grad, w_naive.grad)

            fwd_pass = loss_diff < 5e-3
            bwd_pass = (h_metrics.max_abs < 3e-2 and w_metrics.max_abs < 3e-2)

            results.correctness.append(CorrectnessResult(
                kernel="fused_linear_ce", dtype=dtype_name,
                fwd_pass=fwd_pass, bwd_pass=bwd_pass,
                fwd_max_diff=loss_diff,
                bwd_max_diff=max(h_metrics.max_abs, w_metrics.max_abs),
            ))
            status = "✅" if (fwd_pass and bwd_pass) else "❌"
            print(f"    [{dtype_name:<14}] Loss diff: {loss_diff:.2e} Grad max: {max(h_metrics.max_abs, w_metrics.max_abs):.2e} {status}")

        # ── Benchmark ──
        print("  [FusedLinearCE] Running benchmarks...")
        import gc
        dtype = torch.bfloat16

        for T, D, V in [(2048, 4096, 32000), (4096, 4096, 32000), (2048, 768, 32000)]:
            torch.manual_seed(1234)
            hidden, weight, target = create_inputs(
                num_tokens=T, hidden_dim=D, vocab_size=V,
                dtype=dtype, device=device,
                ignore_fraction=0.05, seed=1234,
            )
            n_non_ignore = torch.sum(target != -100, dtype=torch.int32)

            h_c = hidden.detach().clone().requires_grad_(True)
            w_c = weight.detach().clone().requires_grad_(True)

            def custom_step():
                h_c.grad = None
                w_c.grad = None
                loss = fused_linear_cross_entropy(
                    h_c, w_c, target, -100, n_non_ignore, workspace_bytes,
                )
                loss.backward()
                return loss

            # Warmup
            for _ in range(3):
                custom_step()
            torch.cuda.synchronize()

            # Timed iterations
            timings = []
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                custom_step()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))

            import statistics
            ms_custom = statistics.median(timings)

            # Try naive benchmark (may OOM for large V)
            ms_naive = float("nan")
            try:
                h_n = hidden.detach().clone().requires_grad_(True)
                w_n = weight.detach().clone().requires_grad_(True)

                def naive_step():
                    h_n.grad = None
                    w_n.grad = None
                    loss = naive_linear_cross_entropy(h_n, w_n, target, -100)
                    loss.backward()
                    return loss

                for _ in range(3):
                    naive_step()
                torch.cuda.synchronize()

                naive_timings = []
                for _ in range(10):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    naive_step()
                    e.record()
                    e.synchronize()
                    naive_timings.append(s.elapsed_time(e))
                ms_naive = statistics.median(naive_timings)
                del h_n, w_n
            except torch.cuda.OutOfMemoryError:
                ms_naive = float("inf")
                print(f"    ⚠ Naive OOM for ({T},{D},{V})")
            except Exception:
                pass

            speedup = ms_naive / ms_custom if ms_custom > 0 and ms_naive != float("inf") else 0
            results.benchmarks.append(BenchmarkEntry(
                kernel="fused_linear_ce", shape=f"T={T},D={D},V={V}",
                pytorch_ms=ms_naive if ms_naive != float("inf") else -1,
                triton_ms=ms_custom, speedup=speedup,
            ))
            print(f"    T={T},D={D},V={V}: Naive={ms_naive:.3f}ms Custom={ms_custom:.3f}ms {speedup:.2f}x")

            del h_c, w_c, hidden, weight, target
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        results.error = str(e)
        print(f"  [FusedLinearCE] ❌ Error: {e}")

    return results


# ═══════════════════════════════════════════════════════════════
# W&B Upload
# ═══════════════════════════════════════════════════════════════

def upload_results_to_wandb(
    all_results: Dict[str, KernelResults],
    output_dir: Path,
    wandb_entity: str,
    wandb_project: str,
):
    """Upload all kernel profiling results to W&B."""
    import wandb

    run_name = f"kernel_profile_{datetime.now().strftime('%m%d_%H%M%S')}"
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        group="kernel_profiling",
        tags=["ncu", "correctness", "benchmark", "triton_kernels"],
        config={
            "profiler": "nsight_compute",
            "kernels": list(all_results.keys()),
        },
    )

    # ── Correctness Results Table ──
    correctness_rows = []
    for kernel_name, result in all_results.items():
        for c in result.correctness:
            correctness_rows.append([
                c.kernel, c.dtype,
                "✅" if c.fwd_pass else "❌",
                "✅" if c.bwd_pass else "❌",
                c.fwd_max_diff, c.bwd_max_diff,
                c.error or "",
            ])

    if correctness_rows:
        correctness_table = wandb.Table(
            columns=["kernel", "dtype", "fwd_pass", "bwd_pass",
                     "fwd_max_diff", "bwd_max_diff", "error"],
            data=correctness_rows,
        )
        run.log({"correctness_results": correctness_table})

    # ── Benchmark Results Table ──
    benchmark_rows = []
    for kernel_name, result in all_results.items():
        for b in result.benchmarks:
            benchmark_rows.append([
                b.kernel, b.shape, b.pytorch_ms, b.triton_ms, b.speedup,
            ])

    if benchmark_rows:
        benchmark_table = wandb.Table(
            columns=["kernel", "shape", "pytorch_ms", "triton_ms", "speedup"],
            data=benchmark_rows,
        )
        run.log({"benchmark_results": benchmark_table})

    # ── Summary metrics ──
    for kernel_name, result in all_results.items():
        # Correctness summary
        all_pass = all(
            c.fwd_pass and c.bwd_pass for c in result.correctness
        )
        run.log({f"{kernel_name}/all_correct": 1 if all_pass else 0})

        # Best speedup
        if result.benchmarks:
            best_speedup = max(b.speedup for b in result.benchmarks)
            avg_speedup = sum(b.speedup for b in result.benchmarks) / len(result.benchmarks)
            run.log({
                f"{kernel_name}/best_speedup": best_speedup,
                f"{kernel_name}/avg_speedup": avg_speedup,
            })

    # ── Upload .ncu-rep files as Artifacts ──
    ncu_files = list(output_dir.glob("*.ncu-rep"))
    if ncu_files:
        artifact = wandb.Artifact(
            name="ncu_kernel_profiles",
            type="ncu-profile",
            description=(
                "Nsight Compute kernel profiles for all Triton kernels. "
                "Download and open in Nsight Compute GUI on your local machine."
            ),
        )
        for f in ncu_files:
            artifact.add_file(str(f))
        run.log_artifact(artifact)
        print(f"  ✅ Uploaded {len(ncu_files)} .ncu-rep files as W&B Artifact")

    run.finish()
    print(f"  ✅ W&B run complete: {run_name}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

ALL_KERNELS = ["apply_rope", "swiglu", "fused_add_rms_norm", "fused_linear_cross_entropy"]

KERNEL_RUNNERS = {
    "apply_rope": run_rope_tests,
    "swiglu": run_swiglu_tests,
    "fused_add_rms_norm": run_fused_add_rms_norm_tests,
    "fused_linear_cross_entropy": run_fused_linear_ce_tests,
}


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Profiler — Nsight Compute + Correctness + Benchmarks → W&B"
    )
    parser.add_argument(
        "--kernels", nargs="+", default=ALL_KERNELS,
        choices=ALL_KERNELS,
        help="Which kernels to profile (default: all)",
    )
    parser.add_argument("--skip-ncu", action="store_true",
                        help="Skip Nsight Compute profiling")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness tests (only benchmarks)")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip benchmarks (only correctness)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for ncu output (default: private/ncu_profiles/)")
    parser.add_argument("--wandb-entity", type=str, default="akshithmarepally-akai")
    parser.add_argument("--wandb-project", type=str, default="828_kernel_profiling")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B upload")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "ncu_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  Kernel Profiler — Nsight Compute + Correctness + Benchmarks")
    print("="*70)
    print(f"  Kernels:    {args.kernels}")
    print(f"  NCU:        {'skip' if args.skip_ncu else 'enabled'}")
    print(f"  Correct:    {'skip' if args.skip_correctness else 'enabled'}")
    print(f"  Benchmarks: {'skip' if args.skip_benchmarks else 'enabled'}")
    print(f"  Output:     {output_dir}")
    print(f"  W&B:        {'skip' if args.no_wandb else args.wandb_project}")
    print("="*70)

    all_results: Dict[str, KernelResults] = {}

    # ── NCU Profiling ──
    if not args.skip_ncu:
        ncu_bin = find_ncu_binary()
        if ncu_bin:
            print(f"\n[NCU] Found ncu at: {ncu_bin}")
            for kernel_name in args.kernels:
                ncu_rep = run_ncu_profile(kernel_name, output_dir, ncu_bin)
                if kernel_name not in all_results:
                    all_results[kernel_name] = KernelResults(kernel_name=kernel_name)
                if ncu_rep:
                    all_results[kernel_name].ncu_rep_path = str(ncu_rep)
        else:
            print("\n[NCU] ⚠ ncu not found — skipping Nsight Compute profiling")
            print("       Install: apt-get install -y nsight-compute")

    # ── Correctness + Benchmarks ──
    for kernel_name in args.kernels:
        print(f"\n{'─'*70}")
        print(f"  {kernel_name}")
        print(f"{'─'*70}")

        runner = KERNEL_RUNNERS.get(kernel_name)
        if runner is None:
            print(f"  ⚠ No runner for {kernel_name}")
            continue

        result = runner()

        # Filter based on flags
        if args.skip_correctness:
            result.correctness = []
        if args.skip_benchmarks:
            result.benchmarks = []

        # Merge with any existing NCU results
        if kernel_name in all_results:
            existing = all_results[kernel_name]
            result.ncu_rep_path = existing.ncu_rep_path
        all_results[kernel_name] = result

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    for kernel_name, result in all_results.items():
        n_correct = sum(1 for c in result.correctness if c.fwd_pass and c.bwd_pass)
        n_total = len(result.correctness)
        avg_speedup = (
            sum(b.speedup for b in result.benchmarks) / len(result.benchmarks)
            if result.benchmarks else 0
        )
        ncu_status = "✅" if result.ncu_rep_path else "—"

        print(f"  {kernel_name:<30s}  "
              f"Correct: {n_correct}/{n_total}  "
              f"Avg Speedup: {avg_speedup:.2f}x  "
              f"NCU: {ncu_status}"
              + (f"  ❌ {result.error}" if result.error else ""))

    # ── Upload to W&B ──
    if not args.no_wandb:
        print(f"\n[W&B] Uploading results to {args.wandb_project}...")
        upload_results_to_wandb(
            all_results=all_results,
            output_dir=output_dir,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )

    # ── Save JSON report ──
    report_path = output_dir / "kernel_profiling_report.json"
    json_report = {}
    for kernel_name, result in all_results.items():
        json_report[kernel_name] = {
            "correctness": [asdict(c) for c in result.correctness],
            "benchmarks": [asdict(b) for b in result.benchmarks],
            "ncu_rep_path": result.ncu_rep_path,
            "error": result.error,
        }
    with open(report_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[Report] Saved: {report_path}")

    print("\n" + "="*70)
    print("  DONE — Download .ncu-rep files from W&B Artifacts")
    print("  and open in Nsight Compute GUI on your Mac.")
    print("="*70)


if __name__ == "__main__":
    main()
