import torch
import triton
import triton.language as tl
try:
    from .utils import ensure_contiguous
except ImportError:
    try:
        from src.kernels.utils import ensure_contiguous
    except ImportError:
        from utils import ensure_contiguous


# ── Autotuning configs for RoPE forward kernel ───────────────────────
_ROPE_FWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32},    num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 128},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 128},   num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 256},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},   num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},   num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=16, num_stages=2),
]


# ── Autotuning configs for RoPE backward kernel ─────────────────────
_ROPE_BWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32},    num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 64},    num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 128},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 128},   num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 256},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 256},   num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 512},   num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=16, num_stages=2),
]


@triton.autotune(configs=_ROPE_FWD_CONFIGS, key=["N"])
@triton.jit
def _apply_rope_fwd_kernel(
    X_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    Y_ptr,
    stride_yb, stride_ys, stride_yh, stride_yd,
    cos_ptr,
    stride_cos_b, stride_cos_s, stride_cos_d,
    sin_ptr,
    stride_sin_b, stride_sin_s, stride_sin_d,
    S, H, N,
    BLOCK_SIZE: tl.constexpr,
    HAS_BATCH_COS: tl.constexpr,
):
    row_pid = tl.program_id(axis=0)

    h = row_pid % H
    tmp = row_pid // H
    s = tmp % S
    b = tmp // S

    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < N

    x_base = X_ptr + b * stride_xb + s * stride_xs + h * stride_xh
    y_base = Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh

    if HAS_BATCH_COS:
        cos_base = cos_ptr + b * stride_cos_b + s * stride_cos_s
        sin_base = sin_ptr + b * stride_sin_b + s * stride_sin_s
    else:
        cos_base = cos_ptr + s * stride_cos_s
        sin_base = sin_ptr + s * stride_sin_s

    cos = tl.load(cos_base + col_offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_base + col_offs * stride_sin_d, mask=mask, other=0.0).to(tl.float32)

    x1 = tl.load(x_base + col_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_base + (N + col_offs) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    tl.store(y_base + col_offs * stride_yd, o1.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(y_base + (N + col_offs) * stride_yd, o2.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_ROPE_BWD_CONFIGS, key=["N"])
@triton.jit
def _apply_rope_bwd_kernel(
    dY_ptr,
    stride_dy_b, stride_dy_s, stride_dy_h, stride_dy_d,
    dX_ptr,
    stride_dx_b, stride_dx_s, stride_dx_h, stride_dx_d,
    cos_ptr,
    stride_cos_b, stride_cos_s, stride_cos_d,
    sin_ptr,
    stride_sin_b, stride_sin_s, stride_sin_d,
    S, H, N,
    BLOCK_SIZE: tl.constexpr,
    HAS_BATCH_COS: tl.constexpr,
):
  row_pid = tl.program_id(axis=0)

  h = row_pid % H
  tmp = row_pid // H
  b = tmp // S
  s = tmp % S

  dx_base = dX_ptr + b * stride_dx_b + s * stride_dx_s + h * stride_dx_h
  dy_base = dY_ptr + b * stride_dy_b + s * stride_dy_s + h * stride_dy_h

  if HAS_BATCH_COS:
    cos_base = cos_ptr + b * stride_cos_b + s * stride_cos_s
    sin_base = sin_ptr + b * stride_sin_b + s * stride_sin_s
  else:
    cos_base = cos_ptr + s * stride_cos_s
    sin_base = sin_ptr + s * stride_sin_s

  col_offs = tl.arange(0, BLOCK_SIZE)
  mask = col_offs < N

  cos = tl.load(cos_base + col_offs * stride_cos_d, mask=mask, other=0.).to(tl.float32)
  sin = tl.load(sin_base + col_offs * stride_sin_d, mask=mask, other=0.).to(tl.float32)

  d_o1 = tl.load(dy_base + col_offs * stride_dy_d, mask=mask, other=0.).to(tl.float32)
  d_o2 = tl.load(dy_base + (N + col_offs) * stride_dy_d, mask=mask, other=0.).to(tl.float32)

  d_x1 = d_o1 * cos + d_o2 * sin
  d_x2 = -d_o1 * sin + d_o2 * cos

  tl.store(dx_base + col_offs * stride_dx_d, d_x1.to(dX_ptr.dtype.element_ty), mask=mask)
  tl.store(dx_base + (N + col_offs) * stride_dx_d, d_x2.to(dX_ptr.dtype.element_ty), mask=mask)


class TritonRoPEFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        orig_cos, orig_sin = cos, sin
        while cos.dim() > 2 and (cos.shape[0] == 1 or cos.shape[-2] == 1):
            cos = cos.squeeze(0) if cos.shape[0] == 1 else cos.squeeze(-2)
            sin = sin.squeeze(0) if sin.shape[0] == 1 else sin.squeeze(-2)

        B, S, H, D = x.shape
        N = D // 2
        y = torch.empty_like(x)

        HAS_BATCH_COS = (cos.dim() == 3 and cos.shape[0] > 1)
        grid = (B * S * H,)

        _apply_rope_fwd_kernel[grid](
            x, x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y, y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            cos,
            cos.stride(0) if HAS_BATCH_COS else 0,
            cos.stride(1) if HAS_BATCH_COS else cos.stride(0),
            cos.stride(-1),
            sin,
            sin.stride(0) if HAS_BATCH_COS else 0,
            sin.stride(1) if HAS_BATCH_COS else sin.stride(0),
            sin.stride(-1),
            S, H, N,
            HAS_BATCH_COS=HAS_BATCH_COS,
        )

        ctx.save_for_backward(cos, sin)
        ctx.HAS_BATCH_COS = HAS_BATCH_COS
        ctx.shape = (B, S, H, N)

        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy: torch.Tensor):
        cos, sin = ctx.saved_tensors
        B, S, H, N = ctx.shape
        dx = torch.empty_like(dy)

        grid = (B * S * H,)

        _apply_rope_bwd_kernel[grid](
            dy, dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            dx, dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            cos,
            cos.stride(0) if ctx.HAS_BATCH_COS else 0,
            cos.stride(1) if ctx.HAS_BATCH_COS else cos.stride(0),
            cos.stride(-1),
            sin,
            sin.stride(0) if ctx.HAS_BATCH_COS else 0,
            sin.stride(1) if ctx.HAS_BATCH_COS else sin.stride(0),
            sin.stride(-1),
            S, H, N,
            HAS_BATCH_COS=ctx.HAS_BATCH_COS,
        )

        return dx, None, None


# ------ Testing script --------
def pytorch_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch eager implementation for RoPE."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)

    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1)


# =====================================================================
# 2. NUMERICAL CORRECTNESS SUITE (FWD & BWD)
# =====================================================================
def run_correctness_tests(triton_fn):
    print("=" * 70)
    print(" 1. NUMERICAL CORRECTNESS CHECKS (FP32, FP16, BF16)")
    print("=" * 70)

    device = "cuda"
    B, S, H, D = 4, 2048, 32, 128

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    atols = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1.5e-1}
    rtols = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

    for dtype in dtypes:
        torch.manual_seed(42)

        # Inputs
        x_ref = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)
        x_tri = x_ref.detach().clone().requires_grad_(True)

        cos = torch.randn((S, D // 2), device=device, dtype=dtype)
        sin = torch.randn((S, D // 2), device=device, dtype=dtype)

        dy = torch.randn((B, S, H, D), device=device, dtype=dtype)

        # Forward Pass
        out_ref = pytorch_apply_rope(x_ref, cos, sin)
        out_tri = triton_fn(x_tri, cos, sin)

        # Backward Pass
        out_ref.backward(dy)
        out_tri.backward(dy)

        # Checks
        fwd_diff = torch.max(torch.abs(out_ref - out_tri)).item()
        fwd_pass = torch.allclose(out_ref, out_tri, atol=atols[dtype], rtol=rtols[dtype])

        bwd_diff = torch.max(torch.abs(x_ref.grad - x_tri.grad)).item()
        bwd_pass = torch.allclose(x_ref.grad, x_tri.grad, atol=atols[dtype], rtol=rtols[dtype])

        status_fwd = "✅ PASSED" if fwd_pass else "❌ FAILED"
        status_bwd = "✅ PASSED" if bwd_pass else "❌ FAILED"

        print(
            f"[{str(dtype):<14}] "
            f"Fwd Max Diff: {fwd_diff:.6e} ({status_fwd}) | "
            f"Bwd Max Diff: {bwd_diff:.6e} ({status_bwd})"
        )


# =====================================================================
# 3. LATENCY & THROUGHPUT BENCHMARK
# =====================================================================
def run_benchmark(triton_fn):
    print("\n" + "=" * 70)
    print(" 2. LATENCY BENCHMARK (Fwd + Bwd Pass in BF16)")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    B, H, D = 4, 32, 128
    seq_lengths = [1024, 2048, 4096, 8192, 16384]

    print(f"{'Shape (B, S, H, D)':<22} | {'PyTorch (ms)':<14} | {'Triton (ms)':<14} | {'Speedup':<10}")
    print("-" * 70)

    for S in seq_lengths:
        x = torch.randn((B, S, H, D), device=device, dtype=dtype, requires_grad=True)
        cos = torch.randn((S, D // 2), device=device, dtype=dtype)
        sin = torch.randn((S, D // 2), device=device, dtype=dtype)
        dy = torch.randn((B, S, H, D), device=device, dtype=dtype)

        # PyTorch Eager benchmark
        def bench_pytorch():
            x_ref = x.detach().clone().requires_grad_(True)
            out = pytorch_apply_rope(x_ref, cos, sin)
            out.backward(dy)

        # Triton benchmark
        def bench_triton():
            x_tri = x.detach().clone().requires_grad_(True)
            out = triton_fn(x_tri, cos, sin)
            out.backward(dy)

        ms_pytorch = triton.testing.do_bench(bench_pytorch)
        ms_triton = triton.testing.do_bench(bench_triton)
        speedup = ms_pytorch / ms_triton

        shape_str = f"({B}, {S}, {H}, {D})"
        print(f"{shape_str:<22} | {ms_pytorch:<14.4f} | {ms_triton:<14.4f} | {speedup:.2f}x")

    print("=" * 70)

if __name__ == "__main__":
    import sys
    run_correctness_tests(TritonRoPEFunction.apply)
    if "--correctness-only" not in sys.argv:
        run_benchmark(TritonRoPEFunction.apply)