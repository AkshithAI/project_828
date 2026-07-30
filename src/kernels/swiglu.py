import torch
import triton
import triton.language as tl
import torch.nn.functional as F
try:
    from .utils import ensure_contiguous
except ImportError:
    try:
        from src.kernels.utils import ensure_contiguous
    except ImportError:
        from utils import ensure_contiguous


# ── Autotuning configs for SwiGLU forward kernel ─────────────────────
_SWIGLU_FWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
]


# ── Autotuning configs for SwiGLU backward kernel ───────────────────
_SWIGLU_BWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=3),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=32, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
]


@triton.jit
def silu(x):
  return x * tl.sigmoid(x)

@triton.autotune(configs=_SWIGLU_FWD_CONFIGS, key=["N"])
@triton.jit
def _swiglu_fwd_kernel(
    X_ptr,
    X_row_stride,
    Y_ptr,
    Y_row_stride,
    TANH_ptr,
    TANH_row_stride,
    N,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
  row_pid = tl.program_id(axis=0)

  X_ptr += row_pid * X_row_stride
  Y_ptr += row_pid * Y_row_stride
  if TANH_ptr:
    TANH_ptr += row_pid * TANH_row_stride

  col_offs = tl.arange(0, BLOCK_SIZE // 2)
  mask = col_offs < N

  x_gate = tl.load(X_ptr + col_offs, mask=mask, other=0.).to(tl.float32)
  x_up   = tl.load(X_ptr + N + col_offs, mask=mask, other=0.).to(tl.float32)

  xhat = silu(x_gate)
  out = xhat * x_up

  if limit > 0:
    _tanh = tl.extra.cuda.libdevice.tanh(out / limit)
    y = limit * _tanh
    if TANH_ptr:
      tl.store(TANH_ptr + col_offs, _tanh, mask=mask)
  else:
    y = out

  tl.store(Y_ptr + col_offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_SWIGLU_BWD_CONFIGS, key=["N"])
@triton.jit
def _swiglu_bwd_kernel(
    dX_ptr, dX_stride,
    dY_ptr, dY_stride,
    X_ptr,  X_stride,
    TANH_ptr, TANH_stride,
    N,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_pid = tl.program_id(axis=0)

    dX_ptr += row_pid * dX_stride
    dY_ptr += row_pid * dY_stride
    X_ptr  += row_pid * X_stride
    if TANH_ptr:
        TANH_ptr += row_pid * TANH_stride

    col_offs = tl.arange(0, BLOCK_SIZE // 2)
    mask = col_offs < N

    dy = tl.load(dY_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)

    x_gate = tl.load(X_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    x_up   = tl.load(X_ptr + N + col_offs, mask=mask, other=0.0).to(tl.float32)

    if limit > 0:
        if TANH_ptr:
            t = tl.load(TANH_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
        else:
            raw_out = silu(x_gate) * x_up
            t = tl.extra.cuda.libdevice.tanh(raw_out / limit)

        dout = dy * (1.0 - t * t)
    else:
        dout = dy

    sig_g = tl.sigmoid(x_gate)
    silu_g = x_gate * sig_g

    dx_up = dout * silu_g
    dx_gate = dout * x_up * sig_g * (1.0 + x_gate * (1.0 - sig_g))

    tl.store(dX_ptr + col_offs, dx_gate.to(dX_ptr.dtype.element_ty), mask=mask)
    tl.store(dX_ptr + N + col_offs, dx_up.to(dX_ptr.dtype.element_ty), mask=mask)


def swiglu_forward(X, limit):
  shape = X.shape
  dim = shape[-1]
  X = X.view(-1, dim)
  M, N = X.shape
  N_out = N // 2

  Y = torch.empty((M, N_out), dtype=X.dtype, device=X.device)
  tanh = torch.empty((M, N_out), dtype=torch.float32, device=X.device)

  _swiglu_fwd_kernel[(M,)](
      X,
      X.stride(0),
      Y,
      Y.stride(0),
      tanh,
      tanh.stride(0) if tanh is not None else 0,
      N_out,
      limit,
  )
  out_shape = list(shape[:-1]) + [N_out]
  return Y.view(out_shape), tanh


def swiglu_backward(dY, X, limit, tanh):
  shape = X.shape
  dim = shape[-1]
  X = X.view(-1, dim)
  M, N = X.shape
  N_out = N // 2

  dX = torch.empty_like(X)
  dY = dY.view(-1, N_out)

  _swiglu_bwd_kernel[(M,)](
      dX,
      dX.stride(0),
      dY,
      dY.stride(0),
      X,
      X.stride(0),
      tanh,
      tanh.stride(0) if tanh is not None else 0,
      N_out,
      limit,
  )

  return dX.view(*shape)


class TritonSwigluFunction(torch.autograd.Function):
  @staticmethod
  @ensure_contiguous
  def forward(ctx, X, limit):
    Y, tanh = swiglu_forward(X, limit)

    ctx.limit = limit
    ctx.save_for_backward(X, tanh)
    return Y

  @staticmethod
  @ensure_contiguous
  def backward(ctx, dY):
    X, tanh = ctx.saved_tensors
    dX = swiglu_backward(dY, X, ctx.limit, tanh)
    return dX, None


# ------ Testing script --------

def triton_swiglu(x: torch.Tensor, limit: float = 30.0) -> torch.Tensor:
    return TritonSwigluFunction.apply(x, limit)


# =====================================================================
# 2. NAIVE PYTORCH IMPLEMENTATION
# =====================================================================

def soft_clamp_torch(x: torch.Tensor, limit: float = 5.0):
    return limit * torch.tanh(x / limit)


def naive_swiglu(x: torch.Tensor, limit: float = 30.0):
    x_gate, x_up = x.chunk(2, dim=-1)
    out = F.silu(x_gate) * x_up
    return soft_clamp_torch(out, limit)


# =====================================================================
# 3. VERIFICATION & BENCHMARK SUITE
# =====================================================================

def test_correctness():
    print("=" * 70)
    print(" 1. NUMERICAL CORRECTNESS CHECKS (FP32, FP16, BF16)")
    print("=" * 70)

    device = "cuda"
    limit = 30.0
    shape = (16, 2048, 4096)  # Batch, SeqLen, Hidden (2N)

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    atols = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1.5e-1}
    rtols = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

    for dtype in dtypes:
        torch.manual_seed(42)

        # Setup inputs
        x_naive = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        x_triton = x_naive.detach().clone().requires_grad_(True)
        dout = torch.randn((shape[0], shape[1], shape[2] // 2), device=device, dtype=dtype)

        # Forward pass
        out_naive = naive_swiglu(x_naive, limit=limit)
        out_triton = triton_swiglu(x_triton, limit=limit)

        # Backward pass
        out_naive.backward(dout)
        out_triton.backward(dout)

        # Check Forward Match
        fwd_diff = torch.max(torch.abs(out_naive - out_triton)).item()
        fwd_pass = torch.allclose(out_naive, out_triton, atol=atols[dtype], rtol=rtols[dtype])

        # Check Backward Match
        bwd_diff = torch.max(torch.abs(x_naive.grad - x_triton.grad)).item()
        bwd_pass = torch.allclose(x_naive.grad, x_triton.grad, atol=atols[dtype], rtol=rtols[dtype])

        status_fwd = "✅ PASSED" if fwd_pass else "❌ FAILED"
        status_bwd = "✅ PASSED" if bwd_pass else "❌ FAILED"

        print(f"[{str(dtype):<14}] Fwd Max Diff: {fwd_diff:.6e} ({status_fwd}) | Bwd Max Diff: {bwd_diff:.6e} ({status_bwd})")

    print("\n")


def benchmark_performance():
    print("=" * 70)
    print(" 2. LATENCY & THROUGHPUT BENCHMARK (ms)")
    print("=" * 70)

    device = "cuda"
    limit = 30.0
    dtype = torch.bfloat16

    # Test shapes: (Tokens, Hidden Dim 2N)
    test_configs = [
        (1024, 4096),
        (4096, 4096),
        (8192, 8192),
        (16384, 11008),  # LLaMA 3 8B MLP width
    ]

    print(f"{'Shape (M, 2N)':<22} | {'Naive Fwd (ms)':<14} | {'Triton Fwd (ms)':<15} | {'Speedup':<10}")
    print("-" * 70)

    for M, N_in in test_configs:
        x = torch.randn((M, N_in), device=device, dtype=dtype, requires_grad=True)
        dout = torch.randn((M, N_in // 2), device=device, dtype=dtype)

        # Benchmark Naive Forward
        ms_naive_fwd = triton.testing.do_bench(lambda: naive_swiglu(x, limit=limit))

        # Benchmark Triton Forward
        ms_triton_fwd = triton.testing.do_bench(lambda: triton_swiglu(x, limit=limit))

        speedup = ms_naive_fwd / ms_triton_fwd

        print(f"({M}, {N_in})".ljust(22) + f" | {ms_naive_fwd:<14.4f} | {ms_triton_fwd:<15.4f} | {speedup:.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    test_correctness()
    benchmark_performance()