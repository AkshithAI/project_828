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


# ── Autotuning configs for Gemma-style SwiGLU forward kernel ─────────
_GEMMA_SWIGLU_FWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
]

# ── Autotuning configs for Gemma-style SwiGLU backward kernel ────────
_GEMMA_SWIGLU_BWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
]


# ═════════════════════════════════════════════════════════════════════
# GEMMA-2 STYLE SWIGLU (exact numerics of model_flash_attn.swiglu):
#
#     x_glu,   x_linear = x.chunk(2, dim=-1)
#     g = x_glu.clamp(max=limit)              # upper clamp only
#     l = x_linear.clamp(-limit, limit)       # symmetric clamp
#     out = g * sigmoid(alpha * g) * (l + 1)
#
# NOTE: this is NOT the same function as swiglu.py's silu-gate variant.
# The two are not interchangeable — each matches a different checkpoint.
# ═════════════════════════════════════════════════════════════════════


@triton.autotune(configs=_GEMMA_SWIGLU_FWD_CONFIGS, key=["N"])
@triton.jit
def _gemma_swiglu_fwd_kernel(
    X_ptr,
    X_row_stride,
    Y_ptr,
    Y_row_stride,
    N,                      # size of EACH half (out dim), input is 2N wide
    alpha,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
  row_pid = tl.program_id(axis=0)

  X_ptr += row_pid * X_row_stride
  Y_ptr += row_pid * Y_row_stride

  col_offs = tl.arange(0, BLOCK_SIZE // 2)
  mask = col_offs < N

  g_raw = tl.load(X_ptr + col_offs, mask=mask, other=0.).to(tl.float32)
  l_raw = tl.load(X_ptr + N + col_offs, mask=mask, other=0.).to(tl.float32)

  # Clamps (forward values)
  g = tl.minimum(g_raw, limit)
  l = tl.minimum(tl.maximum(l_raw, -limit), limit)

  sig = tl.sigmoid(alpha * g)
  y = g * sig * (l + 1.0)

  tl.store(Y_ptr + col_offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_GEMMA_SWIGLU_BWD_CONFIGS, key=["N"])
@triton.jit
def _gemma_swiglu_bwd_kernel(
    dX_ptr, dX_stride,
    dY_ptr, dY_stride,
    X_ptr,  X_stride,
    N,
    alpha,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_pid = tl.program_id(axis=0)

    dX_ptr += row_pid * dX_stride
    dY_ptr += row_pid * dY_stride
    X_ptr  += row_pid * X_stride

    col_offs = tl.arange(0, BLOCK_SIZE // 2)
    mask = col_offs < N

    dy = tl.load(dY_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)

    g_raw = tl.load(X_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    l_raw = tl.load(X_ptr + N + col_offs, mask=mask, other=0.0).to(tl.float32)

    g = tl.minimum(g_raw, limit)
    l = tl.minimum(tl.maximum(l_raw, -limit), limit)

    sig = tl.sigmoid(alpha * g)
    base = dy * g * sig                       # common factor for dl branch

    # d(out)/dg = [sig + g*alpha*sig*(1-sig)] * (l+1), zero when clamped
    dg_active = g_raw < limit                 # clamp(max=...) kills grad above limit
    dx_gate = tl.where(
        dg_active,
        dy * (sig + g * alpha * sig * (1.0 - sig)) * (l + 1.0),
        0.0,
    )

    # d(out)/dl = g * sig, zero when clamped on either side
    dl_active = (l_raw <= limit) & (l_raw >= -limit)
    dx_linear = tl.where(dl_active, base, 0.0)

    tl.store(dX_ptr + col_offs, dx_gate.to(dX_ptr.dtype.element_ty), mask=mask)
    tl.store(dX_ptr + N + col_offs, dx_linear.to(dX_ptr.dtype.element_ty), mask=mask)


def gemma_swiglu_forward(X, alpha, limit):
  shape = X.shape
  dim = shape[-1]
  X = X.view(-1, dim)
  M, N = X.shape
  N_out = N // 2

  Y = torch.empty((M, N_out), dtype=X.dtype, device=X.device)

  _gemma_swiglu_fwd_kernel[(M,)](
      X,
      X.stride(0),
      Y,
      Y.stride(0),
      N_out,
      alpha,
      limit,
  )
  out_shape = list(shape[:-1]) + [N_out]
  return Y.view(out_shape)


def gemma_swiglu_backward(dY, X, alpha, limit):
  shape = X.shape
  dim = shape[-1]
  X = X.view(-1, dim)
  M, N = X.shape
  N_out = N // 2

  dX = torch.empty_like(X)
  dY = dY.view(-1, N_out)

  _gemma_swiglu_bwd_kernel[(M,)](
      dX,
      dX.stride(0),
      dY,
      dY.stride(0),
      X,
      X.stride(0),
      N_out,
      alpha,
      limit,
  )

  return dX.view(*shape)


class TritonGemmaSwigluFunction(torch.autograd.Function):
  @staticmethod
  @ensure_contiguous
  def forward(ctx, X, alpha, limit):
    Y = gemma_swiglu_forward(X, alpha, limit)
    ctx.alpha = alpha
    ctx.limit = limit
    ctx.save_for_backward(X)
    return Y

  @staticmethod
  @ensure_contiguous
  def backward(ctx, dY):
    X, = ctx.saved_tensors
    dX = gemma_swiglu_backward(dY, X, ctx.alpha, ctx.limit)
    return dX, None, None


# ------ Testing script --------

def gemma_swiglu_torch(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Naive PyTorch reference — bit-exact mirror of model_flash_attn.swiglu."""
    x_glu, x_linear = x.chunk(2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


# =====================================================================
#  VERIFICATION & BENCHMARK SUITE
# =====================================================================

def test_correctness():
    print("=" * 70)
    print(" NUMERICAL CORRECTNESS CHECKS (FP32, FP16, BF16)")
    print("=" * 70)

    device = "cuda"
    shape = (16, 2048, 1520)  # B, S, 2*I  (I=760 for model_flash_attn config)

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    atols = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1.5e-1}
    rtols = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}

    for dtype in dtypes:
        torch.manual_seed(42)

        x_naive = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        x_triton = x_naive.detach().clone().requires_grad_(True)
        dout = torch.randn((shape[0], shape[1], shape[2] // 2), device=device, dtype=dtype)

        out_naive = gemma_swiglu_torch(x_naive)
        out_triton = TritonGemmaSwigluFunction.apply(x_triton, 1.702, 7.0)

        out_naive.backward(dout)
        out_triton.backward(dout)

        fwd_pass = torch.allclose(out_naive, out_triton, atol=atols[dtype], rtol=rtols[dtype])
        bwd_pass = torch.allclose(x_naive.grad, x_triton.grad, atol=atols[dtype], rtol=rtols[dtype])

        status_fwd = "✅ PASSED" if fwd_pass else "❌ FAILED"
        status_bwd = "✅ PASSED" if bwd_pass else "❌ FAILED"
        print(f"[{str(dtype):<14}] Fwd ({status_fwd}) | Bwd ({status_bwd})")

    print("\nClamp-boundary sanity (values near ±7 must match reference exactly):")
    edge = torch.tensor([[-10.0, -10.0], [-7.001, -7.001], [0.0, 0.0],
                         [6.999, 6.999], [7.0, 7.0], [10.0, 10.0]],
                        device=device, dtype=torch.float32, requires_grad=True)
    edge_tri = edge.detach().clone().requires_grad_(True)
    ref = gemma_swiglu_torch(edge)
    tri = TritonGemmaSwigluFunction.apply(edge_tri, 1.702, 7.0)
    torch.testing.assert_close(ref, tri, atol=1e-6, rtol=1e-6)
    print("✅ boundary values match")


if __name__ == "__main__":
    test_correctness()
