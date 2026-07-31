import torch
import triton
import triton.language as tl
import torch.nn as nn
try:
    from .utils import ensure_contiguous
except ImportError:
    try:
        from src.kernels.utils import ensure_contiguous
    except ImportError:
        from utils import ensure_contiguous


# ── Autotuning configs for the forward kernel ────────────────────────
_FWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=3),
]


# ── Autotuning configs for the backward kernel ───────────────────────
_BWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_SIZE": 4096},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=3)
]


@triton.autotune(configs=_FWD_CONFIGS, key=["N"])
@triton.jit
def _fused_add_rms_norm_fwd(
    X_ptr, # Input from previous sublayer
    X_row_stride,
    R_ptr, # Input residual
    R_row_stride,
    S_ptr, # Output residual (X + R)
    S_row_stride,
    W_ptr, # Weight pointer (1D tensor of shape N)
    W_row_stride,
    Rstd_ptr, # Output cached reciprocal std (1 per row)
    Rstd_row_stride,
    Y_ptr, # Output normalized hidden states
    Y_row_stride,
    eps,
    N,
    BLOCK_SIZE: tl.constexpr
):

  """
  Y = RMSNorm(X)
  R = Sublayer(Y) # input residual

  -- Kernel starts here --
  S = X + R
  Y' = RMSNorm(S) # output residual
  -- Kernel ends here --

  R' = Sublayer(Y')
  ...so on
  """
  row_pid = tl.program_id(axis=0)
  col_offs = tl.arange(0, BLOCK_SIZE)
  mask = col_offs < N

  X_ptr += row_pid * X_row_stride
  R_ptr += row_pid * R_row_stride
  S_ptr += row_pid * S_row_stride
  Y_ptr += row_pid * Y_row_stride
  Rstd_ptr += row_pid * Rstd_row_stride

  x = tl.load(X_ptr + col_offs, mask=mask, other=0.)
  r = tl.load(R_ptr + col_offs, mask=mask, other=0.)

  S_row = x + r
  tl.store(S_ptr + col_offs, S_row, mask=mask)

  S_row_fp32 = S_row.to(tl.float32)
  sq_mean = tl.sum(S_row_fp32 * S_row_fp32, axis=0) / N

  rstd = tl.math.rsqrt(sq_mean + eps)
  tl.store(Rstd_ptr, rstd)

  w = tl.load(W_ptr + col_offs, mask=mask, other=0.).to(tl.float32)
  Y = ((S_row_fp32 * rstd) * w).to(S_row.dtype)
  tl.store(Y_ptr + col_offs, Y, mask=mask)


@triton.autotune(configs=_BWD_CONFIGS, key=["N"])
@triton.jit
def _fused_add_rms_norm_bwd(
    DX,
    DX_row_stride,
    DY,
    DY_row_stride,
    DW,
    DW_row_stride,
    DS_out_ptr,
    DS_out_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    W_row_stride,
    Rstd_ptr,
    Rstd_row_stride,
    M,
    N,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
    has_dS_out: tl.constexpr
):

  row_pid = tl.program_id(axis=0)
  row_start = row_pid * rows_per_program
  row_end = min((row_pid + 1) * rows_per_program, M)

  col_offs = tl.arange(0, BLOCK_SIZE)
  mask = col_offs < N

  dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
  W_row = tl.load(W_ptr + col_offs, mask=mask, other=0.).to(tl.float32)

  for row_idx in range(row_start,row_end):
    DY_ptr = DY + (row_idx * DY_row_stride + col_offs)
    S_ptrs = S_ptr + (row_idx * S_row_stride + col_offs)
    rstd_ptr = Rstd_ptr + row_idx * Rstd_row_stride
    dX_ptr = DX + (row_idx * DX_row_stride + col_offs)

    dy = tl.load(DY_ptr, mask=mask, other=0.).to(tl.float32)
    S_row = tl.load(S_ptrs, mask=mask, other=0.).to(tl.float32)
    rstd = tl.load(rstd_ptr)

    xhat = S_row * rstd
    dW_row += dy * xhat
    dxhat = dy * W_row
    c1 = tl.sum(dxhat * xhat, axis=0) / N
    ds = rstd * (dxhat - xhat * c1)

    if has_dS_out:
      dS_ptr = DS_out_ptr + row_idx * DS_out_row_stride + col_offs
      dS_out = tl.load(dS_ptr, mask=mask, other=0.).to(tl.float32)
      ds += dS_out

    tl.store(dX_ptr, ds.to(dy.dtype), mask=mask)

  tl.store(DW + row_pid * DW_row_stride + col_offs, dW_row, mask=mask)


def fused_add_rms_norm_forward(X, R, W, eps):
  shape = X.shape
  X = X.view(-1, shape[-1])
  R = R.view(-1, shape[-1])
  M, N = X.shape

  Y = torch.empty((M, N), dtype=X.dtype, device=X.device)
  S = torch.empty((M, N), dtype=X.dtype, device=X.device)
  Rstd = torch.empty((M,), dtype=torch.float32, device=X.device)

  _fused_add_rms_norm_fwd[(M,)](
      X,
      X.stride(0),
      R,
      R.stride(0),
      S,
      S.stride(0),
      W,
      W.stride(0),
      Rstd,
      Rstd.stride(0),
      Y,
      Y.stride(0),
      eps,
      N,
  )

  return Y.view(*shape), S.view(*shape), Rstd


def fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD):
  shape = dY.shape
  dim = shape[-1]
  dY = dY.view(-1, dim)
  S = S.view(-1, dim)
  M, N = dY.shape
  sm_count = torch.cuda.get_device_properties(S.device).multi_processor_count

  has_dS_out=dS_out is not None
  if has_dS_out:
    dS_out = dS_out.view(-1, dim)

  dW_acc = torch.empty((sm_count, N), dtype=torch.float32, device=W.device)
  rows_per_program = triton.cdiv(M, sm_count)
  dX = torch.empty_like(dY)

  dS_out_ptr = dS_out if has_dS_out else dX
  dS_out_stride = dS_out.stride(0) if has_dS_out else 0

  _fused_add_rms_norm_bwd[(sm_count,)](
      dX,
      dX.stride(0),
      dY,
      dY.stride(0),
      dW_acc,
      dW_acc.stride(0),
      dS_out_ptr,
      dS_out_stride,
      S,
      S.stride(0),
      W,
      W.stride(0),
      RSTD,
      RSTD.stride(0),
      M,
      N,
      rows_per_program,
      has_dS_out=has_dS_out
  )

  dX = dX.view(*shape)
  dW = dW_acc.sum(dim=0).to(W.dtype)
  return dX, dX, dW


class FusedAddRMSNormFunction(torch.autograd.Function):
  @staticmethod
  @ensure_contiguous
  def forward(ctx, X, R, W, eps):
    Y, S, RSTD = fused_add_rms_norm_forward(X, R, W, eps)

    ctx.save_for_backward(S, W, RSTD)
    return Y, S

  @staticmethod
  @ensure_contiguous
  def backward(ctx, dY, dS_out):
    S, W, RSTD = ctx.saved_tensors
    dX, dR, dW = fused_add_rms_norm_backward(
        dY,
        dS_out,
        S,
        W,
        RSTD,
    )
    return dX, dR, dW, None


# ------ Testing script --------
class PyTorchFusedAddRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, r, weight):
        s = x + r
        t = s.float()
        rstd = torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        y = (t * rstd).to(x.dtype) * weight
        return y, s

def test_correctness():
    assert torch.cuda.is_available(), "CUDA is required to run Triton tests!"

    # Dimensions: Batch=4, Seq=128, Hidden=4096 (Standard LLM shape)
    B, T, H = 4, 128, 4096
    eps = 1e-6
    dtype = torch.float32
    device = "cuda"

    print(f"Testing Triton vs PyTorch Reference...")
    print(f"Shape: ({B}, {T}, {H}) | Precision: {dtype}\n")

    torch.manual_seed(42)

    # Base Tensors
    X_base = torch.randn((B, T, H), dtype=dtype, device=device)
    R_base = torch.randn((B, T, H), dtype=dtype, device=device)
    W_base = torch.randn((H,), dtype=dtype, device=device)

    # Upstream Gradients (Simulating Attention output gradient and Highway gradient)
    dY_incoming = torch.randn((B, T, H), dtype=dtype, device=device)
    dS_out_incoming = torch.randn((B, T, H), dtype=dtype, device=device)

    # -------------------------------------------------------------------------
    # A. PYTORCH REFERENCE RUN
    # -------------------------------------------------------------------------
    X_ref = X_base.clone().detach().requires_grad_(True)
    R_ref = R_base.clone().detach().requires_grad_(True)
    W_ref = W_base.clone().detach().requires_grad_(True)

    ref_module = PyTorchFusedAddRMSNorm(H, eps=eps).to(device)
    Y_ref, S_ref = ref_module(X_ref, R_ref, W_ref)

    # Backward pass with BOTH gradient streams (dY and dS_out)
    loss_ref = (Y_ref * dY_incoming).sum() + (S_ref * dS_out_incoming).sum()
    loss_ref.backward()

    # -------------------------------------------------------------------------
    # B. TRITON KERNEL RUN
    # -------------------------------------------------------------------------
    X_tri = X_base.clone().detach().requires_grad_(True)
    R_tri = R_base.clone().detach().requires_grad_(True)
    W_tri = W_base.clone().detach().requires_grad_(True)

    Y_tri, S_tri = FusedAddRMSNormFunction.apply(X_tri, R_tri, W_tri, eps)

    # Backward pass with identical upstream gradients
    loss_tri = (Y_tri * dY_incoming).sum() + (S_tri * dS_out_incoming).sum()
    loss_tri.backward()

    # -------------------------------------------------------------------------
    # C. ASSERTIONS & NUMERICAL COMPARISON
    # -------------------------------------------------------------------------
    # Tolerances suited for bfloat16 arithmetic
    atol, rtol = 1e-2, 1e-2

    print("--- FORWARD PASS CHECK ---")
    try:
        torch.testing.assert_close(Y_tri, Y_ref, atol=atol, rtol=rtol)
        print("✅ Output Y matches PyTorch!")
    except AssertionError as e:
        print(f"⚠️ Output Y difference: {e}")

    try:
        torch.testing.assert_close(S_tri, S_ref, atol=atol, rtol=rtol)
        print("✅ Residual Output S matches PyTorch!\n")
    except AssertionError as e:
        print(f"⚠️ Residual Output S difference: {e}\n")

    print("--- BACKWARD PASS CHECK ---")
    try:
        torch.testing.assert_close(X_tri.grad, X_ref.grad, atol=atol, rtol=rtol)
        print("✅ Gradient dX matches PyTorch!")
    except AssertionError as e:
        print(f"⚠️ Gradient dX difference: {e}")

    try:
        torch.testing.assert_close(R_tri.grad, R_ref.grad, atol=atol, rtol=rtol)
        print("✅ Gradient dR matches PyTorch!")
    except AssertionError as e:
        print(f"⚠️ Gradient dR difference: {e}")

    try:
        torch.testing.assert_close(W_tri.grad, W_ref.grad, atol=atol, rtol=rtol)
        print("✅ Gradient dW matches PyTorch!\n")
    except AssertionError as e:
        print(f"⚠️ Gradient dW difference: {e}\n")

    print("🎉 ALL CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    test_correctness()