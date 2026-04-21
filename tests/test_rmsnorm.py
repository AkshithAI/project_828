"""
RMS_Norm — Production Correctness Suite
========================================

Validates the RMSNorm implementation used in every TransformerDecoderBLK
(pre-norm) and as the final norm in GPT_FLASH. Covers:

  - Normalization correctness (output RMS ≈ 1.0 when scale=1)
  - Shape preservation across 2D, 3D, 4D inputs
  - Scale parameter learning and effect
  - Gradient flow through normalization
  - Numerical stability (tiny, huge, zero inputs)
  - fp32 / bf16 consistency
  - eps parameter sensitivity
"""

import sys
import os
import math
import types
import pytest
import torch
import torch.nn as nn
from torch.amp import autocast

# ── Mock flash_attn when not installed (CPU/Mac) ─────────────────────
_mock = types.ModuleType("flash_attn")
def _fa_mock(Q, K, V, causal=False):
    Q_t = Q.transpose(1, 2)
    K_t = K.transpose(1, 2)
    V_t = V.transpose(1, 2)
    n_q_heads = Q_t.shape[1]
    n_kv_heads = K_t.shape[1]
    if n_q_heads != n_kv_heads:
        n_rep = n_q_heads // n_kv_heads
        K_t = K_t.repeat_interleave(n_rep, dim=1)
        V_t = V_t.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q_t, K_t, V_t, is_causal=causal
    ).transpose(1, 2)
_mock.flash_attn_func = _fa_mock
sys.modules.setdefault("flash_attn", _mock)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_flash_attn import RMS_Norm


# ── Helpers ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _rms(t: torch.Tensor) -> torch.Tensor:
    """Compute RMS of t along last dimension."""
    return torch.sqrt(torch.mean(t.float() ** 2, dim=-1))


# ── SECTION 1: Shape preservation ──────────────────────────────────────────────

class TestShapePreservation:
    """RMSNorm must not change tensor shapes — ever."""

    def test_2d_input(self):
        """(N, D) input → (N, D) output."""
        norm = RMS_Norm(768)
        x = torch.randn(64, 768)
        out = norm(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    def test_3d_input(self):
        """(B, S, D) input → (B, S, D) output."""
        norm = RMS_Norm(768)
        x = torch.randn(4, 128, 768)
        out = norm(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    def test_4d_input(self):
        """(B, S, H, D) input — used for per-head Q/K normalization."""
        head_dim = 64
        norm = RMS_Norm(head_dim)
        x = torch.randn(4, 128, 12, head_dim)
        out = norm(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    def test_single_element(self):
        """(1, D) edge case."""
        norm = RMS_Norm(768)
        x = torch.randn(1, 768)
        out = norm(x)
        assert out.shape == (1, 768)

    def test_batch_1_seq_1(self):
        """(1, 1, D) — minimal 3D input."""
        norm = RMS_Norm(768)
        x = torch.randn(1, 1, 768)
        out = norm(x)
        assert out.shape == (1, 1, 768)


# ── SECTION 2: Normalization correctness ─────────────────────────────────────

class TestNormalizationCorrectness:
    """
    When scale=1 (default), the RMS of the output along the last dim
    should be approximately 1.0 for each token position.
    """

    def test_output_rms_close_to_one(self):
        """RMS of output ≈ 1.0 when scale = ones."""
        norm = RMS_Norm(768)
        x = torch.randn(64, 768)
        out = norm(x)

        rms_vals = _rms(out)
        torch.testing.assert_close(
            rms_vals,
            torch.ones_like(rms_vals),
            atol=1e-4, rtol=1e-4,
            msg="RMS of output should be ≈ 1.0 with unit scale",
        )

    def test_output_rms_3d(self):
        """RMS ≈ 1.0 for 3D input across all positions."""
        norm = RMS_Norm(768)
        x = torch.randn(4, 128, 768)
        out = norm(x)

        rms_vals = _rms(out)  # (4, 128)
        torch.testing.assert_close(
            rms_vals,
            torch.ones_like(rms_vals),
            atol=1e-4, rtol=1e-4,
        )

    def test_idempotent_on_unit_rms(self):
        """Normalizing already-normalized data should be close to identity × scale."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768)
        # First normalize
        out1 = norm(x)
        # Normalizing again should give very similar result
        out2 = norm(out1)

        rms_vals = _rms(out2)
        torch.testing.assert_close(
            rms_vals,
            torch.ones_like(rms_vals),
            atol=1e-3, rtol=1e-3,
        )

    def test_different_from_layer_norm(self):
        """RMSNorm differs from LayerNorm — no mean subtraction."""
        norm = RMS_Norm(768)
        ln = nn.LayerNorm(768, elementwise_affine=False)

        x = torch.randn(32, 768) + 5.0  # non-zero mean
        out_rms = norm(x)
        out_ln = ln(x)

        # They should differ because RMSNorm doesn't subtract mean
        assert not torch.allclose(out_rms, out_ln, atol=1e-3), (
            "RMSNorm should differ from LayerNorm on non-zero-mean inputs"
        )


# ── SECTION 3: Scale parameter ──────────────────────────────────────────────

class TestScaleParameter:

    def test_scale_is_learnable(self):
        """Scale is a nn.Parameter with requires_grad=True."""
        norm = RMS_Norm(768)
        assert isinstance(norm.scale, nn.Parameter)
        assert norm.scale.requires_grad is True

    def test_scale_initialized_to_ones(self):
        """Scale starts as all ones."""
        norm = RMS_Norm(768)
        torch.testing.assert_close(
            norm.scale, torch.ones(768),
            atol=0.0, rtol=0.0,
        )

    def test_scale_is_float32(self):
        """Scale is always float32 regardless of input dtype (stability)."""
        norm = RMS_Norm(768)
        assert norm.scale.dtype == torch.float32

    def test_scale_affects_output(self):
        """Changing scale should proportionally change output."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768)

        with torch.no_grad():
            out_unit = norm(x.clone())
            norm.scale.fill_(2.0)
            out_double = norm(x.clone())

        torch.testing.assert_close(
            out_double, out_unit * 2.0,
            atol=1e-5, rtol=1e-4,
            msg="Doubling scale should double output",
        )

    def test_scale_per_feature(self):
        """Per-feature scaling: scaling only one feature affects only that feature."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768)

        with torch.no_grad():
            out_base = norm(x.clone())
            norm.scale[0] = 3.0
            out_scaled = norm(x.clone())

        # Feature 0 should change
        assert not torch.allclose(out_base[:, 0], out_scaled[:, 0], atol=1e-5)
        # Feature 1 should NOT change
        torch.testing.assert_close(out_base[:, 1], out_scaled[:, 1], atol=1e-6, rtol=0)

    def test_scale_gradient_flow(self):
        """Gradients flow to scale parameter during training."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768)
        out = norm(x)
        out.sum().backward()
        assert norm.scale.grad is not None, "No gradient on scale"
        assert norm.scale.grad.abs().sum() > 0, "Zero gradient on scale"


# ── SECTION 4: Dtype handling ───────────────────────────────────────────────

class TestDtypeHandling:
    """
    RMSNorm computes in float32 then casts back to input dtype.
    This is critical for bf16 training stability.
    """

    def test_output_preserves_input_dtype_fp32(self):
        norm = RMS_Norm(768)
        x = torch.randn(32, 768, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_output_preserves_input_dtype_bf16(self):
        """bf16 input → bf16 output (computation done in fp32 internally)."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_output_preserves_input_dtype_fp16(self):
        norm = RMS_Norm(768)
        x = torch.randn(32, 768, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_bf16_fp32_close(self):
        """bf16 and fp32 outputs should be close (computation is fp32 internally)."""
        norm = RMS_Norm(768)
        torch.manual_seed(42)
        x_fp32 = torch.randn(32, 768)
        x_bf16 = x_fp32.bfloat16()

        out_fp32 = norm(x_fp32)
        out_bf16 = norm(x_bf16)

        torch.testing.assert_close(
            out_bf16.float(), out_fp32,
            atol=5e-3, rtol=5e-3,
            msg="bf16 and fp32 RMSNorm outputs should be close",
        )

    @requires_cuda
    def test_cuda_bf16_autocast(self):
        """RMSNorm under bf16 autocast on CUDA."""
        norm = RMS_Norm(768, device="cuda")
        x = torch.randn(32, 768, device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = norm(x)

        assert not torch.isnan(out).any(), "NaN in CUDA bf16 autocast output"
        assert not torch.isinf(out).any(), "Inf in CUDA bf16 autocast output"


# ── SECTION 5: Numerical stability ──────────────────────────────────────────────

class TestNumericalStability:

    @pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0, 100.0, 1e4])
    def test_varied_input_scales(self, scale):
        """No NaN/Inf across a wide range of input magnitudes."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768) * scale
        out = norm(x)
        assert not torch.isnan(out).any(), f"NaN at scale {scale}"
        assert not torch.isinf(out).any(), f"Inf at scale {scale}"

    def test_zero_input(self):
        """All-zeros input should not produce NaN (eps prevents div-by-zero)."""
        norm = RMS_Norm(768)
        x = torch.zeros(32, 768)
        out = norm(x)
        assert not torch.isnan(out).any(), "NaN on zero input"
        # Zero input → zero output (0 * anything = 0)
        assert out.abs().max() == 0.0, "Zero input should produce zero output"

    def test_near_zero_input(self):
        """Very small inputs — eps should stabilize."""
        norm = RMS_Norm(768)
        x = torch.full((32, 768), 1e-20)
        out = norm(x)
        assert not torch.isnan(out).any(), "NaN on near-zero input"
        assert not torch.isinf(out).any(), "Inf on near-zero input"

    def test_constant_input(self):
        """Constant input should produce constant output (per feature)."""
        norm = RMS_Norm(768)
        x = torch.full((32, 768), 3.14)
        out = norm(x)
        # All values in a row should be the same (since input is constant)
        row_std = out[0].std().item()
        assert row_std < 1e-5, f"Constant input should give constant output, std={row_std}"

    def test_eps_prevents_instability(self):
        """Smaller eps should still produce finite outputs."""
        for eps in [1e-12, 1e-8, 1e-5, 1e-2]:
            norm = RMS_Norm(768, eps=eps)
            x = torch.randn(32, 768) * 1e-6
            out = norm(x)
            assert not torch.isnan(out).any(), f"NaN with eps={eps}"
            assert not torch.isinf(out).any(), f"Inf with eps={eps}"


# ── SECTION 6: Gradient flow ─────────────────────────────────────────────────────

class TestGradientFlow:

    def test_input_gradient(self):
        """Gradients flow through RMSNorm to the input."""
        norm = RMS_Norm(768)
        x = torch.randn(32, 768, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Zero gradient on input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not explode through RMSNorm."""
        norm = RMS_Norm(768)
        x = torch.randn(128, 768, requires_grad=True)
        norm(x).sum().backward()
        max_grad = x.grad.abs().max().item()
        assert max_grad < 100, f"Gradient exploded: max={max_grad}"

    def test_gradient_through_chain(self):
        """Two sequential RMSNorms — gradient should flow through both."""
        norm1 = RMS_Norm(768)
        norm2 = RMS_Norm(768)
        x = torch.randn(32, 768, requires_grad=True)
        out = norm2(norm1(x))
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        assert norm1.scale.grad is not None
        assert norm2.scale.grad is not None


# ── SECTION 7: Determinism ─────────────────────────────────────────────

class TestDeterminism:

    def test_deterministic_same_input(self):
        """Same input → same output, every time."""
        norm = RMS_Norm(768)
        norm.eval()
        x = torch.randn(32, 768)
        with torch.no_grad():
            out1 = norm(x.clone())
            out2 = norm(x.clone())
        torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)

    def test_state_dict_roundtrip(self):
        """Save → load state dict, output must be identical."""
        norm = RMS_Norm(768)
        with torch.no_grad():
            norm.scale.fill_(2.5)

        x = torch.randn(32, 768)
        with torch.no_grad():
            out_before = norm(x.clone())

        state = norm.state_dict()
        norm2 = RMS_Norm(768)
        norm2.load_state_dict(state)

        with torch.no_grad():
            out_after = norm2(x.clone())

        torch.testing.assert_close(out_before, out_after, atol=0.0, rtol=0.0)


# ── SECTION 8: Device consistency ───────────────────────────────────────────

class TestDeviceConsistency:

    @requires_cuda
    def test_cpu_cuda_match(self):
        """CPU and CUDA outputs should match."""
        torch.manual_seed(42)
        norm_cpu = RMS_Norm(768)
        norm_cuda = RMS_Norm(768, device="cuda")
        norm_cuda.load_state_dict(norm_cpu.state_dict())

        x = torch.randn(32, 768)
        with torch.no_grad():
            out_cpu = norm_cpu(x)
            out_cuda = norm_cuda(x.to("cuda"))

        torch.testing.assert_close(
            out_cpu, out_cuda.cpu(),
            atol=1e-6, rtol=1e-5,
            msg="CPU and CUDA RMSNorm outputs differ",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
