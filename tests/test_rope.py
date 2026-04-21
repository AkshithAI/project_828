"""
RoPE & Rotary Embedding — Production Correctness Suite
======================================================
Validates apply_rope and RotaryEmbedding with YaRN scaling.

Coverage:
  - Rotation correctness (norm preservation)
  - Position sensitivity (different positions → different outputs)
  - YaRN scaling (scaling_factor > 1)
  - cos/sin table shapes and precomputation
  - Offset and position_ids support
  - Gradient flow through RoPE
  - 2D vs 3D cos/sin dispatch in apply_rope
"""
import sys, os, math, types, pytest, torch, torch.nn as nn

_mock = types.ModuleType("flash_attn")
def _fa_mock(Q, K, V, causal=False):
    Q_t, K_t, V_t = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)
    nq, nkv = Q_t.shape[1], K_t.shape[1]
    if nq != nkv:
        K_t = K_t.repeat_interleave(nq//nkv, dim=1)
        V_t = V_t.repeat_interleave(nq//nkv, dim=1)
    return nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=causal).transpose(1,2)
_mock.flash_attn_func = _fa_mock
sys.modules.setdefault("flash_attn", _mock)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_flash_attn import apply_rope, RotaryEmbedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ── SECTION 1: apply_rope function ─────────────────────────────────────────

class TestApplyRoPE:

    def test_output_shape_preserved(self):
        """apply_rope preserves input shape."""
        head_dim = 64
        x = torch.randn(2, 128, 12, head_dim)
        cos = torch.randn(128, head_dim // 2)
        sin = torch.randn(128, head_dim // 2)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_norm_preservation(self):
        """Rotation should preserve vector norms (orthogonal transform)."""
        head_dim = 64
        torch.manual_seed(42)
        x = torch.randn(2, 128, 12, head_dim)

        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=256)
        cos = rope.cos[:128]
        sin = rope.sin[:128]
        out = apply_rope(x, cos, sin)

        norm_in = x.float().norm(dim=-1)
        norm_out = out.float().norm(dim=-1)
        torch.testing.assert_close(
            norm_in, norm_out, atol=1e-4, rtol=1e-4,
            msg="RoPE should preserve vector norms",
        )

    def test_2d_cos_sin(self):
        """2D cos/sin: (seq_len, head_dim//2) — broadcast over batch."""
        head_dim = 64
        x = torch.randn(4, 32, 6, head_dim)
        cos = torch.ones(32, head_dim // 2)
        sin = torch.zeros(32, head_dim // 2)
        out = apply_rope(x, cos, sin)
        # With cos=1, sin=0: rotation is identity
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-5)

    def test_3d_cos_sin(self):
        """3D cos/sin: (batch, seq_len, head_dim//2) — per-batch positions."""
        head_dim = 64
        x = torch.randn(2, 16, 6, head_dim)
        cos = torch.ones(2, 16, head_dim // 2)
        sin = torch.zeros(2, 16, head_dim // 2)
        out = apply_rope(x, cos, sin)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-5)

    def test_position_sensitivity(self):
        """Different positions should produce different rotations."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=256)
        cos = rope.cos[:128]
        sin = rope.sin[:128]

        x = torch.randn(1, 1, 1, head_dim).expand(1, 128, 1, head_dim).clone()
        out = apply_rope(x, cos, sin)

        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-5), (
            "Different positions should produce different rotations"
        )

    def test_gradient_flow(self):
        """Gradients flow through apply_rope."""
        head_dim = 64
        x = torch.randn(2, 32, 6, head_dim, requires_grad=True)
        cos = torch.randn(32, head_dim // 2)
        sin = torch.randn(32, head_dim // 2)
        out = apply_rope(x, cos, sin)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ── SECTION 2: RotaryEmbedding Module ─────────────────────────────────────────

class TestRotaryEmbedding:

    def test_cos_sin_table_shapes(self):
        """Precomputed cos/sin tables have correct shapes."""
        head_dim = 64
        max_ctx = 2048
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=max_ctx)
        assert rope.cos.shape == (max_ctx, head_dim // 2)
        assert rope.sin.shape == (max_ctx, head_dim // 2)

    def test_cos_sin_bounded(self):
        """cos/sin values should be in [-1, 1] (when no YaRN scaling)."""
        rope = RotaryEmbedding(64, 10000, torch.float32, max_context_len=2048)
        assert rope.cos.max().item() <= 1.0 + 1e-6
        assert rope.cos.min().item() >= -1.0 - 1e-6
        assert rope.sin.max().item() <= 1.0 + 1e-6
        assert rope.sin.min().item() >= -1.0 - 1e-6

    def test_forward_shapes(self):
        """Forward should return Q, K with same shapes as input."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 128, 12, head_dim)
        k = torch.randn(2, 128, 6, head_dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_offset_support(self):
        """Offset should shift the position window."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(1, 32, 12, head_dim)
        k = torch.randn(1, 32, 6, head_dim)

        q_rot0, k_rot0 = rope(q, k, offset=0)
        q_rot10, k_rot10 = rope(q, k, offset=10)
        assert not torch.allclose(q_rot0, q_rot10, atol=1e-5), (
            "Different offsets should produce different rotations"
        )

    def test_position_ids_override(self):
        """position_ids should override offset-based position computation."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 4, 12, head_dim)
        k = torch.randn(2, 4, 6, head_dim)

        # Custom position_ids (non-sequential, e.g., for batched generation with padding)
        pos_ids = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]])
        q_rot, k_rot = rope(q, k, position_ids=pos_ids)
        assert q_rot.shape == q.shape

        # Different position_ids should give different results
        pos_ids2 = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        q_rot2, k_rot2 = rope(q, k, position_ids=pos_ids2)
        assert not torch.allclose(q_rot, q_rot2, atol=1e-5)

    def test_deterministic(self):
        """Same input → same output."""
        rope = RotaryEmbedding(64, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 32, 12, 64)
        k = torch.randn(2, 32, 6, 64)
        q1, k1 = rope(q.clone(), k.clone())
        q2, k2 = rope(q.clone(), k.clone())
        torch.testing.assert_close(q1, q2, atol=0, rtol=0)
        torch.testing.assert_close(k1, k2, atol=0, rtol=0)

    def test_q_k_rotated_differently(self):
        """Q and K with different number of heads should still get rotated."""
        rope = RotaryEmbedding(64, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 32, 12, 64)
        k = torch.randn(2, 32, 6, 64)
        q_rot, k_rot = rope(q, k)
        # Both should differ from input
        assert not torch.allclose(q_rot, q, atol=1e-5)
        assert not torch.allclose(k_rot, k, atol=1e-5)


# ── SECTION 3: YaRN Scaling ─────────────────────────────────────────────

class TestYaRNScaling:
    """YaRN scaling should alter frequency computation when scaling_factor > 1."""

    def test_no_scaling_concentration_is_one(self):
        """With scaling_factor=1.0, concentration should be 1.0."""
        rope = RotaryEmbedding(64, 10000, torch.float32, scaling_factor=1.0,
                               max_context_len=2048)
        conc, inv_freq = rope._compute_concentration_and_inv_freq()
        assert conc == 1.0

    def test_scaling_changes_frequencies(self):
        """scaling_factor > 1 should modify inverse frequencies."""
        rope_base = RotaryEmbedding(64, 10000, torch.float32, scaling_factor=1.0,
                                    initial_context_len=2048, max_context_len=4096)
        rope_scaled = RotaryEmbedding(64, 10000, torch.float32, scaling_factor=2.0,
                                      initial_context_len=2048, max_context_len=4096)
        _, inv_freq_base = rope_base._compute_concentration_and_inv_freq()
        _, inv_freq_scaled = rope_scaled._compute_concentration_and_inv_freq()
        assert not torch.allclose(inv_freq_base, inv_freq_scaled, atol=1e-6)

    def test_scaling_concentration_greater_than_one(self):
        """scaling_factor > 1 → concentration > 1.0."""
        rope = RotaryEmbedding(64, 10000, torch.float32, scaling_factor=2.0,
                               initial_context_len=2048, max_context_len=4096)
        conc, _ = rope._compute_concentration_and_inv_freq()
        assert conc > 1.0

    def test_compute_cos_sin_dynamic(self):
        """compute_cos_sin should work for arbitrary token counts."""
        rope = RotaryEmbedding(64, 10000, torch.float32, max_context_len=2048)
        for n in [1, 10, 128, 2048]:
            cos, sin = rope.compute_cos_sin(n)
            assert cos.shape == (n, 32)
            assert sin.shape == (n, 32)


# ── SECTION 4: Numerical stability & edge cases ────────────────────────────

class TestRoPENumericalStability:

    def test_long_sequences(self):
        """No NaN/Inf at max context length."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(1, 2048, 12, head_dim)
        k = torch.randn(1, 2048, 6, head_dim)
        q_rot, k_rot = rope(q, k)
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()
        assert not torch.isinf(q_rot).any()
        assert not torch.isinf(k_rot).any()

    def test_single_position(self):
        """Single position (seq_len=1) — used in decode step."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(1, 1, 12, head_dim)
        k = torch.randn(1, 1, 6, head_dim)
        q_rot, k_rot = rope(q, k, offset=100)
        assert q_rot.shape == q.shape
        assert not torch.isnan(q_rot).any()

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 100.0])
    def test_varied_input_scales(self, scale):
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 32, 12, head_dim) * scale
        k = torch.randn(2, 32, 6, head_dim) * scale
        q_rot, k_rot = rope(q, k)
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()

    def test_gradient_through_rope(self):
        """Gradients must flow through RotaryEmbedding for training."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048)
        q = torch.randn(2, 32, 12, head_dim, requires_grad=True)
        k = torch.randn(2, 32, 6, head_dim, requires_grad=True)
        q_rot, k_rot = rope(q, k)
        (q_rot.sum() + k_rot.sum()).backward()
        assert q.grad is not None and q.grad.abs().sum() > 0
        assert k.grad is not None and k.grad.abs().sum() > 0

    @requires_cuda
    def test_cuda_bf16(self):
        head_dim = 64
        rope = RotaryEmbedding(head_dim, 10000, torch.float32, max_context_len=2048,
                               device="cuda")
        q = torch.randn(2, 128, 12, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 128, 6, head_dim, device="cuda", dtype=torch.bfloat16)
        q_rot, k_rot = rope(q, k)
        assert not torch.isnan(q_rot).any()
        assert q_rot.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
