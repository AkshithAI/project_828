"""
Attention (GQA + RoPE + KV Cache) — Production Correctness Suite
================================================================
Validates the Attention module from model_flash_attn.py including:

  - GQA correctness (n_heads=12, n_kv_heads=6)
  - Q/K RMSNorm pre-rotation
  - RoPE integration
  - KV cache for autoregressive inference
  - Causal masking (future tokens don't leak)
  - Flash attention fallback to SDPA
  - Shape contracts and gradient flow
  - Numerical stability
"""
import sys, os, copy, math, types, pytest, torch, torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

_mock = types.ModuleType("flash_attn")
def _fa_mock(Q, K, V, causal=False):
    Q_t, K_t, V_t = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)
    nq, nkv = Q_t.shape[1], K_t.shape[1]
    if nq != nkv:
        K_t = K_t.repeat_interleave(nq//nkv, dim=1)
        V_t = V_t.repeat_interleave(nq//nkv, dim=1)
    return F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=causal).transpose(1,2)
_mock.flash_attn_func = _fa_mock
sys.modules.setdefault("flash_attn", _mock)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_flash_attn import Attention
from src.scripts.configs.model_config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

def _make_config(**overrides) -> ModelConfig:
    cfg = ModelConfig.__new__(ModelConfig)
    cfg.hidden_dim = overrides.get("hidden_dim", 768)
    cfg.intermediate_size = overrides.get("intermediate_size", 760)
    cfg.num_experts = overrides.get("num_experts", 4)
    cfg.num_experts_per_tok = overrides.get("num_experts_per_tok", 2)
    cfg.update_param = overrides.get("update_param", 1e-3)
    cfg.route_scale = overrides.get("route_scale", 1.0)
    cfg.ffn_dropout = overrides.get("ffn_dropout", 0.0)
    cfg.dtype = overrides.get("dtype", torch.float32)
    cfg.vocab_size = overrides.get("vocab_size", 100)
    cfg.num_attn_heads = overrides.get("num_attn_heads", 12)
    cfg.num_key_value_heads = overrides.get("num_key_value_heads", 6)
    cfg.head_dim = cfg.hidden_dim // cfg.num_attn_heads
    cfg.num_hidden_layers = overrides.get("num_hidden_layers", 2)
    cfg.base = 10000
    cfg.initial_context_len = 2048
    cfg.max_context_len = overrides.get("max_context_len", 2048)
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    cfg.attn_logit_cap = overrides.get("attn_logit_cap", 0.0)
    return cfg



# ── SECTION 1: Shape contracts ─────────────────────────────────────────────


class TestAttentionShapes:

    def test_output_shape(self):
        """(B, S, D) → (B, S, D)."""
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 128, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_single_token(self):
        """(1, 1, D) — minimal input."""
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(1, 1, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == (1, 1, cfg.hidden_dim)

    def test_batch_1(self):
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(1, 64, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == (1, 64, cfg.hidden_dim)

    def test_large_batch(self):
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(8, 32, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == (8, 32, cfg.hidden_dim)



# ── SECTION 2: GQA correctness ──────────────────────────────────────────────


class TestGQA:

    def test_gqa_ratio(self):
        """Production config: 12 query heads, 6 kv heads (2:1 ratio)."""
        cfg = _make_config(num_attn_heads=12, num_key_value_heads=6)
        attn = Attention(cfg)
        assert attn.n_heads == 12
        assert attn.n_kv_heads == 6
        assert attn.n_heads // attn.n_kv_heads == 2

    def test_projection_dimensions(self):
        """wq projects to n_heads*head_dim, wk/wv to n_kv_heads*head_dim."""
        cfg = _make_config(hidden_dim=768, num_attn_heads=12, num_key_value_heads=6)
        attn = Attention(cfg)
        assert attn.wq.weight.shape == (12 * 64, 768)
        assert attn.wk.weight.shape == (6 * 64, 768)
        assert attn.wv.weight.shape == (6 * 64, 768)
        assert attn.wo.weight.shape == (768, 12 * 64)

    def test_mha_equivalent(self):
        """When n_heads == n_kv_heads, it's standard MHA — should still work."""
        cfg = _make_config(num_attn_heads=8, num_key_value_heads=8,
                           hidden_dim=512)
        cfg.head_dim = 512 // 8
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 32, 512)
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_mqa_equivalent(self):
        """When n_kv_heads == 1, it's MQA — should still work."""
        cfg = _make_config(num_attn_heads=8, num_key_value_heads=1,
                           hidden_dim=512)
        cfg.head_dim = 512 // 8
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 32, 512)
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()



# ── SECTION 3: Q/K Normalization ─────────────────────────────────────────


class TestQKNorm:

    def test_qk_norm_modules_exist(self):
        """q_norm and k_norm should be RMS_Norm instances."""
        cfg = _make_config()
        attn = Attention(cfg)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')
        assert attn.q_norm.num_features == cfg.head_dim
        assert attn.k_norm.num_features == cfg.head_dim

    def test_qk_norm_reduces_variance(self):
        """After Q/K normalization, per-head vectors should have unit RMS."""
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()

        # Manually check by running partial forward
        x = torch.randn(2, 32, cfg.hidden_dim)
        Q = attn.wq(x).view(2, 32, attn.n_heads, attn.head_dim)
        Q_normed = attn.q_norm(Q)

        # RMS should be ≈ 1.0 per head
        rms = torch.sqrt(torch.mean(Q_normed.float() ** 2, dim=-1))
        torch.testing.assert_close(
            rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3,
        )



# ── SECTION 4: Causal masking ──────────────────────────────────────────


class TestCausalMasking:

    def test_future_tokens_dont_affect_past(self):
        """
        Changing future token should not change output for earlier positions.
        This verifies the causal mask works correctly.
        """
        cfg = _make_config()
        torch.manual_seed(42)
        attn = Attention(cfg); attn.eval()

        x = torch.randn(1, 8, cfg.hidden_dim)
        with torch.no_grad():
            out_full = attn(x.clone())

        # Modify the last token
        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(cfg.hidden_dim)
        with torch.no_grad():
            out_modified = attn(x_modified)

        # Positions 0-6 should be identical (causal: can't see position 7)
        torch.testing.assert_close(
            out_full[0, :7], out_modified[0, :7],
            atol=1e-5, rtol=1e-4,
            msg="Causal violation: future token affected past positions",
        )

    def test_first_token_sees_only_itself(self):
        """First token output should only depend on first token input."""
        cfg = _make_config()
        torch.manual_seed(42)
        attn = Attention(cfg); attn.eval()

        x1 = torch.randn(1, 16, cfg.hidden_dim)
        x2 = x1.clone()
        x2[0, 1:] = torch.randn(15, cfg.hidden_dim)  # Change all but first

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        torch.testing.assert_close(
            out1[0, 0], out2[0, 0], atol=1e-5, rtol=1e-4,
            msg="First token should only see itself",
        )



# ── SECTION 5: KV Cache for Inference ─────────────────────────────────────


class TestKVCache:

    def test_inference_mode_flag(self):
        """Inference mode creates cache buffers."""
        cfg = _make_config(max_context_len=256)
        attn = Attention(cfg, inference=True)
        assert attn.inference is True

    def test_cache_allocation(self):
        cfg = _make_config(max_context_len=256)
        attn = Attention(cfg, inference=True)
        attn.reset_cache(batch_size=4)
        assert attn.cache_k.shape == (4, cfg.num_key_value_heads, 256, cfg.head_dim)
        assert attn.cache_v.shape == (4, cfg.num_key_value_heads, 256, cfg.head_dim)

    def test_prefill_then_decode_matches_full(self):
        """
        Prefill + decode one-by-one should produce the same output
        as processing the full sequence at once (non-cached).
        """
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)

        attn_full = Attention(cfg, inference=False)
        attn_cached = Attention(cfg, inference=True)
        attn_cached.load_state_dict(attn_full.state_dict())
        attn_full.eval(); attn_cached.eval()

        seq_len = 8
        x = torch.randn(1, seq_len, cfg.hidden_dim)

        # Full forward (no cache)
        with torch.no_grad():
            out_full = attn_full(x)

        # Cached: prefill all at once, then check
        attn_cached.reset_cache(batch_size=1)
        with torch.no_grad():
            out_cached = attn_cached(x, start_pos=0)

        torch.testing.assert_close(
            out_full, out_cached, atol=1e-4, rtol=1e-3,
            msg="Cached prefill should match non-cached forward",
        )

    def test_incremental_decode(self):
        """
        Prefill with first N-1 tokens, then decode the Nth token.
        The Nth output should match the full-sequence result.
        """
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)

        attn_full = Attention(cfg, inference=False)
        attn_cached = Attention(cfg, inference=True)
        attn_cached.load_state_dict(attn_full.state_dict())
        attn_full.eval(); attn_cached.eval()

        seq_len = 8
        x = torch.randn(1, seq_len, cfg.hidden_dim)

        # Full forward
        with torch.no_grad():
            out_full = attn_full(x)

        # Cached: prefill first 7, then decode position 7
        attn_cached.reset_cache(batch_size=1)
        with torch.no_grad():
            _ = attn_cached(x[:, :7, :], start_pos=0)
            out_last = attn_cached(x[:, 7:8, :], start_pos=7)

        torch.testing.assert_close(
            out_full[0, 7:8], out_last[0],
            atol=1e-4, rtol=1e-3,
            msg="Incremental decode should match full forward at last position",
        )

    def test_cache_reuse_across_steps(self):
        """Multiple single-token decode steps should accumulate correctly."""
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)
        attn = Attention(cfg, inference=True); attn.eval()
        attn.reset_cache(batch_size=1)

        outputs = []
        for pos in range(8):
            x = torch.randn(1, 1, cfg.hidden_dim)
            with torch.no_grad():
                out = attn(x, start_pos=pos)
            outputs.append(out)
            assert out.shape == (1, 1, cfg.hidden_dim)
            assert not torch.isnan(out).any()



# ── SECTION 6: Gradient flow ─────────────────────────────────────────────


class TestAttentionGradients:

    def test_gradient_to_all_projections(self):
        """Gradients flow to wq, wk, wv, wo."""
        cfg = _make_config()
        attn = Attention(cfg); attn.train()
        x = torch.randn(2, 32, cfg.hidden_dim)
        out = attn(x)
        out.sum().backward()

        for name in ['wq', 'wk', 'wv', 'wo']:
            w = getattr(attn, name)
            assert w.weight.grad is not None, f"No gradient on {name}"
            assert w.weight.grad.abs().sum() > 0, f"Zero gradient on {name}"

    def test_gradient_to_qk_norms(self):
        """Gradients flow through Q/K normalization."""
        cfg = _make_config()
        attn = Attention(cfg); attn.train()
        x = torch.randn(2, 32, cfg.hidden_dim)
        attn(x).sum().backward()
        assert attn.q_norm.scale.grad is not None
        assert attn.k_norm.scale.grad is not None

    def test_gradient_to_input(self):
        cfg = _make_config()
        attn = Attention(cfg); attn.train()
        x = torch.randn(2, 32, cfg.hidden_dim, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        assert not torch.isnan(x.grad).any()

    def test_gradient_magnitude_reasonable(self):
        cfg = _make_config()
        attn = Attention(cfg); attn.train()
        x = torch.randn(2, 64, cfg.hidden_dim, requires_grad=True)
        attn(x).sum().backward()
        max_grad = x.grad.abs().max().item()
        assert max_grad < 1e4, f"Gradient exploded: max={max_grad}"



# ── SECTION 7: Numerical stability ───────────────────────────────────────────


class TestAttentionStability:

    def test_no_nan_fp32(self):
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 64, cfg.hidden_dim)
        with torch.no_grad():
            out = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("scale", [1e-4, 1.0, 10.0])
    def test_varied_input_scales(self, scale):
        cfg = _make_config()
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 32, cfg.hidden_dim) * scale
        with torch.no_grad():
            out = attn(x)
        assert not torch.isnan(out).any(), f"NaN at scale {scale}"

    def test_deterministic(self):
        cfg = _make_config()
        torch.manual_seed(42)
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 32, cfg.hidden_dim)
        with torch.no_grad():
            o1 = attn(x.clone())
            o2 = attn(x.clone())
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-5)

    def test_state_dict_roundtrip(self):
        cfg = _make_config()
        torch.manual_seed(42)
        attn = Attention(cfg); attn.eval()
        x = torch.randn(2, 32, cfg.hidden_dim)
        with torch.no_grad():
            out_before = attn(x.clone())
        attn2 = Attention(cfg)
        attn2.load_state_dict(attn.state_dict()); attn2.eval()
        with torch.no_grad():
            out_after = attn2(x.clone())
        torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-5)

    @requires_cuda
    def test_cuda_bf16(self):
        cfg = _make_config(dtype=torch.bfloat16)
        attn = Attention(cfg, device="cuda"); attn.eval()
        x = torch.randn(2, 64, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = attn(x)
        assert not torch.isnan(out).any()
        assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
