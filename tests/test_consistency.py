"""
Consistency Tests for model_improv.py — Triton Kernels vs Eager Fallbacks
=========================================================================
model_improv is the checkpoint-compatible twin of model_flash_attn with the
custom Triton kernels wired in. These tests validate that every accelerated
path produces results equivalent to its eager PyTorch fallback:

  - FusedRMSNorm        vs eager RMSNorm math          (CUDA-gated A/B)
  - FusedAddRMSNorm     vs residual-add + norm         (CUDA-gated A/B)
  - TritonRoPE          vs apply_rope reference        (CUDA-gated A/B)
  - TritonGemmaSwiglu   vs swiglu() reference          (CUDA-gated A/B)
  - FlashAttention      vs SDPA reference              (shim on CPU, real on GPU)
  - Full-model parity   with all kernels toggled ON vs OFF

On CPU every kernel flag degrades to its fallback automatically, so the
toggle-parity checks are bit-exact there; on CUDA they verify numerics
within bf16/fp32 tolerances.
"""
import sys, os, types, pytest, torch
import torch.nn.functional as F

# ── FlashAttention mock ─────────────────────────────────────────────
_mock_fa = types.ModuleType("flash_attn")

def _fa_sdpa(Q, K, V, causal=False):
    """SDPA-based mock of flash_attn_func: (B, S, H, D) → (B, S, H, D)."""
    Q_t, K_t, V_t = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
    nq, nkv = Q_t.shape[1], K_t.shape[1]
    if nq != nkv:
        K_t = K_t.repeat_interleave(nq // nkv, dim=1)
        V_t = V_t.repeat_interleave(nq // nkv, dim=1)
    return F.scaled_dot_product_attention(
        Q_t, K_t, V_t, is_causal=causal
    ).transpose(1, 2)

_mock_fa.flash_attn_func = _fa_sdpa
sys.modules.setdefault("flash_attn", _mock_fa)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.models.model_improv as mi
from src.models.model_improv import (
    GPT_FLASH, Attention, MoE, MLPBlock, RMS_Norm, RotaryEmbedding,
    swiglu, apply_rope,
)
from src.scripts.configs.model_config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

# Module attributes consulted inside forward to pick kernel vs fallback.
_KERNEL_ATTRS = ["FusedRMSNormFunction", "FusedAddRMSNormFunction",
                 "TritonRoPEFunction", "TritonGemmaSwigluFunction"]


# ── Config helper ───────────────────────────────────────────────────

def _make_config(**overrides) -> ModelConfig:
    """Build a lightweight ModelConfig without triggering the tokenizer."""
    cfg = ModelConfig.__new__(ModelConfig)
    cfg.hidden_dim = overrides.get("hidden_dim", 256)
    cfg.intermediate_size = overrides.get("intermediate_size", 320)
    cfg.num_experts = overrides.get("num_experts", 4)
    cfg.num_experts_per_tok = overrides.get("num_experts_per_tok", 2)
    cfg.update_param = overrides.get("update_param", 1e-3)
    cfg.route_scale = overrides.get("route_scale", 1.0)
    cfg.ffn_dropout = overrides.get("ffn_dropout", 0.0)
    cfg.dtype = overrides.get("dtype", torch.float32)
    cfg.vocab_size = overrides.get("vocab_size", 100)
    cfg.num_attn_heads = overrides.get("num_attn_heads", 8)
    cfg.num_key_value_heads = overrides.get("num_key_value_heads", 4)
    cfg.head_dim = cfg.hidden_dim // cfg.num_attn_heads
    cfg.num_hidden_layers = overrides.get("num_hidden_layers", 2)
    cfg.base = 10000
    cfg.initial_context_len = 2048
    cfg.max_context_len = overrides.get("max_context_len", 256)
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    return cfg


def _disable_kernels(monkeypatch):
    for attr in _KERNEL_ATTRS:
        monkeypatch.setattr(mi, attr, None)


# ── RMSNorm consistency ─────────────────────────────────────────────

class TestRMSNormConsistency:
    @pytest.mark.parametrize("shape", [(8, 64), (4, 16, 128)])
    def test_matches_reference_math(self, shape):
        norm = RMS_Norm(shape[-1]).to(DEVICE).eval()
        x = torch.randn(*shape, device=DEVICE)
        out = norm(x)
        t = x.float()
        ref = (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + norm.eps)
               * norm.scale).to(x.dtype)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    @requires_cuda
    def test_kernel_vs_fallback(self, monkeypatch):
        norm = RMS_Norm(128).cuda().eval()
        x = torch.randn(4, 16, 128, device="cuda")
        with torch.no_grad():
            out_kernel = norm(x.clone())
        _disable_kernels(monkeypatch)
        with torch.no_grad():
            out_fallback = norm(x.clone())
        torch.testing.assert_close(out_kernel, out_fallback, atol=1e-5, rtol=1e-5)


# ── RoPE consistency ────────────────────────────────────────────────

class TestRoPEConsistency:
    def test_matches_apply_rope_reference(self):
        B, S, H, D = 2, 32, 4, 64
        rope = RotaryEmbedding(D, 10000, torch.float32,
                               max_context_len=S).to(DEVICE)
        q = torch.randn(B, S, H, D, device=DEVICE)
        k = torch.randn(B, S, H, D, device=DEVICE)
        qr, kr = rope(q, k)
        cos = rope.cos[:S]; sin = rope.sin[:S]
        assert torch.allclose(qr, apply_rope(q.view(B, S, H, D), cos, sin),
                              atol=1e-5, rtol=1e-5)
        # halves swapped-in-place property preserved
        assert qr.shape == q.shape and kr.shape == k.shape

    @requires_cuda
    def test_kernel_vs_fallback(self, monkeypatch):
        B, S, H, D = 2, 32, 4, 64
        rope = RotaryEmbedding(D, 10000, torch.float32,
                               max_context_len=S).cuda()
        q = torch.randn(B, S, H, D, device="cuda")
        k = torch.randn(B, S, H, D, device="cuda")
        with torch.no_grad():
            q_kern, k_kern = rope(q.clone(), k.clone())
        monkeypatch.setattr(mi, "TritonRoPEFunction", None)
        with torch.no_grad():
            q_fb, k_fb = rope(q.clone(), k.clone())
        torch.testing.assert_close(q_kern, q_fb, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_kern, k_fb, atol=1e-5, rtol=1e-5)


# ── Gemma SwiGLU consistency ────────────────────────────────────────

class TestGemmaSwigluConsistency:
    def test_mlpblock_matches_reference_math(self):
        cfg = _make_config()
        mlp = MLPBlock(cfg).to(DEVICE).eval()
        x = torch.randn(4, 16, cfg.hidden_dim, device=DEVICE)
        with torch.no_grad():
            out = mlp(x)
            h = mlp.w1(x)
            act = swiglu(h)
            ref = mlp.w2(act * mlp.w3(x))
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    @requires_cuda
    def test_kernel_vs_fallback(self, monkeypatch):
        cfg = _make_config()
        mlp = MLPBlock(cfg).cuda().eval()
        x = torch.randn(4, 16, cfg.hidden_dim, device="cuda")
        with torch.no_grad():
            out_kernel = mlp(x.clone())
        monkeypatch.setattr(mi, "TritonGemmaSwigluFunction", None)
        with torch.no_grad():
            out_fallback = mlp(x.clone())
        torch.testing.assert_close(out_kernel, out_fallback,
                                   atol=1e-5, rtol=1e-5)


# ── Block residual fusion consistency ───────────────────────────────

class TestBlockFusionConsistency:
    def _build(self):
        cfg = _make_config(num_hidden_layers=1)
        torch.manual_seed(0)
        blk = mi.TransformerDecoderBLK(cfg, device=DEVICE).eval()
        x = torch.randn(2, 16, cfg.hidden_dim, device=DEVICE)
        return blk, x

    def test_output_shape_and_finiteness(self):
        blk, x = self._build()
        out = blk(x.clone())
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @requires_cuda
    def test_fused_add_norm_vs_manual(self, monkeypatch):
        blk, x = self._build()
        with torch.no_grad():
            out_kernel = blk(x.clone())
        _disable_kernels(monkeypatch)
        with torch.no_grad():
            out_fallback = blk(x.clone())
        torch.testing.assert_close(out_kernel, out_fallback,
                                   atol=1e-5, rtol=1e-5)

    def test_gradient_flows_through_block(self):
        blk, x = self._build()
        x = x.clone().requires_grad_(True)
        out = blk(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ── MoE behavior (architecture unchanged across versions) ───────────

class TestMoEBehavior:
    def _make_moe(self, num_experts=4, topk=2):
        cfg = _make_config(num_experts=num_experts,
                           num_experts_per_tok=topk)
        torch.manual_seed(0)
        return MoE(cfg, device=DEVICE).eval(), cfg

    def test_output_shape_preserved(self):
        moe, cfg = self._make_moe()
        x = torch.randn(2, 8, cfg.hidden_dim, device=DEVICE)
        assert moe(x).shape == x.shape

    def test_shared_expert_always_contributes(self):
        moe, cfg = self._make_moe()
        x = torch.randn(2, 8, cfg.hidden_dim, device=DEVICE)
        with torch.no_grad():
            full = moe(x)
            routed_only = torch.zeros_like(full)
            inp = x.view(-1, cfg.hidden_dim)
            w, idx, counts = moe.gate(inp)
            flat_idx = idx.reshape(-1)
            order = flat_idx.argsort(stable=True)
            tok = torch.arange(inp.shape[0], device=DEVICE)\
                .unsqueeze(1).expand_as(idx).reshape(-1)[order]
            sx = inp[tok]
            # NOTE: dispatch operates over N*k assignment rows, not T rows.
            sorted_out = torch.zeros_like(sx)
            b = torch.searchsorted(flat_idx[order].contiguous(),
                                   torch.arange(moe.num_experts + 1,
                                                device=DEVICE))
            bounds = b.tolist()
            for i, e in enumerate(moe.experts):
                s, en = bounds[i], bounds[i + 1]
                if s < en:
                    sorted_out[s:en] = e(sx[s:en])
            routed_only = (
                torch.zeros_like(inp).scatter_add_(
                    0, tok.unsqueeze(1).expand_as(sorted_out),
                    sorted_out * w.view(-1, 1)[order])
                .view_as(x)
            )
            diff = (full - (routed_only + moe.shared_experts(x))).abs().max()
            assert diff.item() < 1e-4

    def test_expert_counts_updated_in_training(self):
        moe, cfg = self._make_moe()
        moe.train()
        before = moe.expert_counts.clone()
        x = torch.randn(2, 8, cfg.hidden_dim, device=DEVICE)
        moe(x)
        assert moe.expert_counts.sum() == 2 * 8 * cfg.num_experts_per_tok
        assert not torch.equal(before, moe.expert_counts)

    @pytest.mark.parametrize("num_experts,topk", [(2, 1), (4, 2), (8, 2)])
    def test_varied_expert_configs(self, num_experts, topk):
        moe, cfg = self._make_moe(num_experts, topk)
        x = torch.randn(2, 8, cfg.hidden_dim, device=DEVICE)
        out = moe(x)
        assert out.shape == x.shape and torch.isfinite(out).all()


# ── Whole-model kernel toggle parity ────────────────────────────────

class TestModelKernelToggleParity:
    def _build(self):
        cfg = _make_config(num_hidden_layers=2, max_context_len=64)
        torch.manual_seed(0)
        model = GPT_FLASH(cfg, device=DEVICE).eval()
        x = torch.randint(0, cfg.vocab_size, (2, 32), device=DEVICE)
        return model, x

    def test_cpu_bit_exact_with_and_without_kernels(self, monkeypatch):
        model, x = self._build()
        with torch.no_grad():
            out_default = model(x.clone())       # CPU: everything eager anyway
        _disable_kernels(monkeypatch)
        with torch.no_grad():
            out_no_kernel = model(x.clone())
        assert torch.equal(out_default, out_no_kernel)

    @requires_cuda
    def test_gpu_kernel_vs_fallback(self, monkeypatch):
        model, x = self._build()
        model = model.cuda(); x = x.cuda()
        with torch.no_grad():
            out_kernel = model(x.clone())
        _disable_kernels(monkeypatch)
        with torch.no_grad():
            out_fallback = model(x.clone())
        torch.testing.assert_close(out_kernel, out_fallback,
                                   atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
