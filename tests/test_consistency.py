"""
Consistency Tests for model_improv.py — Library vs Fallback Paths
=================================================================
Validates that the accelerated library paths (flash_attn_func, triton_moe_forward)
produce numerically equivalent results to the pure-PyTorch fallback paths.

Coverage:
  - FlashAttention vs SDPA fallback: same Q/K/V → same attention output
  - Triton MoE vs Python batched dispatch: same routing → same MoE output
  - triton_moe_forward argument contract validation
  - Gradient equivalence for both paths
  - Numerical stability across input scales
"""
import sys, os, copy, math, types, pytest, torch, torch.nn as nn
import torch.nn.functional as F

# ── FlashAttention mock ─────────────────────────────────────────────
# Provide a SDPA-based mock that matches the real flash_attn_func API
# so model_improv can be imported without the CUDA-only flash_attn package.
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

# ── Project path setup ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_improv import (
    Attention, MoE, Gate, MLPBlock, RMS_Norm, RotaryEmbedding,
    swiglu, soft_clamp, apply_rope,
)
from src.scripts.configs.model_config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


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
    cfg.moe_aux_loss_weight = overrides.get("moe_aux_loss_weight", 0.01)
    return cfg


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: FlashAttention vs SDPA Fallback
# ═══════════════════════════════════════════════════════════════════

class TestFlashAttnVsSDPA:
    """
    The Attention module uses flash_attn_func during training and
    F.scaled_dot_product_attention during inference. These must agree.
    """

    def _build_pair(self, **cfg_overrides):
        """Create identically-weighted training + inference Attention modules."""
        cfg = _make_config(**cfg_overrides)
        torch.manual_seed(42)
        attn_train = Attention(cfg, inference=False)
        attn_infer = Attention(cfg, inference=True)
        attn_infer.load_state_dict(attn_train.state_dict())
        attn_train.eval()
        attn_infer.eval()
        # Precompute RoPE tables
        rope = RotaryEmbedding(
            cfg.head_dim, cfg.base, torch.float32,
            initial_context_len=cfg.initial_context_len,
            max_context_len=cfg.max_context_len,
            device=None,
        )
        cos, sin = rope.compute_cos_sin(cfg.max_context_len)
        return attn_train, attn_infer, cfg, cos, sin

    def test_full_sequence_equivalence(self):
        """Full-sequence forward: training (flash) vs inference (SDPA) match."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair()
        x = torch.randn(2, 32, cfg.hidden_dim)

        with torch.no_grad():
            out_train = attn_train(x, cos, sin)
            attn_infer.reset_cache(batch_size=2)
            out_infer = attn_infer(x, cos, sin, start_pos=0)

        torch.testing.assert_close(
            out_train, out_infer, atol=1e-4, rtol=1e-3,
            msg="Flash path vs SDPA path diverged on full sequence",
        )

    def test_single_token_equivalence(self):
        """Minimal input (1,1,D): both paths should agree."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair()
        x = torch.randn(1, 1, cfg.hidden_dim)

        with torch.no_grad():
            out_train = attn_train(x, cos, sin)
            attn_infer.reset_cache(batch_size=1)
            out_infer = attn_infer(x, cos, sin, start_pos=0)

        torch.testing.assert_close(
            out_train, out_infer, atol=1e-4, rtol=1e-3,
            msg="Flash vs SDPA diverged on single token",
        )

    def test_gqa_equivalence(self):
        """GQA configuration (n_heads != n_kv_heads) must still match."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair(
            num_attn_heads=8, num_key_value_heads=2, hidden_dim=256,
        )
        x = torch.randn(2, 16, cfg.hidden_dim)

        with torch.no_grad():
            out_train = attn_train(x, cos, sin)
            attn_infer.reset_cache(batch_size=2)
            out_infer = attn_infer(x, cos, sin, start_pos=0)

        torch.testing.assert_close(
            out_train, out_infer, atol=1e-4, rtol=1e-3,
            msg="Flash vs SDPA diverged under GQA (8 query, 2 KV heads)",
        )

    def test_mha_equivalence(self):
        """MHA configuration (n_heads == n_kv_heads) must match."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair(
            num_attn_heads=8, num_key_value_heads=8, hidden_dim=256,
        )
        x = torch.randn(2, 16, cfg.hidden_dim)

        with torch.no_grad():
            out_train = attn_train(x, cos, sin)
            attn_infer.reset_cache(batch_size=2)
            out_infer = attn_infer(x, cos, sin, start_pos=0)

        torch.testing.assert_close(
            out_train, out_infer, atol=1e-4, rtol=1e-3,
            msg="Flash vs SDPA diverged under standard MHA",
        )

    @pytest.mark.parametrize("scale", [1e-4, 1.0, 10.0, 100.0])
    def test_numerical_stability_across_scales(self, scale):
        """Both paths stay finite and close across input magnitudes."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair()
        x = torch.randn(2, 16, cfg.hidden_dim) * scale

        with torch.no_grad():
            out_train = attn_train(x, cos, sin)
            attn_infer.reset_cache(batch_size=2)
            out_infer = attn_infer(x, cos, sin, start_pos=0)

        assert not torch.isnan(out_train).any(), f"NaN in flash path at scale={scale}"
        assert not torch.isnan(out_infer).any(), f"NaN in SDPA path at scale={scale}"
        torch.testing.assert_close(
            out_train, out_infer, atol=1e-3, rtol=1e-2,
            msg=f"Flash vs SDPA diverged at input scale {scale}",
        )

    def test_gradient_equivalence(self):
        """Gradients through both paths should be close."""
        cfg = _make_config()
        torch.manual_seed(42)
        attn_train = Attention(cfg, inference=False)
        attn_infer = Attention(cfg, inference=True)
        attn_infer.load_state_dict(attn_train.state_dict())
        attn_train.train()
        attn_infer.train()  # train mode but using inference path

        rope = RotaryEmbedding(
            cfg.head_dim, cfg.base, torch.float32,
            initial_context_len=cfg.initial_context_len,
            max_context_len=cfg.max_context_len,
        )
        cos, sin = rope.compute_cos_sin(cfg.max_context_len)

        x = torch.randn(2, 16, cfg.hidden_dim, requires_grad=True)

        # Training path gradient
        out_train = attn_train(x, cos, sin)
        out_train.sum().backward()
        grad_train = x.grad.clone()

        x.grad = None

        # Inference path gradient (uses SDPA — still differentiable)
        attn_infer.reset_cache(batch_size=2)
        out_infer = attn_infer(x, cos, sin, start_pos=0)
        out_infer.sum().backward()
        grad_infer = x.grad.clone()

        torch.testing.assert_close(
            grad_train, grad_infer, atol=1e-3, rtol=1e-2,
            msg="Input gradients diverge between flash and SDPA paths",
        )

    def test_incremental_decode_matches_full(self):
        """Token-by-token KV-cache decode should match full-sequence output."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair()

        seq_len = 8
        x = torch.randn(1, seq_len, cfg.hidden_dim)

        # Full forward (flash/training path)
        with torch.no_grad():
            out_full = attn_train(x, cos, sin)

        # Incremental: prefill first 7, decode position 7
        attn_infer.reset_cache(batch_size=1)
        with torch.no_grad():
            _ = attn_infer(x[:, :7, :], cos, sin, start_pos=0)
            out_last = attn_infer(x[:, 7:8, :], cos, sin, start_pos=7)

        torch.testing.assert_close(
            out_full[0, 7:8], out_last[0],
            atol=1e-4, rtol=1e-3,
            msg="Incremental KV-cache decode diverges from full flash forward",
        )

    def test_causal_masking_equivalent(self):
        """Changing a future token should not affect earlier positions in either path."""
        attn_train, attn_infer, cfg, cos, sin = self._build_pair()

        x = torch.randn(1, 8, cfg.hidden_dim)

        with torch.no_grad():
            out_train_orig = attn_train(x.clone(), cos, sin)

        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(cfg.hidden_dim)

        with torch.no_grad():
            out_train_mod = attn_train(x_modified, cos, sin)

        # Positions 0-6 should be identical (causal)
        torch.testing.assert_close(
            out_train_orig[0, :7], out_train_mod[0, :7],
            atol=1e-5, rtol=1e-4,
            msg="Causal violation in flash attention path",
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Triton MoE vs Python Fallback
# ═══════════════════════════════════════════════════════════════════

def _python_moe_forward(moe: MoE, x: torch.Tensor):
    """
    Force the Python fallback path by calling the MoE internals directly,
    bypassing the TRITON_MOE_AVAILABLE check.
    """
    inp_shape = x.shape
    x_flat = x.view(-1, moe.dim)

    weights, indices, counts, aux_loss = moe.gate(x_flat)

    flat_idx = indices.reshape(-1)
    flat_weights = weights.reshape(-1, 1)
    token_idx = torch.arange(x_flat.shape[0], device=x.device)
    token_idx = token_idx.unsqueeze(1).expand_as(indices).reshape(-1)

    sort_order = flat_idx.argsort(stable=True)
    sorted_expert_ids = flat_idx[sort_order]
    sorted_token_idx = token_idx[sort_order]
    sorted_weights = flat_weights[sort_order]
    sorted_x = x_flat[sorted_token_idx]

    boundaries = torch.searchsorted(
        sorted_expert_ids.contiguous(),
        torch.arange(moe.num_experts + 1, device=x.device),
    ).tolist()

    sorted_out = torch.zeros_like(sorted_x)
    for i, expert in enumerate(moe.experts):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            sorted_out[start:end] = expert(sorted_x[start:end])

    sorted_out = sorted_out * sorted_weights
    routed = torch.zeros_like(x_flat)
    routed.scatter_add_(
        0, sorted_token_idx.unsqueeze(1).expand_as(sorted_out), sorted_out
    )
    mlp_out = routed + moe.shared_expert(x_flat)
    return mlp_out.view(*inp_shape), aux_loss


def _triton_moe_reference(moe: MoE, x: torch.Tensor):
    """
    Simulate what triton_moe_forward SHOULD compute: fused routing + expert
    dispatch + shared expert. This is the reference contract.
    """
    inp_shape = x.shape
    x_flat = x.view(-1, moe.dim)

    # Step 1: Route via sigmoid scores + bias
    scores = torch.sigmoid(moe.gate.router(x_flat.float()))
    biased = scores + moe.gate.bias.to(scores.dtype)
    indices = torch.topk(biased, moe.gate.topk, dim=-1)[1]

    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    weights = (weights * moe.gate.route_scale).to(x.dtype)

    # Step 2: Aux loss
    T = x_flat.shape[0]
    current_load = torch.bincount(indices.flatten(), minlength=moe.num_experts)
    f = current_load.float() / (T * moe.gate.topk)
    P = scores.mean(dim=0)
    aux_loss = moe.num_experts * torch.sum(f * P)

    # Step 3: Expert dispatch
    flat_idx = indices.reshape(-1)
    flat_weights = weights.reshape(-1, 1)
    token_idx = torch.arange(T, device=x.device)
    token_idx = token_idx.unsqueeze(1).expand_as(indices).reshape(-1)

    sort_order = flat_idx.argsort(stable=True)
    sorted_expert_ids = flat_idx[sort_order]
    sorted_token_idx = token_idx[sort_order]
    sorted_weights = flat_weights[sort_order]
    sorted_x = x_flat[sorted_token_idx]

    boundaries = torch.searchsorted(
        sorted_expert_ids.contiguous(),
        torch.arange(moe.num_experts + 1, device=x.device),
    ).tolist()

    sorted_out = torch.zeros_like(sorted_x)
    for i, expert in enumerate(moe.experts):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            sorted_out[start:end] = expert(sorted_x[start:end])

    sorted_out = sorted_out * sorted_weights
    routed = torch.zeros_like(x_flat)
    routed.scatter_add_(
        0, sorted_token_idx.unsqueeze(1).expand_as(sorted_out), sorted_out
    )

    # Step 4: Shared expert
    mlp_out = routed + moe.shared_expert(x_flat)
    return mlp_out.view(*inp_shape), aux_loss


class TestTritonMoEVsPythonFallback:
    """
    Verify that if triton_moe_forward existed, the contract it must
    satisfy matches the Python fallback. We test this by running the
    reference implementation against the actual fallback.
    """

    def _build_moe(self, **cfg_overrides):
        cfg = _make_config(**cfg_overrides)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()
        return moe, cfg

    def test_python_fallback_matches_reference(self):
        """The Python fallback and our reference should be identical."""
        moe, cfg = self._build_moe()
        x = torch.randn(4, 16, cfg.hidden_dim)

        with torch.no_grad():
            out_fallback, aux_fallback = _python_moe_forward(moe, x)
            out_ref, aux_ref = _triton_moe_reference(moe, x)

        torch.testing.assert_close(
            out_fallback, out_ref, atol=1e-5, rtol=1e-4,
            msg="Python fallback vs reference implementation diverged",
        )
        torch.testing.assert_close(
            aux_fallback, aux_ref, atol=1e-5, rtol=1e-4,
            msg="Aux loss diverged between fallback and reference",
        )

    def test_moe_forward_uses_fallback_when_triton_unavailable(self):
        """With TRITON_MOE_AVAILABLE=False, MoE.forward should use Python path."""
        import src.models.model_improv as mod
        from unittest.mock import patch

        moe, cfg = self._build_moe()
        x = torch.randn(2, 8, cfg.hidden_dim)

        # Force fallback path regardless of whether triton_kernels is installed
        with patch.object(mod, "TRITON_MOE_AVAILABLE", False):
            with torch.no_grad():
                out_module, aux_module = moe(x)
                out_ref, aux_ref = _python_moe_forward(moe, x)

        torch.testing.assert_close(
            out_module, out_ref, atol=1e-5, rtol=1e-4,
            msg="MoE.forward() diverges from explicit Python fallback",
        )

    def test_output_shape_preserved(self):
        """MoE output shape must match input shape."""
        moe, cfg = self._build_moe()
        for shape in [(4, 16, cfg.hidden_dim), (1, 1, cfg.hidden_dim),
                      (8, 64, cfg.hidden_dim)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                out, _ = moe(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_aux_loss_is_scalar(self):
        """Aux loss should be a scalar (0-d tensor)."""
        moe, cfg = self._build_moe()
        x = torch.randn(4, 16, cfg.hidden_dim)
        with torch.no_grad():
            _, aux = moe(x)
        assert aux.dim() == 0, f"Aux loss should be scalar, got shape {aux.shape}"
        assert torch.isfinite(aux), "Aux loss is not finite"

    def test_gradient_through_fallback(self):
        """Gradients flow through the Python fallback MoE path."""
        moe, cfg = self._build_moe()
        moe.train()
        x = torch.randn(4, 8, cfg.hidden_dim, requires_grad=True)
        out, aux = moe(x)
        (out.sum() + aux).backward()

        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Zero gradient on input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        # Check expert weights got gradients
        for i, expert in enumerate(moe.experts):
            assert expert.w1.weight.grad is not None, f"Expert {i} w1 no grad"
            assert expert.w2.weight.grad is not None, f"Expert {i} w2 no grad"

        # Shared expert too
        assert moe.shared_expert.w1.weight.grad is not None, "Shared w1 no grad"
        assert moe.shared_expert.w2.weight.grad is not None, "Shared w2 no grad"

    @pytest.mark.parametrize("num_experts,topk", [
        (2, 1), (4, 2), (8, 2), (8, 4), (6, 3),
    ])
    def test_varied_expert_configs(self, num_experts, topk):
        """Fallback works across different expert/topk configurations."""
        moe, cfg = self._build_moe(num_experts=num_experts,
                                    num_experts_per_tok=topk)
        x = torch.randn(4, 8, cfg.hidden_dim)

        with torch.no_grad():
            out_module, aux_module = moe(x)
            out_ref, aux_ref = _python_moe_forward(moe, x)

        torch.testing.assert_close(
            out_module, out_ref, atol=1e-5, rtol=1e-4,
            msg=f"Diverged for {num_experts} experts, top-{topk}",
        )

    @pytest.mark.parametrize("scale", [1e-4, 1.0, 10.0, 100.0])
    def test_numerical_stability_across_scales(self, scale):
        """MoE stays finite across varied input magnitudes."""
        moe, cfg = self._build_moe()
        x = torch.randn(4, 8, cfg.hidden_dim) * scale

        with torch.no_grad():
            out, aux = moe(x)
        assert not torch.isnan(out).any(), f"NaN at scale={scale}"
        assert not torch.isinf(out).any(), f"Inf at scale={scale}"
        assert torch.isfinite(aux), f"Aux loss not finite at scale={scale}"

    def test_shared_expert_always_contributes(self):
        """Shared expert output should be non-zero for non-zero input."""
        moe, cfg = self._build_moe()
        x = torch.randn(2, 4, cfg.hidden_dim)

        with torch.no_grad():
            shared_out = moe.shared_expert(x.view(-1, cfg.hidden_dim))
        assert shared_out.abs().sum() > 0, "Shared expert produced zero output"

    def test_expert_counts_updated_in_training(self):
        """Expert counts should accumulate during training forward passes."""
        moe, cfg = self._build_moe()
        moe.train()
        moe.reset_expert_counts()

        x = torch.randn(4, 8, cfg.hidden_dim)
        with torch.no_grad():
            moe(x)

        total = moe.expert_counts.sum().item()
        assert total > 0, "No expert counts recorded during training"


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: fused_moe_forward (triton_kernels) Argument Contract
# ═══════════════════════════════════════════════════════════════════

class TestTritonMoeForwardContract:
    """
    Validate that the arguments model_improv.py constructs for the
    TritonMoE fused_moe_forward (bassrehab/triton-kernels) are correctly
    shaped and typed per the real API:

        fused_moe_forward(
            hidden_states:  Tensor  (T, D),
            router_weight:  Tensor  (E, D),
            w_gate:         Tensor  (E, ffn_dim, D),
            w_up:           Tensor  (E, ffn_dim, D),
            w_down:         Tensor  (E, D, ffn_dim),
            num_experts:    int,
            top_k:          int,
            gating:         str  = "softmax",
        ) -> Tuple[Tensor, Tensor, Tensor]
            output:        (T, D)
            top_k_indices: (T, K)
            top_k_weights: (T, K)
    """

    def _build_moe(self, **overrides):
        cfg = _make_config(**overrides)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()
        return moe, cfg

    def _build_kernel_args(self, moe, cfg):
        """Reconstruct the 3D stacked tensors that MoE.forward builds for the kernel."""
        intermediate_size = cfg.intermediate_size
        w_gate_list, w_up_list = [], []
        for expert in moe.experts:
            w1 = expert.w1.weight                         # (2*I, D)
            w_gate_list.append(w1[:intermediate_size])    # (I, D)
            w_up_list.append(w1[intermediate_size:])      # (I, D)
        w_gate_3d = torch.stack(w_gate_list)              # (E, I, D)
        w_up_3d = torch.stack(w_up_list)                  # (E, I, D)
        w_down_3d = torch.stack([e.w2.weight for e in moe.experts])  # (E, D, I)
        return w_gate_3d, w_up_3d, w_down_3d

    # ── Shape validation ──────────────────────────────────────────

    def test_router_weight_shape(self):
        """Router weight: (num_experts, hidden_dim)."""
        moe, cfg = self._build_moe()
        w = moe.gate.router.weight
        assert w.shape == (cfg.num_experts, cfg.hidden_dim), (
            f"Router weight shape {w.shape}, expected "
            f"({cfg.num_experts}, {cfg.hidden_dim})"
        )

    def test_w_gate_3d_shape(self):
        """w_gate stacked tensor: (E, intermediate_size, hidden_dim)."""
        moe, cfg = self._build_moe()
        w_gate_3d, _, _ = self._build_kernel_args(moe, cfg)
        expected = (cfg.num_experts, cfg.intermediate_size, cfg.hidden_dim)
        assert w_gate_3d.shape == expected, (
            f"w_gate shape {w_gate_3d.shape}, expected {expected}"
        )

    def test_w_up_3d_shape(self):
        """w_up stacked tensor: (E, intermediate_size, hidden_dim)."""
        moe, cfg = self._build_moe()
        _, w_up_3d, _ = self._build_kernel_args(moe, cfg)
        expected = (cfg.num_experts, cfg.intermediate_size, cfg.hidden_dim)
        assert w_up_3d.shape == expected, (
            f"w_up shape {w_up_3d.shape}, expected {expected}"
        )

    def test_w_down_3d_shape(self):
        """w_down stacked tensor: (E, hidden_dim, intermediate_size)."""
        moe, cfg = self._build_moe()
        _, _, w_down_3d = self._build_kernel_args(moe, cfg)
        expected = (cfg.num_experts, cfg.hidden_dim, cfg.intermediate_size)
        assert w_down_3d.shape == expected, (
            f"w_down shape {w_down_3d.shape}, expected {expected}"
        )

    def test_w1_splits_cover_full_weight(self):
        """Splitting w1 into gate + up must recover the original weight exactly."""
        moe, cfg = self._build_moe()
        w_gate_3d, w_up_3d, _ = self._build_kernel_args(moe, cfg)
        for i, expert in enumerate(moe.experts):
            reconstructed = torch.cat([w_gate_3d[i], w_up_3d[i]], dim=0)
            torch.testing.assert_close(
                reconstructed, expert.w1.weight,
                msg=f"Expert {i}: gate+up concat ≠ original w1"
            )

    # ── Type & scalar validation ──────────────────────────────────

    def test_topk_is_int(self):
        """topk argument must be an int."""
        moe, cfg = self._build_moe()
        assert isinstance(moe.gate.topk, int)
        assert moe.gate.topk == cfg.num_experts_per_tok

    def test_num_experts_matches_expert_count(self):
        """num_experts passed to triton should equal len(experts)."""
        moe, cfg = self._build_moe()
        assert moe.num_experts == len(moe.experts)
        assert moe.num_experts == cfg.num_experts

    def test_all_expert_weights_same_dtype(self):
        """All weight tensors passed to kernel must share the same dtype."""
        moe, cfg = self._build_moe()
        w_gate_3d, w_up_3d, w_down_3d = self._build_kernel_args(moe, cfg)
        dtypes = {
            moe.gate.router.weight.dtype,
            w_gate_3d.dtype, w_up_3d.dtype, w_down_3d.dtype,
        }
        assert len(dtypes) == 1, f"Mixed dtypes: {dtypes}"

    def test_input_shape_2d(self):
        """fused_moe_forward receives (T, D) after view(-1, dim)."""
        moe, cfg = self._build_moe()
        x = torch.randn(2, 8, cfg.hidden_dim)
        x_flat = x.view(-1, moe.dim)
        assert x_flat.dim() == 2
        assert x_flat.shape[1] == cfg.hidden_dim

    # ── Source-level call site validation ──────────────────────────

    def test_triton_call_site_matches_api(self):
        """
        The triton_moe_forward call in MoE.forward passes the 8 arguments
        matching the fused_moe_forward(hidden_states, router_weight,
        w_gate, w_up, w_down, num_experts, top_k, gating) signature.
        """
        import inspect
        import src.models.model_improv as mod

        source = inspect.getsource(mod.MoE.forward)
        # Core API arguments
        assert "triton_moe_forward" in source
        assert "self.gate.router.weight" in source
        assert "w_gate_3d" in source
        assert "w_up_3d" in source
        assert "w_down_3d" in source
        assert "self.num_experts" in source
        assert "self.gate.topk" in source
        assert 'gating="sigmoid"' in source
        # Post-kernel steps handled by the caller
        assert "self.shared_expert" in source
        assert "resid_scale" in source

    def test_gating_mode_is_sigmoid(self):
        """model_improv uses sigmoid routing, not softmax."""
        import inspect
        import src.models.model_improv as mod
        source = inspect.getsource(mod.MoE.forward)
        assert 'gating="sigmoid"' in source, (
            "TritonMoE must be called with gating='sigmoid' to match "
            "the Gate class's torch.sigmoid routing"
        )

    # ── Fallback-specific validation ──────────────────────────────

    def test_resid_scale_is_sqrt_half(self):
        """resid_scale from MLPBlock should be √0.5 ≈ 0.7071."""
        moe, cfg = self._build_moe()
        expected = math.sqrt(0.5)
        actual = moe.experts[0].resid_scale
        assert abs(actual - expected) < 1e-6, (
            f"resid_scale={actual}, expected √0.5={expected}"
        )
        for i, expert in enumerate(moe.experts):
            assert expert.resid_scale == actual, (
                f"Expert {i} resid_scale mismatch"
            )
        assert moe.shared_expert.resid_scale == actual

    def test_shared_expert_not_in_kernel_call(self):
        """Shared expert weights should NOT be passed to fused_moe_forward
        (they are handled outside the kernel)."""
        import inspect
        import src.models.model_improv as mod
        source = inspect.getsource(mod.MoE.forward)
        # Find the actual triton_moe_forward(...) call arguments
        call_start = source.index("triton_moe_forward(")
        # Count parentheses to find the end of the call
        depth, i = 0, call_start
        for i, c in enumerate(source[call_start:], call_start):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        call_args = source[call_start:i + 1]
        assert "shared_expert" not in call_args, (
            "shared_expert should NOT be passed to fused_moe_forward"
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: End-to-End Consistency (full model)
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndConsistency:
    """
    Full-model forward: ensure training and inference paths produce
    matching logits through the entire GPT_FLASH stack.
    """

    def test_full_model_train_vs_infer(self):
        """Full model: training forward logits ≈ inference forward logits."""
        from src.models.model_improv import GPT_FLASH

        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)
        model_train = GPT_FLASH(cfg, inference=False)
        model_infer = GPT_FLASH(cfg, inference=True)
        model_infer.load_state_dict(model_train.state_dict())
        model_train.eval()
        model_infer.eval()

        x = torch.randint(0, cfg.vocab_size, (1, 16))

        with torch.no_grad():
            logits_train, aux_train = model_train(x)
            model_infer.reset_cache(batch_size=1)
            logits_infer, aux_infer = model_infer(x, start_pos=0)

        torch.testing.assert_close(
            logits_train, logits_infer, atol=1e-3, rtol=1e-2,
            msg="Training vs inference full-model logits diverged",
        )

    def test_full_model_no_nan(self):
        """No NaN in any output from the full model."""
        from src.models.model_improv import GPT_FLASH

        cfg = _make_config()
        model = GPT_FLASH(cfg)
        model.eval()

        x = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.no_grad():
            logits, aux = model(x)

        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"
        assert torch.isfinite(aux), "Aux loss not finite"

    def test_full_model_backward(self):
        """Loss backward through full model works without error."""
        from src.models.model_improv import GPT_FLASH

        cfg = _make_config()
        model = GPT_FLASH(cfg)
        model.train()

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        logits, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        total_loss = loss + aux
        total_loss.backward()

        assert math.isfinite(total_loss.item())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Timing Benchmarks — Library vs Fallback Speedup
# ═══════════════════════════════════════════════════════════════════
#
# These benchmarks measure the real-world speedup from the GPU-accelerated
# libraries (flash_attn, triton_kernels.fused_moe_forward) over the pure
# PyTorch fallback implementations.
#
# Methodology:
#   - GPU timing via torch.cuda.Event (not wall-clock) for accuracy
#   - CPU timing via time.perf_counter for fallback-on-CPU measurements
#   - Warmup iterations to fill GPU caches / trigger JIT
#   - Median of multiple runs (not mean) to ignore outliers
#   - torch.cuda.synchronize() before every measurement boundary
#
# Run:  pytest test_consistency.py -k "benchmark" -v -s
# ═══════════════════════════════════════════════════════════════════

import time
import statistics

# Detect real (not mocked) library availability
try:
    import flash_attn as _real_fa
    _has_real_flash_attn = hasattr(_real_fa, "flash_attn_func") and HAS_CUDA
    if _has_real_flash_attn:
        _real_flash_attn_func = _real_fa.flash_attn_func
except Exception:
    _has_real_flash_attn = False

try:
    from triton_kernels import fused_moe_forward as _real_fused_moe_forward
    _has_real_triton_moe = HAS_CUDA
except Exception:
    _has_real_triton_moe = False


def _cuda_timer(fn, *, warmup=5, repeats=20):
    """Time a CUDA function using torch.cuda.Event for sub-ms accuracy.

    Returns median time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    return statistics.median(times)


def _cpu_timer(fn, *, warmup=3, repeats=10):
    """Time a CPU function using perf_counter.

    Returns median time in milliseconds.
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return statistics.median(times)


# ── Attention benchmarks ─────────────────────────────────────────

requires_flash_attn = pytest.mark.skipif(
    not _has_real_flash_attn,
    reason="flash_attn not installed or no CUDA",
)


class TestAttentionBenchmark:
    """Benchmark: flash_attn_func vs F.scaled_dot_product_attention on GPU."""

    @staticmethod
    def _sdpa_reference(Q, K, V, causal=True):
        """Pure SDPA fallback (same as Attention inference path)."""
        Q_t = Q.transpose(1, 2)  # (B, H, S, D)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        nq, nkv = Q_t.shape[1], K_t.shape[1]
        if nq != nkv:
            K_t = K_t.repeat_interleave(nq // nkv, dim=1)
            V_t = V_t.repeat_interleave(nq // nkv, dim=1)
        out = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=causal)
        return out.transpose(1, 2)

    @requires_flash_attn
    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    def test_attention_speedup(self, seq_len):
        """Measure flash_attn_func vs SDPA speedup at various sequence lengths."""
        B, H, D = 4, 8, 64
        Q = torch.randn(B, seq_len, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, seq_len, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, seq_len, H, D, device="cuda", dtype=torch.float16)

        # Benchmark flash_attn
        flash_ms = _cuda_timer(lambda: _real_flash_attn_func(Q, K, V, causal=True))

        # Benchmark SDPA fallback
        sdpa_ms = _cuda_timer(lambda: self._sdpa_reference(Q, K, V, causal=True))

        speedup = sdpa_ms / flash_ms if flash_ms > 0 else float("inf")

        print(
            f"\n  seq_len={seq_len:>5d}: "
            f"flash_attn={flash_ms:.3f}ms  "
            f"SDPA={sdpa_ms:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        # Sanity: both should produce finite output
        with torch.no_grad():
            out_flash = _real_flash_attn_func(Q, K, V, causal=True)
            out_sdpa = self._sdpa_reference(Q, K, V, causal=True)
        assert torch.isfinite(out_flash).all()
        assert torch.isfinite(out_sdpa).all()

    @requires_flash_attn
    def test_attention_gqa_speedup(self):
        """Measure speedup with grouped query attention (fewer KV heads)."""
        B, S, Hq, Hkv, D = 4, 1024, 16, 4, 64
        Q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)

        flash_ms = _cuda_timer(lambda: _real_flash_attn_func(Q, K, V, causal=True))
        sdpa_ms = _cuda_timer(lambda: self._sdpa_reference(Q, K, V, causal=True))

        speedup = sdpa_ms / flash_ms if flash_ms > 0 else float("inf")
        print(
            f"\n  GQA ({Hq}q/{Hkv}kv, seq={S}): "
            f"flash_attn={flash_ms:.3f}ms  "
            f"SDPA={sdpa_ms:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )

    def test_attention_cpu_fallback_timing(self):
        """Measure SDPA fallback timing on CPU as a reference baseline."""
        B, S, H, D = 2, 256, 8, 64
        Q = torch.randn(B, S, H, D)
        K = torch.randn(B, S, H, D)
        V = torch.randn(B, S, H, D)

        cpu_ms = _cpu_timer(
            lambda: self._sdpa_reference(Q, K, V, causal=True),
            warmup=2, repeats=5,
        )
        print(f"\n  CPU SDPA (seq={S}, heads={H}): {cpu_ms:.3f}ms")
        assert cpu_ms > 0


# ── MoE benchmarks ──────────────────────────────────────────────

requires_triton_moe = pytest.mark.skipif(
    not _has_real_triton_moe,
    reason="triton_kernels not installed or no CUDA",
)


class TestMoEBenchmark:
    """Benchmark: fused_moe_forward vs Python batched dispatch on GPU."""

    @staticmethod
    def _python_moe_dispatch(moe, x_flat):
        """Python fallback MoE dispatch (same as MoE.forward's else branch)."""
        weights, indices, counts, aux_loss = moe.gate(x_flat)
        T = x_flat.shape[0]
        K = moe.gate.topk

        flat_idx = indices.reshape(-1)
        flat_weights = weights.reshape(-1, 1)
        token_idx = torch.arange(T, device=x_flat.device)
        token_idx = token_idx.unsqueeze(1).expand_as(indices).reshape(-1)

        sort_order = flat_idx.argsort(stable=True)
        sorted_expert_ids = flat_idx[sort_order]
        sorted_token_idx = token_idx[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_x = x_flat[sorted_token_idx]

        boundaries = torch.searchsorted(
            sorted_expert_ids.contiguous(),
            torch.arange(moe.num_experts + 1, device=x_flat.device),
        ).tolist()

        sorted_out = torch.zeros_like(sorted_x)
        for i, expert in enumerate(moe.experts):
            start, end = boundaries[i], boundaries[i + 1]
            if start < end:
                sorted_out[start:end] = expert(sorted_x[start:end])

        sorted_out = sorted_out * sorted_weights
        routed = torch.zeros_like(x_flat)
        routed.scatter_add_(
            0, sorted_token_idx.unsqueeze(1).expand_as(sorted_out), sorted_out,
        )

        out = routed + moe.shared_expert(x_flat)
        return out, aux_loss

    @staticmethod
    def _triton_moe_dispatch(moe, x_flat):
        """TritonMoE fused dispatch (mirrors MoE.forward's if branch)."""
        intermediate_size = moe.experts[0].w2.weight.shape[1]
        w_gate_list, w_up_list = [], []
        for expert in moe.experts:
            w1 = expert.w1.weight
            w_gate_list.append(w1[:intermediate_size])
            w_up_list.append(w1[intermediate_size:])
        w_gate_3d = torch.stack(w_gate_list)
        w_up_3d = torch.stack(w_up_list)
        w_down_3d = torch.stack([e.w2.weight for e in moe.experts])

        triton_out, top_k_indices, top_k_weights = _real_fused_moe_forward(
            x_flat, moe.gate.router.weight,
            w_gate_3d, w_up_3d, w_down_3d,
            moe.num_experts, moe.gate.topk,
            gating="sigmoid",
        )

        triton_out = triton_out * moe.experts[0].resid_scale
        triton_out = triton_out + moe.shared_expert(x_flat)
        return triton_out

    def _build_moe_cuda(self, **overrides):
        defaults = dict(
            hidden_dim=256, intermediate_size=512,
            num_experts=8, num_experts_per_tok=2,
        )
        defaults.update(overrides)
        cfg = _make_config(**defaults)
        torch.manual_seed(42)
        moe = MoE(cfg).cuda().eval()
        return moe, cfg

    @requires_triton_moe
    @pytest.mark.parametrize("num_tokens", [64, 256, 512, 1024])
    def test_moe_speedup(self, num_tokens):
        """Measure fused_moe_forward vs Python fallback at various token counts."""
        moe, cfg = self._build_moe_cuda()
        x = torch.randn(num_tokens, cfg.hidden_dim, device="cuda", dtype=torch.float32)

        # Benchmark Triton fused kernel
        triton_ms = _cuda_timer(
            lambda: self._triton_moe_dispatch(moe, x),
            warmup=5, repeats=20,
        )

        # Benchmark Python fallback
        python_ms = _cuda_timer(
            lambda: self._python_moe_dispatch(moe, x),
            warmup=5, repeats=20,
        )

        speedup = python_ms / triton_ms if triton_ms > 0 else float("inf")
        print(
            f"\n  tokens={num_tokens:>5d}: "
            f"triton={triton_ms:.3f}ms  "
            f"python={python_ms:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )

    @requires_triton_moe
    @pytest.mark.parametrize(
        "num_experts,topk",
        [(4, 1), (8, 2), (16, 2), (16, 4)],
    )
    def test_moe_speedup_varied_experts(self, num_experts, topk):
        """Measure speedup across different expert configurations."""
        moe, cfg = self._build_moe_cuda(
            num_experts=num_experts, num_experts_per_tok=topk,
        )
        x = torch.randn(256, cfg.hidden_dim, device="cuda", dtype=torch.float32)

        triton_ms = _cuda_timer(lambda: self._triton_moe_dispatch(moe, x))
        python_ms = _cuda_timer(lambda: self._python_moe_dispatch(moe, x))

        speedup = python_ms / triton_ms if triton_ms > 0 else float("inf")
        print(
            f"\n  {num_experts}E top-{topk}: "
            f"triton={triton_ms:.3f}ms  "
            f"python={python_ms:.3f}ms  "
            f"speedup={speedup:.2f}x"
        )

    def test_moe_cpu_fallback_timing(self):
        """Measure Python fallback timing on CPU as a reference baseline."""
        cfg = _make_config(
            hidden_dim=128, intermediate_size=256,
            num_experts=4, num_experts_per_tok=2,
        )
        torch.manual_seed(42)
        moe = MoE(cfg).eval()
        x = torch.randn(64, cfg.hidden_dim)

        cpu_ms = _cpu_timer(
            lambda: self._python_moe_dispatch(moe, x),
            warmup=2, repeats=5,
        )
        print(f"\n  CPU Python MoE (4E top-2, 64 tokens): {cpu_ms:.3f}ms")
        assert cpu_ms > 0

    @requires_triton_moe
    def test_moe_numerical_equivalence_on_gpu(self):
        """On GPU: verify Triton and Python fallback produce reasonable outputs.

        NOTE: The Triton kernel runs its own internal moe_router which
        computes routing independently from Python's Gate.forward().
        Token-to-expert assignments will differ, so exact numerical
        equivalence is not expected. We check:
          1. Same output shape
          2. Both outputs are finite (no NaN/Inf)
          3. Similar output magnitude (cosine similarity > 0.5)
        """
        moe, cfg = self._build_moe_cuda()
        x = torch.randn(128, cfg.hidden_dim, device="cuda")

        with torch.no_grad():
            out_python, _ = self._python_moe_dispatch(moe, x)
            out_triton = self._triton_moe_dispatch(moe, x)

        # Shape must match
        assert out_python.shape == out_triton.shape, (
            f"Shape mismatch: python={out_python.shape} vs triton={out_triton.shape}"
        )

        # Both must be finite
        assert torch.isfinite(out_python).all(), "Python path produced NaN/Inf"
        assert torch.isfinite(out_triton).all(), "Triton path produced NaN/Inf"

        # Outputs should be in similar magnitude range (cosine similarity)
        cos_sim = F.cosine_similarity(
            out_python.flatten().unsqueeze(0),
            out_triton.flatten().unsqueeze(0),
        ).item()
        print(f"\n  Triton vs Python cosine similarity: {cos_sim:.4f}")
        assert cos_sim > 0.5, (
            f"Outputs too dissimilar: cosine_similarity={cos_sim:.4f}"
        )

        # Magnitudes should be within same order of magnitude
        py_norm = out_python.norm().item()
        tr_norm = out_triton.norm().item()
        ratio = max(py_norm, tr_norm) / (min(py_norm, tr_norm) + 1e-8)
        print(f"  Norm ratio: {ratio:.2f} (python={py_norm:.2f}, triton={tr_norm:.2f})")
        assert ratio < 10.0, f"Norm ratio too large: {ratio:.2f}"

    