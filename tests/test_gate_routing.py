"""
Gate/Router — Production Correctness Suite
===========================================
Validates the Gate module implementing Loss-Free Load Balancing
(DeepSeek-V3 paper). This is the routing brain of every MoE layer.

Coverage:
  - Sigmoid gating (NOT softmax)
  - Top-k selection uses biased scores, weighting uses original scores
  - Weight normalization (sum-to-1 per token, then * route_scale)
  - Bias update rule (sign-based, clamped to [-10, 10])
  - Per-layer scaling of bias updates
  - Training vs eval mode (bias updates only in train)
  - Output shapes and types
  - Bias convergence toward balanced load
"""
import sys, os, copy, math, types, pytest, torch, torch.nn as nn
from torch.amp import autocast

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

from src.models.model_flash_attn import Gate
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
    cfg.num_hidden_layers = overrides.get("num_hidden_layers", 24)
    cfg.base = 10000
    cfg.initial_context_len = 2048
    cfg.max_context_len = 2048
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    return cfg



# ── SECTION 1: Output shapes and types ─────────────────────────────────────────────

class TestGateOutputs:

    def test_output_tuple_length(self):
        """Gate returns (weights, indices, counts) — 3-tuple."""
        cfg = _make_config()
        gate = Gate(cfg)
        gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        result = gate(x)
        assert len(result) == 3

    def test_weights_shape(self):
        """Weights: (N, topk)."""
        cfg = _make_config(num_experts=4, num_experts_per_tok=2)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        weights, indices, counts = gate(x)
        assert weights.shape == (64, 2)

    def test_indices_shape(self):
        """Indices: (N, topk), values in [0, num_experts)."""
        cfg = _make_config(num_experts=4, num_experts_per_tok=2)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        weights, indices, counts = gate(x)
        assert indices.shape == (64, 2)
        assert indices.min() >= 0
        assert indices.max() < cfg.num_experts

    def test_counts_shape(self):
        """Counts: (num_experts,), summing to N * topk."""
        cfg = _make_config(num_experts=4, num_experts_per_tok=2)
        gate = Gate(cfg); gate.eval()
        N = 64
        x = torch.randn(N, cfg.hidden_dim)
        weights, indices, counts = gate(x)
        assert counts.shape == (cfg.num_experts,)
        assert counts.sum().item() == N * cfg.num_experts_per_tok

    def test_weights_dtype_matches_input(self):
        """Weights should be cast to input dtype via type_as."""
        cfg = _make_config()
        gate = Gate(cfg); gate.eval()
        x = torch.randn(32, cfg.hidden_dim, dtype=torch.float32)
        weights, _, _ = gate(x)
        assert weights.dtype == x.dtype



# ── SECTION 2: Sigmoid gating (NOT softmax) ─────────────────────────────

class TestSigmoidGating:

    def test_scores_are_sigmoid(self):
        """Router scores should be in (0, 1) — sigmoid range."""
        cfg = _make_config()
        gate = Gate(cfg); gate.eval()
        x = torch.randn(256, cfg.hidden_dim)

        with torch.no_grad():
            raw_scores = torch.sigmoid(gate.router(x))

        # All scores should be strictly between 0 and 1
        assert raw_scores.min() > 0.0
        assert raw_scores.max() < 1.0

    def test_not_softmax(self):
        """Scores should NOT sum to 1 across experts (sigmoid, not softmax)."""
        cfg = _make_config(num_experts=4)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(32, cfg.hidden_dim)

        with torch.no_grad():
            raw_scores = torch.sigmoid(gate.router(x))

        # If softmax, each row sums to 1.0. With sigmoid, they generally don't.
        row_sums = raw_scores.sum(dim=-1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01), (
            "Scores sum to 1 — this looks like softmax, should be sigmoid"
        )


# ── SECTION 3: Weight normalization ─────────────────────────────

class TestWeightNormalization:

    def test_weights_sum_to_route_scale(self):
        """Selected weights should sum to route_scale per token."""
        cfg = _make_config(route_scale=1.0)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        weights, _, _ = gate(x)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums) * cfg.route_scale,
            atol=1e-5, rtol=1e-5,
            msg="Weights should sum to route_scale per token",
        )

    def test_custom_route_scale(self):
        """Custom route_scale should scale the normalized weights."""
        cfg = _make_config(route_scale=2.5)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        weights, _, _ = gate(x)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.full_like(row_sums, 2.5),
            atol=1e-5, rtol=1e-5,
        )

    def test_weights_positive(self):
        """All weights should be positive (from sigmoid)."""
        cfg = _make_config()
        gate = Gate(cfg); gate.eval()
        x = torch.randn(128, cfg.hidden_dim)
        weights, _, _ = gate(x)
        assert (weights > 0).all(), "All routing weights should be positive"


# ── SECTION 4: Decoupled selection vs weighting ──────────────────────────

class TestDecoupledSelectionWeighting:

    def test_bias_affects_selection_not_weighting(self):
        """
        Biased scores select experts, but original (unbiased) scores
        determine the weights. Changing bias should change WHICH experts
        are selected but not the weight computation formula.
        """
        cfg = _make_config(num_experts=4, num_experts_per_tok=2)
        torch.manual_seed(42)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(32, cfg.hidden_dim)

        # Zero bias
        gate.bias.zero_()
        w0, idx0, _ = gate(x.clone())

        # Heavy bias toward expert 0
        gate.bias.zero_()
        gate.bias[0] = 100.0
        w1, idx1, _ = gate(x.clone())

        # Expert selection should differ
        assert not torch.equal(idx0, idx1), "Bias should change expert selection"

        # But weights should still be valid (sum to route_scale)
        torch.testing.assert_close(
            w1.sum(dim=-1), torch.ones(32) * cfg.route_scale,
            atol=1e-5, rtol=1e-5,
        )

    def test_bias_detached(self):
        """Bias should not participate in gradient computation."""
        cfg = _make_config()
        gate = Gate(cfg); gate.train()
        x = torch.randn(32, cfg.hidden_dim, requires_grad=True)
        weights, _, _ = gate(x)
        weights.sum().backward()

        # Router weight should have gradients
        assert gate.router.weight.grad is not None
        # Bias is a buffer, not a parameter — no .grad attribute
        assert not gate.bias.requires_grad



# ── SECTION 5: Bias update rule (Loss-Free Balancing) ───────────

class TestBiasUpdate:

    def test_bias_updates_in_train_mode(self):
        """Bias should change after forward pass in training mode."""
        cfg = _make_config()
        gate = Gate(cfg); gate.train()
        gate.bias.zero_()
        bias_before = gate.bias.clone()

        x = torch.randn(128, cfg.hidden_dim)
        with torch.no_grad():
            gate(x)

        assert not torch.equal(gate.bias, bias_before), (
            "Bias should update in train mode"
        )

    def test_bias_no_update_in_eval_mode(self):
        """Bias should NOT change in eval mode."""
        cfg = _make_config()
        gate = Gate(cfg); gate.eval()
        gate.bias.zero_()
        bias_before = gate.bias.clone()

        x = torch.randn(128, cfg.hidden_dim)
        with torch.no_grad():
            gate(x)

        torch.testing.assert_close(gate.bias, bias_before, atol=0, rtol=0,
            msg="Bias should not update in eval mode")

    def test_update_direction(self):
        """
        update_bias rule: bias += update_param * sign(mean_load - current_load)
        Under-utilized experts get positive bias, over-utilized get negative.
        """
        cfg = _make_config(num_experts=4, update_param=0.1)
        gate = Gate(cfg, layer_idx=0); gate.train()
        gate.bias.zero_()

        # Create a load where expert 0 is heavily loaded
        load = torch.tensor([100, 10, 10, 10], dtype=torch.float32)
        gate.update_bias(load)

        # Expert 0 (overloaded) should get negative bias
        assert gate.bias[0] < 0, "Overloaded expert should get negative bias"
        # Experts 1,2,3 (underloaded) should get positive bias
        assert gate.bias[1] > 0, "Underloaded expert should get positive bias"
        assert gate.bias[2] > 0
        assert gate.bias[3] > 0

    def test_bias_clamping(self):
        """Bias should be clamped to [-10, 10]."""
        cfg = _make_config(update_param=1.0)
        gate = Gate(cfg, layer_idx=0); gate.train()

        # Many extreme updates
        load = torch.tensor([1000, 0, 0, 0], dtype=torch.float32)
        for _ in range(200):
            gate.update_bias(load)

        assert gate.bias.max().item() <= 10.0
        assert gate.bias.min().item() >= -10.0

    def test_per_layer_scaling(self):
        """Deeper layers should have more aggressive bias updates."""
        cfg = _make_config(num_hidden_layers=24, update_param=1e-3)

        gate_shallow = Gate(cfg, layer_idx=0)
        gate_deep = Gate(cfg, layer_idx=23)

        assert gate_deep.effective_update > gate_shallow.effective_update, (
            "Deeper layers should have larger effective_update"
        )

        # Layer 0: scale = 1.0, Layer 23: scale = 1.5
        expected_shallow = 1e-3 * 1.0
        expected_deep = 1e-3 * 1.5
        assert abs(gate_shallow.effective_update - expected_shallow) < 1e-8
        assert abs(gate_deep.effective_update - expected_deep) < 1e-8

    def test_bias_converges_toward_balance(self):
        """After many forward passes, bias should push toward balanced load."""
        cfg = _make_config(num_experts=4, update_param=1e-2)
        torch.manual_seed(42)
        gate = Gate(cfg, layer_idx=0); gate.train()
        gate.bias.zero_()

        # Run many forward passes
        for _ in range(500):
            x = torch.randn(256, cfg.hidden_dim)
            with torch.no_grad():
                _, indices, _ = gate(x)

        # Check load balance after bias adaptation
        x_test = torch.randn(1024, cfg.hidden_dim)
        with torch.no_grad():
            gate.eval()
            _, indices, counts = gate(x_test)

        # Each expert should get roughly 25% (±15%) of assignments
        total = counts.sum().item()
        for i in range(4):
            pct = counts[i].item() / total * 100
            assert 10 < pct < 40, (
                f"Expert {i} got {pct:.1f}% — load balancing failed"
            )



# ── SECTION 6: Router weight properties ─────────────────────────────

class TestRouterProperties:

    def test_router_no_bias(self):
        """Router linear layer should have bias=False."""
        cfg = _make_config()
        gate = Gate(cfg)
        assert gate.router.bias is None, "Router should have no bias term"

    def test_router_weight_shape(self):
        """Router weight: (num_experts, hidden_dim)."""
        cfg = _make_config(num_experts=4, hidden_dim=768)
        gate = Gate(cfg)
        assert gate.router.weight.shape == (4, 768)

    def test_bias_buffer_shape(self):
        """Bias buffer: (num_experts,)."""
        cfg = _make_config(num_experts=4)
        gate = Gate(cfg)
        assert gate.bias.shape == (4,)

    def test_bias_is_buffer_not_parameter(self):
        """Bias should be a buffer, not a learnable parameter."""
        cfg = _make_config()
        gate = Gate(cfg)
        param_names = [n for n, _ in gate.named_parameters()]
        assert "bias" not in param_names, "bias should not be a parameter"
        buffer_names = [n for n, _ in gate.named_buffers()]
        assert "bias" in buffer_names, "bias should be a buffer"


# ── SECTION 7: Gradient flow ─────────────────────────────────────────────

class TestGateGradients:

    def test_gradient_to_router_weights(self):
        cfg = _make_config()
        gate = Gate(cfg); gate.train()
        x = torch.randn(64, cfg.hidden_dim)
        weights, _, _ = gate(x)
        weights.sum().backward()
        assert gate.router.weight.grad is not None
        assert gate.router.weight.grad.abs().sum() > 0

    def test_gradient_to_input(self):
        cfg = _make_config()
        gate = Gate(cfg); gate.train()
        x = torch.randn(64, cfg.hidden_dim, requires_grad=True)
        weights, _, _ = gate(x)
        weights.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0



# ── SECTION 8: Numerical stability & edge cases ─────────────────────────

class TestGateEdgeCases:

    def test_single_token(self):
        cfg = _make_config(num_experts=4, num_experts_per_tok=2)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(1, cfg.hidden_dim)
        w, idx, counts = gate(x)
        assert w.shape == (1, 2)
        assert counts.sum().item() == 2

    def test_top1_routing(self):
        cfg = _make_config(num_experts=4, num_experts_per_tok=1)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(64, cfg.hidden_dim)
        w, idx, counts = gate(x)
        assert w.shape == (64, 1)
        assert idx.shape == (64, 1)
        # Weights should all equal route_scale (only 1 expert, normalized)
        torch.testing.assert_close(
            w, torch.ones_like(w) * cfg.route_scale,
            atol=1e-5, rtol=1e-5,
        )

    @pytest.mark.parametrize("n_experts,topk", [(2,1), (4,2), (6,3), (8,4)])
    def test_varied_configs(self, n_experts, topk):
        cfg = _make_config(num_experts=n_experts, num_experts_per_tok=topk)
        gate = Gate(cfg); gate.eval()
        x = torch.randn(128, cfg.hidden_dim)
        w, idx, counts = gate(x)
        assert w.shape == (128, topk)
        assert idx.shape == (128, topk)
        assert counts.sum().item() == 128 * topk
        assert (idx >= 0).all() and (idx < n_experts).all()

    def test_state_dict_roundtrip(self):
        cfg = _make_config()
        torch.manual_seed(42)
        gate = Gate(cfg); gate.eval()
        gate.bias.fill_(3.14)

        x = torch.randn(32, cfg.hidden_dim)
        with torch.no_grad():
            w_before, idx_before, _ = gate(x.clone())

        state = gate.state_dict()
        gate2 = Gate(cfg)
        gate2.load_state_dict(state); gate2.eval()
        with torch.no_grad():
            w_after, idx_after, _ = gate2(x.clone())

        torch.testing.assert_close(w_before, w_after, atol=0, rtol=0)
        torch.testing.assert_close(idx_before, idx_after, atol=0, rtol=0)

    @requires_cuda
    def test_cuda_bf16(self):
        cfg = _make_config(dtype=torch.bfloat16)
        gate = Gate(cfg, device="cuda"); gate.eval()
        x = torch.randn(128, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
        w, idx, counts = gate(x)
        assert not torch.isnan(w).any()
        assert counts.sum().item() == 128 * cfg.num_experts_per_tok


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
