"""
Weight Initialization — Production Correctness Suite
====================================================
Validates init_gpt_model from weight_init.py to ensure every parameter
in GPT_FLASH is initialized with the correct distribution and scale.

Coverage:
  - Embedding: N(0, 0.02)
  - Q/K/V projections: N(0, 0.02)
  - Output projections (wo, w2, unembedding): N(0, 0.02/√(2*num_layers))
  - Router weights: N(0, 0.01)
  - Router bias buffer: zeros
  - RMSNorm scales: ones
  - Expert weights: correct std per type
  - Deterministic re-initialization
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

from src.models.model_flash_attn import GPT_FLASH
from src.models.weight_init import init_gpt_model
from src.scripts.configs.model_config import ModelConfig

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
    cfg.vocab_size = overrides.get("vocab_size", 200)
    cfg.num_attn_heads = overrides.get("num_attn_heads", 12)
    cfg.num_key_value_heads = overrides.get("num_key_value_heads", 6)
    cfg.head_dim = cfg.hidden_dim // cfg.num_attn_heads
    cfg.num_hidden_layers = overrides.get("num_hidden_layers", 4)
    cfg.base = 10000
    cfg.initial_context_len = 2048
    cfg.max_context_len = 2048
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    return cfg

def _init_model(cfg):
    """Create and initialize a GPT_FLASH model."""
    torch.manual_seed(42)
    model = GPT_FLASH(cfg)
    init_gpt_model(model, cfg)
    return model

def _check_normal(tensor, expected_std, name, tolerance=0.3):
    """Check that tensor is approximately N(0, expected_std)."""
    t = tensor.float().detach()
    actual_mean = t.mean().item()
    actual_std = t.std().item()
    assert abs(actual_mean) < 0.05, (
        f"{name}: mean={actual_mean:.4f}, expected ≈ 0.0"
    )
    rel_err = abs(actual_std - expected_std) / expected_std
    assert rel_err < tolerance, (
        f"{name}: std={actual_std:.4f}, expected={expected_std:.4f}, "
        f"rel_err={rel_err:.2f} (tolerance={tolerance})"
    )


# ── SECTION 1: Embedding initialization ─────────────────────────────────────────

class TestEmbeddingInit:

    def test_embedding_std(self):
        """Embeddings should be N(0, 0.02)."""
        cfg = _make_config()
        model = _init_model(cfg)
        _check_normal(model.embeddings.weight, 0.02, "embeddings")

    def test_unembedding_std(self):
        """Unembedding: N(0, 0.02) — not depth-scaled in init_gpt_model."""
        cfg = _make_config(num_hidden_layers=4)
        model = _init_model(cfg)
        _check_normal(model.unembedding.weight, 0.02, "unembedding")


# ── SECTION 2: Attention projection initialization ─────────────────────────────────────────

class TestAttentionInit:

    def test_qkv_projections_std(self):
        """wq, wk, wv should be N(0, 0.02)."""
        cfg = _make_config()
        model = _init_model(cfg)
        for layer in model.layers:
            attn = layer.attention
            _check_normal(attn.wq.weight, 0.02, "wq")
            _check_normal(attn.wk.weight, 0.02, "wk")
            _check_normal(attn.wv.weight, 0.02, "wv")

    def test_wo_scaled_std(self):
        """wo should be N(0, 0.02 / √(2 * num_layers))."""
        cfg = _make_config(num_hidden_layers=4)
        model = _init_model(cfg)
        expected_std = 0.02 / math.sqrt(2 * 4)
        for layer in model.layers:
            _check_normal(layer.attention.wo.weight, expected_std, "wo",
                          tolerance=0.4)


# ── SECTION 3: MoE / Expert initialization ─────────────────────────────────────────

class TestMoEInit:

    def test_router_weights_std(self):
        """Router weights should be N(0, 0.01)."""
        cfg = _make_config()
        model = _init_model(cfg)
        for layer in model.layers:
            router_w = layer.mlp.gate.router.weight
            _check_normal(router_w, 0.01, "router", tolerance=0.5)

    def test_router_bias_zero(self):
        """Router bias buffer should be all zeros."""
        cfg = _make_config()
        model = _init_model(cfg)
        for i, layer in enumerate(model.layers):
            bias = layer.mlp.gate.bias
            assert (bias == 0).all(), f"Layer {i}: router bias not zero"

    def test_expert_w1_w3_std(self):
        """Expert w1 and w3 should be N(0, 0.02)."""
        cfg = _make_config()
        model = _init_model(cfg)
        for layer in model.layers:
            for expert in layer.mlp.experts:
                _check_normal(expert.w1.weight, 0.02, "expert.w1")
                _check_normal(expert.w3.weight, 0.02, "expert.w3")

    def test_expert_w2_scaled_std(self):
        """Expert w2 should be N(0, 0.02 / √(2 * num_layers))."""
        cfg = _make_config(num_hidden_layers=4)
        model = _init_model(cfg)
        expected_std = 0.02 / math.sqrt(2 * 4)
        for layer in model.layers:
            for expert in layer.mlp.experts:
                _check_normal(expert.w2.weight, expected_std, "expert.w2",
                              tolerance=0.4)

    def test_shared_expert_w1_w3_std(self):
        """Shared expert w1/w3: N(0, 0.02)."""
        cfg = _make_config()
        model = _init_model(cfg)
        for layer in model.layers:
            shared = layer.mlp.shared_experts
            _check_normal(shared.w1.weight, 0.02, "shared.w1")
            _check_normal(shared.w3.weight, 0.02, "shared.w3")

    def test_shared_expert_w2_scaled_std(self):
        """Shared expert w2: N(0, 0.02 / √(2 * num_layers))."""
        cfg = _make_config(num_hidden_layers=4)
        model = _init_model(cfg)
        expected_std = 0.02 / math.sqrt(2 * 4)
        for layer in model.layers:
            shared = layer.mlp.shared_experts
            _check_normal(shared.w2.weight, expected_std, "shared.w2",
                          tolerance=0.4)


# ── SECTION 4: RMSNorm initialization ─────────────────────────────────────────

class TestNormInit:

    def test_layer_norms_ones(self):
        """All RMSNorm scale parameters should be initialized to ones."""
        cfg = _make_config()
        model = _init_model(cfg)

        # Final norm
        torch.testing.assert_close(
            model.norm.scale, torch.ones_like(model.norm.scale),
            atol=0, rtol=0, msg="Final norm scale not ones",
        )

        # Per-layer norms
        for i, layer in enumerate(model.layers):
            torch.testing.assert_close(
                layer.norm1.scale, torch.ones_like(layer.norm1.scale),
                atol=0, rtol=0, msg=f"Layer {i} norm1 scale not ones",
            )
            torch.testing.assert_close(
                layer.norm2.scale, torch.ones_like(layer.norm2.scale),
                atol=0, rtol=0, msg=f"Layer {i} norm2 scale not ones",
            )


# ── SECTION 5: Initialization properties ─────────────────────────────────────────

class TestInitProperties:

    def test_no_nan_after_init(self):
        """No NaN in any parameter after initialization."""
        cfg = _make_config()
        model = _init_model(cfg)
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in {name}"

    def test_no_inf_after_init(self):
        """No Inf in any parameter after initialization."""
        cfg = _make_config()
        model = _init_model(cfg)
        for name, p in model.named_parameters():
            assert not torch.isinf(p).any(), f"Inf in {name}"

    def test_deterministic_init(self):
        """Same seed → same weights."""
        cfg = _make_config()
        torch.manual_seed(42)
        m1 = GPT_FLASH(cfg); init_gpt_model(m1, cfg)
        torch.manual_seed(42)
        m2 = GPT_FLASH(cfg); init_gpt_model(m2, cfg)

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, atol=0, rtol=0,
                                       msg=f"Non-deterministic init: {n1}")

    def test_forward_works_after_init(self):
        """Model should produce valid output after initialization."""
        cfg = _make_config()
        model = _init_model(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_projections_smaller_than_input_projections(self):
        """
        Output projections (wo, w2) should have smaller std than input
        projections (wq, wk, wv, w1, w3) — this is the scaled init.
        """
        cfg = _make_config(num_hidden_layers=4)
        model = _init_model(cfg)
        layer = model.layers[0]

        input_std = layer.attention.wq.weight.float().std().item()
        output_std = layer.attention.wo.weight.float().std().item()

        assert output_std < input_std, (
            f"Output projection std ({output_std:.4f}) should be < "
            f"input projection std ({input_std:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
