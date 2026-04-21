"""
SwiGLU Activation & MLP/Expert — Production Correctness Suite
=============================================================
Validates swiglu activation, MLPBlock, and Expert FFN modules.
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

from src.models.model_flash_attn import swiglu, MLPBlock, Expert
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
    cfg.max_context_len = 2048
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    return cfg


# ── SECTION 1: SwiGLU Activation Correctness ──────────────────────────────────────────

class TestSwiGLUActivation:

    def test_output_shape(self):
        """SwiGLU halves the last dimension."""
        x = torch.randn(32, 1520)
        out = swiglu(x)
        assert out.shape == (32, 760)

    def test_output_shape_3d(self):
        x = torch.randn(4, 128, 1520)
        out = swiglu(x)
        assert out.shape == (4, 128, 760)

    def test_manual_computation(self):
        """Verify SwiGLU formula: (glu * sigmoid(alpha * glu)) * (linear + 1)."""
        alpha, limit = 1.702, 7.0
        torch.manual_seed(42)
        x = torch.randn(8, 20)
        x_glu, x_linear = x.chunk(2, dim=-1)
        x_glu_c = x_glu.clamp(max=limit)
        x_linear_c = x_linear.clamp(min=-limit, max=limit)
        expected = (x_glu_c * torch.sigmoid(alpha * x_glu_c)) * (x_linear_c + 1)
        actual = swiglu(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_clamping_glu_upper(self):
        """GLU half above limit=7.0 should be clamped."""
        x = torch.zeros(1, 20)
        x[0, :10] = 100.0
        out = swiglu(x)
        assert out.max().item() < 10.0, "GLU clamping failed"

    def test_clamping_linear_symmetric(self):
        """Linear half clamped to [-limit, +limit]."""
        x = torch.zeros(1, 20)
        x[0, :10] = 1.0
        x[0, 10:] = 100.0
        out = swiglu(x)
        assert not torch.isinf(out).any()

    def test_zero_input(self):
        x = torch.zeros(8, 20)
        out = swiglu(x)
        assert not torch.isnan(out).any()
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-7, rtol=0)

    def test_gradient_through_clamping(self):
        x = torch.randn(32, 1520, requires_grad=True)
        swiglu(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 10.0, 1e3])
    def test_numerical_stability(self, scale):
        x = torch.randn(32, 1520) * scale
        out = swiglu(x)
        assert not torch.isnan(out).any(), f"NaN at scale {scale}"
        assert not torch.isinf(out).any(), f"Inf at scale {scale}"


# ── SECTION 2: MLPBlock Correctness ────────────────────────────────────────────

class TestMLPBlock:

    def test_output_shape_2d(self):
        cfg = _make_config()
        mlp = MLPBlock(cfg); mlp.eval()
        x = torch.randn(64, cfg.hidden_dim)
        assert mlp(x).shape == x.shape

    def test_output_shape_3d(self):
        cfg = _make_config()
        mlp = MLPBlock(cfg); mlp.eval()
        x = torch.randn(4, 128, cfg.hidden_dim)
        assert mlp(x).shape == x.shape

    def test_weight_dimensions(self):
        cfg = _make_config(hidden_dim=768, intermediate_size=760)
        mlp = MLPBlock(cfg)
        assert mlp.w1.weight.shape == (2 * 760, 768)
        assert mlp.w2.weight.shape == (768, 760)
        assert mlp.w3.weight.shape == (760, 768)

    def test_gradient_flow(self):
        cfg = _make_config()
        mlp = MLPBlock(cfg); mlp.train()
        x = torch.randn(32, cfg.hidden_dim)
        mlp(x).sum().backward()
        for name in ['w1', 'w2', 'w3']:
            w = getattr(mlp, name)
            assert w.weight.grad is not None, f"No gradient on {name}"
            assert w.weight.grad.abs().sum() > 0

    def test_dropout_train_vs_eval(self):
        cfg = _make_config(ffn_dropout=0.5)
        mlp = MLPBlock(cfg)
        x = torch.randn(64, cfg.hidden_dim)
        mlp.eval()
        with torch.no_grad():
            o1 = mlp(x.clone()); o2 = mlp(x.clone())
        torch.testing.assert_close(o1, o2, atol=0, rtol=0)
        mlp.train()
        with torch.no_grad():
            torch.manual_seed(1); o3 = mlp(x.clone())
            torch.manual_seed(2); o4 = mlp(x.clone())
        assert not torch.allclose(o3, o4, atol=1e-6)

    def test_state_dict_roundtrip(self):
        cfg = _make_config()
        torch.manual_seed(42)
        mlp = MLPBlock(cfg); mlp.eval()
        x = torch.randn(32, cfg.hidden_dim)
        with torch.no_grad():
            out_before = mlp(x.clone())
        mlp2 = MLPBlock(cfg)
        mlp2.load_state_dict(mlp.state_dict()); mlp2.eval()
        with torch.no_grad():
            out_after = mlp2(x.clone())
        torch.testing.assert_close(out_before, out_after, atol=0, rtol=0)

    @requires_cuda
    def test_cuda_bf16(self):
        cfg = _make_config(dtype=torch.bfloat16)
        mlp = MLPBlock(cfg, device="cuda"); mlp.train()
        x = torch.randn(64, cfg.hidden_dim, device="cuda")
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = mlp(x)
        assert not torch.isnan(out).any()
        out.sum().backward()


# ── SECTION 3: Expert Module ───────────────────────────────────────────────────

class TestExpert:

    def test_expert_shape(self):
        cfg = _make_config()
        expert = Expert(cfg); expert.eval()
        x = torch.randn(64, cfg.hidden_dim)
        assert expert(x).shape == x.shape

    def test_expert_matches_mlpblock_same_weights(self):
        """Expert and MLPBlock with identical weights produce identical output."""
        cfg = _make_config()
        torch.manual_seed(42)
        expert = Expert(cfg)
        mlp = MLPBlock(cfg)
        mlp.load_state_dict(expert.state_dict())
        expert.eval(); mlp.eval()
        x = torch.randn(32, cfg.hidden_dim)
        with torch.no_grad():
            torch.testing.assert_close(expert(x.clone()), mlp(x.clone()), atol=0, rtol=0)

    def test_expert_gradient_flow(self):
        cfg = _make_config()
        expert = Expert(cfg); expert.train()
        x = torch.randn(32, cfg.hidden_dim, requires_grad=True)
        expert(x).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    @pytest.mark.parametrize("scale", [1e-5, 1.0, 50.0])
    def test_expert_numerical_stability(self, scale):
        cfg = _make_config()
        expert = Expert(cfg); expert.eval()
        x = torch.randn(32, cfg.hidden_dim) * scale
        with torch.no_grad():
            out = expert(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ── SECTION 4: Training simulation ───────────────────────────────────────────

class TestMLPTraining:

    def test_loss_decreases(self):
        cfg = _make_config()
        mlp = MLPBlock(cfg); mlp.train()
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
        losses = []
        for step in range(20):
            opt.zero_grad()
            torch.manual_seed(42)
            x = torch.randn(32, cfg.hidden_dim)
            loss = mlp(x).pow(2).mean()
            loss.backward(); opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_all_weights_updated(self):
        cfg = _make_config()
        torch.manual_seed(42)
        mlp = MLPBlock(cfg); mlp.train()
        before = {n: p.clone() for n, p in mlp.named_parameters()}
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
        mlp(torch.randn(32, cfg.hidden_dim)).sum().backward()
        opt.step()
        for n, p in mlp.named_parameters():
            assert not torch.equal(p, before[n]), f"{n} not updated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
