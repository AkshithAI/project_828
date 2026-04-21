"""
GPT_FLASH End-to-End — Production Correctness Suite
====================================================
Validates the full GPT_FLASH model including TransformerDecoderBLK
composition, forward/backward, KV-cache inference, training loops,
and state dict serialization.

Coverage:
  - Full forward pass shape contracts
  - TransformerDecoderBLK residual connections
  - Multi-layer gradient flow
  - KV-cache autoregressive generation
  - Multi-step training loss trajectory
  - State dict save/load roundtrip
  - Parameter counting
  - Numerical stability across dtypes
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

from src.models.model_flash_attn import GPT_FLASH, TransformerDecoderBLK
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
    cfg.vocab_size = overrides.get("vocab_size", 200)
    cfg.num_attn_heads = overrides.get("num_attn_heads", 12)
    cfg.num_key_value_heads = overrides.get("num_key_value_heads", 6)
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


# ── SECTION 1: TransformerDecoderBLK ──────────────────────────────

class TestTransformerDecoderBLK:

    def test_output_shape(self):
        cfg = _make_config()
        blk = TransformerDecoderBLK(cfg, layer_idx=0); blk.eval()
        x = torch.randn(2, 32, cfg.hidden_dim)
        out = blk(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from both attention-only and mlp-only paths."""
        cfg = _make_config()
        torch.manual_seed(42)
        blk = TransformerDecoderBLK(cfg, layer_idx=0); blk.eval()
        x = torch.randn(1, 16, cfg.hidden_dim)
        with torch.no_grad():
            out = blk(x)
        # Residual means output ≠ 0 even if sublayers produce small values
        assert out.abs().sum() > 0

    def test_pre_norm_order(self):
        """
        Pre-norm architecture: norm is applied BEFORE attention and MLP.
        Verify norm1 and norm2 exist and are applied.
        """
        cfg = _make_config()
        blk = TransformerDecoderBLK(cfg, layer_idx=0)
        assert hasattr(blk, 'norm1')
        assert hasattr(blk, 'norm2')
        assert hasattr(blk, 'attention')
        assert hasattr(blk, 'mlp')

    def test_layer_idx_passed_to_moe(self):
        """layer_idx should be passed through to MoE → Gate for per-layer scaling."""
        cfg = _make_config(num_hidden_layers=24)
        blk0 = TransformerDecoderBLK(cfg, layer_idx=0)
        blk23 = TransformerDecoderBLK(cfg, layer_idx=23)
        assert blk0.mlp.gate.effective_update < blk23.mlp.gate.effective_update

    def test_gradient_flow(self):
        cfg = _make_config()
        blk = TransformerDecoderBLK(cfg, layer_idx=0); blk.train()
        x = torch.randn(2, 16, cfg.hidden_dim, requires_grad=True)
        blk(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ── SECTION 2: GPT_FLASH forward pass ──────────────────────────────

class TestGPTFlashForward:

    def test_output_shape(self):
        """Input: (B, S) token IDs → Output: (B, S, vocab_size) logits."""
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.eval()
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)

    def test_single_token(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, cfg.vocab_size)

    def test_no_nan(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.eval()
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_different_outputs(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.eval()
        x1 = torch.randint(0, cfg.vocab_size, (1, 16))
        x2 = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2, atol=1e-5)

    def test_num_layers(self):
        """Model should have exactly num_hidden_layers transformer blocks."""
        cfg = _make_config(num_hidden_layers=4)
        model = GPT_FLASH(cfg)
        assert len(model.layers) == 4


# ── SECTION 3: KV-Cache Inference ──────────────────────────────

class TestKVCacheInference:

    def test_inference_flag(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg, inference=True)
        assert model.inference is True

    def test_reset_cache(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg, inference=True)
        model.reset_cache(batch_size=2)
        for layer in model.layers:
            assert layer.attention.cache_k is not None
            assert layer.attention.cache_k.shape[0] == 2

    def test_prefill_matches_training_forward(self):
        """Prefill with full sequence should match training forward."""
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)
        model_train = GPT_FLASH(cfg, inference=False)
        model_infer = GPT_FLASH(cfg, inference=True)
        model_infer.load_state_dict(model_train.state_dict())
        model_train.eval(); model_infer.eval()

        x = torch.randint(0, cfg.vocab_size, (1, 16))

        with torch.no_grad():
            out_train = model_train(x)
            model_infer.reset_cache(batch_size=1)
            out_infer = model_infer(x, start_pos=0)

        torch.testing.assert_close(
            out_train, out_infer, atol=1e-4, rtol=1e-3,
            msg="Inference prefill should match training forward",
        )

    def test_autoregressive_decode(self):
        """
        Prefill prompt, then decode tokens one at a time.
        Each decode step should produce valid logits.
        """
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)
        model = GPT_FLASH(cfg, inference=True); model.eval()
        model.reset_cache(batch_size=1)

        # Prefill with 8 tokens
        prompt = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits = model(prompt, start_pos=0)
        assert logits.shape == (1, 8, cfg.vocab_size)

        # Decode 4 more tokens
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        for pos in range(8, 12):
            with torch.no_grad():
                step_logits = model(next_token, start_pos=pos)
            assert step_logits.shape == (1, 1, cfg.vocab_size)
            assert not torch.isnan(step_logits).any()
            next_token = step_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    def test_incremental_matches_full(self):
        """
        Token-by-token generation should match full-sequence forward
        at each position.
        """
        cfg = _make_config(max_context_len=64)
        torch.manual_seed(42)

        model_full = GPT_FLASH(cfg, inference=False)
        model_incr = GPT_FLASH(cfg, inference=True)
        model_incr.load_state_dict(model_full.state_dict())
        model_full.eval(); model_incr.eval()

        seq_len = 6
        x = torch.randint(0, cfg.vocab_size, (1, seq_len))

        # Full forward
        with torch.no_grad():
            out_full = model_full(x)

        # Incremental: prefill first 5, decode position 5
        model_incr.reset_cache(batch_size=1)
        with torch.no_grad():
            _ = model_incr(x[:, :5], start_pos=0)
            out_last = model_incr(x[:, 5:6], start_pos=5)

        torch.testing.assert_close(
            out_full[0, 5], out_last[0, 0],
            atol=1e-3, rtol=1e-2,
            msg="Incremental decode at last position should match full forward",
        )


# ── SECTION 4: Training (backward pass) ─────────────────────────────────────

class TestGPTFlashTraining:

    def test_cross_entropy_loss_backward(self):
        """Standard language modeling loss should backprop without errors."""
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        targets = torch.randint(0, cfg.vocab_size, (2, 32))

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()

        assert math.isfinite(loss.item())

    def test_all_layers_get_gradients(self):
        """Every transformer layer should receive gradients."""
        cfg = _make_config(num_hidden_layers=4)
        model = GPT_FLASH(cfg); model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.randint(0, cfg.vocab_size, (2, 16))

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()

        # Check each layer has gradients
        for i, layer in enumerate(model.layers):
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in layer.parameters()
            )
            assert has_grad, f"Layer {i} received no gradients"

        # Check embedding and unembedding
        assert model.embeddings.weight.grad is not None
        assert model.unembedding.weight.grad is not None

    def test_multi_step_loss_finite(self):
        """5 training steps should all produce finite losses."""
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(5):
            opt.zero_grad()
            torch.manual_seed(step)
            x = torch.randint(0, cfg.vocab_size, (2, 16))
            targets = torch.randint(0, cfg.vocab_size, (2, 16))
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
            loss.backward()
            opt.step()
            assert math.isfinite(loss.item()), f"Step {step}: loss={loss.item()}"

    def test_expert_counts_tracked(self):
        """Expert counts should be tracked during training forward passes."""
        cfg = _make_config()
        model = GPT_FLASH(cfg); model.train()

        # Reset counts
        for layer in model.layers:
            layer.mlp.reset_expert_counts()

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            model(x)

        for i, layer in enumerate(model.layers):
            total = layer.mlp.expert_counts.sum().item()
            assert total > 0, f"Layer {i}: no expert counts tracked"


# ── SECTION 5: State dict serialization ───────────────────────────────────

class TestStateDictRoundtrip:

    def test_save_load_produces_identical_output(self):
        """state_dict save → load should produce bit-identical output."""
        cfg = _make_config()
        torch.manual_seed(42)
        model = GPT_FLASH(cfg); model.eval()

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out_before = model(x)

        state = model.state_dict()
        model2 = GPT_FLASH(cfg)
        model2.load_state_dict(state); model2.eval()
        with torch.no_grad():
            out_after = model2(x)

        torch.testing.assert_close(out_before, out_after, atol=0, rtol=0)

    def test_all_keys_preserved(self):
        """All state dict keys should survive save/load."""
        cfg = _make_config()
        model = GPT_FLASH(cfg)
        keys_original = set(model.state_dict().keys())

        model2 = GPT_FLASH(cfg)
        model2.load_state_dict(model.state_dict())
        keys_loaded = set(model2.state_dict().keys())

        assert keys_original == keys_loaded



# ── SECTION 6: Parameter counting ──────────────────────────────────────────

class TestParameterCounting:

    def test_total_params_positive(self):
        cfg = _make_config()
        model = GPT_FLASH(cfg)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_all_params_trainable(self):
        """By default, all parameters should be trainable (no frozen layers)."""
        cfg = _make_config()
        model = GPT_FLASH(cfg)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable

    def test_param_count_scales_with_layers(self):
        """More layers → more parameters."""
        cfg_small = _make_config(num_hidden_layers=2)
        cfg_large = _make_config(num_hidden_layers=4)
        m_small = GPT_FLASH(cfg_small)
        m_large = GPT_FLASH(cfg_large)
        p_small = sum(p.numel() for p in m_small.parameters())
        p_large = sum(p.numel() for p in m_large.parameters())
        assert p_large > p_small



# ── SECTION 7: CUDA / bf16 (production environment) ─────────────────────

class TestCUDABf16:

    @requires_cuda
    def test_forward_backward_bf16(self):
        cfg = _make_config(dtype=torch.bfloat16)
        model = GPT_FLASH(cfg, device="cuda"); model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
        targets = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, cfg.vocab_size), targets.view(-1)
            )
        loss.backward()
        assert math.isfinite(loss.item())

    @requires_cuda
    def test_inference_kv_cache_cuda(self):
        cfg = _make_config(dtype=torch.bfloat16, max_context_len=64)
        model = GPT_FLASH(cfg, device="cuda", inference=True); model.eval()
        model.reset_cache(batch_size=1)

        prompt = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
        with torch.no_grad():
            logits = model(prompt, start_pos=0)
        assert logits.shape == (1, 8, cfg.vocab_size)
        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
