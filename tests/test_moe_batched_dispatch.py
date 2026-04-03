"""
Comprehensive test suite for MoE Batched Expert Dispatch.

Tests verify correctness, gradient flow, edge cases, numerical stability,
and equivalence against the original sequential loop-based implementation.

All tests run on CPU to be hardware-agnostic (the dispatch logic is
device-independent). GPU-specific performance tests are out of scope here
and should be validated in production with profiling tools.
"""

import sys
import os
import copy
import types
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Mock flash_attn before any project imports (GPU-only package) ─────
# flash_attn is a CUDA-only C extension. We provide a pure-PyTorch fallback
# so that MoE tests can run on CPU without the package installed.
_flash_attn_mock = types.ModuleType("flash_attn")


def _flash_attn_func_mock(Q, K, V, causal=False):
    """Pure-PyTorch fallback for flash_attn_func (used only in tests)."""
    Q_t = Q.transpose(1, 2)
    K_t = K.transpose(1, 2)
    V_t = V.transpose(1, 2)
    out = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=causal)
    return out.transpose(1, 2)


_flash_attn_mock.flash_attn_func = _flash_attn_func_mock
sys.modules.setdefault("flash_attn", _flash_attn_mock)

# ── Path setup so we can import from the project ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_flash_attn import (
    MoE,
    Gate,
    Expert,
    MLPBlock,
    swiglu,
)
from src.scripts.configs.model_config import ModelConfig


# ══════════════════════════════════════════════════════════════════════
# Reference implementation: the ORIGINAL sequential loop-based dispatch
# ══════════════════════════════════════════════════════════════════════

def _moe_forward_original(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """
    Exact replica of the old sequential loop-based MoE.forward().
    Used as the ground-truth reference to verify the batched version
    produces identical outputs.
    """
    inp_shape = x.shape
    x = x.view(-1, moe.dim)
    xprt_weights, xprt_idxs, counts = moe.gate(x)

    moe.expert_counts += counts
    moe.total_tokens += x.shape[0] * moe.n_routed_experts
    routed_xprt_out = torch.zeros_like(x)

    for i, expert in enumerate(moe.experts):
        batch_idx, expert_idx = torch.where(xprt_idxs == i)
        if batch_idx.numel() == 0:
            continue
        routed_xprt_out[batch_idx] += (
            xprt_weights[batch_idx, expert_idx, None] * expert(x[batch_idx])
        )
    mlp_out = routed_xprt_out + moe.shared_experts(x)

    return mlp_out.reshape(inp_shape)


# ══════════════════════════════════════════════════════════════════════
# Test fixtures
# ══════════════════════════════════════════════════════════════════════

def _make_config(**overrides) -> ModelConfig:
    """Create a lightweight ModelConfig for testing."""
    cfg = ModelConfig.__new__(ModelConfig)
    # Defaults: small dims for fast tests
    cfg.hidden_dim = overrides.get("hidden_dim", 64)
    cfg.intermediate_size = overrides.get("intermediate_size", 48)
    cfg.num_experts = overrides.get("num_experts", 4)
    cfg.num_experts_per_tok = overrides.get("num_experts_per_tok", 2)
    cfg.update_param = overrides.get("update_param", 1e-3)
    cfg.route_scale = overrides.get("route_scale", 1.0)
    cfg.ffn_dropout = overrides.get("ffn_dropout", 0.0)
    cfg.dtype = overrides.get("dtype", torch.float32)
    cfg.vocab_size = overrides.get("vocab_size", 100)
    cfg.num_attn_heads = overrides.get("num_attn_heads", 4)
    cfg.num_key_value_heads = overrides.get("num_key_value_heads", 2)
    cfg.head_dim = cfg.hidden_dim // cfg.num_attn_heads
    cfg.num_hidden_layers = overrides.get("num_hidden_layers", 2)
    cfg.base = overrides.get("base", 10000)
    cfg.initial_context_len = 256
    cfg.max_context_len = 256
    cfg.ntk_alpha = 1.0
    cfg.ntk_beta = 32.0
    cfg.scaling_factor = 1.0
    cfg.dropout = 0.0
    return cfg


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def moe(config):
    torch.manual_seed(42)
    m = MoE(config, device=None)
    m.eval()
    return m


@pytest.fixture
def sample_input(config):
    """Standard 2D input: (N, dim) with N=32 tokens."""
    torch.manual_seed(123)
    return torch.randn(32, config.hidden_dim)


@pytest.fixture
def sample_input_3d(config):
    """Standard 3D input: (batch, seq_len, dim)."""
    torch.manual_seed(123)
    return torch.randn(4, 8, config.hidden_dim)


# ══════════════════════════════════════════════════════════════════════
# Core correctness tests
# ══════════════════════════════════════════════════════════════════════

class TestOutputEquivalence:
    """Verify the batched dispatch produces identical results to the original."""

    def test_output_equivalence_2d(self, config):
        """Batched dispatch output matches original loop on 2D input."""
        torch.manual_seed(42)
        moe_batched = MoE(config, device=None)
        moe_batched.eval()

        # Deep-copy to create an identical reference
        moe_original = copy.deepcopy(moe_batched)

        torch.manual_seed(999)
        x = torch.randn(32, config.hidden_dim)

        # Reset gate biases to same state
        moe_batched.gate.bias.zero_()
        moe_original.gate.bias.zero_()

        with torch.no_grad():
            out_batched = moe_batched(x.clone())
            # Reset counters so the reference doesn't double-count
            moe_original.expert_counts.zero_()
            moe_original.total_tokens = 0
            out_original = _moe_forward_original(moe_original, x.clone())

        torch.testing.assert_close(
            out_batched, out_original,
            atol=1e-6, rtol=1e-5,
            msg="Batched dispatch output diverges from original loop implementation",
        )

    def test_output_equivalence_3d(self, config):
        """Batched dispatch output matches original loop on 3D input."""
        torch.manual_seed(42)
        moe_batched = MoE(config, device=None)
        moe_batched.eval()
        moe_original = copy.deepcopy(moe_batched)

        torch.manual_seed(999)
        x = torch.randn(4, 8, config.hidden_dim)

        moe_batched.gate.bias.zero_()
        moe_original.gate.bias.zero_()

        with torch.no_grad():
            out_batched = moe_batched(x.clone())
            moe_original.expert_counts.zero_()
            moe_original.total_tokens = 0
            out_original = _moe_forward_original(moe_original, x.clone())

        torch.testing.assert_close(
            out_batched, out_original,
            atol=1e-6, rtol=1e-5,
            msg="Batched dispatch output diverges from original on 3D input",
        )

    def test_output_equivalence_large_batch(self, config):
        """Equivalence holds for a larger batch (256 tokens)."""
        torch.manual_seed(42)
        moe_batched = MoE(config, device=None)
        moe_batched.eval()
        moe_original = copy.deepcopy(moe_batched)

        torch.manual_seed(777)
        x = torch.randn(256, config.hidden_dim)

        moe_batched.gate.bias.zero_()
        moe_original.gate.bias.zero_()

        with torch.no_grad():
            out_batched = moe_batched(x.clone())
            moe_original.expert_counts.zero_()
            moe_original.total_tokens = 0
            out_original = _moe_forward_original(moe_original, x.clone())

        torch.testing.assert_close(
            out_batched, out_original,
            atol=1e-6, rtol=1e-5,
        )

    def test_output_equivalence_varied_configs(self):
        """Equivalence holds across different num_experts and top-k settings."""
        for num_experts, topk in [(2, 1), (4, 2), (6, 3), (8, 2)]:
            cfg = _make_config(num_experts=num_experts, num_experts_per_tok=topk)
            torch.manual_seed(42)
            moe_batched = MoE(cfg, device=None)
            moe_batched.eval()
            moe_original = copy.deepcopy(moe_batched)

            torch.manual_seed(999)
            x = torch.randn(64, cfg.hidden_dim)

            moe_batched.gate.bias.zero_()
            moe_original.gate.bias.zero_()

            with torch.no_grad():
                out_batched = moe_batched(x.clone())
                moe_original.expert_counts.zero_()
                moe_original.total_tokens = 0
                out_original = _moe_forward_original(moe_original, x.clone())

            torch.testing.assert_close(
                out_batched, out_original,
                atol=1e-6, rtol=1e-5,
                msg=f"Mismatch with num_experts={num_experts}, topk={topk}",
            )


class TestGradientEquivalence:
    """Verify gradients are identical between batched and original dispatch."""

    def test_gradient_equivalence(self, config):
        """Parameter gradients match between batched and original implementations."""
        torch.manual_seed(42)
        moe_batched = MoE(config, device=None)
        moe_batched.train()
        moe_original = copy.deepcopy(moe_batched)

        torch.manual_seed(999)
        x = torch.randn(32, config.hidden_dim)

        # Reset biases for determinism
        moe_batched.gate.bias.zero_()
        moe_original.gate.bias.zero_()

        # Forward + backward through batched
        out_batched = moe_batched(x.clone())
        loss_batched = out_batched.sum()
        loss_batched.backward()
        grads_batched = {
            name: p.grad.clone()
            for name, p in moe_batched.named_parameters()
            if p.grad is not None
        }

        # Forward + backward through original
        moe_original.expert_counts.zero_()
        moe_original.total_tokens = 0
        out_original = _moe_forward_original(moe_original, x.clone())
        loss_original = out_original.sum()
        loss_original.backward()
        grads_original = {
            name: p.grad.clone()
            for name, p in moe_original.named_parameters()
            if p.grad is not None
        }

        # Every parameter that has a gradient should match
        assert set(grads_batched.keys()) == set(grads_original.keys()), (
            f"Gradient key mismatch: "
            f"batched has {grads_batched.keys() - grads_original.keys()} extra, "
            f"original has {grads_original.keys() - grads_batched.keys()} extra"
        )

        for name in grads_batched:
            torch.testing.assert_close(
                grads_batched[name],
                grads_original[name],
                atol=1e-5, rtol=1e-4,
                msg=f"Gradient mismatch for parameter '{name}'",
            )

    def test_input_gradient_equivalence(self, config):
        """Gradients w.r.t. input tensor match between implementations."""
        torch.manual_seed(42)
        moe_batched = MoE(config, device=None)
        moe_batched.train()
        moe_original = copy.deepcopy(moe_batched)

        torch.manual_seed(999)
        x_batched = torch.randn(32, config.hidden_dim, requires_grad=True)
        x_original = x_batched.detach().clone().requires_grad_(True)

        moe_batched.gate.bias.zero_()
        moe_original.gate.bias.zero_()

        out_batched = moe_batched(x_batched)
        out_batched.sum().backward()

        moe_original.expert_counts.zero_()
        moe_original.total_tokens = 0
        out_original = _moe_forward_original(moe_original, x_original)
        out_original.sum().backward()

        torch.testing.assert_close(
            x_batched.grad,
            x_original.grad,
            atol=1e-5, rtol=1e-4,
            msg="Input gradient mismatch between batched and original",
        )


# ══════════════════════════════════════════════════════════════════════
# Shape and structural tests
# ══════════════════════════════════════════════════════════════════════

class TestShapesAndStructure:
    """Verify output shapes and module structure are correct."""

    def test_output_shape_2d(self, moe, sample_input):
        """Output shape matches input shape for 2D input."""
        out = moe(sample_input)
        assert out.shape == sample_input.shape, (
            f"Expected shape {sample_input.shape}, got {out.shape}"
        )

    def test_output_shape_3d(self, moe, sample_input_3d):
        """Output shape matches input shape for 3D input."""
        out = moe(sample_input_3d)
        assert out.shape == sample_input_3d.shape, (
            f"Expected shape {sample_input_3d.shape}, got {out.shape}"
        )

    def test_output_dtype_preserved(self, moe, sample_input):
        """Output dtype matches input dtype."""
        out = moe(sample_input)
        assert out.dtype == sample_input.dtype

    def test_shared_expert_contribution(self, config):
        """
        Shared expert always contributes to the output,
        even when all routed experts produce zero.
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        # Zero out all routed experts' weights so only shared expert contributes
        with torch.no_grad():
            for expert in moe.experts:
                for param in expert.parameters():
                    param.zero_()

        x = torch.randn(16, config.hidden_dim)
        out = moe(x)

        # Output should not be zero (shared expert is non-zero)
        assert not torch.allclose(out, torch.zeros_like(out)), (
            "Output is all zeros even though shared expert has non-zero weights"
        )


# ══════════════════════════════════════════════════════════════════════
# Expert tracking and metrics tests
# ══════════════════════════════════════════════════════════════════════

class TestExpertTracking:
    """Verify expert counters and utilization metrics work correctly."""

    def test_expert_counts_updated(self, config):
        """Expert counts are updated after forward pass."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(32, config.hidden_dim)
        moe.reset_expert_counts()

        with torch.no_grad():
            moe(x)

        total_assignments = moe.expert_counts.sum().item()
        expected_assignments = 32 * config.num_experts_per_tok  # N * top-k
        assert total_assignments == expected_assignments, (
            f"Expected {expected_assignments} total assignments, got {total_assignments}"
        )

    def test_total_tokens_updated(self, config):
        """total_tokens counter is updated correctly."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()
        moe.reset_expert_counts()

        x = torch.randn(32, config.hidden_dim)
        with torch.no_grad():
            moe(x)

        expected = 32 * config.num_experts_per_tok
        assert moe.total_tokens == expected, (
            f"Expected total_tokens={expected}, got {moe.total_tokens}"
        )

    def test_reset_expert_counts(self, moe, sample_input):
        """reset_expert_counts() zeroes all counters."""
        with torch.no_grad():
            moe(sample_input)

        moe.reset_expert_counts()
        assert moe.expert_counts.sum().item() == 0
        assert moe.total_tokens == 0

    def test_expert_utilization_metrics(self, config):
        """get_expert_utilization() returns valid metrics after forward pass."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()
        moe.reset_expert_counts()

        x = torch.randn(64, config.hidden_dim)
        with torch.no_grad():
            moe(x)

        metrics = moe.get_expert_utilization()
        assert len(metrics) == config.num_experts
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Utilization {key}={val} out of [0,1]"

    def test_wandb_metrics(self, config):
        """get_wandb_metrics() returns correctly structured metrics."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()
        moe.reset_expert_counts()

        x = torch.randn(64, config.hidden_dim)
        with torch.no_grad():
            moe(x)

        metrics = moe.get_wandb_metrics()
        assert "load_balance_score" in metrics
        assert 0.0 <= metrics["load_balance_score"] <= 100.0
        for i in range(config.num_experts):
            assert f"expert_{i}" in metrics


# ══════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases that could break the sort/scatter dispatch logic."""

    def test_single_token(self, config):
        """Single token input works without errors."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(1, config.hidden_dim)
        with torch.no_grad():
            out = moe(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any(), "NaN in output for single token"
        assert not torch.isinf(out).any(), "Inf in output for single token"

    def test_single_token_3d(self, config):
        """Single token in 3D format (1, 1, dim) works."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(1, 1, config.hidden_dim)
        with torch.no_grad():
            out = moe(x)

        assert out.shape == x.shape

    def test_two_tokens(self, config):
        """Two tokens — minimal batch for top-2 routing."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(2, config.hidden_dim)
        with torch.no_grad():
            out = moe(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_all_tokens_same_expert(self, config):
        """
        Force all tokens to be routed to the same expert by heavily
        biasing the gate. The output should still be valid and gradient
        should flow through that expert.
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        # Heavily bias gate toward expert 0
        with torch.no_grad():
            moe.gate.bias.zero_()
            moe.gate.bias[0] = 100.0  # overwhelmingly prefer expert 0

        x = torch.randn(16, config.hidden_dim, requires_grad=True)
        out = moe(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any(), "NaN when all tokens routed to same expert"

        # Gradient should flow
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_large_batch(self, config):
        """Large batch (1024 tokens) processes correctly."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(1024, config.hidden_dim)
        with torch.no_grad():
            out = moe(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ══════════════════════════════════════════════════════════════════════
# Gradient flow tests
# ══════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Verify gradients propagate correctly through the batched dispatch."""

    def test_gradient_flows_to_input(self, config):
        """Gradient flows from loss through MoE back to input tensor."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        x = torch.randn(32, config.hidden_dim, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input tensor"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"
        assert x.grad.abs().sum() > 0, "Input gradient is all zeros"

    def test_gradient_flows_to_all_experts(self, config):
        """
        At least some experts receive non-zero gradients.
        (Not all experts are guaranteed to receive every token, but with
        enough tokens and top-2 routing, most experts should be active.)
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        # Use enough tokens that with 4 experts and top-2 routing,
        # each expert is very likely to receive at least some tokens
        x = torch.randn(128, config.hidden_dim)
        out = moe(x)
        out.sum().backward()

        experts_with_grad = 0
        for i, expert in enumerate(moe.experts):
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            if has_grad:
                experts_with_grad += 1

        # With 128 tokens and top-2 routing, all 4 experts should get tokens.
        # But we allow some slack in case of extreme routing imbalance.
        assert experts_with_grad >= config.num_experts - 1, (
            f"Only {experts_with_grad}/{config.num_experts} experts received gradients"
        )

    def test_gradient_flows_to_shared_expert(self, config):
        """Shared expert always receives gradients."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        x = torch.randn(32, config.hidden_dim)
        out = moe(x)
        out.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in moe.shared_experts.parameters()
        )
        assert has_grad, "Shared expert received no gradients"

    def test_gradient_flows_to_gate(self, config):
        """Gate router parameters receive gradients."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        x = torch.randn(32, config.hidden_dim)
        out = moe(x)
        out.sum().backward()

        gate_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in moe.gate.router.parameters()
        )
        assert gate_has_grad, "Gate router received no gradients"


# ══════════════════════════════════════════════════════════════════════
# Mode and determinism tests
# ══════════════════════════════════════════════════════════════════════

class TestModeAndDeterminism:
    """Verify correct behavior in train/eval mode and determinism."""

    def test_training_vs_eval_mode(self, config):
        """Module produces valid output in both train and eval mode."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)

        x = torch.randn(32, config.hidden_dim)

        moe.train()
        out_train = moe(x.clone())
        assert out_train.shape == x.shape
        assert not torch.isnan(out_train).any()

        moe.eval()
        moe.reset_expert_counts()
        with torch.no_grad():
            out_eval = moe(x.clone())
        assert out_eval.shape == x.shape
        assert not torch.isnan(out_eval).any()

    def test_deterministic_with_seed(self, config):
        """Same seed + same input → identical output (reproducibility)."""
        x = torch.randn(32, config.hidden_dim)

        def run_forward():
            torch.manual_seed(42)
            moe = MoE(config, device=None)
            moe.eval()
            moe.gate.bias.zero_()
            with torch.no_grad():
                return moe(x.clone())

        out1 = run_forward()
        out2 = run_forward()

        torch.testing.assert_close(
            out1, out2,
            atol=0.0, rtol=0.0,
            msg="Non-deterministic output with same seed",
        )

    def test_gate_bias_not_updated_in_eval(self, config):
        """Gate bias should NOT be updated during eval mode."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        initial_bias = moe.gate.bias.clone()
        x = torch.randn(32, config.hidden_dim)
        with torch.no_grad():
            moe(x)

        torch.testing.assert_close(
            moe.gate.bias, initial_bias,
            msg="Gate bias was updated during eval mode",
        )

    def test_gate_bias_updated_in_train(self, config):
        """Gate bias should be updated during train mode (Loss-Free Balancing)."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        initial_bias = moe.gate.bias.clone()
        x = torch.randn(64, config.hidden_dim)
        moe(x)

        # Bias should have changed (unless perfectly balanced, extremely unlikely)
        assert not torch.allclose(moe.gate.bias, initial_bias), (
            "Gate bias was not updated during training"
        )


# ══════════════════════════════════════════════════════════════════════
# Numerical stability tests
# ══════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Tests for numerical robustness under adversarial inputs."""

    def test_no_nan_with_large_inputs(self, config):
        """No NaN/Inf with large but finite input values."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        # Large values that could cause overflow in intermediate computations
        x = torch.randn(32, config.hidden_dim) * 10.0
        with torch.no_grad():
            out = moe(x)

        assert not torch.isnan(out).any(), "NaN in output with large inputs"
        assert not torch.isinf(out).any(), "Inf in output with large inputs"

    def test_no_nan_with_small_inputs(self, config):
        """No NaN/Inf with very small input values."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(32, config.hidden_dim) * 1e-6
        with torch.no_grad():
            out = moe(x)

        assert not torch.isnan(out).any(), "NaN in output with small inputs"
        assert not torch.isinf(out).any(), "Inf in output with small inputs"

    def test_no_nan_with_zero_input(self, config):
        """Zero input produces valid (zero or near-zero) output."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.zeros(32, config.hidden_dim)
        with torch.no_grad():
            out = moe(x)

        assert not torch.isnan(out).any(), "NaN in output with zero input"
        assert not torch.isinf(out).any(), "Inf in output with zero input"

    def test_gradient_stability(self, config):
        """Gradients are finite and reasonable magnitude."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()

        x = torch.randn(32, config.hidden_dim, requires_grad=True)
        out = moe(x)
        out.sum().backward()

        assert not torch.isnan(x.grad).any(), "NaN in gradient"
        assert not torch.isinf(x.grad).any(), "Inf in gradient"

        # Gradients should be reasonable magnitude (not exploding)
        max_grad = x.grad.abs().max().item()
        assert max_grad < 1e6, f"Gradient magnitude too large: {max_grad}"


# ══════════════════════════════════════════════════════════════════════
# Integration-style tests
# ══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests simulating real training scenarios."""

    def test_multiple_forward_backward_passes(self, config):
        """
        Simulate multiple training steps to ensure no state corruption
        between passes (accumulators, gate bias, etc.).
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.train()
        optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

        for step in range(5):
            torch.manual_seed(step)
            x = torch.randn(16, config.hidden_dim)
            out = moe(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should still produce valid output after multiple steps
        x_test = torch.randn(16, config.hidden_dim)
        with torch.no_grad():
            out = moe(x_test)
        assert not torch.isnan(out).any(), "NaN after multiple training steps"
        assert not torch.isinf(out).any(), "Inf after multiple training steps"

    def test_with_autocast_bfloat16(self, config):
        """
        Works correctly under torch.autocast (bfloat16 mixed precision),
        simulating the actual training loop.
        """
        # Skip if no CUDA available (autocast requires GPU for bf16)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bf16 autocast test")

        torch.manual_seed(42)
        moe = MoE(config, device="cuda")
        moe.train()

        x = torch.randn(32, config.hidden_dim, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = moe(x)

        assert not torch.isnan(out).any()
        out.sum().backward()

    def test_state_dict_roundtrip(self, config):
        """
        Model can be saved and loaded via state_dict without affecting
        output. This ensures checkpoint compatibility.
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(32, config.hidden_dim)
        moe.gate.bias.zero_()

        with torch.no_grad():
            out_before = moe(x.clone())

        # Save and reload state dict
        state_dict = moe.state_dict()
        torch.manual_seed(42)
        moe_loaded = MoE(config, device=None)
        moe_loaded.load_state_dict(state_dict)
        moe_loaded.eval()

        with torch.no_grad():
            out_after = moe_loaded(x.clone())

        torch.testing.assert_close(
            out_before, out_after,
            atol=0.0, rtol=0.0,
            msg="Output changed after state_dict save/load roundtrip",
        )


# ══════════════════════════════════════════════════════════════════════
# Specialized dispatch logic tests
# ══════════════════════════════════════════════════════════════════════

class TestDispatchLogic:
    """Directly test the sort-and-scatter dispatch mechanics."""

    def test_scatter_add_correctness(self, config):
        """
        Manually verify that the scatter_add_ produces correct weighted
        combination of expert outputs for a known routing.
        """
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        # Create a small input
        x = torch.randn(4, config.hidden_dim)
        moe.gate.bias.zero_()

        with torch.no_grad():
            # Get gate outputs to understand the routing
            x_flat = x.view(-1, config.hidden_dim)
            weights, indices, counts = moe.gate(x_flat)

            # Manually compute expected output using original method
            routed_out = torch.zeros_like(x_flat)
            for i, expert in enumerate(moe.experts):
                batch_idx, expert_idx = torch.where(indices == i)
                if batch_idx.numel() == 0:
                    continue
                routed_out[batch_idx] += (
                    weights[batch_idx, expert_idx, None] * expert(x_flat[batch_idx])
                )
            expected = routed_out + moe.shared_experts(x_flat)

            # Compare with actual module output
            moe.reset_expert_counts()
            actual = moe(x.clone())

        torch.testing.assert_close(
            actual, expected,
            atol=1e-6, rtol=1e-5,
            msg="Dispatch output doesn't match manual computation",
        )

    def test_top_k_routing_respected(self, config):
        """Each token is dispatched to exactly top-k experts."""
        torch.manual_seed(42)
        moe = MoE(config, device=None)
        moe.eval()

        x = torch.randn(32, config.hidden_dim)
        with torch.no_grad():
            x_flat = x.view(-1, config.hidden_dim)
            _, indices, _ = moe.gate(x_flat)

        # Each token should have exactly num_experts_per_tok assignments
        assert indices.shape == (32, config.num_experts_per_tok), (
            f"Expected indices shape (32, {config.num_experts_per_tok}), "
            f"got {indices.shape}"
        )

        # All indices should be valid expert IDs
        assert (indices >= 0).all() and (indices < config.num_experts).all(), (
            "Expert indices out of valid range"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
