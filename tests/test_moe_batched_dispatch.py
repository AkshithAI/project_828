"""
MoE Batched Expert Dispatch — Production Correctness Suite
==========================================================

Designed to run on the H200 with CUDA + bf16 to validate the batched
dispatch under the EXACT conditions of production training:

  - Real model dimensions  (hidden_dim=768, intermediate_size=760)
  - Real expert config     (4 experts, top-2 routing)
  - Real batch sizes       (micro_batch=37, seq_len=2048 → 75,776 tokens)
  - Real dtype             (bf16 via torch.autocast)
  - torch.compile          (max-autotune-no-cudagraphs)

Every core test compares the NEW batched dispatch against the OLD
sequential loop, running both on the same weights + same input.
"""

import sys
import os
import copy
import math
import types
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# ── Mock flash_attn when not installed (CPU/Mac) ─────────────────────
# On the H200, flash_attn is installed so setdefault() is a no-op.
# On CPU (Mac), we provide a pure-PyTorch fallback.
_mock = types.ModuleType("flash_attn")
def _fa_mock(Q, K, V, causal=False):
    # Q: (B, S, Hq, D), K/V: (B, S, Hkv, D) — need GQA head repeat
    Q_t = Q.transpose(1, 2)   # (B, Hq, S, D)
    K_t = K.transpose(1, 2)   # (B, Hkv, S, D)
    V_t = V.transpose(1, 2)   # (B, Hkv, S, D)
    n_q_heads = Q_t.shape[1]
    n_kv_heads = K_t.shape[1]
    if n_q_heads != n_kv_heads:
        n_rep = n_q_heads // n_kv_heads
        K_t = K_t.repeat_interleave(n_rep, dim=1)  # (B, Hq, S, D)
        V_t = V_t.repeat_interleave(n_rep, dim=1)  # (B, Hq, S, D)
    return F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=causal).transpose(1, 2)
_mock.flash_attn_func = _fa_mock
sys.modules.setdefault("flash_attn", _mock)

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_flash_attn import MoE, GPT_FLASH
from src.scripts.configs.model_config import ModelConfig


# ── SECTION: Reference: the ORIGINAL sequential loop-based dispatch (ground truth) ──

def _moe_forward_original(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """
    Exact replica of the pre-optimization MoE.forward().
    This is the ground truth — if the batched version diverges from
    this, the batched version is WRONG.
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


# ── SECTION: Config factories ──────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _make_config(**overrides) -> ModelConfig:
    """Create a ModelConfig with sensible test defaults."""
    cfg = ModelConfig.__new__(ModelConfig)
    cfg.hidden_dim = overrides.get("hidden_dim", 768)
    cfg.intermediate_size = overrides.get("intermediate_size", 760)
    cfg.num_experts = overrides.get("num_experts", 4)
    cfg.num_experts_per_tok = overrides.get("num_experts_per_tok", 2)
    cfg.update_param = overrides.get("update_param", 1e-3)
    cfg.route_scale = overrides.get("route_scale", 1.0)
    cfg.ffn_dropout = overrides.get("ffn_dropout", 0.0)
    cfg.dtype = overrides.get("dtype", torch.bfloat16)
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


def _production_config() -> ModelConfig:
    """The EXACT config used in production training."""
    return _make_config(
        hidden_dim=768,
        intermediate_size=760,
        num_experts=4,
        num_experts_per_tok=2,
        num_attn_heads=12,
        num_key_value_heads=6,
        num_hidden_layers=24,
        dtype=torch.bfloat16,
    )


def _build_pair(cfg, device=None):
    """Build two identical MoE modules (batched + original reference)."""
    torch.manual_seed(42)
    moe_batched = MoE(cfg, device=device)
    moe_original = copy.deepcopy(moe_batched)
    # Zero gate biases so routing is deterministic from weights alone
    moe_batched.gate.bias.zero_()
    moe_original.gate.bias.zero_()
    return moe_batched, moe_original


# ── SECTION 1: Output equivalence  (batched == original loop) ──────────────────────

class TestOutputEquivalence:
    """
    The most critical tests. If ANY of these fail, the batched dispatch
    is producing wrong results and must not be deployed.
    """

    # ── CPU / fp32 (bit-exact baseline) ──────────────────────────────

    def test_equivalence_fp32_small(self):
        """fp32, 64 tokens — should be essentially bit-exact."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(64, cfg.hidden_dim)

        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-6, rtol=1e-5,
            msg="fp32 small batch: batched != original")

    def test_equivalence_fp32_large(self):
        """fp32, 2048 tokens (one full sequence length)."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(2048, cfg.hidden_dim)

        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-5, rtol=1e-4,
            msg="fp32 full seq_len: batched != original")

    def test_equivalence_fp32_3d(self):
        """fp32, 3D input (batch=4, seq=512, dim=768)."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(4, 512, cfg.hidden_dim)

        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-5, rtol=1e-4,
            msg="fp32 3D input: batched != original")

    # ── CUDA / bf16 (production dtype) ───────────────────────────────

    @requires_cuda
    def test_equivalence_bf16_production_dims(self):
        """bf16, production dims (768/760), 2048 tokens on CUDA."""
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(2048, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        # bf16 has ~0.4% relative error; use atol=1e-2 for safety
        torch.testing.assert_close(out_b, out_o, atol=1e-2, rtol=5e-2,
            msg="bf16 CUDA production dims: batched != original")

    @requires_cuda
    def test_equivalence_bf16_autocast(self):
        """
        bf16 via torch.autocast (exactly how train.py calls it).
        Production dims, 37 × 2048 = 75,776 tokens.
        """
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.eval(); original.eval()

        torch.manual_seed(999)
        # Exact production shape: micro_batch=37, seq_len=2048
        x = torch.randn(37, 2048, cfg.hidden_dim, device="cuda")

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-2, rtol=5e-2,
            msg="bf16 autocast production batch: batched != original")

    # ── Varied expert/top-k configs ──────────────────────────────────

    @pytest.mark.parametrize("num_experts,topk", [
        (2, 1), (4, 2), (6, 3), (8, 2), (8, 4),
    ])
    def test_equivalence_varied_configs(self, num_experts, topk):
        """Equivalence holds across various expert/top-k configurations."""
        cfg = _make_config(
            num_experts=num_experts,
            num_experts_per_tok=topk,
            dtype=torch.float32,
        )
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(256, cfg.hidden_dim)

        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-5, rtol=1e-4,
            msg=f"Mismatch: num_experts={num_experts}, topk={topk}")


# ── SECTION 2: Gradient equivalence (backward pass correctness) ────────

class TestGradientEquivalence:
    """
    If gradients diverge, the model will silently learn wrong things.
    These tests catch subtle issues in scatter_add_ backward, weight
    broadcasting, and gather indexing.
    """

    def test_param_gradients_fp32(self):
        """Every parameter's gradient matches between batched and original."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.train(); original.train()

        torch.manual_seed(999)
        x = torch.randn(128, cfg.hidden_dim)

        # Batched
        out_b = batched(x.clone())
        out_b.sum().backward()
        grads_b = {n: p.grad.clone() for n, p in batched.named_parameters() if p.grad is not None}

        # Original
        original.expert_counts.zero_(); original.total_tokens = 0
        out_o = _moe_forward_original(original, x.clone())
        out_o.sum().backward()
        grads_o = {n: p.grad.clone() for n, p in original.named_parameters() if p.grad is not None}

        assert set(grads_b.keys()) == set(grads_o.keys()), "Gradient key sets differ"

        for name in grads_b:
            torch.testing.assert_close(
                grads_b[name], grads_o[name], atol=1e-5, rtol=1e-4,
                msg=f"Gradient mismatch: {name}",
            )

    def test_input_gradient_fp32(self):
        """Gradient w.r.t. input tensor matches."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.train(); original.train()

        torch.manual_seed(999)
        x_b = torch.randn(128, cfg.hidden_dim, requires_grad=True)
        x_o = x_b.detach().clone().requires_grad_(True)

        batched(x_b).sum().backward()
        original.expert_counts.zero_(); original.total_tokens = 0
        _moe_forward_original(original, x_o).sum().backward()

        torch.testing.assert_close(x_b.grad, x_o.grad, atol=1e-5, rtol=1e-4,
            msg="Input gradient mismatch")

    @requires_cuda
    def test_param_gradients_bf16_cuda(self):
        """Parameter gradients match under bf16 autocast on CUDA."""
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.train(); original.train()

        torch.manual_seed(999)
        x = torch.randn(512, cfg.hidden_dim, device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out_b = batched(x.clone())
        out_b.sum().backward()
        grads_b = {n: p.grad.clone() for n, p in batched.named_parameters() if p.grad is not None}

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())
        out_o.sum().backward()
        grads_o = {n: p.grad.clone() for n, p in original.named_parameters() if p.grad is not None}

        for name in grads_b:
            torch.testing.assert_close(
                grads_b[name], grads_o[name], atol=1e-1, rtol=1e-1,
                msg=f"bf16 gradient mismatch: {name}",
            )

    @requires_cuda
    def test_production_batch_gradients(self):
        """
        Gradient equivalence with the EXACT production shape:
        (37, 2048, 768) → 75,776 tokens under bf16 autocast.
        """
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.train(); original.train()

        torch.manual_seed(999)
        x = torch.randn(37, 2048, cfg.hidden_dim, device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out_b = batched(x.clone())
        out_b.sum().backward()
        grads_b = {n: p.grad.clone() for n, p in batched.named_parameters() if p.grad is not None}

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())
        out_o.sum().backward()
        grads_o = {n: p.grad.clone() for n, p in original.named_parameters() if p.grad is not None}

        for name in grads_b:
            # bf16 accumulation over 75K tokens = larger numerical drift
            torch.testing.assert_close(
                grads_b[name], grads_o[name], atol=5e-1, rtol=2e-1,
                msg=f"Production batch gradient mismatch: {name}",
            )


# ── SECTION 3: torch.compile compatibility ─────────────────────────

class TestTorchCompile:
    """
    torch.compile is CRITICAL to our throughput. If the batched dispatch
    causes graph breaks, we lose the compilation gains.
    """

    @requires_cuda
    def test_compile_forward_backward(self):
        """Compiled MoE produces valid output and gradients."""
        cfg = _make_config(dtype=torch.bfloat16)
        torch.manual_seed(42)
        moe = MoE(cfg, device="cuda")
        moe.train()

        compiled_moe = torch.compile(moe, mode="max-autotune-no-cudagraphs")

        torch.manual_seed(999)
        x = torch.randn(256, cfg.hidden_dim, device="cuda", requires_grad=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = compiled_moe(x)

        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in compiled output"
        assert not torch.isinf(out).any(), "Inf in compiled output"

        out.sum().backward()
        assert x.grad is not None, "No gradient through compiled MoE"
        assert not torch.isnan(x.grad).any(), "NaN in compiled gradient"

    @requires_cuda
    def test_compile_matches_eager(self):
        """Compiled and eager MoE produce the same output."""
        cfg = _make_config(dtype=torch.bfloat16)
        torch.manual_seed(42)
        moe_eager = MoE(cfg, device="cuda")
        moe_eager.eval()
        moe_compiled_raw = copy.deepcopy(moe_eager)
        moe_compiled = torch.compile(moe_compiled_raw, mode="max-autotune-no-cudagraphs")

        moe_eager.gate.bias.zero_()
        moe_compiled_raw.gate.bias.zero_()

        torch.manual_seed(999)
        x = torch.randn(512, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            out_eager = moe_eager(x.clone())
            # Warm-up compile with same shape to avoid recompilation
            _ = moe_compiled(x.clone())
            moe_compiled_raw.expert_counts.zero_()
            moe_compiled_raw.total_tokens = 0
            out_compiled = moe_compiled(x.clone())

        # Triton autotuned kernels can diverge from cuBLAS in bf16;
        # use cosine similarity (robust to per-element kernel differences)
        cos_sim = F.cosine_similarity(
            out_eager.flatten().float(), out_compiled.flatten().float(), dim=0,
        )
        assert cos_sim > 0.999, (
            f"Compiled output diverges from eager: cosine_sim={cos_sim:.6f}"
        )

    @requires_cuda
    def test_compile_multi_step_training(self):
        """
        Run 5 optimizer steps through compiled MoE — verifies no state
        corruption between steps (grad accumulation, gate bias updates, etc.).
        """
        cfg = _make_config(dtype=torch.bfloat16)
        torch.manual_seed(42)
        moe = MoE(cfg, device="cuda")
        moe.train()

        compiled_moe = torch.compile(moe, mode="max-autotune-no-cudagraphs")
        optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)

        losses = []
        for step in range(5):
            optimizer.zero_grad()
            torch.manual_seed(step + 100)
            x = torch.randn(64, cfg.hidden_dim, device="cuda")

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out = compiled_moe(x)

            loss = out.sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All losses should be finite
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"Step {i}: loss = {l} (not finite)"


# ── SECTION 4: Edge cases (sort/scatter failure modes) ────────────────────────

class TestEdgeCases:

    def test_single_token(self):
        """1 token — minimal possible dispatch."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        x = torch.randn(1, cfg.hidden_dim)
        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-6, rtol=1e-5)
        assert not torch.isnan(out_b).any()

    def test_two_tokens(self):
        """2 tokens — with top-2 routing, every token goes to 2 experts."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        x = torch.randn(2, cfg.hidden_dim)
        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-6, rtol=1e-5)

    def test_all_tokens_to_one_expert(self):
        """
        Force all tokens to expert 0 by biasing the gate.
        Experts 1-3 get empty slices — searchsorted boundaries must
        handle start==end without crashing.
        """
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        # Overwhelmingly bias expert 0
        with torch.no_grad():
            batched.gate.bias[0] = 100.0
            original.gate.bias[0] = 100.0

        x = torch.randn(64, cfg.hidden_dim)
        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-5, rtol=1e-4,
            msg="All-to-one-expert: batched != original")

    def test_all_tokens_to_one_expert_backward(self):
        """Gradient flows correctly when all tokens hit one expert."""
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.train(); original.train()

        with torch.no_grad():
            batched.gate.bias[0] = 100.0
            original.gate.bias[0] = 100.0

        x_b = torch.randn(64, cfg.hidden_dim, requires_grad=True)
        x_o = x_b.detach().clone().requires_grad_(True)

        batched(x_b).sum().backward()
        original.expert_counts.zero_(); original.total_tokens = 0
        _moe_forward_original(original, x_o).sum().backward()

        torch.testing.assert_close(x_b.grad, x_o.grad, atol=1e-5, rtol=1e-4,
            msg="All-to-one-expert gradient mismatch")

    def test_top1_routing(self):
        """top-1 routing — each token goes to exactly 1 expert."""
        cfg = _make_config(num_experts=4, num_experts_per_tok=1, dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.eval(); original.eval()

        torch.manual_seed(999)
        x = torch.randn(256, cfg.hidden_dim)
        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-5, rtol=1e-4)

    @requires_cuda
    def test_single_token_cuda_bf16(self):
        """Single token on CUDA bf16 — boundary of empty expert slices."""
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.eval(); original.eval()

        x = torch.randn(1, cfg.hidden_dim, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out_b = batched(x.clone())
            original.expert_counts.zero_(); original.total_tokens = 0
            out_o = _moe_forward_original(original, x.clone())

        torch.testing.assert_close(out_b, out_o, atol=1e-2, rtol=5e-2)


# ── SECTION 5: Multi-step training simulation ─────────────────────────

class TestTrainingSimulation:
    """
    Simulate actual training loops to catch issues that only appear
    after multiple steps (state accumulation bugs, gate bias drift, etc.).
    """

    def test_loss_trajectory_matches(self):
        """
        Run 10 optimizer steps with both implementations on identical data.
        Loss trajectories must match — if they diverge, the batched dispatch
        is computing something subtly different.
        """
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.train(); original.train()

        opt_b = torch.optim.AdamW(batched.parameters(), lr=1e-3)
        opt_o = torch.optim.AdamW(original.parameters(), lr=1e-3)

        losses_b, losses_o = [], []

        for step in range(10):
            torch.manual_seed(step + 1000)
            x = torch.randn(64, cfg.hidden_dim)

            # Batched
            opt_b.zero_grad()
            batched.reset_expert_counts()
            out_b = batched(x.clone())
            loss_b = out_b.pow(2).mean()
            loss_b.backward()
            opt_b.step()
            losses_b.append(loss_b.item())

            # Original
            opt_o.zero_grad()
            original.reset_expert_counts()
            out_o = _moe_forward_original(original, x.clone())
            loss_o = out_o.pow(2).mean()
            loss_o.backward()
            opt_o.step()
            losses_o.append(loss_o.item())

        # Losses should match at every step
        for step_i, (lb, lo) in enumerate(zip(losses_b, losses_o)):
            assert abs(lb - lo) < 1e-4, (
                f"Step {step_i}: loss_batched={lb:.6f} vs loss_original={lo:.6f} "
                f"(diff={abs(lb-lo):.2e})"
            )

    @requires_cuda
    def test_loss_trajectory_bf16_cuda(self):
        """
        Same as above but under bf16 autocast on CUDA — the actual
        training conditions.
        """
        cfg = _make_config(dtype=torch.bfloat16)
        batched, original = _build_pair(cfg, device="cuda")
        batched.train(); original.train()

        opt_b = torch.optim.AdamW(batched.parameters(), lr=1e-3)
        opt_o = torch.optim.AdamW(original.parameters(), lr=1e-3)

        losses_b, losses_o = [], []

        for step in range(10):
            torch.manual_seed(step + 1000)
            x = torch.randn(64, cfg.hidden_dim, device="cuda")

            # Batched
            opt_b.zero_grad()
            batched.reset_expert_counts()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out_b = batched(x.clone())
                loss_b = out_b.float().pow(2).mean()
            loss_b.backward()
            opt_b.step()
            losses_b.append(loss_b.item())

            # Original
            opt_o.zero_grad()
            original.reset_expert_counts()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out_o = _moe_forward_original(original, x.clone())
                loss_o = out_o.float().pow(2).mean()
            loss_o.backward()
            opt_o.step()
            losses_o.append(loss_o.item())

        for step_i, (lb, lo) in enumerate(zip(losses_b, losses_o)):
            rel_diff = abs(lb - lo) / max(abs(lo), 1e-10)
            assert rel_diff < 0.01, (
                f"Step {step_i}: loss_batched={lb:.6f} vs loss_original={lo:.6f} "
                f"(rel_diff={rel_diff:.2e})"
            )

    def test_gate_bias_evolution_matches(self):
        """
        After N forward passes in train mode, gate biases should evolve
        identically between batched and original implementations.
        """
        cfg = _make_config(dtype=torch.float32)
        batched, original = _build_pair(cfg)
        batched.train(); original.train()

        for step in range(20):
            torch.manual_seed(step + 500)
            x = torch.randn(64, cfg.hidden_dim)

            batched.reset_expert_counts()
            original.reset_expert_counts()

            with torch.no_grad():
                batched(x.clone())
                _moe_forward_original(original, x.clone())

        torch.testing.assert_close(
            batched.gate.bias, original.gate.bias,
            atol=1e-6, rtol=1e-5,
            msg="Gate bias diverged between batched and original after 20 steps",
        )


# ── SECTION 6: Full model integration ─────────────────────────

class TestFullModelIntegration:
    """
    Test the batched dispatch through the complete GPT_FLASH model,
    not just the isolated MoE module.
    """

    @requires_cuda
    def test_full_model_forward_backward(self):
        """
        Full GPT_FLASH forward + backward with 2 layers.
        Checks the MoE works correctly when composed with attention,
        norms, embeddings, and unembedding.
        """
        cfg = _make_config(num_hidden_layers=2, dtype=torch.bfloat16)
        torch.manual_seed(42)
        model = GPT_FLASH(cfg, device="cuda")
        model.train()

        # (batch=4, seq=128) — small but representative
        input_ids = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)

        assert logits.shape == (4, 128, cfg.vocab_size), f"Logits shape: {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in full model logits"

        # Cross-entropy loss + backward
        targets = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")
        loss = nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), targets.view(-1),
        )
        loss.backward()

        assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"

        # Check gradients exist on MoE params
        for layer in model.layers:
            for expert in layer.mlp.experts:
                assert any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in expert.parameters()
                ), "Expert received no gradients through full model"

    @requires_cuda
    def test_full_model_compiled(self):
        """Full compiled GPT_FLASH forward+backward — the ultimate test."""
        cfg = _make_config(num_hidden_layers=2, dtype=torch.bfloat16)
        torch.manual_seed(42)
        model = GPT_FLASH(cfg, device="cuda")
        model.train()

        compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")

        input_ids = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")
        targets = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = compiled_model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), targets.view(-1),
            )

        loss.backward()
        assert math.isfinite(loss.item()), f"Compiled model loss: {loss.item()}"


# ── SECTION 7: Expert counting & checkpoint compatibility ─────────

class TestBookkeeping:

    def test_expert_counts_correct(self):
        """Total expert assignments = N_tokens × top_k."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()
        moe.reset_expert_counts()

        N = 256
        x = torch.randn(N, cfg.hidden_dim)
        with torch.no_grad():
            moe(x)

        total = moe.expert_counts.sum().item()
        expected = N * cfg.num_experts_per_tok
        assert total == expected, f"Expected {expected} assignments, got {total}"

    def test_state_dict_roundtrip(self):
        """Save → load state dict, output must be identical."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()
        moe.gate.bias.zero_()

        x = torch.randn(64, cfg.hidden_dim)
        with torch.no_grad():
            out_before = moe(x.clone())

        state = moe.state_dict()
        torch.manual_seed(42)
        moe2 = MoE(cfg)
        moe2.load_state_dict(state)
        moe2.eval()

        with torch.no_grad():
            out_after = moe2(x.clone())

        torch.testing.assert_close(out_before, out_after, atol=0.0, rtol=0.0,
            msg="Output changed after state_dict roundtrip")

    def test_wandb_metrics_structure(self):
        """get_wandb_metrics returns expected keys and valid ranges."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()
        moe.reset_expert_counts()

        with torch.no_grad():
            moe(torch.randn(128, cfg.hidden_dim))

        metrics = moe.get_wandb_metrics()
        assert "load_balance_score" in metrics
        assert 0 <= metrics["load_balance_score"] <= 100
        for i in range(cfg.num_experts):
            assert f"expert_{i}" in metrics
            assert 0 <= metrics[f"expert_{i}"] <= 100


# ── SECTION 8: Numerical stability under adversarial inputs ───────────

class TestNumericalStability:

    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 10.0, 100.0])
    def test_varied_input_scales(self, scale):
        """No NaN/Inf across a wide range of input magnitudes."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()

        x = torch.randn(64, cfg.hidden_dim) * scale
        with torch.no_grad():
            out = moe(x)

        assert not torch.isnan(out).any(), f"NaN at scale {scale}"
        assert not torch.isinf(out).any(), f"Inf at scale {scale}"

    def test_zero_input(self):
        """Zero input → no NaN."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.eval()

        with torch.no_grad():
            out = moe(torch.zeros(64, cfg.hidden_dim))

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not explode through the dispatch."""
        cfg = _make_config(dtype=torch.float32)
        torch.manual_seed(42)
        moe = MoE(cfg)
        moe.train()

        x = torch.randn(128, cfg.hidden_dim, requires_grad=True)
        moe(x).sum().backward()

        max_grad = x.grad.abs().max().item()
        assert max_grad < 1e4, f"Input gradient exploded: max={max_grad}"

        for name, p in moe.named_parameters():
            if p.grad is not None:
                pg_max = p.grad.abs().max().item()
                assert pg_max < 1e6, f"Param {name} gradient exploded: max={pg_max}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
