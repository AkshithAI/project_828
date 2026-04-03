"""
Fused Cross-Entropy Correctness Tests
======================================

Validates that ``LigerFusedLinearCrossEntropyLoss`` produces numerically
equivalent results to the standard ``nn.CrossEntropyLoss`` + linear
unembedding path used during validation.

**Why this matters:**
The fused kernel never materializes the full (batch*seq, vocab_size) logits
tensor — it computes cross-entropy chunk-by-chunk inside a Triton kernel.
Any bug in that Triton code could silently produce wrong loss values or
wrong gradients, degrading model quality without any obvious error.

**Run on H200:**
    python tests/test_fused_cross_entropy.py

All tests will auto-skip on non-CUDA environments.
"""

import sys
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# ═══════════════════════════════════════════════════════════════
#  CUDA / liger-kernel availability checks
# ═══════════════════════════════════════════════════════════════

CUDA_AVAILABLE = torch.cuda.is_available()
LIGER_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        LIGER_AVAILABLE = True
    except ImportError:
        pass

# Model import — only needed for end-to-end test (needs flash_attn on CUDA)
MODEL_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
        from src.models.model_flash_attn import GPT_FLASH
        from src.scripts.configs.model_config import ModelConfig
        from src.scripts.tokenizer import tokenizer
        MODEL_AVAILABLE = True
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════
#  Test infrastructure
# ═══════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.error = None
        self.duration = 0.0

    def __repr__(self):
        if self.skipped:
            status = "⏭️  SKIP"
        elif self.passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        return f"{status} [{self.duration:.3f}s] {self.name}"


def run_test(name: str, fn, requires_cuda=True, requires_liger=True, requires_model=False):
    r = TestResult(name)
    if requires_cuda and not CUDA_AVAILABLE:
        r.skipped = True
        print(f"⏭️  SKIP [no CUDA] {name}")
        return r
    if requires_liger and not LIGER_AVAILABLE:
        r.skipped = True
        print(f"⏭️  SKIP [no liger-kernel] {name}")
        return r
    if requires_model and not MODEL_AVAILABLE:
        r.skipped = True
        print(f"⏭️  SKIP [no model/flash_attn] {name}")
        return r
    t0 = time.perf_counter()
    try:
        fn()
        r.passed = True
    except Exception as e:
        r.error = e
        r.passed = False
    r.duration = time.perf_counter() - t0
    print(r)
    if r.error:
        traceback.print_exception(type(r.error), r.error, r.error.__traceback__)
    return r


# ═══════════════════════════════════════════════════════════════
#  Helper: Reference (standard) cross-entropy path
# ═══════════════════════════════════════════════════════════════

def standard_ce_loss(hidden, weight, targets, ignore_index=-100, bias=None):
    """
    Standard path: linear projection → CrossEntropyLoss.

    Args:
        hidden:  (N, D) float tensor
        weight:  (V, D) float tensor — unembedding weight matrix
        targets: (N,)   long tensor
        ignore_index: token id to ignore in loss
        bias:    optional (V,) bias vector

    Returns:
        (loss, grad_hidden, grad_weight)
    """
    hidden = hidden.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    b = None
    if bias is not None:
        b = bias.clone().detach().requires_grad_(True)

    logits = F.linear(hidden, weight, b)         # (N, V)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = criterion(logits, targets)
    loss.backward()

    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


def fused_ce_loss(hidden, weight, targets, ignore_index=-100, bias=None):
    """
    Fused path: LigerFusedLinearCrossEntropyLoss.

    Same interface as standard_ce_loss for direct comparison.
    """
    hidden = hidden.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    b = None
    if bias is not None:
        b = bias.clone().detach().requires_grad_(True)

    fused = LigerFusedLinearCrossEntropyLoss(
        ignore_index=ignore_index, reduction="mean"
    )
    loss = fused(weight, hidden, targets)
    loss.backward()

    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


# ═══════════════════════════════════════════════════════════════
#  Production-scale constants (matching model_config.py)
# ═══════════════════════════════════════════════════════════════

HIDDEN_DIM = 768
VOCAB_SIZE = 49152
SEQ_LEN = 2047       # max_context_len - 1 (training input length)


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════

def test_forward_loss_equivalence():
    """
    Core test: fused CE loss value must match standard CE loss value.

    Tests across multiple batch sizes including the production micro_batch_size=36.
    Uses small sequence length for speed; scale test covers production dimensions.
    """
    torch.manual_seed(42)
    D, V = HIDDEN_DIM, VOCAB_SIZE

    for batch_label, N in [("single", 1), ("small_batch", 4), ("production", 36)]:
        # Use shorter seq_len for fast tests; production scale test covers full size
        T = 64
        total_tokens = N * T

        hidden = torch.randn(total_tokens, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
        targets = torch.randint(0, V, (total_tokens,), device="cuda")

        std_loss, _, _ = standard_ce_loss(hidden, weight, targets)
        fsd_loss, _, _ = fused_ce_loss(hidden, weight, targets)

        diff = (std_loss - fsd_loss).abs().item()
        rel_diff = diff / (std_loss.abs().item() + 1e-8)

        assert torch.allclose(std_loss, fsd_loss, atol=1e-4, rtol=1e-3), (
            f"[{batch_label}] Loss mismatch: standard={std_loss.item():.6f}, "
            f"fused={fsd_loss.item():.6f}, abs_diff={diff:.2e}, rel_diff={rel_diff:.2e}"
        )
        print(f"    → [{batch_label}] N={total_tokens}: "
              f"std={std_loss.item():.6f}, fused={fsd_loss.item():.6f}, "
              f"abs_diff={diff:.2e}")


def test_backward_gradient_equivalence():
    """
    Gradient on hidden states must match between fused and standard paths.

    This is critical — if gradients diverge, the optimizer will update weights
    differently, causing silent model quality degradation over thousands of steps.
    """
    torch.manual_seed(123)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 128  # reasonable batch for gradient test

    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(0, V, (N,), device="cuda")

    _, std_grad_h, _ = standard_ce_loss(hidden, weight, targets)
    _, fsd_grad_h, _ = fused_ce_loss(hidden, weight, targets)

    # Per-element comparison with tolerance
    max_abs_diff = (std_grad_h - fsd_grad_h).abs().max().item()
    mean_abs_diff = (std_grad_h - fsd_grad_h).abs().mean().item()
    cos_sim = F.cosine_similarity(
        std_grad_h.flatten().unsqueeze(0),
        fsd_grad_h.flatten().unsqueeze(0),
    ).item()

    assert torch.allclose(std_grad_h, fsd_grad_h, atol=1e-4, rtol=1e-3), (
        f"Hidden gradient mismatch: max_diff={max_abs_diff:.2e}, "
        f"mean_diff={mean_abs_diff:.2e}, cos_sim={cos_sim:.6f}"
    )
    assert cos_sim > 0.9999, (
        f"Gradient direction diverged: cosine_similarity={cos_sim:.6f}"
    )

    print(f"    → Hidden grad: max_diff={max_abs_diff:.2e}, "
          f"mean_diff={mean_abs_diff:.2e}, cosine_sim={cos_sim:.6f}")


def test_unembedding_weight_gradient():
    """
    Gradient on the unembedding weight matrix must match.

    The fused kernel computes dL/dW internally (never forming the full logits
    tensor), so this gradient path is entirely different from standard PyTorch.
    """
    torch.manual_seed(456)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 128

    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(0, V, (N,), device="cuda")

    _, _, std_grad_w = standard_ce_loss(hidden, weight, targets)
    _, _, fsd_grad_w = fused_ce_loss(hidden, weight, targets)

    max_abs_diff = (std_grad_w - fsd_grad_w).abs().max().item()
    mean_abs_diff = (std_grad_w - fsd_grad_w).abs().mean().item()
    cos_sim = F.cosine_similarity(
        std_grad_w.flatten().unsqueeze(0),
        fsd_grad_w.flatten().unsqueeze(0),
    ).item()

    assert torch.allclose(std_grad_w, fsd_grad_w, atol=1e-4, rtol=1e-3), (
        f"Weight gradient mismatch: max_diff={max_abs_diff:.2e}, "
        f"mean_diff={mean_abs_diff:.2e}, cos_sim={cos_sim:.6f}"
    )
    assert cos_sim > 0.9999, (
        f"Weight gradient direction diverged: cosine_similarity={cos_sim:.6f}"
    )

    print(f"    → Weight grad: max_diff={max_abs_diff:.2e}, "
          f"mean_diff={mean_abs_diff:.2e}, cosine_sim={cos_sim:.6f}")


def test_ignore_index_handling():
    """
    Tokens with ignore_index must be excluded from loss computation.

    Our training uses ignore_index=eos_token_id. If the fused kernel doesn't
    handle this correctly, EOS tokens contribute to the loss, biasing gradients.
    """
    torch.manual_seed(789)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 256
    IGNORE_IDX = 0  # use 0 as a stand-in for eos_token_id

    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(1, V, (N,), device="cuda")  # no ignore tokens initially

    # Sprinkle ~25% ignore tokens
    ignore_mask = torch.rand(N, device="cuda") < 0.25
    targets[ignore_mask] = IGNORE_IDX
    n_ignored = ignore_mask.sum().item()

    std_loss, std_grad_h, _ = standard_ce_loss(hidden, weight, targets, ignore_index=IGNORE_IDX)
    fsd_loss, fsd_grad_h, _ = fused_ce_loss(hidden, weight, targets, ignore_index=IGNORE_IDX)

    diff = (std_loss - fsd_loss).abs().item()
    assert torch.allclose(std_loss, fsd_loss, atol=1e-4, rtol=1e-3), (
        f"ignore_index loss mismatch: std={std_loss.item():.6f}, "
        f"fused={fsd_loss.item():.6f}, diff={diff:.2e}"
    )

    # Gradients for ignored positions should be zero (or near-zero)
    # in both paths — verify fused handles this
    grad_diff_max = (std_grad_h - fsd_grad_h).abs().max().item()
    assert torch.allclose(std_grad_h, fsd_grad_h, atol=1e-4, rtol=1e-3), (
        f"ignore_index gradient mismatch: max_diff={grad_diff_max:.2e}"
    )

    print(f"    → {n_ignored}/{N} tokens ignored, loss_diff={diff:.2e}, "
          f"grad_max_diff={grad_diff_max:.2e}")


def test_all_ignored():
    """
    Edge case: all targets are ignore_index.

    Standard CE returns 0.0 loss with no gradients. Fused kernel must match.
    """
    torch.manual_seed(101)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 64
    IGNORE_IDX = 0

    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.full((N,), IGNORE_IDX, device="cuda", dtype=torch.long)

    std_loss, std_grad_h, _ = standard_ce_loss(hidden, weight, targets, ignore_index=IGNORE_IDX)
    fsd_loss, fsd_grad_h, _ = fused_ce_loss(hidden, weight, targets, ignore_index=IGNORE_IDX)

    assert std_loss.item() == 0.0, f"Standard CE should be 0.0, got {std_loss.item()}"
    assert fsd_loss.item() == 0.0, f"Fused CE should be 0.0, got {fsd_loss.item()}"

    assert torch.all(std_grad_h == 0), "Standard grad should be all zeros"
    assert torch.all(fsd_grad_h == 0), "Fused grad should be all zeros"

    print(f"    → All-ignored: both losses = 0.0, both grads = 0.0 ✓")


def test_numerical_stability_at_scale():
    """
    Test with production dimensions to catch overflow/underflow issues.

    Uses hidden_dim=768, vocab_size=49152, batch_size×seq_len=36×128.
    The full production size (36×2047) would require ~28 GB for logits alone
    in the standard path, so we use a reduced seq_len but full hidden/vocab dims.
    """
    torch.manual_seed(2024)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    B, T = 36, 128  # production batch, reduced seq for memory
    N = B * T

    # Initialize weight like real unembedding: small values
    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32) * 0.1
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(0, V, (N,), device="cuda")

    std_loss, _, _ = standard_ce_loss(hidden, weight, targets)
    fsd_loss, _, _ = fused_ce_loss(hidden, weight, targets)

    diff = (std_loss - fsd_loss).abs().item()
    rel_diff = diff / (std_loss.abs().item() + 1e-8)

    assert torch.allclose(std_loss, fsd_loss, atol=1e-4, rtol=1e-3), (
        f"Scale test loss mismatch: std={std_loss.item():.6f}, "
        f"fused={fsd_loss.item():.6f}, abs_diff={diff:.2e}"
    )

    # Check loss is in a reasonable range (not NaN/Inf)
    assert not torch.isnan(fsd_loss), "Fused loss is NaN!"
    assert not torch.isinf(fsd_loss), "Fused loss is Inf!"
    assert fsd_loss.item() > 0, "Loss should be positive"

    print(f"    → Scale ({B}×{T}={N} tokens, V={V}): "
          f"std={std_loss.item():.4f}, fused={fsd_loss.item():.4f}, "
          f"rel_diff={rel_diff:.2e}")


def test_bf16_autocast_precision():
    """
    Test under bf16 autocast — matching the actual training configuration.

    bf16 has lower mantissa precision than fp32, so we use looser tolerances.
    The key property is that the fused kernel and standard path produce the
    same result under the same precision regime.
    """
    torch.manual_seed(555)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 256

    # Create inputs in bf16 (as they would be with autocast)
    hidden_base = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    weight_base = torch.randn(V, D, device="cuda", dtype=torch.bfloat16) * 0.02
    targets = torch.randint(0, V, (N,), device="cuda")

    # Standard path under autocast
    hidden_std = hidden_base.float().clone().detach().requires_grad_(True)
    weight_std = weight_base.float().clone().detach().requires_grad_(True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = F.linear(hidden_std, weight_std)
        std_loss = nn.CrossEntropyLoss()(logits, targets)
    std_loss.backward()

    # Fused path under autocast
    hidden_fsd = hidden_base.float().clone().detach().requires_grad_(True)
    weight_fsd = weight_base.float().clone().detach().requires_grad_(True)
    fused = LigerFusedLinearCrossEntropyLoss(reduction="mean")
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        fsd_loss = fused(weight_fsd, hidden_fsd, targets)
    fsd_loss.backward()

    diff = (std_loss.detach() - fsd_loss.detach()).abs().item()
    # bf16 requires looser tolerance
    assert torch.allclose(std_loss.detach(), fsd_loss.detach(), atol=5e-3, rtol=5e-3), (
        f"bf16 loss mismatch: std={std_loss.item():.6f}, "
        f"fused={fsd_loss.item():.6f}, diff={diff:.2e}"
    )

    # Gradient direction check (cosine similarity)
    cos_sim = F.cosine_similarity(
        hidden_std.grad.flatten().unsqueeze(0),
        hidden_fsd.grad.flatten().unsqueeze(0),
    ).item()
    assert cos_sim > 0.999, (
        f"bf16 gradient direction diverged: cosine_sim={cos_sim:.6f}"
    )

    print(f"    → bf16: loss_diff={diff:.2e}, grad_cos_sim={cos_sim:.6f}")


def test_determinism():
    """
    Same inputs must produce exactly the same outputs across runs.

    Non-deterministic behavior would make training reproducibility impossible
    and debugging extremely difficult.
    """
    torch.manual_seed(999)
    D, V = HIDDEN_DIM, VOCAB_SIZE
    N = 128

    hidden = torch.randn(N, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02
    targets = torch.randint(0, V, (N,), device="cuda")

    results = []
    for run in range(5):
        loss, grad_h, grad_w = fused_ce_loss(hidden, weight, targets)
        results.append((loss.item(), grad_h.clone(), grad_w.clone()))

    # All runs must produce identical results
    ref_loss, ref_grad_h, ref_grad_w = results[0]
    for i, (loss, grad_h, grad_w) in enumerate(results[1:], 1):
        assert loss == ref_loss, (
            f"Run {i}: loss {loss} != reference {ref_loss}"
        )
        assert torch.equal(grad_h, ref_grad_h), (
            f"Run {i}: hidden gradient differs from reference"
        )
        assert torch.equal(grad_w, ref_grad_w), (
            f"Run {i}: weight gradient differs from reference"
        )

    print(f"    → 5 runs identical: loss={ref_loss:.6f}")


def test_end_to_end_model_integration():
    """
    End-to-end test using the actual GPT_FLASH model.

    Compares:
      Path A (standard): model.forward(x) → logits → nn.CrossEntropyLoss
      Path B (fused):    model.forward_with_hidden(x) → LigerFused(weight, hidden, targets)

    This validates the actual code path used in train.py.
    """
    torch.manual_seed(42)

    # Use a tiny config to save memory
    cfg = ModelConfig()
    # Override to make test feasible on any GPU
    cfg.num_hidden_layers = 2
    cfg.max_context_len = 128
    cfg.initial_context_len = 128

    device = "cuda"
    model = GPT_FLASH(cfg, device)
    model.eval()  # deterministic behavior

    B, T = 4, 127  # batch_size, seq_len (context - 1)
    inputs = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    eos_id = tokenizer.eos_token_id

    # ── Path A: Standard ──
    model.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        std_criterion = nn.CrossEntropyLoss(ignore_index=eos_id)
        std_loss = std_criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
    std_loss.backward()
    std_grad_snapshot = {
        name: p.grad.clone() for name, p in model.named_parameters()
        if p.grad is not None
    }

    # ── Path B: Fused ──
    model.zero_grad()
    fused_criterion = LigerFusedLinearCrossEntropyLoss(
        ignore_index=eos_id, reduction="mean"
    )
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = model.forward_with_hidden(inputs)
        hidden_flat = hidden.view(-1, hidden.shape[-1])
        targets_flat = targets.view(-1)
        fsd_loss = fused_criterion(model.unembedding.weight, hidden_flat, targets_flat)
    fsd_loss.backward()
    fsd_grad_snapshot = {
        name: p.grad.clone() for name, p in model.named_parameters()
        if p.grad is not None
    }

    # ── Compare losses ──
    loss_diff = (std_loss.detach() - fsd_loss.detach()).abs().item()
    assert torch.allclose(std_loss.detach(), fsd_loss.detach(), atol=5e-3, rtol=5e-3), (
        f"E2E loss mismatch: std={std_loss.item():.6f}, "
        f"fused={fsd_loss.item():.6f}, diff={loss_diff:.2e}"
    )

    # ── Compare gradients on all parameters ──
    mismatched_params = []
    for name in std_grad_snapshot:
        if name not in fsd_grad_snapshot:
            mismatched_params.append(f"{name}: missing in fused path")
            continue

        std_g = std_grad_snapshot[name]
        fsd_g = fsd_grad_snapshot[name]

        cos_sim = F.cosine_similarity(
            std_g.flatten().float().unsqueeze(0),
            fsd_g.flatten().float().unsqueeze(0),
        ).item()

        if cos_sim < 0.99:
            mismatched_params.append(
                f"{name}: cosine_sim={cos_sim:.4f}"
            )

    assert len(mismatched_params) == 0, (
        f"Gradient mismatches in {len(mismatched_params)} parameters:\n"
        + "\n".join(f"  - {m}" for m in mismatched_params)
    )

    # Report a few representative gradient similarities
    sample_params = list(std_grad_snapshot.keys())[:5]
    for name in sample_params:
        cos_sim = F.cosine_similarity(
            std_grad_snapshot[name].flatten().float().unsqueeze(0),
            fsd_grad_snapshot[name].flatten().float().unsqueeze(0),
        ).item()
        print(f"    → {name}: cos_sim={cos_sim:.6f}")

    print(f"    → E2E loss_diff={loss_diff:.2e}, all {len(std_grad_snapshot)} "
          f"parameter gradients match")

    del model
    torch.cuda.empty_cache()


def test_gradient_flow_through_training_step():
    """
    Simulate a complete training step: forward → fused CE → backward → optimizer.step().
    
    Verifies that the fused path produces valid parameter updates (no NaN/Inf in weights
    after optimization), matching the behavior of the standard path.
    """
    torch.manual_seed(77)

    D, V = 256, 1024  # smaller dims for speed
    N = 64

    # Create a simple linear model standing in for the unembedding
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(D, D, bias=False)
            self.unembed = nn.Linear(D, V, bias=False)

        def forward_hidden(self, x):
            return self.proj(x)

        def forward(self, x):
            return self.unembed(self.proj(x))

    # ── Standard path ──
    torch.manual_seed(77)
    model_std = MiniModel().cuda()
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3)
    inputs = torch.randn(N, D, device="cuda")
    targets = torch.randint(0, V, (N,), device="cuda")

    opt_std.zero_grad()
    logits = model_std(inputs)
    loss_std = nn.CrossEntropyLoss()(logits, targets)
    loss_std.backward()
    opt_std.step()
    w_std_after = model_std.proj.weight.detach().clone()

    # ── Fused path ──
    torch.manual_seed(77)
    model_fsd = MiniModel().cuda()
    opt_fsd = torch.optim.AdamW(model_fsd.parameters(), lr=1e-3)

    opt_fsd.zero_grad()
    hidden = model_fsd.forward_hidden(inputs)
    fused = LigerFusedLinearCrossEntropyLoss(reduction="mean")
    loss_fsd = fused(model_fsd.unembed.weight, hidden, targets)
    loss_fsd.backward()
    opt_fsd.step()
    w_fsd_after = model_fsd.proj.weight.detach().clone()

    # Weights should be nearly identical after one optimizer step
    assert not torch.isnan(w_fsd_after).any(), "NaN in fused weights after optim step"
    assert not torch.isinf(w_fsd_after).any(), "Inf in fused weights after optim step"

    weight_diff = (w_std_after - w_fsd_after).abs().max().item()
    cos_sim = F.cosine_similarity(
        w_std_after.flatten().unsqueeze(0),
        w_fsd_after.flatten().unsqueeze(0),
    ).item()

    assert cos_sim > 0.999, (
        f"Weight divergence after optim step: cos_sim={cos_sim:.6f}"
    )

    print(f"    → Post-optim weight max_diff={weight_diff:.2e}, cos_sim={cos_sim:.6f}")


# ═══════════════════════════════════════════════════════════════
#  Main runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Fused Cross-Entropy Correctness — Test Suite")
    print("=" * 70)
    if not CUDA_AVAILABLE:
        print("  ⚠️  CUDA not available — all tests will be skipped.")
        print("  Run this on the H200 training node.")
    elif not LIGER_AVAILABLE:
        print("  ⚠️  liger-kernel not installed — all tests will be skipped.")
        print("  Install: pip install liger-kernel>=0.5.0")
    print()

    tests = [
        # (name, function, requires_cuda, requires_liger, requires_model)
        ("Forward loss equivalence",         test_forward_loss_equivalence,         True, True, False),
        ("Backward gradient equivalence",    test_backward_gradient_equivalence,    True, True, False),
        ("Unembedding weight gradient",      test_unembedding_weight_gradient,      True, True, False),
        ("Ignore-index (EOS) handling",      test_ignore_index_handling,            True, True, False),
        ("All-ignored edge case",            test_all_ignored,                      True, True, False),
        ("Numerical stability at scale",     test_numerical_stability_at_scale,     True, True, False),
        ("bf16 autocast precision",          test_bf16_autocast_precision,          True, True, False),
        ("Determinism (5 runs)",             test_determinism,                      True, True, False),
        ("Gradient flow through optim step", test_gradient_flow_through_training_step, True, True, False),
        ("End-to-end model integration",     test_end_to_end_model_integration,     True, True, True),
    ]

    results = []
    for name, fn, req_cuda, req_liger, req_model in tests:
        r = run_test(name, fn, req_cuda, req_liger, req_model)
        results.append(r)

    # ── Summary ──
    print()
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    total_time = sum(r.duration for r in results)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped "
          f"({total_time:.2f}s total)")

    if failed > 0:
        print()
        print("  FAILED TESTS:")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"    ❌ {r.name}: {r.error}")

    print("=" * 70)

    if not CUDA_AVAILABLE or not LIGER_AVAILABLE:
        print("\n  ℹ️  Re-run on H200 with liger-kernel installed for full results.\n")
        sys.exit(0)

    sys.exit(1 if failed > 0 else 0)
