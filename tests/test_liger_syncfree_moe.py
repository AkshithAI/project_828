"""Sync-free liger MoE routing metadata — A/B correctness suite.

The sync-free patch (src/kernels/liger_moe_syncfree.py) replaces liger's
`compute_routing_metadata` with an upper-bound variant that removes the
per-layer `.item()` pipeline drain. These tests verify it is numerically
identical to upstream liger for forward AND backward, including skewed
routing where some experts receive zero tokens.

Run on a CUDA machine:  pytest tests/test_liger_syncfree_moe.py -v
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# ── Mock flash_attn when not installed (CPU/Mac) ─────────────────────
_mock = types.ModuleType("flash_attn")
sys.modules.setdefault("flash_attn", _mock)

# ── Stub triton ONLY for module import; removed before real use ──────
_triton_stub = types.ModuleType("triton")
_triton_stub.jit = lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f)
_triton_stub.autotune = lambda configs, key: (lambda f: f)
_triton_stub.Config = MagicMock()
_triton_stub.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
_triton_stub.cdiv = lambda a, b: -(-a // b)
sys.modules["triton"] = _triton_stub
sys.modules["triton.language"] = MagicMock()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.models.model_adv import LigerFusedMoEFunction, RoutedExperts  # noqa: F401
    from src.kernels.liger_moe_syncfree import (
        apply_sync_free_routing_metadata,
        restore_original_routing_metadata,
        _LIGER_IMPORT_OK,
    )
    _HAS_LIGER = _LIGER_IMPORT_OK and LigerFusedMoEFunction is not None
except ImportError:
    _HAS_LIGER = False

# Undo the triton stub so transformers / other libs see the truth.
sys.modules.pop("triton", None)
sys.modules.pop("triton.language", None)

requires_liger_cuda = pytest.mark.skipif(
    not (_HAS_LIGER and torch.cuda.is_available()),
    reason="CUDA + liger_kernel required",
)

BF16_ATOL = BF16_RTOL = 2e-2


def _make_inputs(T=256, H=128, I=64, E=8, K=2, seed=0, skew=False):
    g = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn(T, H, generator=g).to("cuda", torch.bfloat16)
    gate_up = (torch.randn(E, 2 * I, H, generator=g) * 0.1).to("cuda", torch.bfloat16)
    down = (torch.randn(E, H, I, generator=g) * 0.1).to("cuda", torch.bfloat16)

    if skew:
        idx = torch.randint(0, 3, (T, K), generator=g)  # experts 3..7 stay empty
    else:
        idx = torch.randint(0, E, (T, K), generator=g)
    idx = idx.to("cuda", dtype=torch.int32)

    w = torch.rand(T, K, generator=g).to("cuda", torch.float32)
    w = w / w.sum(dim=-1, keepdim=True)

    dO = torch.randn(T, H, generator=g).to("cuda", torch.bfloat16)
    return x, gate_up, down, idx, w, dO


def _run(x, gate_up, down, idx, w, dO):
    x = x.detach().clone().requires_grad_(True)
    gate_up = gate_up.detach().clone().requires_grad_(True)
    down = down.detach().clone().requires_grad_(True)
    out = LigerFusedMoEFunction.apply(x, gate_up, down, idx, w)
    out.backward(dO)
    return out.detach().clone(), x.grad.clone(), gate_up.grad.clone(), down.grad.clone()


class TestSyncFreeMoEPatching:
    @classmethod
    def setup_class(cls):
        # Ensure we start from UPSTREAM liger regardless of earlier imports.
        restore_original_routing_metadata()

    @classmethod
    def teardown_class(cls):
        restore_original_routing_metadata()

    @requires_liger_cuda
    @pytest.mark.parametrize("seed,skew", [(42, False), (7, True)])
    def test_patched_matches_upstream(self, seed, skew):
        inputs = _make_inputs(seed=seed, skew=skew)

        restore_original_routing_metadata()
        reference = _run(*inputs)

        assert apply_sync_free_routing_metadata(verbose=False)
        patched = _run(*inputs)

        names = ["out", "dx", "d_gate_up_proj", "d_down_proj"]
        for name, ref_t, pat_t in zip(names, reference, patched):
            torch.testing.assert_close(
                pat_t.float(), ref_t.float(),
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"{name} mismatch (seed={seed}, skew={skew})",
            )
