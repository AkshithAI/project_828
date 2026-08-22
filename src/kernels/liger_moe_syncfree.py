"""Sync-free routing metadata for liger_kernel's fused MoE.

WHY THIS EXISTS
===============
LigerFusedMoEFunction.compute_routing_metadata() performs exactly one
`.item()` per MoE layer per forward pass:

    num_m_tiles = int(expert_tile_offset[-1].item())

That single line forces a full pipeline drain (cudaMemcpyAsync DtoH +
cudaStreamSynchronize) in the middle of an otherwise async forward pass.
At 24 MoE layers x 16 microbatches/step that is 384 hidden drains per
optimizer step — visible in nsys as a comb of red memcpy blocks inside
forward_and_loss, and confirmed by torch.cuda.set_sync_debug_mode.

HOW THIS FIXES IT
=================
`num_m_tiles` is only needed host-side for buffer sizing and grid dims.
Instead of reading the exact value from device, we compute a safe upper
bound and make every padding tile provably inert:

    total M-tiles = sum_e ceil(count_e / BLOCK_M_TOKEN),  sum_e count_e = T*K
    ceil(c/BM) <= c/BM + 1  =>  total <= cdiv(T*K, BM) + E

Padding tiles are initialised with `tile_row_start = TK` (sentinel) and
`tile_expert = 0`. Every GEMM/backward kernel computes
`row_mask = (row_start + m_offs) < expert_end` where expert_end <= TK,
so padding tiles see row_mask == False everywhere:
  * all loads return zeros (masked),
  * all stores are masked out,
  * the dS atomic_add in _moe_bwd_down_proj_kernel is masked out
    (this is why the sentinel must be >= TK — a zero-init would instead
    re-execute real tile 0 and double-count its dS contributions).

Downstream, LigerFusedMoEFunction.forward derives num_m_tiles from
`tile_row_start.shape[0]`, so patching ONLY this function propagates the
upper bound through forward grids and ctx.num_m_tiles in backward with no
further changes.

Overhead: up to E-1 extra no-op tiles out of ~cdiv(T*K, 64) real ones
(<1% grid inflation at our shapes).
"""

import torch

try:
    import triton
    from liger_kernel.ops import fused_moe as _liger_fused_moe
    from liger_kernel.ops.fused_moe_kernels import (
        _moe_router_histogram_kernel,
        _moe_router_prefix_sum_kernel,
        _moe_router_scatter_kernel,
    )
    _LIGER_IMPORT_OK = True
except ImportError:
    _LIGER_IMPORT_OK = False


def compute_routing_metadata_sync_free(
    topk_indices: torch.Tensor,
    E: int,
    block_m_token: int = None,
):
    """Drop-in replacement for liger_kernel.ops.fused_moe.compute_routing_metadata.

    Identical outputs for all real tiles; tile_row_start/tile_expert are
    allocated at a safe upper bound instead of the exact count, with
    sentinel values in padding slots that render those tiles inert.
    No device->host synchronization anywhere.
    """
    if block_m_token is None:
        block_m_token = _liger_fused_moe.BLOCK_M_TOKEN

    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # ── Kernel 1: tiled histogram (identical to liger) ──
    tile_expert_counts = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _moe_router_histogram_kernel[(n_tiles,)](
        topk_indices,
        tile_expert_counts,
        T,
        E=E,
        n_tiles=n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_token_count = tile_expert_counts.sum(dim=1, dtype=torch.int32)

    # ── Kernel 2: prefix sums + offsets (identical to liger) ──
    expert_start_idx = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    _moe_router_prefix_sum_kernel[(E + 2,)](
        expert_token_count,
        expert_start_idx,
        expert_tile_offset,
        E=E,
        partial_sum_ptr=tile_expert_counts,
        n_tiles=n_tiles,
        TK=TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
        BLOCK_M_TOKEN=block_m_token,
    )

    # ── SYNC-FREE: upper-bound tile count (replaces expert_tile_offset[-1].item()) ──
    # Padding tiles carry sentinel row_start = TK so their row_mask is empty.
    max_m_tiles = triton.cdiv(TK, block_m_token) + E
    tile_row_start = torch.full((max_m_tiles,), TK, dtype=torch.int32, device=device)
    tile_expert = torch.zeros(max_m_tiles, dtype=torch.int32, device=device)

    # ── Kernel 3: sort + scatter + tile metadata (identical to liger) ──
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            T,
            tile_expert_counts,
            n_tiles,
            expert_start_idx[:E],
            expert_tile_offset[:E],
            K_POW2=K_POW2,
            K=K,
            TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
            BLOCK_M_TOKEN=block_m_token,
        )

    return (
        expert_token_count,
        expert_start_idx,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        tile_row_start,
        tile_expert,
    )


_applied = False
_original_fn = None


def apply_sync_free_routing_metadata(verbose: bool = True) -> bool:
    """Monkey-patch liger_kernel.ops.fused_moe.compute_routing_metadata.

    Idempotent. Returns True if the patch is (already) active.
    """
    global _applied, _original_fn
    if not _LIGER_IMPORT_OK:
        raise ImportError("liger_kernel / triton unavailable — nothing to patch")
    if _applied:
        return True
    # Stash the original so tests can A/B against it (and restore afterwards).
    if _original_fn is None:
        _original_fn = _liger_fused_moe.compute_routing_metadata
    _liger_fused_moe.compute_routing_metadata = compute_routing_metadata_sync_free
    _applied = True
    if verbose:
        print("[liger_sync_free] Patched compute_routing_metadata: "
              "MoE routing metadata is now synchronization-free.")
    return True


def restore_original_routing_metadata() -> bool:
    """Undo the patch (used by correctness tests to A/B against upstream)."""
    global _applied
    if not _LIGER_IMPORT_OK or _original_fn is None:
        return False
    _liger_fused_moe.compute_routing_metadata = _original_fn
    _applied = False
    return True
