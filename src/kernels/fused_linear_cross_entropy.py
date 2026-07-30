import torch
import triton
import triton.language as tl
import torch.nn.functional as F

import argparse
import gc
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Optional
from .utils import amp_custom_fwd, amp_custom_bwd


LOG2_E = tl.constexpr(1.4426950408889634)

def choose_chunk_size(
    num_tokens: int,
    vocab_size: int,
    dtype: torch.dtype,
    workspace_bytes: int,
    alignment: int = 128,
) -> int:
    """
    Select the largest token chunk that fits in the requested logit workspace.

    Workspace usage is approximately:

        chunk_size * vocab_size * element_size(dtype)

    A large chunk is intentionally preferred because every additional chunk
    causes another accumulation into the complete [V, D] FP32 weight gradient.
    """
    if num_tokens == 0:
        return 0

    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_token = vocab_size * element_size

    max_chunk = max(1, workspace_bytes // bytes_per_token)
    max_chunk = min(num_tokens, max_chunk)

    if max_chunk >= alignment:
        max_chunk = (max_chunk // alignment) * alignment

    return max(1, max_chunk)


def choose_ce_block_size(vocab_size: int) -> int:
    """
    Conservative streamed vocabulary tile.

    This should ultimately be replaced by architecture-specific autotuning.
    Avoid constructing a 32K-element vector inside every Triton program.
    """
    if vocab_size <= 2048:
        return triton.next_power_of_2(vocab_size)
    if vocab_size <= 32768:
        return 4096
    return 8192


def ce_num_warps(block_size: int) -> int:
    if block_size <= 2048:
        return 4
    return 8


@triton.jit
def _cross_entropy_fwd_inplace_kernel(
    logits_ptr,
    logits_row_stride,
    target_ptr,
    target_stride,
    loss_ptr,
    loss_stride,
    n_non_ignore_ptr,
    n_cols,
    ignore_index,
    WRITE_GRADIENTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Each program owns one token row.

    Pass 1:
        Compute online log-sum-exp.

    Pass 2:
        Optionally overwrite logits with normalized dL/dlogits.

    The vocabulary is streamed in moderate BLOCK_V tiles to avoid the register
    pressure of a next-power-of-two block spanning the complete vocabulary.
    """
    row = tl.program_id(axis=0)

    row_logits_ptr = logits_ptr + row * logits_row_stride
    row_target_ptr = target_ptr + row * target_stride
    row_loss_ptr = loss_ptr + row * loss_stride

    target = tl.load(row_target_ptr)

    if target == ignore_index:
        tl.store(row_loss_ptr, 0.0)

        if WRITE_GRADIENTS:
            for col_start in range(0, n_cols, BLOCK_V):
                cols = col_start + tl.arange(0, BLOCK_V)
                mask = cols < n_cols

                tl.store(
                    row_logits_ptr + cols,
                    0.0,
                    mask=mask,
                )
        return

    running_max = float("-inf")
    running_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols

        logits = tl.load(
            row_logits_ptr + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        tile_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, tile_max)

        old_scale = tl.exp2((running_max - new_max) * LOG2_E)
        tile_sum = tl.sum(
            tl.exp2((logits - new_max) * LOG2_E),
            axis=0,
        )

        running_sum = running_sum * old_scale + tile_sum
        running_max = new_max

    lse = running_max + tl.log(running_sum)
    target_logit = tl.load(row_logits_ptr + target).to(tl.float32)

    tl.store(row_loss_ptr, lse - target_logit)

    if WRITE_GRADIENTS:
        n_non_ignore = tl.load(n_non_ignore_ptr).to(tl.float32)
        inv_n = 1.0 / n_non_ignore

        for col_start in range(0, n_cols, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols

            logits = tl.load(
                row_logits_ptr + cols,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)

            probabilities = tl.exp2((logits - lse) * LOG2_E)
            target_delta = tl.where(cols == target, 1.0, 0.0)

            grad_logits = (probabilities - target_delta) * inv_n

            tl.store(
                row_logits_ptr + cols,
                grad_logits,
                mask=mask,
            )


_DW_CONFIGS = [
    triton.Config(
        {
            "BLOCK_V": 32,
            "BLOCK_D": 64,
            "BLOCK_K": 32,
            "GROUP_V": 8,
        },
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_V": 64,
            "BLOCK_D": 64,
            "BLOCK_K": 32,
            "GROUP_V": 8,
        },
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_V": 64,
            "BLOCK_D": 128,
            "BLOCK_K": 32,
            "GROUP_V": 8,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_V": 128,
            "BLOCK_D": 64,
            "BLOCK_K": 32,
            "GROUP_V": 8,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_V": 64,
            "BLOCK_D": 128,
            "BLOCK_K": 64,
            "GROUP_V": 8,
        },
        num_warps=8,
        num_stages=4,
    ),
]


@triton.autotune(
    configs=_DW_CONFIGS,
    key=["V", "D", "K"],
)
@triton.jit
def _grad_weight_kernel(
    grad_logits_ptr,
    hidden_ptr,
    grad_weight_ptr,
    stride_gk,
    stride_gv,
    stride_xk,
    stride_xd,
    stride_wv,
    stride_wd,
    V: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_V: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_d = tl.cdiv(D, BLOCK_D)

    # Grouped program ordering improves reuse of the token/hidden-state tiles.
    programs_per_group = GROUP_V * num_pid_d
    group_id = pid // programs_per_group

    first_pid_v = group_id * GROUP_V
    group_size_v = tl.minimum(num_pid_v - first_pid_v, GROUP_V)

    pid_in_group = pid % programs_per_group
    pid_v = first_pid_v + (pid_in_group % group_size_v)
    pid_d = pid_in_group // group_size_v

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # grad_logits has physical layout [K, V], but is consumed as [V, K].
    grad_ptrs = (
        grad_logits_ptr
        + offs_v[:, None] * stride_gv
        + offs_k[None, :] * stride_gk
    )

    hidden_ptrs = (
        hidden_ptr
        + offs_k[:, None] * stride_xk
        + offs_d[None, :] * stride_xd
    )

    accumulator = tl.zeros(
        (BLOCK_V, BLOCK_D),
        dtype=tl.float32,
    )

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k

        grad_tile = tl.load(
            grad_ptrs,
            mask=(
                (offs_v[:, None] < V)
                & (k_offsets[None, :] < K)
            ),
            other=0.0,
        )

        hidden_tile = tl.load(
            hidden_ptrs,
            mask=(
                (k_offsets[:, None] < K)
                & (offs_d[None, :] < D)
            ),
            other=0.0,
        )

        accumulator += tl.dot(grad_tile, hidden_tile)

        grad_ptrs += BLOCK_K * stride_gk
        hidden_ptrs += BLOCK_K * stride_xk

    output_ptrs = (
        grad_weight_ptr
        + offs_v[:, None] * stride_wv
        + offs_d[None, :] * stride_wd
    )

    output_mask = (
        (offs_v[:, None] < V)
        & (offs_d[None, :] < D)
    )

    if ACCUMULATE:
        previous = tl.load(
            output_ptrs,
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)

        accumulator += previous

    tl.store(
        output_ptrs,
        accumulator,
        mask=output_mask,
    )


def accumulate_grad_weight(
    grad_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    grad_weight_fp32: torch.Tensor,
    accumulate: bool,
):
    K, V = grad_logits.shape
    _, D = hidden_states.shape

    grid = lambda meta: (
        triton.cdiv(V, meta["BLOCK_V"])
        * triton.cdiv(D, meta["BLOCK_D"]),
    )

    _grad_weight_kernel[grid](
        grad_logits,
        hidden_states,
        grad_weight_fp32,
        grad_logits.stride(0),
        grad_logits.stride(1),
        hidden_states.stride(0),
        hidden_states.stride(1),
        grad_weight_fp32.stride(0),
        grad_weight_fp32.stride(1),
        V=V,
        D=D,
        K=K,
        ACCUMULATE=accumulate,
    )


@triton.jit
def _scale_inplace_kernel(
    input_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    scale = tl.load(scale_ptr).to(tl.float32)
    values = tl.load(input_ptr + offsets, mask=mask)

    tl.store(
        input_ptr + offsets,
        values * scale,
        mask=mask,
    )


def scale_inplace(tensor: Optional[torch.Tensor], scale: torch.Tensor):
    if tensor is None or tensor.numel() == 0:
        return

    n_elements = tensor.numel()
    block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    _scale_inplace_kernel[grid](
        tensor,
        scale,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

def normalize_non_ignore_count(
    target: torch.Tensor,
    ignore_index: int,
    total_n_non_ignore=None,
):
    """
    Return a device-resident scalar.

    No .item() call is permitted in the hot path.
    """
    if total_n_non_ignore is None:
        return torch.sum(
            target != ignore_index,
            dtype=torch.int32,
        )

    if isinstance(total_n_non_ignore, torch.Tensor):
        return total_n_non_ignore.to(
            device=target.device,
            dtype=torch.int32,
        )

    return torch.tensor(
        total_n_non_ignore,
        device=target.device,
        dtype=torch.int32,
    )


def fused_linear_cross_entropy_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    total_n_non_ignore=None,
    workspace_bytes: int = 512 * 1024 * 1024,
):
    """
    Production-oriented chunked fused linear cross entropy.

    The implementation computes unit-upstream gradients during forward.
    Backward only applies grad_output.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must have shape [T, D], got {hidden_states.shape}"
        )

    if weight.ndim != 2:
        raise ValueError(
            f"weight must have shape [V, D], got {weight.shape}"
        )

    if target.ndim != 1:
        raise ValueError(
            f"target must have shape [T], got {target.shape}"
        )

    T, D = hidden_states.shape
    V, weight_dim = weight.shape

    if weight_dim != D:
        raise ValueError(
            f"Hidden dimension mismatch: X has D={D}, W has D={weight_dim}"
        )

    if target.shape[0] != T:
        raise ValueError(
            f"Token count mismatch: X has T={T}, target has {target.shape[0]}"
        )

    if hidden_states.device != weight.device:
        raise ValueError("hidden_states and weight must be on the same device")

    if hidden_states.device != target.device:
        raise ValueError("hidden_states and target must be on the same device")

    if hidden_states.dtype != weight.dtype:
        raise ValueError(
            "The optimized path requires hidden_states and weight "
            "to have the same dtype"
        )

    if hidden_states.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            "The optimized path currently supports FP16 and BF16 inputs"
        )

    if hidden_states.stride(1) != 1:
        hidden_states = hidden_states.contiguous()

    if weight.stride(1) != 1:
        weight = weight.contiguous()

    if target.stride(0) != 1:
        target = target.contiguous()

    need_grad_input = hidden_states.requires_grad
    need_grad_weight = weight.requires_grad
    need_grad_logits = need_grad_input or need_grad_weight

    n_non_ignore = normalize_non_ignore_count(
        target,
        ignore_index,
        total_n_non_ignore,
    )

    chunk_size = choose_chunk_size(
        num_tokens=T,
        vocab_size=V,
        dtype=hidden_states.dtype,
        workspace_bytes=workspace_bytes,
    )

    num_chunks = triton.cdiv(T, chunk_size)

    # Reusable logit/grad-logit workspace.
    logits_workspace = torch.empty(
        (chunk_size, V),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    loss_1d = torch.empty(
        T,
        device=hidden_states.device,
        dtype=torch.float32,
    )

    grad_input = None
    if need_grad_input:
        grad_input = torch.empty_like(hidden_states)

    grad_weight_fp32 = None
    if need_grad_weight:
        grad_weight_fp32 = torch.empty(
            (V, D),
            device=weight.device,
            dtype=torch.float32,
        )

    block_v = choose_ce_block_size(V)
    num_warps = ce_num_warps(block_v)

    wrote_grad_weight = False

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, T)
        current_chunk_size = chunk_end - chunk_start

        hidden_chunk = hidden_states[chunk_start:chunk_end]
        target_chunk = target[chunk_start:chunk_end]
        loss_chunk = loss_1d[chunk_start:chunk_end]

        logits_chunk = logits_workspace[:current_chunk_size]

        # Backend GEMM writes directly into reusable workspace.
        torch.mm(
            hidden_chunk,
            weight.t(),
            out=logits_chunk,
        )

        _cross_entropy_fwd_inplace_kernel[
            (current_chunk_size,)
        ](
            logits_chunk,
            logits_chunk.stride(0),
            target_chunk,
            target_chunk.stride(0),
            loss_chunk,
            loss_chunk.stride(0),
            n_non_ignore,
            V,
            ignore_index,
            WRITE_GRADIENTS=need_grad_logits,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )

        # logits_chunk now contains normalized dL/dlogits.
        if need_grad_input:
            torch.mm(
                logits_chunk,
                weight,
                out=grad_input[chunk_start:chunk_end],
            )

        if need_grad_weight:
            accumulate_grad_weight(
                grad_logits=logits_chunk,
                hidden_states=hidden_chunk,
                grad_weight_fp32=grad_weight_fp32,
                accumulate=wrote_grad_weight,
            )
            wrote_grad_weight = True

    # Keep the loss in FP32.
    #
    # clamp_min handles the all-ignored case without host synchronization.
    safe_n_non_ignore = n_non_ignore.clamp_min(1).to(torch.float32)
    loss = loss_1d.sum(dtype=torch.float32) / safe_n_non_ignore

    grad_weight = None
    if need_grad_weight:
        # PyTorch parameter gradients conventionally match parameter dtype.
        grad_weight = grad_weight_fp32.to(weight.dtype)

    return loss, grad_input, grad_weight


def fused_linear_cross_entropy_backward(
    grad_output: torch.Tensor,
    grad_input: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
):
    """
    Apply the scalar upstream derivative.

    No torch.equal(), .item(), or Python branch on a GPU value.
    """
    if grad_output is None:
        return grad_input, grad_weight

    scale_inplace(grad_input, grad_output)
    scale_inplace(grad_weight, grad_output)

    return grad_input, grad_weight


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        hidden_states,
        weight,
        target,
        ignore_index=-100,
        total_n_non_ignore=None,
        workspace_bytes=512 * 1024 * 1024,
    ):
        loss, grad_input, grad_weight = fused_linear_cross_entropy_forward(
            hidden_states=hidden_states,
            weight=weight,
            target=target,
            ignore_index=ignore_index,
            total_n_non_ignore=total_n_non_ignore,
            workspace_bytes=workspace_bytes,
        )

        ctx.has_grad_input = grad_input is not None
        ctx.has_grad_weight = grad_weight is not None

        # save_for_backward is kept tensor-only. Empty placeholders represent
        # gradients that were not requested.
        grad_input_saved = (
            grad_input.detach()
            if grad_input is not None
            else hidden_states.new_empty(0)
        )

        grad_weight_saved = (
            grad_weight.detach()
            if grad_weight is not None
            else weight.new_empty(0)
        )

        ctx.save_for_backward(
            grad_input_saved,
            grad_weight_saved,
        )

        # This implementation deliberately does not support double backward.
        ctx.set_materialize_grads(False)

        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input_saved, grad_weight_saved = ctx.saved_tensors

        grad_input = (
            grad_input_saved
            if ctx.has_grad_input
            else None
        )

        grad_weight = (
            grad_weight_saved
            if ctx.has_grad_weight
            else None
        )

        grad_input, grad_weight = fused_linear_cross_entropy_backward(
            grad_output,
            grad_input,
            grad_weight,
        )

        return (
            grad_input,   # hidden_states
            grad_weight,  # weight
            None,         # target
            None,         # ignore_index
            None,         # total_n_non_ignore
            None,         # workspace_bytes
        )


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    total_n_non_ignore=None,
    workspace_bytes: int = 512 * 1024 * 1024,
):
    return FusedLinearCrossEntropyFunction.apply(
        hidden_states,
        weight,
        target,
        ignore_index,
        total_n_non_ignore,
        workspace_bytes,
    )


# ------ Testing script --------

@dataclass
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    relative_l2: float
    cosine_similarity: float


@dataclass
class BenchmarkResult:
    name: str
    median_ms: float
    mean_ms: float
    min_ms: float
    std_ms: float
    tokens_per_second: float
    approximate_tflops: float
    peak_allocated_gib: float
    incremental_peak_gib: float


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dtype {name!r}. "
            f"Choose from: {sorted(mapping)}"
        ) from error


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda_memory():
    synchronize()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def tensor_error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> ErrorMetrics:
    actual_fp32 = actual.detach().float().reshape(-1)
    reference_fp32 = reference.detach().float().reshape(-1)

    difference = actual_fp32 - reference_fp32
    absolute_difference = difference.abs()

    max_abs = absolute_difference.max().item()
    mean_abs = absolute_difference.mean().item()

    reference_norm = reference_fp32.norm()
    difference_norm = difference.norm()

    relative_l2 = (
        difference_norm / reference_norm.clamp_min(1e-12)
    ).item()

    actual_norm = actual_fp32.norm()

    if actual_norm.item() == 0.0 and reference_norm.item() == 0.0:
        cosine_similarity = 1.0
    elif actual_norm.item() == 0.0 or reference_norm.item() == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = F.cosine_similarity(
            actual_fp32,
            reference_fp32,
            dim=0,
            eps=1e-12,
        ).item()

    return ErrorMetrics(
        max_abs=max_abs,
        mean_abs=mean_abs,
        relative_l2=relative_l2,
        cosine_similarity=cosine_similarity,
    )


def print_error_metrics(
    name: str,
    metrics: ErrorMetrics,
):
    print(
        f"{name:<16}"
        f" max_abs={metrics.max_abs:10.4e}"
        f" mean_abs={metrics.mean_abs:10.4e}"
        f" relative_l2={metrics.relative_l2:10.4e}"
        f" cosine={metrics.cosine_similarity:12.9f}"
    )


def tensor_bytes(shape, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return math.prod(shape) * element_size


def naive_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """
    PyTorch reference matching the fused kernel's intended numerical path:

        1. Linear projection in model dtype.
        2. Cross entropy and log-sum-exp in FP32.
        3. Gradient through the FP32 conversion back into model dtype.

    Casting logits to FP32 before cross-entropy makes the CE reduction
    numerically comparable to the fused kernel.
    """
    logits = hidden_states @ weight.t()

    return F.cross_entropy(
        logits.float(),
        target,
        ignore_index=ignore_index,
        reduction="mean",
    )


def custom_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    n_non_ignore: torch.Tensor,
    workspace_bytes: int,
) -> torch.Tensor:
    return fused_linear_cross_entropy(
        hidden_states=hidden_states,
        weight=weight,
        target=target,
        ignore_index=ignore_index,
        total_n_non_ignore=n_non_ignore,
        workspace_bytes=workspace_bytes,
    )


def create_inputs(
    num_tokens: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    ignore_fraction: float,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # A moderate standard deviation avoids constructing unrealistically
    # saturated initial logits.
    hidden_states = torch.randn(
        (num_tokens, hidden_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ) / math.sqrt(hidden_dim)

    weight = torch.randn(
        (vocab_size, hidden_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    target = torch.randint(
        low=0,
        high=vocab_size,
        size=(num_tokens,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )

    if ignore_fraction > 0.0:
        ignored = torch.rand(
            num_tokens,
            device=device,
            generator=generator,
        ) < ignore_fraction

        target[ignored] = -100

        # Ensure at least one valid token.
        target[0] = torch.randint(
            low=0,
            high=vocab_size,
            size=(),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

    return hidden_states, weight, target


def run_correctness_test(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    workspace_bytes: int,
    upstream_scale: float,
    loss_atol: float,
    loss_rtol: float,
    grad_atol: float,
    grad_rtol: float,
):
    print("\n" + "=" * 80)
    print("Correctness test")
    print("=" * 80)

    # Keep the valid count resident on the GPU.
    n_non_ignore = torch.sum(
        target != ignore_index,
        dtype=torch.int32,
    )

    naive_hidden = (
        hidden_states.detach()
        .clone()
        .requires_grad_(True)
    )
    naive_weight = (
        weight.detach()
        .clone()
        .requires_grad_(True)
    )

    custom_hidden = (
        hidden_states.detach()
        .clone()
        .requires_grad_(True)
    )
    custom_weight = (
        weight.detach()
        .clone()
        .requires_grad_(True)
    )

    naive_loss = naive_linear_cross_entropy(
        naive_hidden,
        naive_weight,
        target,
        ignore_index,
    )

    # Non-unit scale validates custom backward's grad_output handling.
    (naive_loss * upstream_scale).backward()

    custom_loss = custom_linear_cross_entropy(
        custom_hidden,
        custom_weight,
        target,
        ignore_index,
        n_non_ignore,
        workspace_bytes,
    )

    (custom_loss * upstream_scale).backward()

    synchronize()

    loss_difference = abs(
        custom_loss.float().item() - naive_loss.float().item()
    )

    print(f"Naive loss:       {naive_loss.float().item():.10f}")
    print(f"Custom loss:      {custom_loss.float().item():.10f}")
    print(f"Absolute error:   {loss_difference:.6e}")
    print(f"Upstream scale:   {upstream_scale}")

    hidden_metrics = tensor_error_metrics(
        custom_hidden.grad,
        naive_hidden.grad,
    )

    weight_metrics = tensor_error_metrics(
        custom_weight.grad,
        naive_weight.grad,
    )

    print()
    print_error_metrics("grad_hidden", hidden_metrics)
    print_error_metrics("grad_weight", weight_metrics)

    torch.testing.assert_close(
        custom_loss.float(),
        naive_loss.float(),
        atol=loss_atol,
        rtol=loss_rtol,
        msg="Loss mismatch",
    )

    torch.testing.assert_close(
        custom_hidden.grad.float(),
        naive_hidden.grad.float(),
        atol=grad_atol,
        rtol=grad_rtol,
        msg="Hidden-state gradient mismatch",
    )

    torch.testing.assert_close(
        custom_weight.grad.float(),
        naive_weight.grad.float(),
        atol=grad_atol,
        rtol=grad_rtol,
        msg="Weight gradient mismatch",
    )

    print("\nCorrectness: PASS")

    del naive_hidden
    del naive_weight
    del custom_hidden
    del custom_weight
    del naive_loss
    del custom_loss

    clear_cuda_memory()


def benchmark_cuda_function(
    name: str,
    function: Callable,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    num_tokens: int,
    hidden_dim: int,
    vocab_size: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-event benchmarking requires a CUDA or ROCm device"
        )

    
    # Warmup
    for _ in range(warmup):
        hidden_states.grad = None
        weight.grad = None

        loss = function()
        loss.backward()

    synchronize()

    hidden_states.grad = None
    weight.grad = None
    loss = None

    synchronize()
    gc.collect()

    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    timings_ms = []

    for _ in range(iterations):
        hidden_states.grad = None
        weight.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        loss = function()
        loss.backward()

        end_event.record()
        end_event.synchronize()

        timings_ms.append(start_event.elapsed_time(end_event))

    synchronize()

    peak_memory = torch.cuda.max_memory_allocated()
    incremental_peak = max(0, peak_memory - baseline_memory)

    median_ms = statistics.median(timings_ms)
    mean_ms = statistics.mean(timings_ms)
    min_ms = min(timings_ms)
    std_ms = (
        statistics.stdev(timings_ms)
        if len(timings_ms) > 1
        else 0.0
    )

    tokens_per_second = num_tokens / (median_ms / 1000.0)

    # Approximate conventional linear-head training FLOPs:
    #
    #   forward projection: 2*T*D*V
    #   input gradient:     2*T*D*V
    #   weight gradient:    2*T*D*V
    #
    # Cross-entropy elementwise FLOPs are not included.
    approximate_flops = 6.0 * num_tokens * hidden_dim * vocab_size

    approximate_tflops = (
        approximate_flops / (median_ms / 1000.0) / 1e12
    )

    return BenchmarkResult(
        name=name,
        median_ms=median_ms,
        mean_ms=mean_ms,
        min_ms=min_ms,
        std_ms=std_ms,
        tokens_per_second=tokens_per_second,
        approximate_tflops=approximate_tflops,
        peak_allocated_gib=gib(peak_memory),
        incremental_peak_gib=gib(incremental_peak),
    )


def print_benchmark_result(result: BenchmarkResult):
    print(
        f"{result.name:<20}"
        f" median={result.median_ms:9.3f} ms"
        f" mean={result.mean_ms:9.3f} ms"
        f" min={result.min_ms:9.3f} ms"
        f" std={result.std_ms:8.3f} ms"
    )

    print(
        f"{'':20}"
        f" tokens/s={result.tokens_per_second:12,.0f}"
        f" approximate={result.approximate_tflops:9.2f} TFLOP/s"
        f" peak={result.peak_allocated_gib:7.3f} GiB"
        f" incremental={result.incremental_peak_gib:7.3f} GiB"
    )


def run_performance_test(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    workspace_bytes: int,
    upstream_scale: float,
    warmup: int,
    iterations: int,
):
    print("\n" + "=" * 80)
    print("Performance test: forward + backward")
    print("=" * 80)

    T, D = hidden_states.shape
    V, _ = weight.shape

    n_non_ignore = torch.sum(
        target != ignore_index,
        dtype=torch.int32,
    )

    custom_hidden = (
        hidden_states.detach()
        .clone()
        .requires_grad_(True)
    )
    custom_weight = (
        weight.detach()
        .clone()
        .requires_grad_(True)
    )

    def custom_step():
        loss = custom_linear_cross_entropy(
            custom_hidden,
            custom_weight,
            target,
            ignore_index,
            n_non_ignore,
            workspace_bytes,
        )

        return loss * upstream_scale

    custom_result = benchmark_cuda_function(
        name="Custom",
        function=custom_step,
        hidden_states=custom_hidden,
        weight=custom_weight,
        num_tokens=T,
        hidden_dim=D,
        vocab_size=V,
        warmup=warmup,
        iterations=iterations,
    )

    print_benchmark_result(custom_result)

    del custom_hidden
    del custom_weight
    clear_cuda_memory()

    naive_result: Optional[BenchmarkResult] = None

    try:
        naive_hidden = (
            hidden_states.detach()
            .clone()
            .requires_grad_(True)
        )
        naive_weight = (
            weight.detach()
            .clone()
            .requires_grad_(True)
        )

        def naive_step():
            loss = naive_linear_cross_entropy(
                naive_hidden,
                naive_weight,
                target,
                ignore_index,
            )

            return loss * upstream_scale

        naive_result = benchmark_cuda_function(
            name="Naive PyTorch",
            function=naive_step,
            hidden_states=naive_hidden,
            weight=naive_weight,
            num_tokens=T,
            hidden_dim=D,
            vocab_size=V,
            warmup=warmup,
            iterations=iterations,
        )

        print_benchmark_result(naive_result)

        del naive_hidden
        del naive_weight

    except torch.cuda.OutOfMemoryError:
        print(
            "Naive PyTorch: OOM. The custom kernel completed, "
            "but the full logits path did not fit in HBM."
        )

        clear_cuda_memory()

    if naive_result is not None:
        speedup = naive_result.median_ms / custom_result.median_ms

        memory_reduction = (
            naive_result.peak_allocated_gib
            - custom_result.peak_allocated_gib
        )

        print()
        print(f"Custom speedup:        {speedup:.3f}x")
        print(f"Peak HBM reduction:    {memory_reduction:.3f} GiB")

        if speedup < 1.0:
            print(
                "Warning: the custom implementation is slower than the "
                "naive implementation for this shape."
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Correctness and performance benchmark for fused linear "
            "cross entropy."
        )
    )

    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=32000)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
    )

    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=512,
        help="Maximum custom logit workspace in MiB.",
    )

    parser.add_argument(
        "--ignore-fraction",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
    )

    parser.add_argument(
        "--upstream-scale",
        type=float,
        default=128.0,
        help=(
            "Non-unit upstream loss scale used to validate and benchmark "
            "custom backward scaling."
        ),
    )

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--correctness-only",
        action="store_true",
    )

    parser.add_argument(
        "--performance-only",
        action="store_true",
    )

    parser.add_argument("--loss-atol", type=float, default=5e-3)
    parser.add_argument("--loss-rtol", type=float, default=5e-3)
    parser.add_argument("--grad-atol", type=float, default=3e-2)
    parser.add_argument("--grad-rtol", type=float, default=3e-2)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test requires a CUDA or ROCm device supported by Triton."
        )

    if not 0.0 <= args.ignore_fraction < 1.0:
        raise ValueError("--ignore-fraction must be in [0, 1)")

    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda")
    workspace_bytes = args.workspace_mib * 1024 * 1024

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Keep TF32 settings explicit so benchmark results are reproducible.
    torch.backends.cuda.matmul.allow_tf32 = True

    hidden_states, weight, target = create_inputs(
        num_tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        dtype=dtype,
        device=device,
        ignore_fraction=args.ignore_fraction,
        seed=args.seed,
    )

    logit_bytes = tensor_bytes(
        (args.tokens, args.vocab_size),
        dtype,
    )

    weight_bytes = tensor_bytes(
        (args.vocab_size, args.hidden_dim),
        dtype,
    )

    workspace_tokens = max(
        1,
        workspace_bytes
        // (
            args.vocab_size
            * torch.empty((), dtype=dtype).element_size()
        ),
    )

    expected_chunks = math.ceil(
        args.tokens / min(args.tokens, workspace_tokens)
    )

    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"Device:                 {torch.cuda.get_device_name()}")
    print(f"PyTorch:                {torch.__version__}")
    print(f"Triton:                 {triton.__version__}")
    print(f"Dtype:                  {dtype}")
    print(f"Tokens T:               {args.tokens:,}")
    print(f"Hidden dimension D:     {args.hidden_dim:,}")
    print(f"Vocabulary V:           {args.vocab_size:,}")
    print(f"Ignore fraction:        {args.ignore_fraction:.3f}")
    print(f"Upstream scale:         {args.upstream_scale}")
    print(f"Full logits:            {gib(logit_bytes):.3f} GiB")
    print(f"Unembedding weight:     {gib(weight_bytes):.3f} GiB")
    print(f"Workspace budget:       {args.workspace_mib} MiB")
    print(f"Approx. chunk tokens:   {workspace_tokens:,}")
    print(f"Approx. chunk count:    {expected_chunks:,}")

    if not args.performance_only:
        run_correctness_test(
            hidden_states=hidden_states,
            weight=weight,
            target=target,
            ignore_index=args.ignore_index,
            workspace_bytes=workspace_bytes,
            upstream_scale=args.upstream_scale,
            loss_atol=args.loss_atol,
            loss_rtol=args.loss_rtol,
            grad_atol=args.grad_atol,
            grad_rtol=args.grad_rtol,
        )

    if not args.correctness_only:
        run_performance_test(
            hidden_states=hidden_states,
            weight=weight,
            target=target,
            ignore_index=args.ignore_index,
            workspace_bytes=workspace_bytes,
            upstream_scale=args.upstream_scale,
            warmup=args.warmup,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()