import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ..scripts.tokenizer import tokenizer


def nvtx_push(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


try:
    from liger_kernel.ops.fused_moe import LigerFusedMoEFunction

    LIGER_FUSED_MOE_AVAILABLE = True
except ImportError:
    LigerFusedMoEFunction = None
    LIGER_FUSED_MOE_AVAILABLE = False


try:
    from ..kernels.fused_linear_cross_entropy import fused_linear_cross_entropy
    FUSED_LINEAR_CE_AVAILABLE = True
except ImportError:
    try:
        from src.kernels.fused_linear_cross_entropy import fused_linear_cross_entropy
        FUSED_LINEAR_CE_AVAILABLE = True
    except ImportError:
        try:
            from kernels.fused_linear_cross_entropy import fused_linear_cross_entropy
            FUSED_LINEAR_CE_AVAILABLE = True
        except ImportError:
            fused_linear_cross_entropy = None
            FUSED_LINEAR_CE_AVAILABLE = False


try:
    from ..kernels.apply_rope import TritonRoPEFunction
    from ..kernels.swiglu import TritonSwigluFunction
except ImportError:
    try:
        from src.kernels.apply_rope import TritonRoPEFunction
        from src.kernels.swiglu import TritonSwigluFunction
    except ImportError:
        try:
            from kernels.apply_rope import TritonRoPEFunction
            from kernels.swiglu import TritonSwigluFunction
        except ImportError as e:
            print(f"[WARNING] Failed to import Triton kernels (RoPE/SwiGLU): {e}")
            TritonRoPEFunction = None
            TritonSwigluFunction = None


try:
    from ..kernels.fused_add_rms_norm import FusedAddRMSNormFunction
    FUSED_ADD_RMS_NORM_AVAILABLE = True
except ImportError:
    try:
        from src.kernels.fused_add_rms_norm import FusedAddRMSNormFunction
        FUSED_ADD_RMS_NORM_AVAILABLE = True
    except ImportError:
        try:
            from kernels.fused_add_rms_norm import FusedAddRMSNormFunction
            FUSED_ADD_RMS_NORM_AVAILABLE = True
        except ImportError:
            FusedAddRMSNormFunction = None
            FUSED_ADD_RMS_NORM_AVAILABLE = False


class RMS_Norm(nn.Module):
    def __init__(self,
                 num_features,
                 eps : float = 1e-8,
                 device : torch.device|None = None
        ) -> None:
        """
            Normalizing weights along num_features using RMSNorm.
    
            Args:
                num_features: dim along which the weights are normalized
                eps : a small factor to handle divide-by-zero error
                config: Optional ModelConfig object
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FUSED_ADD_RMS_NORM_AVAILABLE and x.is_cuda:
            zeros = torch.zeros_like(x)
            nvtx_push("triton_rms_norm")
            y, _ = FusedAddRMSNormFunction.apply(x, zeros, self.scale, self.eps)
            nvtx_pop()
            return y
        t, dtype = x.float(), x.dtype
        y = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        return (y * self.scale).to(dtype)
    

def soft_clamp(x: torch.Tensor, limit: float = 5.0):
    """Smoothly clamp *x* to the range ``[-limit, limit]`` via ``tanh``.

    Args:
        x: Input tensor.
        limit: Symmetric clamping bound.

    Returns:
        Soft-clamped tensor of the same shape.
    """
    return limit * torch.tanh(x / limit)

def swiglu(x: torch.Tensor, limit: float = 30.0):
    """SwiGLU activation with soft-clamping for gradient stability.

    Splits *x* along the last dimension into a gating half and a linear half,
    applies SiLU-style gating, and returns the fused result.

    Args:
        x: Input tensor whose last dimension is even.
        limit: Soft-clamping bound applied before gating.

    Returns:
        Activated tensor with last dimension halved.
    """
    x_gate, x_up = x.chunk(2, dim=-1)
    out = F.silu(x_gate) * x_up
    return soft_clamp(out, limit)


class MLPBlock(nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
    ) -> None:
        """
            Multi-Layer Perceptron Block with SwiGLU activation.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = config.intermediate_size

        self.w1 = nn.Linear(
            config.hidden_dim, 2 * config.intermediate_size, device=device, dtype=config.dtype, bias=False,
        )

        self.w2 = nn.Linear(
            config.intermediate_size, config.hidden_dim, device=device, dtype=config.dtype, bias=False,
        )

        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        if TritonSwigluFunction is not None and h.is_cuda:
            nvtx_push("triton_swiglu")
            h = TritonSwigluFunction.apply(h, 30.0)
            nvtx_pop()
        else:
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = soft_clamp(h, 30.0)
        h = self.dropout(h)
        return self.w2(h)    


class Gate(nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
        layer_idx: int = 0,
    ):
        """Initialise the routing gate.

        Args:
            config: :class:`ModelConfig` with MoE hyper-parameters.
            device: Torch device for parameter placement.
            layer_idx: Index of the parent transformer layer (used for
                per-layer bias update scaling).
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.route_scale = config.route_scale

        self.bias_update_rate = getattr(config, "router_bias_update_rate", getattr(config, "update_param", 2e-3))
        self.bias_max = getattr(config, "router_bias_max", 1.0)

        self.router = nn.Linear(
            self.hidden_dim, self.num_experts, bias=False, device=device, dtype=config.dtype,
        )

        self.register_buffer(
            "routing_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device,),
        )

        self.register_buffer(
            "load_accum",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device,),
        )

        self.register_buffer(
            "last_mean_scores",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device,), persistent=False,
        )

        self.register_buffer(
            "last_load",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device,), persistent=False,
        )
        self.last_routing_probs = None

    def forward(self, x: torch.Tensor, retain_full_probs: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                Flattened hidden states [T, D].
            retain_full_probs:
                When True, stash the full ``[T, E]`` routing-probability tensor on
                ``last_routing_probs`` for expensive validation-interval telemetry.
                When False (the default hot path) only compact per-expert
                statistics are kept, avoiding a large FP32 tensor being pinned
                until the next forward pass (~192 MiB at 128K tokens × 24 layers).

        Returns:
            routing_weights:
                Normalized top-k weights [T, K].

            expert_indices:
                Selected expert indices [T, K].

            current_load:
                Number of assignments per expert [E].

            auxiliary_loss:
                Differentiable balancing objective.
        """
        router_logits = F.linear(x, self.router.weight, bias=None).float()
        scores = torch.sigmoid(router_logits)

        biased_scores = scores + self.routing_bias.unsqueeze(0)
        _, expert_indices = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)

        routing_weights = torch.gather(scores, dim=1, index=expert_indices)
        routing_weights = routing_weights / routing_weights.sum(dim=-1,keepdim=True).clamp_min(1e-9)
        routing_weights = (routing_weights * self.route_scale).to(x.dtype)

        current_load = torch.bincount(expert_indices.reshape(-1), minlength=self.num_experts)
        num_tokens = x.shape[0]
        load_fraction = (current_load.float() / max(num_tokens * self.top_k, 1))
        mean_probability = scores.mean(dim=0)
        auxiliary_loss = self.num_experts * torch.sum(load_fraction.detach() * mean_probability)

        if self.training:
            with torch.no_grad():
                self.load_accum.add_(current_load.float())
                self.last_load.copy_(current_load.float())
                self.last_mean_scores.copy_(mean_probability.detach())
                if retain_full_probs:
                    self.last_routing_probs = scores.detach()
                else:
                    self.last_routing_probs = None

        return routing_weights, expert_indices, current_load, auxiliary_loss

    @torch.no_grad()
    def commit_bias_update(self):
        """
        Apply one load-balancing bias update.

        Call this once per optimizer update, after all gradient-accumulation
        microbatches have completed. Do not call it after every microbatch.
        """
        load = self.load_accum.clone()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)

        mean_load = load.mean()

        direction = torch.sign(mean_load - load)

        self.routing_bias.add_(
            self.bias_update_rate * direction
        )

        self.routing_bias.clamp_(min=-self.bias_max, max=self.bias_max)

        self.load_accum.zero_()

    @torch.no_grad()
    def reset_load_statistics(self):
        self.load_accum.zero_()
        self.last_load.zero_()
        self.last_mean_scores.zero_()
    

class RoutedExperts(nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = config.intermediate_size
        self.use_liger = getattr(config, "use_liger_moe", True)

        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * self.intermediate_size, self.hidden_dim,
                device=device, dtype=config.dtype,
            )
        )

        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden_dim, self.intermediate_size,
                device=device, dtype=config.dtype,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                [T, D]

            expert_indices:
                [T, K]

            routing_weights:
                [T, K]

        Returns:
            Routed expert output [T, D].
        """
        if hidden_states.is_cuda and self.use_liger:
            if not LIGER_FUSED_MOE_AVAILABLE:
                raise RuntimeError(
                    "[RoutedExperts] CUDA device detected and use_liger_moe=True but "
                    "liger-kernel is not installed."
                )
            return LigerFusedMoEFunction.apply(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                expert_indices.to(torch.int32),
                routing_weights,
            )

        # Reference path: CPU inference/testing, or CUDA with use_liger_moe=False.
        return self._reference_forward(
            hidden_states,
            expert_indices,
            routing_weights,
        )

    def _reference_forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training-correct fallback.

        It retains the Python loop over 16 experts, but all routing metadata
        remains on the GPU. There is no .tolist(), .item(), or CPU boundary
        calculation in the forward pass.
        """
        output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            assignment_mask = expert_indices == expert_id

            token_indices, slot_indices = torch.where(
                assignment_mask
            )

            expert_input = hidden_states.index_select(
                0,
                token_indices,
            )

            gate_up = F.linear(
                expert_input,
                self.gate_up_proj[expert_id],
                bias=None,
            )

            if TritonSwigluFunction is not None and gate_up.is_cuda:
                nvtx_push("triton_swiglu")
                expert_hidden = TritonSwigluFunction.apply(gate_up, 30.0)
                nvtx_pop()
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                expert_hidden = F.silu(gate) * up
                expert_hidden = soft_clamp(expert_hidden, 30.0)

            expert_output = F.linear(
                expert_hidden,
                self.down_proj[expert_id],
                bias=None,
            )

            selected_weights = routing_weights[
                token_indices,
                slot_indices,
            ].unsqueeze(-1)

            weighted_output = expert_output * selected_weights

            output.index_add_(
                0,
                token_indices,
                weighted_output,
            )

        return output


class MoE(nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = Gate(config,device=device,layer_idx=layer_idx,)
        self.routed_experts = RoutedExperts(config,device=device,)
        self.shared_expert = MLPBlock(config,device=device,)

        self.register_buffer("expert_counts", torch.zeros(self.num_experts,dtype=torch.long,device=device,), persistent=False,)
        self.register_buffer("total_tokens", torch.zeros((), dtype=torch.long, device=device), persistent=False)

    def forward(self, x: torch.Tensor, retain_full_probs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_dim)

        routing_weights, expert_indices, current_load, auxiliary_loss = self.gate(
            x_flat, retain_full_probs=retain_full_probs,
        )

        routed_output = self.routed_experts(
            hidden_states=x_flat, expert_indices=expert_indices, routing_weights=routing_weights,
        )

        shared_output = self.shared_expert(x_flat)
        output = routed_output + shared_output

        if self.training:
            with torch.no_grad():
                self.expert_counts.add_(current_load.to(self.expert_counts.dtype))
                self.total_tokens.add_(x_flat.shape[0])
        return output.view(original_shape), auxiliary_loss

    @torch.no_grad()
    def get_wandb_metrics(self) -> Dict[str, float]:
        """Per-expert utilization + load-balance health for W&B.

        Returns fractional load per expert (share of the ``top_k`` assignment
        budget) plus a load-balance score in ``[0, 1]`` where 1.0 is perfectly
        uniform routing. Safe to call only when ``total_tokens > 0``.
        """
        total_tokens_val = int(self.total_tokens.item())
        total_assignments = max(total_tokens_val * self.top_k, 1)
        counts = self.expert_counts.float()
        fractions = counts / total_assignments

        metrics: Dict[str, float] = {}
        for expert_id in range(self.num_experts):
            metrics[f"expert_{expert_id}"] = fractions[expert_id].item()

        uniform = 1.0 / self.num_experts
        max_frac = fractions.max().item()
        metrics["load_balance_score"] = max(0.0, 1.0 - (max_frac - uniform) / (1.0 - uniform))
        return metrics

    @torch.no_grad()
    def reset_expert_counts(self):
        self.expert_counts.zero_()
        self.total_tokens.zero_()

    @torch.no_grad()
    def commit_bias_update(self):
        self.gate.commit_bias_update()


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 head_dim : int,
                 base : int,
                 dtype : torch.dtype,
                 initial_context_len : int = 2048,
                 max_context_len : int = 2048,
                 ntk_alpha : float = 1.0,
                 ntk_beta : float = 32.0,
                 scaling_factor : float = 1.0,
                 device: torch.device | None = None
        ) -> None:
        """
            Rotary Position Embedding with YaRN scaling support.
    
            Args:
                head_dim: dimension of each attention head
                base: base frequency for rotary embeddings
                dtype: data type for computations
                initial_context_len: original context length for YaRN scaling
                max_context_len: maximum context length to precompute
                ntk_alpha: NTK-aware scaling alpha parameter
                ntk_beta: NTK-aware scaling beta parameter
                scaling_factor: context length scaling factor
                device: torch device to place the module on
        """
        super().__init__()
        self.head_dim  = head_dim
        self.base = base
        self.initial_context_len = initial_context_len
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.scaling_factor = scaling_factor
        self.device = device

    @staticmethod
    def apply_rope(x : torch.Tensor,
                cos : torch.Tensor,
                sin : torch.Tensor
        ) -> torch.Tensor:
        """Apply Rotary Position Embedding to tensor *x*.

        Splits the last dimension of *x* in half and applies the standard RoPE
        rotation using the precomputed *cos* and *sin* tables.

        Args:
            x: Input tensor of shape ``(B, S, H, D)``.
            cos: Cosine table, either ``(S, D//2)`` or ``(B, S, D//2)``.
            sin: Sine table with the same shape as *cos*.

        Returns:
            Rotated tensor of the same shape as *x*.
        """
        if cos.dim() == 2: 
            # (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
            cos = cos.unsqueeze(0).unsqueeze(-2)
            sin = sin.unsqueeze(0).unsqueeze(-2)
        else: # inference
            # (batch, seq_len, head_dim//2) -> (batch, seq_len, 1, head_dim//2) * (batch, seq_len, n_heads, head_dim)
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        cos = cos.to(x.device).to(x.dtype)
        sin = sin.to(x.device).to(x.dtype)
        x1,x2 = torch.chunk(x,2,dim = -1)
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.cat([o1,o2],dim = -1)

    def _compute_concentration_and_inv_freq(self) -> Tuple[float,torch.Tensor]:
        """Refer gpt-oss implemention of YaRN and See YaRN paper for more details: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_len / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_len / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def compute_cos_sin(self,num_tokens : int) -> Tuple[torch.Tensor,torch.Tensor]:
        """Precompute cosine and sine tables for the first *num_tokens* positions.

        Args:
            num_tokens: Number of positions to compute.

        Returns:
            Tuple of ``(cos, sin)`` tensors, each of shape
            ``(num_tokens, head_dim // 2)``.
        """
        concentration , inv_freq  = self._compute_concentration_and_inv_freq()
        pos = torch.arange(num_tokens,dtype = torch.float32 ,device = self.device)
        freqs = torch.einsum('i,j->ij',pos,inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos,sin

    def _apply_rope_pytorch(self, x, cos, sin):
        """PyTorch fallback for RoPE when Triton kernel is unavailable."""
        # x: (B, T, nH, head_dim), cos/sin: (T, head_dim//2) or (B, T, head_dim//2)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        # Broadcast cos/sin to match x shape
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, half)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(2)  # (B, T, 1, half)
            sin = sin.unsqueeze(2)
        out = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)
        return out

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor,
                cos : torch.Tensor,
                sin : torch.Tensor,
                offset : int = 0,
                position_ids : torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size,seq_len,_,_ = q.shape
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[offset:offset+seq_len,:]
            sin = sin[offset:offset+seq_len,:]

        query_shape = q.shape
        q = q.view(batch_size,seq_len,-1,self.head_dim)
        if TritonRoPEFunction is not None and q.is_cuda:
            nvtx_push("triton_rope_q")
            q = TritonRoPEFunction.apply(q,cos,sin)
            nvtx_pop()
        else:
            q = self._apply_rope_pytorch(q, cos, sin)
        q = q.reshape(query_shape)

        key_shape = k.shape
        k = k.view(batch_size,seq_len,-1,self.head_dim)
        if TritonRoPEFunction is not None and k.is_cuda:
            nvtx_push("triton_rope_k")
            k = TritonRoPEFunction.apply(k,cos,sin)
            nvtx_pop()
        else:
            k = self._apply_rope_pytorch(k, cos, sin)
        k = k.reshape(key_shape)

        return q,k


class Attention(nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
        inference: bool = False,
    ):
        super().__init__()

        self.n_heads = config.num_attn_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.inference = inference
        self.max_cache_len = config.max_context_len
        self.attn_dropout = getattr(config, "dropout", 0.0)

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                "num_attn_heads must be divisible by "
                "num_key_value_heads"
            )

        self.q_dim = config.num_attn_heads * config.head_dim
        self.kv_dim = config.num_key_value_heads * config.head_dim

        self.w_qkv = nn.Linear(
            config.hidden_dim, self.q_dim + 2 * self.kv_dim,
            device=device, dtype=config.dtype, bias=False,
        )

        self.wo = nn.Linear(
            config.num_attn_heads * config.head_dim, config.hidden_dim, device=device,
            dtype=config.dtype, bias=False,
        )

        self.q_norm = RMS_Norm(config.head_dim, device=device)
        self.k_norm = RMS_Norm(config.head_dim, device=device)

        self.rope = RotaryEmbedding(
            config.head_dim,
            config.base,
            torch.float32,
            initial_context_len=config.initial_context_len,
            max_context_len=config.max_context_len,
            ntk_alpha=config.ntk_alpha,
            ntk_beta=config.ntk_beta,
            scaling_factor=config.scaling_factor,
            device=device,
        )

        if self.inference:
            self.register_buffer("cache_k", None, persistent=False)
            self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(
        self,
        batch_size: int = 1,
    ):
        if not self.inference:
            return

        device = self.w_qkv.weight.device
        dtype = self.w_qkv.weight.dtype

        self.cache_k = torch.empty(batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim, device=device, dtype=dtype)
        self.cache_v = torch.empty(batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim, device=device, dtype=dtype)

    @staticmethod
    def _build_offset_causal_mask(
        start_pos: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Query row q corresponds to absolute position start_pos + q.

        A query may attend to key k iff:

            k <= start_pos + q
        """
        query_positions = (start_pos + torch.arange(query_length, device=device))
        key_positions = torch.arange(key_length, device=device)
        mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        return mask.unsqueeze(0).unsqueeze(0)

    def _training_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:

        dropout_p = self.attn_dropout if self.training else 0.0

        # ── Document-aware packing (variable-length FlashAttention) ──
        if cu_seqlens is not None:
            if flash_attn_varlen_func is None:
                raise RuntimeError(
                    "cu_seqlens was provided for document-aware packing but "
                    "flash_attn_varlen_func is unavailable. Install flash-attn "
                    "(>=2.x) or pass a block-diagonal attn_mask instead."
                )
            batch_size, seq_len = q.shape[0], q.shape[1]
            q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
            k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
            v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
            eff_max = max_seqlen if max_seqlen is not None else seq_len
            out = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=eff_max,
                max_seqlen_k=eff_max,
                dropout_p=dropout_p,
                causal=True,
            )
            return out.reshape(batch_size, seq_len, out.shape[-2], out.shape[-1])

        if flash_attn_func is not None and attn_mask is None:
            return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=(attn_mask is None),enable_gqa=(self.n_heads != self.n_kv_heads),
        )

        return output.transpose(1, 2)

    def _inference_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
        end_pos: int,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, query_length, _, _ = q.shape

        self.cache_k[:batch_size, :, start_pos:end_pos, :].copy_(k.transpose(1, 2))
        self.cache_v[:batch_size, :, start_pos:end_pos, :].copy_(v.transpose(1, 2))

        cached_k = self.cache_k[:batch_size, :, :end_pos, :]
        cached_v = self.cache_v[:batch_size, :, :end_pos, :]

        q = q.transpose(1, 2)

        if attn_mask is not None:
            effective_mask = attn_mask
            is_causal = False

        elif start_pos == 0:
            effective_mask = None
            is_causal = query_length > 1

        elif query_length == 1:
            effective_mask = None
            is_causal = False

        else:
            effective_mask = self._build_offset_causal_mask(
                start_pos=start_pos,query_length=query_length,key_length=end_pos,device=q.device,
            )

            is_causal = False

        output = F.scaled_dot_product_attention(
            q,cached_k, cached_v, attn_mask=effective_mask, is_causal=is_causal, enable_gqa=(self.n_heads != self.n_kv_heads)
        )

        return output.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len

        if self.inference:
            if self.cache_k is None or self.cache_k.shape[0] != batch_size:
                self.reset_cache(batch_size)

            if end_pos > self.max_cache_len:
                raise ValueError(
                    f"Requested cache position {end_pos}, but "
                    f"max_cache_len={self.max_cache_len}"
                )

        qkv = self.w_qkv(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rope(q, k, cos, sin, offset=start_pos, position_ids=position_ids)

        if self.inference:
            attention_output = self._inference_attention(
                q, k, v, start_pos=start_pos, end_pos=end_pos, attn_mask=attn_mask,
            )
        else:
            attention_output = self._training_attention(
                q, k, v, attn_mask=attn_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            )

        attention_output = attention_output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        return self.wo(attention_output)
        
    
class TransformerDecoderBLK(nn.Module):
    def __init__(
        self,
        config,
        device=None,
        inference=False,
        layer_idx=0,
    ):
        """Initialise a decoder block.

        Args:
            config: :class:`ModelConfig` with model hyper-parameters.
            device: Torch device for parameter placement.
            inference: Whether to enable KV caching in the attention layer.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim,device=device)
        self.norm2 = RMS_Norm(config.hidden_dim,device=device)

        self.attention = Attention(config, device=device, inference=inference)
        self.mlp = MoE(config, device=device,layer_idx=layer_idx)

    def forward(
        self,
        x,
        cos,
        sin,
        start_pos=0,
        position_ids=None,
        attn_mask=None,
        retain_full_probs=False,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        attention_output = self.attention(
            self.norm1(x), cos, sin, start_pos, position_ids, attn_mask,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )

        if FUSED_ADD_RMS_NORM_AVAILABLE and x.is_cuda:
            normed, x = FusedAddRMSNormFunction.apply(
                attention_output, x, self.norm2.scale, self.norm2.eps,
            )
        else:
            x = x + attention_output
            normed = self.norm2(x)

        mlp_output, auxiliary_loss = self.mlp(normed, retain_full_probs=retain_full_probs)
        x = x + mlp_output

        return x, auxiliary_loss


class GPT_FLASH(nn.Module):
    def __init__(
        self,
        config,
        device=None,
        inference=False,
    ):
        """Initialise the model.

        Args:
            config: :class:`ModelConfig` defining architecture hyper-parameters.
            device: Torch device for parameter and buffer placement.
            inference: If ``True``, enables KV caching in all attention layers.
        """
        super().__init__()
        self.config = config
        self.inference = inference
        self.num_hidden_layers = config.num_hidden_layers

        self.moe_aux_loss_weight = getattr(config, "moe_aux_loss_weight", 1e-3)
        self.loss_workspace_bytes = getattr(config, "loss_workspace_bytes", 512 * 1024 * 1024)
        self.ignore_index = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        self.norm = RMS_Norm(config.hidden_dim, device=device)

        self._rope = RotaryEmbedding(
            config.head_dim,
            config.base,
            torch.float32,
            initial_context_len=config.initial_context_len,
            max_context_len=config.max_context_len,
            ntk_alpha=config.ntk_alpha,
            ntk_beta=config.ntk_beta,
            scaling_factor=config.scaling_factor,
            device=device,
        )

        cos, sin = self._rope.compute_cos_sin(config.max_context_len)

        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)

        self.embeddings = nn.Embedding(config.vocab_size,config.hidden_dim,device=device,dtype=config.dtype)

        self.layers = nn.ModuleList([
            TransformerDecoderBLK(config,device=device,
                    inference=inference,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.unembedding = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=config.dtype,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        base_std = getattr(self.config, "initializer_std", 0.02)
        router_std = getattr(self.config, "router_initializer_std", 0.01)
        residual_std = base_std / math.sqrt(2 * self.num_hidden_layers)

        # Base initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=base_std,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=base_std,
                )

            elif isinstance(module, RMS_Norm):
                module.scale.fill_(1.0)

        # Specialized residual and router initialization.
        for layer in self.layers:
            nn.init.normal_(
                layer.attention.wo.weight,
                mean=0.0,
                std=residual_std,
            )

            nn.init.normal_(
                layer.mlp.shared_expert.w2.weight,
                mean=0.0,
                std=residual_std,
            )

            nn.init.normal_(
                layer.mlp.gate.router.weight,
                mean=0.0,
                std=router_std,
            )

            nn.init.normal_(
                layer.mlp.routed_experts.gate_up_proj,
                mean=0.0,
                std=base_std,
            )

            nn.init.normal_(
                layer.mlp.routed_experts.down_proj,
                mean=0.0,
                std=residual_std,
            )

    def reset_cache(
        self,
        batch_size: int = 1,
    ):
        if not self.inference:
            return

        for layer in self.layers:
            layer.attention.reset_cache(batch_size)

    @torch.no_grad()
    def commit_moe_bias_updates(self):
        """
        Call once after each optimizer update.

        Example:

            scaler.step(optimizer)
            scaler.update()
            model.commit_moe_bias_updates()
        """
        for layer in self.layers:
            layer.mlp.commit_bias_update()

    @torch.no_grad()
    def reset_moe_statistics(self):
        for layer in self.layers:
            layer.mlp.reset_expert_counts()
            layer.mlp.gate.reset_load_statistics()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        collect_routing_telemetry: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        if input_ids.numel() > 0:
            torch._assert(input_ids.min() >= 0, "Negative token ID")
            torch._assert(
                input_ids.max() < self.config.vocab_size,
                "Token ID exceeds padded vocabulary (50_304)",
            )

        x = self.embeddings(input_ids)

        total_auxiliary_loss = x.new_zeros((), dtype=torch.float32)

        for layer in self.layers:
            x, layer_auxiliary_loss = layer(
                x, self.cos, self.sin, start_pos, position_ids, attn_mask,
                retain_full_probs=collect_routing_telemetry,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            )
            total_auxiliary_loss = total_auxiliary_loss + layer_auxiliary_loss.float()

        x = self.norm(x)

        auxiliary_loss = self.moe_aux_loss_weight * total_auxiliary_loss / self.num_hidden_layers

        if labels is not None and not return_logits:
            hidden_flat = x.reshape(-1, x.shape[-1])
            labels_flat = labels.reshape(-1)
            n_non_ignore = torch.sum(labels_flat != self.ignore_index, dtype=torch.int32)

            if FUSED_LINEAR_CE_AVAILABLE and hidden_flat.is_cuda:
                nvtx_push("triton_fused_linear_ce")
                lm_loss = fused_linear_cross_entropy(
                    hidden_states=hidden_flat,
                    weight=self.unembedding.weight,
                    target=labels_flat,
                    ignore_index=self.ignore_index,
                    total_n_non_ignore=n_non_ignore,
                    workspace_bytes=self.loss_workspace_bytes,
                )
                nvtx_pop()
            else:
                logits = self.unembedding(x)
                lm_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    labels_flat,
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )

            return lm_loss, auxiliary_loss

        logits = self.unembedding(x)

        return logits, auxiliary_loss

    # ── Attention Diagnostics for W&B ────────────────────────────
    @torch.no_grad()
    def get_attention_diagnostics(self) -> Dict[str, float]:
        """
        Compute per-layer attention health metrics for W&B logging.
        Lightweight: only inspects QK-norm scale parameters (no forward pass).

        Returns dict with keys like:
            attn/layer_0/q_scale_max, attn/layer_0/k_scale_max,
            attn/layer_0/q_norm_est, attn/layer_0/k_norm_est,
            attn/q_scale_max_global, attn/k_scale_max_global
        """
        metrics = {}
        q_maxes, k_maxes = [], []

        for i, layer in enumerate(self.layers):
            q_scale = layer.attention.q_norm.scale.data
            k_scale = layer.attention.k_norm.scale.data

            q_max = q_scale.max().item()
            k_max = k_scale.max().item()
            q_mean = q_scale.mean().item()
            k_mean = k_scale.mean().item()
            q_maxes.append(q_max)
            k_maxes.append(k_max)

            # Estimated vector norm: scale_rms × √head_dim
            head_dim = q_scale.shape[0]
            q_rms = (q_scale.float() ** 2).mean().sqrt().item()
            k_rms = (k_scale.float() ** 2).mean().sqrt().item()
            q_norm_est = q_rms * math.sqrt(head_dim)
            k_norm_est = k_rms * math.sqrt(head_dim)

            metrics[f"attn/layer_{i}/q_scale_max"] = q_max
            metrics[f"attn/layer_{i}/k_scale_max"] = k_max
            metrics[f"attn/layer_{i}/q_scale_mean"] = q_mean
            metrics[f"attn/layer_{i}/k_scale_mean"] = k_mean
            metrics[f"attn/layer_{i}/q_norm_est"] = q_norm_est
            metrics[f"attn/layer_{i}/k_norm_est"] = k_norm_est

        metrics["attn/q_scale_max_global"] = max(q_maxes)
        metrics["attn/k_scale_max_global"] = max(k_maxes)

        return metrics

    # ── Comprehensive Telemetry ──────────────────────────────
    @torch.no_grad()
    def get_telemetry_diagnostics(
        self,
        input_ids: torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 0.0,
        include_hidden_states: bool = False,
    ) -> Dict[str, float]:
        """
        Collect all telemetry metrics in a single call.

        Args:
            input_ids:  Last training batch (needed for hidden state telemetry).
            optimizer:  The optimizer (needed for weight update ratio telemetry).
            lr:         Current learning rate.
            include_hidden_states: Whether to run the expensive hidden-state
                                   SVD + cosine-sim analysis (only at val_interval).

        Returns:
            Dict of telemetry metrics for W&B logging.
        """
        from ..scripts.training.telemetry import (
            compute_routing_telemetry,
            compute_weight_update_ratios,
            compute_hidden_state_telemetry,
        )
        metrics = {}

        # 1. Routing entropy + router weight cosine sim (lightweight)
        metrics.update(compute_routing_telemetry(self))

        # 2. Weight update ratios (expensive — run only at val_interval)
        if include_hidden_states and optimizer is not None:
            metrics.update(compute_weight_update_ratios(self, optimizer, lr))

        # 3. Hidden state collapse (expensive — run only at val_interval)
        if include_hidden_states and input_ids is not None:
            metrics.update(compute_hidden_state_telemetry(self, input_ids))

        return metrics


def build_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
):
    decay = []
    no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim < 2
            or "norm" in name.lower()
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]