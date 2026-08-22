"""
model_improv — kernel-integrated twin of model_flash_attn
=========================================================

This is model_flash_attn (the pretrained architecture) with the custom
Triton kernels from src/kernels wired into the compute paths:

    * RMS_Norm            -> FusedRMSNorm (src/kernels/fused_add_rms_norm.py)
    * norm2 + residual add-> FusedAddRMSNorm (same module)
    * RoPE                -> TritonRoPE (src/kernels/apply_rope.py)
    * Gemma-2 SwiGLU      -> TritonGemmaSwiglu (src/kernels/gemma_swiglu.py)
                             (NEW kernel: exact numerics of
                              model_flash_attn.swiglu — clamp(glu,max=L),
                              clamp(linear,±L), glu*sigmoid(alpha*glu)*(lin+1).
                              NOT interchangeable with swiglu.py's silu variant!)
    * MoE                 -> UNCHANGED (custom grouped-GEMM MoE = future work)

WEIGHT COMPATIBILITY CONTRACT
=============================
Every parameter and buffer name/shape is identical to model_flash_attn.
Only computation inside forward methods changed, and every Triton path is
mathematically equivalent to its PyTorch fallback. A checkpoint trained on
model_flash_attn loads here with strict state_dict equality and produces
the same outputs (within bf16 tolerance).

All kernels degrade gracefully to eager PyTorch when unavailable
(CPU / missing CUDA), where both files are bit-identical.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict
from ..scripts.configs.model_config import ModelConfig
try:
    from flash_attn import flash_attn_func
    if flash_attn_func is not None:
        from ..scripts.training.telemetry import _safe_flash_attn_func
        flash_attn_func = _safe_flash_attn_func
except ImportError:
    flash_attn_func = None
try:
    # Variable-length path for document-aware sequence packing: tokens from
    # different documents in one packed batch never attend to each other.
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:
    _flash_attn_varlen_func = None

# ── Custom Triton kernels (with graceful CPU fallback) ────────────────
try:
    from ..kernels.fused_add_rms_norm import FusedRMSNormFunction, FusedAddRMSNormFunction
    _FUSED_RMS_OK = True
except ImportError:
    try:
        from src.kernels.fused_add_rms_norm import FusedRMSNormFunction, FusedAddRMSNormFunction
        _FUSED_RMS_OK = True
    except ImportError:
        try:
            from kernels.fused_add_rms_norm import FusedRMSNormFunction, FusedAddRMSNormFunction
            _FUSED_RMS_OK = True
        except ImportError:
            FusedRMSNormFunction = None
            FusedAddRMSNormFunction = None
            _FUSED_RMS_OK = False

try:
    from ..kernels.apply_rope import TritonRoPEFunction
    _TRITON_ROPE_OK = True
except ImportError:
    try:
        from src.kernels.apply_rope import TritonRoPEFunction
        _TRITON_ROPE_OK = True
    except ImportError:
        try:
            from kernels.apply_rope import TritonRoPEFunction
            _TRITON_ROPE_OK = True
        except ImportError:
            TritonRoPEFunction = None
            _TRITON_ROPE_OK = False

try:
    from ..kernels.gemma_swiglu import TritonGemmaSwigluFunction
    _TRITON_GEMMA_SWIGLU_OK = True
except ImportError:
    try:
        from src.kernels.gemma_swiglu import TritonGemmaSwigluFunction
        _TRITON_GEMMA_SWIGLU_OK = True
    except ImportError:
        try:
            from kernels.gemma_swiglu import TritonGemmaSwigluFunction
            _TRITON_GEMMA_SWIGLU_OK = True
        except ImportError:
            TritonGemmaSwigluFunction = None
            _TRITON_GEMMA_SWIGLU_OK = False


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

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        # ── Kernel integration: fused RMSNorm (fp32 internal math, identical
        #    semantics; reuses the backward of the add-variant internally). ──
        if FusedRMSNormFunction is not None and x.is_cuda:
            return FusedRMSNormFunction.apply(x, self.scale, self.eps)
        t,dtype = x.float(),x.dtype
        t = t * torch.rsqrt(torch.mean(t**2,dim = -1,keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """Reference (eager) implementation — also used as kernel fallback."""
    x_glu, x_linear = x.chunk(2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class MLPBlock(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device|None = None
        ) -> None:
        """
            Multi-Layer Perceptron Block with SwiGLU activation.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
        )
        self.w2 = nn.Linear(
            config.intermediate_size, config.hidden_dim, device = device, dtype=config.dtype
        )
        self.w3 = nn.Linear(
            config.hidden_dim, config.intermediate_size, device = device, dtype=config.dtype
        )

        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        # ── Kernel integration: fused Gemma-2 style SwiGLU ──
        if TritonGemmaSwigluFunction is not None and h.is_cuda:
            act = TritonGemmaSwigluFunction.apply(h, 1.702, 7.0)
        else:
            act = swiglu(h)
        return self.w2(self.dropout(act * self.w3(x)))


class Expert(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device|None = None
        ) -> None:
        """
            A Multi-Layer Perceptron Block for Experts in MoE.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
        )
        self.w2 = nn.Linear(
            config.intermediate_size, config.hidden_dim, device = device, dtype=config.dtype
        )
        self.w3 = nn.Linear(
            config.hidden_dim, config.intermediate_size, device = device, dtype=config.dtype
        )

        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        # ── Kernel integration: fused Gemma-2 style SwiGLU ──
        if TritonGemmaSwigluFunction is not None and h.is_cuda:
            act = TritonGemmaSwigluFunction.apply(h, 1.702, 7.0)
        else:
            act = swiglu(h)
        return self.w2(self.dropout(act * self.w3(x)))


class Gate(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
                layer_idx : int = 0,
    ) -> None:
        """
            - Router/Gate module for Mixture of Experts.
            - Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy
            - Per-layer update scaling: deeper layers get more aggressive bias updates
              (1.0× at layer 0, up to 1.5× at the deepest layer) to counteract
              deep-layer expert polarization.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                layer_idx: index of this layer (0-indexed), used for per-layer scaling
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias = False, device = device, dtype = config.dtype)
        self.register_buffer("bias", torch.zeros(config.num_experts, dtype=torch.float32, device=device))
        self.last_routing_probs: torch.Tensor | None = None

        num_layers = config.num_hidden_layers
        layer_scale = 1.0 + 0.5 * (layer_idx / max(num_layers - 1, 1))
        self.effective_update = config.update_param * layer_scale

    def update_bias(self, current_load: torch.Tensor) -> None:
        """Update bias in-place using Loss-Free Balancing rule."""
        load_float = current_load.float()
        e = torch.sign(load_float.mean() - load_float)
        self.bias.add_(self.effective_update * e)
        self.bias.clamp_(-10.0, 10.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See this paper for implementation details: https://arxiv.org/abs/2408.15664"""
        scores = self.router(x)
        scores = torch.sigmoid(scores)
        original_scores = scores
        self.last_routing_probs = original_scores.detach()
        biased_scores = scores + self.bias.detach().to(scores.dtype)
        indices = torch.topk(biased_scores, self.topk, dim=-1)[1]
        # NOTE: torch.bincount is FORBIDDEN here — on CUDA it sizes its output
        # from max(input)+1, forcing an internal .max() + host read (= hidden
        # pipeline drains). Scatter-add has a statically-known output size.
        flat_idx = indices.reshape(-1)
        current_load = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        weights = original_scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale

        # Bias term update rule
        if self.training:
            with torch.no_grad():
                self.update_bias(current_load)

        return weights.type_as(x), indices, current_load


class MoE(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device|None = None,
                 layer_idx : int = 0,
    ) -> None:
        """
            Mixture of Experts module with shared experts.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                layer_idx: index of this layer (0-indexed), passed to Gate for per-layer scaling
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.gate = Gate(config, device, layer_idx=layer_idx)
        self.n_routed_experts = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList(
            [Expert(config,device)
             for _ in range(config.num_experts)]
        )
        self.shared_experts = MLPBlock(config,device)
        self.register_buffer(
            'expert_counts',
            torch.zeros(config.num_experts, dtype=torch.long, device = device)
        )
        self.total_tokens = 0

    def get_expert_utilization(self):
        """Return expert utilization statistics for logging"""
        if self.total_tokens == 0:
            return {}
        utilization = self.expert_counts.float() / self.total_tokens
        return {
            f"experts/expert_{i}_util": utilization[i].item()
            for i in range(self.num_experts)
        }

    def get_wandb_metrics(self):
        """Return expert utilization metrics formatted for wandb real-time dashboard"""
        if self.total_tokens == 0:
            return {}
        utilization = self.expert_counts.float() / self.total_tokens
        util_list = [utilization[i].item() * 100 for i in range(self.num_experts)]
        ideal = 100.0 / self.num_experts
        load_balance = (1.0 - sum(abs(u - ideal) for u in util_list) / (2 * 100.0)) * 100.0

        metrics = {
            **{f"expert_{i}": util_list[i] for i in range(self.num_experts)},
            "load_balance_score": load_balance,
        }

        return metrics

    def reset_expert_counts(self):
        """Reset counters (call periodically during training)"""
        self.expert_counts.zero_()
        self.total_tokens = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_shape = x.shape
        x = x.view(-1, self.dim)
        xprt_weights, xprt_idxs, counts = self.gate(x)

        self.expert_counts += counts
        self.total_tokens += x.shape[0] * self.n_routed_experts

        # ── Batched expert dispatch: sort tokens by expert ID ──
        # Flatten top-k indices: each token appears k times
        flat_idx = xprt_idxs.view(-1)                                        # (N*k,)
        flat_weights = xprt_weights.view(-1, 1)                              # (N*k, 1)
        token_idx = torch.arange(x.shape[0], device=x.device)
        token_idx = token_idx.unsqueeze(1).expand_as(xprt_idxs).reshape(-1)  # (N*k,)

        # Sort by expert ID for contiguous memory access
        sort_order = flat_idx.argsort(stable=True)
        sorted_expert_ids = flat_idx[sort_order]
        sorted_token_idx = token_idx[sort_order]
        sorted_weights = flat_weights[sort_order]

        # Gather tokens in expert-sorted order (single contiguous gather)
        sorted_x = x[sorted_token_idx]                                       # (N*k, dim)

        # Find boundaries between experts via searchsorted
        expert_boundaries = torch.searchsorted(
            sorted_expert_ids.contiguous(),
            torch.arange(self.num_experts + 1, device=x.device),
        )

        # PERF: one batched DtoH read instead of 2*E per-expert .item() calls.
        # The old pattern (start/end .item() inside the loop) drained the
        # pipeline 32x per layer per microbatch; this drains it once.
        # (Your upcoming grouped-GEMM MoE kernel removes this last sync too.)
        bounds = expert_boundaries.tolist()

        # Run each expert on its contiguous slice
        sorted_out = torch.zeros_like(sorted_x)
        for i, expert in enumerate(self.experts):
            start, end = bounds[i], bounds[i + 1]
            if start < end:
                sorted_out[start:end] = expert(sorted_x[start:end])

        # Weighted scatter-add back to original token positions
        sorted_out = sorted_out * sorted_weights
        routed_xprt_out = torch.zeros_like(x)
        routed_xprt_out.scatter_add_(
            0,
            sorted_token_idx.unsqueeze(1).expand_as(sorted_out),
            sorted_out,
        )

        mlp_out = routed_xprt_out + self.shared_experts(x)
        return mlp_out.reshape(inp_shape)


def apply_rope(x : torch.Tensor,
               cos : torch.Tensor,
               sin : torch.Tensor
    ) -> torch.Tensor:
    if cos.dim() == 2:
        # (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
    else:
        # (batch, seq_len, head_dim//2) -> (batch, seq_len, 1, head_dim//2)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    cos = cos.to(x.device).to(x.dtype)
    sin = sin.to(x.device).to(x.dtype)
    x1,x2 = torch.chunk(x,2,dim = -1)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1,o2],dim = -1)


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
                max_context_len: max context length for precomputed tables
                ntk_alpha: NTK-aware scaling alpha parameter
                ntk_beta: NTK-aware scaling beta parameter
                scaling_factor: context length scaling factor
                device: torch.device | None
        """
        super().__init__()
        self.head_dim  = head_dim
        self.base = base
        self.initial_context_len = initial_context_len
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.scaling_factor = scaling_factor
        self.device = device
        cos, sin = self.compute_cos_sin(max_context_len)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

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
        concentration , inv_freq  = self._compute_concentration_and_inv_freq()
        pos = torch.arange(num_tokens,dtype = torch.float32 ,device = self.device)
        freqs = torch.einsum('i,j->ij',pos,inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos,sin

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor,
                offset : int = 0,
                position_ids : torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size,seq_len,_,_ = q.shape
        if position_ids is not None:
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]
        else:
            cos = self.cos[offset:offset+seq_len,:]
            sin = self.sin[offset:offset+seq_len,:]

        query_shape = q.shape
        q = q.view(batch_size,seq_len,-1,self.head_dim)
        # ── Kernel integration: fused RoPE (identical math, single launch
        #    per tensor instead of chunk/mul/cat chains) ──
        if TritonRoPEFunction is not None and q.is_cuda:
            q = TritonRoPEFunction.apply(q,cos,sin)
        else:
            q = apply_rope(q,cos,sin)
        q = q.reshape(query_shape)

        key_shape = k.shape
        k = k.view(batch_size,seq_len,-1,self.head_dim)
        if TritonRoPEFunction is not None and k.is_cuda:
            k = TritonRoPEFunction.apply(k,cos,sin)
        else:
            k = apply_rope(k,cos,sin)
        k = k.reshape(key_shape)

        return q,k


class Attention(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
                inference : bool = False,
    ) -> None:
        """
            Multi-Head Attention with Grouped Query Attention and Flash Attention.
            Supports Gemma-2 style logit soft-capping for attention stability.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                inference: whether to enable KV caching for inference
        """
        super().__init__()
        self.n_heads = config.num_attn_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inference = inference
        self.max_cache_len = config.max_context_len

        self.wq = nn.Linear(
            config.hidden_dim, config.num_attn_heads * config.head_dim, device = device, dtype = config.dtype
        )
        self.wk = nn.Linear(
            config.hidden_dim,config.num_key_value_heads * config.head_dim , device = device, dtype = config.dtype
        )
        self.wv = nn.Linear(
            config.hidden_dim,config.num_key_value_heads * config.head_dim , device = device, dtype = config.dtype
        )
        self.wo = nn.Linear(
            config.num_attn_heads * config.head_dim, config.hidden_dim, device = device, dtype = config.dtype
        )
        if self.inference:
            self.register_buffer("cache_k", None, persistent=False)
            self.register_buffer("cache_v", None, persistent=False)
        self.q_norm = RMS_Norm(config.head_dim, device = device)
        self.k_norm = RMS_Norm(config.head_dim, device = device)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.base,
            torch.float32,
            initial_context_len = config.initial_context_len,
            max_context_len = config.max_context_len,
            ntk_alpha = config.ntk_alpha,
            ntk_beta = config.ntk_beta,
            scaling_factor = config.scaling_factor,
            device = device
        )

    def reset_cache(self, batch_size: int = 1) -> None:
        """Allocate (or reallocate) KV cache for the given batch size."""
        if self.inference:
            device = self.wq.weight.device
            dtype = self.wq.weight.dtype
            self.cache_k = torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=dtype,
            )
            self.cache_v = torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=dtype,
            )

    def forward(self,
                x : torch.Tensor,
                start_pos : int = 0,
                position_ids : torch.Tensor | None = None,
                attn_mask : torch.Tensor | None = None,
                cu_seqlens : torch.Tensor | None = None,
                max_seqlen : int | None = None,
        ) -> torch.Tensor:
        batch_size,seq_len,_ = x.shape
        end_pos = start_pos + seq_len

        if self.inference and cu_seqlens is not None:
            raise ValueError("cu_seqlens packing is a training-only path")

        if self.inference:
            if self.cache_k is None or self.cache_k.shape[0] != batch_size:
                self.reset_cache(batch_size)
            assert end_pos <= self.max_cache_len, (
                f"Sequence length {end_pos} exceeds max cache length {self.max_cache_len}. "
                f"Increase max_context_len in ModelConfig."
            )

        Q,K,V = self.wq(x),self.wk(x),self.wv(x)

        Q = Q.view(batch_size,seq_len,self.n_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)

        Q,K = self.q_norm(Q),self.k_norm(K)
        Q,K = self.rope(Q,K,offset = start_pos,position_ids = position_ids)

        if self.inference:
            # Write: transpose small new tokens (B,S,Hkv,D) → (B,Hkv,S,D)
            self.cache_k[:, :, start_pos:end_pos, :] = K.transpose(1, 2)
            self.cache_v[:, :, start_pos:end_pos, :] = V.transpose(1, 2)
            # Read: already in (B, Hkv, end_pos, D) — SDPA format, no transpose
            K = self.cache_k[:, :, :end_pos, :]
            V = self.cache_v[:, :, :end_pos, :]

            Q = Q.transpose(1, 2)  # (B, H, S, D)

            is_causal = (seq_len > 1 and attn_mask is None)

            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                is_causal=is_causal,
                enable_gqa=(self.n_heads != self.n_kv_heads),
            )

            attn_out = attn_out.transpose(1, 2)
        else:
            # ── Training: FlashAttention ──
            if cu_seqlens is not None:
                # Document-aware packing: (B, S, H, D) rows are packed
                # sequences; flatten to varlen layout and let the kernel
                # mask cross-document attention via cumulative offsets.
                if _flash_attn_varlen_func is None:
                    raise RuntimeError(
                        "cu_seqlens was provided but flash_attn_varlen_func "
                        "is unavailable. Install flash-attn (>=2.x)."
                    )
                q_flat = Q.reshape(-1, Q.shape[-2], Q.shape[-1])
                k_flat = K.reshape(-1, K.shape[-2], K.shape[-1])
                v_flat = V.reshape(-1, V.shape[-2], V.shape[-1])
                eff_max = max_seqlen if max_seqlen is not None else seq_len
                attn_out = _flash_attn_varlen_func(
                    q_flat, k_flat, v_flat,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=eff_max,
                    max_seqlen_k=eff_max,
                    dropout_p=0.0,
                    causal=True,
                )
            else:
                attn_out = flash_attn_func(Q, K, V, causal=True)
        attn_out = attn_out.reshape(batch_size,seq_len,-1)
        attn_out = self.wo(attn_out)

        return attn_out


class TransformerDecoderBLK(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
                inference : bool = False,
                layer_idx : int = 0,
    ) -> None:
        """
            Transformer Decoder Block with pre-normalization.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                inference: whether to enable KV caching for inference
                layer_idx: index of this layer (0-indexed), passed to MoE for per-layer scaling
        """
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim,device = device)
        self.norm2 = RMS_Norm(config.hidden_dim,device = device)
        self.attention = Attention(config,device,inference)
        self.mlp = MoE(config, device, layer_idx=layer_idx)

    def forward(self,x,start_pos : int = 0,position_ids = None,attn_mask = None,
                cu_seqlens : torch.Tensor | None = None, max_seqlen : int | None = None):
        attention_output = self.attention(self.norm1(x),start_pos,position_ids,attn_mask,
                                          cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        # ── Kernel integration: residual add + RMSNorm fused into one
        #    kernel (S = x + attn_out, Y = RMSNorm(S)) — mathematically
        #    identical to `x = x + attention_output; normed = norm2(x)`. ──
        if FusedAddRMSNormFunction is not None and x.is_cuda:
            normed, x = FusedAddRMSNormFunction.apply(
                attention_output, x, self.norm2.scale, self.norm2.eps
            )
        else:
            x = x + attention_output
            normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x


class GPT_FLASH(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device | None = None,
                 inference : bool = False,
    ) -> None:
        """
            GPT model with Flash Attention and Mixture of Experts.

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                inference: whether to enable KV caching for inference
        """
        super().__init__()
        self.inference = inference
        self.norm = RMS_Norm(config.hidden_dim,device = device)
        self.embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_dim,
                device=device,
                dtype=config.dtype
        )
        self.layers = nn.ModuleList(
            [TransformerDecoderBLK(config, device, inference, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
        self.unembedding = nn.Linear(config.hidden_dim,config.vocab_size,device = device, dtype=config.dtype)

    def reset_cache(self, batch_size: int = 1) -> None:
        """Reset KV caches across all layers. Call before each new generation."""
        if self.inference:
            for layer in self.layers:
                layer.attention.reset_cache(batch_size)

    def forward(self,
                x : torch.Tensor,
                start_pos : int = 0,
                position_ids : torch.Tensor | None = None,
                attn_mask : torch.Tensor | None = None,
                cu_seqlens : torch.Tensor | None = None,
                max_seqlen : int | None = None,
        ) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x,start_pos,position_ids,attn_mask,
                      cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    # ── QK-Norm Scale Annealing ──────────────────────────────────
    def step_qk_scale_anneal(
        self,
        current_step: int,
        anneal_start_step: int,
        anneal_steps: int = 1000,
        target_max_scale: float = 1.0,
    ) -> float:
        """
        Gradually clamp QK-norm scale parameters from their current values
        toward `target_max_scale` over `anneal_steps` optimizer steps.

        Call this once per optimizer step during training.
        Returns the current clamp value being applied.
        """
        if current_step < anneal_start_step:
            return float('inf')  # no clamping yet

        progress = min(1.0, (current_step - anneal_start_step) / max(anneal_steps, 1))

        # Compute the tightest current scale across all QK norms
        all_max = []
        for layer in self.layers:
            all_max.append(layer.attention.q_norm.scale.data.max().item())
            all_max.append(layer.attention.k_norm.scale.data.max().item())
        current_max = max(all_max) if all_max else 1.0

        # Linearly interpolate clamp: start_max → target_max_scale
        clamp_val = current_max + progress * (target_max_scale - current_max)
        clamp_val = max(clamp_val, target_max_scale)  # never below target

        # Apply clamp to all QK-norm scale parameters
        with torch.no_grad():
            for layer in self.layers:
                layer.attention.q_norm.scale.data.clamp_(max=clamp_val)
                layer.attention.k_norm.scale.data.clamp_(max=clamp_val)

        return clamp_val

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

            # Estimated max attention logit: q_norm × k_norm / √head_dim
            max_logit_est = q_norm_est * k_norm_est / math.sqrt(head_dim)

            metrics[f"attn/layer_{i}/q_scale_max"] = q_max
            metrics[f"attn/layer_{i}/k_scale_max"] = k_max
            metrics[f"attn/layer_{i}/q_scale_mean"] = q_mean
            metrics[f"attn/layer_{i}/k_scale_mean"] = k_mean
            metrics[f"attn/layer_{i}/q_norm_est"] = q_norm_est
            metrics[f"attn/layer_{i}/k_norm_est"] = k_norm_est
            metrics[f"attn/layer_{i}/max_logit_est"] = max_logit_est

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

        # 3. Hidden state collapse (expensive — only at val_interval)
        if include_hidden_states and input_ids is not None:
            metrics.update(compute_hidden_state_telemetry(self, input_ids))

        return metrics
