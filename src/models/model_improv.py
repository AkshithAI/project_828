import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict
from ..scripts.configs.model_config import ModelConfig
import torch.distributed as dist
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
try:
    from triton_kernels import fused_moe_forward as triton_moe_forward
    TRITON_MOE_AVAILABLE = True
except ImportError:
    TRITON_MOE_AVAILABLE = False


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

def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 5.0):
    """SwiGLU activation with soft-clamping for gradient stability.

    Splits *x* along the last dimension into a gating half and a linear half,
    applies soft-clamped SiLU-style gating, and returns the fused result.

    Args:
        x: Input tensor whose last dimension is even.
        alpha: Temperature coefficient for the sigmoid gate.
        limit: Soft-clamping bound applied before gating.

    Returns:
        Activated tensor with last dimension halved.
    """
    x_glu, x_linear = x.chunk(2, dim=-1)
    x_glu_s = soft_clamp(x_glu, limit)
    x_lin_s = soft_clamp(x_linear, limit)
    return x_glu_s * torch.sigmoid(alpha * x_glu_s) * (x_lin_s + 1)

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
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype, bias=False
        )
        self.w2 = nn.Linear(
            config.intermediate_size, config.hidden_dim, device = device, dtype=config.dtype, bias=False
        )

        self.resid_scale = math.sqrt(0.5)
        self.dropout = nn.Dropout(config.ffn_dropout)
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = swiglu(h)
        h = self.dropout(h)
        out = self.w2(h)
        return out * self.resid_scale 
    

class Gate(nn.Module):
    def __init__(self, config, device=None, layer_idx=0):
        """Initialise the routing gate.

        Args:
            config: :class:`ModelConfig` with MoE hyper-parameters.
            device: Torch device for parameter placement.
            layer_idx: Index of the parent transformer layer (used for
                per-layer bias update scaling).
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.num_experts = config.num_experts
        self.router = nn.Linear(self.dim, self.num_experts,
                                bias=False, device=device, dtype=config.dtype)
        self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer("load_accum", torch.zeros(self.num_experts, dtype=torch.float32, device=device))
        self.last_routing_probs = None

        num_layers = config.num_hidden_layers
        layer_scale = 1.0 + 0.5 * (layer_idx / max(num_layers - 1, 1))
        self.effective_update = config.update_param * layer_scale

    def forward(self, x: torch.Tensor):

        scores = torch.sigmoid(self.router(x.float()))          # (T, N)
        self.last_routing_probs = scores.detach()

        biased = scores + self.bias.to(scores.dtype)
        indices = torch.topk(biased, self.topk, dim=-1)[1]      # (T, K)

        current_load = torch.bincount(indices.flatten(),
                                      minlength=self.num_experts)

        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        weights = (weights * self.route_scale).to(x.dtype)

        if self.training:
            with torch.no_grad():
                self.load_accum += current_load.float()

        T = x.shape[0]
        f = current_load.float() / (T * self.topk)              
        P = scores.mean(dim=0)                                  
        aux_loss = self.num_experts * torch.sum(f * P)

        return weights.type_as(x), indices, current_load, aux_loss

    def commit_bias_update(self):
        """Apply a sign-based bias correction to rebalance expert utilisation.

        Aggregates load counts across distributed workers (if available),
        then nudges each expert's bias toward the mean load. Should be
        called once per training step after the backward pass.
        """
        load = self.load_accum.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)
        mean = load.mean()
        e = torch.sign(mean - load)
        self.bias.data.add_(self.effective_update * e)
        self.bias.data.clamp_(-10.0, 10.0)
        self.load_accum.zero_()

    def commit_bias_update(self):
        """Apply a sign-based bias correction to rebalance expert utilisation.

        Aggregates load counts across distributed workers (if available),
        then nudges each expert's bias toward the mean load. Should be
        called once per training step after the backward pass.
        """
        load = self.load_accum.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)
        mean = load.mean()
        e = torch.sign(mean - load)                            
        self.bias.data.add_(self.effective_update * e)
        self.bias.data.clamp_(-10.0, 10.0)
        self.load_accum.zero_()  
    

class MoE(nn.Module):
    def __init__(self, config, device=None, layer_idx=0):
        """Initialise MoE layer.

        Args:
            config: :class:`ModelConfig` with MoE hyper-parameters.
            device: Torch device for parameter placement.
            layer_idx: Transformer layer index forwarded to :class:`Gate`.
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.gate = Gate(config, device, layer_idx=layer_idx)
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([MLPBlock(config, device) for _ in range(self.num_experts)])
        self.shared_expert = MLPBlock(config, device)
        self.register_buffer('expert_counts', torch.zeros(self.num_experts, dtype=torch.long, device=device))
        self.total_tokens = 0

    def reset_expert_counts(self):
        """Zero the running expert-utilisation counters."""
        self.expert_counts.zero_()
        self.total_tokens = 0

    def forward(self, x: torch.Tensor):
        inp_shape = x.shape
        x_flat = x.view(-1, self.dim)

        if TRITON_MOE_AVAILABLE:
            # TritonMoE fused_moe_forward (bassrehab/triton-kernels)
            # Signature:
            #   fused_moe_forward(
            #       hidden_states:  Tensor  (T, D)            — input tokens
            #       router_weight:  Tensor  (E, D)            — router projection
            #       w_gate:         Tensor  (E, ffn_dim, D)   — expert gate weights
            #       w_up:           Tensor  (E, ffn_dim, D)   — expert up weights
            #       w_down:         Tensor  (E, D, ffn_dim)   — expert down weights
            #       num_experts:    int                       — number of experts
            #       top_k:          int                       — experts per token
            #       gating:         str                       — "softmax" or "sigmoid"
            #   ) -> Tuple[Tensor, Tensor, Tensor]
            #       output:       (T, D)     — MoE output
            #       top_k_indices: (T, K)    — expert assignments
            #       top_k_weights: (T, K)    — gating weights
            #
            # NOTE: The kernel handles routing + expert dispatch internally.
            # Shared expert, gate bias, route_scale, resid_scale, and aux loss
            # are NOT handled by the kernel — we apply them here.

            # Stack per-expert weights into 3D tensors for the kernel.
            # MLPBlock.w1 is fused gate+up: (2*I, D) → split into gate (I, D) and up (I, D)
            intermediate_size = self.experts[0].w2.weight.shape[1]  # I
            w_gate_list, w_up_list = [], []
            for expert in self.experts:
                w1_full = expert.w1.weight                         # (2*I, D)
                w_gate_list.append(w1_full[:intermediate_size])    # (I, D)
                w_up_list.append(w1_full[intermediate_size:])      # (I, D)
            w_gate_3d = torch.stack(w_gate_list)                   # (E, I, D)
            w_up_3d = torch.stack(w_up_list)                       # (E, I, D)
            w_down_3d = torch.stack([e.w2.weight for e in self.experts])  # (E, D, I)

            triton_out, top_k_indices, top_k_weights = triton_moe_forward(
                x_flat, self.gate.router.weight,
                w_gate_3d, w_up_3d, w_down_3d,
                self.num_experts, self.gate.topk,
                gating="sigmoid",
            )

            # Apply resid_scale (MLPBlock scales output by √0.5)
            triton_out = triton_out * self.experts[0].resid_scale

            # Add shared expert contribution (not part of TritonMoE kernel)
            triton_out = triton_out + self.shared_expert(x_flat)

            # Compute aux loss from kernel's routing decisions
            T = x_flat.shape[0]
            scores = torch.sigmoid(self.gate.router(x_flat.float()))
            current_load = torch.bincount(
                top_k_indices.flatten(), minlength=self.num_experts
            )
            f = current_load.float() / (T * self.gate.topk)
            P = scores.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(f * P)

            # Update expert counts for telemetry
            if self.training:
                self.expert_counts += current_load.to(self.expert_counts.dtype)
                self.total_tokens += T * self.gate.topk

            return triton_out.view(*inp_shape), aux_loss

        weights, indices, counts, aux_loss = self.gate(x_flat)
        if self.training:
            self.expert_counts += counts.to(self.expert_counts.dtype)
            self.total_tokens += x_flat.shape[0] * self.gate.topk

        flat_idx = indices.reshape(-1)                            # (T*K,)
        flat_weights = weights.reshape(-1, 1)                     # (T*K, 1)
        token_idx = torch.arange(x_flat.shape[0], device=x.device)
        token_idx = token_idx.unsqueeze(1).expand_as(indices).reshape(-1)

        sort_order = flat_idx.argsort(stable=True)
        sorted_expert_ids = flat_idx[sort_order]
        sorted_token_idx  = token_idx[sort_order]
        sorted_weights    = flat_weights[sort_order]
        sorted_x          = x_flat[sorted_token_idx]

        boundaries = torch.searchsorted(
            sorted_expert_ids.contiguous(),
            torch.arange(self.num_experts + 1, device=x.device),
        ).tolist()

        sorted_out = torch.zeros_like(sorted_x)
        for i, expert in enumerate(self.experts):
            start, end = boundaries[i], boundaries[i + 1]
            if start < end:
                sorted_out[start:end] = expert(sorted_x[start:end])

        sorted_out = sorted_out * sorted_weights
        routed = torch.zeros_like(x_flat)
        routed.scatter_add_(
            0, sorted_token_idx.unsqueeze(1).expand_as(sorted_out), sorted_out
        )
        mlp_out = routed + self.shared_expert(x_flat)
        return mlp_out.view(*inp_shape), aux_loss


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
        q = apply_rope(q,cos,sin)
        q = q.reshape(query_shape)

        key_shape = k.shape
        k = k.view(batch_size,seq_len,-1,self.head_dim)
        k = apply_rope(k,cos,sin)
        k = k.reshape(key_shape)

        return q,k


class Attention(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
                inference : bool = False,
    ) -> None:
        """Initialise the attention layer.

        Args:
            config: :class:`ModelConfig` with attention hyper-parameters.
            device: Torch device for parameter placement.
            inference: If ``True``, allocate a KV cache and use
                ``scaled_dot_product_attention`` instead of FlashAttention.
        """
        super().__init__()
        self.n_heads = config.num_attn_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inference = inference
        self.max_cache_len = config.max_context_len

        self.wq = nn.Linear(
            config.hidden_dim, config.num_attn_heads * config.head_dim, device = device, dtype = config.dtype, bias=False
        )
        self.wk = nn.Linear(
            config.hidden_dim,config.num_key_value_heads * config.head_dim , device = device, dtype = config.dtype, bias=False
        )
        self.wv = nn.Linear(
            config.hidden_dim,config.num_key_value_heads * config.head_dim , device = device, dtype = config.dtype, bias=False
        )
        self.wo = nn.Linear(
            config.num_attn_heads * config.head_dim, config.hidden_dim, device = device, dtype = config.dtype, bias=False
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
                batch_size, self.max_cache_len, self.n_kv_heads, self.head_dim,
                device=device, dtype=dtype,
            )
            self.cache_v = torch.zeros(
                batch_size, self.max_cache_len, self.n_kv_heads, self.head_dim,
                device=device, dtype=dtype,
            )

    def forward(self,
                x : torch.Tensor,
                cos : torch.Tensor,
                sin : torch.Tensor,
                start_pos : int = 0,
                position_ids : torch.Tensor | None = None,
                attn_mask : torch.Tensor | None = None,
        ) -> torch.Tensor:
        batch_size,seq_len,_ = x.shape
        end_pos = start_pos + seq_len

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
        Q,K = self.rope(Q, K, cos, sin, offset = start_pos, position_ids = position_ids)

        if self.inference:
            self.cache_k[:,start_pos:end_pos,:,:] = K
            self.cache_v[:,start_pos:end_pos,:,:] = V
            K = self.cache_k[:,:end_pos,:,:]
            V = self.cache_v[:,:end_pos,:,:]

            Q = Q.transpose(1, 2)  # (B, H, S, D)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(
                Q,K,V,
                attn_mask=attn_mask,
                is_causal=(seq_len > 1 and attn_mask is None),
                enable_gqa=(self.n_heads != self.n_kv_heads)
            )

            attn_out = attn_out.transpose(1,2)
        else:
            # ── Training: FlashAttention with native softcap ──
            attn_out = flash_attn_func(
                Q, K, V, causal=True,
            )
        attn_out = attn_out.reshape(batch_size,seq_len,-1)
        attn_out = self.wo(attn_out)

        return attn_out
        
    
class TransformerDecoderBLK(nn.Module):
    def __init__(self, config, device=None, inference=False, layer_idx=0):
        """Initialise a decoder block.

        Args:
            config: :class:`ModelConfig` with model hyper-parameters.
            device: Torch device for parameter placement.
            inference: Whether to enable KV caching in the attention layer.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim, device=device)
        self.norm2 = RMS_Norm(config.hidden_dim, device=device)
        self.attention = Attention(config, device, inference)
        self.mlp = MoE(config, device, layer_idx=layer_idx)
        self.resid_scale = math.sqrt(0.5)

    def forward(self, x, cos, sin, start_pos=0, position_ids=None, attn_mask=None):
        attn_out = self.attention(self.norm1(x), cos, sin, start_pos, position_ids, attn_mask)
        x = x + attn_out * self.resid_scale
        mlp_out, aux_loss = self.mlp(self.norm2(x))
        x = x + mlp_out * self.resid_scale
        return x, aux_loss
        
class GPT_FLASH(nn.Module):
    def __init__(self, config, device=None, inference=False):
        """Initialise the model.

        Args:
            config: :class:`ModelConfig` defining architecture hyper-parameters.
            device: Torch device for parameter and buffer placement.
            inference: If ``True``, enables KV caching in all attention layers.
        """
        super().__init__()
        self.inference = inference
        self.num_hidden_layers = config.num_hidden_layers
        self.moe_aux_loss_weight = getattr(config, "moe_aux_loss_weight", 0.01)

        self.norm = RMS_Norm(config.hidden_dim, device=device)
        self._rope = RotaryEmbedding(
            config.head_dim, config.base, torch.float32,
            initial_context_len=config.initial_context_len,
            max_context_len=config.max_context_len,
            ntk_alpha=config.ntk_alpha, ntk_beta=config.ntk_beta,
            scaling_factor=config.scaling_factor, device=device,
        )
        cos, sin = self._rope.compute_cos_sin(config.max_context_len)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim,
                                       device=device, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [TransformerDecoderBLK(config, device, inference, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
        self.unembedding = nn.Linear(config.hidden_dim, config.vocab_size,
                                     bias=False, device=device, dtype=config.dtype)

    def reset_cache(self, batch_size: int = 1) -> None:
        """Reset the KV cache in every attention layer for *batch_size*.

        Args:
            batch_size: Number of sequences in the upcoming batch.
        """
        if self.inference:
            for layer in self.layers:
                layer.attention.reset_cache(batch_size)

    def forward(self, x, start_pos=0, position_ids=None, attn_mask=None):
        x = self.embeddings(x)
        total_aux_loss = x.new_zeros(())
        for layer in self.layers:
            x, aux_loss = layer(x, self.cos, self.sin, start_pos, position_ids, attn_mask)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.norm(x)
        logits = self.unembedding(x)

        aux_loss = self.moe_aux_loss_weight * (total_aux_loss / self.num_hidden_layers)
        return logits, aux_loss


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

        # 2. Weight update ratios (needs optimizer state)
        if optimizer is not None:
            metrics.update(compute_weight_update_ratios(self, optimizer, lr))

        # 3. Hidden state collapse (expensive — only at val_interval)
        if include_hidden_states and input_ids is not None:
            metrics.update(compute_hidden_state_telemetry(self, input_ids))

        return metrics
