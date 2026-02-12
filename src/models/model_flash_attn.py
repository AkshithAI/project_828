import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from ..scripts.configs.model_config import ModelConfig

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
        t,dtype = x.float(),x.dtype
        t = t * torch.rsqrt(torch.mean(t**2,dim = -1,keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
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
        
        return self.w2(self.dropout(swiglu(self.w1(x)) * self.w3(x)))
    

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
        
        return self.w2(self.dropout(swiglu(self.w1(x)) * self.w3(x)))
    

class Gate(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device|None = None
    ) -> None:
        """
            - Router/Gate module for Mixture of Experts.
            - Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy

            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.update_param = config.update_param
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias = False, device = device, dtype = config.dtype)
        self.register_buffer("bias", torch.zeros(config.num_experts, dtype=torch.float32, device=device))

    def update_bias(self, current_load: torch.Tensor) -> None:
        """Update bias in-place using Loss-Free Balancing rule."""
        load_float = current_load.float()
        e = torch.sign(load_float.mean() - load_float) 
        self.bias.add_(self.update_param * e)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See this paper for implementation details: https://arxiv.org/abs/2408.15664"""
        scores = self.router(x)
        scores = torch.sigmoid(scores)
        original_scores = scores
        biased_scores = scores + self.bias.detach().to(scores.dtype)
        indices = torch.topk(biased_scores, self.topk, dim=-1)[1]
        current_load = torch.bincount(indices.flatten(), minlength=self.num_experts)
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
                 device : torch.device|None = None
    ) -> None:
        """
            Mixture of Experts module with shared experts.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.gate = Gate(config,device)
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
        
        metrics = {
            **{f"expert_{i}": util_list[i] for i in range(self.num_experts)},
        }
        
        return metrics

    def reset_expert_counts(self):
        """Reset counters (call periodically during training)"""
        self.expert_counts.zero_()
        self.total_tokens = 0
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        inp_shape = x.shape
        x = x.view(-1,self.dim) 
        xprt_weights,xprt_idxs,counts = self.gate(x)

        self.expert_counts += counts
        self.total_tokens += x.shape[0] * self.n_routed_experts 
        routed_xprt_out = torch.zeros_like(x)

        for i,expert in enumerate(self.experts):
            if not counts[i]:
                continue
            batch_idx,expert_idx = torch.where(xprt_idxs == i)
            routed_xprt_out[batch_idx] += xprt_weights[batch_idx,expert_idx,None] * expert(x[batch_idx])
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
        """
            Multi-Head Attention with Grouped Query Attention and Flash Attention.
    
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
                batch_size, self.max_cache_len, self.n_kv_heads, self.head_dim,
                device=device, dtype=dtype,
            )
            self.cache_v = torch.zeros(
                batch_size, self.max_cache_len, self.n_kv_heads, self.head_dim,
                device=device, dtype=dtype,
            )

    def forward(self,
                x : torch.Tensor,
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
        Q,K = self.rope(Q,K,offset = start_pos,position_ids = position_ids)

        if self.inference:
            self.cache_k[:,start_pos:end_pos,:,:] = K
            self.cache_v[:,start_pos:end_pos,:,:] = V
            K = self.cache_k[:,:end_pos,:,:]
            V = self.cache_v[:,:end_pos,:,:]

            Q = Q.transpose(1,2)
            K = K.transpose(1,2)
            V = V.transpose(1,2)

            attn_out = F.scaled_dot_product_attention(
                Q,K,V,
                attn_mask=attn_mask,
                is_causal=(seq_len > 1 and attn_mask is None),
                enable_gqa=(self.n_heads != self.n_kv_heads)
            )
            attn_out = attn_out.transpose(1,2)
        else:
            from flash_attn import flash_attn_func
            attn_out = flash_attn_func(Q,K,V,causal = True)
        attn_out = attn_out.view(batch_size,seq_len,-1)
        attn_out = self.wo(attn_out)

        return attn_out
        
    
class TransformerDecoderBLK(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
                inference : bool = False,
    ) -> None:
        """
            Transformer Decoder Block with pre-normalization.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
                inference: whether to enable KV caching for inference
        """
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim,device = device)
        self.norm2 = RMS_Norm(config.hidden_dim,device = device)
        self.attention = Attention(config,device,inference)
        self.mlp = MoE(config,device)

    def forward(self,x,start_pos : int = 0,position_ids = None,attn_mask = None): 
        x = x + self.attention(self.norm1(x),start_pos,position_ids,attn_mask)        
        x = x + self.mlp(self.norm2(x))
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
            [TransformerDecoderBLK(config,device,inference)
             for _ in range(config.num_hidden_layers)]
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
        ) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x,start_pos,position_ids,attn_mask)
        x = self.norm(x)
        x = self.unembedding(x)  
        return x
