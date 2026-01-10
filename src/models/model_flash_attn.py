import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from ..scripts.configs import ModelConfig
from flash_attn import flash_attn_func

class RMS_Norm(nn.Module):
    def __init__(self,
                 num_features,
                 eps : float = 1e-5,
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
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
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
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
        )
        
        self.dropout = nn.Dropout(config.ffn_dropout)
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        
        return self.w2(self.dropout(swiglu(self.w1(x) * self.w3(x))))
    

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
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
        )
        
        self.dropout = nn.Dropout(config.ffn_dropout)
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        
        return self.w2(self.dropout(swiglu(self.w1(x) * self.w3(x))))
    

class Gate(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device|None = None
    ) -> None:
        """
            Router/Gate module for Mixture of Experts.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.dim = config.hidden_dim
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_dim), device=device, dtype=config.dtype))
        self.bias = nn.Parameter(torch.empty((config.num_experts), dtype=config.dtype, device=device))

    def forward(self,x : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor] :
        scores = F.linear(x,self.weight)
        scores = scores.softmax(dim = -1,dtype = torch.float32)
        original_scores = scores
        scores = scores + self.bias
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights *= self.route_scale
        return weights.type_as(x),indices     
    

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
            # Individual expert utilization (percentage)
            **{f"expert_{i}": util_list[i] for i in range(self.num_experts)},
            # Summary statistics
            "expert_util_mean": sum(util_list) / len(util_list),
            "expert_util_max": max(util_list),
            "expert_util_min": min(util_list),
            "expert_util_std": (sum((x - sum(util_list)/len(util_list))**2 for x in util_list) / len(util_list)) ** 0.5,
            # Load balance score (higher = more balanced, 100 = perfect)
            "load_balance_score": (min(util_list) / max(util_list) * 100) if max(util_list) > 0 else 0,
        }
        return metrics

    def reset_expert_counts(self):
        """Reset counters (call periodically during training)"""
        self.expert_counts.zero_()
        self.total_tokens = 0
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        inp_shape = x.shape
        x = x.view(-1,self.dim) 
        xprt_weights,xprt_idxs = self.gate(x)
        counts = torch.bincount(xprt_idxs.flatten(), minlength=self.num_experts)
        self.expert_counts += counts
        self.total_tokens += x.shape[0] * self.n_routed_experts 
        routed_xprt_out = torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            mask = (xprt_idxs == i).any(dim=-1)
            if not mask.any():
                continue
            batch_idx,expert_idx = torch.where(xprt_idxs == i)
            routed_xprt_out[batch_idx] += xprt_weights[batch_idx,expert_idx,None] * expert(x[batch_idx])
        mlp_out = routed_xprt_out + self.shared_experts(x)
        return mlp_out.reshape(inp_shape)


def apply_rope(x : torch.Tensor,
               cos : torch.Tensor,
               sin : torch.Tensor
    ) -> torch.Tensor:
    cos = cos.unsqueeze(-2).unsqueeze(0).to(x.device).to(x.dtype)
    sin = sin.unsqueeze(-2).unsqueeze(0).to(x.device).to(x.dtype)
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
                 max_context_len : int = 4096,
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
        self.cos,self.sin = self.compute_cos_sin(max_context_len)

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
                offset : int
        ) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size,seq_len,_,_ = q.shape
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
            self.register_buffer("cache_k", torch.zeros(
                1, config.seq_len, config.num_key_value_heads, config.head_dim, device = device , dtype = config.dtype
            ), persistent=False)
            self.register_buffer("cache_v", torch.zeros(
                1, config.seq_len, config.num_key_value_heads, config.head_dim, device = device , dtype = config.dtype
            ), persistent=False)
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
        
    def forward(self,
                x : torch.Tensor,
                start_pos : int = 0,
        ) -> torch.Tensor:
        batch_size,seq_len,_ = x.shape
        end_pos = start_pos + seq_len
        Q,K,V = self.wq(x),self.wk(x),self.wv(x)
        
        Q = Q.view(batch_size,seq_len,self.n_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        
        # Test 1 : Q and K norms along head_dim
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q,K = self.rope(Q,K,offset = start_pos)          
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
                is_causal=(seq_len > 1),
                enable_gqa=(self.n_heads != self.n_kv_heads)
            )
            attn_out = attn_out.transpose(1,2)
        else:
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

    def forward(self, x, start_pos : int = 0): 
        x = x + self.attention(self.norm1(x),start_pos)        
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

    def forward(self,
                x : torch.Tensor,
                start_pos : int = 0,
        ) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x,start_pos)
        x = self.norm(x)
        x = self.unembedding(x)  
        return x
