import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from ..scripts.tokenizer import tokenizer
from ..scripts.configs import ModelConfig
from flash_attn import flash_attn_func

class RMS_Norm(nn.Module):
    def __init__(self,
                 num_features,
                 eps : float = 1e-5,
                 device : torch.device|None = None
        ) -> None:
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
        super().__init__()
        self.dim = config.hidden_dim
        self.topk = config.num_experts_per_tok
        self.topk_groups = config.topk_groups
        self.n_groups = config.n_groups
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_dim),device = device))
        self.bias = nn.Parameter(torch.empty((config.num_experts), dtype=torch.float32,device = device))

    def forward(self,x : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor] :
        scores = F.linear(x,self.weight)
        scores = scores.softmax(dim = -1,dtype = torch.float32)
        original_scores = scores
        scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.shape[0],self.n_groups,-1)
            group_scores = scores.topk(2,dim = -1)[0].sum(dim = -1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.shape[0], self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights *= self.route_scale
        return weights.type_as(x),indices     
    

class MoE(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device|None = None
    ) -> None:
        super().__init__()
        self.dim = config.hidden_dim
        self.gate = Gate(config,device)
        self.n_routed_experts = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [Expert(config,device) 
             for _ in range(config.num_experts)]
        )
        self.shared_experts = MLPBlock(config,device)
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        inp_shape = x.shape
        x = x.view(-1,self.dim) 
        xprt_weights,xprt_idxs = self.gate(x)
        xprt_count = torch.zeros_like(x) 
        xprt_count = xprt_count.scatter_(-1, xprt_idxs, 1)
        routed_xprt_out = torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            if not xprt_count[:,i].any():
                continue
            batch_idx,expert_idx = torch.where(xprt_idxs == i)
            routed_xprt_out[batch_idx] += xprt_weights[batch_idx,expert_idx,None] * expert(x[batch_idx])
        mlp_out = routed_xprt_out + self.shared_experts(x)
        return mlp_out.reshape(inp_shape)
    

def apply_rope(x : torch.Tensor,
               cos : torch.Tensor,
               sin : torch.Tensor
    ) -> torch.Tensor:
    cos = cos.unsqueeze(-2).unsqueeze(0).to(x.device)
    sin = sin.unsqueeze(-2).unsqueeze(0).to(x.device)
    x1,x2 = torch.chunk(x,2,dim = -1)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1,o2],dim = -1)


class RotaryEmbedding(nn.Module):
    def __init__(self,
                 head_dim : int,
                 base : int,
                 dtype : torch.dtype,
                 initial_context_len : int = 4096,
                 ntk_alpha : float = 1.0,
                 ntk_beta : float = 32.0,
                 scaling_factor : float = 1.0,
                 device: torch.device | None = None
        ) -> None:
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
        concentration , inv_freq  = self._compute_concentration_and_inv_freq()
        pos = torch.arange(num_tokens,dtype = torch.float32 ,device = self.device)
        freqs = torch.einsum('i,j->ij',pos,inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos,sin

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor
        ) -> Tuple[torch.Tensor,torch.Tensor]:
        batch_size,seq_len,_,_ = q.shape
        cos,sin = self.compute_cos_sin(seq_len)

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
    ) -> None:
        super().__init__()
        self.n_heads = config.num_attn_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
       
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

        self.rope = RotaryEmbedding(
            config.head_dim,
            config.base,
            torch.float32,
            initial_context_len = config.initial_context_len,
            ntk_alpha = config.ntk_alpha,
            ntk_beta = config.ntk_beta,
            scaling_factor = config.scaling_factor,
            device = device
        )
        
    def forward(self,
                x : torch.Tensor
        ) -> torch.Tensor:
        batch_size,seq_len,hidden_dim = x.shape
        Q,K,V = self.wq(x),self.wk(x),self.wv(x)
        
        Q = Q.view(batch_size,seq_len,self.n_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        
        
        Q,K = self.rope(Q,K)
        attn_out = flash_attn_func(Q,K,V,causal = True)
        attn_out = attn_out.view(batch_size,seq_len,-1)
        attn_out = self.wo(attn_out)

        return attn_out.reshape(batch_size,seq_len,hidden_dim)
        
    
class TransformerDecoderBLK(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None
    ) -> None:
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim,device = device)
        self.norm2 = RMS_Norm(config.hidden_dim,device = device)
        self.attention = Attention(config,device)
        self.mlp = MoE(config,device)

    def forward(self,x): 
        x = x + self.attention(self.norm1(x))        
        x = x + self.mlp(self.norm2(x))
        return x
        
class GPT_FLASH(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device | None = None
    ) -> None:
        super().__init__()
        self.norm = RMS_Norm(config.hidden_dim,device = device)
        self.embeddings = nn.Embedding(
                config.vocab_size, 
                config.hidden_dim, 
                device=device,
                dtype=config.dtype
        )
        self.layers = nn.ModuleList(
            [TransformerDecoderBLK(config,device)
             for _ in range(config.num_hidden_layers)]
        )
        self.unembedding = nn.Linear(config.hidden_dim,config.vocab_size,device = device, dtype=config.dtype)

    def forward(self,
                x : torch.Tensor
        ) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.unembedding(x)  
        return x
