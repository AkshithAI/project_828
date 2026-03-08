import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple
from dataclasses import dataclass
from ..scripts.tokenizer import tokenizer

@dataclass
class ModelConfig:
        vocab_size : int = tokenizer.vocab_size   
        num_attn_heads : int = 12 
        num_key_value_heads : int = 6
        hidden_dim : int = 768  
        intermediate_size : int = 760
        ffn_dropout : float = 0.0
        head_dim : float = hidden_dim // num_attn_heads 
        num_hidden_layers : int = 24 
        num_experts : int = 4
        num_experts_per_tok : int = 2 
        update_param : float = 1e-3
        route_scale : float = 1.0
        base : int = 10000
        initial_context_len : int = 2048
        max_context_len : int = 2048 
        ntk_alpha : float = 1.0
        ntk_beta : float = 32.0
        scaling_factor : float = 1.0
 

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
                eps: a small factor to handle divide-by-zero error
                device: torch device to place the module on
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
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
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
            config.hidden_dim, 2 * config.intermediate_size, device = device, dtype=config.dtype
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
        self.bias.clamp_(-10.0, 10.0)
    
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

        if self.training:
            with torch.no_grad():
                self.update_bias(current_load)

        return weights.type_as(x), indices    
    

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
        self.experts = nn.ModuleList(
            [Expert(config,device) 
             for _ in range(config.num_experts)]
        )
        self.shared_experts = MLPBlock(config,device) 
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        inp_shape = x.shape
        x = x.view(-1,self.dim) 
        xprt_weights,xprt_idxs = self.gate(x)
        routed_xprt_out = torch.zeros_like(x)

        for i,expert in enumerate(self.experts):
            batch_idx,expert_idx = torch.where(xprt_idxs == i)
            routed_xprt_out[batch_idx] += xprt_weights[batch_idx,expert_idx,None] * expert(x[batch_idx])
        mlp_out = routed_xprt_out + self.shared_experts(x)

        return mlp_out.reshape(inp_shape)
    

def apply_rope(x : torch.Tensor,
               cos : torch.Tensor,
               sin : torch.Tensor
    ) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(-2).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(-2).to(x.device)
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
        """
            Rotary Position Embedding with YaRN scaling support.
    
            Args:
                head_dim: dimension of each attention head
                base: base frequency for rotary embeddings
                dtype: data type for computations
                initial_context_len: original context length for YaRN scaling
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
        batch_size,seq_len,_,_,_ = q.shape
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
    
    
def expand_kv(
              K : torch.Tensor,
              V : torch.Tensor,
              S : torch.Tensor,
              q_shape
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    batch_size,seq_len,n_heads,q_mult,head_dim = q_shape
    assert K.shape == (batch_size,seq_len,n_heads,head_dim)
    assert V.shape == (batch_size,seq_len,n_heads,head_dim)
    
    K = K[:,:,:,None,:].expand(batch_size,seq_len,n_heads,q_mult,head_dim)
    V = V[:,:,:,None,:].expand(batch_size,seq_len,n_heads,q_mult,head_dim)
    S = S.reshape(1,n_heads,q_mult,1,1).expand(batch_size,-1,-1,seq_len,-1)
    
    return K.contiguous(),V.contiguous(),S.contiguous()

class Attention(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None,
    ) -> None:
        """
            Multi-Head Attention with Grouped Query Attention and Attention Sinks.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.n_heads = config.num_attn_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attn_heads, device=device, dtype=config.dtype)
        )
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

        self.q_norm = RMS_Norm(config.head_dim, device = device)
        self.k_norm = RMS_Norm(config.head_dim, device = device) 

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
        
        Q = Q.view(batch_size,seq_len,self.n_kv_heads,self.n_heads // self.n_kv_heads,self.head_dim)
        K = K.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        V = V.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        Q,K = self.q_norm(Q),self.k_norm(K)
        Q,K = self.rope(Q,K)
        K,V,S = expand_kv(K,V,self.sinks,Q.shape)
        mask = torch.triu(Q.new_full((seq_len,seq_len),-float('inf')),diagonal = 1)

        scores = torch.einsum("bqhmd,bkhmd->bhmqk",Q,K) / math.sqrt(self.head_dim)
        scores += mask[None,None,None,:,:]
        scores = torch.cat([scores,S],dim = -1)
        
        attn_scores = torch.softmax(scores,dim = -1)
        attn_scores = attn_scores[...,:-1]

        attn_out = torch.einsum("bhmqk,bvhmd->bqhmd",attn_scores,V)
        attn_out = attn_out.reshape(batch_size,seq_len,-1).to(x.dtype)
        attn_out = self.wo(attn_out) 
        
        return attn_out.reshape(batch_size,seq_len,hidden_dim)
        
    
class TransformerDecoderBLK(nn.Module):
    def __init__(self,
                config : ModelConfig,
                device : torch.device | None = None
    ) -> None:
        """
            Transformer Decoder Block with pre-normalization.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
        """
        super().__init__()
        self.norm1 = RMS_Norm(config.hidden_dim,device = device)
        self.norm2 = RMS_Norm(config.hidden_dim,device = device)
        self.attention = Attention(config,device)
        self.mlp = MoE(config,device)

    def forward(self,x): 
        x = x + self.attention(self.norm1(x))        
        x = x + self.mlp(self.norm2(x))
        return x
        
class GPT(nn.Module):
    def __init__(self,
                 config : ModelConfig,
                 device : torch.device | None = None
    ) -> None:
        """
            GPT model with Mixture of Experts.
    
            Args:
                config: ModelConfig object containing model hyperparameters
                device: torch device to place the module on
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
