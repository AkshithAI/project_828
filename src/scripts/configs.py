import torch
from .tokenizer import tokenizer
from dataclasses import dataclass

@dataclass
class ModelConfig:
        # Model architecture
        vocab_size : int = tokenizer.vocab_size   
        num_attn_heads : int = 8
        num_key_value_heads : int = 2
        hidden_dim : int = 512 
        intermediate_size : int = 768
        ffn_dropout : float = 0.0
        head_dim : float = hidden_dim // num_attn_heads 
        num_hidden_layers : int = 1
        num_experts : int = 4
        num_experts_per_tok : int = 2
        n_groups : int = 2
        route_scale : int = 1
        topk_groups : int = 1
        base : int = 10000
        initial_context_len : int = 2048
        ntk_alpha : float = 1.0
        ntk_beta : float = 32.0
        scaling_factor : float = 1.0
        # Training
        dropout : float = 0.0
        learning_rate : float = 3e-4
        batch_size : int = 8
        seq_len : int = 2048
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        local_rank : int = -1
        global_rank : int = -1

if __name__ == '__main__':
    config = ModelConfig()
    