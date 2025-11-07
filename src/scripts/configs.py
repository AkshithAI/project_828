import torch
from .tokenizer import tokenizer

class ModelConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = tokenizer.vocab_size   
        self.num_attn_heads : int = 8
        self.num_key_value_heads : int = 2
        self.hidden_dim : int = 512 
        self.intermediate_size : int = 768
        self.ffn_dropout = 0.0
        self.head_dim = 8
        self.num_hidden_layers = 1
        self.num_experts = 4
        self.num_experts_per_tok = 2
        self.n_groups = 2
        self.route_scale = 1
        self.topk_groups = 1
        self.base = 10000
        self.initial_context_len = 2048
        self.ntk_alpha = 1.0
        self.ntk_beta = 32.0
        self.scaling_factor = 1.0
        # Training
        self.dropout = 0.0
        self.learning_rate = 3e-4
        self.batch_size = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "project_828/assets/ckpts" 

config = ModelConfig()