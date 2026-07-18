import torch
from ..tokenizer import tokenizer
from dataclasses import dataclass, field
from typing import List, Optional
from .model_config import DatasetEntry, PhaseConfig

@dataclass
class ModelConfig:
    # ── Model architecture ────────────────────────────────────
    # ~818M total params, ~328M active per token
    #   Embeddings:  100.7M  (embed 50.3M + unembed 50.3M + final norm 1K)
    #   Per layer:    29.9M  × 24 layers = 717.7M
    #     Attention:   3.1M  (GQA 2:1 — 16 Q heads, 8 KV heads)
    #     MoE:        26.8M  (16 routed + 1 shared expert, top-3 routing)
    #   Active/tok:  328M    (attn + shared + 3 routed experts + embeds)
    vocab_size: int = tokenizer.vocab_size
    num_attn_heads: int = 16
    num_key_value_heads: int = 8       # GQA 2:1 ratio (each KV head shared by 2 Q heads)
    hidden_dim: int = 1024
    intermediate_size: int = 512
    ffn_dropout: float = 0.0
    head_dim: int = hidden_dim // num_attn_heads  # 64
    num_hidden_layers: int = 24
    num_experts: int = 16
    num_experts_per_tok: int = 3
    update_param: float = 2e-3
    route_scale: float = 1.0
    base: int = 10000
    initial_context_len: int = 2048
    max_context_len: int = 4096
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0
    scaling_factor: float = 1.0

    # ── Training hyperparameters ──────────────────────────────
    dropout: float = 0.0
    learning_rate: float = 3e-4
    batch_size: int = 8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 3000
    total_updates: int = 57200
    total_tokens: int = 120_000_000_000
    phase1_tokens: int = 85_000_000_000
    phase2_tokens: int = 35_000_000_000
    dtype = torch.bfloat16
    device = "cuda"
    local_rank: int = -1
    global_rank: int = -1