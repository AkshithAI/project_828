import torch
from ..tokenizer import tokenizer
from dataclasses import dataclass
from .model_config import PhaseConfig

@dataclass
class ModelConfig:
    vocab_size: int = tokenizer.vocab_size
    num_attn_heads: int = 16
    num_key_value_heads: int = 8
    hidden_dim: int = 1024
    intermediate_size: int = 520
    head_dim: int = 64
    num_hidden_layers: int = 24

    num_experts: int = 16
    num_experts_per_tok: int = 3
    route_scale: float = 1.0

    use_liger_moe: bool = True
    router_bias_update_rate: float = 2e-3
    update_param: float = 2e-3
    router_bias_max: float = 1.0
    moe_aux_loss_weight: float = 1e-3
    initializer_std: float = 0.02
    router_initializer_std: float = 0.01

    loss_workspace_bytes: int = 512 * 1024 * 1024

    base: int = 10000
    initial_context_len: int = 2048
    max_context_len: int = 4096
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0
    scaling_factor: float = 1.0

    ffn_dropout: float = 0.0
    dropout: float = 0.0

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    warmup_steps: int = 3000

    dtype = torch.bfloat16
    device = "cuda"

    # Weight init
    initializer_std: float = 0.02
    embedding_initializer_std: float = 0.02
    unembedding_initializer_std: float = 0.02
    router_initializer_std: float = 0.01

    residual_initializer_multiplier: float = 1.0
    norm_initial_scale: float = 1.0

    initialize_low_precision_from_fp32: bool = True
    zero_padding_embedding: bool = True
    initialization_seed: int = 1234



PRETRAINING_PHASE_CONFIG = PhaseConfig(
    phase_name="pretraining_phase",
    phase_num=1,
    peak_lr=3e-4,
    min_lr=3e-5,
    warmup_steps="--placeholder_for_warmup_steps--",
    total_steps="--placeholder_for_total_steps--",
    scheduler_type="wsd",
    wsd_stable_frac=0.795,
    micro_batch_size="--placeholder_for_micro_batch_size--",
    grad_accum_steps="--placeholder_for_grad_accum_steps--",
    grad_clip=1.0,
    val_interval="--placeholder_for_val_interval--",
    val_steps="--placeholder_for_val_steps--",
    eval_suite_interval="--placeholder_for_eval_suite_interval--",
    datasets=[
        # Yet to be filled
    ],
)


config = ModelConfig()