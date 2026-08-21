import math
import torch
from dataclasses import dataclass
from .model_config import PhaseConfig

@dataclass
class ModelConfig:
    vocab_size: int = 50_304  # Phi-2 tokenizer padded to 128-byte boundary
    num_attn_heads: int = 16
    num_key_value_heads: int = 8
    hidden_dim: int = 1024
    intermediate_size: int = 518
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

    loss_workspace_bytes: int = 512 * 1024 * 1024

    # RoPE — native 2K for base pretraining (YaRN is a separate post-training stage)
    base: int = 10000
    initial_context_len: int = 2048
    max_context_len: int = 2048
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

    def __post_init__(self):
        if self.vocab_size != 50_304:
            raise ValueError(
                f"GPT_FLASH requires vocab_size=50_304 (128-aligned), "
                f"got {self.vocab_size}"
            )
        if self.max_context_len > self.initial_context_len:
            expected_scale = self.max_context_len / self.initial_context_len
            if not math.isclose(self.scaling_factor, expected_scale):
                raise ValueError(
                    f"scaling_factor={self.scaling_factor} does not match "
                    f"max_context_len/initial_context_len = {expected_scale}"
                )



from .model_config import PhaseConfig, DatasetEntry

PRETRAINING_PHASE_CONFIG = PhaseConfig(
    phase_name="pretraining_phase",
    phase_num=1,
    peak_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    total_steps=34300,
    scheduler_type="wsd",
    wsd_stable_frac=0.795,
    micro_batch_size=16,
    grad_accum_steps=4,
    grad_clip=1.0,
    val_interval=2500,
    val_steps=5000,
    eval_suite_interval=0,
    datasets=[
        # ── Source Code (56%) ─────────────────────────────────
        DatasetEntry(
            name="starcoderdata-python",
            repo_id="bigcode/starcoderdata",
            weight=20,
            format_fn="starcoder",
            data_dir="python",
        ),
        DatasetEntry(
            name="starcoderdata-javascript",
            repo_id="bigcode/starcoderdata",
            weight=8,
            format_fn="starcoder",
            data_dir="javascript",
        ),
        DatasetEntry(
            name="starcoderdata-java",
            repo_id="bigcode/starcoderdata",
            weight=7,
            format_fn="starcoder",
            data_dir="java",
        ),
        DatasetEntry(
            name="starcoderdata-typescript",
            repo_id="bigcode/starcoderdata",
            weight=5,
            format_fn="starcoder",
            data_dir="typescript",
        ),
        DatasetEntry(
            name="starcoderdata-cpp",
            repo_id="bigcode/starcoderdata",
            weight=6,
            format_fn="starcoder",
            data_dir="cpp",
        ),
        DatasetEntry(
            name="starcoderdata-go",
            repo_id="bigcode/starcoderdata",
            weight=5,
            format_fn="starcoder",
            data_dir="go",
        ),
        DatasetEntry(
            name="starcoderdata-rust",
            repo_id="bigcode/starcoderdata",
            weight=5,
            format_fn="starcoder",
            data_dir="rust",
        ),
        # ── Educational Code (9%) ─────────────────────────────
        DatasetEntry(
            name="tiny-codes",
            repo_id="nampdn-ai/tiny-codes",
            weight=9,
            format_fn="tiny_codes",
            max_epochs=2,
        ),
        # ── CS Knowledge (17%) ────────────────────────────────
        DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=12,
            format_fn="stackexchange_programming_cs",
        ),
        DatasetEntry(
            name="dclm-edu",
            repo_id="HuggingFaceTB/dclm-edu",
            weight=5,
            format_fn="dclm_edu",
        ),
        # ── General Knowledge (18%) ───────────────────────────
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=15,
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ),
        DatasetEntry(
            name="wikipedia-en",
            repo_id="wikimedia/wikipedia",
            weight=3,
            format_fn="wikipedia",
            config_name="20231101.en",
        ),
    ],
)


config = ModelConfig()