import torch
from ..tokenizer import tokenizer
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
        # Model architecture
        vocab_size : int = tokenizer.vocab_size   
        num_attn_heads : int = 12 
        num_key_value_heads : int = 6
        hidden_dim : int = 768  
        intermediate_size : int = 1776
        ffn_dropout : float = 0.0
        head_dim : float = hidden_dim // num_attn_heads 
        num_hidden_layers : int = 6 
        num_experts : int = 4
        num_experts_per_tok : int = 2 
        update_param : float = 1e-3
        route_scale : float = 1.0
        base : int = 10000
        initial_context_len : int = 2048
        max_context_len : int = 2048  # Set during 2nd phase of pretraining (Dynamic context scaling using YaRN)
        ntk_alpha : float = 1.0
        ntk_beta : float = 32.0
        scaling_factor : float = 1.0
        # Training
        dropout : float = 0.0
        learning_rate : float = 3e-4
        batch_size : int = 8
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        local_rank : int = -1
        global_rank : int = -1


@dataclass
class DatasetEntry:
    """Configuration for a single dataset in the mix."""
    name: str                          
    repo_id: str                       
    weight: int                        
    format_fn: str = "default"         
    config_name: Optional[str] = None  
    data_dir: Optional[str] = None     
    split: str = "train"               
    streaming: bool = True
    use_local_download: bool = False   # If True, .jsonl.zst shards are downloaded
                                       # to local cache before reading (legacy).
                                       # Default False: stream over HTTP with retry.


@dataclass
class PhaseConfig:
    """Per-phase training hyperparameters and dataset specification."""
    phase_name: str                    
    phase_num: int = 1                 

    # --- Learning rate schedule ---
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 34_300             
    scheduler_type: str = "wsd"           
    wsd_stable_frac: float = 0.76         

    # --- Batch / accumulation ---
    micro_batch_size: int = 128
    grad_accum_steps: int = 8             
    grad_clip: float = 1.0

    # --- Validation / early stopping ---
    val_interval: int = 2500              
    val_steps: int = 5000                 
    patience: int = 5                     

    # --- Datasets ---
    datasets: List[DatasetEntry] = field(default_factory=list)

    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


# ──────────────────────────────────────────────────────────────
# Phase 1:  72B tokens  —  Math / Science / General Knowledge
# ──────────────────────────────────────────────────────────────
#   effective_batch = 66 * 8 = 528 seqs  (bumped from 40 after step 17945)
#   tokens_per_step ≈ 528 * 2048 ≈ 1.08M
#   total_steps     = 73,655  (17,945 @ old bs + 55,710 @ new bs = 72B)
#
#   WSD schedule (stable_frac=0.60):
#     warmup:  0 → 1,999          (2,000 steps)
#     stable:  2,000 → 44,992     (42,993 steps)
#     decay:   44,993 → 73,655    (28,663 steps, ~39% of training)
#
#   Dataset mix (weights sum to 100):
#     openmath-instruct-2          25   — math reasoning (problem + solution)
#     proof-pile-algebraic-stack   12   — mathematical code (11B tokens)
#     proof-pile-open-web-math      9   — accessible math text (15B tokens)
#     proof-pile-arxiv              4   — arXiv papers, low weight (29B tokens)
#     fineweb-edu                  30   — general knowledge
#     cosmopedia-v2                20   — synthetic textbooks
# ──────────────────────────────────────────────────────────────
PHASE_1_CONFIG = PhaseConfig(
    phase_name="phase_1_math_science",
    phase_num=1,
    peak_lr=3.5e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    total_steps=73_655,
    scheduler_type="wsd",
    wsd_stable_frac=0.60,
    micro_batch_size=66,
    grad_accum_steps=8,
    grad_clip=1.0,
    val_interval=1500,
    val_steps=3000,
    patience=5,
    datasets=[
        DatasetEntry(
            name="openmath-instruct-2",
            repo_id="nvidia/OpenMathInstruct-2",
            weight=25,
            format_fn="openmath",
        ),
        DatasetEntry(
            name="proof-pile-algebraic-stack",
            repo_id="EleutherAI/proof-pile-2",
            weight=12,
            format_fn="default",
            data_dir="algebraic-stack",
        ),
        DatasetEntry(
            name="proof-pile-open-web-math",
            repo_id="EleutherAI/proof-pile-2",
            weight=9,
            format_fn="default",
            data_dir="open-web-math",
        ),
        DatasetEntry(
            name="proof-pile-arxiv",
            repo_id="EleutherAI/proof-pile-2",
            weight=4,
            format_fn="default",
            data_dir="arxiv",
        ),
        DatasetEntry(
            name="fineweb-edu",
            repo_id="HuggingFaceFW/fineweb-edu",
            weight=30,
            format_fn="fineweb_edu",
            config_name="sample-100BT",
        ),
        DatasetEntry(
            name="cosmopedia-v2",
            repo_id="HuggingFaceTB/cosmopedia-v2",
            weight=20,
            format_fn="default",
            config_name="cosmopedia-v2",
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Phase 2:  18B tokens  —  Code / Instruction / Replay
# (Placeholder — user will configure after Phase 1 completes)
# ──────────────────────────────────────────────────────────────
PHASE_2_CONFIG = PhaseConfig(
    phase_name="phase_2_code",
    phase_num=2,
    peak_lr=1e-4,
    min_lr=1e-5,
    warmup_steps=500,
    total_steps=8_600,
    scheduler_type="cosine",
    wsd_stable_frac=0.0,           
    micro_batch_size=32,
    grad_accum_steps=8,
    grad_clip=1.0,
    val_interval=10000,
    val_steps=3000,
    patience=5,
    datasets=[],                    # to be filled after Phase 1
)


config = ModelConfig()

if __name__ == '__main__':
    print(config)
    print(PHASE_1_CONFIG)
    
