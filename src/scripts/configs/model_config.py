import torch
from ..tokenizer import tokenizer
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
        # ── Model architecture ────────────────────────────────────
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
        update_param : float = 2e-3
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
    max_epochs: int = 1                # How many times to iterate through this dataset.
                                       # 1 = single pass (default). Increase for smaller
                                       # datasets that should be repeated in the mix.


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

    # --- Validation ---
    val_interval: int = 2500              
    val_steps: int = 5000                 

    # --- Datasets ---
    datasets: List[DatasetEntry] = field(default_factory=list)

    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


# ──────────────────────────────────────────────────────────────
# Phase 1 (post-growth):  ~60B tokens  —  Code / Math / General Knowledge
# ──────────────────────────────────────────────────────────────
#   effective_batch = 36 * 8 = 288 seqs
#   tokens_per_step = 288 * 2048 ≈ 0.59M
#   total_steps     = 101,726  (~60B tokens)
#   lifetime tokens = 27B (pre-growth) + 60B (post) ≈ 87B
#
#   WSD schedule (stable_frac=0.895):
#     warmup:  0 → 499            (500 steps)
#     stable:  500 → 91,044       (90,545 steps)
#     decay:   91,045 → 101,726   (10,682 steps, ~10.5% of training)
#
#   Dataset mix (weights sum to 100):
#     starcoderdata-python         14   — primary code corpus (Python)
#     starcoderdata-javascript      6   — web scripting (JavaScript)
#     starcoderdata-java            5   — enterprise/Android (Java)
#     starcoderdata-typescript      4   — typed web (TypeScript)
#     starcoderdata-cpp             4   — systems programming (C++)
#     starcoderdata-c               3   — low-level systems (C)
#     starcoderdata-csharp          3   — .NET ecosystem (C#)
#     starcoderdata-go              3   — cloud-native (Go)
#     starcoderdata-rust            2   — safety-focused systems (Rust)
#     starcoderdata-php             1   — web back-end (PHP)
#     fineweb-edu-dedup            15   — deduplicated educational web (220B tokens)
#     cosmopedia-v2                12   — synthetic textbooks
#     openmath-instruct-2           7   — math reasoning (problem + solution)
#     numina-math-cot               4   — step-by-step math reasoning
#     stack-exchange-preferences    5   — technical Q&A (code↔NL bridge)
#     proof-pile-algebraic-stack    4   — mathematical code (11B tokens)
#     magicoder-oss-instruct        5   — code instruction (OSS-grounded)
#     openhermes-2.5                3   — instruction/chat corpus
#
#   Category breakdown:
#     Source Code          45%  (10 languages from starcoderdata)
#     General Knowledge    27%  (fineweb-edu-dedup + cosmopedia)
#     Code-adjacent        14%  (stack-exchange + algebraic-stack + magicoder)
#     Math/Reasoning       11%  (openmath + numina)
#     Instruction           3%  (openhermes)
# ──────────────────────────────────────────────────────────────
PHASE_1_CONFIG = PhaseConfig(
    phase_name="phase_1_post_growth",
    phase_num=1,
    peak_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=500,
    total_steps=101_726,
    scheduler_type="wsd",
    wsd_stable_frac=0.895,
    micro_batch_size=37,
    grad_accum_steps=8,
    grad_clip=1.0,
    val_interval=2000,
    val_steps=500,
    datasets=[
        # ── Source Code (45%) — Top 10 Languages ────────────
        DatasetEntry(
            name="starcoderdata-python",
            repo_id="bigcode/starcoderdata",
            weight=14,
            format_fn="starcoder",
            data_dir="python",
        ),
        DatasetEntry(
            name="starcoderdata-javascript",
            repo_id="bigcode/starcoderdata",
            weight=6,
            format_fn="starcoder",
            data_dir="javascript",
        ),
        DatasetEntry(
            name="starcoderdata-java",
            repo_id="bigcode/starcoderdata",
            weight=5,
            format_fn="starcoder",
            data_dir="java",
        ),
        DatasetEntry(
            name="starcoderdata-typescript",
            repo_id="bigcode/starcoderdata",
            weight=4,
            format_fn="starcoder",
            data_dir="typescript",
        ),
        DatasetEntry(
            name="starcoderdata-cpp",
            repo_id="bigcode/starcoderdata",
            weight=4,
            format_fn="starcoder",
            data_dir="cpp",
        ),
        DatasetEntry(
            name="starcoderdata-c",
            repo_id="bigcode/starcoderdata",
            weight=3,
            format_fn="starcoder",
            data_dir="c",
        ),
        DatasetEntry(
            name="starcoderdata-csharp",
            repo_id="bigcode/starcoderdata",
            weight=3,
            format_fn="starcoder",
            data_dir="c-sharp",
        ),
        DatasetEntry(
            name="starcoderdata-go",
            repo_id="bigcode/starcoderdata",
            weight=3,
            format_fn="starcoder",
            data_dir="go",
        ),
        DatasetEntry(
            name="starcoderdata-rust",
            repo_id="bigcode/starcoderdata",
            weight=2,
            format_fn="starcoder",
            data_dir="rust",
        ),
        DatasetEntry(
            name="starcoderdata-php",
            repo_id="bigcode/starcoderdata",
            weight=1,
            format_fn="starcoder",
            data_dir="php",
        ),
        # ── General Knowledge (27%) ─────────────────────────
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=15,
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ),
        DatasetEntry(
            name="cosmopedia-v2",
            repo_id="HuggingFaceTB/cosmopedia-v2",
            weight=12,
            format_fn="default",
            config_name="cosmopedia-v2",
        ),
        # ── Math/Reasoning (11%) ────────────────────────────
        DatasetEntry(
            name="openmath-instruct-2",
            repo_id="nvidia/OpenMathInstruct-2",
            weight=7,
            format_fn="openmath",
        ),
        DatasetEntry(
            name="numina-math-cot",
            repo_id="PrimeIntellect/NuminaMath-QwQ-CoT-5M",
            weight=4,
            format_fn="numina",
        ),
        # ── Code-Adjacent (14%) ─────────────────────────────
        DatasetEntry(
            name="stack-exchange-preferences",
            repo_id="HuggingFaceH4/stack-exchange-preferences",
            weight=5,
            format_fn="stackexchange",
        ),
        DatasetEntry(
            name="proof-pile-algebraic-stack",
            repo_id="EleutherAI/proof-pile-2",
            weight=4,
            format_fn="default",
            data_dir="algebraic-stack",
        ),
        DatasetEntry(
            name="magicoder-oss-instruct",
            repo_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
            weight=5,
            format_fn="magicoder",
            max_epochs=3,
        ),
        # ── Instruction (3%) ────────────────────────────────
        DatasetEntry(
            name="openhermes-2.5",
            repo_id="teknium/OpenHermes-2.5",
            weight=3,
            format_fn="openhermes",
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
    datasets=[],                    # to be filled after Phase 1
)


config = ModelConfig()

if __name__ == '__main__':
    print(config)
    print(PHASE_1_CONFIG)
    
