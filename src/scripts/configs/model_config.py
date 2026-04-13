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
# Phase 1 (post-growth):  ~60B tokens  —  Code / CS Engineering / Knowledge
# ──────────────────────────────────────────────────────────────
#   CONTEXT: First ~19B tokens used a 50% math-heavy mix which caused
#   LaTeX contamination and undertrained code/CS skills. The remaining
#   ~26B tokens course-correct hard toward code + CS/engineering.
#
#   effective_batch = 37 * 8 = 296 seqs
#   tokens_per_step = 296 * 2048 ≈ 0.61M
#   total_steps     = 101,726  (~60B tokens)
#   lifetime tokens = 27B (pre-growth) + 60B (post) ≈ 87B
#
#   WSD schedule (stable_frac=0.895):
#     warmup:  0 → 499            (500 steps)
#     stable:  500 → 91,044       (90,545 steps)
#     decay:   91,045 → 101,726   (10,682 steps, ~10.5% of training)
#
#   Dataset mix (weights sum to 100):
#     starcoderdata-python         16   — primary code corpus (Python)
#     starcoderdata-javascript      7   — web scripting (JavaScript)
#     starcoderdata-java            5   — enterprise/Android (Java)
#     starcoderdata-typescript      4   — typed web (TypeScript)
#     starcoderdata-cpp             5   — systems programming (C++)
#     starcoderdata-c               3   — low-level systems (C)
#     starcoderdata-csharp          3   — .NET ecosystem (C#)
#     starcoderdata-go              3   — cloud-native (Go)
#     starcoderdata-rust            2   — safety-focused systems (Rust)
#     starcoderdata-php             2   — web back-end (PHP)
#     fineweb-edu-dedup            18   — deduplicated educational web (220B tokens, includes CS tutorials/docs)
#     finemath-4plus                6   — highest quality math web (9.6B tokens, decontaminated)
#     finemath-3plus                4   — broader math web content (34B tokens)
#     stackexchange-programming-cs 10   — strict programming/CS StackExchange Q&A
#     opencodeinstruct             12   — 5M execution-verified Python code instructions
#
#   Category breakdown:
#     Source Code          50%  (10 languages from starcoderdata)
#     CS/Engineering       22%  (stackexchange + opencodeinstruct, all LaTeX-free)
#     General Knowledge    18%  (fineweb-edu-dedup)
#     Math/Reasoning       10%  (finemath-4plus + finemath-3plus, CLEAN — no LaTeX-heavy instruct data)
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
        # ── Source Code (50%) — Top 10 Languages ────────────
        DatasetEntry(
            name="starcoderdata-python",
            repo_id="bigcode/starcoderdata",
            weight=16,
            format_fn="starcoder",
            data_dir="python",
        ),
        DatasetEntry(
            name="starcoderdata-javascript",
            repo_id="bigcode/starcoderdata",
            weight=7,
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
            weight=5,
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
            weight=2,
            format_fn="starcoder",
            data_dir="php",
        ),
        # ── General Knowledge (18%) ─────────────────────────
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=18,
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ),
        # ── Math/Reasoning (10%) ────────────────────────────
        DatasetEntry(
            name="finemath-4plus",
            repo_id="HuggingFaceTB/finemath",
            weight=6,
            format_fn="finemath",
            config_name="finemath-4plus",
        ),
        DatasetEntry(
            name="finemath-3plus",
            repo_id="HuggingFaceTB/finemath",
            weight=4,
            format_fn="finemath",
            config_name="finemath-3plus",
        ),
        # ── CS/Engineering (22%) ────────────────────────────
        DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=10,
            format_fn="stackexchange_programming_cs",
        ),
        DatasetEntry(
            name="opencodeinstruct",
            repo_id="nvidia/OpenCodeInstruct",
            weight=12,
            format_fn="opencodeinstruct",
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
    
