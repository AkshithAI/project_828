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
        # ── Source Code (55%) — Coding Strength ─────────────
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
            weight=8,
            format_fn="starcoder",
            data_dir="javascript",
        ),
        DatasetEntry(
            name="starcoderdata-java",
            repo_id="bigcode/starcoderdata",
            weight=6,
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
            weight=6,
            format_fn="starcoder",
            data_dir="cpp",
        ),
        DatasetEntry(
            name="starcoderdata-c",
            repo_id="bigcode/starcoderdata",
            weight=4,
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
            weight=4,
            format_fn="starcoder",
            data_dir="go",
        ),
        DatasetEntry(
            name="starcoderdata-rust",
            repo_id="bigcode/starcoderdata",
            weight=3,
            format_fn="starcoder",
            data_dir="rust",
        ),
        DatasetEntry(
            name="starcoderdata-php",
            repo_id="bigcode/starcoderdata",
            weight=3,
            format_fn="starcoder",
            data_dir="php",
        ),
        # ── General Knowledge (20%) — Linguistic Anchor ─────
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=20,
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ),
        # ── High-Quality Knowledge (10%) — NEW ──────────────
        DatasetEntry(
            name="cosmopedia-v2",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=7,
            format_fn="cosmopedia",
            config_name="cosmopedia-v2",
        ),
        DatasetEntry(
            name="wikipedia-en",
            repo_id="wikimedia/wikipedia",
            weight=3,
            format_fn="wikipedia",
            config_name="20231101.en",
        ),
        # ── Math/Reasoning (8%) — Only Highest Quality ──────
        DatasetEntry(
            name="finemath-4plus",
            repo_id="HuggingFaceTB/finemath",
            weight=8,
            format_fn="finemath",
            config_name="finemath-4plus",
        ),
        # ── CS/Engineering (7%) — Cleaned StackExchange ─────
        DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=7,
            format_fn="stackexchange_programming_cs",
        ),
    ],
)


# ──────────────────────────────────────────────────────────────
# Phase 2:  30B tokens  —  Code Replay / Educational Code / CS Knowledge
# ──────────────────────────────────────────────────────────────
#   CONTEXT: Phase 1 (87B lifetime tokens) produced a strong Python
#   autocomplete engine but with critical gaps:
#     - Code understanding is broken (can write code but can't reason about it)
#     - CS knowledge is shallow/wrong (PUT vs PATCH, BST explanations)
#     - Non-Python languages degrade (C++ Stack::pop returns front())
#     - Textbook-mode contamination from Cosmopedia leaking into code context
#
#   Phase 2 strategy:
#     - Drop C, C#, PHP, Java (model has these from Phase 1, not core focus)
#     - Redistribute weight to CS Knowledge (StackExchange 10%→18%)
#     - Add educational code dataset (Tiny-Codes, 13 languages) to teach WHY
#     - Replace Cosmopedia with DCLM-Edu (real web content, not synthetic)
#     - Drop math datasets (per user decision)
#
#   effective_batch = 64 * 8 = 512 seqs
#   tokens_per_step = 512 * 2048 ≈ 1.05M
#   total_steps     = 28_600  (~30B tokens)
#   lifetime tokens = 87B (Phase 1) + 30B (Phase 2) ≈ 117B
#
#   Cosine schedule:
#     warmup:  0 → 999           (1,000 steps)
#     decay:   1,000 → 28,600    (27,600 steps, smooth cosine decay)
#
#   Dataset mix (weights sum to 100):
#     Code Replay          35%  Python/JS/TS/C++/Go/Rust (core languages only)
#     Educational Code     15%  Tiny-Codes (multi-lang educational snippets)
#     CS Knowledge         18%  Cleaned StackExchange programming/CS Q&A
#     General Knowledge    32%  DCLM-Edu + Wikipedia + FineWeb-Edu
# ──────────────────────────────────────────────────────────────
PHASE_2_CONFIG = PhaseConfig(
    phase_name="phase_2_continued",
    phase_num=2,
    peak_lr=6e-5,
    min_lr=6e-6,
    warmup_steps=1000,
    total_steps=28_600,
    scheduler_type="cosine",
    wsd_stable_frac=0.0,
    micro_batch_size=64,
    grad_accum_steps=8,
    grad_clip=1.0,
    val_interval=5000,
    val_steps=500,
    datasets=[
        # ── Code Replay (35%) — Prevent catastrophic forgetting ──
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
            weight=7,
            format_fn="starcoder",
            data_dir="javascript",
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
        # ── Educational Code (15%) — Teach code reasoning ────────
        DatasetEntry(
            name="tiny-codes",
            repo_id="nampdn-ai/tiny-codes",
            weight=15,
            format_fn="tiny_codes",
            max_epochs=5,
        ),
        # ── CS Knowledge (18%) — Fix shallow/wrong CS facts ──────
        DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=18,
            format_fn="stackexchange_programming_cs",
        ),
        # ── General Knowledge (32%) — Grounded factual content ───
        DatasetEntry(
            name="dclm-edu",
            repo_id="HuggingFaceTB/dclm-edu",
            weight=12,
            format_fn="dclm_edu",
        ),
        DatasetEntry(
            name="wikipedia-en",
            repo_id="wikimedia/wikipedia",
            weight=5,
            format_fn="wikipedia",
            config_name="20231101.en",
        ),
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=10,
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ),
    ],
)


config = ModelConfig()

if __name__ == '__main__':
    print(config)
    print(PHASE_1_CONFIG)
    
