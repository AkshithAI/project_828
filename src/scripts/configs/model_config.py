import torch
try:
    from ..tokenizer import tokenizer_v1 as tokenizer
except (ImportError, ValueError):
    from src.scripts.tokenizer import tokenizer_v1 as tokenizer
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
        head_dim : int = hidden_dim // num_attn_heads
        num_hidden_layers : int = 24 
        num_experts : int = 4
        num_experts_per_tok : int = 2 
        update_param : float = 2e-3
        route_scale : float = 1.0
        base : int = 10000
        initial_context_len : int = 2048
        max_context_len : int = 2048
        ntk_alpha : float = 1.0
        ntk_beta : float = 32.0
        scaling_factor : float = 1.0

        # Training
        dropout : float = 0.0
        learning_rate : float = 3e-4
        weight_decay : float = 0.1
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
    yarn_fmt_fn: Optional[str] = None         
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

    # --- Token-based schedule (optional, preferred for WSD) ---
    # When ``total_tokens`` is set, the WSD scheduler is driven by cumulative
    # non-padding tokens instead of optimizer steps, so context-length or
    # batch-size changes do not distort warmup/stable/decay.
    total_tokens: Optional[int] = None
    warmup_tokens: Optional[int] = None
    decay_start_tokens: Optional[int] = None

    # --- Continuation warmup (optional) ---
    # For extension/continuation runs, warmup rewarms from this LR up to peak_lr
    # instead of ramping from zero. Leave None for standard from-scratch warmup.
    start_lr: Optional[float] = None

    # --- Batch / accumulation ---
    micro_batch_size: int = 128
    grad_accum_steps: int = 8             
    grad_clip: float = 1.0

    # --- Validation ---
    val_interval: int = 2500              
    val_steps: int = 5000                 
    eval_suite_interval: int = 0          

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
    eval_suite_interval=0,
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


# ──────────────────────────────────────────────────────────
# Phase 2 — Continued pre-training (code-heavy mix)
#
#   Purpose: Strengthen code + add CS knowledge depth.
#   Resumes from Phase 1 checkpoint (model_101002.pt).
#
#   Hardware: H100 80GB
#   effective_batch = 25 * 22 = 550 seqs
#   tokens_per_step = 550 * 2048 ≈ 1.13M
#   total_steps     = 28_000  (~30B tokens)
#   lifetime tokens = 87B (Phase 1) + 30B (Phase 2) ≈ 117B
#
#   WSD schedule (MiniCPM-style):
#     warmup:  0 → 1,999           (2,000 steps)
#     stable:  2,000 → 23,279      (21,280 steps at peak LR)
#     decay:   23,280 → 28,000     (4,720 steps cosine → min_lr)
#
#   Dataset mix (weights sum to 100):
#     Source Code          56%  Python/JS/Java/TS/C++/Go/Rust
#     Educational Code      9%  Tiny-Codes (max_epochs=2)
#     CS Knowledge         17%  StackExchange (12) + DCLM-Edu (5)
#     General Knowledge    18%  FineWeb-Edu (15) + Wikipedia (3)
# ──────────────────────────────────────────────────────────
PHASE_2_CONFIG = PhaseConfig(
    phase_name="phase_2_continued",
    phase_num=2,
    peak_lr=4e-5,
    min_lr=4e-6,
    warmup_steps=2000,
    total_steps=28_000,
    scheduler_type="wsd",
    wsd_stable_frac=0.76,
    micro_batch_size=25,
    grad_accum_steps=22,
    grad_clip=1.0,
    val_interval=2000,
    val_steps=500,
    eval_suite_interval=5000,
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
        # ── Educational Code (9%) — Teach code reasoning ─────
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


PHASE_3_CONFIG = PhaseConfig(
    phase_name="yarn_extension",
    phase_num=3,
    peak_lr=5e-5,
    min_lr=3e-6,
    warmup_steps=2000,
    total_steps=28_000,
    scheduler_type="wsd",
    wsd_stable_frac=0.76,
    micro_batch_size=16,
    grad_accum_steps=4,
    grad_clip=1.0,
    val_interval=2000,
    val_steps=500,
    eval_suite_interval=5000,
    datasets=[
        # ── Source Code (45%) ─────────────────────────────────
        DatasetEntry(
            name="starcoderdata-python",
            repo_id="bigcode/starcoderdata",
            weight=14,
            yarn_fmt_fn="starcoder_python",
            data_dir="python",
        ),
        DatasetEntry(
            name="starcoderdata-javascript",
            repo_id="bigcode/starcoderdata",
            weight=5,
            yarn_fmt_fn="starcoder_javascript",
            data_dir="javascript",
        ),
        DatasetEntry(
            name="starcoderdata-java",
            repo_id="bigcode/starcoderdata",
            weight=5,
            yarn_fmt_fn="starcoder_java",
            data_dir="java",
        ),
        DatasetEntry(
            name="starcoderdata-typescript",
            repo_id="bigcode/starcoderdata",
            weight=5,
            yarn_fmt_fn="starcoder_typescript",
            data_dir="typescript",
        ),
        DatasetEntry(
            name="starcoderdata-cpp",
            repo_id="bigcode/starcoderdata",
            weight=5,
            yarn_fmt_fn="starcoder_cpp",
            data_dir="cpp",
        ),
        DatasetEntry(
            name="starcoderdata-c",
            repo_id="bigcode/starcoderdata",
            weight=4,
            yarn_fmt_fn="starcoder_c",
            data_dir="c",
        ),
        DatasetEntry(
            name="starcoderdata-go",
            repo_id="bigcode/starcoderdata",
            weight=7,
            yarn_fmt_fn="starcoder_go",
            data_dir="go",
        ),
        DatasetEntry(
            name="starcoderdata-rust",
            repo_id="bigcode/starcoderdata",
            weight=5,
            yarn_fmt_fn="starcoder_rust",
            data_dir="rust",
        ),
        # ── CS Knowledge (17%) ────────────────────────────────
        DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=10,
            yarn_fmt_fn="stackexchange_programming_cs",
        ),
        DatasetEntry(
            name="dclm-edu",
            repo_id="HuggingFaceTB/dclm-edu",
            weight=5,
            yarn_fmt_fn="dclm_edu",
        ),
        # ── General Knowledge (18%) ───────────────────────────
        DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=15,
            yarn_fmt_fn="fineweb_dedup",
            config_name="fineweb-edu-dedup",
        ),
        DatasetEntry(
            name="fineweb-finepdfs-edu",
            repo_id="HuggingFaceFW/finepdfs-edu",
            weight=15,
            yarn_fmt_fn="finepdfs",
        ),
        DatasetEntry(
            name="wikipedia-en",
            repo_id="wikimedia/wikipedia",
            weight=2,
            yarn_fmt_fn="wikipedia",
            config_name="20231101.en",
        ),
    ]
)

config = ModelConfig()

if __name__ == '__main__':
    print(config)
    print(PHASE_1_CONFIG)
    
