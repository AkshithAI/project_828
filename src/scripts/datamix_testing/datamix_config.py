"""
Datamix Testing Configuration
===============================

Defines all configuration dataclasses for the proxy-scale mixture
experiment pipeline described in new_model_plan.md.

Proxy Model:
    ~102M total params, ~74M active — a reduced MoE matching the
    production architecture's routing dynamics (8 experts, top-2).

Mixture Grid:
    8 points (4 code × 2 book), 500M tokens each, single-GPU sequential.

Domain Buckets:
    web, code, books, science, synthetic, qa — mapped to existing HF datasets.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ..configs.model_config import DatasetEntry


# ══════════════════════════════════════════════════════════════
#  W&B Configuration
# ══════════════════════════════════════════════════════════════

WANDB_ENTITY = "akshithmarepally-akai"
WANDB_PROJECT = "828_datamix_proxy"


# ══════════════════════════════════════════════════════════════
#  Proxy Model Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class ProxyModelConfig:
    """Reduced-scale model config for proxy mixture experiments.

    Mirrors the production architecture (model_improv.py GPT_FLASH)
    but with smaller dimensions to enable fast iteration.

    ~102M total params, ~74M active per token.
      Embeddings:   50.3M  (embed 25.2M + unembed 25.2M)
      Per layer:     4.3M  × 12 layers = 52.0M
        Attention:   0.8M  (GQA 2:1 — 8 Q heads, 4 KV heads)
        MoE:         3.5M  (8 routed + 1 shared expert, top-2 routing)
      Active/tok:   74M    (attn + shared + 2 routed experts + embeds)
    """
    vocab_size: int = 49152          # StarCoder2-15B tokenizer
    num_attn_heads: int = 8
    num_key_value_heads: int = 4     # GQA 2:1
    hidden_dim: int = 512
    intermediate_size: int = 256
    ffn_dropout: float = 0.0
    head_dim: int = 64               # 512 // 8
    num_hidden_layers: int = 12
    num_experts: int = 8
    num_experts_per_tok: int = 2
    update_param: float = 2e-3
    route_scale: float = 1.0
    moe_aux_loss_weight: float = 0.01
    base: int = 10000
    initial_context_len: int = 2048
    max_context_len: int = 2048      # Short context for proxy speed
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0
    scaling_factor: float = 1.0
    dtype: torch.dtype = torch.bfloat16


# ══════════════════════════════════════════════════════════════
#  Mixture Point & Grid
# ══════════════════════════════════════════════════════════════

@dataclass
class MixturePoint:
    """A single point in the mixture experiment grid.

    The six domain fractions must sum to 100.
    ``web_pct`` is auto-calculated as the residual.
    """
    label: str                # e.g. "code15_book10"
    code_pct: int
    book_pct: int
    science_pct: int = 5     # Fixed
    synthetic_pct: int = 5   # Fixed
    qa_pct: int = 5          # Fixed

    @property
    def web_pct(self) -> int:
        return 100 - self.code_pct - self.book_pct - self.science_pct \
               - self.synthetic_pct - self.qa_pct

    def to_weights_dict(self) -> Dict[str, int]:
        """Return domain → weight mapping."""
        return {
            "web": self.web_pct,
            "code": self.code_pct,
            "book": self.book_pct,
            "science": self.science_pct,
            "synthetic": self.synthetic_pct,
            "qa": self.qa_pct,
        }

    def __post_init__(self):
        assert self.web_pct >= 0, (
            f"Negative web_pct={self.web_pct} — reduce other fractions "
            f"(code={self.code_pct}, book={self.book_pct})"
        )


# 8-point grid: 4 code fractions × 2 book fractions
MIXTURE_GRID: List[MixturePoint] = [
    MixturePoint("code05_book05", code_pct=5,  book_pct=5),
    MixturePoint("code05_book15", code_pct=5,  book_pct=15),
    MixturePoint("code15_book05", code_pct=15, book_pct=5),
    MixturePoint("code15_book15", code_pct=15, book_pct=15),
    MixturePoint("code25_book05", code_pct=25, book_pct=5),
    MixturePoint("code25_book15", code_pct=25, book_pct=15),
    MixturePoint("code35_book05", code_pct=35, book_pct=5),
    MixturePoint("code35_book15", code_pct=35, book_pct=15),
]


# ══════════════════════════════════════════════════════════════
#  Domain → HuggingFace Dataset Mapping
# ══════════════════════════════════════════════════════════════

def build_datasets_for_mixture(mix: MixturePoint) -> List[DatasetEntry]:
    """Convert a MixturePoint into a list of DatasetEntry objects
    compatible with the existing load_phase_datasets() infrastructure.

    Each domain maps to a single representative HuggingFace dataset
    using the existing format functions in dataloader.py.
    """
    datasets = []
    weights = mix.to_weights_dict()

    if weights["web"] > 0:
        datasets.append(DatasetEntry(
            name="fineweb-edu-dedup",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=weights["web"],
            format_fn="default",
            config_name="fineweb-edu-dedup",
        ))

    if weights["code"] > 0:
        datasets.append(DatasetEntry(
            name="starcoderdata-python",
            repo_id="bigcode/starcoderdata",
            weight=weights["code"],
            format_fn="starcoder",
            data_dir="python",
        ))

    if weights["book"] > 0:
        datasets.append(DatasetEntry(
            name="cosmopedia-v2",
            repo_id="HuggingFaceTB/smollm-corpus",
            weight=weights["book"],
            format_fn="cosmopedia",
            config_name="cosmopedia-v2",
        ))

    if weights["science"] > 0:
        datasets.append(DatasetEntry(
            name="finemath-4plus",
            repo_id="HuggingFaceTB/finemath",
            weight=weights["science"],
            format_fn="finemath",
            config_name="finemath-4plus",
        ))

    if weights["synthetic"] > 0:
        datasets.append(DatasetEntry(
            name="tiny-codes",
            repo_id="nampdn-ai/tiny-codes",
            weight=weights["synthetic"],
            format_fn="tiny_codes",
        ))

    if weights["qa"] > 0:
        datasets.append(DatasetEntry(
            name="stackexchange-programming-cs",
            repo_id="common-pile/stackexchange",
            weight=weights["qa"],
            format_fn="stackexchange_programming_cs",
        ))

    return datasets


# ══════════════════════════════════════════════════════════════
#  Proxy Experiment Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class ProxyExperimentConfig:
    """Top-level configuration for the proxy experiment grid."""

    # Model
    model_config: ProxyModelConfig = field(default_factory=ProxyModelConfig)

    # Training budget per run
    tokens_per_run: int = 500_000_000       # 500M tokens per mixture point
    context_length: int = 2048
    micro_batch_size: int = 32
    grad_accum_steps: int = 4
    grad_clip: float = 1.0

    # LR schedule (WSD)
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    scheduler_type: str = "wsd"
    wsd_stable_frac: float = 0.80

    # Checkpointing
    checkpoint_interval: int = 500          # Save every N optimizer steps
    checkpoint_dir: str = "checkpoints/datamix_proxy"

    # Evaluation
    eval_interval: int = 250                # Evaluate every N optimizer steps
    eval_batches_per_domain: int = 20       # Lightweight: 20 batches per domain
    eval_batch_size: int = 16

    # Grid
    mixture_grid: List[MixturePoint] = field(default_factory=lambda: MIXTURE_GRID)

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.context_length

    def total_steps_per_run(self) -> int:
        return self.tokens_per_run // self.tokens_per_step


# ══════════════════════════════════════════════════════════════
#  Dynamic Schedule Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class DynamicScheduleConfig:
    """Cosine ramp parameters for the production training schedule.

    During Phase 1 (first 85B tokens), code ramps from ``code_p0``
    to ``code_p_target`` using a cosine curve.  Web proportion adjusts
    as the residual.  All other domains stay fixed.
    """
    code_p0: float = 0.10           # 10% code at start
    code_p_target: float = 0.20     # 20% code at end of ramp
    web_p0: float = 0.65            # 65% web at start (residual)
    web_p_target: float = 0.55      # 55% web at end of ramp
    book_pct: float = 0.10          # Fixed
    science_pct: float = 0.05       # Fixed
    synthetic_pct: float = 0.05     # Fixed
    qa_pct: float = 0.05            # Fixed
    ramp_tokens: int = 85_000_000_000  # Ramp over Phase 1 (85B tokens)
    total_tokens: int = 120_000_000_000


# ══════════════════════════════════════════════════════════════
#  Code Repetition Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class RepetitionConfig:
    """Parameters for code corpus repetition management.

    Models diminishing returns via: Δ(r) = A × (1 − e^{−k·r})
    where r is the repetition factor.
    """
    k: float = 0.15                  # Decay constant — 95% gain at r≈20
    max_repeat: int = 15             # Max repetition factor
    unique_code_tokens: int = 10_000_000_000  # Estimated unique code tokens (10B)


# ══════════════════════════════════════════════════════════════
#  Proxy Run Manifest (for cross-run resumption)
# ══════════════════════════════════════════════════════════════

@dataclass
class ProxyRunResult:
    """Stores results from a single completed proxy run."""
    label: str
    code_pct: int
    book_pct: int
    web_pct: int
    final_step: int
    total_tokens_seen: int
    code_loss: float
    general_loss: float
    reasoning_loss: float
    code_ppl: float
    general_ppl: float
    reasoning_ppl: float
    combined_score: float            # Weighted harmonic mean


@dataclass
class ProxyManifest:
    """Tracks progress across all proxy runs for resumption."""
    completed_runs: Dict[str, ProxyRunResult] = field(default_factory=dict)
    current_run: Optional[str] = None
    current_step: int = 0
    total_runs: int = 8

    def is_run_complete(self, label: str) -> bool:
        return label in self.completed_runs

    def next_pending_run(self, grid: List[MixturePoint]) -> Optional[MixturePoint]:
        for point in grid:
            if not self.is_run_complete(point.label):
                return point
        return None

    def to_dict(self) -> dict:
        return {
            "completed_runs": {
                k: vars(v) for k, v in self.completed_runs.items()
            },
            "current_run": self.current_run,
            "current_step": self.current_step,
            "total_runs": self.total_runs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProxyManifest":
        completed = {}
        for k, v in data.get("completed_runs", {}).items():
            completed[k] = ProxyRunResult(**v)
        return cls(
            completed_runs=completed,
            current_run=data.get("current_run"),
            current_step=data.get("current_step", 0),
            total_runs=data.get("total_runs", 8),
        )
