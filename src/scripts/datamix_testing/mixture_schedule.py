"""
Dynamic Mixture Schedule & Repetition Curves
==============================================

Implements the cosine-ramp mixture proportion scheduler and the
exponential-saturation code repetition model from new_model_plan.md.

Schedule formula (cosine ramp):
    p(t) = p_target − (p_target − p0) × (1 + cos(πt)) / 2
    where t ∈ [0, 1] is training progress fraction.

Repetition gain model:
    Δ(r) = A × (1 − e^{−k·r})
    where r is the repetition factor and k controls saturation speed.
"""

import math
from typing import Dict, List, Tuple
from .datamix_config import DynamicScheduleConfig, RepetitionConfig


# ══════════════════════════════════════════════════════════════
#  Cosine Ramp Schedule
# ══════════════════════════════════════════════════════════════

def cosine_ramp(t: float, p0: float, p_target: float) -> float:
    """Compute proportion at normalized time ``t`` using cosine ramp.

    Args:
        t: Training progress in [0, 1]. 0 = start, 1 = end of ramp.
        p0: Starting proportion.
        p_target: Target proportion at end of ramp.

    Returns:
        Proportion at time ``t``.
    """
    t = max(0.0, min(1.0, t))
    return p_target - (p_target - p0) * (1.0 + math.cos(math.pi * t)) / 2.0


def get_proportions_at_step(
    step: int,
    total_steps: int,
    config: DynamicScheduleConfig,
) -> Dict[str, float]:
    """Return domain proportions at a given training step.

    During the ramp phase (Phase 1), code increases and web decreases
    following cosine curves.  After the ramp, proportions are held constant.

    Args:
        step: Current optimizer step.
        total_steps: Total steps across all phases.
        config: Dynamic schedule configuration.

    Returns:
        Dict mapping domain name → proportion (sums to ~1.0).
    """
    ramp_steps = int(total_steps * (config.ramp_tokens / config.total_tokens))

    if step >= ramp_steps:
        # Post-ramp: hold at target proportions
        t = 1.0
    else:
        t = step / max(ramp_steps, 1)

    code = cosine_ramp(t, config.code_p0, config.code_p_target)
    web = cosine_ramp(t, config.web_p0, config.web_p_target)

    return {
        "code": code,
        "web": web,
        "book": config.book_pct,
        "science": config.science_pct,
        "synthetic": config.synthetic_pct,
        "qa": config.qa_pct,
    }


def generate_schedule_table(
    total_steps: int,
    config: DynamicScheduleConfig,
    num_checkpoints: int = 20,
) -> List[Dict[str, float]]:
    """Generate a table of proportions at evenly-spaced checkpoints.

    Useful for visualization and report generation.

    Args:
        total_steps: Total training steps.
        config: Dynamic schedule configuration.
        num_checkpoints: Number of evenly-spaced points to sample.

    Returns:
        List of dicts, each containing 'step', 'progress', and per-domain proportions.
    """
    table = []
    for i in range(num_checkpoints + 1):
        step = int(i * total_steps / num_checkpoints)
        props = get_proportions_at_step(step, total_steps, config)
        entry = {"step": step, "progress": step / max(total_steps, 1)}
        entry.update(props)
        table.append(entry)
    return table


# ══════════════════════════════════════════════════════════════
#  Code Repetition Curve
# ══════════════════════════════════════════════════════════════

class RepetitionCurve:
    """Models diminishing returns from code corpus repetition.

    The gain function Δ(r) = A × (1 − e^{−k·r}) saturates as
    repetition factor r increases.  At r ≈ 20/k ≈ 20 (for k=0.15),
    ~95% of maximum gain is reached.

    Args:
        config: Repetition configuration parameters.
    """

    def __init__(self, config: RepetitionConfig):
        self.k = config.k
        self.max_repeat = config.max_repeat
        self.unique_tokens = config.unique_code_tokens

    def gain(self, r: float, a: float = 1.0) -> float:
        """Compute relative performance gain at repetition factor ``r``.

        Args:
            r: Repetition factor (1 = no repetition, 15 = 15× reuse).
            a: Maximum gain amplitude (default 1.0 = normalized).

        Returns:
            Gain value in [0, A].
        """
        return a * (1.0 - math.exp(-self.k * r))

    def marginal_gain(self, r: float, a: float = 1.0) -> float:
        """Marginal gain from one additional repetition at factor ``r``.

        Args:
            r: Current repetition factor.
            a: Maximum gain amplitude.

        Returns:
            Derivative dΔ/dr at ``r``.
        """
        return a * self.k * math.exp(-self.k * r)

    def compute_effective_repetitions(
        self,
        code_fraction: float,
        total_tokens: int,
    ) -> Tuple[float, bool]:
        """Compute how many times the code corpus must be repeated.

        Args:
            code_fraction: Fraction of total tokens allocated to code (e.g. 0.15).
            total_tokens: Total training token budget.

        Returns:
            Tuple of (repetition_factor, within_budget) where within_budget
            is True if the factor is ≤ max_repeat.
        """
        code_tokens_needed = code_fraction * total_tokens
        if self.unique_tokens <= 0:
            return float('inf'), False

        r = code_tokens_needed / self.unique_tokens
        return r, r <= self.max_repeat

    def generate_curve_table(self, max_r: int = 20) -> List[Dict[str, float]]:
        """Generate a table of gain values for visualization.

        Args:
            max_r: Maximum repetition factor to tabulate.

        Returns:
            List of dicts with 'r', 'gain', 'marginal_gain' keys.
        """
        table = []
        for r in range(1, max_r + 1):
            table.append({
                "r": r,
                "gain": self.gain(r),
                "marginal_gain": self.marginal_gain(r),
                "pct_of_max": self.gain(r) / self.gain(max_r) * 100,
            })
        return table


# ══════════════════════════════════════════════════════════════
#  Visualization (optional — requires matplotlib)
# ══════════════════════════════════════════════════════════════

def plot_schedule(
    config: DynamicScheduleConfig,
    total_steps: int = 57200,
    output_path: str = "datamix_schedule.png",
) -> str:
    """Plot the dynamic mixture schedule over training.

    Args:
        config: Dynamic schedule configuration.
        total_steps: Total training steps.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[MixSchedule] matplotlib not available — skipping plot.")
        return ""

    table = generate_schedule_table(total_steps, config, num_checkpoints=100)
    steps = [e["step"] for e in table]
    domains = ["code", "web", "book", "science", "synthetic", "qa"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for domain, color in zip(domains, colors):
        values = [e[domain] * 100 for e in table]
        ax.plot(steps, values, label=domain.capitalize(), color=color, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax.set_title("Dynamic Data Mixture Schedule (Cosine Ramp)", fontsize=14)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 70)

    # Mark Phase 1 / Phase 2 boundary
    ramp_step = int(total_steps * config.ramp_tokens / config.total_tokens)
    ax.axvline(ramp_step, color="gray", linestyle="--", alpha=0.7, label="Phase 1→2")
    ax.text(ramp_step + 200, 65, "Phase 2 →", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[MixSchedule] Schedule plot saved to {output_path}")
    return output_path


def plot_repetition_curve(
    config: RepetitionConfig,
    output_path: str = "repetition_curve.png",
) -> str:
    """Plot the code repetition gain curve.

    Args:
        config: Repetition configuration.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[MixSchedule] matplotlib not available — skipping plot.")
        return ""

    curve = RepetitionCurve(config)
    table = curve.generate_curve_table(max_r=20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rs = [e["r"] for e in table]
    gains = [e["gain"] for e in table]
    marginals = [e["marginal_gain"] for e in table]

    ax1.plot(rs, gains, "o-", color="#e74c3c", linewidth=2, markersize=5)
    ax1.axhline(curve.gain(config.max_repeat), color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(config.max_repeat, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Repetition Factor (r)")
    ax1.set_ylabel("Cumulative Gain")
    ax1.set_title("Code Repetition — Cumulative Gain")
    ax1.grid(True, alpha=0.3)

    ax2.bar(rs, marginals, color="#3498db", alpha=0.7)
    ax2.set_xlabel("Repetition Factor (r)")
    ax2.set_ylabel("Marginal Gain")
    ax2.set_title("Code Repetition — Marginal Gain per Repeat")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[MixSchedule] Repetition curve saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick sanity check
    cfg = DynamicScheduleConfig()
    print("=== Schedule boundary conditions ===")
    p0 = get_proportions_at_step(0, 57200, cfg)
    p_mid = get_proportions_at_step(28600, 57200, cfg)
    p_end = get_proportions_at_step(57200, 57200, cfg)
    print(f"t=0.0: code={p0['code']:.3f}, web={p0['web']:.3f}")
    print(f"t=0.5: code={p_mid['code']:.3f}, web={p_mid['web']:.3f}")
    print(f"t=1.0: code={p_end['code']:.3f}, web={p_end['web']:.3f}")

    rep = RepetitionConfig()
    curve = RepetitionCurve(rep)
    print("\n=== Repetition curve ===")
    for entry in curve.generate_curve_table(max_r=20):
        print(f"  r={entry['r']:2d}  gain={entry['gain']:.4f}  "
              f"marginal={entry['marginal_gain']:.4f}  "
              f"pct_max={entry['pct_of_max']:.1f}%")
