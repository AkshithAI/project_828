"""
Data Mixing Law Fitting & Optimal Mixture Prediction
======================================================

Implements the quadratic surface regression from the plan:

    Perf(code%, book%) = α + β₁·code% + β₂·book%
                       + β₃·code%² + β₄·book%²
                       + β₅·code%·book%

Fits separate surfaces for code metric and general metric,
then finds the mixture that optimizes the combined score
(weighted harmonic mean: 60% code, 25% general, 15% reasoning).

Also generates a markdown report with tables, fitted coefficients,
and the recommended mixture for full-scale pretraining.
"""

import math
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .datamix_config import ProxyRunResult, MixturePoint, DynamicScheduleConfig


# ══════════════════════════════════════════════════════════════
#  Quadratic Surface Fitting
# ══════════════════════════════════════════════════════════════

@dataclass
class QuadraticFit:
    """Coefficients of the fitted quadratic surface.

    Perf = alpha + b1*code + b2*book + b3*code² + b4*book² + b5*code*book
    """
    metric_name: str
    alpha: float
    b1: float    # code linear
    b2: float    # book linear
    b3: float    # code quadratic
    b4: float    # book quadratic
    b5: float    # interaction
    r_squared: float
    residuals: List[float]

    def predict(self, code_pct: float, book_pct: float) -> float:
        """Predict metric value at a given mixture point."""
        return (self.alpha
                + self.b1 * code_pct + self.b2 * book_pct
                + self.b3 * code_pct ** 2 + self.b4 * book_pct ** 2
                + self.b5 * code_pct * book_pct)


def fit_quadratic_surface(
    results: List[ProxyRunResult],
    metric_key: str = "code_loss",
) -> QuadraticFit:
    """Fit a quadratic surface to proxy experiment results.

    Uses ordinary least squares via the normal equation:
        β = (X^T X)^{-1} X^T y

    Implemented without numpy to avoid adding a dependency — the
    matrices are at most 8×6, so manual inversion is fine.

    Args:
        results: List of ProxyRunResult from completed proxy runs.
        metric_key: Which metric to fit ('code_loss', 'general_loss',
                    'reasoning_loss', 'combined_score').

    Returns:
        QuadraticFit with coefficients and R² value.
    """
    n = len(results)
    if n < 3:
        raise ValueError(
            f"Need at least 3 data points for fitting, got {n}. "
            f"Run more proxy experiments."
        )

    # Build design matrix X and target vector y
    # Features: [1, code, book, code², code*book]
    # NOTE: book² is omitted because book_pct only has 2 unique values (5%, 15%),
    # making book² collinear with 1 and book.
    X = []
    y = []
    for r in results:
        c, b = float(r.code_pct), float(r.book_pct)
        X.append([1.0, c, b, c * c, c * b])
        y.append(getattr(r, metric_key))

    num_features = min(5, n)

    if n < 5:
        if n < 4:
            # Linear only: [1, code, book]
            X = [[row[0], row[1], row[2]] for row in X]
            num_features = 3
        else:
            # Linear + code²: [1, code, book, code²]
            X = [[row[0], row[1], row[2], row[3]] for row in X]
            num_features = 4

    # Solve via normal equation: β = (X^T X)^{-1} X^T y
    beta = _solve_ols(X, y, num_features)

    # Compute R²
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    predictions = []
    residuals = []
    for i in range(n):
        pred = sum(X[i][j] * beta[j] for j in range(min(len(X[i]), len(beta))))
        predictions.append(pred)
        residuals.append(y[i] - pred)
    ss_res = sum(r ** 2 for r in residuals)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Map the solved coefficients back to the full 6-parameter QuadraticFit
    alpha_val = beta[0] if len(beta) > 0 else 0.0
    b1_val = beta[1] if len(beta) > 1 else 0.0
    b2_val = beta[2] if len(beta) > 2 else 0.0
    b3_val = beta[3] if len(beta) > 3 else 0.0
    b4_val = 0.0  # book² is omitted due to collinearity
    b5_val = beta[4] if len(beta) > 4 else 0.0

    return QuadraticFit(
        metric_name=metric_key,
        alpha=alpha_val,
        b1=b1_val,
        b2=b2_val,
        b3=b3_val,
        b4=b4_val,
        b5=b5_val,
        r_squared=r_squared,
        residuals=residuals,
    )


def _solve_ols(
    X: List[List[float]],
    y: List[float],
    num_features: int,
) -> List[float]:
    """Solve ordinary least squares via the normal equation.

    For small matrices (≤8×6), this is perfectly numerically stable
    and avoids numpy dependency.
    """
    n = len(X)
    p = num_features

    # X^T X  (p×p)
    XtX = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            for k in range(n):
                XtX[i][j] += X[k][i] * X[k][j]

    # X^T y  (p×1)
    Xty = [0.0] * p
    for i in range(p):
        for k in range(n):
            Xty[i] += X[k][i] * y[k]

    # Solve via Gaussian elimination with partial pivoting
    # Augmented matrix [XtX | Xty]
    aug = [XtX[i][:] + [Xty[i]] for i in range(p)]

    for col in range(p):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, p):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            # Near-singular — set this coefficient to 0
            aug[col] = [0.0] * (p + 1)
            continue

        for j in range(col, p + 1):
            aug[col][j] /= pivot

        for row in range(p):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, p + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][p] for i in range(p)]


# ══════════════════════════════════════════════════════════════
#  Optimal Mixture Search
# ══════════════════════════════════════════════════════════════

@dataclass
class OptimalMixture:
    """The recommended optimal mixture from surface fitting."""
    code_pct: float
    book_pct: float
    web_pct: float
    predicted_code_loss: float
    predicted_general_loss: float
    predicted_combined: float
    science_pct: float = 5.0
    synthetic_pct: float = 5.0
    qa_pct: float = 5.0


def find_optimal_mixture(
    code_fit: QuadraticFit,
    general_fit: QuadraticFit,
    reasoning_fit: Optional[QuadraticFit] = None,
    code_weight: float = 0.60,
    general_weight: float = 0.25,
    reasoning_weight: float = 0.15,
    code_range: Tuple[float, float] = (5, 35),
    book_range: Tuple[float, float] = (5, 15),
    grid_resolution: float = 1.0,
) -> OptimalMixture:
    """Find the mixture that minimizes the weighted combined loss.

    Searches a fine grid over [code_range] × [book_range] and evaluates
    the weighted harmonic mean of predicted losses.

    Args:
        code_fit: Fitted surface for code loss.
        general_fit: Fitted surface for general loss.
        reasoning_fit: Optional fitted surface for reasoning loss.
        code_weight: Weight for code in combined metric.
        general_weight: Weight for general in combined metric.
        reasoning_weight: Weight for reasoning in combined metric.
        code_range: (min, max) code percentage to search.
        book_range: (min, max) book percentage to search.
        grid_resolution: Step size for the search grid.

    Returns:
        OptimalMixture with the best found mixture.
    """
    fixed_pct = 15  # science(5) + synthetic(5) + qa(5)
    best_score = float("inf")
    best_code = code_range[0]
    best_book = book_range[0]

    c = code_range[0]
    while c <= code_range[1]:
        b = book_range[0]
        while b <= book_range[1]:
            web = 100 - c - b - fixed_pct
            if web < 10:  # Minimum web percentage
                b += grid_resolution
                continue

            code_loss = code_fit.predict(c, b)
            gen_loss = general_fit.predict(c, b)

            if code_loss <= 0 or gen_loss <= 0:
                b += grid_resolution
                continue

            # Weighted harmonic mean of losses (lower is better)
            w_sum = code_weight + general_weight
            inv_sum = code_weight / code_loss + general_weight / gen_loss

            if reasoning_fit is not None:
                reas_loss = reasoning_fit.predict(c, b)
                if reas_loss > 0:
                    w_sum += reasoning_weight
                    inv_sum += reasoning_weight / reas_loss

            combined = w_sum / inv_sum if inv_sum > 0 else float("inf")

            if combined < best_score:
                best_score = combined
                best_code = c
                best_book = b

            b += grid_resolution
        c += grid_resolution

    return OptimalMixture(
        code_pct=best_code,
        book_pct=best_book,
        web_pct=100 - best_code - best_book - fixed_pct,
        predicted_code_loss=code_fit.predict(best_code, best_book),
        predicted_general_loss=general_fit.predict(best_code, best_book),
        predicted_combined=best_score,
    )


# ══════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════

def generate_report(
    results: List[ProxyRunResult],
    code_fit: QuadraticFit,
    general_fit: QuadraticFit,
    reasoning_fit: Optional[QuadraticFit],
    optimal: OptimalMixture,
    output_path: str = "datamix_report.md",
) -> str:
    """Generate a markdown report with experiment results and recommendation.

    Args:
        results: All proxy run results.
        code_fit: Fitted code loss surface.
        general_fit: Fitted general loss surface.
        reasoning_fit: Optional reasoning loss surface.
        optimal: The recommended optimal mixture.
        output_path: Path to save the report.

    Returns:
        Path to the generated report.
    """
    lines = [
        "# Datamix Proxy Experiment Report",
        "",
        f"> Generated from {len(results)} proxy runs at 500M tokens each.",
        "",
        "---",
        "",
        "## 1. Proxy Run Results",
        "",
        "| Label | Code% | Book% | Web% | Code Loss | General Loss | "
        "Reasoning Loss | Combined |",
        "|-------|-------|-------|------|-----------|-------------|"
        "---------------|----------|",
    ]

    for r in sorted(results, key=lambda x: x.combined_score):
        lines.append(
            f"| {r.label} | {r.code_pct} | {r.book_pct} | {r.web_pct} "
            f"| {r.code_loss:.4f} | {r.general_loss:.4f} "
            f"| {r.reasoning_loss:.4f} | **{r.combined_score:.4f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Fitted Surfaces",
        "",
        "### Code Loss Surface",
        "",
        f"```",
        f"Perf(c, b) = {code_fit.alpha:.4f}",
        f"           + {code_fit.b1:.6f} × code%",
        f"           + {code_fit.b2:.6f} × book%",
        f"           + {code_fit.b3:.8f} × code%²",
        f"           + {code_fit.b4:.8f} × book%²",
        f"           + {code_fit.b5:.8f} × code% × book%",
        f"",
        f"R² = {code_fit.r_squared:.4f}",
        f"```",
        "",
        "### General Loss Surface",
        "",
        f"```",
        f"Perf(c, b) = {general_fit.alpha:.4f}",
        f"           + {general_fit.b1:.6f} × code%",
        f"           + {general_fit.b2:.6f} × book%",
        f"           + {general_fit.b3:.8f} × code%²",
        f"           + {general_fit.b4:.8f} × book%²",
        f"           + {general_fit.b5:.8f} × code% × book%",
        f"",
        f"R² = {general_fit.r_squared:.4f}",
        f"```",
    ])

    if reasoning_fit is not None:
        lines.extend([
            "",
            "### Reasoning Loss Surface",
            "",
            f"```",
            f"Perf(c, b) = {reasoning_fit.alpha:.4f}",
            f"           + {reasoning_fit.b1:.6f} × code%",
            f"           + {reasoning_fit.b2:.6f} × book%",
            f"           + {reasoning_fit.b3:.8f} × code%²",
            f"           + {reasoning_fit.b4:.8f} × book%²",
            f"           + {reasoning_fit.b5:.8f} × code% × book%",
            f"",
            f"R² = {reasoning_fit.r_squared:.4f}",
            f"```",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 3. Optimal Mixture Recommendation",
        "",
        "| Domain | Recommended % |",
        "|--------|--------------|",
        f"| **Code** | **{optimal.code_pct:.0f}%** |",
        f"| **Web** | **{optimal.web_pct:.0f}%** |",
        f"| **Books** | **{optimal.book_pct:.0f}%** |",
        f"| Science/Math | {optimal.science_pct:.0f}% |",
        f"| Synthetic | {optimal.synthetic_pct:.0f}% |",
        f"| Q&A | {optimal.qa_pct:.0f}% |",
        "",
        f"**Predicted code loss**: {optimal.predicted_code_loss:.4f}",
        f"**Predicted general loss**: {optimal.predicted_general_loss:.4f}",
        f"**Predicted combined score**: {optimal.predicted_combined:.4f}",
        "",
        "---",
        "",
        "## 4. Recommended Dynamic Schedule",
        "",
        "Based on the optimal mixture, the recommended cosine ramp for "
        "Phase 1 (85B tokens):",
        "",
        f"- Code: {max(optimal.code_pct - 5, 5):.0f}% → {optimal.code_pct:.0f}% "
        f"(cosine ramp)",
        f"- Web: {optimal.web_pct + 5:.0f}% → {optimal.web_pct:.0f}% "
        f"(inverse ramp)",
        f"- Books: {optimal.book_pct:.0f}% (fixed)",
        f"- Science: {optimal.science_pct:.0f}% (fixed)",
        f"- Synthetic: {optimal.synthetic_pct:.0f}% (fixed)",
        f"- Q&A: {optimal.qa_pct:.0f}% (fixed)",
        "",
        "---",
        "",
        "## 5. Residual Analysis",
        "",
        "### Code Loss Fit Residuals",
        "",
    ])

    for i, r in enumerate(results):
        lines.append(
            f"- {r.label}: residual = {code_fit.residuals[i]:+.4f}"
        )

    lines.extend([
        "",
        "### General Loss Fit Residuals",
        "",
    ])
    for i, r in enumerate(results):
        lines.append(
            f"- {r.label}: residual = {general_fit.residuals[i]:+.4f}"
        )

    report_text = "\n".join(lines) + "\n"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"[MixingLaw] Report saved to {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  Visualization (optional — requires matplotlib)
# ══════════════════════════════════════════════════════════════

def plot_mixture_surface(
    fit: QuadraticFit,
    results: List[ProxyRunResult],
    metric_key: str = "code_loss",
    output_path: str = "mixture_surface.png",
) -> str:
    """Plot the fitted performance surface as a contour plot.

    Args:
        fit: The fitted quadratic surface.
        results: Proxy run results (plotted as scatter points).
        metric_key: Which metric was fitted.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("[MixingLaw] matplotlib not available — skipping plot.")
        return ""

    # Generate grid
    code_range = list(range(0, 41))
    book_range = list(range(0, 26))
    Z = []
    for b in book_range:
        row = []
        for c in code_range:
            row.append(fit.predict(float(c), float(b)))
        Z.append(row)

    fig, ax = plt.subplots(figsize=(10, 7))
    cs = ax.contourf(
        code_range, book_range, Z,
        levels=20, cmap=cm.RdYlGn_r,
    )
    plt.colorbar(cs, ax=ax, label=f"{metric_key} (lower = better)")

    # Scatter actual data points
    for r in results:
        val = getattr(r, metric_key, None)
        if val is not None:
            ax.scatter(r.code_pct, r.book_pct, c="black", s=80,
                       zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(
                f"{val:.3f}",
                (r.code_pct, r.book_pct),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Code %", fontsize=12)
    ax.set_ylabel("Book %", fontsize=12)
    ax.set_title(f"Fitted Surface: {metric_key} (R²={fit.r_squared:.3f})",
                 fontsize=14)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[MixingLaw] Surface plot saved to {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  Convenience: Full Analysis Pipeline
# ══════════════════════════════════════════════════════════════

def run_full_analysis(
    results: List[ProxyRunResult],
    output_dir: str = "datamix_results",
) -> OptimalMixture:
    """Run the complete analysis pipeline: fit → optimize → report.

    Args:
        results: Completed proxy run results.
        output_dir: Directory for output files.

    Returns:
        The recommended OptimalMixture.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Data Mixing Law Analysis ({len(results)} runs)")
    print(f"{'='*70}\n")

    # Fit surfaces
    print("[MixingLaw] Fitting code loss surface...")
    code_fit = fit_quadratic_surface(results, "code_loss")
    print(f"  R² = {code_fit.r_squared:.4f}")

    print("[MixingLaw] Fitting general loss surface...")
    general_fit = fit_quadratic_surface(results, "general_loss")
    print(f"  R² = {general_fit.r_squared:.4f}")

    reasoning_fit = None
    if all(not math.isnan(r.reasoning_loss) for r in results):
        print("[MixingLaw] Fitting reasoning loss surface...")
        reasoning_fit = fit_quadratic_surface(results, "reasoning_loss")
        print(f"  R² = {reasoning_fit.r_squared:.4f}")

    # Find optimal
    print("\n[MixingLaw] Searching for optimal mixture...")
    optimal = find_optimal_mixture(
        code_fit, general_fit, reasoning_fit,
        code_weight=0.60, general_weight=0.25, reasoning_weight=0.15,
    )
    print(f"  → code={optimal.code_pct:.0f}%, book={optimal.book_pct:.0f}%, "
          f"web={optimal.web_pct:.0f}%")
    print(f"  → predicted combined loss: {optimal.predicted_combined:.4f}")

    # Generate report
    report_path = str(out / "datamix_report.md")
    generate_report(results, code_fit, general_fit, reasoning_fit,
                    optimal, report_path)

    # Generate plots
    plot_mixture_surface(code_fit, results, "code_loss",
                         str(out / "surface_code_loss.png"))
    plot_mixture_surface(general_fit, results, "general_loss",
                         str(out / "surface_general_loss.png"))

    # Save results as JSON for reproducibility
    results_json = [vars(r) for r in results]
    with open(out / "proxy_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[MixingLaw] Raw results saved to {out / 'proxy_results.json'}")

    # Save optimal as JSON
    with open(out / "optimal_mixture.json", "w") as f:
        json.dump(vars(optimal), f, indent=2)
    print(f"[MixingLaw] Optimal mixture saved to {out / 'optimal_mixture.json'}")

    return optimal
