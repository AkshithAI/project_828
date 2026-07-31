"""
W&B Pull and Recalculate script for Project 828
==============================================

Pulls the evaluation results of all 8 proxy grid experiments directly
from W&B, fits the corrected non-collinear quadratic surface, searches the
valid grid space, and proves the optimal data mixture.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import wandb
except ImportError:
    print("ERROR: 'wandb' package is not installed. Run: pip install wandb")
    sys.exit(1)

# W&B Configuration
WANDB_ENTITY = "akshithmarepally-akai"
WANDB_PROJECT = "828_datamix_proxy"

# The 8 proxy mixtures in the grid
GRID_MIXTURES = [
    {"label": "code05_book05", "code": 5,  "book": 5},
    {"label": "code05_book15", "code": 5,  "book": 15},
    {"label": "code15_book05", "code": 15, "book": 5},
    {"label": "code15_book15", "code": 15, "book": 15},
    {"label": "code25_book05", "code": 25, "book": 5},
    {"label": "code25_book15", "code": 25, "book": 15},
    {"label": "code35_book05", "code": 35, "book": 5},
    {"label": "code35_book15", "code": 35, "book": 15},
]

@dataclass
class RunData:
    label: str
    code_pct: float
    book_pct: float
    web_pct: float
    code_loss: float
    general_loss: float
    reasoning_loss: float
    combined_loss: float

@dataclass
class QuadraticFit:
    metric_name: str
    alpha: float
    b1: float  # code
    b2: float  # book
    b3: float  # code^2
    b4: float  # book^2 (always 0.0, omitted to avoid collinearity)
    b5: float  # code * book
    r_squared: float

    def predict(self, c: float, b: float) -> float:
        return self.alpha + self.b1 * c + self.b2 * b + self.b3 * c * c + self.b5 * c * b


def _solve_ols(X: List[List[float]], y: List[float]) -> List[float]:
    """Solve ordinary least squares via Normal Equations: beta = (X^T X)^{-1} X^T y"""
    n = len(X)
    p = len(X[0])

    # X^T X (p x p)
    XtX = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            for k in range(n):
                XtX[i][j] += X[k][i] * X[k][j]

    # X^T y (p x 1)
    Xty = [0.0] * p
    for i in range(p):
        for k in range(n):
            Xty[i] += X[k][i] * y[k]

    # Solve via Gaussian elimination with pivoting
    aug = [XtX[i] + [Xty[i]] for i in range(p)]
    for i in range(p):
        pivot_row = i
        for r in range(i + 1, p):
            if abs(aug[r][i]) > abs(aug[pivot_row][i]):
                pivot_row = r
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            # Singular matrix, set coefficient to 0
            for r in range(i, p):
                aug[r][i] = 0.0
            continue

        for col in range(i, p + 1):
            aug[i][col] /= pivot

        for r in range(p):
            if r != i:
                factor = aug[r][i]
                for col in range(i, p + 1):
                    aug[r][col] -= factor * aug[i][col]

    return [row[p] for row in aug]


def fit_surface(data: List[RunData], metric_key: str) -> QuadraticFit:
    """Fits the non-collinear surface: y = a + b1*c + b2*b + b3*c^2 + b5*c*b"""
    X = []
    y = []
    for r in data:
        c, b = r.code_pct, r.book_pct
        X.append([1.0, c, b, c * c, c * b])
        y.append(getattr(r, metric_key))

    beta = _solve_ols(X, y)

    # Calculate R^2
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    residuals = []
    for i in range(len(y)):
        pred = beta[0] + beta[1] * X[i][1] + beta[2] * X[i][2] + beta[3] * X[i][3] + beta[4] * X[i][4]
        residuals.append(y[i] - pred)
    ss_res = sum(r ** 2 for r in residuals)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return QuadraticFit(
        metric_name=metric_key,
        alpha=beta[0],
        b1=beta[1],
        b2=beta[2],
        b3=beta[3],
        b4=0.0,
        b5=beta[4],
        r_squared=r_squared
    )


def main():
    print(f"Connecting to W&B: {WANDB_ENTITY}/{WANDB_PROJECT}...")
    api = wandb.Api()
    try:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    except Exception as e:
        print(f"Failed to fetch runs from W&B: {e}")
        return

    # Find the run metrics for our grid
    run_dict = {}
    for run in runs:
        if run.name.startswith("proxy_"):
            label = run.name[len("proxy_"):]
            run_dict[label] = run

    print("\nSuccessfully found runs on W&B. Retrieving final metrics:")
    print("-" * 110)
    print(f"{'Label':<15} | {'Code%':<5} | {'Book%':<5} | {'Web%':<5} | {'Code Loss':<10} | {'Gen Loss':<10} | {'Reas Loss':<10} | {'Combined':<10} | {'W&B State':<10}")
    print("-" * 110)

    pulled_data: List[RunData] = []
    
    # Sort mixtures by label to display them cleanly
    for mix in sorted(GRID_MIXTURES, key=lambda x: x["label"]):
        label = mix["label"]
        code_pct = mix["code"]
        book_pct = mix["book"]
        web_pct = 100 - code_pct - book_pct - 15  # fixed 15% other domains

        if label not in run_dict:
            print(f"{label:<15} | {code_pct:<5} | {book_pct:<5} | {web_pct:<5} | {'MISSING':<10} | {'MISSING':<10} | {'MISSING':<10} | {'MISSING':<10} | {'N/A':<10}")
            continue

        run = run_dict[label]
        s = run.summary
        
        # Pull required metrics
        c_loss = s.get("eval/code/loss", float("nan"))
        g_loss = s.get("eval/general/loss", float("nan"))
        r_loss = s.get("eval/reasoning/loss", float("nan"))
        comb_loss = s.get("eval/combined_loss", float("nan"))

        print(f"{label:<15} | {code_pct:<5} | {book_pct:<5} | {web_pct:<5} | {c_loss:<10.4f} | {g_loss:<10.4f} | {r_loss:<10.4f} | {comb_loss:<10.4f} | {run.state:<10}")

        # Check if values are valid numbers
        if any(map(lambda x: isinstance(x, float) and (x != x or x is None), [c_loss, g_loss, r_loss, comb_loss])):
            # Skip if invalid metrics
            continue

        pulled_data.append(RunData(
            label=label,
            code_pct=code_pct,
            book_pct=book_pct,
            web_pct=web_pct,
            code_loss=c_loss,
            general_loss=g_loss,
            reasoning_loss=r_loss,
            combined_loss=comb_loss
        ))

    if len(pulled_data) < 3:
        print(f"\nERROR: Only found {len(pulled_data)} valid data points on W&B. Need at least 3 to perform regression.")
        return

    print("-" * 110)
    print(f"Total valid runs loaded: {len(pulled_data)}/8")

    # Fit surfaces
    print("\n" + "=" * 50)
    print("  FITTING DATA MIXING LAW SURFACES")
    print("=" * 50)
    
    code_fit = fit_surface(pulled_data, "code_loss")
    gen_fit = fit_surface(pulled_data, "general_loss")
    reas_fit = fit_surface(pulled_data, "reasoning_loss")
    comb_fit = fit_surface(pulled_data, "combined_loss")

    for fit in [code_fit, gen_fit, reas_fit, comb_fit]:
        print(f"\n[Surface Fit] Metric: {fit.metric_name}")
        print(f"  R² = {fit.r_squared:.4f}")
        print(f"  Equation: Loss = {fit.alpha:.6f} + ({fit.b1:.6f} * Code) + ({fit.b2:.6f} * Book) + ({fit.b3:.6f} * Code^2) + ({fit.b5:.6f} * Code*Book)")

    # Grid search for optimal mixture
    print("\n" + "=" * 50)
    print("  OPTIMIZATION GRID SEARCH")
    print("=" * 50)
    print("Search bounds:")
    print("  Code range: [5%, 35%]")
    print("  Book range: [5%, 15%]  (constrained to tested domain range)")
    print("  Web range:  100 - Code - Book - 15 (min 10% web)")
    print("Weights:")
    print("  Code: 60%, General: 25%, Reasoning: 15%")

    best_score = float("inf")
    best_code = 0.0
    best_book = 0.0
    best_web = 0.0

    c = 5.0
    while c <= 35.0:
        b = 5.0
        while b <= 15.0:
            web = 100.0 - c - b - 15.0
            if web < 10.0:
                b += 0.1
                continue

            # Predict individual domain losses
            pred_code = code_fit.predict(c, b)
            pred_gen = gen_fit.predict(c, b)
            pred_reas = reas_fit.predict(c, b)

            # Compute combined score as weighted harmonic mean
            w_sum = 0.60 + 0.25 + 0.15
            inv_sum = 0.60 / pred_code + 0.25 / pred_gen + 0.15 / pred_reas
            score = w_sum / inv_sum

            if score < best_score:
                best_score = score
                best_code = c
                best_book = b
                best_web = web

            b += 0.1
        c += 0.1

    print("\n" + "=" * 50)
    print("  OPTIMIZATION RESULTS & PROOF")
    print("=" * 50)
    print(f"Optimal Data Mixture:")
    print(f"  → Code:      {best_code:.2f}%")
    print(f"  → Book:      {best_book:.2f}%")
    print(f"  → Web:       {best_web:.2f}%")
    print(f"  → Others:    15.00% (fixed: 5% science, 5% synthetic, 5% qa)")
    print(f"  → Predicted Combined Loss: {best_score:.4f}")

    # Diminishing returns & repetitions budget validation
    # Management config
    k = 0.15
    max_repeat = 15
    total_tokens = 120_000_000_000
    unique_code_tokens = 10_000_000_000

    opt_code_fraction = best_code / 100.0
    opt_code_tokens = total_tokens * opt_code_fraction
    required_repetitions = opt_code_tokens / unique_code_tokens

    print("\n" + "=" * 50)
    print("  CODE REPETITION & BUDGET VERIFICATION")
    print("=" * 50)
    print(f"Total pretraining budget:  {total_tokens/1e9:.1f}B tokens")
    print(f"Unique code tokens on disk: {unique_code_tokens/1e9:.1f}B tokens")
    print(f"Optimal code tokens:       {opt_code_tokens/1e9:.1f}B tokens ({best_code:.2f}%)")
    print(f"Required repetition rate:  {required_repetitions:.2f}x")
    print(f"Maximum allowed repetition: {max_repeat}x")

    if required_repetitions <= max_repeat:
        # Calculate utility gain
        gain = 1.0 - (2.71828 ** (-k * required_repetitions))
        max_gain = 1.0 - (2.71828 ** (-k * max_repeat))
        gain_pct = (gain / max_gain) * 100.0
        print(f"Status: ✓ WITHIN BUDGET ({gain_pct:.1f}% of maximum potential utility gain)")
    else:
        print("Status: ✗ EXCEEDS REPETITION BUDGET!")


if __name__ == "__main__":
    main()
