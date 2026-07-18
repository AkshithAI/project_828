"""
Datamix Testing Pipeline — Main Entry Point
=============================================

Orchestrates the full datamix proxy experiment pipeline:

    1. Load or create a ProxyManifest for cross-run resumption
    2. For each pending mixture point in the grid:
       a. Train a proxy model (500M tokens, with within-run checkpointing)
       b. Evaluate on 3 domains (code, general, reasoning)
       c. Save results to manifest
    3. Fit Data Mixing Laws (quadratic regression)
    4. Find optimal mixture
    5. Generate report + plots
    6. Optionally validate code repetition budget

Usage:
    # Full pipeline (8 runs sequentially)
    python -m src.scripts.datamix_testing.run_datamix_tests

    # Resume after interruption (picks up where it left off)
    python -m src.scripts.datamix_testing.run_datamix_tests

    # Dry run (no GPU training, mock metrics)
    python -m src.scripts.datamix_testing.run_datamix_tests --dry-run

    # Skip training, run analysis on existing results
    python -m src.scripts.datamix_testing.run_datamix_tests --analysis-only

    # Custom output directory
    python -m src.scripts.datamix_testing.run_datamix_tests --output-dir results/run_01
"""

import os
import sys
import json
import math
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional

import torch

from .datamix_config import (
    ProxyExperimentConfig,
    ProxyManifest,
    ProxyRunResult,
    MixturePoint,
    MIXTURE_GRID,
    DynamicScheduleConfig,
    RepetitionConfig,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from .proxy_runner import train_proxy
from .mixing_law_fit import run_full_analysis
from .mixture_schedule import (
    RepetitionCurve,
    plot_schedule,
    plot_repetition_curve,
    DynamicScheduleConfig,
)


# ══════════════════════════════════════════════════════════════
#  Manifest Persistence
# ══════════════════════════════════════════════════════════════

def load_manifest(output_dir: str) -> ProxyManifest:
    """Load the proxy manifest from disk, or create a fresh one."""
    manifest_path = Path(output_dir) / "proxy_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        manifest = ProxyManifest.from_dict(data)
        n_done = len(manifest.completed_runs)
        print(f"[Pipeline] Loaded manifest: {n_done}/{manifest.total_runs} "
              f"runs complete")
        if manifest.current_run:
            print(f"[Pipeline] Resuming interrupted run: {manifest.current_run} "
                  f"(step {manifest.current_step})")
        return manifest
    else:
        print("[Pipeline] No manifest found — starting fresh.")
        return ProxyManifest(total_runs=len(MIXTURE_GRID))


def save_manifest(manifest: ProxyManifest, output_dir: str) -> None:
    """Persist the manifest to disk."""
    manifest_path = Path(output_dir) / "proxy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def reconstruct_manifest_from_checkpoints(
    output_dir: str,
    experiment_config: ProxyExperimentConfig,
    device: str = "cuda",
) -> ProxyManifest:
    """Load proxy manifest, auto-reconstructing completed runs from disk checkpoints or W&B if missing."""
    manifest = ProxyManifest(total_runs=len(experiment_config.mixture_grid))
    manifest_path = Path(output_dir) / "proxy_manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            manifest = ProxyManifest.from_dict(data)
            n_done = len(manifest.completed_runs)
            print(f"[Pipeline] Loaded manifest: {n_done}/{manifest.total_runs} runs complete")
        except Exception as e:
            print(f"[Pipeline] Warning: Could not parse manifest file: {e}. Reconstructing...")

    grid = experiment_config.mixture_grid

    # ── Scan W&B to restore completed runs and download checkpoints for active run ──
    try:
        import wandb
        print(f"[Pipeline] Scanning W&B project '{WANDB_PROJECT}' for completed runs...")
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
        
        # 1. First, reconstruct all completed runs from W&B
        for run in runs:
            if run.name.startswith("proxy_"):
                label = run.name[len("proxy_"):]
                if label not in manifest.completed_runs:
                    # Find matching mix point in grid
                    mix_point = None
                    for mp in grid:
                        if mp.label == label:
                            mix_point = mp
                            break
                    if mix_point is None:
                        continue

                    s = run.summary
                    # Check if the run finished and has final evaluation metrics
                    if "eval/combined_loss" in s:
                        print(f"[Pipeline] Found completed run '{label}' on W&B.")
                        from .proxy_runner import _build_result
                        metrics = {
                            "eval/code/loss": s.get("eval/code/loss", 0.0),
                            "eval/general/loss": s.get("eval/general/loss", 0.0),
                            "eval/reasoning/loss": s.get("eval/reasoning/loss", 0.0),
                            "eval/code/ppl": s.get("eval/code/ppl", 1.0),
                            "eval/general/ppl": s.get("eval/general/ppl", 1.0),
                            "eval/reasoning/ppl": s.get("eval/reasoning/ppl", 1.0),
                            "eval/combined_loss": s.get("eval/combined_loss", 0.0),
                        }
                        final_step = s.get("global_step", experiment_config.total_steps_per_run())
                        result = _build_result(mix_point, final_step, experiment_config, metrics)
                        manifest.completed_runs[mix_point.label] = result
                        save_manifest(manifest, output_dir)
                        print(f"[Pipeline] ✓ Restored completed run '{label}' from W&B cloud logs.")

        # 2. Next, identify the first incomplete run and download its checkpoint from W&B if missing locally
        first_incomplete = None
        for mp in grid:
            if mp.label not in manifest.completed_runs:
                first_incomplete = mp
                break
        
        if first_incomplete is not None:
            label = first_incomplete.label
            run_dir = Path(experiment_config.checkpoint_dir) / label
            ckpt_dir = run_dir / "checkpoints"
            
            # Check if checkpoint exists locally
            from ..helper_funcs import get_latest_checkpoint_step
            local_step = get_latest_checkpoint_step(ckpt_dir) if ckpt_dir.exists() else None
            
            # If no local checkpoint, search for it on W&B
            if local_step is None:
                print(f"[Pipeline] Checking W&B for active run '{label}' checkpoints...")
                for run in runs:
                    if run.name == f"proxy_{label}":
                        artifacts = run.logged_artifacts()
                        model_artifacts = [a for a in artifacts if a.type == "model"]
                        if model_artifacts:
                            # Sort by step number in the name
                            def get_step(art):
                                try:
                                    name_part = art.name.split(":")[0]
                                    return int(name_part.split("-")[-1])
                                except Exception:
                                    return -1
                            model_artifacts.sort(key=get_step, reverse=True)
                            latest_art = model_artifacts[0]
                            latest_art_step = get_step(latest_art)
                            
                            if latest_art_step > 0:
                                print(f"[Pipeline] Downloading latest checkpoint '{latest_art.name}' (step {latest_art_step}) from W&B...")
                                ckpt_dir.mkdir(parents=True, exist_ok=True)
                                latest_art.download(root=str(ckpt_dir))
                                print(f"[Pipeline] ✓ Checkpoint files downloaded successfully to {ckpt_dir}")
                        break

    except Exception as wb_err:
        print(f"[Pipeline] Warning: Could not scan/restore from W&B: {wb_err}")

    # ── Scan local checkpoints directory to see if any runs completed locally ──
    for mix_point in grid:
        if mix_point.label in manifest.completed_runs:
            continue

        run_dir = Path(experiment_config.checkpoint_dir) / mix_point.label
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        from ..helper_funcs import get_latest_checkpoint_step
        latest_step = get_latest_checkpoint_step(ckpt_dir)
        if latest_step is None:
            continue

        total_steps = experiment_config.total_steps_per_run()
        # If it reached the final training step, it's completed
        if latest_step >= total_steps:
            print(f"[Pipeline] Found completed checkpoint on disk for '{mix_point.label}' (step {latest_step}).")
            print(f"[Pipeline] Automatically running final evaluation to reconstruct manifest entry...")

            try:
                # Create model
                from .proxy_runner import create_proxy_model
                model = create_proxy_model(experiment_config.model_config, device)

                # Load checkpoints
                from ..helper_funcs import load_checkpoint
                load_checkpoint(ckpt_dir, model, device=device)

                # Evaluate
                from .proxy_eval import evaluate_proxy
                metrics = evaluate_proxy(
                    model, device,
                    experiment_config.eval_batch_size,
                    experiment_config.context_length,
                    experiment_config.eval_batches_per_domain,
                )

                from .proxy_runner import _build_result
                result = _build_result(mix_point, latest_step, experiment_config, metrics)
                manifest.completed_runs[mix_point.label] = result
                save_manifest(manifest, output_dir)
                print(f"[Pipeline] ✓ Reconstructed manifest entry for '{mix_point.label}' successfully.")
            except Exception as eval_err:
                print(f"[Pipeline] Failed to evaluate completed run '{mix_point.label}': {eval_err}")
            finally:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()

    return manifest


# ══════════════════════════════════════════════════════════════
#  Dry Run (Mock Metrics for Testing)
# ══════════════════════════════════════════════════════════════

def mock_proxy_result(mix: MixturePoint) -> ProxyRunResult:
    """Generate plausible mock metrics for a dry run.

    Simulates the expected trend: more code → lower code loss but
    higher general loss; more books → lower general loss.
    """
    import random
    random.seed(hash(mix.label))

    base_code_loss = 3.5 - 0.03 * mix.code_pct + 0.01 * mix.book_pct
    base_gen_loss = 2.8 + 0.02 * mix.code_pct - 0.02 * mix.book_pct
    base_reas_loss = 3.0 + 0.01 * mix.code_pct - 0.01 * mix.book_pct

    noise = lambda: random.uniform(-0.05, 0.05)
    code_loss = max(base_code_loss + noise(), 1.0)
    gen_loss = max(base_gen_loss + noise(), 1.0)
    reas_loss = max(base_reas_loss + noise(), 1.0)

    # Combined: weighted harmonic mean (60% code, 25% general, 15% reasoning)
    w_sum = 0.60 + 0.25 + 0.15
    inv_sum = 0.60 / code_loss + 0.25 / gen_loss + 0.15 / reas_loss
    combined = w_sum / inv_sum

    return ProxyRunResult(
        label=mix.label,
        code_pct=mix.code_pct,
        book_pct=mix.book_pct,
        web_pct=mix.web_pct,
        final_step=1920,
        total_tokens_seen=500_000_000,
        code_loss=code_loss,
        general_loss=gen_loss,
        reasoning_loss=reas_loss,
        code_ppl=math.exp(min(code_loss, 20)),
        general_ppl=math.exp(min(gen_loss, 20)),
        reasoning_ppl=math.exp(min(reas_loss, 20)),
        combined_score=combined,
    )


# ══════════════════════════════════════════════════════════════
#  Code Repetition Validation
# ══════════════════════════════════════════════════════════════

def validate_repetition_budget(
    optimal_code_pct: float,
    total_tokens: int = 120_000_000_000,
    rep_config: Optional[RepetitionConfig] = None,
) -> None:
    """Check if the optimal code fraction requires excessive repetition."""
    if rep_config is None:
        rep_config = RepetitionConfig()

    curve = RepetitionCurve(rep_config)
    r, within_budget = curve.compute_effective_repetitions(
        optimal_code_pct / 100.0, total_tokens,
    )

    print(f"\n{'─'*60}")
    print(f"  Code Repetition Analysis")
    print(f"{'─'*60}")
    print(f"  Optimal code fraction: {optimal_code_pct:.0f}%")
    print(f"  Total training tokens: {total_tokens/1e9:.0f}B")
    print(f"  Unique code tokens:    {rep_config.unique_code_tokens/1e9:.0f}B")
    print(f"  Required repetitions:  {r:.1f}×")
    print(f"  Max allowed:           {rep_config.max_repeat}×")

    if within_budget:
        gain_pct = curve.gain(r) / curve.gain(rep_config.max_repeat) * 100
        print(f"  ✓ Within budget ({gain_pct:.0f}% of max gain)")
    else:
        print(f"  ✗ EXCEEDS budget — consider reducing code fraction or "
              f"adding more unique code data")
        # Suggest the max code fraction that stays within budget
        max_code_tokens = rep_config.unique_code_tokens * rep_config.max_repeat
        max_code_pct = max_code_tokens / total_tokens * 100
        print(f"  → Max code fraction at {rep_config.max_repeat}× repetition: "
              f"{max_code_pct:.1f}%")

    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full datamix testing pipeline."""
    output_dir = os.path.abspath(args.output_dir)
    device = args.device
    dry_run = args.dry_run
    analysis_only = args.analysis_only

    os.makedirs(output_dir, exist_ok=True)

    experiment_config = ProxyExperimentConfig(
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
    )
    grid = experiment_config.mixture_grid

    if args.run:
        matching = [mix for mix in grid if mix.label == args.run]
        if not matching:
            print(f"[Pipeline] ERROR: No mixture found matching label '{args.run}'. ")
            print(f"Available labels: {[m.label for m in grid]}")
            sys.exit(1)
        grid = matching
        print(f"[Pipeline] Filtering grid to run ONLY '{args.run}'")

    # ── Load manifest (reconstructing completed runs if missing) ──
    manifest = reconstruct_manifest_from_checkpoints(output_dir, experiment_config, device)

    # ── Phase 1: Run proxy experiments ──
    if not analysis_only:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Proxy Grid Experiments")
        print(f"  {len(grid)} mixtures × {experiment_config.tokens_per_run/1e6:.0f}M "
              f"tokens each")
        print(f"  Device: {device} | Sequential execution")
        print(f"{'='*70}\n")

        for i, mix_point in enumerate(grid):
            if manifest.is_run_complete(mix_point.label):
                print(f"[{i+1}/{len(grid)}] {mix_point.label}: "
                      f"ALREADY COMPLETE — skipping")
                continue

            print(f"\n[{i+1}/{len(grid)}] Starting: {mix_point.label}")
            print(f"  Mix: {mix_point.to_weights_dict()}")

            manifest.current_run = mix_point.label
            save_manifest(manifest, output_dir)

            if dry_run:
                result = mock_proxy_result(mix_point)
                print(f"  [DRY RUN] Mock result: combined={result.combined_score:.4f}")
            else:
                try:
                    result = train_proxy(
                        mixture=mix_point,
                        experiment_config=experiment_config,
                        device=device,
                    )
                except KeyboardInterrupt:
                    print(f"\n[Pipeline] Interrupted during {mix_point.label}. "
                          f"Manifest saved — resume with same command.")
                    save_manifest(manifest, output_dir)
                    sys.exit(1)
                except Exception as e:
                    print(f"\n[Pipeline] ERROR in {mix_point.label}: {e}")
                    print("[Pipeline] Saving manifest and continuing to next run...")
                    save_manifest(manifest, output_dir)
                    continue

            manifest.completed_runs[mix_point.label] = result
            manifest.current_run = None
            manifest.current_step = 0
            save_manifest(manifest, output_dir)

            print(f"  ✓ {mix_point.label} complete: "
                  f"code_loss={result.code_loss:.4f}, "
                  f"gen_loss={result.general_loss:.4f}, "
                  f"combined={result.combined_score:.4f}")

    # ── Phase 2: Analysis ──
    completed = list(manifest.completed_runs.values())
    if len(completed) < 3:
        print(f"\n[Pipeline] Only {len(completed)} runs complete — need at least 3 "
              f"for analysis. Run more experiments first.")
        return

    print(f"\n{'='*70}")
    print(f"  PHASE 2: Data Mixing Law Analysis")
    print(f"{'='*70}")

    optimal = run_full_analysis(completed, output_dir)

    # ── Phase 3: Repetition validation ──
    validate_repetition_budget(optimal.code_pct)

    # ── Phase 4: Generate schedule plots ──
    schedule_config = DynamicScheduleConfig(
        code_p0=max(optimal.code_pct / 100 - 0.05, 0.05),
        code_p_target=optimal.code_pct / 100,
        web_p0=optimal.web_pct / 100 + 0.05,
        web_p_target=optimal.web_pct / 100,
        book_pct=optimal.book_pct / 100,
    )
    plot_schedule(schedule_config, output_path=str(Path(output_dir) / "schedule.png"))
    plot_repetition_curve(RepetitionConfig(),
                          output_path=str(Path(output_dir) / "repetition.png"))

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Runs completed:  {len(completed)}/{len(grid)}")
    print(f"  Optimal mixture: code={optimal.code_pct:.0f}% "
          f"book={optimal.book_pct:.0f}% web={optimal.web_pct:.0f}%")
    print(f"  Report:          {output_dir}/datamix_report.md")
    print(f"  Results JSON:    {output_dir}/proxy_results.json")
    print(f"  W&B project:     {WANDB_PROJECT}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Datamix Proxy Experiment Pipeline for Project 828",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (8 runs, single GPU)
  python -m src.scripts.datamix_testing.run_datamix_tests

  # Resume after interruption
  python -m src.scripts.datamix_testing.run_datamix_tests

  # Dry run with mock metrics (for testing the pipeline)
  python -m src.scripts.datamix_testing.run_datamix_tests --dry-run

  # Only run analysis on existing results
  python -m src.scripts.datamix_testing.run_datamix_tests --analysis-only
        """,
    )

    parser.add_argument(
        "--output-dir", type=str, default="datamix_results",
        help="Directory for checkpoints, results, and reports (default: datamix_results)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for training (default: cuda)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual training, use mock metrics (for testing pipeline logic)",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Skip training, only run analysis on existing results in manifest",
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run only a specific mixture configuration by its label (e.g. code25_book05)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    args = parse_args()
    run_pipeline(args)
