#!/usr/bin/env python3
"""
Nsight Systems Training Profiler — Standalone GPU Script
=========================================================

Profiles the training pass for both model_adv and model_improv using
Nsight Systems (nsys). Captures model speed, dataloader stalls, CUDA kernel
timing, NVTX ranges, and uploads everything to W&B.

Usage (on a GPU VM):
    # Profile both models (default):
    python profile_nsys_training.py

    # Profile only one model:
    python profile_nsys_training.py --models model_adv

    # Custom profiling window:
    python profile_nsys_training.py --warmup-steps 3 --active-steps 15

    # Custom W&B project:
    python profile_nsys_training.py --wandb-project my_project

Requirements:
    - NVIDIA GPU with CUDA
    - nsys (Nsight Systems CLI) installed
    - wandb, torch, triton, etc. (same as training requirements)
"""

import argparse
import json
import os
import shutil
import subprocess
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime
from glob import glob

# ── Ensure project root is importable ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # project_828/
sys.path.insert(0, str(PROJECT_ROOT))


def find_nsys_binary() -> str:
    """Locate the nsys binary on the system."""
    nsys = shutil.which("nsys")
    if nsys:
        return nsys

    candidates = [
        "/usr/local/cuda/bin/nsys",
        "/usr/bin/nsys",
        "/opt/nvidia/nsight-systems/*/bin/nsys",
    ]
    for candidate in candidates:
        matches = glob(candidate)
        if matches:
            return matches[0]

    # Search common Nsight Systems install paths
    nsight_dirs = [
        Path("/opt/nvidia/nsight-systems"),
        Path("/usr/local/cuda"),
    ]
    for d in nsight_dirs:
        if d.exists():
            for p in d.rglob("nsys"):
                if p.is_file() and os.access(str(p), os.X_OK):
                    return str(p)

    raise FileNotFoundError(
        "Could not locate 'nsys' binary. "
        "Install Nsight Systems: apt-get install -y nsight-systems "
        "or download from https://developer.nvidia.com/nsight-systems"
    )


def run_nsys_profile(
    model_name: str,
    output_dir: Path,
    warmup_steps: int,
    active_steps: int,
    nsys_bin: str,
) -> tuple:
    """
    Run nsys profile on the training script for a specific model.

    Returns (nsys_rep_path, sqlite_path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = output_dir / f"nsys_training_{model_name}_{timestamp}"

    # The training script is run as a module from the project root
    train_module = "src.scripts.training.train"

    cmd = [
        nsys_bin, "profile",
        "-f", "true",                           # Overwrite existing output
        "-o", str(output_prefix),                # Output file prefix
        "--trace=cuda,nvtx,osrt,cudnn,cublas",   # Trace sources
        "--capture-range=cudaProfilerApi",        # Only capture our profiling window
        "--capture-range-end=stop",
        "--cuda-memory-usage=true",              # Track CUDA memory allocations
        "--stats=true",                          # Generate stats summary
        "--export=sqlite",                       # Also export SQLite for analysis
        "python3", "-m", train_module,
        "--model", model_name,
        "--profile",
        "--profile-warmup-steps", str(warmup_steps),
        "--profile-active-steps", str(active_steps),
        "--no-compile",                          # Disable torch.compile for cleaner traces
    ]

    print(f"\n{'='*70}")
    print(f" Profiling: {model_name}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["PROFILE_NSYS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"[WARNING] nsys exited with code {result.returncode} for {model_name}")
        print("  This may be expected if the training script exits after profiling window.")

    # Find the generated files
    nsys_rep = Path(f"{output_prefix}.nsys-rep")
    sqlite_file = Path(f"{output_prefix}.sqlite")

    if not nsys_rep.exists():
        # Try finding any .nsys-rep file matching the prefix
        matches = list(output_dir.glob(f"nsys_training_{model_name}*.nsys-rep"))
        if matches:
            nsys_rep = sorted(matches)[-1]  # Latest
            sqlite_file = nsys_rep.with_suffix(".sqlite")

    return nsys_rep, sqlite_file


def extract_nsys_stats(sqlite_path: Path, model_name: str) -> dict:
    """
    Extract profiling statistics from the Nsight Systems SQLite export.

    Returns a dict of metrics suitable for W&B logging.
    """
    metrics = {"model": model_name}

    if not sqlite_path.exists():
        print(f"[WARNING] SQLite file not found: {sqlite_path}")
        return metrics

    try:
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        # Get table names to understand the schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # ── NVTX Ranges (our custom annotations) ──
        nvtx_table = None
        for t in tables:
            if "nvtx" in t.lower() and "event" in t.lower():
                nvtx_table = t
                break

        if nvtx_table:
            try:
                # Get column names to handle schema differences
                cursor.execute(f"PRAGMA table_info({nvtx_table})")
                columns = {row[1].lower(): row[1] for row in cursor.fetchall()}

                text_col = columns.get("text", columns.get("name", None))
                start_col = columns.get("start", columns.get("timestamp", None))
                end_col = columns.get("end", None)
                dur_col = columns.get("duration", None)

                if text_col and (dur_col or (start_col and end_col)):
                    dur_expr = dur_col if dur_col else f"({end_col} - {start_col})"
                    cursor.execute(f"""
                        SELECT {text_col},
                               COUNT(*) as count,
                               AVG({dur_expr}) / 1e6 as avg_ms,
                               MIN({dur_expr}) / 1e6 as min_ms,
                               MAX({dur_expr}) / 1e6 as max_ms,
                               SUM({dur_expr}) / 1e6 as total_ms
                        FROM {nvtx_table}
                        WHERE {text_col} IS NOT NULL
                        GROUP BY {text_col}
                        ORDER BY total_ms DESC
                    """)
                    for row in cursor.fetchall():
                        name, count, avg_ms, min_ms, max_ms, total_ms = row
                        safe_name = name.replace(" ", "_").replace("/", "_")
                        metrics[f"nvtx/{safe_name}/count"] = count
                        metrics[f"nvtx/{safe_name}/avg_ms"] = avg_ms or 0
                        metrics[f"nvtx/{safe_name}/min_ms"] = min_ms or 0
                        metrics[f"nvtx/{safe_name}/max_ms"] = max_ms or 0
                        metrics[f"nvtx/{safe_name}/total_ms"] = total_ms or 0
            except Exception as e:
                print(f"[WARNING] Failed to query NVTX ranges: {e}")

        # ── CUDA Kernel Statistics ──
        kernel_table = None
        for t in tables:
            if "cupti_activity_kind_kernel" in t.lower():
                kernel_table = t
                break
        if kernel_table is None:
            for t in tables:
                if "kernel" in t.lower() and "api" not in t.lower():
                    kernel_table = t
                    break

        if kernel_table:
            try:
                cursor.execute(f"PRAGMA table_info({kernel_table})")
                columns = {row[1].lower(): row[1] for row in cursor.fetchall()}

                name_col = columns.get("shortname", columns.get("demangledname",
                           columns.get("name", None)))
                dur_col = columns.get("duration", columns.get("end", None))

                if name_col and dur_col:
                    # Check if duration is direct or needs computation
                    if dur_col.lower() == "end" and "start" in columns:
                        dur_expr = f"({columns['end']} - {columns['start']})"
                    else:
                        dur_expr = dur_col

                    cursor.execute(f"""
                        SELECT {name_col},
                               COUNT(*) as invocations,
                               AVG({dur_expr}) / 1e3 as avg_us,
                               SUM({dur_expr}) / 1e6 as total_ms
                        FROM {kernel_table}
                        WHERE {name_col} IS NOT NULL
                        GROUP BY {name_col}
                        ORDER BY total_ms DESC
                        LIMIT 25
                    """)

                    total_kernel_time_ms = 0
                    top_kernels = []
                    for row in cursor.fetchall():
                        kname, invocations, avg_us, total_ms = row
                        total_kernel_time_ms += total_ms or 0
                        top_kernels.append({
                            "name": kname,
                            "invocations": invocations,
                            "avg_us": avg_us or 0,
                            "total_ms": total_ms or 0,
                        })

                    metrics["kernels/total_kernel_time_ms"] = total_kernel_time_ms
                    metrics["kernels/num_unique_kernels"] = len(top_kernels)
                    # Store top 5 kernel names for quick reference
                    for i, k in enumerate(top_kernels[:5]):
                        metrics[f"kernels/top{i+1}_name"] = k["name"][:80]
                        metrics[f"kernels/top{i+1}_total_ms"] = k["total_ms"]
                        metrics[f"kernels/top{i+1}_invocations"] = k["invocations"]
            except Exception as e:
                print(f"[WARNING] Failed to query kernel stats: {e}")

        # ── CUDA memcpy/memset ──
        for t in tables:
            if "memcpy" in t.lower():
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {t}")
                    count = cursor.fetchone()[0]
                    metrics["memory/memcpy_operations"] = count
                except:
                    pass
            if "memset" in t.lower():
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {t}")
                    count = cursor.fetchone()[0]
                    metrics["memory/memset_operations"] = count
                except:
                    pass

        conn.close()

    except Exception as e:
        print(f"[WARNING] Failed to parse SQLite stats: {e}")

    return metrics


def run_nsys_text_stats(output_dir: Path, model_name: str) -> dict:
    """
    Run nsys stats on the .nsys-rep file and capture text output.
    """
    text_reports = {}

    try:
        nsys_bin = find_nsys_binary()
    except FileNotFoundError:
        return text_reports

    nsys_rep_files = sorted(output_dir.glob(f"nsys_training_{model_name}*.nsys-rep"))
    if not nsys_rep_files:
        return text_reports

    nsys_rep = nsys_rep_files[-1]

    report_types = ["nvtx_sum", "cuda_gpu_kern_sum", "cuda_api_sum", "osrt_sum"]
    for report in report_types:
        try:
            result = subprocess.run(
                [nsys_bin, "stats", "--report", report, str(nsys_rep)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout:
                text_reports[f"nsys_{report}"] = result.stdout[:8000]
        except Exception as e:
            print(f"[WARNING] nsys stats --report {report} failed: {e}")

    return text_reports


def upload_to_wandb(
    model_name: str,
    nsys_rep_path: Path,
    sqlite_path: Path,
    stats: dict,
    text_reports: dict,
    wandb_entity: str,
    wandb_project: str,
):
    """Upload profiling results to W&B."""
    import wandb

    run_name = f"nsys_{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        group="nsys_training_profiling",
        tags=[model_name, "nsys", "training_profile"],
        config={
            "model": model_name,
            "profiler": "nsight_systems",
            "profile_type": "training_pass",
        },
    )

    # ── Log numeric metrics ──
    numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    if numeric_stats:
        run.log(numeric_stats)

    # ── Log text reports as HTML artifacts ──
    for key, text in text_reports.items():
        if isinstance(text, str) and text.strip():
            run.log({key: wandb.Html(f"<pre style='font-size:12px'>{text}</pre>")})

    # ── Upload .nsys-rep as a W&B Artifact ──
    if nsys_rep_path.exists():
        artifact = wandb.Artifact(
            name=f"nsys_rep_{model_name}",
            type="nsys-profile",
            description=(
                f"Nsight Systems trace for {model_name} training pass. "
                f"Download and open in Nsight Systems GUI on your local machine."
            ),
        )
        artifact.add_file(str(nsys_rep_path))
        run.log_artifact(artifact)
        print(f"  ✅ Uploaded {nsys_rep_path.name} as W&B Artifact")

    # ── Upload SQLite export as a W&B Artifact ──
    if sqlite_path.exists():
        artifact_sql = wandb.Artifact(
            name=f"nsys_sqlite_{model_name}",
            type="nsys-sqlite",
            description=f"Nsight Systems SQLite export for {model_name}.",
        )
        artifact_sql.add_file(str(sqlite_path))
        run.log_artifact(artifact_sql)
        print(f"  ✅ Uploaded {sqlite_path.name} as W&B Artifact")

    # ── NVTX breakdown as a W&B Table ──
    nvtx_data = []
    for key in sorted(stats.keys()):
        if key.startswith("nvtx/") and key.endswith("/avg_ms"):
            range_name = key.replace("nvtx/", "").replace("/avg_ms", "")
            nvtx_data.append([
                range_name,
                stats.get(f"nvtx/{range_name}/avg_ms", 0),
                stats.get(f"nvtx/{range_name}/min_ms", 0),
                stats.get(f"nvtx/{range_name}/max_ms", 0),
                stats.get(f"nvtx/{range_name}/total_ms", 0),
                stats.get(f"nvtx/{range_name}/count", 0),
            ])

    if nvtx_data:
        table = wandb.Table(
            columns=["range", "avg_ms", "min_ms", "max_ms", "total_ms", "count"],
            data=nvtx_data,
        )
        run.log({"nvtx_breakdown": table})

    # ── Top kernels table ──
    kernel_rows = []
    for i in range(1, 6):
        name = stats.get(f"kernels/top{i}_name")
        if name:
            kernel_rows.append([
                name,
                stats.get(f"kernels/top{i}_total_ms", 0),
                stats.get(f"kernels/top{i}_invocations", 0),
            ])
    if kernel_rows:
        kernel_table = wandb.Table(
            columns=["kernel_name", "total_ms", "invocations"],
            data=kernel_rows,
        )
        run.log({"top_cuda_kernels": kernel_table})

    run.finish()
    print(f"  ✅ W&B run complete: {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Nsight Systems Training Profiler — profiles model_adv and model_improv"
    )
    parser.add_argument(
        "--models", nargs="+", default=["model_adv", "model_improv"],
        choices=["model_adv", "model_improv"],
        help="Which models to profile (default: both)",
    )
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Optimizer steps before starting CUDA profiler")
    parser.add_argument("--active-steps", type=int, default=10,
                        help="Optimizer steps to profile")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (default: private/nsys_profiles/)")
    parser.add_argument("--wandb-entity", type=str, default="akshithmarepally-akai")
    parser.add_argument("--wandb-project", type=str, default="828_nsys_profiling")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip W&B upload (just generate profiles)")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "nsys_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find nsys
    nsys_bin = find_nsys_binary()
    print(f"[NSys Profiler] Found nsys at: {nsys_bin}")
    print(f"[NSys Profiler] Output directory: {output_dir}")
    print(f"[NSys Profiler] Models to profile: {args.models}")
    print(f"[NSys Profiler] Warmup steps: {args.warmup_steps}, Active steps: {args.active_steps}")

    all_results = {}

    for model_name in args.models:
        print(f"\n{'#'*70}")
        print(f"  PROFILING: {model_name}")
        print(f"{'#'*70}")

        # Run nsys profile
        nsys_rep, sqlite_file = run_nsys_profile(
            model_name=model_name,
            output_dir=output_dir,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            nsys_bin=nsys_bin,
        )

        print(f"\n[{model_name}] nsys-rep: {nsys_rep} (exists={nsys_rep.exists()})")
        print(f"[{model_name}] sqlite:   {sqlite_file} (exists={sqlite_file.exists()})")

        # Extract statistics from SQLite
        stats = extract_nsys_stats(sqlite_file, model_name)

        # Run nsys stats for text reports
        text_reports = run_nsys_text_stats(output_dir, model_name)

        all_results[model_name] = {
            "nsys_rep": nsys_rep,
            "sqlite": sqlite_file,
            "stats": stats,
            "text_reports": text_reports,
        }

        # Print NVTX summary
        print(f"\n[{model_name}] NVTX Range Summary:")
        print(f"  {'Range':<30s}  {'Avg(ms)':>10s}  {'Total(ms)':>12s}  {'Count':>8s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}  {'-'*8}")
        for key in sorted(stats.keys()):
            if key.startswith("nvtx/") and key.endswith("/avg_ms"):
                rn = key.replace("nvtx/", "").replace("/avg_ms", "")
                avg = stats.get(f"nvtx/{rn}/avg_ms", 0)
                total = stats.get(f"nvtx/{rn}/total_ms", 0)
                count = stats.get(f"nvtx/{rn}/count", 0)
                print(f"  {rn:<30s}  {avg:>10.2f}  {total:>12.2f}  {count:>8}")

        # Upload to W&B
        if not args.no_wandb:
            print(f"\n[{model_name}] Uploading to W&B ({args.wandb_project})...")
            upload_to_wandb(
                model_name=model_name,
                nsys_rep_path=nsys_rep,
                sqlite_path=sqlite_file,
                stats=stats,
                text_reports=text_reports,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
            )

    # ── Comparison summary ──
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Model':<20s}  {'Forward(ms)':>12s}  {'Backward(ms)':>13s}  {'DataLoad(ms)':>13s}  {'OptStep(ms)':>12s}")
        print(f"  {'-'*20}  {'-'*12}  {'-'*13}  {'-'*13}  {'-'*12}")
        for model_name, result in all_results.items():
            s = result["stats"]
            fwd = s.get("nvtx/forward/avg_ms", "N/A")
            bwd = s.get("nvtx/backward/avg_ms", "N/A")
            data = s.get("nvtx/data_to_device/avg_ms", "N/A")
            opt = s.get("nvtx/optimizer_step/avg_ms", "N/A")
            fwd_s = f"{fwd:.2f}" if isinstance(fwd, (int, float)) else str(fwd)
            bwd_s = f"{bwd:.2f}" if isinstance(bwd, (int, float)) else str(bwd)
            data_s = f"{data:.2f}" if isinstance(data, (int, float)) else str(data)
            opt_s = f"{opt:.2f}" if isinstance(opt, (int, float)) else str(opt)
            print(f"  {model_name:<20s}  {fwd_s:>12s}  {bwd_s:>13s}  {data_s:>13s}  {opt_s:>12s}")

    # Save combined JSON report
    report_path = output_dir / "profiling_report.json"
    json_report = {}
    for model_name, result in all_results.items():
        json_report[model_name] = {
            "nsys_rep": str(result["nsys_rep"]),
            "sqlite": str(result["sqlite"]),
            "stats": {k: v for k, v in result["stats"].items()
                      if isinstance(v, (int, float, str, bool))},
        }
    with open(report_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[NSys Profiler] Combined report saved: {report_path}")

    print("\n" + "="*70)
    print("  DONE — Download .nsys-rep files from W&B Artifacts")
    print("  and open in Nsight Systems GUI on your Mac.")
    print("="*70)


if __name__ == "__main__":
    main()
