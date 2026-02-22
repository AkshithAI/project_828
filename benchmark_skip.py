#!/usr/bin/env python3
"""
benchmark_skip.py — Load the dataloader checkpoint, time how long the
document-skipping takes on resume, then exit immediately.

Usage (from project-828/):
    PYTHONPATH=$(pwd) python project_828/benchmark_skip.py
    PYTHONPATH=$(pwd) python project_828/benchmark_skip.py --checkpoint project_828/checkpoints/dataloader_49999.pt
"""

import argparse
import time
import sys
import os
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Benchmark dataloader resume skip time")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to dataloader checkpoint (.pt)")
    args = parser.parse_args()

    # ── Find checkpoint ──────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-detect from project_828/checkpoints/
        ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        dl_files = sorted(ckpt_dir.glob("dataloader_*.pt"))
        if not dl_files:
            print("ERROR: No dataloader_*.pt found in checkpoints/")
            sys.exit(1)
        ckpt_path = dl_files[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Print summary ────────────────────────────────────────
    version = state.get("version", 1)
    phase = state.get("phase", 1)
    samples_yielded = state.get("samples_yielded", 0)
    batches_yielded = state.get("batches_yielded", 0)
    ds_states = state.get("dataset_states", {})
    total_docs = sum(s.get("documents_processed", 0) for s in ds_states.values())

    print("\n" + "=" * 65)
    print("  DATALOADER CHECKPOINT SUMMARY")
    print("=" * 65)
    print(f"  Version:         {version}")
    print(f"  Phase:           {phase}")
    print(f"  Samples yielded: {samples_yielded:,}")
    print(f"  Batches yielded: {batches_yielded:,}")
    print(f"  Total documents: {total_docs:,}")
    print()
    for name, ds_s in ds_states.items():
        docs = ds_s.get("documents_processed", 0)
        buf = len(ds_s.get("buffer_tokens", []))
        shard_info = ds_s.get("docs_per_shard", [])
        shard_str = f"  {len(shard_info)} shards tracked" if shard_info else "  no shard info"
        print(f"  {name:30s}  {docs:>12,} docs   {buf:>6,} buf tokens  {shard_str}")
    print("=" * 65)

    # ── Import project modules (needs PYTHONPATH set) ─────────
    print("\nImporting project modules...")
    try:
        from project_828.src.scripts.dataloader import create_phase_dataloaders
        from project_828.src.scripts.configs.model_config import PHASE_1_CONFIG, PHASE_2_CONFIG
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Make sure to run from project-828/ with: PYTHONPATH=$(pwd) python project_828/benchmark_skip.py")
        sys.exit(1)

    phase_config = PHASE_1_CONFIG if phase == 1 else PHASE_2_CONFIG

    # ── Time the resume (this is where skipping happens) ─────
    print(f"\n>>> Starting dataloader resume (Phase {phase})...")
    print(f">>> This will skip {total_docs:,} total documents across {len(ds_states)} datasets.")
    print(f">>> Timing...\n")

    t_start = time.perf_counter()

    train_data, val_data = create_phase_dataloaders(
        phase_config=phase_config,
        train_state=state,
        val_repo_id="HuggingFaceFW/fineweb-edu",
        batch_size_val=16,
        context_length=2048,
    )

    t_dataloader_created = time.perf_counter()
    print(f"\n[Timer] Dataloader created in {t_dataloader_created - t_start:.2f}s")

    # Pull the first batch to force actual iteration / skipping
    print("[Timer] Pulling first batch (forces skip to complete)...")
    t_first_batch_start = time.perf_counter()

    first_batch = None
    for batch in train_data:
        first_batch = batch
        break

    t_first_batch = time.perf_counter()

    t_total = t_first_batch - t_start

    # ── Results ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Dataloader creation : {t_dataloader_created - t_start:>10.2f}s  (stream setup + file resolution)")
    print(f"  First batch pull    : {t_first_batch - t_first_batch_start:>10.2f}s  (actual document skipping)")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL RESUME TIME   : {t_total:>10.2f}s")
    print(f"  Documents skipped   : {total_docs:>10,}")
    if t_total > 0:
        print(f"  Skip rate           : {total_docs / t_total:>10,.0f} docs/s")
    if first_batch is not None:
        print(f"  First batch shape   : {first_batch.shape}")
    print("=" * 65)

    print("\nDone. Exiting.")


if __name__ == "__main__":
    main()
