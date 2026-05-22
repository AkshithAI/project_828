"""
Eval Suite CLI — Standalone entry point for running the eval suite.

Usage:
    # Full suite on GPU
    venv/bin/python -m tests.test_eval_suite \
        --checkpoint checkpoints/model_06767.pt --device cuda

    # Quick run (fewer problems)
    venv/bin/python -m tests.test_eval_suite \
        --checkpoint checkpoints/model_06767.pt --device cpu --quick

    # Specific benchmarks only
    venv/bin/python -m tests.test_eval_suite \
        --checkpoint checkpoints/model_06767.pt \
        --bench mbpp cruxeval cs_qa

    # Cross-checkpoint comparison
    venv/bin/python -m tests.test_eval_suite \
        --checkpoint checkpoints/model_06767.pt checkpoints/model_101002.pt \
        --device cuda --bench mbpp cs_qa
"""

import os, sys, json, argparse, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.scripts.data.eval_suite import (
    load_model_for_eval, run_eval_suite, print_eval_summary, ALL_BENCHMARKS,
)


def main():
    parser = argparse.ArgumentParser(description="Project 828 Eval Suite")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="Path(s) to model checkpoint(s)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--bench", type=str, nargs="+", default=None,
                        choices=ALL_BENCHMARKS,
                        help=f"Benchmarks to run (default: all). Options: {ALL_BENCHMARKS}")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode — fewer problems per benchmark")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "private"),
                        help="Directory to save results")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_checkpoint_results = {}

    for ckpt_path in args.checkpoint:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(__file__), "..", ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue

        ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
        print(f"\n{'='*70}")
        print(f"  Evaluating: {ckpt_name}")
        print(f"  Device: {device}")
        print(f"  Quick: {args.quick}")
        print(f"{'='*70}")

        model = load_model_for_eval(ckpt_path, device)
        results = run_eval_suite(model, device, benchmarks=args.bench, quick=args.quick)
        print_eval_summary(results)

        all_checkpoint_results[ckpt_name] = {
            "checkpoint": ckpt_name,
            "checkpoint_path": ckpt_path,
            "device": device,
            "timestamp": datetime.now().isoformat(),
            "quick": args.quick,
            "results": results,
        }

        # Cleanup
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(args.output_dir, "eval_suite_results.json")
    compact = {}
    for name, data in all_checkpoint_results.items():
        compact[name] = {
            "checkpoint": data["checkpoint"],
            "timestamp": data["timestamp"],
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                        for k, v in data["results"].items()},
        }
    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Cross-checkpoint comparison
    if len(all_checkpoint_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-CHECKPOINT COMPARISON")
        print(f"{'='*70}")
        for bench in (args.bench or ALL_BENCHMARKS):
            print(f"\n  {bench}:")
            for name, data in all_checkpoint_results.items():
                r = data["results"].get(bench, {})
                if "error" in r:
                    print(f"    {name:25s}  ERROR")
                    continue
                if bench == "mbpp":
                    print(f"    {name:25s}  pass@1={r.get('pass_at_1',0):.1%}")
                elif bench == "cruxeval":
                    print(f"    {name:25s}  output={r.get('output_pass_at_1',0):.1%}  "
                          f"input={r.get('input_pass_at_1',0):.1%}")
                elif bench == "code_completion":
                    print(f"    {name:25s}  score={r.get('overall_score',0):.3f}")
                elif bench == "cs_qa":
                    print(f"    {name:25s}  accuracy={r.get('accuracy',0):.1%}")
                elif bench == "domain_ppl":
                    print(f"    {name:25s}  ppl={r.get('weighted_ppl',0):.1f}  "
                          f"loss={r.get('weighted_loss',0):.4f}")


if __name__ == "__main__":
    main()
