"""
Comprehensive Evaluation Suite for Project 828
===============================================
Lab-grade benchmarks aligned with the Phase 2 datamix.

Benchmarks:
  1. MBPP          — Execution-based Python code generation (→ Code Replay 35%)
  2. CRUXEval      — Code reasoning: output + input prediction (→ Educational Code 15%)
  3. Code Completion — Multilingual structural validation (→ Code Replay 35%)
  4. CS Knowledge QA — MCQ log-likelihood scoring (→ CS Knowledge 18%)
  5. Domain PPL     — Per-domain perplexity on held-out data (→ All categories)

Usage (standalone):
    python -m tests.test_eval_suite --checkpoint checkpoints/model_06767.pt --device cuda

Usage (from training loop):
    from ..data.eval_suite import run_training_eval
    run_training_eval(model, device, wandb_run, train_step)
"""

import os, sys, math, time, re, json, subprocess, traceback
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ..inference import generate
from ..tokenizer import tokenizer
from ..configs.model_config import config


# ═══════════════════════════════════════════════════════════════════
#  Shared Utilities
# ═══════════════════════════════════════════════════════════════════

def load_model_for_eval(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint for standalone evaluation."""
    from ...models.model_flash_attn import GPT_FLASH
    print(f"[EvalSuite] Loading model from {checkpoint_path}...")
    model = GPT_FLASH(config, device, inference=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    del state
    model.eval()
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    print(f"[EvalSuite] Model loaded on {device}")
    return model


def _execute_code(code: str, timeout: int = 3) -> Tuple[bool, str]:
    """Execute code in a subprocess sandbox. Returns (passed, error_msg)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        err = result.stderr.strip().split('\n')
        return False, err[-1] if err else "Unknown error"
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@torch.inference_mode()
def _compute_log_likelihood(
    model, prompt_text: str, continuation_text: str, device: str,
) -> float:
    """Compute average log-likelihood of continuation given prompt."""
    from ..inference import _enable_kv_cache, _disable_kv_cache

    prompt_ids = tokenizer.encode(prompt_text)
    cont_ids = tokenizer.encode(continuation_text)
    full_ids = prompt_ids + cont_ids

    max_len = config.max_context_len
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)

    # Use a clean forward pass (no KV cache needed for scoring)
    needs_toggle = getattr(model, 'inference', False)
    if needs_toggle and hasattr(model, 'reset_cache'):
        model.reset_cache()

    use_ac = device.startswith("cuda") if isinstance(device, str) else False
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_ac):
        logits = model(input_ids, start_pos=0)

    cont_start = max(prompt_len - 1, 0)
    cont_logits = logits[0, cont_start:, :]
    target_ids = torch.tensor(full_ids[prompt_len:], dtype=torch.long, device=device)

    min_len = min(cont_logits.shape[0], target_ids.shape[0])
    if min_len == 0:
        return float('-inf')

    log_probs = F.log_softmax(cont_logits[:min_len].float(), dim=-1)
    token_lp = log_probs.gather(1, target_ids[:min_len].unsqueeze(1)).squeeze(1)
    return token_lp.mean().item()


def _generate_completion(model, prompt: str, device: str, max_tokens: int = 128) -> str:
    """Generate a completion using the model. Returns only the generated part."""
    if hasattr(model, 'reset_cache'):
        model.reset_cache()
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    try:
        output = generate(
            model, prompt, device,
            max_tokens=max_tokens, temp=0.1, k=40, top_p=0.95,
            repetition_penalty=1.0, report_perf=False, show_progress=False,
        )
        return output[len(prompt):]
    except Exception as e:
        return f"[ERROR: {e}]"


# ═══════════════════════════════════════════════════════════════════
#  1. MBPP Benchmark  →  Code Replay (35%)
# ═══════════════════════════════════════════════════════════════════

def run_mbpp(
    model, device: str = "cuda",
    n_problems: int = 100, max_tokens: int = 256, timeout: int = 3,
) -> Dict[str, Any]:
    """
    MBPP: Mostly Basic Python Problems — execution-based pass@1.
    Generates code from task description, runs against provided test assertions.
    """
    print(f"\n{'='*60}\n  MBPP Benchmark ({n_problems} problems)\n{'='*60}")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test",
                      trust_remote_code=True)
    problems = list(ds)[:n_problems]
    print(f"  Loaded {len(problems)} problems")

    model.eval()
    passed, results = 0, []

    for i, p in enumerate(problems):
        prompt = (
            f"# Task: {p['prompt']}\n"
            f"# Write a Python function to solve this.\n\n"
        )
        generated = _generate_completion(model, prompt, device, max_tokens)

        # Build test: generated code + assertions
        test_code = generated.strip()
        for t in p['test_list']:
            test_code += f"\n{t}"

        ok, err = _execute_code(test_code, timeout)
        if ok:
            passed += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] pass@1: {passed/(i+1):.1%}")

        results.append({"task_id": p['task_id'], "passed": ok, "error": err})

    score = passed / len(problems) if problems else 0.0
    print(f"\n  MBPP pass@1: {score:.1%} ({passed}/{len(problems)})")
    return {
        "benchmark": "MBPP", "pass_at_1": score,
        "passed": passed, "total": len(problems), "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  2. CRUXEval Benchmark  →  Educational Code (15%)
# ═══════════════════════════════════════════════════════════════════

def run_cruxeval(
    model, device: str = "cuda",
    n_problems: int = 100, max_tokens: int = 64, timeout: int = 3,
) -> Dict[str, Any]:
    """
    CRUXEval: Code Reasoning, Understanding, and eXecution.
    Two tasks:
      - Output prediction: given f + input → predict output
      - Input prediction:  given f + output → predict valid input
    """
    print(f"\n{'='*60}\n  CRUXEval Benchmark ({n_problems} problems)\n{'='*60}")
    from datasets import load_dataset
    ds = load_dataset("cruxeval-org/cruxeval", split="test", trust_remote_code=True)
    problems = list(ds)[:n_problems]
    print(f"  Loaded {len(problems)} problems")

    model.eval()
    output_passed, input_passed = 0, 0
    results = []

    for i, p in enumerate(problems):
        code = p['code']
        inp = p['input']
        expected_out = p['output']

        # ── Output Prediction ──
        o_prompt = f"{code}\n\n# What is the output of: {inp}\n# Output: "
        o_gen = _generate_completion(model, o_prompt, device, max_tokens=32)
        o_gen_clean = o_gen.strip().split('\n')[0].strip()

        # Verify by execution
        o_test = f"{code}\nassert {inp} == {o_gen_clean}"
        o_ok, o_err = _execute_code(o_test, timeout)
        if o_ok:
            output_passed += 1

        # ── Input Prediction ──
        i_prompt = f"{code}\n\n# Find an input x such that the output equals {expected_out}\n# Input: "
        i_gen = _generate_completion(model, i_prompt, device, max_tokens=32)
        i_gen_clean = i_gen.strip().split('\n')[0].strip()

        # Verify by execution
        i_test = f"{code}\nassert {i_gen_clean} == {expected_out}"
        i_ok, i_err = _execute_code(i_test, timeout)
        if i_ok:
            input_passed += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] output: {output_passed/(i+1):.1%}  "
                  f"input: {input_passed/(i+1):.1%}")

        results.append({
            "idx": i, "output_ok": o_ok, "input_ok": i_ok,
            "output_err": o_err, "input_err": i_err,
        })

    n = len(problems)
    o_score = output_passed / n if n else 0.0
    i_score = input_passed / n if n else 0.0
    print(f"\n  CRUXEval-O pass@1: {o_score:.1%} ({output_passed}/{n})")
    print(f"  CRUXEval-I pass@1: {i_score:.1%} ({input_passed}/{n})")
    return {
        "benchmark": "CRUXEval",
        "output_pass_at_1": o_score, "input_pass_at_1": i_score,
        "output_passed": output_passed, "input_passed": input_passed,
        "total": n, "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  3. Multilingual Code Completion  →  Code Replay (35%)
# ═══════════════════════════════════════════════════════════════════

def _score_completion(text: str, lang: str, keywords: List[str]) -> Dict[str, float]:
    """Score a code completion on structural quality metrics (0-1 each)."""
    scores = {}

    # Bracket balance
    opens = text.count('{') + text.count('(') + text.count('[')
    closes = text.count('}') + text.count(')') + text.count(']')
    scores["bracket_balance"] = 1.0 - min(abs(opens - closes) / max(opens + closes, 1), 1.0)

    # Keyword presence
    kw_found = sum(1 for kw in keywords if kw in text) if keywords else 0
    scores["keyword_hit"] = kw_found / max(len(keywords), 1)

    # Sufficient length
    scores["sufficient_length"] = 1.0 if len(text.strip()) > 20 else 0.0

    # No repetition (detect repeated 4-grams)
    words = text.split()
    if len(words) > 15:
        ngrams = [" ".join(words[j:j+4]) for j in range(len(words) - 3)]
        from collections import Counter
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 2)
        scores["no_repetition"] = 1.0 if repeated / max(len(counts), 1) < 0.15 else 0.0
    else:
        scores["no_repetition"] = 1.0

    # Language-specific structure
    if lang == "python":
        scores["has_structure"] = float("return " in text or "def " in text or "class " in text)
    elif lang in ("javascript", "typescript"):
        scores["has_structure"] = float("return" in text or "=>" in text or "function" in text or "}" in text)
    elif lang == "go":
        scores["has_structure"] = float("return " in text or "func " in text or "}" in text)
    elif lang == "rust":
        scores["has_structure"] = float("fn " in text or "let " in text or "}" in text)
    elif lang == "cpp":
        scores["has_structure"] = float("return " in text or "}" in text or "::" in text)
    else:
        scores["has_structure"] = 1.0

    return scores


def run_code_completion(
    model, device: str = "cuda", max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Multilingual code completion with structural scoring.
    Uses curated prompts from eval_suite_prompts across 6 languages.
    """
    # Import prompts — handle both relative and absolute import contexts
    try:
        from tests.eval_suite_prompts import CODE_PROMPTS
    except ImportError:
        import importlib, pathlib
        spec = importlib.util.spec_from_file_location(
            "eval_suite_prompts",
            pathlib.Path(__file__).parent.parent.parent.parent / "tests" / "eval_suite_prompts.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        CODE_PROMPTS = mod.CODE_PROMPTS

    print(f"\n{'='*60}\n  Code Completion ({len(CODE_PROMPTS)} prompts, 6 languages)\n{'='*60}")
    model.eval()

    lang_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "scores": defaultdict(float)})
    results = []

    for i, p in enumerate(CODE_PROMPTS):
        lang = p["lang"]
        generated = _generate_completion(model, p["prompt"], device, max_tokens)
        scores = _score_completion(generated, lang, p.get("kw", []))
        avg = sum(scores.values()) / max(len(scores), 1)

        lang_stats[lang]["count"] += 1
        lang_stats[lang]["total_score"] += avg
        for k, v in scores.items():
            lang_stats[lang]["scores"][k] += v

        results.append({
            "lang": lang, "desc": p["desc"],
            "avg_score": round(avg, 3), "scores": scores,
            "preview": generated[:200],
        })

    # Print summary
    print(f"\n  {'Language':<15} {'Prompts':>7} {'Avg Score':>10} {'Bracket':>8} "
          f"{'Keywords':>8} {'Structure':>9} {'No Repeat':>9}")
    print(f"  {'─'*68}")
    overall_score, overall_count = 0.0, 0
    for lang in ["python", "javascript", "typescript", "cpp", "go", "rust"]:
        s = lang_stats.get(lang)
        if not s or s["count"] == 0:
            continue
        n = s["count"]
        avg = s["total_score"] / n
        bk = s["scores"]["bracket_balance"] / n
        kw = s["scores"]["keyword_hit"] / n
        st = s["scores"]["has_structure"] / n
        nr = s["scores"]["no_repetition"] / n
        print(f"  {lang:<15} {n:>7} {avg:>10.3f} {bk:>8.3f} {kw:>8.3f} {st:>9.3f} {nr:>9.3f}")
        overall_score += s["total_score"]
        overall_count += n

    overall = overall_score / max(overall_count, 1)
    print(f"  {'─'*68}")
    print(f"  {'OVERALL':<15} {overall_count:>7} {overall:>10.3f}")

    return {
        "benchmark": "CodeCompletion", "overall_score": round(overall, 4),
        "per_language": {l: {"avg": round(s["total_score"]/max(s["count"],1), 4), "n": s["count"]}
                        for l, s in lang_stats.items()},
        "total": overall_count, "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  4. CS Knowledge QA  →  CS Knowledge (18%)
# ═══════════════════════════════════════════════════════════════════

def run_cs_qa(model, device: str = "cuda") -> Dict[str, Any]:
    """
    CS Knowledge evaluation via MCQ log-likelihood scoring.
    40 curated questions across 5 CS subcategories.
    """
    try:
        from tests.eval_suite_prompts import CS_QA_QUESTIONS
    except ImportError:
        import importlib, pathlib
        spec = importlib.util.spec_from_file_location(
            "eval_suite_prompts",
            pathlib.Path(__file__).parent.parent.parent.parent / "tests" / "eval_suite_prompts.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        CS_QA_QUESTIONS = mod.CS_QA_QUESTIONS

    print(f"\n{'='*60}\n  CS Knowledge QA ({len(CS_QA_QUESTIONS)} MCQ questions)\n{'='*60}")
    model.eval()

    LETTERS = ["A", "B", "C", "D"]
    correct, total = 0, 0
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for i, q in enumerate(CS_QA_QUESTIONS):
        # Format prompt
        prompt = f"Question: {q['q']}\n"
        for j, c in enumerate(q['c']):
            prompt += f"{LETTERS[j]}. {c}\n"
        prompt += "Answer:"

        # Score each choice
        scores = []
        for j in range(len(q['c'])):
            score = _compute_log_likelihood(model, prompt, f" {LETTERS[j]}", device)
            scores.append(score)

        pred_idx = max(range(len(scores)), key=lambda x: scores[x])
        is_correct = (pred_idx == q['a'])
        if is_correct:
            correct += 1
        total += 1
        cat_stats[q['cat']]["total"] += 1
        if is_correct:
            cat_stats[q['cat']]["correct"] += 1

        results.append({
            "question": q['q'][:80], "predicted": LETTERS[pred_idx],
            "correct": LETTERS[q['a']], "is_correct": is_correct, "category": q['cat'],
        })

    accuracy = correct / total if total else 0.0

    # Print per-category breakdown
    print(f"\n  {'Category':<20} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print(f"  {'─'*44}")
    for cat in ["algorithms", "networking", "systems", "software_eng", "databases"]:
        s = cat_stats.get(cat, {"correct": 0, "total": 0})
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"  {cat:<20} {s['correct']:>8} {s['total']:>6} {acc:>7.1%}")
    print(f"  {'─'*44}")
    print(f"  {'OVERALL':<20} {correct:>8} {total:>6} {accuracy:>7.1%}")

    return {
        "benchmark": "CS_QA", "accuracy": accuracy,
        "correct": correct, "total": total,
        "per_category": {c: dict(s) for c, s in cat_stats.items()},
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  5. Domain Perplexity  →  All Phase 2 categories
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _DomainSpec:
    name: str
    key: str
    repo_id: str
    weight: int
    fmt_fn: str
    config_name: Optional[str] = None
    data_dir: Optional[str] = None

_PHASE2_DOMAINS = [
    _DomainSpec("Python Code", "code_python", "bigcode/starcoderdata", 14, "starcoder", data_dir="python"),
    _DomainSpec("JS Code", "code_js", "bigcode/starcoderdata", 7, "starcoder", data_dir="javascript"),
    _DomainSpec("Edu Code", "edu_code", "nampdn-ai/tiny-codes", 15, "tiny_codes"),
    _DomainSpec("CS Knowledge", "cs_knowledge", "common-pile/stackexchange", 18, "stackexchange"),
    _DomainSpec("FineWeb-Edu", "fineweb_edu", "HuggingFaceFW/fineweb-edu", 10, "fineweb_edu", config_name="sample-100BT"),
    _DomainSpec("Wikipedia", "wikipedia", "wikimedia/wikipedia", 5, "wikipedia", config_name="20231101.en"),
]

def _fmt_for_ppl(row, fmt_fn: str) -> Optional[str]:
    """Minimal format functions for domain PPL (mirrors dataloader.py)."""
    if fmt_fn == "starcoder":
        c = row.get("content", "")
        return c if c and 100 < len(c) < 100_000 else None
    elif fmt_fn == "tiny_codes":
        c = row.get("response", "")
        return c.strip() if c and len(c.strip()) > 100 else None
    elif fmt_fn == "stackexchange":
        t = row.get("text", "")
        return t if t and len(t) > 200 else None
    elif fmt_fn == "fineweb_edu":
        s = row.get("score", 0.0)
        if s is None or s < 3.5:
            return None
        return row.get("text", "") or None
    elif fmt_fn == "wikipedia":
        t = row.get("text", "")
        title = row.get("title", "")
        if not t or len(t.strip()) < 500:
            return None
        return f"{title}\n\n{t}" if title else t
    else:
        return row.get("text", "") or None


@torch.inference_mode()
def run_domain_ppl(
    model, device: str = "cuda",
    samples_per_domain: int = 50, batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Compute per-domain perplexity on held-out data from each Phase 2 source.
    Uses streaming to avoid downloading full datasets.
    """
    print(f"\n{'='*60}\n  Domain Perplexity ({samples_per_domain} samples/domain)\n{'='*60}")
    from datasets import load_dataset

    model.eval()
    ctx_len = config.max_context_len
    chunk_size = ctx_len + 1
    eos_id = tokenizer.eos_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=eos_id)
    domain_results = {}

    for dom in _PHASE2_DOMAINS:
        print(f"  → {dom.name} ({dom.weight}%)...", end=" ", flush=True)
        try:
            kwargs = {}
            if dom.data_dir:
                kwargs["data_dir"] = dom.data_dir
            if dom.config_name:
                kwargs["name"] = dom.config_name
            stream = load_dataset(dom.repo_id, split="train", streaming=True, **kwargs)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # Tokenize and chunk
        buffer, batches = [], []
        rows_scanned, deadline = 0, time.monotonic() + 60  # 60s timeout per domain

        for row in stream:
            if time.monotonic() > deadline:
                break
            rows_scanned += 1
            if rows_scanned > 200_000:
                break

            text = _fmt_for_ppl(row, dom.fmt_fn)
            if not text:
                continue

            buffer.extend(tokenizer.encode(text))
            buffer.append(eos_id)

            while len(buffer) >= chunk_size:
                batches.append(torch.tensor(buffer[:chunk_size], dtype=torch.long))
                buffer = buffer[chunk_size:]
                if len(batches) >= samples_per_domain:
                    break
            if len(batches) >= samples_per_domain:
                break

        if not batches:
            print(f"NO DATA ({rows_scanned} rows scanned)")
            continue

        # Compute loss
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(batches), batch_size):
            batch = torch.stack(batches[start:start+batch_size]).to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=device.startswith("cuda") if isinstance(device, str) else False):
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        domain_results[dom.key] = {
            "name": dom.name, "weight": dom.weight,
            "loss": round(avg_loss, 4), "ppl": round(ppl, 2),
            "n_samples": len(batches),
        }
        print(f"loss={avg_loss:.4f} ppl={ppl:.2f} ({len(batches)} samples)")

    # Weighted average
    if domain_results:
        total_w = sum(r["weight"] for r in domain_results.values())
        w_loss = sum(r["loss"] * r["weight"] for r in domain_results.values()) / max(total_w, 1)
        w_ppl = math.exp(min(w_loss, 20))
        print(f"\n  Weighted avg: loss={w_loss:.4f} ppl={w_ppl:.2f}")
    else:
        w_loss, w_ppl = 0.0, 0.0

    return {
        "benchmark": "DomainPPL",
        "weighted_loss": round(w_loss, 4), "weighted_ppl": round(w_ppl, 2),
        "per_domain": domain_results,
    }


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════

ALL_BENCHMARKS = ["mbpp", "cruxeval", "code_completion", "cs_qa", "domain_ppl"]

def run_eval_suite(
    model, device: str = "cuda",
    benchmarks: Optional[List[str]] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite (or a subset).

    Args:
        model:      GPT_FLASH model (eval mode, unwrapped from torch.compile).
        device:     Device string.
        benchmarks: List of benchmark names to run (default: all).
        quick:      If True, use smaller problem counts for faster execution.

    Returns:
        Dict mapping benchmark names to their results.
    """
    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    # Problem counts: quick mode for training, full for standalone
    mbpp_n = 30 if quick else 100
    crux_n = 30 if quick else 100
    ppl_n = 25 if quick else 50

    runners = {
        "mbpp": lambda: run_mbpp(model, device, n_problems=mbpp_n),
        "cruxeval": lambda: run_cruxeval(model, device, n_problems=crux_n),
        "code_completion": lambda: run_code_completion(model, device),
        "cs_qa": lambda: run_cs_qa(model, device),
        "domain_ppl": lambda: run_domain_ppl(model, device, samples_per_domain=ppl_n),
    }

    all_results = {}
    suite_start = time.time()

    for name in benchmarks:
        runner = runners.get(name)
        if runner is None:
            print(f"[EvalSuite] Unknown benchmark: {name}. Available: {list(runners.keys())}")
            continue
        t0 = time.time()
        try:
            result = runner()
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            all_results[name] = result
        except Exception as e:
            print(f"[EvalSuite] ERROR in {name}: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    total_elapsed = round(time.time() - suite_start, 1)
    print(f"\n[EvalSuite] Total time: {total_elapsed:.0f}s")
    return all_results


def print_eval_summary(results: Dict[str, Any]):
    """Print a formatted summary table of all benchmark results."""
    print(f"\n{'='*70}")
    print(f"  EVAL SUITE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<22} {'Score':>12} {'Detail':>22} {'Time':>8}")
    print(f"  {'─'*64}")

    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<22} {'ERROR':>12} {r['error'][:22]:>22}")
            continue
        elapsed = r.get("elapsed_seconds", 0)
        if name == "mbpp":
            s = f"{r['pass_at_1']:.1%}"
            d = f"{r['passed']}/{r['total']} passed"
        elif name == "cruxeval":
            s = f"O:{r['output_pass_at_1']:.0%} I:{r['input_pass_at_1']:.0%}"
            d = f"O:{r['output_passed']} I:{r['input_passed']}/{r['total']}"
        elif name == "code_completion":
            s = f"{r['overall_score']:.3f}"
            d = f"{r['total']} prompts"
        elif name == "cs_qa":
            s = f"{r['accuracy']:.1%}"
            d = f"{r['correct']}/{r['total']} correct"
        elif name == "domain_ppl":
            s = f"PPL:{r['weighted_ppl']:.1f}"
            d = f"loss={r['weighted_loss']:.4f}"
        else:
            s = "—"
            d = "—"
        print(f"  {name:<22} {s:>12} {d:>22} {elapsed:>6.0f}s")

    print(f"  {'─'*64}\n")


# ═══════════════════════════════════════════════════════════════════
#  Training Integration
# ═══════════════════════════════════════════════════════════════════

def run_training_eval(
    model, device: str, wandb_run=None, train_step: int = 0,
    grad_accum_steps: int = 14,
    benchmarks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run eval suite during training. Manages model state transitions.

    Called from train.py — the model should be the UNWRAPPED raw model
    (via _unwrap), not the torch.compiled wrapper.

    Args:
        model:            Raw GPT_FLASH model.
        device:           Device string.
        wandb_run:        Active W&B run for metric logging.
        train_step:       Current optimizer step.
        grad_accum_steps: Grad accumulation steps (for W&B step alignment).
        benchmarks:       Which benchmarks to run (default: all, quick mode).
    """
    print(f"\n{'='*70}")
    print(f"  EVAL SUITE — Training Step {train_step}")
    print(f"{'='*70}")

    was_training = model.training
    model.eval()

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    results = run_eval_suite(model, device, benchmarks=benchmarks, quick=True)
    print_eval_summary(results)

    # Log to W&B
    if wandb_run is not None:
        metrics = {}
        for name, r in results.items():
            if "error" in r:
                continue
            if name == "mbpp":
                metrics["eval/mbpp_pass_at_1"] = r["pass_at_1"]
            elif name == "cruxeval":
                metrics["eval/cruxeval_output"] = r["output_pass_at_1"]
                metrics["eval/cruxeval_input"] = r["input_pass_at_1"]
            elif name == "code_completion":
                metrics["eval/code_completion_score"] = r["overall_score"]
                for lang, ls in r.get("per_language", {}).items():
                    metrics[f"eval/code_completion/{lang}"] = ls["avg"]
            elif name == "cs_qa":
                metrics["eval/cs_qa_accuracy"] = r["accuracy"]
                for cat, cs in r.get("per_category", {}).items():
                    if cs["total"] > 0:
                        metrics[f"eval/cs_qa/{cat}"] = cs["correct"] / cs["total"]
            elif name == "domain_ppl":
                metrics["eval/domain_ppl_weighted"] = r["weighted_ppl"]
                metrics["eval/domain_loss_weighted"] = r["weighted_loss"]
                for dk, dv in r.get("per_domain", {}).items():
                    metrics[f"eval/domain_ppl/{dk}"] = dv["ppl"]

        wandb_run.log(metrics, step=grad_accum_steps * train_step, commit=False)

    # Restore model state
    if was_training:
        model.train()

    return results
