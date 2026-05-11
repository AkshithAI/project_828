"""
Standardized Evaluation Benchmarks — HumanEval, MMLU-CS, MMLU-Pro, ARC
=======================================================================
Evaluates the Project 828 model on standard benchmarks:

  1. HumanEval       — Code generation with execution-based pass@k
  2. MMLU-CS         — Multiple-choice CS knowledge (log-likelihood)
  3. MMLU-Pro-CS     — Official TIGER-AI-Lab CoT evaluation (CS only)
  4. ARC             — AI2 Reasoning Challenge (log-likelihood)

Usage:
    python -m private.eval_benchmarks --checkpoint checkpoints/model_101002.pt
    python -m private.eval_benchmarks --checkpoint checkpoints/model_101002.pt --bench humaneval
    python -m private.eval_benchmarks --checkpoint checkpoints/model_101002.pt --bench mmlu_cs
    python -m private.eval_benchmarks --checkpoint checkpoints/model_101002.pt --bench mmlu_pro_cs
    python -m private.eval_benchmarks --checkpoint checkpoints/model_101002.pt --bench arc
"""

import os, sys, json, time, re, math, argparse, signal, traceback
import multiprocessing
from typing import Dict, Any, Optional, List, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scripts.inference import generate
from src.scripts.tokenizer import tokenizer
from src.scripts.configs.model_config import config
from src.models.model_flash_attn import GPT_FLASH


# ═══════════════════════════════════════════════════════════════════
#  Lightweight generation with early stopping (for CoT benchmarks)
# ═══════════════════════════════════════════════════════════════════

def _generate_with_early_stop(
    model, prompt: str, device: str,
    max_tokens: int = 384,
    stop_strings: list = None,
    check_every: int = 8,
) -> str:
    """
    Greedy generation with early stopping on stop_strings.

    Unlike the main generate() function, this:
      - Uses greedy decoding (argmax, no sampling)
      - Checks for stop_strings every `check_every` tokens
      - Returns only the generated text (not the prompt)

    This is ~10-50x faster than generate(max_tokens=1024) for CoT
    evaluation where answers typically appear within 50-200 tokens.
    """
    from src.scripts.inference import (
        _enable_kv_cache, _disable_kv_cache, _sync_device, _autocast_ctx,
    )

    if stop_strings is None:
        stop_strings = []

    was_training = model.training
    model.eval()

    needs_cache_toggle = not getattr(model, 'inference', False)
    if needs_cache_toggle:
        _enable_kv_cache(model)
    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    all_prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(all_prompt_ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
    predicted_token = torch.tensor(all_prompt_ids[-1], device=device, dtype=torch.long).unsqueeze(0)

    with _autocast_ctx(device):
        model(tokens, 0)

    start_pos = len(all_prompt_ids) - 1
    generated_ids = []

    for step in range(max_tokens):
        with _autocast_ctx(device):
            logits = model(predicted_token.view(1, 1), start_pos)

        # Greedy (argmax)
        idx = logits[:, -1, :].argmax(dim=-1)
        idx_item = idx.item()
        generated_ids.append(idx_item)
        start_pos += 1
        predicted_token = idx

        if idx_item == tokenizer.eos_token_id:
            break

        # Check stop strings periodically (decoding is expensive, don't do it every token)
        if stop_strings and (step + 1) % check_every == 0:
            partial_text = tokenizer.decode(generated_ids)
            for ss in stop_strings:
                if ss in partial_text:
                    # Truncate at the stop string
                    generated_ids = tokenizer.encode(partial_text[:partial_text.index(ss)])
                    break
            else:
                continue
            break  # stop string was found

    # Restore model state
    if needs_cache_toggle:
        _disable_kv_cache(model)
    if was_training:
        model.train()

    return tokenizer.decode(generated_ids)


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = "cuda") -> GPT_FLASH:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = GPT_FLASH(config, device, inference=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    del state
    model.eval()
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    print(f"Model loaded. Device: {device}")
    return model


@torch.inference_mode()
def compute_log_likelihood(
    model: GPT_FLASH,
    prompt_text: str,
    continuation_text: str,
    device: str = "cuda",
) -> float:
    """
    Compute the average log-likelihood of `continuation_text` given `prompt_text`.

    Used for MCQ benchmarks (MMLU, ARC) where we score each answer choice
    and pick the one with highest log-likelihood.

    Keeps the model in inference mode (uses SDPA attention path) to avoid
    requiring flash-attn. Resets KV cache and uses start_pos=0 for a
    full-sequence forward pass.
    """
    prompt_ids = tokenizer.encode(prompt_text)
    continuation_ids = tokenizer.encode(continuation_text)
    full_ids = prompt_ids + continuation_ids

    # Truncate to context length if needed
    max_len = config.max_context_len
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)

    # Reset KV cache for a clean full-sequence forward pass
    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    use_autocast = device.startswith("cuda") if isinstance(device, str) else False
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        logits = model(input_ids, start_pos=0)  # (1, seq_len, vocab_size)

    # Logits at position i predict token i+1
    # Continuation starts at prompt_len in full_ids
    # So we need logits at positions [prompt_len-1, ..., len(full_ids)-2]
    cont_start = max(prompt_len - 1, 0)
    cont_logits = logits[0, cont_start:, :]  # (cont_len, vocab)

    target_ids = torch.tensor(full_ids[prompt_len:], dtype=torch.long, device=device)

    min_len = min(cont_logits.shape[0], target_ids.shape[0])
    cont_logits = cont_logits[:min_len]
    target_ids = target_ids[:min_len]

    if min_len == 0:
        return float('-inf')

    log_probs = F.log_softmax(cont_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs.mean().item()


# ═══════════════════════════════════════════════════════════════════
#  1. HumanEval Benchmark
# ═══════════════════════════════════════════════════════════════════

def _load_humaneval() -> List[Dict[str, Any]]:
    """Load HumanEval dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return problems


def _unsafe_execute(code: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Execute code in a subprocess with timeout.
    Uses subprocess.run to avoid multiprocessing pickling issues.
    Returns (passed: bool, error_msg: str).
    """
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        else:
            error_msg = result.stderr.strip()
            # Extract just the last line (the actual error) for brevity
            if error_msg:
                lines = error_msg.strip().split('\n')
                error_msg = lines[-1] if lines else error_msg
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: execution exceeded time limit"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _extract_function_body(generated: str, entry_point: str) -> str:
    """Extract the generated function completion, stopping at the next top-level def/class."""
    lines = generated.split('\n')
    result_lines = []
    for line in lines:
        # Stop at a new top-level definition (not indented)
        stripped = line.lstrip()
        if result_lines and stripped and not line[0].isspace():
            if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                break
        result_lines.append(line)
    return '\n'.join(result_lines)


def run_humaneval(
    model: GPT_FLASH,
    device: str = "cuda",
    num_samples: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Run HumanEval benchmark.

    Args:
        model: The loaded model.
        device: Device string.
        num_samples: Number of samples per problem (for pass@k).
        max_tokens: Max tokens to generate per completion.
        temperature: Sampling temperature.

    Returns:
        Dict with pass@1 score and per-problem results.
    """
    print("\n" + "=" * 70)
    print("  HumanEval Benchmark")
    print("=" * 70)

    problems = _load_humaneval()
    print(f"  Loaded {len(problems)} problems")

    results = []
    passed_count = 0

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        # Reset KV cache
        if hasattr(model, 'reset_cache'):
            model.reset_cache()
        for layer in model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
                layer.mlp.reset_expert_counts()

        # Generate completion
        sample_passed = False
        sample_error = ""
        try:
            output = generate(
                model, prompt, device,
                max_tokens=max_tokens, temp=temperature,
                k=50, top_p=0.95, repetition_penalty=1.0,
                report_perf=False, show_progress=False,
            )
            generated = output[len(prompt):]
            completion = _extract_function_body(generated, entry_point)

            # Build full test program
            full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

            sample_passed, sample_error = _unsafe_execute(full_code, timeout=10)
        except Exception as e:
            generated = ""
            completion = ""
            sample_error = f"GenerationError: {e}"

        if sample_passed:
            passed_count += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"

        print(f"  [{i+1:3d}/{len(problems)}] {task_id:30s} {status}")
        if not sample_passed and sample_error:
            print(f"           Error: {sample_error[:100]}")

        results.append({
            "task_id": task_id,
            "passed": sample_passed,
            "error": sample_error,
            "completion": completion[:500] if not sample_passed else "[passed]",
        })

    pass_at_1 = passed_count / len(problems) if problems else 0.0
    print(f"\n  {'─' * 50}")
    print(f"  HumanEval pass@1: {pass_at_1:.1%} ({passed_count}/{len(problems)})")
    print(f"  {'─' * 50}")

    return {
        "benchmark": "HumanEval",
        "pass_at_1": pass_at_1,
        "passed": passed_count,
        "total": len(problems),
        "temperature": temperature,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  2. MMLU-CS Benchmark
# ═══════════════════════════════════════════════════════════════════

# CS-related MMLU subjects
MMLU_CS_SUBJECTS = [
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "electrical_engineering",
]

MMLU_ANSWER_CHOICES = ["A", "B", "C", "D"]


def _load_mmlu_cs() -> List[Dict[str, Any]]:
    """Load MMLU CS-related subjects from HuggingFace."""
    from datasets import load_dataset
    all_questions = []

    for subject in MMLU_CS_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            for row in ds:
                choices = row["choices"]
                answer_idx = row["answer"]
                all_questions.append({
                    "subject": subject,
                    "question": row["question"],
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "answer_letter": MMLU_ANSWER_CHOICES[answer_idx],
                })
        except Exception as e:
            print(f"  WARNING: Could not load MMLU subject '{subject}': {e}")
            continue

    return all_questions


def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Format an MMLU question as a multiple-choice prompt."""
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{MMLU_ANSWER_CHOICES[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def run_mmlu_cs(
    model: GPT_FLASH,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run MMLU-CS benchmark using log-likelihood scoring.

    For each question, we compute the log-likelihood of each answer choice
    given the prompt and select the highest-scoring one.
    """
    print("\n" + "=" * 70)
    print("  MMLU-CS Benchmark")
    print("=" * 70)

    questions = _load_mmlu_cs()
    print(f"  Loaded {len(questions)} questions across {len(MMLU_CS_SUBJECTS)} subjects")

    if not questions:
        return {"benchmark": "MMLU-CS", "accuracy": 0.0, "total": 0, "results": []}

    model.eval()

    correct = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for i, q in enumerate(questions):
        prompt = _format_mmlu_prompt(q["question"], q["choices"])

        # Score each answer choice
        scores = []
        for j, choice_letter in enumerate(MMLU_ANSWER_CHOICES[:len(q["choices"])]):
            # The continuation is just the answer letter (+ the choice text for better signal)
            continuation = f" {choice_letter}"
            score = compute_log_likelihood(model, prompt, continuation, device)
            scores.append(score)

        predicted_idx = max(range(len(scores)), key=lambda x: scores[x])
        predicted_letter = MMLU_ANSWER_CHOICES[predicted_idx]
        is_correct = predicted_idx == q["answer_idx"]

        if is_correct:
            correct += 1
        subject_stats[q["subject"]]["total"] += 1
        if is_correct:
            subject_stats[q["subject"]]["correct"] += 1

        if (i + 1) % 50 == 0 or i == len(questions) - 1:
            running_acc = correct / (i + 1)
            print(f"  [{i+1:4d}/{len(questions)}] Running accuracy: {running_acc:.1%}")

        results.append({
            "subject": q["subject"],
            "question": q["question"][:100],
            "predicted": predicted_letter,
            "correct": q["answer_letter"],
            "is_correct": is_correct,
            "scores": scores,
        })


    accuracy = correct / len(questions) if questions else 0.0

    print(f"\n  {'─' * 60}")
    print(f"  {'Subject':<35s} {'Acc':>8s} {'N':>6s}")
    print(f"  {'─' * 60}")
    for subj in MMLU_CS_SUBJECTS:
        stats = subject_stats[subj]
        if stats["total"] > 0:
            subj_acc = stats["correct"] / stats["total"]
            print(f"  {subj:<35s} {subj_acc:>7.1%} {stats['total']:>6d}")
    print(f"  {'─' * 60}")
    print(f"  {'OVERALL':<35s} {accuracy:>7.1%} {len(questions):>6d}")
    print(f"  {'─' * 60}")

    return {
        "benchmark": "MMLU-CS",
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
        "per_subject": dict(subject_stats),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  2.5 MMLU-Pro Benchmark — Official TIGER-AI-Lab Evaluation (CS Only)
#       https://github.com/TIGER-AI-Lab/MMLU-Pro
#       Uses CoT few-shot prompting + generative answer extraction
# ═══════════════════════════════════════════════════════════════════

MMLU_PRO_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _load_mmlu_pro_dataset():
    """Load MMLU-Pro test and validation sets, filtered for CS."""
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = _preprocess_mmlu_pro(dataset["test"])
    val_df = _preprocess_mmlu_pro(dataset["validation"])
    # Filter to computer science only
    test_df = [q for q in test_df if q["category"] == "computer science"]
    val_df = [q for q in val_df if q["category"] == "computer science"]
    return test_df, val_df


def _preprocess_mmlu_pro(split):
    """Remove N/A options (official preprocessing)."""
    res = []
    for each in split:
        options = [opt for opt in each["options"] if opt != "N/A"]
        entry = dict(each)
        entry["options"] = options
        res.append(entry)
    return res


def _format_cot_example(example, including_answer=True):
    """Format a single MMLU-Pro example in official CoT style."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(MMLU_PRO_CHOICES[i], opt)
    if including_answer:
        cot = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def _generate_cot_prompt(val_df, curr, k=5):
    """
    Build the official MMLU-Pro few-shot CoT prompt.

    Format:
      <system instruction>
      <k few-shot examples with CoT answers from val set>
      <current question ending with 'Answer: Let's think step by step.'>
    """
    system = (
        'The following are multiple choice questions (with answers) about '
        'computer science. Think step by step and then finish your answer '
        'with "the answer is (X)" where X is the correct letter choice.\n\n'
    )
    prompt = system
    examples = val_df[:k]
    for ex in examples:
        prompt += _format_cot_example(ex, including_answer=True)
    prompt += _format_cot_example(curr, including_answer=False)
    return prompt


def _extract_answer_l1(text):
    """Level 1: Official regex — 'answer is (X)'."""
    match = re.search(r"answer is \(?([A-J])\)?", text)
    return match.group(1) if match else None


def _extract_answer_l2(text):
    """Level 2: Fallback — 'Answer: X'."""
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    return match.group(1) if match else None


def _extract_answer_l3(text):
    """Level 3: Last standalone letter A-J in text."""
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def _extract_answer(text):
    """
    Official TIGER-AI-Lab 3-level answer extraction cascade.
    Returns the predicted letter or None.
    """
    pred = _extract_answer_l1(text)
    if pred is not None:
        return pred
    pred = _extract_answer_l2(text)
    if pred is not None:
        return pred
    return _extract_answer_l3(text)


def run_mmlu_pro_cs(
    model: GPT_FLASH,
    device: str = "cuda",
    n_shots: int = 5,
    max_tokens: int = 384,
) -> Dict[str, Any]:
    """
    Run MMLU-Pro CS using the official TIGER-AI-Lab CoT evaluation protocol.

    - 5-shot Chain-of-Thought prompting from the validation set
    - Greedy generation with early stopping on "Question:" (official stop token)
    - Answer extraction via the official 3-level regex cascade
    - Output format matches official submission schema

    Args:
        model:      The loaded GPT_FLASH model.
        device:     Device string.
        n_shots:    Number of few-shot examples (default 5, per official).
        max_tokens: Max new tokens to generate for CoT reasoning.

    Returns:
        Dict with accuracy, per-question results, and official-format output.
    """
    print("\n" + "=" * 70)
    print("  MMLU-Pro Benchmark — Official CoT Eval (Computer Science)")
    print("=" * 70)

    test_df, val_df = _load_mmlu_pro_dataset()
    print(f"  Loaded {len(test_df)} CS test questions, {len(val_df)} CS val examples")

    if not test_df:
        return {"benchmark": "MMLU-Pro-CS", "accuracy": 0.0, "total": 0, "results": []}

    model.eval()
    correct, wrong, no_answer = 0, 0, 0
    official_results = []  # Official submission format
    results = []           # Our internal format

    for i, curr in enumerate(test_df):
        # Build few-shot CoT prompt, reducing shots if it exceeds context
        k = n_shots
        prompt = _generate_cot_prompt(val_df, curr, k)
        prompt_ids = tokenizer.encode(prompt)

        # Shrink few-shot count until prompt fits in context
        # Leave room for generation: prompt + max_tokens <= context_len
        max_prompt_len = config.max_context_len - max_tokens
        while len(prompt_ids) >= max_prompt_len and k > 0:
            k -= 1
            prompt = _generate_cot_prompt(val_df, curr, k)
            prompt_ids = tokenizer.encode(prompt)

        if len(prompt_ids) >= config.max_context_len:
            # Even 0-shot doesn't fit — skip
            official_results.append({**curr, "pred": None, "model_outputs": "[SKIPPED: prompt too long]"})
            results.append({"id": curr.get("question_id", ""), "predicted": None,
                            "correct": curr["answer"], "is_correct": False, "skipped": True})
            wrong += 1
            continue

        # Generate with early stopping (stops at "Question:" like official vLLM config)
        try:
            generated_text = _generate_with_early_stop(
                model, prompt, device,
                max_tokens=max_tokens,
                stop_strings=["Question:"],
            )
        except Exception as e:
            generated_text = f"[ERROR: {e}]"

        # Extract answer using official regex cascade
        pred = _extract_answer(generated_text)
        is_correct = (pred == curr["answer"])

        if pred is None:
            no_answer += 1
            wrong += 1
        elif is_correct:
            correct += 1
        else:
            wrong += 1

        # Official submission format entry
        entry = dict(curr)
        entry["pred"] = pred
        entry["model_outputs"] = generated_text
        official_results.append(entry)

        status = "✓" if is_correct else ("? (no answer)" if pred is None else "✗")
        results.append({
            "id": curr.get("question_id", ""),
            "question": curr["question"][:80],
            "predicted": pred,
            "correct": curr["answer"],
            "is_correct": is_correct,
            "n_shots_used": k,
        })

        if (i + 1) % 25 == 0 or i == len(test_df) - 1:
            running_acc = correct / (i + 1)
            print(f"  [{i+1:4d}/{len(test_df)}] Acc: {running_acc:.1%}  "
                  f"(✓{correct} ✗{wrong} ?{no_answer})  {status} pred={pred} gt={curr['answer']}")

    accuracy = correct / len(test_df) if test_df else 0.0

    print(f"\n  {'─' * 60}")
    print(f"  MMLU-Pro-CS (Official CoT) accuracy: {accuracy:.1%} ({correct}/{len(test_df)})")
    print(f"  No answer extracted: {no_answer}")
    print(f"  {'─' * 60}")

    # Save official-format results for submission
    _save_official_results(official_results)

    return {
        "benchmark": "MMLU-Pro-CS",
        "accuracy": accuracy,
        "correct": correct,
        "wrong": wrong,
        "no_answer": no_answer,
        "total": len(test_df),
        "n_shots": n_shots,
        "results": results,
        "official_results_path": _official_results_path(),
    }


def _official_results_path() -> str:
    """Path for the official-format submission JSON."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mmlu_pro_cs_official_results.json",
    )


def _save_official_results(results: List[Dict]):
    """Save results in official TIGER-AI-Lab submission format."""
    out_path = _official_results_path()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Official results saved to: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  3. ARC Benchmark
# ═══════════════════════════════════════════════════════════════════

def _load_arc(challenge: bool = True) -> List[Dict[str, Any]]:
    """
    Load ARC dataset from HuggingFace.

    Args:
        challenge: If True, load ARC-Challenge. If False, load ARC-Easy.
    """
    from datasets import load_dataset
    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    ds = load_dataset("allenai/ai2_arc", subset, split="test")

    questions = []
    for row in ds:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = row["answerKey"]

        # Find correct answer index
        answer_idx = None
        for j, label in enumerate(labels):
            if label == answer_key:
                answer_idx = j
                break
        if answer_idx is None:
            continue

        questions.append({
            "id": row["id"],
            "question": row["question"],
            "choices": texts,
            "labels": labels,
            "answer_key": answer_key,
            "answer_idx": answer_idx,
        })
    return questions


def _format_arc_prompt(question: str, choices: List[str], labels: List[str]) -> str:
    """Format an ARC question."""
    prompt = f"Question: {question}\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:"
    return prompt


def run_arc(
    model: GPT_FLASH,
    device: str = "cuda",
    challenge: bool = True,
) -> Dict[str, Any]:
    """
    Run ARC benchmark using log-likelihood scoring.
    """
    subset_name = "ARC-Challenge" if challenge else "ARC-Easy"
    print("\n" + "=" * 70)
    print(f"  {subset_name} Benchmark")
    print("=" * 70)

    questions = _load_arc(challenge=challenge)
    print(f"  Loaded {len(questions)} questions")

    if not questions:
        return {"benchmark": subset_name, "accuracy": 0.0, "total": 0, "results": []}

    model.eval()

    correct = 0
    results = []

    for i, q in enumerate(questions):
        prompt = _format_arc_prompt(q["question"], q["choices"], q["labels"])

        # Score each answer choice
        scores = []
        for j, (label, choice_text) in enumerate(zip(q["labels"], q["choices"])):
            continuation = f" {label}"
            score = compute_log_likelihood(model, prompt, continuation, device)
            scores.append(score)

        predicted_idx = max(range(len(scores)), key=lambda x: scores[x])
        predicted_label = q["labels"][predicted_idx]
        is_correct = predicted_idx == q["answer_idx"]

        if is_correct:
            correct += 1

        if (i + 1) % 100 == 0 or i == len(questions) - 1:
            running_acc = correct / (i + 1)
            print(f"  [{i+1:4d}/{len(questions)}] Running accuracy: {running_acc:.1%}")

        results.append({
            "id": q["id"],
            "question": q["question"][:100],
            "predicted": predicted_label,
            "correct": q["answer_key"],
            "is_correct": is_correct,
            "scores": scores,
        })

    accuracy = correct / len(questions) if questions else 0.0

    print(f"\n  {'─' * 50}")
    print(f"  {subset_name} accuracy: {accuracy:.1%} ({correct}/{len(questions)})")
    print(f"  {'─' * 50}")

    return {
        "benchmark": subset_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main Runner
# ═══════════════════════════════════════════════════════════════════

def run_all_benchmarks(
    model: GPT_FLASH,
    device: str = "cuda",
    benchmarks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run selected benchmarks and return combined results."""

    available = {
        "humaneval": lambda: run_humaneval(model, device),
        "mmlu_cs": lambda: run_mmlu_cs(model, device),
        "mmlu_pro_cs": lambda: run_mmlu_pro_cs(model, device),
        "arc_challenge": lambda: run_arc(model, device, challenge=True),
        "arc_easy": lambda: run_arc(model, device, challenge=False),
    }

    if benchmarks is None:
        benchmarks = ["humaneval", "mmlu_cs", "arc_challenge"]

    all_results = {}
    for bench_name in benchmarks:
        runner = available.get(bench_name)
        if runner is None:
            print(f"Unknown benchmark: {bench_name}. Available: {list(available.keys())}")
            continue
        t0 = time.time()
        try:
            result = runner()
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            all_results[bench_name] = result
        except Exception as e:
            print(f"ERROR running {bench_name}: {e}")
            traceback.print_exc()
            all_results[bench_name] = {"error": str(e)}

    return all_results


def print_summary(results: Dict[str, Any]):
    """Print a final summary table."""
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Benchmark':<25s} {'Score':>10s} {'Details':>20s} {'Time':>10s}")
    print(f"  {'─' * 65}")

    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<25s} {'ERROR':>10s} {r['error'][:20]:>20s}")
            continue
        elapsed = r.get("elapsed_seconds", 0)
        if name == "humaneval":
            score = f"{r['pass_at_1']:.1%}"
            detail = f"{r['passed']}/{r['total']} passed"
        else:
            score = f"{r.get('accuracy', 0):.1%}"
            detail = f"{r.get('correct', 0)}/{r.get('total', 0)} correct"
        print(f"  {name:<25s} {score:>10s} {detail:>20s} {elapsed:>8.0f}s")

    print(f"  {'─' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--bench", type=str, nargs="+",
                        default=["humaneval", "mmlu_cs", "mmlu_pro_cs", "arc_challenge"],
                        choices=["humaneval", "mmlu_cs", "mmlu_pro_cs", "arc_challenge", "arc_easy"],
                        help="Which benchmarks to run")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: auto)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ckpt_path,
        )

    model = load_model(ckpt_path, device)

    results = run_all_benchmarks(model, device, benchmarks=args.bench)
    print_summary(results)

    # Save results
    out_path = args.output
    if out_path is None:
        # Auto-name based on checkpoint
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"eval_benchmarks_{ckpt_stem}.json",
        )

    # Strip per-problem results for compact JSON (keep summary)
    compact = {}
    for name, r in results.items():
        compact[name] = {k: v for k, v in r.items() if k != "results"}

    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"Results saved to: {out_path}")

    # Also save full results with per-problem details
    full_path = out_path.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full results saved to: {full_path}")
