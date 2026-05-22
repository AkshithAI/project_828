from ..models.model_flash_attn import GPT_FLASH
from .tokenizer import tokenizer
from .configs.model_config import config
import torch
import os
import torch.nn.functional as F
import time
from contextlib import nullcontext
from tqdm import tqdm

def display_expert_stats(model):
    """
    Display expert usage statistics for each layer.

    """
    print("\n" + "-"*50)
    print("Expert Usage Statistics".center(70))
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'expert_counts'):
            moe = layer.mlp
            expert_counts = moe.expert_counts
            total_tokens = moe.total_tokens
            
            if total_tokens == 0:
                continue
                
            print(f"\nLayer {layer_idx}:")
            print(f"  Total routed tokens: {total_tokens}")
            print(f"  Expert distribution:")
            
            # Sort experts by usage (descending)
            sorted_indices = torch.argsort(expert_counts, descending=True)
            
            for idx in sorted_indices:
                count = expert_counts[idx].item()
                percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
                bar_length = int(percentage / 2)  # Scale bar to 50 chars max
                bar = "█" * bar_length
                print(f"    Expert {idx:2d}: {count:6d} ({percentage:5.1f}%) {bar}")
            
            # Calculate load balance metrics
            num_experts = len(expert_counts)
            mean_count = total_tokens / num_experts
            variance = ((expert_counts - mean_count) ** 2).mean().item()
            std_dev = variance ** 0.5
            cv = (std_dev / mean_count * 100) if mean_count > 0 else 0  # Coefficient of variation
            
            print(f"  Load balance metrics:")
            print(f"    Mean: {mean_count:.1f} | Std Dev: {std_dev:.1f} | CV: {cv:.1f}%")


def _is_cuda_device(device):
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    if isinstance(device, str):
        return device.startswith('cuda')
    return False


def _sync_device(device):
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def _autocast_ctx(device):
    if _is_cuda_device(device):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()
    

def _build_prefill_mask(padding_lengths, max_prompt_len, device):
    """
    Build a float attention mask for the prefill phase.
    Combines causal masking with left-padding masking.

    Returns:
        Tensor of shape (batch, 1, max_prompt_len, max_prompt_len)
        with 0.0 for attend and -inf for masked positions.
    """
    batch_size = len(padding_lengths)
    causal = torch.tril(
        torch.ones(max_prompt_len, max_prompt_len, dtype=torch.bool, device=device)
    )
    is_real = torch.zeros(batch_size, max_prompt_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        is_real[i, padding_lengths[i]:] = True

    attend = causal.unsqueeze(0) & is_real.unsqueeze(1) # [1,mpl,mpl] & [bs,1,mpl]
    mask = torch.where(attend, 0.0, float('-inf'))
    return mask.unsqueeze(1)   # (batch, 1, seq, seq)


def _build_kv_mask(padding_lengths, kv_len, device):
    """
    Build a float attention mask for autoregressive decoding steps.
    Masks out left-padding positions in the KV cache.

    Returns:
        Tensor of shape (batch, 1, 1, kv_len)
    """
    batch_size = len(padding_lengths)
    mask = torch.zeros(batch_size, 1, 1, kv_len, device=device)
    for i in range(batch_size):
        if padding_lengths[i] > 0:
            mask[i, :, :, :padding_lengths[i]] = float('-inf')
    return mask


def _enable_kv_cache(model):
    """Temporarily enable KV-cache inference on a model that was built without it."""
    model.inference = True
    for layer in model.layers:
        layer.attention.inference = True


def _disable_kv_cache(model):
    """Restore the model to its non-inference (training) state."""
    model.inference = False
    for layer in model.layers:
        attn = layer.attention
        attn.inference = False
        attn.cache_k = None
        attn.cache_v = None


def _apply_sampling(logits, temp, k, top_p, repetition_penalty, generated_ids):
    """
    Shared sampling logic: repetition penalty -> temperature -> top-k -> top-p -> sample.

    Args:
        logits:             Raw logits of shape (batch, vocab_size).
        temp:               Temperature (>0). Lower = more deterministic.
        k:                  Top-k filtering. 0 disables top-k.
        top_p:              Nucleus (top-p) probability mass threshold in (0, 1].
        repetition_penalty: Multiplicative penalty (>=1.0). 1.0 = no penalty.
        generated_ids:      Tensor of shape (batch, seq) with previously generated
                            token ids used for the repetition penalty.

    Returns:
        Sampled token ids of shape (batch, 1).
    """

    scores = logits.float()

    if repetition_penalty > 1.0 and generated_ids.numel() > 0:
        for i in range(scores.size(0)):
            prev_tokens = generated_ids[i].unique()
            prev_tokens = prev_tokens[prev_tokens >= 0]  
            token_scores = scores[i, prev_tokens]

            scores[i, prev_tokens] = torch.where(
                token_scores > 0,
                token_scores / repetition_penalty,
                token_scores * repetition_penalty,
            )

    scores = scores / max(temp, 1e-5)

    if k > 0:
        top_k_logits, top_k_indices = torch.topk(scores, min(k, scores.size(-1)), dim=-1)
        filter_mask = torch.full_like(scores, float('-inf'))
        filter_mask.scatter_(-1, top_k_indices, top_k_logits)
        scores = filter_mask

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        scores = torch.zeros_like(scores).scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(scores, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled


@torch.inference_mode()
def generate(model, seed_txt, device, max_tokens=500, k=50, temp=0.8,
            top_p=0.9, repetition_penalty=1.15,
            report_perf=True, show_progress=True):
    """
    Sample Inference on the model
    
    Args:
        model: model object
        seed_txt: prompt for sequence generation
        device: torch.device() object
        max_tokens: max sequence length
        k: topk param for selecting top 'k' words from probability distribution
        temp: temperature for sequence generation
        top_p: nucleus sampling threshold (0.0-1.0). 1.0 disables it.
        repetition_penalty: penalise repeated tokens (>=1.0). 1.0 disables it.
    """
    was_training = model.training
    model.eval()

    needs_cache_toggle = not getattr(model, 'inference', False)
    if needs_cache_toggle:
        _enable_kv_cache(model)

    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    sampled_tokens = []
    start_pos = 0
    all_prompt_ids = tokenizer.encode(seed_txt)
    tokens = torch.tensor(all_prompt_ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
    predicted_token = torch.tensor(all_prompt_ids[-1], device=device, dtype=torch.long).unsqueeze(0)
    sampled_tokens.extend(all_prompt_ids)

    _sync_device(device)
    prefill_start = time.perf_counter()
    with _autocast_ctx(device):
        model(tokens, start_pos)
    _sync_device(device)
    prefill_sec = time.perf_counter() - prefill_start

    start_pos = len(all_prompt_ids) - 1
    prompt_len = len(all_prompt_ids)
    generated_ids = torch.full((1, prompt_len + max_tokens), -1, device=device, dtype=torch.long)
    generated_ids[0, :prompt_len] = torch.tensor(all_prompt_ids, device=device, dtype=torch.long)
    gen_idx = prompt_len  

    _sync_device(device)
    decode_start = time.perf_counter()
    for _ in tqdm(range(max_tokens), desc="Decode", disable=not show_progress):
        with _autocast_ctx(device):
            logits = model(predicted_token.view(1, 1), start_pos)

        idx = _apply_sampling(
            logits[:, -1, :], temp, k, top_p, repetition_penalty,
            generated_ids[:, :gen_idx],
        )

        idx_item = idx.item()
        sampled_tokens.append(idx_item)
        generated_ids[0, gen_idx] = idx_item
        gen_idx += 1
        start_pos += 1
        predicted_token = idx
        if idx_item == tokenizer.eos_token_id:
            break

    _sync_device(device)
    decode_sec = time.perf_counter() - decode_start

    # Restore original model state
    if needs_cache_toggle:
        _disable_kv_cache(model)
    if was_training:
        model.train()

    if report_perf:
        print(f"Number of tokens sampled : {len(sampled_tokens)}")

        generated_tokens = len(sampled_tokens) - len(all_prompt_ids)
        total_sec = prefill_sec + decode_sec
        decode_tps = generated_tokens / max(decode_sec, 1e-9)
        total_tps = len(sampled_tokens) / max(total_sec, 1e-9)
        decode_ms_per_tok = (decode_sec * 1000.0) / max(generated_tokens, 1)
        print("\n[Perf] single-sequence")
        print(
            f"  prompt_tokens={len(all_prompt_ids)} generated_tokens={generated_tokens} total_tokens={len(sampled_tokens)}"
        )
        print(f"  prefill: {prefill_sec*1000:.2f} ms")
        print(
            f"  decode: {decode_sec:.3f} s | {decode_tps:.2f} tok/s | {decode_ms_per_tok:.2f} ms/tok"
        )
        print(f"  total : {total_sec:.3f} s | {total_tps:.2f} tok/s")
    return tokenizer.decode(sampled_tokens)


@torch.inference_mode()
def generate_batch(model, prompts, device, max_tokens=500, k=50, temp=0.8,
                   top_p=0.9, repetition_penalty=1.15,
                   report_perf=True, show_progress=True):
    """
    Batched inference with KV caching for multiple prompts simultaneously.

    Uses left-padding with per-sequence position IDs and attention masks
    so that prompts of varying lengths are handled correctly with RoPE.

    Args:
        model: GPT model instantiated with inference=True
        prompts: list of prompt strings
        device: device string ('cuda' or 'cpu')
        max_tokens: max *new* tokens to generate per sequence
        k: top-k sampling parameter
        temp: temperature for sampling
        top_p: nucleus sampling threshold (0.0-1.0). 1.0 disables it.
        repetition_penalty: penalise repeated tokens (>=1.0). 1.0 disables it.

    Returns:
        list of generated text strings (prompt + continuation)
    """
    model.eval()
    batch_size = len(prompts)

    encoded = [tokenizer.encode(p) for p in prompts]
    prompt_lengths = [len(e) for e in encoded]
    max_prompt_len = max(prompt_lengths)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded, padding_lengths = [], []
    for e in encoded:
        pad_len = max_prompt_len - len(e)
        padding_lengths.append(pad_len)
        padded.append([pad_id] * pad_len + e)

    tokens = torch.tensor(padded, device=device, dtype=torch.long)

    position_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    for i in range(batch_size):
        position_ids[i, padding_lengths[i]:] = torch.arange(
            prompt_lengths[i], device=device
        )

    prefill_mask = _build_prefill_mask(padding_lengths, max_prompt_len, device)

    if hasattr(model, 'reset_cache'):
        model.reset_cache(batch_size)

    _sync_device(device)
    prefill_start = time.perf_counter()
    with _autocast_ctx(device):
        logits = model(
            tokens, start_pos=0,
            position_ids=position_ids, attn_mask=prefill_mask,
        )
    _sync_device(device)
    prefill_sec = time.perf_counter() - prefill_start

    total_buf_len = max_prompt_len + max_tokens
    generated_ids = torch.full((batch_size, total_buf_len), -1, device=device, dtype=torch.long)
    for i in range(batch_size):
        generated_ids[i, padding_lengths[i]:max_prompt_len] = tokens[i, padding_lengths[i]:]
    gen_idx = max_prompt_len  

    _sync_device(device)
    decode_start = time.perf_counter()
    next_tokens = _apply_sampling(
        logits[:, -1, :], temp, k, top_p, repetition_penalty,
        generated_ids[:, :gen_idx],
    )

    all_tokens = [list(encoded[i]) + [next_tokens[i].item()] for i in range(batch_size)]
    finished = [next_tokens[i].item() == tokenizer.eos_token_id for i in range(batch_size)]
    for i in range(batch_size):
        generated_ids[i, gen_idx] = next_tokens[i]
    gen_idx += 1

    frozen_position = [prompt_lengths[i] for i in range(batch_size)]

    for step in tqdm(range(1, max_tokens), desc="Batch generation", disable=not show_progress):
        if all(finished):
            break

        start_pos = max_prompt_len + step - 1
        kv_len = start_pos + 1

        step_position_ids = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        for i in range(batch_size):
            if finished[i]:
                next_tokens[i, 0] = pad_id
                step_position_ids[i, 0] = frozen_position[i]
            else:
                step_position_ids[i, 0] = prompt_lengths[i] + step - 1

        kv_mask = _build_kv_mask(padding_lengths, kv_len, device)

        with _autocast_ctx(device):
            logits = model(
                next_tokens, start_pos=start_pos,
                position_ids=step_position_ids, attn_mask=kv_mask,
            )

        next_tokens = _apply_sampling(
            logits[:, -1, :], temp, k, top_p, repetition_penalty,
            generated_ids[:, :gen_idx],
        )
        for i in range(batch_size):
            generated_ids[i, gen_idx] = next_tokens[i]
        gen_idx += 1

        for i in range(batch_size):
            if not finished[i]:
                tok = next_tokens[i].item()
                all_tokens[i].append(tok)
                if tok == tokenizer.eos_token_id:
                    finished[i] = True

    _sync_device(device)
    decode_sec = time.perf_counter() - decode_start

    results = []
    generated_counts = []
    for i in range(batch_size):
        text = tokenizer.decode(all_tokens[i])
        results.append(text)
        generated_counts.append(len(all_tokens[i]) - prompt_lengths[i])
        print(f"\n[Sequence {i}] ({len(all_tokens[i])} tokens):")
        print(text)

    if report_perf:
        total_new_tokens = sum(generated_counts)
        total_tokens = sum(len(seq) for seq in all_tokens)
        total_sec = prefill_sec + decode_sec
        decode_tps = total_new_tokens / max(decode_sec, 1e-9)
        total_tps = total_tokens / max(total_sec, 1e-9)
        decode_ms_per_tok = (decode_sec * 1000.0) / max(total_new_tokens, 1)
        print("\n[Perf] batch")
        print(
            f"  batch_size={batch_size} prompt_tokens={sum(prompt_lengths)} generated_tokens={total_new_tokens} total_tokens={total_tokens}"
        )
        print(f"  prefill: {prefill_sec*1000:.2f} ms")
        print(
            f"  decode: {decode_sec:.3f} s | {decode_tps:.2f} tok/s | {decode_ms_per_tok:.2f} ms/tok"
        )
        print(f"  total : {total_sec:.3f} s | {total_tps:.2f} tok/s")

    return results


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used : {device}")
    model = GPT_FLASH(config,device,inference=True)
    model.load_state_dict(torch.load("/Users/apple/Documents/project-828/project_828/checkpoints/model_08000.pt",map_location="cpu"))
    # Reset expert counts from training before inference
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    
    print("  SINGLE SEQUENCE INFERENCE")

    seed_txt = "def sliding_window_average(values, window_size):\n    \"\"\"Compute moving averages for a list of numbers.\"\"\"\n    if window_size <= 0:\n        raise ValueError('window_size must be positive')\n    if len(values) < window_size:\n        return []\n    window_sum = sum(values[:window_size])\n    averages = [window_sum / window_size]\n    for i in range(window_size, len(values)):\n        window_sum += values[i] - values[i - window_size]\n        averages.append(window_sum / window_size)\n    return"
    generated_text = generate(model,seed_txt,device,max_tokens=250,temp=0.7,top_p=0.9,k=50,repetition_penalty=1.25)
    print(generated_text)
    
    print("\n")
    print("  BATCHED INFERENCE")

    # Reset expert counts between runs
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()

    test_prompts = [
        # 1. Python — data structures (Source Code 50%, Python 16%)
        "def lru_cache(capacity):\n    from collections import OrderedDict\n    cache = OrderedDict()\n    def get(key):\n        if key not in cache:\n            return -1\n        cache.move_to_end(key)\n        return cache[key]\n    def put(key, value):",

        # 2. Go — concurrent server (Source Code 50%, Go 3%)
        "package main\n\nimport (\n    \"fmt\"\n    \"net/http\"\n    \"sync\"\n)\n\ntype SafeCounter struct {\n    mu sync.Mutex\n    v  map[string]int\n}\n\nfunc (c *SafeCounter) Inc(key string) {",

        # 3. Rust — ownership and borrowing (Source Code 50%, Rust 2%)
        "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() > y.len() {\n        x\n    } else {\n        y\n    }\n}\n\nfn main() {\n    let string1 = String::from(\"hello world\");\n    let result;",

        # 4. JavaScript — async patterns (Source Code 50%, JS 7%)
        "async function fetchWithRetry(url, maxRetries = 3) {\n  for (let attempt = 0; attempt < maxRetries; attempt++) {\n    try {\n      const response = await fetch(url);\n      if (!response.ok) throw new Error(`HTTP ${response.status}`);\n      return await response.json();\n    } catch (err) {",

        # 5. General Knowledge — educational (General Knowledge 18%, fineweb-edu-dedup)
        "In computer networking, the OSI model defines seven layers of communication. The Transport Layer (Layer 4) is responsible for end-to-end communication and flow control. The two main protocols at this layer are TCP and UDP. TCP provides reliable, ordered delivery through",

        # 6. Math — clean reasoning (Math/Reasoning 10%, finemath)
        "A factory produces items with a 5% defect rate. If a quality inspector randomly selects 20 items, what is the probability that exactly 2 are defective?\n\nUsing the binomial probability formula: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)\nwhere n=20, k=2, p=0.05\n\nFirst, C(20,2) = 20! / (2! * 18!) =",

        # 7. CS Q&A — StackExchange style (CS/Engineering 22%, stackexchange 10%)
        "Question: How does garbage collection work in Java compared to manual memory management in C++?\n\nAnswer: In Java, the JVM automatically manages memory through garbage collection. The garbage collector identifies objects that are no longer reachable from any GC root (such as local variables, static fields, or active threads) and",

        # 8. Code task — OpenCodeInstruct style (CS/Engineering 22%, opencodeinstruct 12%)
        "Write a function that implements binary search on a sorted array and returns the index of the target element, or -1 if not found.\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:",
    ]

    print(f"Running batched inference on {len(test_prompts)} prompts...\n")
    batch_results = generate_batch(model, test_prompts, device, max_tokens=200)

    # Display expert stats after batched inference
    display_expert_stats(model)
