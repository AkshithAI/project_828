from ..models.model import GPT
from ..models.model_flash_attn import GPT_FLASH
from .tokenizer import tokenizer
from .configs.model_config import config
import torch
import os
import torch.nn.functional as F
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
        scores = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(scores, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled


@torch.inference_mode()
def generate(model, seed_txt, device, max_tokens=500, k=50, temp=0.8,
            top_p=0.9, repetition_penalty=1.15):
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
    model(tokens, start_pos)
    start_pos = len(all_prompt_ids) - 1  
    generated_ids = torch.tensor([all_prompt_ids], device=device, dtype=torch.long)

    for _ in tqdm(range(max_tokens)):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(predicted_token.view(1, 1), start_pos)

        idx = _apply_sampling(
            logits[:, -1, :], temp, k, top_p, repetition_penalty, generated_ids
        )

        idx_item = idx.item()
        sampled_tokens.append(idx_item)
        generated_ids = torch.cat([generated_ids, idx], dim=-1)
        start_pos += 1
        predicted_token = idx
        if idx_item == tokenizer.eos_token_id:
            break

    # Restore original model state
    if needs_cache_toggle:
        _disable_kv_cache(model)
    if was_training:
        model.train()

    print(f"Number of tokens sampled : {len(sampled_tokens)}")
    return tokenizer.decode(sampled_tokens)


@torch.inference_mode()
def generate_batch(model, prompts, device, max_tokens=500, k=50, temp=0.8,
                   top_p=0.9, repetition_penalty=1.15):
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

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(
            tokens, start_pos=0,
            position_ids=position_ids, attn_mask=prefill_mask,
        )

    generated_ids = torch.full_like(tokens, -1)
    for i in range(batch_size):
        generated_ids[i, padding_lengths[i]:] = tokens[i, padding_lengths[i]:]

    next_tokens = _apply_sampling(
        logits[:, -1, :], temp, k, top_p, repetition_penalty, generated_ids
    )   # (batch, 1)

    all_tokens = [list(encoded[i]) + [next_tokens[i].item()] for i in range(batch_size)]
    finished = [next_tokens[i].item() == tokenizer.eos_token_id for i in range(batch_size)]
    generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

    for step in tqdm(range(1, max_tokens), desc="Batch generation"):
        if all(finished):
            break

        start_pos = max_prompt_len + step - 1       
        kv_len = start_pos + 1                     

        step_position_ids = torch.tensor(
            [[prompt_lengths[i] + step - 1] for i in range(batch_size)],
            device=device, dtype=torch.long,
        )

        kv_mask = _build_kv_mask(padding_lengths, kv_len, device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(
                next_tokens, start_pos=start_pos,
                position_ids=step_position_ids, attn_mask=kv_mask,
            )

        next_tokens = _apply_sampling(
            logits[:, -1, :], temp, k, top_p, repetition_penalty, generated_ids
        )
        generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

        for i in range(batch_size):
            if not finished[i]:
                tok = next_tokens[i].item()
                all_tokens[i].append(tok)
                if tok == tokenizer.eos_token_id:
                    finished[i] = True

    results = []
    for i in range(batch_size):
        text = tokenizer.decode(all_tokens[i])
        results.append(text)
        print(f"\n[Sequence {i}] ({len(all_tokens[i])} tokens):")
        print(text)

    return results


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used : {device}")
    use_flash_attn = True
    if use_flash_attn:
        model = GPT_FLASH(config,device,inference=True)
    else:
        model = GPT(config,device)
    model.load_state_dict(torch.load("./project-828/project_828/checkpoints/model_00000.pt",map_location="cpu"))
    # Reset expert counts from training before inference
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    
    print("  SINGLE SEQUENCE INFERENCE")

    seed_txt = "The theory of general relativity, published by Albert Einstein in 1915, states that"
    generated_text = generate(model,seed_txt,device)
    print(generated_text)
    
    print("\n")
    print("  BATCHED INFERENCE")

    # Reset expert counts between runs
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()

    test_prompts = [
        "Chapter 1. The dark forest was",
        "The following is a Python function that reverses a string:\n\ndef reverse_string(s):",
        "To solve the quadratic equation x^2 - 5x + 6 = 0, we first",
        "The theory of general relativity, published by Albert Einstein in 1915, states that",
        "In this essay, I will argue that renewable energy is essential for economic growth because",
    ]

    print(f"Running batched inference on {len(test_prompts)} prompts...\n")
    batch_results = generate_batch(model, test_prompts, device, max_tokens=200)

    # Display expert stats after batched inference
    display_expert_stats(model)
