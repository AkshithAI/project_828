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


@torch.inference_mode()
def generate(model,seed_txt,device,max_tokens=500,k=50,temp = 0.8):
    """
    Sample Inference on the model
    
    Args:
        model: model object
        seed_txt: prompt for sequence generation
        device: torch.device() object
        max_tokens: max sequence length
        k: topk param for selecting top 'k' words from probability distribution
        temp: temperature for sequence generation
    """
    model.eval()

    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    sampled_tokens = []
    start_pos = 0
    tokens = torch.tensor(tokenizer.encode(seed_txt)[:-1], device = device, dtype = torch.long).unsqueeze(0)
    predicted_token = torch.tensor(tokenizer.encode(seed_txt)[-1], device = device, dtype = torch.long).unsqueeze(0)
    sampled_tokens.extend(tokens.squeeze(0).tolist())
    model(tokens,start_pos)
    start_pos = len(sampled_tokens)
    for _ in tqdm(range(max_tokens)):
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits = model(predicted_token.view(1, 1),start_pos)
        last_seq = logits[:,-1,:] / max(temp, 1e-5)
        top_k_logits, top_k_indices = torch.topk(last_seq, k, dim=-1)
        preds = F.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(preds, num_samples=1)
        idx = top_k_indices.gather(-1, sampled_idx)
        idx_item = idx.item()
        sampled_tokens.append(idx_item)
        tokens = torch.cat((tokens,idx),dim=-1)
        start_pos += 1
        predicted_token = idx
        if idx_item == tokenizer.eos_token_id:
            break
    print(f"Number of tokens sampled : {len(sampled_tokens)}")
    return tokenizer.decode(sampled_tokens)


@torch.inference_mode()
def generate_batch(model, prompts, device, max_tokens=500, k=50, temp=0.8):
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

    Returns:
        list of generated text strings (prompt + continuation)
    """
    model.eval()
    batch_size = len(prompts)

    # ---- Tokenize & left-pad ----
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

    # ---- Position IDs for prefill (actual positions; 0 for pad slots) ----
    position_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    for i in range(batch_size):
        position_ids[i, padding_lengths[i]:] = torch.arange(
            prompt_lengths[i], device=device
        )

    # ---- Prefill attention mask (causal + padding) ----
    prefill_mask = _build_prefill_mask(padding_lengths, max_prompt_len, device)

    # ---- Reset KV cache for this batch ----
    if hasattr(model, 'reset_cache'):
        model.reset_cache(batch_size)

    # ---- Prefill forward pass ----
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(
            tokens, start_pos=0,
            position_ids=position_ids, attn_mask=prefill_mask,
        )

    next_logits = logits[:, -1, :] / max(temp, 1e-5)
    top_k_logits, top_k_indices = torch.topk(next_logits, k, dim=-1)
    preds = F.softmax(top_k_logits, dim=-1)
    sampled_idx = torch.multinomial(preds, num_samples=1)
    next_tokens = top_k_indices.gather(-1, sampled_idx)        # (batch, 1)

    all_tokens = [list(encoded[i]) + [next_tokens[i].item()] for i in range(batch_size)]
    finished = [next_tokens[i].item() == tokenizer.eos_token_id for i in range(batch_size)]

    for step in tqdm(range(1, max_tokens), desc="Batch generation"):
        if all(finished):
            break

        start_pos = max_prompt_len + step - 1       
        kv_len = start_pos + 1                     

        # Per-sequence position IDs: (batch, 1)
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

        next_logits = logits[:, -1, :] / max(temp, 1e-5)
        top_k_logits, top_k_indices = torch.topk(next_logits, k, dim=-1)
        preds = F.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(preds, num_samples=1)
        next_tokens = top_k_indices.gather(-1, sampled_idx)

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
    
    # Reset expert counts from training before inference
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    
    print("  SINGLE SEQUENCE INFERENCE")

    seed_txt = "The investigation of past cultures of the modern"
    generated_text = generate(model,seed_txt,device)
    print(generated_text)
    
    # Display expert usage statistics
    display_expert_stats(model)
    
    print("\n")
    print("  BATCHED INFERENCE")

    # Reset expert counts between runs
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()

    test_prompts = [
        "The investigation of past cultures of the modern",
        "In the beginning, there was",
        "Artificial intelligence has transformed the way we",
        "The quick brown fox jumped over the",
        "Deep in the ocean, scientists discovered",
    ]

    print(f"Running batched inference on {len(test_prompts)} prompts...\n")
    batch_results = generate_batch(model, test_prompts, device, max_tokens=200)

    # Display expert stats after batched inference
    display_expert_stats(model)
