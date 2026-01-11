from ..models.model import GPT
from ..models.model_flash_attn import GPT_FLASH
from .tokenizer import tokenizer
from .configs import config
import torch
import torch.nn.functional as F
from tqdm import tqdm

def display_expert_stats(model):
    """
    Display expert usage statistics for each layer.

    """
    print("\n" + "="*70)
    print("Expert Usage Statistics".center(70))
    print("="*70)
    
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
    
    print("="*70)

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
        last_seq = logits[:,-1,:]
        preds = F.softmax(last_seq/temp,dim=-1)
        idx = torch.multinomial(preds,num_samples=1)
        idx_item = idx.item()
        sampled_tokens.append(idx_item)
        tokens = torch.cat((tokens,idx),dim=-1)
        start_pos += 1
        predicted_token = idx
        if idx_item == tokenizer.eos_token_id:
            break
    print(f"Number of tokens sampled : {len(sampled_tokens)}")
    return tokenizer.decode(sampled_tokens)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used : {device}")
    use_flash_attn = True
    if use_flash_attn:
        model = GPT_FLASH(config,device,inference=True)
    else:
        model = GPT(config,device)
    model.load_state_dict(torch.load("assets/model_24999.pt",map_location="cpu"))
    
    # Reset expert counts from training before inference
    for layer in model.layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_counts'):
            layer.mlp.reset_expert_counts()
    
    seed_txt = "In the field artifical intelligence"
    generated_text = generate(model,seed_txt,device)
    print(generated_text)
    
    # Display expert usage statistics
    display_expert_stats(model)
