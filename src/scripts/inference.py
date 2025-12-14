from ..models.model import GPT
from ..models.model_flash_attn import GPT_FLASH
from .tokenizer import tokenizer
from .configs import config
import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.inference_mode()
def generate(model,seed_txt,device,max_tokens=500,k=50,temp = 0.8):
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
    seed_txt = "In the field artifical intelligence"
    generated_text = generate(model,seed_txt,device)
    print(generated_text)
    
    # Access MoE expert counts through the transformer layers
    print("\n=== Expert Usage Statistics ===")
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'expert_counts'):
            moe = layer.mlp
            expert_counts = moe.expert_counts
            total_tokens = moe.total_tokens
            
            print(f"\nLayer {layer_idx} MoE:")
            print(f"  Total routed tokens: {total_tokens}")
            print(f"  Expert counts: {dict(enumerate(expert_counts.tolist()))}")
            
            # Calculate utilization percentages
            if total_tokens > 0:
                utilization = moe.get_expert_utilization()
                print("  Expert utilization:")
                for key, val in utilization.items():
                    expert_id = key.split('_')[1]
                    print(f"    Expert {expert_id}: {val*100:.2f}%")
