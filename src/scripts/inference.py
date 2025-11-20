from ..models.model import GPT
from .tokenizer import tokenizer
from .configs import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
@torch.inference_mode
def generate(model,seed_txt,device,max_tokens=100,k=50,temp = 0.8):
    model.eval()
    sampled_tokens = []
    tokens = torch.tensor(tokenizer.encode(seed_txt),dtype=torch.long,device=device).unsqueeze(0)
    sampled_tokens.extend(tokenizer.encode(seed_txt))
    for _ in tqdm(range(max_tokens)):
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits = model(tokens)
        last_seq = logits[:,-1,:]
        preds = F.softmax(last_seq/temp,dim=-1)
        idx = torch.multinomial(preds,num_samples=1)
        idx_item = idx.item()
        sampled_tokens.append(idx_item)
        tokens = torch.cat((tokens,idx),dim=-1)
        if idx_item == tokenizer.eos_token_id:
            break
    print(f"Number of tokens sampled : {len(sampled_tokens)}")
    return tokenizer.decode(sampled_tokens)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used : {device}")
    model = GPT(config,device)
    seed_txt = "app.get"
    print(generate(model,seed_txt,device))