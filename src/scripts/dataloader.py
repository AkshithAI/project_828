import torch
from datasets import load_dataset
from huggingface_hub import list_repo_files
from .tokenizer import tokenizer

repo_id = "codeparrot/codeparrot-clean"   # your TRAIN repo
branch  = "main"                           # or a specific commit SHA for reproducibility
train_shards = sorted(list_repo_files(repo_id, repo_type="dataset"))
selected = train_shards[3:]  # skip the first shard
print(len(selected))

# Turn repo-relative paths into stable resolve URLs
def to_url(path):
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{path}"

train_urls = [to_url(p) for p in selected]

# Build the streaming dataset using the JSON loader
# (It handles .gz automatically; assumes JSON Lines per file.)
ds_dict = load_dataset(
    "json",
    data_files={"train": train_urls},
    split=None,              # we provided our own split mapping
    streaming=True,
)

ds_for_train = ds_dict["train"]
ds_for_val = load_dataset("codeparrot/codeparrot-clean-valid",split = "train",streaming = True)

def prepare_code_data(files, context_length=2048):
    buffer = []
    for i,file in enumerate(files):
        tokens = tokenizer(
                file['content'],
                return_attention_mask = False
                )["input_ids"]
        buffer.extend(tokens)
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= context_length:
            chunk = torch.tensor(buffer[:context_length],dtype = torch.long)
            yield chunk
            buffer = buffer[context_length:]
 
def collate_fn(batch):
  return torch.stack(batch,dim = 0)

from torch.utils.data import IterableDataset,DataLoader
class CustomDataset(IterableDataset):
  def __init__(self,data,context_length = 2048):
    super().__init__()
    self.data = data
    self.context_length = context_length
    
  def __iter__(self):
    yield from prepare_code_data(self.data,self.context_length)
    
dataset_train = CustomDataset(ds_for_train)
dataset_val = CustomDataset(ds_for_val)
train_data = DataLoader(dataset_train,batch_size = 8,collate_fn = collate_fn,pin_memory=True)
val_data = DataLoader(dataset_val,batch_size = 8,collate_fn = collate_fn,pin_memory=True)