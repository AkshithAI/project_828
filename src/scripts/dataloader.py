import torch
from datasets import load_dataset
from huggingface_hub import list_repo_files
from .tokenizer import tokenizer
from torch.utils.data import IterableDataset,DataLoader

# def get_train_files():
#   repo_id = "codeparrot/codeparrot-clean"   
#   branch  = "main"                          
#   train_shards = sorted(list_repo_files(repo_id, repo_type="dataset"))

#   def to_url(path):
#       return f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{path}"

#   train_urls = [to_url(p) for p in train_shards]
#   return train_urls
def get_train_files(lang="en", include_variants=False):
    """
    Return URLs for files that live in the top-level `lang` folder only (e.g. "en/...").
    If include_variants=True, also include top-level folders like "en.noclean" and "en.noblocklist".
    """
    repo_id = "allenai/c4"
    branch = "main"

    # get all file paths from the repo
    all_shards = sorted(list_repo_files(repo_id, repo_type="dataset"))

    # helper: top-level folder name (text before first '/')
    def top_segment(p):
        return p.split("/", 1)[0] if "/" in p else p

    if include_variants:
        allowed = {lang, f"{lang}.noclean", f"{lang}.noblocklist"}
        train_shards = [p for p in all_shards if top_segment(p) in allowed]
    else:
        # strict: first path segment must exactly equal the language (no "multilingual/en/..." etc.)
        train_shards = [p for p in all_shards if top_segment(p) == lang]

    # dedupe while preserving order (defensive)
    seen = set()
    uniq_shards = []
    for p in train_shards:
        if p not in seen:
            seen.add(p)
            uniq_shards.append(p)
    train_shards = uniq_shards

    if not train_shards:
        # helpful debug info
        top_level = sorted({top_segment(p) for p in all_shards})
        raise RuntimeError(f"No files found for language '{lang}'. Top-level entries: {top_level}")

    def to_url(path):
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{path}"

    train_urls = [to_url(p) for p in train_shards]
    return train_urls

def get_hf_datasets(train_files):
  ds_dict = load_dataset(
      "json",
      data_files={"train": train_files},
      split=None,              
      streaming=True,
  )

  ds_for_train = ds_dict["train"]
  ds_for_val = en = load_dataset("allenai/c4", "en",split = "validation", streaming=True)
  return ds_for_train,ds_for_val

def prepare_code_data(files, context_length=2048):
    buffer = []
    for i,file in enumerate(files):
        tokens = tokenizer(
                file['content'],
                return_attention_mask = False
                )["input_ids"]
        buffer.extend(tokens)
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= context_length + 1:
            chunk = torch.tensor(buffer[:context_length + 1],dtype = torch.long)
            yield chunk
            buffer = buffer[context_length + 1:]
 
def collate_fn(batch):
  return torch.stack(batch,dim = 0)

class CustomDataset(IterableDataset):
  def __init__(self,data,context_length = 2048):
    super().__init__()
    self.data = data
    self.context_length = context_length
    
  def __iter__(self):
    yield from prepare_code_data(self.data,self.context_length)

train_files = get_train_files()[0]
ds_for_train,ds_for_val = get_hf_datasets(train_files)
dataset_train = CustomDataset(ds_for_train)
dataset_val = CustomDataset(ds_for_val)
train_data = DataLoader(
      dataset_train,
      batch_size = 8,
      collate_fn = collate_fn,
      pin_memory=True,
      num_workers=0,
)
val_data = DataLoader(
      dataset_val,
      batch_size = 8,
      collate_fn = collate_fn,
      pin_memory=True,
      num_workers=0,
)

