from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from .tokenizer import tokenizer
from .dataloader import CustomDataset,collate_fn

# Load TinyStories
dataset = load_dataset("roneneldan/TinyStories")
train_data = dataset["train"]
val_data = dataset["validation"]

dataset_train = CustomDataset(train_data)
dataset_val = CustomDataset(val_data)

train_data = DataLoader(
      dataset_train,
      batch_size = 8,
      shuffle = True,
      collate_fn = collate_fn,
      pin_memory=True,
      num_workers=0,
)
val_data = DataLoader(
      dataset_val,
      batch_size = 8,
      shuffle = True,
      collate_fn = collate_fn,
      pin_memory=True,
      num_workers=0,
)