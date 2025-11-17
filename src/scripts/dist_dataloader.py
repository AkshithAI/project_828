from torch.utils.data import DataLoader
from .dataloader import get_train_files,get_hf_datasets,collate_fn,CustomDataset
from datasets import load_dataset

train_files = get_train_files()
ds_rank_0,ds_rank_1 = train_files[0::2],train_files[1::2]

ds_rank_0,ds_rank_1 = get_hf_datasets(ds_rank_0)["train"],get_hf_datasets(ds_rank_1)["train"]
ds_for_val = load_dataset("codeparrot/codeparrot-clean-valid",split = "train",streaming = True)

train_data_0 = CustomDataset(ds_rank_0)
train_data_1 = CustomDataset(ds_rank_1)
dataset_val = CustomDataset(ds_for_val)

train_loader_0 = DataLoader(
    train_data_0,
    batch_size=8,  # Matches train_micro_batch_size_per_gpu in ds-config.json
    collate_fn = collate_fn,
    pin_memory=True,
    num_workers=4,  # Increased for better I/O performance
    prefetch_factor=2,
    persistent_workers=True,
)

train_loader_1 = DataLoader(
    train_data_1,
    batch_size=8,  # Matches train_micro_batch_size_per_gpu in ds-config.json
    collate_fn = collate_fn,
    pin_memory=True,
    num_workers=4,  # Increased for better I/O performance
    prefetch_factor=2,
    persistent_workers=True,
)

val_data = DataLoader(
      dataset_val,
      batch_size = 8,  # Match training batch size for consistency
      collate_fn = collate_fn,
      pin_memory=True,
      num_workers=2,
      prefetch_factor=2,
  )