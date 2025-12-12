from torch.utils.data import DataLoader
from .dataloader import get_train_files,get_hf_datasets,collate_fn,CustomDataset
from datasets import load_dataset


def get_distributed_dataloader(rank: int, world_size: int, batch_size: int = 8):
    """
    Create a dataloader for a specific rank in distributed training.
    Shards data files across GPUs for efficient distributed data loading.
    
    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        batch_size: Batch size per GPU
    
    Returns:
        DataLoader for the specified rank
    """
    train_files = get_train_files()
    
    # Shard files across ranks
    rank_files = train_files[rank::world_size]
    
    if len(rank_files) == 0:
        raise ValueError(f"No files assigned to rank {rank}. Total files: {len(train_files)}, world_size: {world_size}")
    
    # get_hf_datasets returns (train, val) tuple - we only need train
    ds_rank, _ = get_hf_datasets(rank_files)
    train_data = CustomDataset(ds_rank)
    
    return DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )


# Legacy support for 2 GPU setup (deprecated, use get_distributed_dataloader instead)
train_files = get_train_files()
ds_rank_0, ds_rank_1 = train_files[0::2], train_files[1::2]

# get_hf_datasets returns (train, val) tuple - we only need train
ds_rank_0, _ = get_hf_datasets(ds_rank_0)
ds_rank_1, _ = get_hf_datasets(ds_rank_1)
ds_for_val = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)

train_data_0 = CustomDataset(ds_rank_0)
train_data_1 = CustomDataset(ds_rank_1)
dataset_val = CustomDataset(ds_for_val)

# Legacy hardcoded loaders (deprecated)
train_loader_0 = DataLoader(
    train_data_0,
    batch_size=8,  
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4,  
    prefetch_factor=2,
    persistent_workers=True,
)

train_loader_1 = DataLoader(
    train_data_1,
    batch_size=8,  
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4,  
    prefetch_factor=2,
    persistent_workers=True,
)

val_data = DataLoader(
    dataset_val,
    batch_size=8,  
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=2,
)
