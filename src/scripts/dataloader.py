import torch
from datasets import load_dataset
from huggingface_hub import list_repo_files
from .tokenizer import tokenizer
from torch.utils.data import IterableDataset, DataLoader
from .configs.model_config import config
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class DataLoaderState:
    """
    State container for resumable dataloader.
    
    Attributes:
        samples_yielded: Total number of complete samples (batches * batch_size) yielded
        batches_yielded: Total number of batches yielded from the dataloader
        documents_processed: Number of documents fully processed from the stream
        buffer_tokens: Leftover tokens in the buffer (not yet formed into a sample)
        batch_size: Batch size used
        context_length: Context length for samples
    """
    samples_yielded: int = 0
    batches_yielded: int = 0
    documents_processed: int = 0
    buffer_tokens: List[int] = field(default_factory=list)
    batch_size: int = 4
    context_length: int = 2048
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataLoaderState':
        """Create state from dictionary."""
        return cls(**data)


def get_train_files(repo_id):
    branch = "main"

    # get all file paths from the repo
    all_shards = sorted(list_repo_files(repo_id, repo_type="dataset"))

    def to_url(path):
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/{path}"

    train_urls = [to_url(p) for p in all_shards]
    return train_urls[2:]  # Exclude readme and .gitattributes file


def get_hf_datasets(train_files, skip_documents: int = 0):
    """
    Load dataset directly using the datasets library with streaming. 
    Uses all shards except last 4 for training, and last 4 for validation.
    
    Args:
        train_files: List of file URLs
        skip_documents: Number of documents to skip for resumption (only for training)
    """
    train_urls = train_files[:-4]
    val_urls = train_files[-4:]
    
    # Detect file format from first file extension
    file_format = 'parquet' if train_urls[0].endswith('.parquet') else 'json'
    
    ds_for_train = load_dataset(
        file_format,
        data_files=train_urls,
        split='train',
        streaming=True
    )
    
    # Skip documents if resuming
    if skip_documents > 0:
        ds_for_train = ds_for_train.skip(skip_documents)
    
    ds_for_val = load_dataset(
        file_format,
        data_files=val_urls,
        split='train',
        streaming=True
    )
    
    return ds_for_train, ds_for_val


class ResumableDataset(IterableDataset):
    """
    An IterableDataset that supports state tracking for resumption.
    
    This dataset processes streaming documents by:
    1. Tokenizing each document
    2. Concatenating tokens with EOS separators  
    3. Chunking into fixed-length samples
    
    State is tracked externally via the state object for checkpointing.
    """
    
    def __init__(
        self, 
        data, 
        context_length: int = 2048,
        state: Optional[DataLoaderState] = None
    ):
        super().__init__()
        self.data = data
        self.context_length = context_length
        # External state reference - this allows the dataloader wrapper to track state
        self.state = state if state is not None else DataLoaderState(context_length=context_length)
        
    def _prepare_data_with_state(self):
        """
        Generator that yields chunks while tracking state.
        
        The state tracks:
        - documents_processed: how many raw documents we've consumed
        - buffer_tokens: leftover tokens that haven't formed a complete sample yet
        """
        # Initialize buffer from saved state (for mid-document resumption)
        buffer = list(self.state.buffer_tokens) if self.state.buffer_tokens else []
        
        for doc in self.data:
            # Tokenize document
            tokens = tokenizer(
                doc['text'],
                return_attention_mask=False
            )["input_ids"]
            
            # Add tokens and EOS to buffer
            buffer.extend(tokens)
            buffer.append(tokenizer.eos_token_id)
            
            # Increment document counter AFTER processing
            self.state.documents_processed += 1
            
            # Yield complete chunks
            while len(buffer) >= self.context_length + 1:
                chunk = torch.tensor(buffer[:self.context_length + 1], dtype=torch.long)
                buffer = buffer[self.context_length + 1:]
                
                # Update buffer state before yielding (in case of interruption)
                self.state.buffer_tokens = buffer.copy()
                self.state.samples_yielded += 1
                
                yield chunk
        
        # Clear buffer state at end of epoch
        self.state.buffer_tokens = []
    
    def __iter__(self):
        yield from self._prepare_data_with_state()


class ResumableDataLoader:
    """
    A wrapper around DataLoader that provides state management for resumption.
    
    This class:
    1. Wraps the underlying DataLoader
    2. Tracks batch-level progress
    3. Provides save/load state functionality
    
    Resumption is handled at the dataset level: on resume, the HuggingFace stream
    skips already-processed documents, and the token buffer is restored. This gives
    exact sample-level resumption without needing batch-level skipping.
    
    Usage:
        # Use the factory function for full resumption support:
        train_loader, val_loader = create_resumable_dataloaders(
            repo_id="...",
            train_state=saved_state,  # from a previous train_loader.get_state()
        )
        
        for batch in train_loader:
            ...
            # Save periodically:
            state = train_loader.get_state()
            torch.save(state, "dataloader_state.pt")
    """
    
    def __init__(
        self,
        dataset: ResumableDataset,
        batch_size: int = 4,
        pin_memory: bool = True,
        num_workers: int = 0,
        collate_fn=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else self._default_collate
        
        # Update state with batch size
        self.dataset.state.batch_size = batch_size
        
        # Create underlying dataloader
        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers
        )
    
    @staticmethod
    def _default_collate(batch):
        return torch.stack(batch, dim=0)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state for checkpointing.
        
        Returns a dictionary containing all information needed to resume.
        """
        return self.dataset.state.to_dict()
    
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state counters from a checkpoint for tracking purposes.
        
        WARNING: This only restores internal counters. For streaming datasets,
        the dataset must be recreated with the correct skip_documents parameter
        to actually resume from the right position in the stream.
        Use create_resumable_dataloaders() for full resumption.
        
        Args:
            state_dict: State dictionary from get_state()
        """
        self.dataset.state = DataLoaderState.from_dict(state_dict)
        print(f"[DataLoader] Loaded state: {self.dataset.state.batches_yielded} batches, "
              f"{self.dataset.state.documents_processed} documents processed")
    
    def __iter__(self):
        """
        Iterate over batches. Resumption is handled at the dataset level via
        document skipping and buffer restoration — no batch-level skipping needed.
        """
        for batch in self._dataloader:
            self.dataset.state.batches_yielded += 1
            yield batch


def create_resumable_dataloaders(
    repo_id: str = "karpathy/fineweb-edu-100b-shuffle",
    train_state: Optional[Dict[str, Any]] = None,
    batch_size_train: int = 4,
    batch_size_val: int = 16,
    context_length: int = 2048
):
    """
    Factory function to create resumable train and validation dataloaders.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        train_state: Optional saved state to resume from
        batch_size_train: Batch size for training
        batch_size_val: Batch size for validation
        context_length: Context length for samples
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    train_files = get_train_files(repo_id)
    
    # Determine how many documents to skip for resumption
    skip_documents = 0
    buffer_tokens = []
    if train_state is not None:
        skip_documents = train_state.get('documents_processed', 0)
        buffer_tokens = train_state.get('buffer_tokens', [])
        print(f"[DataLoader] Resuming: skipping {skip_documents} documents")
    
    # Load datasets with skip for training
    ds_for_train, ds_for_val = get_hf_datasets(train_files, skip_documents=skip_documents)
    
    # Create train dataset with state
    train_dataset_state = DataLoaderState(
        context_length=context_length,
        batch_size=batch_size_train,
        buffer_tokens=buffer_tokens
    )
    if train_state is not None:
        # Restore full state - keep documents_processed at saved value for cumulative tracking
        # The dataset skip handles not re-processing, but we track total for next save
        train_dataset_state.samples_yielded = train_state.get('samples_yielded', 0)
        train_dataset_state.batches_yielded = train_state.get('batches_yielded', 0)
        train_dataset_state.documents_processed = train_state.get('documents_processed', 0)
    
    train_dataset = ResumableDataset(
        ds_for_train, 
        context_length=context_length,
        state=train_dataset_state
    )
    
    # Create validation dataset (no state needed - always from start)
    val_dataset = ResumableDataset(
        ds_for_val,
        context_length=context_length,
        state=DataLoaderState(context_length=context_length, batch_size=batch_size_val)
    )
    
    # Create dataloaders
    train_loader = ResumableDataLoader(
        train_dataset,
        batch_size=batch_size_train,
        pin_memory=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        collate_fn=lambda batch: torch.stack(batch, dim=0),
        pin_memory=True,
        num_workers=0
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    return torch.stack(batch, dim=0)


# Legacy dataset class for backward compatibility
class CustomDataset(IterableDataset):
    def __init__(self, data, context_length=2048):
        super().__init__()
        self.data = data
        self.context_length = context_length
    
    def __iter__(self):
        buffer = []
        for doc in self.data:
            tokens = tokenizer(
                doc['text'],
                return_attention_mask=False
            )["input_ids"]
            buffer.extend(tokens)
            buffer.append(tokenizer.eos_token_id)
            while len(buffer) >= self.context_length + 1:
                chunk = torch.tensor(buffer[:self.context_length + 1], dtype=torch.long)
                buffer = buffer[self.context_length + 1:]
                yield chunk

# Legacy usage (guarded to prevent execution on import):
if __name__ == '__main__':
    repo_id = "karpathy/fineweb-edu-100b-shuffle"
    train_files = get_train_files(repo_id)
    ds_for_train, ds_for_val = get_hf_datasets(train_files)
    dataset_train = CustomDataset(ds_for_train)
    dataset_val = CustomDataset(ds_for_val)
    train_data = DataLoader(
          dataset_train,
          batch_size = 4,
          collate_fn = collate_fn,
          pin_memory=True,
          num_workers=0,
    )
    val_data = DataLoader(
          dataset_val,
          batch_size = 16,
          collate_fn = collate_fn,
          pin_memory=True,
          num_workers=0,
    )        

