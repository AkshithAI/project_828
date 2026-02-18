import torch
import math
from datasets import load_dataset
from huggingface_hub import list_repo_files
from .tokenizer import tokenizer
from torch.utils.data import IterableDataset, DataLoader
from .configs.model_config import config, PhaseConfig, DatasetEntry
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════════
#  Format functions  — one per dataset layout
# ═══════════════════════════════════════════════════════════════

def _fmt_default(row: Dict[str, Any]) -> Optional[str]:
    """Most datasets: use the 'text' column."""
    text = row.get("text", "")
    return text if text else None


def _fmt_openmath(row: Dict[str, Any]) -> Optional[str]:
    """nvidia/OpenMathInstruct-2: problem + solution."""
    problem = row.get("problem", "")
    solution = row.get("generated_solution", "")
    if not problem and not solution:
        return None
    return f"{problem}\n\n{solution}"


def _fmt_fineweb_edu(row: Dict[str, Any]) -> Optional[str]:
    """HuggingFaceFW/fineweb-edu — only keep top-10% (score >= 3.0)."""
    score = row.get("score", 0.0)
    if score is None or score < 3.0:
        return None            
    return row.get("text", "") or None


def _fmt_starcoder(row: Dict[str, Any]) -> Optional[str]:
    """bigcode/the-stack-v2: use 'content' column."""
    content = row.get("content", "")
    return content if content else None


def _fmt_magicoder(row: Dict[str, Any]) -> Optional[str]:
    """ise-uiuc/Magicoder-OSS-Instruct-75K: problem + solution."""
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    if not problem and not solution:
        return None
    return f"{problem}\n\n{solution}"


def _fmt_stackexchange(row: Dict[str, Any]) -> Optional[str]:
    """HuggingFaceH4/stack-exchange-preferences: question + chosen answer."""
    question = row.get("question", "")
    chosen = row.get("chosen", "")
    if not question:
        return None
    return f"{question}\n\n{chosen}" if chosen else question


FORMAT_FNS: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {
    "default": _fmt_default,
    "openmath": _fmt_openmath,
    "fineweb_edu": _fmt_fineweb_edu,
    "starcoder": _fmt_starcoder,
    "magicoder": _fmt_magicoder,
    "stackexchange": _fmt_stackexchange,
}


# ═══════════════════════════════════════════════════════════════
#  State containers
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataLoaderState:
    """
    State container for resumable dataloader (single-dataset).
    
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


@dataclass
class DatasetStreamState:
    """Per-dataset state within a weighted mixer."""
    name: str
    documents_processed: int = 0      
    buffer_tokens: List[int] = field(default_factory=list)
    weight: int = 1                   

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetStreamState':
        return cls(**data)


@dataclass
class MixerState:
    """
    Full state of a WeightedMixerDataset — everything needed for exact resumption.

    Invariants kept across save / load:
        - ``dataset_states[name].documents_processed`` counts *all* raw HF rows
          seen from that stream, including rows filtered out by the format function.
          This makes ``.skip(n)`` always skip exactly *n* raw rows — O(1) per row.
        - ``draw_cycle_position`` is the index into the deterministic round-robin
          schedule, so the draw order is identical after resume.
        - ``samples_yielded`` counts total chunks yielded globally (for logging).
    """
    samples_yielded: int = 0
    batches_yielded: int = 0
    context_length: int = 2048
    batch_size: int = 128
    draw_cycle_position: int = 0
    dataset_states: Dict[str, DatasetStreamState] = field(default_factory=dict)
    version: int = 2                  

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "version": self.version,
            "samples_yielded": self.samples_yielded,
            "batches_yielded": self.batches_yielded,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "draw_cycle_position": self.draw_cycle_position,
            "dataset_states": {
                k: v.to_dict() for k, v in self.dataset_states.items()
            },
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MixerState':
        ds_states = {
            k: DatasetStreamState.from_dict(v)
            for k, v in data.get("dataset_states", {}).items()
        }
        return cls(
            samples_yielded=data.get("samples_yielded", 0),
            batches_yielded=data.get("batches_yielded", 0),
            context_length=data.get("context_length", 2048),
            batch_size=data.get("batch_size", 128),
            draw_cycle_position=data.get("draw_cycle_position", 0),
            dataset_states=ds_states,
            version=data.get("version", 2),
        )


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

    Works with both legacy ``ResumableDataset`` (single-dataset) and the new
    ``WeightedMixerDataset`` (multi-dataset).  The underlying dataset 
    exposes a ``.state`` attribute with a ``.to_dict()`` method and a 
    ``.batches_yielded`` counter.

    Resumption is handled at the dataset level: on resume, each HF stream is
    ``.skip()``-ed by its own ``documents_processed`` counter and token
    buffers are restored.  No batch-level skipping is needed.
    """

    def __init__(
        self,
        dataset,                      # ResumableDataset | WeightedMixerDataset
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

        self.dataset.state.batch_size = batch_size

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
        Works for both single-dataset and multi-dataset mixers.
        """
        return self.dataset.state.to_dict()

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
    
    skip_documents = 0
    buffer_tokens = []
    if train_state is not None:
        skip_documents = train_state.get('documents_processed', 0)
        buffer_tokens = train_state.get('buffer_tokens', [])
        print(f"[DataLoader] Resuming: skipping {skip_documents} documents")
    
    ds_for_train, ds_for_val = get_hf_datasets(train_files, skip_documents=skip_documents)
    
    train_dataset_state = DataLoaderState(
        context_length=context_length,
        batch_size=batch_size_train,
        buffer_tokens=buffer_tokens
    )
    if train_state is not None:
        train_dataset_state.samples_yielded = train_state.get('samples_yielded', 0)
        train_dataset_state.batches_yielded = train_state.get('batches_yielded', 0)
        train_dataset_state.documents_processed = train_state.get('documents_processed', 0)
    
    train_dataset = ResumableDataset(
        ds_for_train, 
        context_length=context_length,
        state=train_dataset_state
    )
    
    val_dataset = ResumableDataset(
        ds_for_val,
        context_length=context_length,
        state=DataLoaderState(context_length=context_length, batch_size=batch_size_val)
    )

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


# ═══════════════════════════════════════════════════════════════
#  Weighted multi-dataset mixer with exact resumption
# ═══════════════════════════════════════════════════════════════

class WeightedMixerDataset(IterableDataset):
    """
    Deterministic round-robin mixer over multiple HF streaming datasets.

    **Resumption guarantees** :

    1.  Draw order is purely deterministic — datasets are consumed in a fixed
        cycle derived from integer weights (e.g. [25,20,30,25] → cycle of 20
        draws repeating ``[0]*5 + [1]*4 + [2]*6 + [3]*5``).  After resume the
        cycle restarts from ``state.draw_cycle_position``.
    2.  Each dataset tracks its own ``documents_processed`` (raw HF rows seen,
        *including* rows rejected by the format function).  On resume the
        stream is ``.skip()``-ed by exactly that count.
    3.  Per-dataset token buffers are saved in full — no truncation.
    4.  ``samples_yielded`` is incremented and the buffer snapshot is taken
        *before* each ``yield``, so the state dict is always consistent even
        if the process is killed between yields.

    Args:
        dataset_entries: list of ``(name, hf_stream, weight, format_fn)`` tuples.
            *hf_stream* should already have ``.skip()`` applied if resuming.
        context_length: number of tokens per sample (chunk size is ctx+1).
        state: optional ``MixerState`` for resumption.
    """

    def __init__(
        self,
        dataset_entries: List[Tuple[str, Any, int, Callable]],
        context_length: int = 2048,
        state: Optional[MixerState] = None,
    ):
        super().__init__() 
        self.context_length = context_length 
        self.entries = dataset_entries          # [(name, stream, weight, fmt_fn), ...]

        if state is not None:
            self.state = state
        else:
            self.state = MixerState(
                context_length=context_length,
                dataset_states={
                    name: DatasetStreamState(name=name, weight=weight)
                    for name, _, weight, _ in dataset_entries
                },
            )

        weights = [w for _, _, w, _ in self.entries]
        g = weights[0]
        for w in weights[1:]:
            g = math.gcd(g, w)
        self._draw_schedule: List[int] = []     
        for idx, w in enumerate(weights):
            self._draw_schedule.extend([idx] * (w // g))
        self._cycle_len = len(self._draw_schedule)

    # ── helpers ───────────────────────────────────────────────

    def _drain_buffer(self, buf: List[int]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Extract as many complete ``(context_length + 1)`` chunks as possible
        from *buf*.  Returns ``(chunks, remaining_buffer)``.
        """
        chunks = []
        chunk_size = self.context_length + 1
        while len(buf) >= chunk_size:
            chunks.append(torch.tensor(buf[:chunk_size], dtype=torch.long))
            buf = buf[chunk_size:]
        return chunks, buf

    # ── main iterator ────────────────────────────────────────

    def __iter__(self):     
        iterators: List[Optional[Any]] = [iter(stream) for _, stream, _, _ in self.entries]
        exhausted: List[bool] = [False] * len(self.entries)

        buffers: List[List[int]] = []
        for name, _, _, _ in self.entries:
            ds_state = self.state.dataset_states.get(name)
            if ds_state is not None and ds_state.buffer_tokens:
                buffers.append(list(ds_state.buffer_tokens))
            else:
                buffers.append([])

        for ds_idx in range(len(self.entries)):
            name = self.entries[ds_idx][0]
            chunks, buffers[ds_idx] = self._drain_buffer(buffers[ds_idx])
            for chunk in chunks:
                self.state.dataset_states[name].buffer_tokens = buffers[ds_idx].copy()
                self.state.samples_yielded += 1
                yield chunk 

        pos = self.state.draw_cycle_position

        while True:
            if all(exhausted):
                break

            ds_idx = self._draw_schedule[pos % self._cycle_len]
            pos += 1
            self.state.draw_cycle_position = pos

            if exhausted[ds_idx]:
                continue

            name, _, _, fmt_fn = self.entries[ds_idx]
            it = iterators[ds_idx]

            try:
                row = next(it)
            except StopIteration:
                exhausted[ds_idx] = True
                print(f"[Mixer] Dataset '{name}' exhausted after "
                      f"{self.state.dataset_states[name].documents_processed} documents.")
                continue

            self.state.dataset_states[name].documents_processed += 1

            text = fmt_fn(row)
            if text is None:
                continue

            tokens = tokenizer( 
                text,
                return_attention_mask=False,
            )["input_ids"]  
            buffers[ds_idx].extend(tokens)
            buffers[ds_idx].append(tokenizer.eos_token_id)

            # Drain complete chunks
            chunks, buffers[ds_idx] = self._drain_buffer(buffers[ds_idx])
            for chunk in chunks:
                self.state.dataset_states[name].buffer_tokens = buffers[ds_idx].copy()
                self.state.samples_yielded += 1 
                yield chunk 
 
            self.state.dataset_states[name].buffer_tokens = buffers[ds_idx].copy()


# ═══════════════════════════════════════════════════════════════
#  Factory: build mixer from a PhaseConfig  (+ optional resume)
# ═══════════════════════════════════════════════════════════════

def load_phase_datasets(
    phase_config: PhaseConfig,
    mixer_state: Optional[Dict[str, Any]] = None,
    context_length: int = 2048,
) -> WeightedMixerDataset:
    """
    Build a ``WeightedMixerDataset`` for a training phase.

    On fresh start, all streams begin at document 0.
    On resume, each stream is ``.skip()``-ed by its saved
    ``documents_processed`` and the token buffer is restored.

    Args:
        phase_config:  ``PhaseConfig`` with ``.datasets`` populated.
        mixer_state:   Saved state dict (from ``ResumableDataLoader.get_state()``).
                       Pass ``None`` to start from scratch.
        context_length: Context length for chunking.

    Returns:
        A ``WeightedMixerDataset`` ready to iterate.
    """
    # Parse saved state if resuming
    restored_state: Optional[MixerState] = None
    if mixer_state is not None and mixer_state.get("version", 1) >= 2:
        restored_state = MixerState.from_dict(mixer_state)
        saved_names = set(restored_state.dataset_states.keys())
        config_names = {ds.name for ds in phase_config.datasets}
        if saved_names != config_names:
            raise ValueError(
                f"[DataLoader] Dataset name mismatch on resume!\n"
                f"  Saved:  {sorted(saved_names)}\n"
                f"  Config: {sorted(config_names)}\n"
                f"  Cannot resume — wrong phase config for this checkpoint."
            )
        if restored_state.context_length != context_length:
            raise ValueError(
                f"[DataLoader] Context length mismatch: "
                f"saved={restored_state.context_length}, requested={context_length}"
            )
        print(f"[DataLoader] Resuming mixer: {restored_state.samples_yielded} samples yielded, "
              f"cycle position {restored_state.draw_cycle_position}")
        for name, ds_state in restored_state.dataset_states.items():
            print(f"  {name}: {ds_state.documents_processed} docs, "
                  f"{len(ds_state.buffer_tokens)} buffered tokens")

    entries: List[Tuple[str, Any, int, Callable]] = []
    for ds_entry in phase_config.datasets:
        fmt_fn = FORMAT_FNS.get(ds_entry.format_fn)
        if fmt_fn is None:
            raise ValueError(
                f"Unknown format_fn={ds_entry.format_fn!r} for dataset {ds_entry.name!r}. "
                f"Available: {list(FORMAT_FNS.keys())}"
            )

        # Load HF streaming dataset
        kwargs = {}
        if ds_entry.data_dir is not None:
            # Bypass custom loading scripts — load raw files directly
            repo_files = list_repo_files(ds_entry.repo_id, repo_type="dataset")
            data_files = [
                f"https://huggingface.co/datasets/{ds_entry.repo_id}/resolve/main/{f}"
                for f in sorted(repo_files)
                if f.startswith(ds_entry.data_dir + "/") and not f.endswith(".md")
            ]
            if not data_files:
                raise ValueError(
                    f"No data files found under '{ds_entry.data_dir}/' in {ds_entry.repo_id}"
                )
            fmt = "parquet" if data_files[0].endswith(".parquet") else "json"
            stream = load_dataset(
                fmt,
                data_files=data_files,
                split=ds_entry.split,
                streaming=ds_entry.streaming,
            )
        else:
            if ds_entry.config_name is not None:
                kwargs["name"] = ds_entry.config_name
            kwargs["trust_remote_code"] = True
            stream = load_dataset(
                ds_entry.repo_id,
                split=ds_entry.split,
                streaming=ds_entry.streaming,
                **kwargs,
            )

        # Skip documents if resuming
        if restored_state is not None:
            skip_n = restored_state.dataset_states[ds_entry.name].documents_processed
            if skip_n > 0:
                print(f"[DataLoader] Skipping {skip_n} documents for '{ds_entry.name}'")
                stream = stream.skip(skip_n)

        entries.append((ds_entry.name, stream, ds_entry.weight, fmt_fn))

    return WeightedMixerDataset(
        dataset_entries=entries,
        context_length=context_length,
        state=restored_state,
    )


def create_phase_dataloaders(
    phase_config: PhaseConfig,
    train_state: Optional[Dict[str, Any]] = None,
    val_repo_id: str = "HuggingFaceFW/fineweb-edu",
    batch_size_val: int = 16,
    context_length: int = 2048,
) -> Tuple['ResumableDataLoader', DataLoader]:
    """
    Factory for phase-aware training: builds a ``WeightedMixerDataset`` for
    training and a simple streaming ``ResumableDataset`` for validation.

    Args:
        phase_config:   Phase configuration with datasets.
        train_state:    Saved mixer state dict (or None).
        val_repo_id:    HF repo for validation data.
        batch_size_val: Batch size for validation loader.
        context_length: Token context length.

    Returns:
        ``(train_loader, val_loader)``
    """

    # ── Train mixer ──
    mixer_dataset = load_phase_datasets(
        phase_config,
        mixer_state=train_state,
        context_length=context_length,
    )
    train_loader = ResumableDataLoader(
        mixer_dataset,
        batch_size=phase_config.micro_batch_size,
        pin_memory=True,
        num_workers=0,
    )   

    # ── Validation ──
    val_stream = load_dataset(
        val_repo_id,
        name="sample-100BT",
        split="train",
        streaming=True,
    )
    val_dataset = ResumableDataset(
        val_stream,
        context_length=context_length,
        state=DataLoaderState(context_length=context_length, batch_size=batch_size_val),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        collate_fn=lambda batch: torch.stack(batch, dim=0),
        pin_memory=True,
        num_workers=0,
    )
            
    return train_loader, val_loader