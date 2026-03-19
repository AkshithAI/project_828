import torch
import math
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download
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

    data_files: List[str] = field(default_factory=list)       
    docs_per_shard: List[int] = field(default_factory=list)   

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

    def to_dict(self) -> Dict[str, Any]:
        d = {
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
        )


# ═══════════════════════════════════════════════════════════════
#  Shard-aware streaming for fast resume
# ═══════════════════════════════════════════════════════════════

class ShardedStream:
    """
    Iterates through data files (shards) one at a time, tracking
    ``current_shard_idx`` and updating per-shard document counts
    in a :class:`DatasetStreamState`.

    On resume, entire shards can be skipped by simply not including
    them in *data_files*, converting an O(total_docs) skip into
    O(offset_within_current_shard).

    Two skip modes are supported:

    * **initial_skip** — skip *N* rows within the **first** shard only.
      Used on shard-aware resume (fast).
    * **total_skip** — skip *N* rows across **all** shards sequentially.
      Used on legacy resume when ``docs_per_shard`` is not yet populated
      (same speed as before, but builds shard tracking for future resumes).

    Supports all formats that HuggingFace ``datasets`` can stream:
    parquet, json, and jsonl.zst (auto-detected via ``fsspec``).
    For unreliable network connections (e.g. large ``.jsonl.zst`` shards),
    per-shard retry logic re-opens the HTTP stream and skips to the
    failure point — no local caching required.

    Args:
        data_files:        Ordered list of data-file URLs.
        file_format:       Format string for ``load_dataset``
                           (``"parquet"`` or ``"json"``). The HF datasets
                           library auto-detects ``.zst`` compression.
        ds_state:          Optional state object for shard tracking.
        initial_skip:      Rows to skip within the **first** shard only.
        total_skip:        Rows to skip across all shards sequentially.
        shard_idx_offset:  Global shard index offset (for shard-aware resume).
        max_retries:       Max retries per shard on network / decompression errors.
        base_backoff:      Base backoff in seconds (capped exponential, max 600s).
        on_shard_complete: Optional callback ``(global_idx, url) -> None``
                           invoked after a shard is fully iterated.
    """

    def __init__(
        self,
        data_files: List[str],
        file_format: str = "parquet",
        ds_state: Optional[DatasetStreamState] = None,
        initial_skip: int = 0,
        total_skip: int = 0,
        shard_idx_offset: int = 0,
        max_retries: int = 8,
        base_backoff: int = 30,
        on_shard_complete: Optional[Callable[[int, str], None]] = None,
    ):
        assert not (initial_skip > 0 and total_skip > 0), (
            "ShardedStream: initial_skip and total_skip are mutually exclusive. "
            f"Got initial_skip={initial_skip}, total_skip={total_skip}."
        )
        self.data_files = data_files
        self.all_data_files = list(data_files)   # full copy for state storage
        self.file_format = file_format
        self.current_shard_idx = shard_idx_offset
        self._shard_idx_offset = shard_idx_offset
        self._ds_state = ds_state
        self._initial_skip = initial_skip
        self._total_skip = total_skip
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._on_shard_complete = on_shard_complete

    def _open_shard(self, url: str, skip: int = 0):
        """
        Open a single shard as a streaming HF dataset, optionally
        skipping *skip* rows from the start.
        """
        ds = load_dataset(
            self.file_format,
            data_files=[url],
            split="train",
            streaming=True,
        )
        if skip > 0:
            ds = ds.skip(skip)
        return ds

    def __iter__(self):
        import time as _time_mod

        remaining_skip = self._total_skip
        for local_idx, url in enumerate(self.data_files):
            global_idx = self._shard_idx_offset + local_idx
            self.current_shard_idx = global_idx

            # How many rows to skip at the start of this shard
            shard_initial_skip = self._initial_skip if local_idx == 0 else 0

            # Track rows yielded from *this* shard for retry recovery
            rows_yielded_this_shard = 0
            # Snapshot docs_per_shard count at shard start for retry reset
            dps_at_shard_start = 0
            if self._ds_state is not None:
                dps = self._ds_state.docs_per_shard
                while len(dps) <= global_idx:
                    dps.append(0)
                dps_at_shard_start = dps[global_idx]

            attempt = 0
            while True:  # retry loop
                try:
                    total_skip_for_open = shard_initial_skip + rows_yielded_this_shard
                    shard_ds = self._open_shard(url, skip=total_skip_for_open)

                    for row in shard_ds:
                        # Update docs_per_shard tracking
                        if self._ds_state is not None:
                            self._ds_state.docs_per_shard[global_idx] = (
                                dps_at_shard_start + rows_yielded_this_shard + 1
                            )

                        # Legacy cross-shard skip (slow, builds shard tracking
                        # for future resumes so subsequent loads are instant)
                        if remaining_skip > 0:
                            remaining_skip -= 1
                            rows_yielded_this_shard += 1
                            skipped = self._total_skip - remaining_skip
                            if skipped % 200_000 == 0:
                                print(f"[ShardedStream] Skip progress: "
                                      f"{skipped:,}/{self._total_skip:,} docs "
                                      f"(shard {global_idx})")
                            continue

                        rows_yielded_this_shard += 1
                        yield row

                    # Shard fully consumed — invoke callback and break retry loop
                    if self._on_shard_complete is not None:
                        try:
                            self._on_shard_complete(global_idx, url)
                        except Exception as cb_exc:
                            print(f"[ShardedStream] on_shard_complete callback "
                                  f"error for shard {global_idx}: {cb_exc!r}")
                    break  # success — move to next shard

                except KeyboardInterrupt:
                    raise
                except StopIteration:
                    break  # shard exhausted normally
                except Exception as exc:
                    attempt += 1
                    if attempt >= self._max_retries:
                        print(f"[ShardedStream] FATAL: shard {global_idx} failed "
                              f"after {self._max_retries} retries: {exc!r}")
                        print(f"[ShardedStream]   URL: {url}")
                        print(f"[ShardedStream]   Rows yielded before failure: "
                              f"{rows_yielded_this_shard}")
                        raise
                    wait = min(
                        self._base_backoff * (2 ** (attempt - 1)), 600
                    )
                    print(f"[ShardedStream] Error in shard {global_idx}: {exc!r}")
                    print(f"[ShardedStream]   Retrying in {wait}s "
                          f"(attempt {attempt}/{self._max_retries}, "
                          f"will skip {rows_yielded_this_shard} already-yielded rows)")
                    _time_mod.sleep(wait)


class LocalZstStream:
    """
    **LEGACY FALLBACK** — Shard-aware streaming for ``.jsonl.zst`` files
    via local download + caching.

    .. deprecated::
        Prefer :class:`ShardedStream` with ``file_format="json"`` which
        streams ``.jsonl.zst`` files over HTTP without downloading them.
        ``ShardedStream`` now includes per-shard retry logic for
        reliability.  Use ``LocalZstStream`` only if your network is
        too unstable for streaming (set ``use_local_download=True``
        on the ``DatasetEntry``).

    Each shard is downloaded to the HuggingFace cache directory with
    ``hf_hub_download()`` **before** any rows are read.  This gives us:

    * **Built-in retry + resume** — ``hf_hub_download`` uses HTTP Range
      headers to resume interrupted downloads automatically.
    * **Disk caching** — once downloaded, the shard is never re-fetched.
    * **Zero network errors during reading** — decompression and JSON
      parsing happen entirely on local disk.

    **Warning:** This approach consumes significant disk space as training
    progresses through shards, since downloaded files are never cleaned up
    automatically.  For proof-pile-2 (arxiv: ~88 shards, open-web-math:
    ~64 shards), this can grow to tens of GB.

    The API mirrors :class:`ShardedStream` so the rest of the pipeline
    (``load_phase_datasets``, ``WeightedMixerDataset``) works unchanged.
    """

    def __init__(
        self,
        repo_id: str,
        data_files: List[str],
        ds_state: Optional[DatasetStreamState] = None,
        initial_skip: int = 0,
        total_skip: int = 0,
        shard_idx_offset: int = 0,
    ):
        assert not (initial_skip > 0 and total_skip > 0), (
            "LocalZstStream: initial_skip and total_skip are mutually exclusive. "
            f"Got initial_skip={initial_skip}, total_skip={total_skip}."
        )
        self.repo_id = repo_id
        self.data_files = data_files
        self.all_data_files = list(data_files)
        self.current_shard_idx = shard_idx_offset
        self._shard_idx_offset = shard_idx_offset
        self._ds_state = ds_state
        self._initial_skip = initial_skip
        self._total_skip = total_skip

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_filename(file_ref: str) -> str:
        """Extract repo-relative path from an HF resolve URL (or pass through)."""
        marker = "/resolve/main/"
        idx = file_ref.find(marker)
        return file_ref[idx + len(marker):] if idx != -1 else file_ref

    def _download_shard(self, file_ref: str) -> str:
        """Download a shard to local cache with aggressive retry."""
        import time as _time_mod

        filename = self._extract_filename(file_ref)
        max_retries = 10
        base_backoff = 30  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                return hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                wait = min(base_backoff * (2 ** (attempt - 1)), 600)
                print(f"[LocalZstStream] Download error for {filename}: {exc!r}")
                print(f"[LocalZstStream] Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})")
                _time_mod.sleep(wait)

    @staticmethod
    def _iter_local_zst(local_path: str):
        """Yield parsed dicts from a local ``.jsonl.zst`` file."""
        import io
        import json as _json
        import zstandard as zstd

        with open(local_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if line:
                        yield _json.loads(line)

    # ── main iterator ────────────────────────────────────────

    def __iter__(self):
        remaining_skip = self._total_skip

        for local_idx, file_ref in enumerate(self.data_files):
            global_idx = self._shard_idx_offset + local_idx
            self.current_shard_idx = global_idx

            fname = self._extract_filename(file_ref)
            print(f"[LocalZstStream] Caching shard {global_idx}: {fname}")
            local_path = self._download_shard(file_ref)
            print(f"[LocalZstStream] Reading from cache: {local_path}")

            rows_in_shard = 0
            for row in self._iter_local_zst(local_path):
                rows_in_shard += 1

                if local_idx == 0 and rows_in_shard <= self._initial_skip:
                    continue

                if self._ds_state is not None:
                    dps = self._ds_state.docs_per_shard
                    while len(dps) <= global_idx:
                        dps.append(0)
                    dps[global_idx] += 1

                if remaining_skip > 0:
                    remaining_skip -= 1
                    skipped = self._total_skip - remaining_skip
                    if skipped % 200_000 == 0:
                        print(f"[LocalZstStream] Skip progress: "
                              f"{skipped:,}/{self._total_skip:,} docs "
                              f"(shard {global_idx})")
                    continue

                yield row


def resolve_dataset_files(
    ds_entry: 'DatasetEntry',
) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Resolve a :class:`DatasetEntry` to an ordered list of data-file
    URLs and a format string (``"parquet"`` or ``"json"``).

    ``.jsonl.zst`` and ``.json.zst`` files are returned with format
    ``"json"`` — the HuggingFace ``datasets`` library auto-detects
    zstandard compression and streams them transparently over HTTP.

    For datasets with ``data_dir`` set, the resolver checks for a
    ``{data_dir}/{split}/`` subdirectory first (e.g. ``arxiv/train/``),
    falling back to ``{data_dir}/`` if no split subdirectory exists.

    Falls back to ``(None, None)`` when the repo layout cannot be
    determined automatically.
    """
    try:
        all_files = sorted(list_repo_files(ds_entry.repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"  [resolve] Could not list files for {ds_entry.repo_id}: {exc}")
        return None, None

    matched: List[str] = []

    _DATA_EXTS = (".parquet", ".json", ".jsonl", ".jsonl.zst", ".json.zst")

    if ds_entry.data_dir is not None:
        split = ds_entry.split or "train"

        prefix_with_split = f"{ds_entry.data_dir}/{split}/"
        matched = [
            f for f in all_files
            if f.startswith(prefix_with_split)
            and f.endswith(_DATA_EXTS)
        ]

        if not matched:
            matched = [
                f for f in all_files
                if f.startswith(ds_entry.data_dir + "/")
                and not f.endswith((".md", ".gitattributes"))
                and f.endswith(_DATA_EXTS)
            ]
    else:
        split = ds_entry.split or "train"
        # 1. {config_name}/*{split}*.parquet
        if ds_entry.config_name:
            matched = [
                f for f in all_files
                if ds_entry.config_name in f and split in f
                and f.endswith(_DATA_EXTS)
            ]
        # 2. data/{split}-*.parquet
        if not matched:
            matched = [
                f for f in all_files
                if f.startswith("data/") and split in f
                and f.endswith(_DATA_EXTS)
            ]
        # 3. {split}-*.parquet at root
        if not matched:
            matched = [
                f for f in all_files
                if f.startswith(f"{split}-")
                and f.endswith(_DATA_EXTS)
            ]

    if not matched:
        return None, None

    urls = [
        f"https://huggingface.co/datasets/{ds_entry.repo_id}/resolve/main/{f}"
        for f in sorted(matched)
    ]
    # HF datasets auto-detects .zst compression when format="json",
    # so .jsonl.zst files are streamed over HTTP without local download.
    fmt = "parquet" if urls[0].endswith(".parquet") else "json"
    return urls, fmt


def _compute_shard_skip(
    docs_per_shard: List[int],
    documents_processed: int,
) -> Tuple[int, int]:
    """
    Given per-shard document counts and total documents processed,
    return ``(start_shard_idx, offset_within_shard)``.
    """
    cumulative = 0
    for i, count in enumerate(docs_per_shard):
        cumulative += count
        if cumulative >= documents_processed:
            offset = documents_processed - (cumulative - count)
            return i, offset

    return len(docs_per_shard), 0


def _count_shard_rows_fast(file_urls: List[str]) -> Optional[List[int]]:
    """
    Read per-shard row counts from parquet metadata without downloading
    the full data.  Uses pyarrow to read only the parquet footer (~few KB
    per file) via HTTP range requests.

    Returns ``None`` if reading fails for any shard (caller should fall
    back to the slow row-by-row skip).
    """
    try:
        import pyarrow.parquet as pq
        import fsspec
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _count_one(url: str) -> int:
            with fsspec.open(url, "rb") as f:
                return pq.ParquetFile(f).metadata.num_rows

        counts = [0] * len(file_urls)
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(_count_one, url): i
                       for i, url in enumerate(file_urls)}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    counts[idx] = future.result()
                except Exception:
                    return None          # any failure → abort
                done += 1
                if done % 200 == 0 or done == len(file_urls):
                    print(f"\r[DataLoader]   Counting shard rows: {done}/{len(file_urls)}",
                          end="", flush=True)
        if file_urls:
            print()
        return counts
    except ImportError:
        return None
    except Exception as e:
        print(f"\n[DataLoader] Fast shard row counting failed: {e}")
        return None


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
            tokens = tokenizer(
                doc['text'],
                return_attention_mask=False
            )["input_ids"]
            
            buffer.extend(tokens)
            buffer.append(tokenizer.eos_token_id)
            
            self.state.documents_processed += 1
            
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
#  Cache cleanup for previously-downloaded .zst shards
# ═══════════════════════════════════════════════════════════════

def _cleanup_cached_shard(repo_id: str, file_url: str) -> None:
    """
    Delete a previously-downloaded ``.jsonl.zst`` shard from the local
    HuggingFace cache (``~/.cache/huggingface/hub/``).

    This is a best-effort operation — if the file was never cached, or
    deletion fails, a warning is printed but no exception is raised.

    Only targets ``.jsonl.zst`` / ``.json.zst`` files to avoid
    accidentally deleting parquet shards (which are never cached).
    """
    import os

    if not file_url.endswith((".jsonl.zst", ".json.zst")):
        return  # only clean up zst files

    # Extract repo-relative filename from HF resolve URL
    marker = "/resolve/main/"
    idx = file_url.find(marker)
    filename = file_url[idx + len(marker):] if idx != -1 else file_url

    try:
        # hf_hub_download returns the cache path — we can use it to find
        # the file.  But calling it would re-download if not cached.
        # Instead, construct the expected cache path directly.
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo_id and repo_info.repo_type == "dataset":
                for revision in repo_info.revisions:
                    for cached_file in revision.files:
                        if cached_file.file_path.name.endswith(
                            os.path.basename(filename)
                        ) and filename.replace("/", os.sep) in str(
                            cached_file.file_path
                        ):
                            size_mb = cached_file.size_on_disk / (1024 * 1024)
                            cached_file.file_path.unlink(missing_ok=True)
                            print(
                                f"[CacheCleanup] Deleted cached shard: "
                                f"{filename} ({size_mb:.1f} MB freed)"
                            )
                            return
        # File not in cache — nothing to do
    except Exception as exc:
        print(f"[CacheCleanup] Warning: could not clean up {filename}: {exc!r}")


def _make_zst_cleanup_callback(
    repo_id: str,
    all_data_files: List[str],
) -> Optional[Callable[[int, str], None]]:
    """
    Create a shard-complete callback that cleans up cached ``.jsonl.zst``
    files from prior ``LocalZstStream`` runs.

    Returns ``None`` if no zst files are present (parquet-only datasets),
    avoiding the overhead of scanning the cache on every shard completion.
    """
    has_zst = any(
        f.endswith((".jsonl.zst", ".json.zst")) for f in all_data_files
    )
    if not has_zst:
        return None

    def _callback(global_idx: int, url: str) -> None:
        _cleanup_cached_shard(repo_id, url)

    return _callback


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

    On fresh start every dataset is resolved to explicit data-file URLs
    and wrapped in a :class:`ShardedStream` that tracks per-shard document
    counts.  On resume the saved ``docs_per_shard`` is used to skip
    entire shard files instantly — only the offset within the *current*
    shard requires iteration, reducing resume time from O(total_docs) to
    O(docs_within_one_shard).

    Falls back to the legacy ``stream.skip(N)`` path when the checkpoint
    was saved before shard tracking was available, or when file-URL
    resolution fails for a dataset.

    Args:
        phase_config:  ``PhaseConfig`` with ``.datasets`` populated.
        mixer_state:   Saved state dict (from ``ResumableDataLoader.get_state()``).
                       Pass ``None`` to start from scratch.
        context_length: Context length for chunking.

    Returns:
        A ``WeightedMixerDataset`` ready to iterate.
    """
    import time as _time

    # ── Parse saved state ────────────────────────────────────
    restored_state: Optional[MixerState] = None
    if mixer_state is not None:
        restored_state = MixerState.from_dict(mixer_state)
        cfg_by_name = {ds.name: ds for ds in phase_config.datasets}

        # Guard against accidental duplicate names in config.
        if len(cfg_by_name) != len(phase_config.datasets):
            names = [ds.name for ds in phase_config.datasets]
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"[DataLoader] Duplicate dataset names in phase config: {dupes}"
            )

        saved_names = set(restored_state.dataset_states.keys())
        config_names = set(cfg_by_name.keys())

        removed = sorted(saved_names - config_names)
        added = sorted(config_names - saved_names)

        if removed or added:
            print("[DataLoader] Dataset set changed since checkpoint; "
                  "migrating mixer state to current config.")
            if removed:
                print(f"  Removed datasets: {removed}")
            if added:
                print(f"  Added datasets:   {added}")

            migrated_states: Dict[str, DatasetStreamState] = {}
            for ds in phase_config.datasets:
                prev = restored_state.dataset_states.get(ds.name)
                if prev is None:
                    migrated_states[ds.name] = DatasetStreamState(
                        name=ds.name,
                        weight=ds.weight,
                    )
                else:
                    prev.weight = ds.weight
                    migrated_states[ds.name] = prev

            restored_state.dataset_states = migrated_states

            restored_state.draw_cycle_position = 0
        else:
            for ds in phase_config.datasets:
                restored_state.dataset_states[ds.name].weight = ds.weight

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

    # ── Build entries ────────────────────────────────────────
    entries: List[Tuple[str, Any, int, Callable]] = []
    for ds_entry in phase_config.datasets:
        fmt_fn = FORMAT_FNS.get(ds_entry.format_fn)
        if fmt_fn is None:
            raise ValueError(
                f"Unknown format_fn={ds_entry.format_fn!r} for dataset {ds_entry.name!r}. "
                f"Available: {list(FORMAT_FNS.keys())}"
            )

        ds_state_ref: Optional[DatasetStreamState] = None
        if restored_state is not None:
            ds_state_ref = restored_state.dataset_states.get(ds_entry.name)

        skip_n = ds_state_ref.documents_processed if ds_state_ref else 0

        # ── Try shard-aware path ─────────────────────────────
        used_shard_path = False

        if (
            ds_state_ref is not None
            and ds_state_ref.docs_per_shard
            and ds_state_ref.data_files
            and skip_n > 0
        ):

            data_files = ds_state_ref.data_files
            docs_per_shard = ds_state_ref.docs_per_shard

            # HF datasets auto-detects .zst compression via format="json"
            if data_files[0].endswith(".parquet"):
                fmt = "parquet"
            else:
                fmt = "json"

            start_shard, shard_offset = _compute_shard_skip(docs_per_shard, skip_n)

            remaining_files = data_files[start_shard:]
            print(
                f"[DataLoader] Shard-aware skip for '{ds_entry.name}': "
                f"skipping {start_shard} full shards, "
                f"offset {shard_offset} in shard {start_shard} "
                f"({len(remaining_files)} shards remaining)"
            )
            _t0 = _time.perf_counter()

            # Opt-in local download for extremely unstable networks
            if getattr(ds_entry, 'use_local_download', False) and fmt == "json":
                stream = LocalZstStream(
                    repo_id=ds_entry.repo_id,
                    data_files=remaining_files,
                    ds_state=ds_state_ref,
                    initial_skip=shard_offset,
                    shard_idx_offset=start_shard,
                )
            else:
                # Build cleanup callback for previously-cached zst shards
                _cleanup_cb = _make_zst_cleanup_callback(
                    ds_entry.repo_id, data_files
                )
                stream = ShardedStream(
                    data_files=remaining_files,
                    file_format=fmt,
                    ds_state=ds_state_ref,
                    initial_skip=shard_offset,
                    shard_idx_offset=start_shard,
                    on_shard_complete=_cleanup_cb,
                )

            ds_state_ref.data_files = data_files
            used_shard_path = True
            _elapsed = _time.perf_counter() - _t0
            print(f"[DataLoader]   Stream ready in {_elapsed:.2f}s")

        if not used_shard_path:
            # ── Resolve files and use ShardedStream ──────────
            file_urls, file_fmt = resolve_dataset_files(ds_entry)

            if file_urls is not None:
                _t0 = _time.perf_counter()

                # Try fast metadata-based shard skip (reads parquet footers only)
                if skip_n > 0 and file_fmt == "parquet":
                    print(f"[DataLoader] Counting rows in {len(file_urls)} shards "
                          f"for '{ds_entry.name}' (parquet metadata)...")
                    shard_row_counts = _count_shard_rows_fast(file_urls)
                    if shard_row_counts is not None:
                        start_shard, shard_offset = _compute_shard_skip(
                            shard_row_counts, skip_n
                        )
                        remaining_files = file_urls[start_shard:]
                        print(
                            f"[DataLoader] Metadata skip for '{ds_entry.name}': "
                            f"skipping {start_shard} full shards, "
                            f"offset {shard_offset} in shard {start_shard} "
                            f"({len(remaining_files)} shards remaining)"
                        )
                        stream = ShardedStream(
                            data_files=remaining_files,
                            file_format=file_fmt,
                            ds_state=ds_state_ref,
                            initial_skip=shard_offset,
                            shard_idx_offset=start_shard,
                        )
                        if ds_state_ref is not None:
                            ds_state_ref.data_files = file_urls
                            # Pre-populate docs_per_shard for skipped shards
                            # + partial count for the current shard (initial_skip rows)
                            ds_state_ref.docs_per_shard = (
                                list(shard_row_counts[:start_shard]) + [shard_offset]
                            )
                        used_shard_path = True
                        _elapsed = _time.perf_counter() - _t0
                        print(f"[DataLoader]   Stream ready in {_elapsed:.2f}s")

                # Fall back to naive skip with progress logging
                if not used_shard_path:
                    if skip_n > 0 and ds_state_ref is not None:
                        print(
                            f"[DataLoader] No shard info for '{ds_entry.name}' — "
                            f"using naive skip ({skip_n:,} docs). "
                            f"Shard tracking will be saved in next checkpoint."
                        )

                    # Opt-in local download fallback for unstable networks
                    if (
                        getattr(ds_entry, 'use_local_download', False)
                        and file_fmt == "json"
                        and file_urls[0].endswith((".jsonl.zst", ".json.zst"))
                    ):
                        stream = LocalZstStream(
                            repo_id=ds_entry.repo_id,
                            data_files=file_urls,
                            ds_state=ds_state_ref,
                            initial_skip=0,
                            total_skip=skip_n,
                            shard_idx_offset=0,
                        )
                    else:
                        _cleanup_cb = _make_zst_cleanup_callback(
                            ds_entry.repo_id, file_urls
                        )
                        stream = ShardedStream(
                            data_files=file_urls,
                            file_format=file_fmt,
                            ds_state=ds_state_ref,
                            initial_skip=0,
                            total_skip=skip_n,
                            shard_idx_offset=0,
                            on_shard_complete=_cleanup_cb,
                        )
                    if ds_state_ref is not None:
                        ds_state_ref.data_files = file_urls
                    _elapsed = _time.perf_counter() - _t0
                    print(f"[DataLoader] '{ds_entry.name}': resolved {len(file_urls)} "
                          f"data files in {_elapsed:.2f}s")
                    used_shard_path = True

            else:
                # ── Fallback: native HF loading ─────────────
                kwargs = {}
                if ds_entry.data_dir is not None:
                    kwargs["data_dir"] = ds_entry.data_dir
                if ds_entry.config_name is not None:
                    kwargs["name"] = ds_entry.config_name

                stream = load_dataset(
                    ds_entry.repo_id,
                    split=ds_entry.split,
                    streaming=ds_entry.streaming,
                    **kwargs,
                )

                # Legacy naive skip (only if not already using ShardedStream)
                if not used_shard_path and skip_n > 0:
                    print(f"[DataLoader] Skipping {skip_n} documents for "
                          f"'{ds_entry.name}' (legacy)")
                    stream = stream.skip(skip_n)

        entries.append((ds_entry.name, stream, ds_entry.weight, fmt_fn))

    # ── Build mixer ──
    mixer = WeightedMixerDataset(
        dataset_entries=entries,
        context_length=context_length,
        state=restored_state,
    )

    if restored_state is None:
        for name, stream, _, _ in entries:
            ds_state = mixer.state.dataset_states.get(name)
            if ds_state is not None and hasattr(stream, 'all_data_files'):
                ds_state.data_files = list(stream.all_data_files)
                if hasattr(stream, '_ds_state'):
                    stream._ds_state = ds_state

    return mixer


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