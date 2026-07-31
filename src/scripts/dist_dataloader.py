from .dataloader import (
    DataLoaderState, DatasetStreamState, MixerState
)
from .dataloader import (
    resolve_dataset_files as _resolve_dataset_files_base,
    _compute_shard_skip, _count_shard_rows_fast,
    ShardedStream, ResumableDataset, ResumableDataLoader, PrefetchedDataLoader,
    ZeroStallDataLoader
)
from .dataloader import FORMAT_FNS
import math
import torch
from typing import Optional, Dict, Any, List, Callable, Tuple
from .configs.model_config import config, PhaseConfig, DatasetEntry
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from .tokenizer import tokenizer


def resolve_dataset_files(
    ds_entry: 'DatasetEntry',
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Distributed variant of :func:`resolve_dataset_files`.

    Resolves all data-file URLs via the base implementation, then
    partitions them across ranks using round-robin shard assignment
    (``urls[rank::world_size]``).

    If a rank ends up with zero shards (fewer shards than GPUs),
    returns ``(None, None)`` so the caller can fall back to native
    HF streaming (which handles partitioning at the row level).
    """
    # Use the base resolver to get the full shard list
    all_urls, fmt = _resolve_dataset_files_base(ds_entry)
    if all_urls is None:
        return None, None

    # Partition shards across ranks
    rank_urls = all_urls[rank::world_size]

    if not rank_urls:
        # Fewer shards than GPUs — this rank gets no shards.
        # Caller must fall back to native HF streaming.
        print(
            f"  [resolve] WARNING: Dataset '{ds_entry.name}' has {len(all_urls)} shards "
            f"but world_size={world_size}. Rank {rank} gets 0 shards — "
            f"falling back to native HF streaming."
        )
        return None, None

    return rank_urls, fmt


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
        rank: int = 0,
        world_size: int = 1,
        context_length: int = 2048,
        state: Optional[MixerState] = None,
        ds_configs: Optional[List] = None,
    ):
        super().__init__() 
        self.rank = rank
        self.world_size = world_size
        self.context_length = context_length 
        self.entries = dataset_entries          # [(name, stream, weight, fmt_fn), ...]

        # Store dataset configs for stream recreation on epoch restart
        self._ds_configs = ds_configs or [None] * len(dataset_entries)
        self._max_epochs = [
            (cfg.max_epochs if cfg is not None and hasattr(cfg, 'max_epochs') else 1)
            for cfg in self._ds_configs
        ]

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

    # ── stream recreation for epoch restart ───────────────────

    def _create_fresh_stream(self, ds_idx: int):
        """Recreate a dataset stream from scratch for a new epoch (no skip)."""
        name = self.entries[ds_idx][0]
        ds_state = self.state.dataset_states[name]
        ds_config = self._ds_configs[ds_idx]

        # ── Use stored data_files (fastest path) ──
        data_files = ds_state.data_files
        if data_files:
            fmt = "parquet" if data_files[0].endswith(".parquet") else "json"
            return ShardedStream(
                data_files=data_files,
                file_format=fmt,
                ds_state=ds_state,
                initial_skip=0,
                shard_idx_offset=0,
            )

        # ── Re-resolve from DatasetEntry config ──
        if ds_config is not None:
            file_urls, file_fmt = resolve_dataset_files(ds_config,self.rank,self.world_size)
            if file_urls:
                ds_state.data_files = file_urls
                return ShardedStream(
                    data_files=file_urls,
                    file_format=file_fmt,
                    ds_state=ds_state,
                )
            # Last resort: native HF streaming
            kwargs = {}
            if ds_config.data_dir is not None:
                kwargs["data_dir"] = ds_config.data_dir
            if ds_config.config_name is not None:
                kwargs["name"] = ds_config.config_name
            return load_dataset(
                ds_config.repo_id,
                split=ds_config.split,
                streaming=True,
                **kwargs,
            )

        return None

    # ── main iterator ────────────────────────────────────────

    def __iter__(self):     
        iterators: List[Optional[Any]] = [iter(stream) for _, stream, _, _ in self.entries]
        exhausted: List[bool] = [
            self.state.dataset_states[name].epochs_completed >= self._max_epochs[i]
            for i, (name, _, _, _) in enumerate(self.entries)
        ]
        if any(exhausted):
            already = [self.entries[i][0] for i, e in enumerate(exhausted) if e]
            print(f"[Mixer] Already exhausted on resume: {already}")

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
                ds_state = self.state.dataset_states[name]
                max_ep = self._max_epochs[ds_idx]

                if ds_state.epochs_completed + 1 < max_ep:
                    # ── Restart this dataset for another epoch ──
                    ds_state.epochs_completed += 1
                    ds_state.documents_processed = 0
                    ds_state.docs_per_shard = []
                    buffers[ds_idx] = []
                    ds_state.buffer_tokens = []

                    print(f"[Mixer] Dataset '{name}' completed epoch "
                          f"{ds_state.epochs_completed}/{max_ep}. Restarting...")

                    new_stream = self._create_fresh_stream(ds_idx)
                    if new_stream is not None:
                        self.entries[ds_idx] = (name, new_stream, self.entries[ds_idx][2], fmt_fn)
                        iterators[ds_idx] = iter(new_stream)
                    else:
                        exhausted[ds_idx] = True
                        print(f"[Mixer] Failed to restart '{name}'. Marking as exhausted.")
                else:
                    ds_state.epochs_completed += 1
                    exhausted[ds_idx] = True
                    print(f"[Mixer] Dataset '{name}' exhausted after "
                          f"{ds_state.epochs_completed} epoch(s) "
                          f"(max_epochs={max_ep}), "
                          f"{ds_state.documents_processed} documents in final epoch.")
                continue

            self.state.dataset_states[name].documents_processed += 1

            text = fmt_fn(row)
            if text is None:
                continue

            tokens = tokenizer.encode(text)  
            buffers[ds_idx].extend(tokens)
            buffers[ds_idx].append(tokenizer.eos_token_id)

            # Drain complete chunks
            chunks, buffers[ds_idx] = self._drain_buffer(buffers[ds_idx])
            for chunk in chunks:
                self.state.dataset_states[name].buffer_tokens = buffers[ds_idx].copy()
                self.state.samples_yielded += 1 
                yield chunk 
 
            self.state.dataset_states[name].buffer_tokens = buffers[ds_idx].copy()

    # ── text-only iterator for ZeroStallDataLoader ─────────────

    def __iter_texts__(self):
        """Yield raw text strings without tokenizing.

        Maintains the same deterministic round-robin draw order, format
        function application, epoch tracking, and document counting as
        :meth:`__iter__`.  Yields text strings instead of token chunks so
        that :class:`ZeroStallDataLoader` can batch-tokenize externally via
        gigatoken's GIL-releasing Rust encoder.
        """
        iterators: List[Optional[Any]] = [
            iter(stream) for _, stream, _, _ in self.entries
        ]
        exhausted: List[bool] = [
            self.state.dataset_states[name].epochs_completed >= self._max_epochs[i]
            for i, (name, _, _, _) in enumerate(self.entries)
        ]
        if any(exhausted):
            already = [self.entries[i][0] for i, e in enumerate(exhausted) if e]
            print(f"[Mixer] Already exhausted on resume: {already}")

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
                ds_state = self.state.dataset_states[name]
                max_ep = self._max_epochs[ds_idx]

                if ds_state.epochs_completed + 1 < max_ep:
                    ds_state.epochs_completed += 1
                    ds_state.documents_processed = 0
                    ds_state.docs_per_shard = []
                    ds_state.buffer_tokens = []

                    print(f"[Mixer] Dataset '{name}' completed epoch "
                          f"{ds_state.epochs_completed}/{max_ep}. Restarting...")

                    new_stream = self._create_fresh_stream(ds_idx)
                    if new_stream is not None:
                        self.entries[ds_idx] = (
                            name, new_stream, self.entries[ds_idx][2], fmt_fn,
                        )
                        iterators[ds_idx] = iter(new_stream)
                    else:
                        exhausted[ds_idx] = True
                        print(f"[Mixer] Failed to restart '{name}'. "
                              f"Marking as exhausted.")
                else:
                    ds_state.epochs_completed += 1
                    exhausted[ds_idx] = True
                    print(f"[Mixer] Dataset '{name}' exhausted after "
                          f"{ds_state.epochs_completed} epoch(s) "
                          f"(max_epochs={max_ep}), "
                          f"{ds_state.documents_processed} documents "
                          f"in final epoch.")
                continue

            self.state.dataset_states[name].documents_processed += 1

            text = fmt_fn(row)
            if text is None:
                continue

            yield text


def load_phase_datasets(
    phase_config: PhaseConfig,
    mixer_state: Optional[Dict[str, Any]] = None,
    rank: int = 0,
    world_size: int = 1,
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

            stream = ShardedStream(
                data_files=remaining_files,
                file_format=fmt,
                ds_state=ds_state_ref,
                initial_skip=shard_offset,
                shard_idx_offset=start_shard,
            )

            ds_state_ref.data_files = data_files
            used_shard_path = True
            _elapsed = _time.perf_counter() - _t0
            print(f"[DataLoader]   Stream ready in {_elapsed:.2f}s")

        if not used_shard_path:
            # ── Resolve files and use ShardedStream ──────────
            file_urls, file_fmt = resolve_dataset_files(ds_entry, rank, world_size)

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

                    stream = ShardedStream(
                        data_files=file_urls,
                        file_format=file_fmt,
                        ds_state=ds_state_ref,
                        initial_skip=0,
                        total_skip=skip_n,
                        shard_idx_offset=0,
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
        rank=rank,
        world_size=world_size,
        context_length=context_length,
        state=restored_state,
        ds_configs=list(phase_config.datasets),
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
    rank: int = 0,
    world_size: int = 1,
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
        rank=rank,
        world_size=world_size,
        context_length=context_length,
    )
    train_loader = ZeroStallDataLoader(
        mixer_dataset,
        batch_size=phase_config.micro_batch_size,
        num_prefetch=16,
        tokenize_chunk_size=512,
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