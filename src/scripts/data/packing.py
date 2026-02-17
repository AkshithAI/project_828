"""
Best-fit bin packing pipeline for HuggingFace datasets.

Stages
------
  tokenize  — download + tokenize HF dataset shards into parquet chunks
  pack      — best-fit bin packing on pre-computed document lengths
  upload    — materialize packed bins and push to HF Hub
  all       — run all three stages sequentially

Usage examples
--------------
  python -m project_828.src.scripts.data.packing tokenize \\
         --repo_id codeparrot/codeparrot-clean --max_seq_len 2048

  python -m project_828.src.scripts.data.packing pack --max_seq_len 2048

  python -m project_828.src.scripts.data.packing upload \\
         --repo_id AkshithAI/packed-data

  python -m project_828.src.scripts.data.packing all \\
         --repo_id codeparrot/codeparrot-clean \\
         --upload_repo_id AkshithAI/packed-data --max_seq_len 2048
"""

import os
import gc
import glob
import logging
import argparse
import numpy as np
import polars as pl
from tqdm import tqdm
from numba import njit
from collections import defaultdict
from datasets import load_dataset, Dataset
from huggingface_hub import list_repo_files, hf_hub_download, HfApi
from ..tokenizer import tokenizer

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Tokenization
# ───────────────────────────────────────────────────────────────

def get_train_filenames(repo_id):
    """Retrieves list of parquet / json files from a HF repo."""
    all_files = list_repo_files(repo_id, repo_type="dataset")
    files = sorted([f for f in all_files if f.endswith('.parquet') or f.endswith('.json.gz')])
    return files


def process_and_save_shard(
    file_path, tokenizer, output_dir, shard_idx,
    max_seq_len, text_column="content", num_proc=2,
):
    """Tokenise a single shard with overflow splitting and save as parquet."""
    if file_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=file_path, split="train")
    else:
        ds = load_dataset("json", data_files=file_path, split="train")

    def tokenize_and_split(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_len,
            return_overflowing_tokens=True,
            return_length=False,
            return_attention_mask=False,
        )

    tokenized_ds = ds.map(
        tokenize_and_split,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )

    df = pl.from_arrow(tokenized_ds.data.table)
    df = df.with_row_index("row_idx")
    df = df.with_columns(
        pl.col("input_ids").list.len().alias("doc_len").cast(pl.Int32)
    )

    save_path = os.path.join(output_dir, f"chunk_{shard_idx}.parquet")
    df.select(["input_ids"]).write_parquet(save_path)

    return df.select(["doc_len"]).to_series().to_numpy(), df.shape[0]


def orchestrate_tokenization(
    repo_id, tokenizer, output_dir,
    max_seq_len, text_column="content", num_proc=2,
):
    """Download every shard, tokenise, and save chunk parquets + metadata."""
    os.makedirs(output_dir, exist_ok=True)

    files = get_train_filenames(repo_id)
    logger.info("Found %d files in %s", len(files), repo_id)

    global_doc_lengths = []
    total_docs = 0

    for i, filename in enumerate(tqdm(files, desc="Processing Files")):
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir="temp_downloads",
            local_dir_use_symlinks=False,
        )

        lengths, num_rows = process_and_save_shard(
            local_path, tokenizer, output_dir, i,
            max_seq_len=max_seq_len,
            text_column=text_column,
            num_proc=num_proc,
        )
        global_doc_lengths.append(lengths)
        total_docs += num_rows

        os.remove(local_path)

    all_lengths = np.concatenate(global_doc_lengths)
    all_indices = np.arange(total_docs, dtype=np.int64)

    logger.info("Saving metadata for %d documents...", total_docs)
    np.save("doc_lengths.npy", all_lengths.astype(np.int32))
    np.save("doc_indices.npy", all_indices)

    chunk_sizes = np.array([len(l) for l in global_doc_lengths], dtype=np.int64)
    np.save("chunk_sizes.npy", chunk_sizes)
    logger.info("Tokenization complete.")


# ───────────────────────────────────────────────────────────────
# Segment-tree accelerated best-fit bin packing
# ───────────────────────────────────────────────────────────────

@njit
def update_tree(tree, node, start, end, idx, val):
    if start == end:
        tree[node] = val
        return
    mid = (start + end) // 2
    if idx <= mid:
        update_tree(tree, 2 * node + 1, start, mid, idx, val)
    else:
        update_tree(tree, 2 * node + 2, mid + 1, end, idx, val)
    tree[node] = max(tree[2 * node + 1], tree[2 * node + 2])


@njit
def query_tree(tree, node, start, end, doc_size):
    if tree[node] < doc_size:
        return -1
    if start == end:
        return start
    mid = (start + end) // 2
    res = -1
    if tree[2 * node + 1] >= doc_size:
        res = query_tree(tree, 2 * node + 1, start, mid, doc_size)
    if res != -1:
        return res
    return query_tree(tree, 2 * node + 2, mid + 1, end, doc_size)


class BestFitPacking:
    def __init__(self, max_seq_len, lengths, indices):
        self.max_seq_len = max_seq_len
        self.lengths = lengths
        self.indices = indices

        self.no_of_bins = 0
        self.n_docs = len(lengths)
        self.space_to_bins = defaultdict(list)

        # Pre-allocate output arrays
        self.out_bin_ids = np.zeros(self.n_docs, dtype=np.int64)
        self.out_doc_indices = np.zeros(self.n_docs, dtype=np.int64)

        self.tree_size = 4 * (max_seq_len + 1)
        self.tree = np.zeros(self.tree_size, dtype=np.int64)

    def pack(self):
        skipped = 0
        for ind in tqdm(range(self.n_docs), desc="Packing"):
            doc_len = self.lengths[ind]
            original_idx = self.indices[ind]

            if doc_len > self.max_seq_len:
                self.out_bin_ids[ind] = -1
                self.out_doc_indices[ind] = original_idx
                skipped += 1
                continue

            best_capacity = query_tree(self.tree, 0, 0, self.max_seq_len, doc_len)

            if best_capacity != -1:
                bin_id = self.space_to_bins[best_capacity].pop()

                self.out_bin_ids[ind] = bin_id
                self.out_doc_indices[ind] = original_idx

                new_space = best_capacity - doc_len
                self.space_to_bins[new_space].append(bin_id)

                if not self.space_to_bins[best_capacity]:
                    update_tree(self.tree, 0, 0, self.max_seq_len, best_capacity, 0)

                if len(self.space_to_bins[new_space]) == 1:
                    update_tree(self.tree, 0, 0, self.max_seq_len, new_space, new_space)

            else:
                new_bin_id = self.no_of_bins
                self.out_bin_ids[ind] = new_bin_id
                self.out_doc_indices[ind] = original_idx

                remaining_space = self.max_seq_len - doc_len
                self.space_to_bins[remaining_space].append(new_bin_id)

                if len(self.space_to_bins[remaining_space]) == 1:
                    update_tree(self.tree, 0, 0, self.max_seq_len, remaining_space, remaining_space)
                self.no_of_bins += 1

        return self.out_bin_ids, self.out_doc_indices, skipped


def run_packing(max_seq_len):
    """Load doc lengths, sort descending, run best-fit packing, save map."""
    lengths = np.load("doc_lengths.npy")
    indices = np.load("doc_indices.npy")

    sorted_idx = np.argsort(-lengths)
    lengths, indices = lengths[sorted_idx], indices[sorted_idx]

    packer = BestFitPacking(max_seq_len=max_seq_len, lengths=lengths, indices=indices)
    bin_ids_res, doc_ids_res, skipped = packer.pack()

    # ── Packing statistics ──
    valid_mask = bin_ids_res != -1
    total_docs = len(lengths)
    packed_docs = int(valid_mask.sum())
    total_bins = packer.no_of_bins
    total_packed_tokens = int(lengths[valid_mask].sum())
    total_capacity = total_bins * max_seq_len
    utilization = (total_packed_tokens / total_capacity * 100) if total_capacity > 0 else 0.0

    logger.info("─── Packing Statistics ───")
    logger.info("  Total documents:      %s", f"{total_docs:,}")
    logger.info("  Packed documents:     %s", f"{packed_docs:,}")
    logger.info("  Skipped (too long):   %s", f"{skipped:,}")
    logger.info("  Total bins:           %s", f"{total_bins:,}")
    logger.info("  Bin capacity:         %s tokens", f"{max_seq_len:,}")
    logger.info("  Total packed tokens:  %s", f"{total_packed_tokens:,}")
    logger.info("  Total bin capacity:   %s", f"{total_capacity:,}")
    logger.info("  Avg utilization:      %.2f%%", utilization)
    logger.info("──────────────────────────")

    # Save packing map
    df_map = pl.DataFrame({
        "original_idx": doc_ids_res,
        "bin_id": bin_ids_res,
    })
    df_map = df_map.filter(pl.col("bin_id") != -1)
    df_map = df_map.with_row_index("seq_order")
    df_map.write_parquet("packing_map.parquet")
    logger.info("Packing map saved (%s rows).", f"{df_map.shape[0]:,}")


# ───────────────────────────────────────────────────────────────
# Materialise + Upload
# ───────────────────────────────────────────────────────────────

def materialize_and_upload(repo_id, data_dir="./tokenized_data"):
    """Join packing map with tokenized chunks, stream packed bins to HF Hub."""
    logger.info("1. Indexing chunk files...")

    chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.parquet")))

    if os.path.exists("chunk_sizes.npy"):
        chunk_sizes = np.load("chunk_sizes.npy")
    else:
        chunk_sizes = [pl.scan_parquet(f).select(pl.len()).collect().item() for f in chunk_files]

    start_offsets = np.concatenate(([0], np.cumsum(chunk_sizes)[:-1]))

    lazy_frames = []
    for f, offset in zip(chunk_files, start_offsets):
        lf = pl.scan_parquet(f).select(
            pl.col("input_ids").cast(pl.List(pl.Int32)),
        )
        lf = lf.with_row_index("row_idx", offset=offset).rename({"row_idx": "original_idx"})
        lazy_frames.append(lf)

    lf_data = pl.concat(lazy_frames).with_columns(pl.col("original_idx").cast(pl.Int64))
    lf_map = pl.scan_parquet("packing_map.parquet")

    logger.info("2. Joining and sorting (lazy)...")
    final_lf = lf_map.join(lf_data, on="original_idx", how="left")
    final_lf = final_lf.sort(["bin_id", "seq_order"])

    temp_file = "sorted_full_dataset.parquet"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    logger.info("3. Materializing sorted dataset to temp file...")
    final_lf.sink_parquet(temp_file)

    del final_lf, lf_data, lazy_frames
    gc.collect()

    logger.info("4. Streaming to HuggingFace...")

    def upload_gen():
        """Yield packed bins, carrying forward rows at batch boundaries
        to prevent splitting a bin across two group_by calls."""
        lf_sorted = pl.scan_parquet(temp_file)
        total_rows = lf_sorted.select(pl.len()).collect().item()
        batch_size = 20_000
        carry = None  # rows whose bin_id may span the next batch

        for i in range(0, total_rows, batch_size):
            is_last = (i + batch_size >= total_rows)
            df_chunk = lf_sorted.slice(i, batch_size).collect()

            # Prepend carry-forward rows from previous batch
            if carry is not None:
                df_chunk = pl.concat([carry, df_chunk])
                carry = None

            if not is_last:
                # The last bin_id in this sorted slice may continue in the
                # next batch — hold those rows back until we can confirm.
                last_bin_id = df_chunk["bin_id"][-1]
                carry = df_chunk.filter(pl.col("bin_id") == last_bin_id)
                df_chunk = df_chunk.filter(pl.col("bin_id") != last_bin_id)

            if df_chunk.height == 0:
                continue

            packed_df = df_chunk.group_by("bin_id", maintain_order=True).agg(
                pl.col("input_ids").flatten()
            )
            for row in packed_df.iter_rows(named=True):
                yield {"input_ids": row["input_ids"]}

            del df_chunk, packed_df
            gc.collect()

        # Flush remaining carry
        if carry is not None and carry.height > 0:
            packed_df = carry.group_by("bin_id", maintain_order=True).agg(
                pl.col("input_ids").flatten()
            )
            for row in packed_df.iter_rows(named=True):
                yield {"input_ids": row["input_ids"]}
            del carry, packed_df
            gc.collect()

    ds = Dataset.from_generator(upload_gen)
    ds.push_to_hub(repo_id, max_shard_size="500MB", private=False)
    logger.info("Upload complete!")


# ───────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Best-fit bin packing pipeline for HF datasets.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── shared parent for max_seq_len ──
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--max_seq_len", type=int, default=2048,
        help="Maximum sequence length — used as both tokenization "
             "truncation length and bin capacity (default: 2048).",
    )

    # ── tokenize ──
    p_tok = sub.add_parser(
        "tokenize", parents=[common],
        help="Download and tokenize a HF dataset into parquet chunks.",
    )
    p_tok.add_argument("--repo_id", required=True,
                       help="HF dataset repo id (source).")
    p_tok.add_argument("--output_dir", default="./tokenized_data",
                       help="Directory for tokenized chunks (default: ./tokenized_data).")
    p_tok.add_argument("--text_column", default="content",
                       help="Name of the text column in the dataset (default: content).")
    p_tok.add_argument("--num_proc", type=int, default=2,
                       help="Parallel workers for tokenization map (default: 2).")

    # ── pack ──
    sub.add_parser(
        "pack", parents=[common],
        help="Run best-fit bin packing on pre-computed doc lengths.",
    )

    # ── upload ──
    p_up = sub.add_parser("upload",
                          help="Materialize packed bins and push to HF Hub.")
    p_up.add_argument("--repo_id", required=True,
                      help="HF dataset repo id (destination).")
    p_up.add_argument("--data_dir", default="./tokenized_data",
                      help="Directory containing chunk parquets (default: ./tokenized_data).")

    # ── all ──
    p_all = sub.add_parser(
        "all", parents=[common],
        help="Run tokenize → pack → upload end-to-end.",
    )
    p_all.add_argument("--repo_id", required=True,
                       help="HF dataset repo id (source).")
    p_all.add_argument("--upload_repo_id", required=True,
                       help="HF dataset repo id (destination for packed data).")
    p_all.add_argument("--output_dir", default="./tokenized_data",
                       help="Directory for tokenized chunks (default: ./tokenized_data).")
    p_all.add_argument("--text_column", default="content",
                       help="Name of the text column in the dataset (default: content).")
    p_all.add_argument("--num_proc", type=int, default=2,
                       help="Parallel workers for tokenization map (default: 2).")

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "tokenize":
        orchestrate_tokenization(
            repo_id=args.repo_id,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            text_column=args.text_column,
            num_proc=args.num_proc,
        )

    elif args.command == "pack":
        run_packing(max_seq_len=args.max_seq_len)

    elif args.command == "upload":
        materialize_and_upload(repo_id=args.repo_id, data_dir=args.data_dir)

    elif args.command == "all":
        orchestrate_tokenization(
            repo_id=args.repo_id,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
            text_column=args.text_column,
            num_proc=args.num_proc,
        )
        run_packing(max_seq_len=args.max_seq_len)
        materialize_and_upload(
            repo_id=args.upload_repo_id,
            data_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()