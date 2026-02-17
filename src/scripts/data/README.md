# Data Processing Pipeline

This directory contains two data processing systems:

1. **NeMo Curator Preprocessing** (`preprocess.py`) — Quality filtering, text cleaning, and deduplication using [NVIDIA NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/index.html)
2. **Best-Fit Bin Packing** (`packing.py`) — Tokenization with overflow splitting + segment-tree accelerated document packing + HuggingFace Hub upload

---

# NeMo Curator Data Preprocessing Pipeline

A configurable data preprocessing pipeline for LLM training using [NVIDIA NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/index.html).

## Overview

This module provides end-to-end data curation capabilities:

- **Custom Dataset Curation** - Process datasets from HuggingFace Hub
- **Predefined Pipelines** - ArXiv, Common Crawl, Wikipedia
- **Quality Filtering** - Word count, punctuation, boilerplate removal
- **Text Cleaning** - Unicode normalization, URL removal, newline standardization
- **Deduplication** - Exact, fuzzy (MinHash), and semantic deduplication

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure credentials (required for Common Crawl)
# Edit .env in project root with your AWS credentials
```

### 2. Configure Pipeline

Edit `configs/preprocess.json` to enable/disable processing stages:

```json
{
    "WordCountFilter": {
        "enabled": true,
        "params": { "min_words": 50, "max_words": 100000 }
    },
    "UnicodeReformatter": true,
    "ExactDeduplicationWorkflow": { "enabled": false }
}
```

### 3. Run Pipeline

```bash
# Custom HuggingFace dataset
python preprocess.py --custom --repo_id username/dataset-name

# With deduplication
python preprocess.py --custom --dedup --repo_id username/dataset-name

# Predefined datasets
python preprocess.py --data_tag arxiv --download_dir ./data
python preprocess.py --data_tag wiki --download_dir ./data
python preprocess.py --data_tag common_crawl --download_dir ./data  # Requires AWS credentials
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--custom` | Use custom HuggingFace dataset | `True` |
| `--repo_id` | HuggingFace repository ID | `karpathy/fineweb-edu-100b-shuffle` |
| `--dedup` | Enable deduplication after curation | `False` |
| `--data_tag` | Predefined pipeline: `arxiv`, `wiki`, `common_crawl` | - |
| `--download_dir` | Download directory for predefined datasets | `./downloads` |
| `--config` | Path to preprocessing config | `configs/preprocess.json` |
| `--num_files` | Number of files to process | `100` |
| `--log_level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--log_file` | Optional log file path | - |

## Configuration Reference

### Filters

| Filter | Description | Parameters |
|--------|-------------|------------|
| `add_id` | Add unique document IDs | `id_field`, `id_prefix`, `overwrite` |
| `WordCountFilter` | Filter by word count | `min_words`, `max_words` |
| `NonAlphaNumericFilter` | Remove symbol-heavy content | `max_non_alpha_numeric_to_text_ratio` |
| `RepeatedLinesFilter` | Remove repetitive content | `max_repeated_line_fraction` |
| `PunctuationFilter` | Ensure proper punctuation | `max_num_sentences_without_endmark_ratio` |
| `BoilerPlateStringFilter` | Remove boilerplate text | - |

### Modifiers

| Modifier | Description |
|----------|-------------|
| `UnicodeReformatter` | Normalize Unicode characters |
| `NewlineNormalizer` | Standardize line breaks |
| `UrlRemover` | Remove URLs from text |

### Deduplication

| Workflow | Description | Requirements |
|----------|-------------|--------------|
| `ExactDeduplicationWorkflow` | Remove identical documents (MD5 hashing) | Ray, GPU |
| `FuzzyDeduplicationWorkflow` | Remove near-duplicates (MinHash + LSH) | Ray, GPU |
| `TextSemanticDeduplicationWorkflow` | Remove semantically similar content | Ray, GPU, sentence-transformers |

## Environment Variables

Configure in `.env` (project root):

```bash
# Required for Common Crawl (S3 access)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# Optional: HuggingFace token for private datasets
HF_TOKEN=your_token
```

## File Format Support

The pipeline automatically detects file formats:
- **Parquet** (`.parquet`)
- **JSONL** (`.jsonl`)

No manual configuration needed - format is detected from downloaded files.

## Output

Curated data is saved to `nemo_curator/data/curated/` in the same format as input.

## Logging

All operations are logged with timestamps. Enable debug mode for detailed output:

```bash
python preprocess.py --custom --log_level DEBUG --log_file ./logs/preprocess.log
```

---

# Best-Fit Bin Packing Pipeline

A high-performance document packing pipeline (`packing.py`) that tokenizes HuggingFace datasets, packs variable-length documents into fixed-capacity bins using a **segment-tree accelerated best-fit** algorithm, and uploads the packed result to HuggingFace Hub.

## Overview

The pipeline runs in three stages:

1. **Tokenize** — Download dataset shards from HuggingFace, tokenize with overflow splitting (`return_overflowing_tokens=True`), and save as parquet chunks.
2. **Pack** — Sort documents by length (descending), then assign each to the tightest-fitting open bin using a numba-accelerated segment tree. Complexity: $O(N \log L)$ where $N$ = documents and $L$ = bin capacity.
3. **Upload** — Join the packing map with tokenized chunks, materialize sorted bins, and stream to HuggingFace Hub.

### Key Design Decisions

- **Single `--max_seq_len`** parameter controls both tokenization truncation length and bin capacity, ensuring consistency.
- **Overflow splitting** is intentional — documents longer than `max_seq_len` are split into multiple segments at the tokenization stage, then each segment is packed independently.
- **Carry-forward buffer** in the upload stage prevents bins from being split across batch boundaries during streaming materialization.

## Quick Start

```bash
# Run the full pipeline end-to-end
python -m project_828.src.scripts.data.packing all \
    --repo_id codeparrot/codeparrot-clean \
    --upload_repo_id username/packed-dataset \
    --max_seq_len 2048
```

## CLI Subcommands

### `tokenize`

Download and tokenize a HuggingFace dataset into parquet chunks.

```bash
python -m project_828.src.scripts.data.packing tokenize \
    --repo_id codeparrot/codeparrot-clean \
    --max_seq_len 2048 \
    --text_column content \
    --output_dir ./tokenized_data \
    --num_proc 2
```

| Option | Description | Default |
|--------|-------------|---------|
| `--repo_id` | HuggingFace source dataset | *required* |
| `--max_seq_len` | Tokenization truncation length / bin capacity | `2048` |
| `--text_column` | Name of the text column in the dataset | `content` |
| `--output_dir` | Directory for tokenized chunk parquets | `./tokenized_data` |
| `--num_proc` | Parallel workers for `dataset.map()` | `2` |

**Outputs:** `chunk_*.parquet` files in `output_dir`, plus `doc_lengths.npy`, `doc_indices.npy`, `chunk_sizes.npy` in the working directory.

### `pack`

Run best-fit bin packing on pre-computed document lengths.

```bash
python -m project_828.src.scripts.data.packing pack --max_seq_len 2048
```

| Option | Description | Default |
|--------|-------------|---------|
| `--max_seq_len` | Bin capacity (must match tokenization) | `2048` |

**Inputs:** `doc_lengths.npy`, `doc_indices.npy` in the working directory.  
**Output:** `packing_map.parquet` — mapping of `(original_idx, bin_id, seq_order)`.

After packing, statistics are logged:

```
─── Packing Statistics ───
  Total documents:      1,234,567
  Packed documents:     1,234,000
  Skipped (too long):   567
  Total bins:           312,000
  Bin capacity:         2,048 tokens
  Total packed tokens:  623,456,789
  Total bin capacity:   639,168,000
  Avg utilization:      97.54%
──────────────────────────
```

### `upload`

Materialize packed bins and push to HuggingFace Hub.

```bash
python -m project_828.src.scripts.data.packing upload \
    --repo_id username/packed-dataset \
    --data_dir ./tokenized_data
```

| Option | Description | Default |
|--------|-------------|---------|
| `--repo_id` | HuggingFace destination repo | *required* |
| `--data_dir` | Directory containing `chunk_*.parquet` | `./tokenized_data` |

### `all`

Run all three stages (`tokenize` → `pack` → `upload`) end-to-end.

```bash
python -m project_828.src.scripts.data.packing all \
    --repo_id codeparrot/codeparrot-clean \
    --upload_repo_id username/packed-dataset \
    --max_seq_len 2048 \
    --text_column content \
    --output_dir ./tokenized_data \
    --num_proc 2
```

| Option | Description | Default |
|--------|-------------|---------|
| `--repo_id` | HuggingFace source dataset | *required* |
| `--upload_repo_id` | HuggingFace destination repo | *required* |
| `--max_seq_len` | Tokenization truncation / bin capacity | `2048` |
| `--text_column` | Text column name | `content` |
| `--output_dir` | Directory for intermediary chunks | `./tokenized_data` |
| `--num_proc` | Parallel workers | `2` |

## Algorithm Details

### Best-Fit Bin Packing with Segment Tree

Documents are sorted by length in descending order (largest first). For each document, the algorithm queries a segment tree to find the bin whose remaining capacity most tightly fits the document. If no bin fits, a new bin is opened.

```
Segment Tree (indexed by remaining capacity)
├── Leaf nodes store the capacity value if any bin has that exact free space
├── Internal nodes store the max of their children
└── query(doc_size) → smallest capacity ≥ doc_size in O(log L)
```

- **Time complexity:** $O(N \log L)$ — one tree query per document
- **Space complexity:** $O(L)$ for the segment tree + $O(N)$ for output arrays
- **Numba JIT:** Tree operations are compiled with `@njit` for near-C performance

### Upload Carry-Forward

When streaming the sorted dataset in 20K-row batches for upload, a bin may span two consecutive batches. The upload generator holds back rows belonging to the last `bin_id` of each batch (the *carry*), prepends them to the next batch, and only emits fully-complete bins. The final carry is flushed at the end.

## Architecture

```
packing.py
├── CLI (argparse)                # tokenize | pack | upload | all
├── Tokenization Stage
│   ├── get_train_filenames()     # List parquet/json files from HF repo
│   ├── process_and_save_shard()  # Tokenize single shard with overflow splitting
│   └── orchestrate_tokenization()# Download → tokenize → save metadata
├── Packing Stage
│   ├── update_tree() / query_tree()  # Numba JIT segment tree ops
│   ├── BestFitPacking.pack()     # Core packing loop
│   └── run_packing()             # Load lengths → pack → save map + stats
└── Upload Stage
    └── materialize_and_upload()  # Join → sort → stream with carry-forward
```

---

## Preprocessing Architecture (NeMo Curator)

```
preprocess.py
├── StageRegistry          # Modular stage registration
│   ├── Filters            # Quality filtering stages
│   ├── Modifiers          # Text transformation stages
│   └── Deduplication      # Dedup workflows
├── custom_pipeline()      # HuggingFace dataset curation
├── dedup_pipeline()       # Deduplication execution
├── arxiv_pipeline()       # ArXiv processing
├── common_crawl_pipeline() # Common Crawl (requires AWS)
└── wikipedia_pipeline()   # Wikipedia dumps
```

## Error Handling

Custom exceptions for clear error messages:
- `ConfigurationError` - Invalid config file
- `PipelineStageError` - Stage execution failure
- `AWSCredentialsError` - Missing S3 credentials
- `DataDownloadError` - HuggingFace download failure

## References

- [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/index.html)
- [Text Processing Concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-processing-concepts.html)
- [Data Curation Pipeline](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-curation-pipeline.html)
- [Fewer Truncations Improve Language Modeling](https://arxiv.org/html/2404.10830v1) 
- [Best fit bin packing blog](https://www.amazon.science/blog/improving-llm-pretraining-with-better-data-organization#:~:text=To%20address%20this%20issue%2C%20we,3)