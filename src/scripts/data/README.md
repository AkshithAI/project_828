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

## Architecture

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
