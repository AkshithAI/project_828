<div align="center">

# Project 828 - MoE Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Enabled-green.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](LICENSE)

**Mixture-of-Experts Transformer with Advanced Training Pipeline**

A Mixture-of-Experts (MoE) transformer implementation featuring a custom GPT-style architecture with Grouped Query Attention, RoPE positional encoding, efficient expert routing, and distributed training support via DeepSpeed.

[Features](#features) • [Architecture](#model-architecture) • [Quick Start](#quick-start) • [Training](#training) • [Results](#training-experiments--results)

---

</div>

## Table of Contents

- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset & Data Mix](#dataset--data-mix)
- [Data Curation](#data-curation)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training](#training)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [Performance Benchmarks](#performance-benchmarks)
- [Monitoring Training](#monitoring-training)
- [Training Experiments & Results](#training-experiments--results)
- [Known Issues & Solutions](#known-issues--solutions)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Features

- **Mixture of Experts (MoE)** - 4 routed experts + 1 shared expert with auxiliary-loss-free load balancing
- **Batched Sort-and-Slice Dispatch** - High-performance expert routing via argsort + searchsorted for contiguous memory access
- **Grouped Query Attention (GQA)** - Efficient attention with 12 attention heads and 6 KV heads (2:1 ratio)
- **Q/K Normalization** - RMSNorm applied to Query and Key projections for attention stability
- **RoPE with YaRN Scaling** - Rotary Position Embeddings with NTK-aware interpolation for context extension
- **SwiGLU Activation** - Gated activation function with clamping (`limit=7.0`) for numerical stability
- **DeepSpeed Integration** - ZeRO optimization stages 1-3 for distributed training
- **Flash Attention 2 Support** - Fully native integration with Flash Attention 2 for optimized memory efficiency
- **Mixed Precision Training** - BFloat16 for optimal performance
- **Comprehensive Logging** - Weights & Biases integration with per-layer expert utilization, per-domain validation metrics, and live eval reports
- **Robust Training Pipeline** - Phase-aware training with async checkpointing, data prefetching, and recovery workflows
- **Lab-Grade Evaluation Suite** - Comprehensive, datamix-aligned validation (MBPP, CRUXEval, Multilingual Completion, CS QA, Domain Perplexity)
- **NeMo Curator Integration** - Robust data preprocessing pipeline with filtering, cleaning, and deduplication
- **Best-Fit Bin Packing** - Segment-tree accelerated document packing with CLI, overflow splitting, and HF Hub upload

---

## Model Architecture

### Overview

- **Model Type**: GPT-style Decoder-only Transformer with Mixture of Experts
- **Total Parameters**: ~398.7M (286M active per token)
- **Context Length**: 2048 tokens (initial), extensible to 4096+ with YaRN scaling
- **Vocabulary**: StarCoder2-15B tokenizer (~49K tokens)
- **Precision**: BFloat16 mixed precision training

---

### Current Model Configuration

```python
# src/scripts/configs/model_config.py
vocab_size: tokenizer.vocab_size    # ~49,152
hidden_dim: 768
intermediate_size: 760
num_hidden_layers: 24
num_attn_heads: 12
num_key_value_heads: 6              # 2:1 GQA ratio
head_dim: 64                        # hidden_dim / num_attn_heads
num_experts: 4                      # Routed experts
num_experts_per_tok: 2              # Active experts per token (top-k)
update_param: 1e-3                  # Bias update rate for load balancing
route_scale: 1.0                    # Expert routing scale
base: 10000                         # RoPE base frequency
initial_context_len: 2048
max_context_len: 2048
dtype: bfloat16                     # Mixed precision training
ffn_dropout: 0.0
```

**Parameter breakdown**: Embedding 37.7M + Unembedding 37.8M + 24×(Attention 1.8M + MoE 11.7M) + Norm 768 = **398.7M total**, **~286M active per token** (2 of 4 routed experts + shared expert).

---

### Architecture Details

#### Core Components

**1. Attention Mechanism**
- **Type**: Grouped Query Attention (GQA) with RoPE
- **Attention Heads**: 12
- **KV Heads**: 6 (2:1 ratio for efficiency)
- **Q/K Normalization**: RMSNorm applied to Query and Key projections before RoPE
- **Position Encoding**: Rotary Position Embeddings (RoPE) with YaRN scaling
- **Special Feature**: Attention sinks (standard model) for improved long-context handling

**2. Mixture of Experts (MoE)**
- **Number of Experts**: 4 routed experts + 1 shared expert
- **Active Experts**: 2 experts per token (top-k routing)
- **Expert Architecture**: SwiGLU-based FFN with clamping for stability
  ```
  Expert(x) = W2(dropout(SwiGLU(W1(x), limit=7.0) * W3(x)))
  ```
- **SwiGLU Activation**: Includes clamping (`limit=7.0`) to prevent activation explosions
- **Routing**: Sigmoid gating with Auxiliary-Loss-Free Load Balancing (no auxiliary loss required)
- **Load Balancing**: Dynamic bias adjustment based on real-time token routing statistics
- **Dispatch**: Batched sort-and-slice — argsort by expert ID + searchsorted for contiguous slices

**3. Feed-Forward Network**
- **Hidden Dimension**: 768
- **Intermediate Size**: 760
- **Activation**: SwiGLU (Swish-Gated Linear Unit) with clamping for numerical stability
- **Dropout**: 0.0

**4. Normalization**
- **Type**: RMSNorm (Root Mean Square Layer Normalization)
- **Epsilon**: 1e-8 (flash model) / 1e-5 (standard model)
- **Applied**: Pre-normalization (before attention and FFN)

**5. Rotary Position Embeddings (RoPE)**
- **Base Frequency**: `base: 10000`
- **Scaling Method**: YaRN (Yet another RoPE extensioN method)
- **Context Extension**: `initial_context_len: 2048`, `max_context_len: 2048` (extensible to 4096+)
- **NTK Scaling Parameters**: `ntk_alpha: 1.0`, `ntk_beta: 32.0`, `scaling_factor: 1.0`

---

## Dataset & Data Mix

### Phase 1 — Post-Growth (~60B tokens)

The training uses 18 datasets in a weighted mix:

| Dataset | Weight | Category |
|:---|:---:|:---|
| `starcoderdata-python` | 14 | Source Code |
| `starcoderdata-javascript` | 6 | Source Code |
| `starcoderdata-java` | 5 | Source Code |
| `starcoderdata-typescript` | 4 | Source Code |
| `starcoderdata-cpp` | 4 | Source Code |
| `starcoderdata-c` | 3 | Source Code |
| `starcoderdata-csharp` | 3 | Source Code |
| `starcoderdata-go` | 3 | Source Code |
| `starcoderdata-rust` | 2 | Source Code |
| `starcoderdata-php` | 1 | Source Code |
| `fineweb-edu` (score ≥ 3.5) | 12 | General Knowledge |
| `cosmopedia-v2` | 7 | General Knowledge |
| `openmath-instruct-2` | 10 | Math/Reasoning |
| `numina-math-cot` (len ≤ 8000) | 4 | Math/Reasoning |
| `stack-exchange-preferences` | 7 | Code-Adjacent |
| `proof-pile-algebraic-stack` | 5 | Code-Adjacent |
| `magicoder-oss-instruct` (3 epochs) | 5 | Code-Adjacent |
| `openhermes-2.5` | 5 | Instruction |

**Category breakdown**:
```
Source Code:         45%  (10 languages from starcoderdata)
General Knowledge:   19%  (fineweb-edu + cosmopedia)
Code-Adjacent:       17%  (stack-exchange + algebraic-stack + magicoder)
Math/Reasoning:      14%  (openmath + numina)
Instruction:          5%  (openhermes)
```

### Phase 2 — Code/Instruction (~18B tokens)

Phase 2 focuses heavily on code generation, reasoning, and factual computer science knowledge with the following target datamix:

| Phase 2 Category | Weight | Target Capability | Aligning Evaluation Benchmark |
| :--- | :---: | :--- | :--- |
| **Code Replay** | **35%** | Multi-language generation correctness (Python, JS, TS, C++, Go, Rust) | MBPP & MultiPL-E |
| **Educational Code** | **15%** | Execution reasoning (predicting input/output of functions) | CRUXEval-O & CRUXEval-I |
| **CS Knowledge** | **18%** | Multi-domain CS factuality (DSA, networks, databases, systems) | CS-QA Curated Benchmark |
| **General Knowledge** | **32%** | High-quality prose and educational/factual texts | Held-out Domain Perplexity |

### Tokenizer

- **Source**: `bigcode/starcoder2-15b` (~49K vocabulary)
- **Format**: Parquet/JSON with text content
- **Preprocessing**: Document-level packing with EOS tokens, resumable dataloader with state checkpointing

---

## Data Curation

This project includes a comprehensive data preprocessing pipeline powered by [NVIDIA NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/).

### Features

- **Quality Filtering** - Word count, punctuation, boilerplate removal
- **Text Cleaning** - Unicode normalization, URL removal, newline standardization  
- **Deduplication** - Exact, fuzzy (MinHash), and semantic deduplication
- **Multiple Sources** - HuggingFace datasets, ArXiv, Common Crawl, Wikipedia

### Quick Usage

```bash
# Curate a HuggingFace dataset
python src/scripts/data/preprocess.py --custom --repo_id username/dataset-name

# With deduplication
python src/scripts/data/preprocess.py --custom --dedup

# Predefined datasets (ArXiv, Wikipedia, Common Crawl)
python src/scripts/data/preprocess.py --data_tag wiki --download_dir ./data
```

### Document Packing

The project includes a **best-fit bin packing** pipeline (`packing.py`) for packing variable-length tokenized documents into fixed-capacity bins:

- Numba-accelerated segment tree — $O(N \log L)$ packing for millions of documents
- Overflow splitting — documents longer than `max_seq_len` are split at tokenization
- Single `--max_seq_len` parameter ensures tokenization truncation and bin capacity always match
- Carry-forward streaming upload

```bash
# End-to-end: tokenize → pack → upload
python -m project_828.src.scripts.data.packing all \
    --repo_id codeparrot/codeparrot-clean \
    --upload_repo_id username/packed-dataset \
    --max_seq_len 2048

# Or run stages independently
python -m project_828.src.scripts.data.packing tokenize --repo_id codeparrot/codeparrot-clean --max_seq_len 2048
python -m project_828.src.scripts.data.packing pack --max_seq_len 2048
python -m project_828.src.scripts.data.packing upload --repo_id username/packed-dataset
```

**[Full Documentation →](src/scripts/data/README.md#best-fit-bin-packing-pipeline)**

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4090/5090, H200)
- 16GB+ GPU memory recommended
- DeepSpeed (for multi-GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/AkshithAI/project_828.git
cd project_828

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

### Training

#### Single GPU Training (Recommended for Development)

```bash
# Set your Weights & Biases credentials
export WANDB_API_KEY="your_wandb_key"

# Start training
python -m src.scripts.training.train
```

#### Multi-GPU Distributed Training with DeepSpeed

**Prerequisites:** Check GPU topology first!

```bash
# Check if your GPUs have good P2P connection
nvidia-smi topo -m
```

**Good topologies (Fast):**
- `PIX` - Single PCIe bridge (best for 2 GPUs)
- `PXB` - Multiple PCIe bridges (good)
- `NV#` - NVLink connection (excellent)

**Bad topologies (Very Slow):**
- `NODE` - Cross NUMA node (use single GPU)
- `SYS` - Cross CPU socket (use single GPU)

**Launch distributed training:**
```bash
export WANDB_API_KEY="your_wandb_key"
export PYTHONPATH=$(pwd)

# Run on 2 GPUs
deepspeed --num_gpus=2 src/scripts/training/distributed_training.py \
    --deepspeed \
    --deepspeed_config src/scripts/configs/ds-config.json \
    --batch_size 8
```

#### Phase 1 Hyperparameters

| Param | Value |
|:---|:---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1, ε=1e-8) |
| Peak LR | 3e-4 |
| Min LR | 3e-5 |
| Scheduler | WSD (Warmup → Stable → Cosine Decay) |
| WSD stable fraction | 0.895 |
| Warmup steps | 500 |
| Total steps | 101,726 |
| Micro batch size | 37 |
| Grad accumulation | 8 steps |
| Effective batch | 296 sequences (~606K tokens/step) |
| Gradient clipping | 1.0 |
| Precision | BF16 autocast |
| Validation interval | every 2,000 steps |

## Project Structure

```
project_828/
├── src/
│   ├── models/
│   │   ├── model_flash_attn.py     # Unified MoE GPT model with Flash Attention, Q/K Norm, and KV Cache
│   │   └── weight_init.py          # Model weight initialization
│   └── scripts/
│       ├── tokenizer.py            # StarCoder2 tokenizer setup
│       ├── dataloader.py           # Resumable data loading with state checkpointing
│       ├── dist_dataloader.py      # Legacy distributed data loading
│       ├── helper_funcs.py         # Utility functions (sync + async checkpointing, paths)
│       ├── inference.py            # Autoregressive generation with KV cache + routing stats
│       ├── configs/
│       │   ├── model_config.py     # Model, Phase, and Dataset configuration (e.g. eval_suite_interval)
│       │   └── ds-config.json      # DeepSpeed configuration
│       ├── training/
│       │   ├── train.py            # Single GPU phase-based training loop
│       │   ├── distributed_training.py  # DeepSpeed multi-GPU training
│       │   ├── schedulers.py       # WSD + Cosine LR schedulers
│       │   └── validate_domains.py # Per-domain validation suite
│       └── data/
│           ├── packing.py          # Best-fit bin packing pipeline
│           ├── preprocess.py       # NeMo Curator data preprocessing
│           ├── preprocess.json     # Preprocessing configuration
│           ├── eval_suite.py       # Datamix-aligned comprehensive evaluation suite
│           ├── eval_benchmarks.py  # Validation metrics and checkpoint evaluation runner
│           └── README.md           # Data curation & packing documentation
├── tests/
│   ├── eval_suite_prompts.py       # Curated multi-language code and CS QA prompts
│   ├── eval_runner.py              # Evaluator run harness
│   ├── test_eval_suite.py          # CLI entry point for evaluation runs & checkpoint comparisons
│   ├── test_phase2_knowledge.py    # Unit tests for CS QA and knowledge validation
│   └── test_moe_batched_dispatch.py # MoE dispatch correctness tests
├── checkpoints/
├── requirements.txt
└── README.md
```

## Advanced Features

### Flash Attention Support

The repository utilizes the high-performance Flash Attention implementation (`GPT_FLASH` from `model_flash_attn.py`) unconditionally as its native architecture. This delivers a 40%+ speedup and substantial GPU memory savings during training and inference.

```python
# Unconditional Native Instantiation
model = GPT_FLASH(config, "cuda")
```

### Lab-Grade Evaluation Suite

We have built a comprehensive evaluation engine (`eval_suite.py`) to systematically measure and log the model's performance on 5 diverse benchmarks tailored to the training mix:

1. **MBPP Benchmark**: Autoregressive code generation evaluated against test suites in a Python sandbox.
2. **CRUXEval**: Multi-turn code reasoning and input/output prediction.
3. **Multilingual Code Completion**: Syntactic and structural code validation across Python, JS, TS, C++, Go, and Rust.
4. **CS Knowledge QA**: Log-likelihood scoring of domain-specific computer science questions.
5. **Domain Perplexity**: Cross-entropy perplexity tracking on held-out datamix shards.

The evaluation runner runs automatically at custom checkpoint intervals (`eval_suite_interval` in config) or via CLI for manual evaluation:

```bash
# Standalone evaluation of a checkpoint on GPU
python -m tests.test_eval_suite \
    --checkpoint checkpoints/model_101002.pt \
    --device cuda \
    --bench mbpp cruxeval multiple code_completion cs_qa domain_ppl

# Quick CPU-based dry-run/smoke test
python -m tests.test_eval_suite \
    --checkpoint checkpoints/model_06767.pt \
    --device cpu --quick
```

### Key-Value (KV) Cache for Inference

The model implements an efficient **KV Cache mechanism** that significantly boosts inference speed during autoregressive text generation:

**How it works:**
- During inference, the attention mechanism caches the Key (K) and Value (V) projections for all previously processed tokens
- For each new token, only the current token's Q, K, V need to be computed
- The cached K and V tensors are concatenated with the new K, V for attention computation
- This avoids redundant recomputation of K, V for the entire sequence at each generation step

**Implementation Details:**
```python
# In Attention class (model_flash_attn.py)
self.register_buffer("cache_k", torch.zeros(
        1, config.initial_context_len, config.num_key_value_heads, config.head_dim, device = device , dtype = config.dtype), persistent=False
)
self.register_buffer("cache_v", torch.zeros(
        1, config.initial_context_len, config.num_key_value_heads, config.head_dim, device = device , dtype = config.dtype), persistent=False
)

# During forward pass with inference=True:
if self.inference:
    self.cache_k[:, start_pos:end_pos, :, :] = K
    self.cache_v[:, start_pos:end_pos, :, :] = V
    K = self.cache_k[:, :end_pos, :, :]
    V = self.cache_v[:, :end_pos, :, :]
```

**Performance Benefits:**
- **Time Complexity**: Reduces from O(n²) to O(n) per token generation (where n is sequence length)
- **Speed Boost**: ~10-50x faster inference compared to full recomputation
- **Memory Trade-off**: Uses additional memory proportional to `initial_context_len × num_kv_heads × head_dim`

**Usage:**
```python
# Enable KV cache by setting inference=True when creating the model
model = GPT_FLASH(config, device, inference=True)

# Generate tokens with positional tracking
start_pos = 0
model(initial_tokens, start_pos)  # Prefill cache
start_pos = len(initial_tokens)
for _ in range(max_new_tokens):
    logits = model(next_token.view(1, 1), start_pos)
    start_pos += 1
    # ... sample next token
```

**Note**: KV cache is automatically used when `inference=True` is passed to the model constructor. During training, the cache is bypassed for efficiency.

### Auxiliary-Loss-Free Load Balancing for MoE

Based on the [DeepSeek-V3 paper](https://arxiv.org/abs/2408.15664):

1. **Sigmoid gating**: `scores = sigmoid(Linear(x))` — not softmax
2. **Decoupled selection vs. weighting**: Biased scores select experts, original scores weight them
3. **Dynamic bias**: `bias += update_param * sign(mean_load - current_load)`, clamped to [-10, 10]
4. **No auxiliary loss**: No interference with the primary training objective

### Batched Expert Dispatch

High-performance sort-and-slice dispatch replaces sequential expert loop:

```python
# Sort tokens by expert ID for contiguous memory access
sort_order = flat_idx.argsort(stable=True)
# Find expert boundaries via searchsorted
expert_boundaries = torch.searchsorted(sorted_expert_ids, arange(num_experts + 1))
# Each expert processes its contiguous slice
for i, expert in enumerate(self.experts):
    start, end = expert_boundaries[i], expert_boundaries[i + 1]
    sorted_out[start:end] = expert(sorted_x[start:end])
# Weighted scatter-add back to original positions
routed_output.scatter_add_(0, sorted_token_idx.expand_as(sorted_out), sorted_out)
```

### Async Checkpointing

Avoids blocking training during checkpoint saves:

1. Pauses data prefetch thread (reduces CPU contention)
2. Snapshots state dicts to CPU (fast D2H transfer, ~1-2s)
3. Resumes prefetch immediately
4. Background thread handles disk I/O + W&B artifact upload
5. Falls back to synchronous save when CPU memory < 5GB

### Domain-Specific Validation

Per-domain loss/perplexity evaluation across 5 macro categories:

```python
from .validate_domains import validate_domains
domain_results = validate_domains(model, wandb_run, optim_step, phase_config)
```

Tracks relative model performance across Source Code, General Knowledge, Math/Reasoning, Code-Adjacent, and Instruction domains.

### DeepSpeed ZeRO Optimization

**ZeRO Stage 3** (recommended): Optimizer + Gradient + Parameter partitioning with CPU offloading. Configure in `ds-config.json`.

### Checkpoint Management

Files per step: `model_{step:05d}.pt`, `optim_{step:05d}.pt`, `scheduler_{step:05d}.pt`, `dataloader_{step:05d}.pt`

Dataloader checkpoints include full mixer state: per-dataset document counts, token buffers, shard positions, and draw cycle position for exact resumption.

---

## Performance Benchmarks

### Single GPU Performance
| GPU | Speed | Memory | Recommendation |
|-----|-------|--------|----------------|
| RTX 4090 | ~12 iter/sec | ~12GB | Recommended for development |
| RTX 5090 | ~14 iter/sec | ~11GB | Recommended |
| H200 | ~110K tokens/sec | — | Primary training GPU |

### Multi-GPU Performance
| Configuration | Topology | Speed | Recommendation |
|--------------|----------|-------|----------------|
| 2x RTX 4090 | PIX | ~20 iter/sec | Good speedup |
| 2x RTX 4090 | NVLink | ~22 iter/sec | Strong speedup |
| 2x RTX 4090 | NODE | 0.01 iter/sec | Use 1 GPU instead |

**Key Takeaway:** Multi-GPU only helps with good P2P topology (PIX/PXB/NVLink). With NODE topology, single GPU is ~1200x faster!

## Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease smoothly
2. **Validation Loss**: Should track training loss closely
3. **Gradient Norm**: Should stay below 5.0 (clipped at 1.0)
4. **Learning Rate**: Should follow WSD schedule
5. **Per-Domain Validation**: Check relative domain losses for data mix balance
6. **Expert Utilization**: Near 25% per expert (for 4 experts)

### Red Flags
- Loss becomes NaN → reduce learning rate
- Loss increases for >5K steps → check data pipeline
- Grad norm consistently >10 → gradient explosion, reduce learning rate
- Val loss >> train loss → possible overfitting, add regularization
- Expert utilization <10% for any expert → check update_param setting
- Multi-GPU < 1 iter/sec → likely topology bottleneck, use single GPU

---

## Training Experiments & Results

### Training Runs Summary

| Run | Steps | Dataset | Key Issue | Grad Norm | Training Stability |
|-----|-------|---------|-----------|-----------|-------------------|
| **Run 1** | 240,000 | Code | RoPE shape mismatch (B*S vs B,S) | N/A | Unstable |
| **Run 2** | 110,000 | Code | Gradient norm explosion | Peak ~25 | Very noisy |
| **Run 3** | 50,000 | Language | None (all fixes applied) | Peak ~6 | Stable |
| **Run 4** | 50,000 | Language | Expert collapsing in MoE | Peak ~7 | Stable + balanced |
| **Run 5** | Ongoing | Multi-dataset | SwiGLU activation overflow | - | Stable (with clamping) |

### Detailed Experiment Analysis

#### Run 1: 240k Steps - Critical RoPE Bug Discovery

**Configuration**: 5090_run_240k_steps

**Critical Issue Discovered**: RoPE Positional Encoding Shape Mismatch
- **Bug Description**: The RoPE positional encoding was calculated for tensor shape `(B*S, ...)` (batch × sequence flattened), but the attention layer performed a reshape operation to `(B, S, ...)` (batch, sequence separate) before applying attention
- **Impact**: This caused a severe position encoding mismatch where tokens received incorrect positional information
- **Symptoms**: 
  - Training loss highly unstable and oscillating (fluctuating between 2-10)
  - High validation loss variance
  - Attention mechanism receiving corrupted positional signals
  - Model unable to learn proper sequence relationships

**Training Metrics**:
- Training loss: Oscillating between 2-10 (no convergence)
- Validation loss: High variance, no improvement trend
- Gradient norms: Not properly tracked in this run
- Result: **Training abandoned** due to fundamental positional encoding bug

**Screenshot**: Training metrics showing unstable loss patterns

![Run 1 - 240k Steps Metrics](assets/screenshots/Screenshot%202025-12-11%20at%208.13.10%E2%80%AFPM.png)

---

#### Run 2: 110k Steps - Post-Fix Gradient Instability

**Configuration**: 5090_run_110k_steps

**Dataset**: Code dataset (CodeParrot-Clean)

**Fixes Applied**:
- Fixed RoPE attention reshape issue - corrected positional encoding to match attention layer tensor shapes

**New Issue Discovered**: Gradient Norm Explosion
- **Problem**: Despite fixing the RoPE bug, training remained highly unstable
- **Cause**: Code dataset characteristics (long sequences, complex patterns) causing gradient instability
- **Symptoms**:
  - Gradient norm spikes up to 25+ (despite gradient clipping at 1.0)
  - Training loss extremely noisy, oscillating between 2-10
  - Perplexity exploding to 20,000+
  - Learning rate schedule functioning correctly, but gradients too unstable

**Training Metrics**:
- Training loss: Still oscillating 2-10, very noisy
- Perplexity: Spikes up to 20,000+
- Gradient norm: **Peak ~25** (indicating severe gradient explosion)
- Learning rate: Following cosine schedule correctly
- Result: **Unstable training**, problematic for convergence

**Screenshot**: Training metrics showing gradient explosion and noisy loss

![Run 2 - 110k Steps Metrics](assets/screenshots/Screenshot%202025-12-11%20at%208.13.31%E2%80%AFPM.png)

---

#### Run 3: 50k Steps - Stable Training Achieved

**Configuration**: Latest run (ongoing/best results)

**Dataset**: **Language dataset** (switched from code)

**All Fixes Applied**:
- RoPE attention reshape fix (from Run 1)
- Switched to language dataset for more stable gradients (from Run 2 insights)

**Results**: Significantly improved stability.

**Training Metrics**:
- **Gradient norm**: Now peaked around **6** (down from 25+) - 4x improvement.
- **Training loss**: Much less noisy, smoothly decreasing from ~11 to ~5
- **Perplexity**: Dropping smoothly from 20,000+ to stable low values (proper convergence)
- **Learning rate**: Following proper cosine warmup schedule
- **Overall**: **Stable, converging training** - ready for long-term runs

**Key Improvements**:
- 4x reduction in gradient norm peaks (25 → 6)
- Smooth loss convergence instead of oscillation
- Perplexity showing proper learning dynamics
- No training instability issues

**Screenshot**: Training metrics showing stable convergence

![Run 3 - 50k Steps Metrics](assets/screenshots/Screenshot%202025-12-13%20at%2012.33.28%E2%80%AFPM.png)

---

#### Run 4: 50k Steps - MoE Load Balancing & Attention Stability

**Configuration**: denim-dew-45

**Dataset**: Language dataset

**Problems Identified**:
- Expert collapsing observed in MoE - some experts receiving disproportionate token allocation
- Residual variance in gradient norms and training loss during training

**Fixes Applied**:
- Added **Q-Norm and K-Norm** to attention for improved stability
- Implemented **auxiliary-loss-free load balancing** in the MoE router

**Results**: Perfect load balancing achieved.

**Training Metrics**:
- **Gradient norm**: Peaked around ~7, with some variance but overall stable
- **Training loss**: Smoothly decreasing from ~11 to ~4, consistent convergence
- **Perplexity**: Proper exponential decay from 20,000+ to stable low values
- **Learning rate**: Following linear warmup + cosine schedule correctly
- **Expert Utilization**: Near-perfect load balancing across all experts

**Key Improvements**:
- **Q/K Normalization**: Added RMSNorm to query and key projections before RoPE application
  - Provides additional stability to attention mechanism
  - Marginal but significant improvement in training dynamics
- **Loss-Free Load Balancing**: Dynamic bias adjustment ensures equal expert utilization
  - No auxiliary loss required (eliminates hyperparameter tuning)
  - Bias term adapts in real-time based on token routing statistics
  - Achieves near-perfect ~25% utilization per expert (for 4 experts)

**Screenshot**: Training metrics showing stable convergence with load-balanced MoE

![Run 4 - 50k Steps Metrics](assets/screenshots/Screenshot%202026-01-14%20at%205.55.34%E2%80%AFPM.png)

---

#### Run 5: SwiGLU Stability & Resumable Training

**Configuration**: 828_testing_5090

**Dataset**: FineWeb-Edu-100B (high-quality educational content)

**New Features Implemented**:
- **SwiGLU with Clamping**: Added `limit=7.0` clamping to prevent activation explosions
- **Resumable Dataloader**: State checkpointing for seamless training resumption
- **Enhanced Checkpointing**: Model, optimizer, scheduler, and dataloader states saved together
- **Inline Inference**: Sample generation during validation for qualitative monitoring
- **Expert Utilization Logging**: Real-time W&B metrics for MoE load balancing

**SwiGLU Clamping Implementation**:
```python
def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x.chunk(2, dim=-1) 
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)
```

**Key Improvements**:
- Prevents NaN/Inf from large activation values
- Maintains gradient flow while bounding outputs
- Training can resume from any checkpoint without re-processing data

---

### Lessons Learned

1. **RoPE shape must match attention**: Positional encoding dimensions must align with attention tensor reshape operations
2. **Code datasets cause gradient instability**: Start with language dataset, then fine-tune on code
3. **SwiGLU needs clamping**: `limit=7.0` prevents activation explosions in long training runs
4. **Loss-Free Balancing works**: Sigmoid gating + adaptive bias achieves near-perfect expert utilization without auxiliary losses
5. **Resumable data loading is critical**: Saves exact data position + token buffers across 18 datasets
6. **Async checkpointing prevents training stalls**: D2H snapshot + background I/O avoids blocking the training loop

---

## Known Issues & Solutions

### Training Instability
- **Fixed**: Loss logging bug (incorrect gradient accumulation scaling)
- **Fixed**: Data pipeline context length mismatch
- **Fixed**: Dtype mismatch in attention output

### Memory Issues
- Use ZeRO Stage 2 or 3 for distributed training
- Reduce batch size if OOM
- Enable Flash Attention for 40% memory reduction

### Distributed Training Issues

**NCCL Timeout Errors**: Poor GPU P2P topology (NODE/SYS). Use single GPU training instead.

**Multi-GPU Slower Than Single GPU**: Check topology with `nvidia-smi topo -m`. If NODE/SYS, use single GPU.

---

## Model Unified Architecture (`model_flash_attn.py`)

The repository has transitioned to a single, unified, high-efficiency architecture:

- **Flash Attention 2**: Native hardware-accelerated attention support.
- **Q/K Normalization**: Attention stability via pre-attention Query/Key RMSNorm scaling.
- **KV Cache for Inference**: Fast autoregressive text generation.
- **Batched Expert Dispatch**: High-performance contiguous memory sort-and-slice MoE dispatch.
- **Auxiliary-Loss-Free Load Balancing**: Load balancing via dynamic router bias adjustment.

## Hardware Requirements

### Minimum (Single GPU)
- 1x RTX 4090 (24GB VRAM)
- 32GB RAM
- 100GB storage

### Recommended (Training at Scale)
- 1x H200 (primary training GPU)
- 64GB+ RAM
- 500GB NVMe SSD

### Cloud Provider Tips
- **Vast.ai:** Filter for "NVLink" in search
- **Always check topology first:** `nvidia-smi topo -m`
- If NODE topology: prefer hardware with PIX, PXB, or NVLink interconnects
- Look for "PCIe 4.0 x16" with modern CPUs (AMD EPYC 7xxx series)

## Troubleshooting Guide

### Before Starting Multi-GPU Training

1. **Check GPU topology:**
   ```bash
   nvidia-smi topo -m
   ```
  - PIX/PXB/NVLink → suitable for multi-GPU training
  - NODE/SYS → use single GPU instead

2. **Test single GPU first:**
   ```bash
   python -m src.scripts.training.train
   ```

3. **Verify NCCL:**
   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{project828,
  author = {AkshithAI},
  title = {Project 828: MoE Transformer with Advanced Training Pipeline},
  year = {2025},
  version = "1.0.0",
  publisher = {GitHub},
  url = {https://github.com/AkshithAI/project_828}
}
```

---

## License

[TBD]

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or issues, please open an issue on GitHub or contact [@AkshithAI](https://github.com/AkshithAI).

---

<div align="center">

**Project 828** | Version 1.0.0

*MoE Transformer Architecture — 398.7M params, 286M active per token*

**Note**: This is a research project. The ~400M model is proving out the training pipeline and architecture. The 800M model is the target configuration for production.

</div>
