<div align="center">

# Project 828 - MoE Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-project--828--gpt--base-yellow.svg)](https://huggingface.co/AkshithAI/project-828-gpt-base)
[![Triton Kernels](https://img.shields.io/badge/Triton_Kernels-Enabled-orange.svg)](src/kernels/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Enabled-green.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](LICENSE)

**Mixture-of-Experts Transformer with Custom Triton Kernels & Advanced Training Pipeline**

A Mixture-of-Experts (MoE) transformer implementation featuring custom Triton CUDA kernels, Grouped Query Attention, RoPE positional encoding with YaRN extension, Gigatoken Rust/SIMD tokenization, Liger-Kernel integration, and distributed training support.

[Features](#features) • [Model Architecture](#model-architecture) • [🤗 HF Model Release](#3987m-model-configuration-baseline--v1--hugging-face-release) • [Custom Triton Kernels](#custom-triton-kernels) • [Quick Start](#quick-start) • [Training](#training) • [Results](#training-experiments--results)

---

- **Model Release (400M Base)**: [🤗 Hugging Face - `AkshithAI/project-828-gpt-base`](https://huggingface.co/AkshithAI/project-828-gpt-base)
- **Training Report** : [Phase 1 Pretraining H200 W&B Report](https://wandb.ai/akshithmarepally-akai/828_pretraining_h200/reports/Phase-1-Pretraining-H200:-Training-Dynamics-and-MoE-Routing--VmlldzoxNzQ4MTcwOA==)
</div>

## Table of Contents

- [Features](#features)
- [Model Architecture](#model-architecture)
  - [Architecture Overview](#architecture-overview)
  - [828M Model Configuration (Current Target)](#828m-model-configuration-current-target)
  - [398.7M Model Configuration (Baseline / v1) & Hugging Face Release](#3987m-model-configuration-baseline--v1--hugging-face-release)
- [Custom Triton Kernels](#custom-triton-kernels)
- [Gigatoken Rust/SIMD Tokenizer](#gigatoken-rustsimd-tokenizer)
- [YaRN Context Extension](#yarn-context-extension)
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

- **828M MoE Model Architecture (New)** - 16 routed experts + 1 shared expert with top-3 routing (~828M total parameters, ~330M active per token).
- **398.7M Baseline Architecture (v1)** - 4 routed experts + 1 shared expert with top-2 routing (398.7M total parameters, 286M active per token).
- **Custom Triton CUDA Kernels** - High-performance custom Triton kernels in [`src/kernels/`](src/kernels/) for fused linear cross-entropy loss, fused add + RMSNorm, SwiGLU with soft-clamping, and Triton RoPE.
- **Gigatoken Rust/SIMD Tokenization** - `microsoft/phi-2` tokenizer (~50,304 padded vocabulary) integrated with `gigatoken` Rust SIMD engine, releasing the Python GIL for 400x–1000x faster tokenization throughput.
- **Liger-Kernel Acceleration** - Integration of LinkedIn's `liger-kernel` package for Triton-fused GPU ops and CUDA MoE dispatch.
- **Grouped Query Attention (GQA)** - Efficient attention with 16 attention heads and 8 KV heads in 828M model (2:1 ratio) / 12 heads and 6 KV heads in 398M model.
- **Document-Aware Packing** - Variable-length Flash Attention 2 (`cu_seqlens`) for block-diagonal document packing without cross-document attention leakage.
- **Token-Based WSD Scheduler** - WSD (Warmup-Stable-Decay) learning rate scheduler driven by cumulative non-padding tokens (`total_tokens`).
- **Auxiliary-Loss-Free Load Balancing** - Dynamic router bias adjustment for uniform expert routing without interfering with training loss.
- **Compact YaRN Context Scaling** - Base context of 2048 tokens, extensible up to 8192 tokens using YaRN context extension (`scaling_factor=4.0`).

---

## Model Architecture

### Architecture Overview

Project 828 supports two primary model architecture configurations: the **828M MoE architecture** (current primary configuration in `new_model_config.py`) and the **398.7M MoE baseline architecture** (`model_config.py`).

| Metric / Property | 828M MoE Model (Target) | 398.7M MoE Model (Baseline) |
| :--- | :---: | :---: |
| **Total Parameters** | **~828 Million** | **398.7 Million** |
| **Active Parameters / Token** | **~330 Million** | **286 Million** |
| **Routed Experts** | 16 | 4 |
| **Active Experts / Token** | 3 (Top-3) | 2 (Top-2) |
| **Shared Experts** | 1 | 1 |
| **Hidden Dimension ($d_{\text{model}}$)** | 1024 | 768 |
| **Intermediate Size ($d_{\text{ff}}$)** | 520 | 760 |
| **Hidden Layers** | 24 | 24 |
| **Attention Heads / KV Heads** | 16 / 8 (GQA 2:1) | 12 / 6 (GQA 2:1) |
| **Head Dimension** | 64 | 64 |
| **Vocabulary Size** | 50,304 (Phi-2 128-byte aligned) | 50,304 (Phi-2 128-byte aligned) |
| **Tokenizer Engine** | Gigatoken Rust/SIMD (`microsoft/phi-2`) | Gigatoken Rust/SIMD (`microsoft/phi-2`) |
| **Precision** | BFloat16 Mixed Precision | BFloat16 Mixed Precision |

---

### 828M Model Configuration (Current Target)

The 828M model configuration defined in [`src/scripts/configs/new_model_config.py`](src/scripts/configs/new_model_config.py) utilizes 16 routed experts with top-3 token routing:

```python
# src/scripts/configs/new_model_config.py
vocab_size: int = 50_304              # Phi-2 tokenizer padded to 128-byte boundary
hidden_dim: int = 1024
intermediate_size: int = 518
num_hidden_layers: int = 24
num_attn_heads: int = 16
num_key_value_heads: int = 8          # 2:1 GQA ratio
head_dim: int = 64
num_experts: int = 16                 # Routed experts
num_experts_per_tok: int = 3          # Active experts per token (top-k)
use_liger_moe: bool = True            # CUDA MoE acceleration
router_bias_update_rate: float = 2e-3
base: int = 10000                     # RoPE base frequency
initial_context_len: int = 2048
max_context_len: int = 2048           # Extensible to 8192 via YaRN
dtype = torch.bfloat16
```

**828M Parameter Breakdown**:
- Embeddings: $50,304 \times 1024 \approx 51.5\text{M}$
- Unembedding: $1024 \times 50,304 \approx 51.5\text{M}$
- Attention per layer: $1024^2 + 2(1024 \times 512) + 1024^2 \approx 3.15\text{M}$
- MoE per layer: $16 \text{ routed} \times (2 \times 518 \times 1024 + 518 \times 1024) + 1 \text{ shared} \times (2 \times 518 \times 1024 + 518 \times 1024) \approx 27.05\text{M}$
- Layer Total ($24 \times (3.15\text{M} + 27.05\text{M})$): $\approx 725.2\text{M}$
- **Grand Total**: **828.2M parameters (828,215,296)** (**~331.7M active per token**).

---

### 398.7M Model Configuration (Baseline / v1) & Hugging Face Release

The 398.7M (~400M) baseline model is published and available on Hugging Face: **[`AkshithAI/project-828-gpt-base`](https://huggingface.co/AkshithAI/project-828-gpt-base)**.

The baseline 398.7M model configuration in [`src/scripts/configs/model_config.py`](src/scripts/configs/model_config.py):

```python
# src/scripts/configs/model_config.py
vocab_size: int = 50_304              # Padded Phi-2 tokenizer
hidden_dim: int = 768
intermediate_size: int = 760
num_hidden_layers: int = 24
num_attn_heads: int = 12
num_key_value_heads: int = 6          # 2:1 GQA ratio
head_dim: int = 64
num_experts: int = 4                  # Routed experts
num_experts_per_tok: int = 2          # Active experts per token
update_param: float = 2e-3
base: int = 10000
initial_context_len: int = 2048
max_context_len: int = 2048
dtype = torch.bfloat16
```

**398.7M Parameter Breakdown**: Embedding 37.7M + Unembedding 37.8M + 24×(Attention 1.8M + MoE 11.7M) + Norm = **398.7M total**, **~286M active per token**.

#### Quickstart with Hugging Face `transformers`

You can directly load and run inference on the model using `transformers`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AkshithAI/project-828-gpt-base"

# Load tokenizer and custom MoE model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "from collections import deque\n\ndef bfs(graph, start):\n    \"\"\"Breadth-first search returning visited nodes in order.\"\"\"\n    visited = set()\n    queue = deque([start])\n    result = []\n    while queue:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Custom Triton Kernels

Project 828 includes custom GPU kernels written in Triton under [`src/kernels/`](src/kernels/) to eliminate memory bottlenecks and accelerate training:

- **Fused Linear Cross-Entropy** ([`src/kernels/fused_linear_cross_entropy.py`](src/kernels/fused_linear_cross_entropy.py)): Combines the final linear LM projection layer with Cross-Entropy loss computation. Computes loss and gradients in a single chunked kernel pass without allocating the full $[B \times S, V]$ logits tensor in VRAM, saving gigabytes of memory at large sequence lengths.
- **Fused Add + RMSNorm** ([`src/kernels/fused_add_rms_norm.py`](src/kernels/fused_add_rms_norm.py)): Fuses residual connection addition with Root Mean Square Layer Normalization into a single Triton kernel to reduce memory read/write passes.
- **SwiGLU with Soft-Clamping** ([`src/kernels/swiglu.py`](src/kernels/swiglu.py)): Triton kernel for Swish-Gated Linear Units with soft-clamping (`limit=30.0` or `7.0`) to prevent activation explosions and ensure numerical stability during long training runs.
- **Triton RoPE** ([`src/kernels/apply_rope.py`](src/kernels/apply_rope.py)): Custom kernel for applying Rotary Position Embeddings directly to Q and K projections.

These custom kernels are combined with LinkedIn's `liger-kernel` for grouped GEMM expert routing in `model_adv.py`.

---

## Gigatoken Rust/SIMD Tokenizer

Tokenization is accelerated using the [`gigatoken`](gigatoken/) Rust SIMD backend:

- **Tokenizer Model**: `microsoft/phi-2` (code and technical domain optimized).
- **Padded Vocabulary**: 50,304 tokens (padded to a 128-byte boundary for CUDA alignment).
- **GIL-Free Parallelism**: Releases the Python Global Interpreter Lock (GIL) during encoding, allowing multi-threaded dataset batching alongside GPU training.
- **Throughput**: ~1000x faster tokenization throughput compared to standard Python tokenizer wrappers.

```python
# src/scripts/tokenizer.py
import gigatoken as gt
from transformers import AutoTokenizer

_hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
gt_tokenizer = gt.Tokenizer(_hf_tokenizer)
tokenizer = gt_tokenizer.as_hf()
```

---

## YaRN Context Extension

The model natively trains with a 2048 base context length. Context extension to 8192 tokens is implemented via YaRN (Yet another RoPE extensioN) as configured in [`src/scripts/configs/yarn_extension_config.py`](src/scripts/configs/yarn_extension_config.py) and [`yarn_extension_plan.md`](yarn_extension_plan.md):

- **Base Context Length**: 2048 tokens.
- **Extended Context Length**: 8192 tokens (`max_context_len = 8192`).
- **Scaling Factor**: `scaling_factor = 4.0` ($8192 / 2048$).
- **Frequency Interpolation**: Modifies high-frequency and low-frequency RoPE components with attention-temperature correction ($m = 1 + 0.1\ln(4) \approx 1.1386$) to preserve short-context recall while expanding sequence capacity.

---

## Dataset & Data Mix

> [!NOTE]
> The dataset specification in [`src/scripts/configs/new_model_config.py`](src/scripts/configs/new_model_config.py) is currently a **placeholder** (TBD) for the 828M pretraining run. Below are the historical dataset mixes utilized during Phase 1 and Phase 2 pretraining.

### Phase 1 — Baseline Pretraining (~60B tokens)

Weighted mix across 18 datasets:

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

**Category Summary**: Source Code (45%), General Knowledge (19%), Code-Adjacent (17%), Math/Reasoning (14%), Instruction (5%).

---

### Phase 2 — Code/Instruction (~18B tokens)

Phase 2 focuses heavily on code generation, execution reasoning, and CS knowledge:

| Category | Weight | Target Capability |
| :--- | :---: | :--- |
| **Code Replay** | **35%** | Multi-language generation correctness |
| **Educational Code** | **15%** | Execution reasoning (input/output prediction) |
| **CS Knowledge** | **18%** | Multi-domain CS factuality (DSA, systems, networks) |
| **General Knowledge** | **32%** | Factual educational prose |

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
- PyTorch 2.0+ with CUDA support
- CUDA-capable GPU (tested on RTX 4090/5090, H200)
- 16GB+ GPU memory recommended
- DeepSpeed (for multi-GPU training)
- Rust toolchain (for building Gigatoken SIMD backend)

### Installation

```bash
# Clone the repository
git clone https://github.com/AkshithAI/project_828.git
cd project_828

# Install dependencies
bash init.sh
```

### Training

#### Single GPU Training

```bash
# Set Weights & Biases API Key
export WANDB_API_KEY="your_wandb_key"

# Run single GPU training
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

# Launch distributed training on 2 GPUs
./launch_distributed.sh
```

---

## Project Structure

```
project_828/
├── src/
│   ├── kernels/
│   │   ├── apply_rope.py                  # Custom Triton RoPE kernel
│   │   ├── fused_add_rms_norm.py          # Fused Add + RMSNorm Triton kernel
│   │   ├── fused_linear_cross_entropy.py  # Fused Linear Cross-Entropy loss kernel
│   │   ├── swiglu.py                      # SwiGLU Triton kernel with soft-clamping
│   │   └── utils.py                       # Triton kernel helpers
│   ├── models/
│   │   ├── model_adv.py                   # Primary 828M MoE model implementation
│   │   ├── model_flash_attn.py            # Unified MoE model with Flash Attention & KV cache
│   │   ├── model_improv.py               # Improved MoE model variant
│   │   ├── new_full_training_plan.md      # Detailed 828M training specification
│   │   └── weight_init.py                 # Weight initialization routines
│   └── scripts/
│       ├── dataloader.py                  # Zero-stall resumable data loader
│       ├── dist_dataloader.py             # Distributed data loader
│       ├── helper_funcs.py                # Async checkpointing & path utilities
│       ├── inference.py                   # Autoregressive generation with KV cache
│       ├── tokenizer.py                   # Phi-2 Gigatoken SIMD tokenizer setup
│       ├── configs/
│       │   ├── ds-config.json             # DeepSpeed configuration
│       │   ├── model_config.py            # 398M baseline model config & Phase 1/2 configs
│       │   ├── new_model_config.py        # 828M target model config (placeholder datamix)
│       │   └── yarn_extension_config.py   # 8K YaRN context extension config
│       ├── training/
│       │   ├── distributed_training.py     # DeepSpeed multi-GPU training entrypoint
│       │   ├── train.py                   # Single-GPU phase-based training loop
│       │   ├── schedulers.py              # Token-based WSD & Cosine schedulers
│       │   ├── telemetry.py               # Async telemetry & routing logging
│       │   └── validate_domains.py        # Per-domain validation suite
│       └── data/
│           ├── packing.py                 # Best-fit bin packing pipeline
│           ├── preprocess.py              # NeMo Curator data preprocessing
│           ├── eval_suite.py              # Benchmark evaluation suite
│           └── README.md                  # Data curation documentation
├── gigatoken/                             # Rust SIMD BPE tokenization engine
├── tests/                                 # Unit & benchmark test suites
├── assets/                                # Telemetry screenshots & figures
├── check_flash_attn_requirements.py
├── gigatoken.sh                           # Gigatoken build script
├── init.sh                                # Environment init script
├── launch_distributed.sh                  # DeepSpeed distributed launcher script
├── yarn_extension_plan.md                 # Concise 8K YaRN context extension plan
├── requirements.txt
└── README.md
```

---

## Advanced Features

### Flash Attention 2 & Document-Aware Packing

The repository supports native Flash Attention 2 with document-aware variable-length sequence packing (`cu_seqlens`):

```python
# Document-aware varlen Flash Attention in model_adv.py
out = flash_attn_varlen_func(
    q_flat, k_flat, v_flat,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
```

Unrelated documents packed into a single sequence attend block-diagonally, avoiding cross-document token leakage while retaining maximum hardware throughput.

### Key-Value (KV) Cache for Inference

The model implements an efficient **KV Cache mechanism** that significantly boosts inference speed during autoregressive text generation:

**How it works:**
- During inference, the attention mechanism caches the Key (K) and Value (V) projections for all previously processed tokens
- For each new token, only the current token's Q, K, V need to be computed
- The cached K and V tensors are concatenated with the new K, V for attention computation
- This avoids redundant recomputation of K, V for the entire sequence at each generation step

**Implementation Details:**
```python
# Generate tokens with KV cache enabled
model = GPT_FLASH(config, device, inference=True)
logits = model(input_ids, start_pos=0)
```

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
| RTX 4090 | ~12 iter/sec | ~12GB | Development |
| RTX 5090 | ~14 iter/sec | ~11GB | Recommended |
| H100 | ~110K tokens/sec | — | Primary Training GPU |

### Multi-GPU Performance
| Configuration | Interconnect | Speed | Recommendation |
|--------------|----------|-------|----------------|
| 2x RTX 4090 | PIX | ~20 iter/sec | Good speedup |
| 2x RTX 4090 | NVLink | ~22 iter/sec | Strong speedup |
| 2x RTX 4090 | NODE | 0.01 iter/sec | Topology bottleneck — use 1 GPU |

**Key Takeaway:** Multi-GPU only helps with good P2P topology (PIX/PXB/NVLink). With NODE topology, single GPU is ~1200x faster!

## Monitoring Training

Key W&B metrics logged during training runs:
1. **Training Loss**: Smooth convergence trend.
2. **Validation Loss**: Monitored per-domain (Source Code, General Knowledge, Math, Instruction).
3. **Gradient Norm**: Clipped at 1.0 (alert if consistently > 5.0).
4. **Expert Utilization**: Balanced allocation across experts (target ~25% per expert for 4 experts, ~6.25% for 16 experts).
5. **Learning Rate**: WSD token-driven schedule tracking.

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

- **Padded Vocab Invariant**: `vocab_size` must equal `50_304` (128-byte aligned Phi-2 vocabulary). Token IDs are verified $< 50,304$.
- **GPU P2P Interconnect Bottlenecks**: Avoid running multi-GPU training over `NODE` or `SYS` PCIe topologys due to NCCL latency. Verify topology with `nvidia-smi topo -m`.
- **Soft-Clamping Parity**: SwiGLU activation bounding (`limit=30.0` or `7.0`) prevents activation explosions in deep MoE layers.

---

## Hardware Requirements

### Minimum Development Setup
- 1x RTX 4090 (24GB VRAM)
- 32GB System RAM
- 100GB Storage

### Production Training Setup
- 1x H200 / H100 GPU (or multi-GPU node with NVLink / PIX interconnects)
- 64GB+ System RAM
- 500GB NVMe SSD

---

## Troubleshooting Guide

1. **Verify GPU Topology**:
   ```bash
   nvidia-smi topo -m
   ```
2. **Run Single GPU Dry Run**:
   ```bash
   python -m src.scripts.training.train
   ```
3. **Verify Flash Attention & Triton Kernels**:
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

**Note**: This is a research project. The ~400M model is proving out the training pipeline and architecture. The 828M model is the target configuration for production.

</div>
