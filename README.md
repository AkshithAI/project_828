# Project 828 - MoE Transformer for Code Generation

A Mixture-of-Experts (MoE) transformer model implementation for code generation, trained on the CodeParrot dataset. This project features a custom GPT-style architecture with advanced attention mechanisms, efficient expert routing, and distributed training support via DeepSpeed.

## 🏗️ Model Architecture

### Overview
- **Model Type**: GPT-style Decoder-only Transformer with Mixture of Experts
- **Current Configuration**: ~60M parameters (test configuration)
- **Target Configuration**: 800M parameters (planned)
- **Context Length**: 2048 tokens
- **Vocabulary**: StarCoder2-15B tokenizer (~49K tokens)

### Architecture Details

#### Core Components

**1. Attention Mechanism**
- **Type**: Grouped Query Attention (GQA) with RoPE
- **Attention Heads**: 8 (configurable)
- **KV Heads**: 2 (4:1 ratio for efficiency)
- **Position Encoding**: Rotary Position Embeddings (RoPE) with YaRN scaling
- **Special Feature**: Attention sinks for improved long-context handling

**2. Mixture of Experts (MoE)**
- **Number of Experts**: 4 routed experts + 1 shared expert
- **Active Experts**: 2 experts per token (top-k routing)
- **Expert Architecture**: SwiGLU-based FFN
  ```
  Expert(x) = W2(dropout(SwiGLU(W1(x) * W3(x))))
  ```
- **Routing**: Learned gating with group-based selection with Auxiliary-Loss-Free Load Balancing 
- **Groups**: 2 groups with top-1 group selection

**3. Feed-Forward Network**
- **Hidden Dimension**: 512 (test) / 1024+ (production)
- **Intermediate Size**: 768 (test) / 2048+ (production)
- **Activation**: SwiGLU (Swish-Gated Linear Unit)
- **Dropout**: 0.0 (disabled for stability)

**4. Normalization**
- **Type**: RMSNorm (Root Mean Square Layer Normalization)
- **Epsilon**: 1e-5
- **Applied**: Pre-normalization (before attention and FFN)

**5. Rotary Position Embeddings (RoPE)**
- **Base Frequency**: 10000
- **Scaling Method**: YaRN (Yet another RoPE extensioN method)
- **NTK-aware interpolation** for context length extension
- **Concentration factor**: 1.0 (default)

### Current Model Configuration (60M params)

```python
# Test Configuration
vocab_size: 49,152 (StarCoder2 tokenizer)
hidden_dim: 512
intermediate_size: 768
num_hidden_layers: 1
num_attn_heads: 8
num_key_value_heads: 2
head_dim: 64  # hidden_dim / num_attn_heads
num_experts: 4
num_experts_per_tok: 2
context_length: 2048
dtype: bfloat16
```

### Planned 800M Configuration

```python
# Production Configuration (Recommended)
hidden_dim: 1536
intermediate_size: 4096
num_hidden_layers: 24
num_attn_heads: 24
num_key_value_heads: 8
head_dim: 64
num_experts: 8
num_experts_per_tok: 2
```

## 📊 Dataset

- **Training**: CodeParrot-Clean (streaming, 54 shards)
- **Validation**: CodeParrot-Clean-Valid
- **Tokenizer**: StarCoder2-15B tokenizer
- **Data Format**: JSON Lines with code content
- **Preprocessing**: Document-level packing with EOS tokens

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4090/5090)
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

**Performance:** ~12 iterations/second on RTX 4090

```bash
# Set your Weights & Biases credentials
export WANDB_API_KEY="your_wandb_key"

# Start training
python -m src.scripts.train
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
- `NODE` - Cross NUMA node ⚠️ **Use single GPU instead!**
- `SYS` - Cross CPU socket ⚠️ **Use single GPU instead!**

**Launch distributed training:**
```bash
export WANDB_API_KEY="your_wandb_key"
export PYTHONPATH=$(pwd)

# Run on 2 GPUs
deepspeed --num_gpus=2 src/scripts/distributed_training.py \
    --deepspeed \
    --deepspeed_config src/scripts/ds-config.json \
    --batch_size 8

# Or use the launch script
bash launch_distributed.sh
```

#### Configuration

Edit `src/scripts/configs.py` to modify model architecture:

```python
class ModelConfig:
    def __init__(self):
        # Model architecture
        self.hidden_dim = 512
        self.num_hidden_layers = 1
        self.num_attn_heads = 8
        self.num_experts = 4
        
        # Training
        self.learning_rate = 3e-4
        self.batch_size = 8
```

Edit `src/scripts/train.py` for single-GPU training hyperparameters:

```python
# Training settings
grad_accumulation_step = 16  # Effective batch size = batch_size * grad_accum
num_warmup_steps = 2000      # 2% of total steps
num_training_steps = 100000  # Total training steps
```

Edit `src/scripts/ds-config.json` for distributed training settings:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 2,
  "zero_optimization": {
    "stage": 2
  }
}
```

### Training Hyperparameters

**Recommended for 60M model (100K steps):**
- **Learning Rate**: 3e-4
- **Warmup Steps**: 2,000 (single GPU) / 100 (distributed)
- **Batch Size**: 8 sequences/GPU
- **Gradient Accumulation**: 16 steps (single GPU) / 2 steps (distributed)
- **Effective Batch Size**: 128 sequences (~262K tokens)
- **Optimizer**: AdamW (β1=0.9, β2=0.95, weight_decay=0.01)
- **LR Schedule**: Cosine with warmup
- **Gradient Clipping**: 1.0
- **Precision**: BFloat16 mixed precision

**Expected Training Time:**
- 60M model (single GPU): ~24-48 hours for 100K steps (RTX 5090)
- 60M model (2 GPUs with good P2P): ~12-24 hours for 100K steps
- 800M model: ~7-14 days for 500K steps (8x A100 with NVLink)

## 📁 Project Structure

```
project_828/
├── src/
│   ├── models/
│   │   ├── model.py              # Main GPT model with standard attention
│   │   ├── model_flash_attn.py   # GPT model with Flash Attention
│   │   └── weight_init.py        # Model weight initialization
│   └── scripts/
│       ├── train.py              # Single GPU training loop
│       ├── distributed_training.py  # DeepSpeed distributed training
│       ├── dist_dataloader.py    # Distributed data loading
│       ├── ds-config.json        # DeepSpeed configuration
│       ├── configs.py            # Model configuration
│       ├── dataloader.py         # Data loading and preprocessing
│       ├── tokenizer.py          # Tokenizer setup
│       └── helper_funcs.py       # Utility functions
├── launch_distributed.sh         # Launch script for distributed training
├── test_setup.py                 # Setup validation script
├── requirements.txt
└── README.md
```

## 🔧 Advanced Features

### Flash Attention Support

The model supports Flash Attention 2 for faster training:

```python
# In train.py or distributed_training.py
use_flash_attn = True  # Set to True to enable
if use_flash_attn:
    model = GPT_FLASH(config, "cuda")
else:
    model = GPT(config, "cuda")
```

### DeepSpeed ZeRO Optimization

**ZeRO Stage 1:** Optimizer state partitioning
- Moderate memory savings
- May cause OOM on 24GB GPUs

**ZeRO Stage 2:** Optimizer + Gradient partitioning (Recommended)
- Good balance of memory and speed
- Works well with NODE topology (though slow)

**ZeRO Stage 3:** Optimizer + Gradient + Parameter partitioning
- Maximum memory savings
- Slower but enables larger models

Configure in `ds-config.json`:
```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": false,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000
  }
}
```

### Checkpoint Management

Checkpoints are automatically saved every 50,000 steps (configurable):

**Single GPU:**
```python
# Checkpoints include:
# - Model weights
# - Optimizer state
# - Scheduler state
# - Training metadata (step, loss, etc.)
```

**DeepSpeed (Distributed):**
```bash
# Saves to checkpoint directory with DeepSpeed format
# Includes ZeRO optimizer states partitioned across GPUs
# Use model_engine.load_checkpoint() to resume
```

### Weights & Biases Logging

The training script automatically logs to W&B:
- Training loss and perplexity
- Validation loss
- Learning rate schedule
- Gradient norms (single GPU only)
- GPU memory usage
- Tokens per second
- Model configuration

## 🐛 Known Issues & Solutions

### Training Instability
- ✅ **Fixed**: Loss logging bug (incorrect gradient accumulation scaling)
- ✅ **Fixed**: Data pipeline context length mismatch
- ✅ **Fixed**: Dtype mismatch in attention output

### Memory Issues
- Use ZeRO Stage 2 or 3 for distributed training
- Reduce batch size if OOM (`train_micro_batch_size_per_gpu: 4`)
- Enable Flash Attention for 40% memory reduction
- Increase gradient accumulation to maintain effective batch size

### Distributed Training Issues

**1. NCCL Timeout Errors**
```
[Rank 0] Watchdog caught collective operation timeout
```

**Solution:** Poor GPU P2P topology (NODE/SYS). Options:
- **Best:** Use single GPU training (12 iter/sec vs 0.01 iter/sec!)
- Increase timeout: `export NCCL_TIMEOUT=3600`
- Reduce bucket sizes in `ds-config.json`

**2. Multi-GPU Slower Than Single GPU**

**Cause:** NODE or SYS topology forces all gradient syncs through CPU

**Solution:**
- Check topology: `nvidia-smi topo -m`
- If NODE/SYS: **Use single GPU** (much faster!)
- If on cloud: Look for instances with NVLink/PIX topology

**3. NCCL P2P Disabled Warning**
```
NCCL INFO P2P is disabled between connected GPUs
```

**Expected with NODE topology.** Configure NCCL:
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
```

## 📊 Performance Benchmarks

### Single GPU Performance
| GPU | Speed | Memory | Recommendation |
|-----|-------|--------|----------------|
| RTX 4090 | 12 iter/sec | ~12GB | ✅ Best for development |
| RTX 5090 | 14 iter/sec | ~11GB | ✅ Excellent |

### Multi-GPU Performance
| Configuration | Topology | Speed | Recommendation |
|--------------|----------|-------|----------------|
| 2x RTX 4090 | PIX | ~20 iter/sec | ✅ Good speedup |
| 2x RTX 4090 | NVLink | ~22 iter/sec | ✅ Excellent |
| 2x RTX 4090 | NODE | 0.01 iter/sec | ❌ **Use 1 GPU instead** |

**Key Takeaway:** Multi-GPU only helps with good P2P topology (PIX/PXB/NVLink). With NODE topology, single GPU is ~1200x faster!

## 📊 Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease smoothly from ~10 to ~4 (60M) or ~3 (800M)
2. **Validation Loss**: Should track training loss closely
3. **Gradient Norm**: Should stay below 5.0 (clipped at 1.0)
4. **Learning Rate**: Should follow cosine schedule
5. **Iterations/sec**: ~12 on single RTX 4090

### Red Flags
- ⚠️ Loss becomes NaN → Reduce learning rate
- ⚠️ Loss increases for >5K steps → Check data pipeline
- ⚠️ Grad norm consistently >10 → Gradient explosion, reduce LR
- ⚠️ Val loss >> train loss → Overfitting, add regularization
- ⚠️ Multi-GPU < 1 iter/sec → Bad topology, use single GPU

## 🔬 Model Variants

### Standard Attention (`model.py`)
- Traditional scaled dot-product attention
- Lower memory but slower
- Better for debugging
- Used in `train.py` and `distributed_training.py`

### Flash Attention (`model_flash_attn.py`)
- Flash Attention 2 implementation
- 40% faster training
- 50% memory reduction
- Requires `flash-attn` package

## 💻 Hardware Requirements

### Minimum (Single GPU)
- 1x RTX 4090 (24GB VRAM)
- 32GB RAM
- 100GB storage

### Recommended (Multi-GPU)
- 2-4x RTX 4090/5090 with **PIX or NVLink** topology
- 64GB+ RAM
- 500GB NVMe SSD

### Cloud Provider Tips
- **Vast.ai:** Filter for "NVLink" in search
- **Always check topology first:** `nvidia-smi topo -m`
- If NODE topology: Destroy and find better hardware
- Look for "PCIe 4.0 x16" with modern CPUs (AMD EPYC 7xxx series)

## 🛠️ Troubleshooting Guide

### Before Starting Multi-GPU Training

1. **Check GPU topology:**
   ```bash
   nvidia-smi topo -m
   ```
   - ✅ PIX/PXB/NVLink → Good to go
   - ❌ NODE/SYS → Use single GPU instead

2. **Test single GPU first:**
   ```bash
   python -m src.scripts.train
   ```
   Should get ~12 iter/sec on RTX 4090

3. **Verify NCCL:**
   ```bash
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

### Common Errors

**"Duplicate keys in DeepSpeed config"**
- Check `ds-config.json` for duplicate parameters
- Each parameter should appear only once

**"destroy_process_group() was not called"**
- Fixed in latest version
- Training script now properly cleans up on exit

**OOM with ZeRO Stage 1**
- Use ZeRO Stage 2 instead
- Reduce `train_micro_batch_size_per_gpu` to 4

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{project828,
  author = {AkshithAI},
  title = {Project 828: MoE Transformer for Code Generation with DeepSpeed},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AkshithAI/project_828}
}
```

## 📄 License

[tbd]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [@AkshithAI](https://github.com/AkshithAI).

---

**Note**: This is a research project. The 60M model is for testing pipeline stability, not for production code generation. The 800M model is the target configuration for practical applications.

**Important**: For multi-GPU training, always check GPU topology first. With NODE topology, single GPU training is significantly faster (~1200x speedup).
