# Project 828 - MoE Transformer for Code Generation

A Mixture-of-Experts (MoE) transformer model implementation for code generation, trained on the CodeParrot dataset. This project features a custom GPT-style architecture with advanced attention mechanisms and efficient expert routing.

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
- **Head Dimension**: Computed as `hidden_dim / num_attn_heads`
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
- CUDA-capable GPU (tested on RTX 5090)
- 16GB+ GPU memory recommended

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

#### Basic Training (60M model, 100K steps)

```bash
# Set your Weights & Biases credentials
export WANDB_API_KEY="your_wandb_key"

# Start training
python -m src.scripts.train
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

Edit `src/scripts/train.py` for training hyperparameters:

```python
# Training settings
grad_accumulation_step = 16  # Effective batch size = batch_size * grad_accum
num_warmup_steps = 2000      # 2% of total steps
num_training_steps = 100000  # Total training steps
```

### Training Hyperparameters

**Recommended for 60M model (100K steps):**
- **Learning Rate**: 3e-4
- **Warmup Steps**: 2,000 (2% of total)
- **Batch Size**: 8 sequences/GPU
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 128 sequences (~262K tokens)
- **Optimizer**: AdamW (β1=0.9, β2=0.95, weight_decay=0.01)
- **LR Schedule**: Cosine with warmup
- **Gradient Clipping**: 1.0
- **Precision**: BFloat16 mixed precision

**Expected Training Time:**
- 60M model: ~24-48 hours for 100K steps (RTX 5090)
- 800M model: ~7-14 days for 500K steps (8x A100)

## 📁 Project Structure

```
project_828/
├── src/
│   ├── models/
│   │   ├── model.py              # Main GPT model with standard attention
│   │   ├── model_flash_attn.py   # GPT model with Flash Attention
│   │   └── weight_init.py        # Model weight initialization
│   └── scripts/
│       ├── train.py              # Training loop
│       ├── configs.py            # Model configuration
│       ├── dataloader.py         # Data loading and preprocessing
│       ├── tokenizer.py          # Tokenizer setup
│       └── helper_funcs.py       # Utility functions
├── requirements.txt
└── README.md
```

## 🔧 Advanced Features

### Flash Attention Support

The model supports Flash Attention 2 for faster training:

```python
# In train.py
use_flash_attn = True  # Set to True to enable
if use_flash_attn:
    model = GPT_FLASH(config, "cuda")
else:
    model = GPT(config, "cuda")
```

### Checkpoint Management

Checkpoints are automatically saved every 50,000 steps (configurable):

```python
# Checkpoints include:
# - Model weights
# - Optimizer state
# - Scheduler state
# - Training metadata (step, loss, etc.)
```

### Weights & Biases Logging

The training script automatically logs to W&B:
- Training loss and perplexity
- Validation loss
- Learning rate schedule
- Gradient norms
- Model configuration


## 🐛 Known Issues & Solutions

### Training Instability
- ✅ **Fixed**: Loss logging bug (incorrect gradient accumulation scaling)
- ✅ **Fixed**: Data pipeline context length mismatch

### Memory Issues
- Use gradient checkpointing for larger models
- Reduce batch size if OOM
- Enable Flash Attention for 40% memory reduction

## 📊 Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease smoothly from ~10 to ~4 (60M) or ~3 (800M)
2. **Validation Loss**: Should track training loss closely
3. **Gradient Norm**: Should stay below 5.0 (clipped at 1.0)
4. **Learning Rate**: Should follow cosine schedule

### Red Flags
- ⚠️ Loss becomes NaN → Reduce learning rate
- ⚠️ Loss increases for >5K steps → Check data pipeline
- ⚠️ Grad norm consistently >10 → Gradient explosion, reduce LR
- ⚠️ Val loss >> train loss → Overfitting, add regularization

## 🔬 Model Variants

### Standard Attention (`model.py`)
- Traditional scaled dot-product attention
- Lower memory but slower
- Better for debugging

### Flash Attention (`model_flash_attn.py`)
- Flash Attention 2 implementation
- 40% faster training
- 50% memory reduction
- Requires `flash-attn` package


## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{project828,
  author = {AkshithAI},
  title = {Project 828: MoE Transformer for Code Generation},
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
