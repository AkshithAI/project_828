## Updated Project Structure
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
├── requirements.txt
└── README.md
```

## 🚀 Distributed Training with DeepSpeed

### Single GPU Training (Recommended for Development)

**Performance:** ~12 iterations/second on RTX 4090

```bash
# Set environment variables
export WANDB_API_KEY="your_wandb_key"
export PYTHONPATH=$(pwd)

# Run single GPU training
python -m src.scripts.train
```

### Multi-GPU Training with DeepSpeed

#### Prerequisites for Multi-GPU:
- ✅ Multiple CUDA-capable GPUs
- ✅ DeepSpeed installed (`pip install deepspeed`)
- ✅ **IMPORTANT:** Check GPU topology before training

#### Check GPU Topology:
```bash
nvidia-smi topo -m
```

**Good P2P Connection (Fast):**
- `PIX` - Single PCIe bridge (best)
- `PXB` - Multiple PCIe bridges (good)
- `NV#` - NVLink connection (excellent)

**Poor P2P Connection (Slow - NOT RECOMMENDED):**
- `NODE` - Cross NUMA node (very slow, ~7000x slower)
- `SYS` - Cross CPU socket (extremely slow)

⚠️ **If you see NODE or SYS topology, stick with single GPU training!**

#### Multi-GPU Training Commands:

**Basic 2-GPU Training:**
```bash
export WANDB_API_KEY="your_wandb_key"
export PYTHONPATH=$(pwd)

deepspeed --num_gpus=2 src/scripts/distributed_training.py \
    --deepspeed \
    --deepspeed_config src/scripts/ds-config.json \
    --batch_size 8
```

**Or use the launch script:**
```bash
bash launch_distributed.sh
```

### DeepSpeed Configuration (`ds-config.json`)

**ZeRO Stage 2 Configuration (Recommended):**
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 2,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "total_num_steps": 100000,
      "warmup_num_steps": 100,
      "warmup_min_ratio": 0.1,
      "warmup_type": "linear",
      "cos_min_ratio": 0.001
    }
  },
  
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": false,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000,
    "round_robin_gradients": false
  },
  
  "bf16": {
    "enabled": true
  },
  
  "wall_clock_breakdown": false
}
```

**ZeRO Optimization Stages:**
- **Stage 1**: Optimizer state partitioning (moderate memory saving)
- **Stage 2**: Optimizer + Gradient partitioning (recommended, good memory/speed trade-off)
- **Stage 3**: Optimizer + Gradient + Parameter partitioning (maximum memory saving)

⚠️ **Note:** ZeRO Stage 1 may cause OOM on smaller GPUs. Use Stage 2 for best results.

### Troubleshooting Distributed Training

#### 1. NCCL Timeout Errors
```
[Rank 0] Watchdog caught collective operation timeout
```

**Solution:** Your GPUs have slow P2P connection (NODE topology). Options:
- **Recommended:** Use single GPU training (faster!)
- Increase timeout: `export NCCL_TIMEOUT=3600`
- Reduce bucket sizes in `ds-config.json`
- Try different hardware with PIX/PXB topology

#### 2. NCCL P2P Disabled Warnings
```
NCCL INFO P2P is disabled between connected GPUs
```

**Solution:** This is expected with NODE topology. Configure NCCL for slow interconnects:
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_BLOCKING_WAIT=1
```

#### 3. OOM (Out of Memory) Errors

**Solutions:**
- Reduce `train_micro_batch_size_per_gpu` from 8 to 4
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable ZeRO Stage 3 for maximum memory efficiency
- Disable activation checkpointing if enabled

#### 4. Slow Training (< 1 iter/sec on multi-GPU)

**Likely cause:** Poor GPU topology (NODE/SYS)

**Solution:** 
- Check topology with `nvidia-smi topo -m`
- If NODE/SYS, **use single GPU** instead (~12 iter/sec vs ~0.01 iter/sec)
- Look for instances with NVLink or PIX topology on cloud providers

### Performance Comparison

| Configuration | Topology | Speed | Recommendation |
|--------------|----------|-------|----------------|
| 1x RTX 4090 | N/A | 12 iter/sec | ✅ Best for development |
| 2x RTX 4090 | PIX/PXB | ~20 iter/sec | ✅ Good speedup |
| 2x RTX 4090 | NVLink | ~22 iter/sec | ✅ Excellent |
| 2x RTX 4090 | NODE | 0.01 iter/sec | ❌ DO NOT USE |

### Hardware Requirements

**Minimum (Single GPU):**
- 1x RTX 4090 (24GB VRAM)
- 32GB RAM
- 100GB storage

**Recommended (Multi-GPU with good P2P):**
- 2-4x RTX 4090 with NVLink or PIX topology
- 64GB RAM
- 500GB NVMe SSD

**Cloud Provider Tips:**
- On Vast.ai: Filter for "NVLink" or check topology before renting
- Check `nvidia-smi topo -m` immediately after SSH
- If NODE topology, destroy instance and find better hardware
