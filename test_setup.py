#!/usr/bin/env python3
"""
Test script to validate the distributed training setup
Run this before launching the full training
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("✓ Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        
        import deepspeed
        print(f"  ✓ DeepSpeed: {deepspeed.__version__}")
        
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
        
        import datasets
        print(f"  ✓ Datasets: {datasets.__version__}")
        
        import wandb
        print(f"  ✓ WandB: {wandb.__version__}")
        
        try:
            import flash_attn
            print(f"  ✓ Flash Attention: {flash_attn.__version__}")
        except ImportError:
            print(f"  ⚠️  Flash Attention: Not installed (optional)")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_config_files():
    """Test configuration files"""
    print("\n✓ Testing configuration files...")
    try:
        # Test ds-config.json
        import json
        config_path = "src/scripts/ds-config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                ds_config = json.load(f)
            print(f"  ✓ ds-config.json found")
            print(f"    - Batch size: {ds_config['train_batch_size']}")
            print(f"    - Micro batch: {ds_config['train_micro_batch_size_per_gpu']}")
            print(f"    - Grad accum: {ds_config['gradient_accumulation_steps']}")
            print(f"    - BF16: {ds_config['bf16']['enabled']}")
            print(f"    - ZeRO stage: {ds_config['zero_optimization']['stage']}")
        else:
            print(f"  ❌ ds-config.json not found!")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_model_imports():
    """Test project imports"""
    print("\n✓ Testing project imports...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.scripts.configs import config
        print(f"  ✓ Config imported")
        print(f"    - Hidden dim: {config.hidden_dim}")
        print(f"    - Num layers: {config.num_hidden_layers}")
        print(f"    - Seq len: {config.seq_len}")
        
        from src.scripts.tokenizer import tokenizer
        print(f"  ✓ Tokenizer imported")
        print(f"    - Vocab size: {tokenizer.vocab_size}")
        
        from src.models.model import GPT
        print(f"  ✓ GPT model imported")
        
        try:
            from src.models.model_flash_attn import GPT_FLASH
            print(f"  ✓ GPT_FLASH model imported")
        except Exception as e:
            print(f"  ⚠️  GPT_FLASH model: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading"""
    print("\n✓ Testing data loading...")
    try:
        from src.scripts.dataloader import get_train_files, CustomDataset
        print(f"  ✓ Data loader imported")
        
        # Test getting train files
        train_files = get_train_files()
        print(f"    - Found {len(train_files)} training files")
        
        return True
    except Exception as e:
        print(f"  ❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\n✓ Testing model creation...")
    try:
        import torch
        from src.models.model import GPT
        from src.scripts.configs import config
        
        # Create model
        model = GPT(config, "cpu")  # Use CPU for testing
        print(f"  ✓ Model created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    - Total params: {total_params:,}")
        print(f"    - Trainable params: {trainable_params:,}")
        print(f"    - Model size: ~{total_params * 2 / 1e6:.1f} MB (bf16)")
        
        # Test forward pass
        dummy_input = torch.randint(0, config.vocab_size, (2, 128))
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✓ Forward pass successful")
        print(f"    - Input shape: {dummy_input.shape}")
        print(f"    - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Project 828 - Distributed Training Setup Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config Files", test_config_files()))
    results.append(("Project Imports", test_model_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Creation", test_model_creation()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Ready to train.")
        print("\nTo start training:")
        print("  ./launch_distributed.sh 2")
    else:
        print("⚠️  Some tests failed. Please fix the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
