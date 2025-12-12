import torch
import torch.nn as nn
import math


def init_weights(module: nn.Module, config=None):
    """
    Initialize model weights .
    
    Args:
        module: PyTorch module to initialize
        config: Optional ModelConfig object for context-aware initialization
    """
    if isinstance(module, nn.Linear):
        # Xavier/Glorot uniform initialization for linear layers
        std = 0.02
        if config is not None and hasattr(config, 'hidden_dim'):
            # Scale by 1/sqrt(hidden_dim) for better gradient flow
            std = 1.0 / math.sqrt(config.hidden_dim)
        
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    elif isinstance(module, nn.Parameter):
        # For standalone parameters (like sinks in attention)
        if module.dim() == 1:
            # For 1D parameters, initialize with small random values
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
        else:
            # For multi-dimensional parameters
            torch.nn.init.xavier_uniform_(module)


def init_model_weights(model, config=None):
    """
    Initialize all weights in a model.
    
    Args:
        model: PyTorch model (e.g., GPT)
        config: Optional ModelConfig object
    """
    # Apply initialization to all modules
    model.apply(lambda m: init_weights(m, config))
    
    # Special initialization for specific components
    for name, param in model.named_parameters():
        # Initialize output projection layers with smaller std for stability
        if 'wo.weight' in name or 'w2.weight' in name or 'unembedding.weight' in name:
            std = 0.02
            if config is not None and hasattr(config, 'hidden_dim'):
                std = 0.02 / math.sqrt(2 * config.num_hidden_layers)
            torch.nn.init.normal_(param, mean=0.0, std=std)
        
        # Initialize gate/routing parameters with small values
        if 'gate.weight' in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)
        
        if 'gate.bias' in name:
            torch.nn.init.zeros_(param)
        
        # Initialize sinks (attention parameters) with small values
        if 'sinks' in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    print(f"✓ Model weights initialized successfully")


def init_gpt_model(model, config):
    """
    Specialized initialization for GPT-style models.
    
    Args:
        model: GPT model instance
        config: ModelConfig object
    """
    print("Initializing GPT model weights...")
    
    # Initialize embeddings
    nn.init.normal_(model.embeddings.weight, mean=0.0, std=0.02)
    
    # Initialize transformer blocks
    for layer_idx, layer in enumerate(model.layers):
        # Attention layers
        if hasattr(layer, 'attention'):
            attn = layer.attention
            # Query, Key, Value projections
            nn.init.normal_(attn.wq.weight, mean=0.0, std=0.02)
            nn.init.normal_(attn.wk.weight, mean=0.0, std=0.02)
            nn.init.normal_(attn.wv.weight, mean=0.0, std=0.02)
            
            # Output projection with scaled initialization
            std = 0.02 / math.sqrt(2 * config.num_hidden_layers)
            nn.init.normal_(attn.wo.weight, mean=0.0, std=std)
            
            # Initialize biases to zero
            if attn.wq.bias is not None:
                nn.init.zeros_(attn.wq.bias)
            if attn.wk.bias is not None:
                nn.init.zeros_(attn.wk.bias)
            if attn.wv.bias is not None:
                nn.init.zeros_(attn.wv.bias)
            if attn.wo.bias is not None:
                nn.init.zeros_(attn.wo.bias)
            
            # Initialize attention sinks
            if hasattr(attn, 'sinks'):
                nn.init.normal_(attn.sinks, mean=0.0, std=0.02)
        
        # MLP/MoE layers
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            
            # Initialize gate (handle bfloat16 by using float32 temp then converting)
            if hasattr(mlp, 'gate'):
                with torch.no_grad():
                    # Initialize in float32 then copy to preserve dtype
                    temp_weight = torch.empty_like(mlp.gate.weight, dtype=torch.float32)
                    nn.init.normal_(temp_weight, mean=0.0, std=0.01)
                    mlp.gate.weight.copy_(temp_weight)
                    nn.init.zeros_(mlp.gate.bias)
            
            # Initialize experts
            if hasattr(mlp, 'experts'):
                for expert in mlp.experts:
                    nn.init.normal_(expert.w1.weight, mean=0.0, std=0.02)
                    nn.init.normal_(expert.w3.weight, mean=0.0, std=0.02)
                    std = 0.02 / math.sqrt(2 * config.num_hidden_layers)
                    nn.init.normal_(expert.w2.weight, mean=0.0, std=std)
                    
                    if expert.w1.bias is not None:
                        nn.init.zeros_(expert.w1.bias)
                    if expert.w2.bias is not None:
                        nn.init.zeros_(expert.w2.bias)
                    if expert.w3.bias is not None:
                        nn.init.zeros_(expert.w3.bias)
            
            # Initialize shared experts
            if hasattr(mlp, 'shared_experts'):
                shared = mlp.shared_experts
                nn.init.normal_(shared.w1.weight, mean=0.0, std=0.02)
                nn.init.normal_(shared.w3.weight, mean=0.0, std=0.02)
                std = 0.02 / math.sqrt(2 * config.num_hidden_layers)
                nn.init.normal_(shared.w2.weight, mean=0.0, std=std)
                
                if shared.w1.bias is not None:
                    nn.init.zeros_(shared.w1.bias)
                if shared.w2.bias is not None:
                    nn.init.zeros_(shared.w2.bias)
                if shared.w3.bias is not None:
                    nn.init.zeros_(shared.w3.bias)
        
        # Layer norms - keep scale at 1.0 (default)
        if hasattr(layer, 'norm1'):
            nn.init.ones_(layer.norm1.scale)
        if hasattr(layer, 'norm2'):
            nn.init.ones_(layer.norm2.scale)
    
    # Final layer norm
    if hasattr(model, 'norm'):
        nn.init.ones_(model.norm.scale)
    
    # Output/unembedding layer
    if hasattr(model, 'unembedding'):
        nn.init.normal_(model.unembedding.weight, mean=0.0, std=0.02)
        if model.unembedding.bias is not None:
            nn.init.zeros_(model.unembedding.bias)
    
    print(f"✓ GPT model weights initialized successfully")
    print(f"  - {config.num_hidden_layers} transformer layers")
    print(f"  - {config.num_experts} experts per MoE layer")
    print(f"  - Hidden dim: {config.hidden_dim}")


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params
