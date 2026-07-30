import math
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Initialization configuration
# =============================================================================

@dataclass
class WeightInitializationConfig:
    """
    Initialization policy for GPT_FLASH.

    base_std:
        Used for input-side projections: Q/K/V, expert gate/up projections,
        shared-expert gate/up projection, embeddings, unembedding (unless tied).
    router_std:
        Smaller initialization for MoE router weights.
    residual_multiplier:
        Residual output projections use:
        residual_multiplier * base_std / sqrt(2 * num_layers)
    initialize_low_precision_from_fp32:
        Sample BF16/FP16 parameters through an FP32 temporary tensor before casting.
    """

    base_std: float = 0.02
    embedding_std: float = 0.02
    unembedding_std: float = 0.02
    router_std: float = 0.01
    residual_multiplier: float = 1.0
    norm_scale: float = 1.0
    attention_sink_std: float = 0.02

    initialize_low_precision_from_fp32: bool = True
    zero_padding_embedding: bool = True
    seed: int = 1234


# =============================================================================
# Low-level initialization utilities
# =============================================================================

@torch.no_grad()
def normal_init_(
    parameter: torch.Tensor,
    std: float,
    mean: float = 0.0,
    initialize_from_fp32: bool = True,
):
    """
    Initialize a parameter from a normal distribution.

    BF16 and FP16 parameters can be sampled through an FP32 temporary tensor
    and then cast to the destination dtype.
    """
    if parameter is None:
        return

    use_fp32_temporary = (
        initialize_from_fp32 and parameter.dtype in (torch.float16, torch.bfloat16)
    )

    if use_fp32_temporary:
        temporary = torch.empty(parameter.shape, device=parameter.device, dtype=torch.float32)
        nn.init.normal_(temporary, mean=mean, std=std)
        parameter.copy_(temporary.to(dtype=parameter.dtype))
        del temporary
    else:
        nn.init.normal_(parameter, mean=mean, std=std)


@torch.no_grad()
def zeros_init_(parameter: Optional[torch.Tensor]):
    if parameter is not None:
        parameter.zero_()


@torch.no_grad()
def ones_init_(parameter: Optional[torch.Tensor], value: float = 1.0):
    if parameter is not None:
        parameter.fill_(value)


def get_initialization_config(
    model_config,
    initialization_config: Optional[WeightInitializationConfig],
) -> WeightInitializationConfig:
    """
    Resolve initialization settings.

    Explicit WeightInitializationConfig values take priority. Otherwise,
    values are read from ModelConfig where available.
    """
    if initialization_config is not None:
        return initialization_config

    return WeightInitializationConfig(
        base_std=getattr(model_config, "initializer_std", 0.02),
        embedding_std=getattr(
            model_config, "embedding_initializer_std", getattr(model_config, "initializer_std", 0.02)
        ),
        unembedding_std=getattr(
            model_config, "unembedding_initializer_std", getattr(model_config, "initializer_std", 0.02)
        ),
        router_std=getattr(model_config, "router_initializer_std", 0.01),
        residual_multiplier=getattr(model_config, "residual_initializer_multiplier", 1.0),
        norm_scale=getattr(model_config, "norm_initial_scale", 1.0),
        attention_sink_std=getattr(model_config, "attention_sink_initializer_std", 0.02),
        initialize_low_precision_from_fp32=getattr(
            model_config, "initialize_low_precision_from_fp32", True
        ),
        zero_padding_embedding=getattr(model_config, "zero_padding_embedding", True),
        seed=getattr(model_config, "initialization_seed", 1234),
    )


# =============================================================================
# Initialization tracker
# =============================================================================

class ParameterInitializationTracker:
    """
    Prevent accidental double initialization of aliased parameters.

    This is particularly important when unembedding.weight is embeddings.weight
    under tied embeddings.
    """

    def __init__(self):
        self._initialized_parameter_ids: Set[int] = set()
        self.initialized_names: Dict[int, str] = {}

    def was_initialized(self, parameter: Optional[torch.Tensor]) -> bool:
        if parameter is None:
            return False
        return id(parameter) in self._initialized_parameter_ids

    def mark_initialized(self, name: str, parameter: Optional[torch.Tensor]):
        if parameter is None:
            return
        parameter_id = id(parameter)
        self._initialized_parameter_ids.add(parameter_id)
        self.initialized_names[parameter_id] = name

    def normal_(
        self,
        name: str,
        parameter: Optional[torch.Tensor],
        std: float,
        initialize_from_fp32: bool,
        force: bool = False,
    ):
        if parameter is None or (self.was_initialized(parameter) and not force):
            return
        normal_init_(parameter, std=std, initialize_from_fp32=initialize_from_fp32)
        self.mark_initialized(name, parameter)

    def zeros_(self, name: str, parameter: Optional[torch.Tensor], force: bool = False):
        if parameter is None or (self.was_initialized(parameter) and not force):
            return
        zeros_init_(parameter)
        self.mark_initialized(name, parameter)

    def ones_(
        self,
        name: str,
        parameter: Optional[torch.Tensor],
        value: float = 1.0,
        force: bool = False,
    ):
        if parameter is None or (self.was_initialized(parameter) and not force):
            return
        ones_init_(parameter, value)
        self.mark_initialized(name, parameter)


# =============================================================================
# Generic base initialization
# =============================================================================

@torch.no_grad()
def initialize_generic_modules(
    model: nn.Module,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    """
    Initialize all ordinary modules.

    Architecture-specific residual and router projections are overridden in a
    later pass.
    """
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            tracker.normal_(
                name=f"{module_name}.weight",
                parameter=module.weight,
                std=init_config.embedding_std,
                initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            )
            if init_config.zero_padding_embedding and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()

        elif isinstance(module, nn.Linear):
            tracker.normal_(
                name=f"{module_name}.weight",
                parameter=module.weight,
                std=init_config.base_std,
                initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            )
            if module.bias is not None:
                tracker.zeros_(name=f"{module_name}.bias", parameter=module.bias)

        # Supports RMS_Norm class without requiring a circular import.
        if hasattr(module, "scale"):
            scale = getattr(module, "scale")
            if isinstance(scale, nn.Parameter) and scale.ndim == 1:
                tracker.ones_(
                    name=f"{module_name}.scale",
                    parameter=scale,
                    value=init_config.norm_scale,
                )


# =============================================================================
# Architecture-specific initialization
# =============================================================================

@torch.no_grad()
def initialize_attention(
    attention: nn.Module,
    layer_idx: int,
    residual_std: float,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    """
    Initialize one attention module.

    Q/K/V are input-side projections and use base_std.
    WO is a residual output projection and uses residual_std.
    """
    prefix = f"layers.{layer_idx}.attention"

    for projection_name in ("wq", "wk", "wv"):
        projection = getattr(attention, projection_name, None)
        if projection is None:
            continue

        tracker.normal_(
            name=f"{prefix}.{projection_name}.weight",
            parameter=projection.weight,
            std=init_config.base_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )
        if projection.bias is not None:
            tracker.zeros_(
                name=f"{prefix}.{projection_name}.bias",
                parameter=projection.bias,
                force=True,
            )

    output_projection = getattr(attention, "wo", None)
    if output_projection is not None:
        tracker.normal_(
            name=f"{prefix}.wo.weight",
            parameter=output_projection.weight,
            std=residual_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )
        if output_projection.bias is not None:
            tracker.zeros_(
                name=f"{prefix}.wo.bias",
                parameter=output_projection.bias,
                force=True,
            )

    # Optional attention sinks.
    sinks = getattr(attention, "sinks", None)
    if isinstance(sinks, nn.Parameter):
        tracker.normal_(
            name=f"{prefix}.sinks",
            parameter=sinks,
            std=init_config.attention_sink_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )

    # QK RMSNorm scales.
    for norm_name in ("q_norm", "k_norm"):
        norm = getattr(attention, norm_name, None)
        if norm is not None and hasattr(norm, "scale"):
            tracker.ones_(
                name=f"{prefix}.{norm_name}.scale",
                parameter=norm.scale,
                value=init_config.norm_scale,
                force=True,
            )


@torch.no_grad()
def initialize_router(
    gate: nn.Module,
    layer_idx: int,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    """Initialize differentiable router and non-differentiable balancing buffers."""
    prefix = f"layers.{layer_idx}.mlp.gate"
    router = getattr(gate, "router", None)

    if router is not None:
        tracker.normal_(
            name=f"{prefix}.router.weight",
            parameter=router.weight,
            std=init_config.router_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )
        if router.bias is not None:
            tracker.zeros_(
                name=f"{prefix}.router.bias",
                parameter=router.bias,
                force=True,
            )

    # Buffers rather than optimizer parameters.
    for buffer_name in ("routing_bias", "load_accum", "last_mean_scores", "last_load"):
        buffer = getattr(gate, buffer_name, None)
        if isinstance(buffer, torch.Tensor):
            buffer.zero_()


@torch.no_grad()
def initialize_routed_experts(
    routed_experts: nn.Module,
    layer_idx: int,
    residual_std: float,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    """
    Initialize grouped Liger expert parameters.

    Expected layouts:
        gate_up_proj: [E, 2I, D]
        down_proj:    [E, D, I]
    """
    prefix = f"layers.{layer_idx}.mlp.routed_experts"
    gate_up_projection = getattr(routed_experts, "gate_up_proj", None)
    down_projection = getattr(routed_experts, "down_proj", None)

    if gate_up_projection is None or down_projection is None:
        return

    if gate_up_projection.ndim == 3 and down_projection.ndim == 3:
        tracker.normal_(
            name=f"{prefix}.gate_up_proj",
            parameter=gate_up_projection,
            std=init_config.base_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )
        tracker.normal_(
            name=f"{prefix}.down_proj",
            parameter=down_projection,
            std=residual_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )


@torch.no_grad()
def initialize_shared_expert(
    shared_expert: nn.Module,
    layer_idx: int,
    residual_std: float,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    """
    Initialize dense shared expert.

    w1, w3 are input projections.
    w2 is the residual output projection.
    """
    prefix = f"layers.{layer_idx}.mlp.shared_expert"
    for proj_name in ("w1", "w3"):
        proj = getattr(shared_expert, proj_name, None)
        if proj is not None and hasattr(proj, "weight"):
            tracker.normal_(
                name=f"{prefix}.{proj_name}.weight",
                parameter=proj.weight,
                std=init_config.base_std,
                initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
                force=True,
            )
            if proj.bias is not None:
                tracker.zeros_(
                    name=f"{prefix}.{proj_name}.bias",
                    parameter=proj.bias,
                    force=True,
                )

    w2 = getattr(shared_expert, "w2", None)
    if w2 is not None and hasattr(w2, "weight"):
        tracker.normal_(
            name=f"{prefix}.w2.weight",
            parameter=w2.weight,
            std=residual_std,
            initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
            force=True,
        )
        if w2.bias is not None:
            tracker.zeros_(
                name=f"{prefix}.w2.bias",
                parameter=w2.bias,
                force=True,
            )


@torch.no_grad()
def initialize_moe(
    moe: nn.Module,
    layer_idx: int,
    residual_std: float,
    init_config: WeightInitializationConfig,
    tracker: ParameterInitializationTracker,
):
    gate = getattr(moe, "gate", None)
    if gate is not None:
        initialize_router(
            gate=gate,
            layer_idx=layer_idx,
            init_config=init_config,
            tracker=tracker,
        )

    # Individual expert modules (ModuleList of Experts)
    experts = getattr(moe, "experts", None)
    if experts is not None:
        prefix = f"layers.{layer_idx}.mlp.experts"
        for i, expert in enumerate(experts):
            for proj_name in ("w1", "w3"):
                proj = getattr(expert, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    tracker.normal_(
                        name=f"{prefix}.{i}.{proj_name}.weight",
                        parameter=proj.weight,
                        std=init_config.base_std,
                        initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
                        force=True,
                    )
                    if proj.bias is not None:
                        tracker.zeros_(
                            name=f"{prefix}.{i}.{proj_name}.bias",
                            parameter=proj.bias,
                            force=True,
                        )
            w2 = getattr(expert, "w2", None)
            if w2 is not None and hasattr(w2, "weight"):
                tracker.normal_(
                    name=f"{prefix}.{i}.w2.weight",
                    parameter=w2.weight,
                    std=residual_std,
                    initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
                    force=True,
                )
                if w2.bias is not None:
                    tracker.zeros_(
                        name=f"{prefix}.{i}.w2.bias",
                        parameter=w2.bias,
                        force=True,
                    )

    # Grouped Liger expert parameters if present
    routed_experts = getattr(moe, "routed_experts", None)
    if routed_experts is not None:
        initialize_routed_experts(
            routed_experts=routed_experts,
            layer_idx=layer_idx,
            residual_std=residual_std,
            init_config=init_config,
            tracker=tracker,
        )

    # Shared expert (singular or plural attribute name)
    shared_expert = getattr(moe, "shared_expert", None) or getattr(moe, "shared_experts", None)
    if shared_expert is not None:
        initialize_shared_expert(
            shared_expert=shared_expert,
            layer_idx=layer_idx,
            residual_std=residual_std,
            init_config=init_config,
            tracker=tracker,
        )

    expert_counts = getattr(moe, "expert_counts", None)
    if isinstance(expert_counts, torch.Tensor):
        expert_counts.zero_()

    if hasattr(moe, "total_assignments"):
        moe.total_assignments = 0


# =============================================================================
# Main model initializer
# =============================================================================

@torch.no_grad()
def initialize_gpt_model(
    model: nn.Module,
    model_config,
    initialization_config: Optional[WeightInitializationConfig] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Initialize the revised GPT_FLASH architecture.

    Initialization policy
    ---------------------
    Embeddings:                 N(0, embedding_std²)
    Q/K/V projections:          N(0, base_std²)
    Attention output proj:      N(0, residual_std²)
    Routed expert gate/up:      N(0, base_std²)
    Routed expert down:         N(0, residual_std²)
    Shared expert gate/up:      N(0, base_std²)
    Shared expert down:         N(0, residual_std²)
    Router:                     N(0, router_std²)
    Unembedding:                N(0, unembedding_std²), unless tied to embeddings
    RMSNorm:                    Constant norm_scale

    where:
        residual_std = residual_multiplier * base_std / sqrt(2 * num_hidden_layers)
    """
    init_config = get_initialization_config(model_config, initialization_config)
    num_layers = getattr(model_config, "num_hidden_layers", 1)

    if num_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    residual_std = init_config.residual_multiplier * init_config.base_std / math.sqrt(2.0 * num_layers)

    # The model is assumed to reside on one device during initialization.
    first_parameter = next(model.parameters(), None)
    cuda_devices = []

    if first_parameter is not None and first_parameter.is_cuda:
        cuda_devices = [
            first_parameter.device.index
            if first_parameter.device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(init_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(init_config.seed)

        tracker = ParameterInitializationTracker()

        # Initialize ordinary modules first.
        initialize_generic_modules(model=model, init_config=init_config, tracker=tracker)

        # Explicit embedding initialization.
        if hasattr(model, "embeddings"):
            tracker.normal_(
                name="embeddings.weight",
                parameter=model.embeddings.weight,
                std=init_config.embedding_std,
                initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
                force=True,
            )
            if init_config.zero_padding_embedding and model.embeddings.padding_idx is not None:
                model.embeddings.weight[model.embeddings.padding_idx].zero_()

        # Transformer blocks.
        if hasattr(model, "layers"):
            for layer_idx, layer in enumerate(model.layers):
                attention = getattr(layer, "attention", None)
                moe = getattr(layer, "mlp", None)

                if attention is not None:
                    initialize_attention(
                        attention=attention,
                        layer_idx=layer_idx,
                        residual_std=residual_std,
                        init_config=init_config,
                        tracker=tracker,
                    )
                if moe is not None:
                    initialize_moe(
                        moe=moe,
                        layer_idx=layer_idx,
                        residual_std=residual_std,
                        init_config=init_config,
                        tracker=tracker,
                    )

                # Transformer hidden-state RMSNorms.
                for norm_name in ("norm1", "norm2"):
                    norm = getattr(layer, norm_name, None)
                    if norm is not None and hasattr(norm, "scale"):
                        tracker.ones_(
                            name=f"layers.{layer_idx}.{norm_name}.scale",
                            parameter=norm.scale,
                            value=init_config.norm_scale,
                            force=True,
                        )

        # Final RMSNorm.
        final_norm = getattr(model, "norm", None)
        if final_norm is not None and hasattr(final_norm, "scale"):
            tracker.ones_(
                name="norm.scale",
                parameter=final_norm.scale,
                value=init_config.norm_scale,
                force=True,
            )

        # Unembedding.
        if hasattr(model, "unembedding"):
            unembedding_weight = model.unembedding.weight
            embeddings_are_tied = hasattr(model, "embeddings") and unembedding_weight is model.embeddings.weight

            if not embeddings_are_tied:
                tracker.normal_(
                    name="unembedding.weight",
                    parameter=unembedding_weight,
                    std=init_config.unembedding_std,
                    initialize_from_fp32=init_config.initialize_low_precision_from_fp32,
                    force=True,
                )

            if model.unembedding.bias is not None:
                tracker.zeros_(
                    name="unembedding.bias",
                    parameter=model.unembedding.bias,
                    force=True,
                )

    report = {
        "base_std": init_config.base_std,
        "embedding_std": init_config.embedding_std,
        "unembedding_std": init_config.unembedding_std,
        "router_std": init_config.router_std,
        "residual_std": residual_std,
        "norm_scale": init_config.norm_scale,
        "seed": float(init_config.seed),
    }

    if verbose:
        tied = (
            hasattr(model, "embeddings")
            and hasattr(model, "unembedding")
            and model.embeddings.weight is model.unembedding.weight
        )

        hidden_dim = getattr(model_config, "hidden_dim", "N/A")
        num_experts = getattr(model_config, "num_experts", "N/A")
        num_experts_per_tok = getattr(model_config, "num_experts_per_tok", "N/A")

        print("✓ GPT model weights initialized")
        print(f"  Layers:                 {num_layers}")
        print(f"  Hidden dimension:       {hidden_dim}")
        print(f"  Experts/layer:          {num_experts}")
        print(f"  Active routed experts:  {num_experts_per_tok}")
        print(f"  Base std:               {init_config.base_std:.8f}")
        print(f"  Residual std:           {residual_std:.8f}")
        print(f"  Router std:             {init_config.router_std:.8f}")
        print(f"  Embeddings tied:        {tied}")
        print(f"  FP32 temporary init:    {init_config.initialize_low_precision_from_fp32}")
        print(f"  Initialization seed:    {init_config.seed}")

    return report


# Alias for backward compatibility / external imports
init_gpt_model = initialize_gpt_model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable