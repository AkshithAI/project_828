import os
import argparse
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from huggingface_hub import HfApi, create_repo, ModelCardData
from src.scripts.tokenizer import tokenizer_v1
from src.models.model_flash_attn import GPT_FLASH

try:
    from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, AutoConfig, AutoModelForCausalLM
except ImportError:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.utils import GenerationMixin
    from transformers.models.auto import AutoConfig, AutoModelForCausalLM


class GPTConfig(PretrainedConfig):
    model_type = "custom_gpt"

    def __init__(
        self,
        vocab_size: int = tokenizer_v1.vocab_size,
        num_attn_heads: int = 12,
        num_key_value_heads: int = 6,
        hidden_dim: int = 768,
        intermediate_size: int = 760,
        ffn_dropout: float = 0.0,
        num_hidden_layers: int = 24,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        update_param: float = 2e-3,
        route_scale: float = 1.0,
        base: int = 10000,
        initial_context_len: int = 2048,
        max_context_len: int = 2048,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        scaling_factor: float = 1.0,
        pad_token_id: int = tokenizer_v1.pad_token_id,
        bos_token_id: int = tokenizer_v1.bos_token_id,
        eos_token_id: int = tokenizer_v1.eos_token_id,
        tie_word_embeddings: bool = False,
        is_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=is_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.num_attn_heads = num_attn_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.ffn_dropout = ffn_dropout
        self.head_dim = hidden_dim // num_attn_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.update_param = update_param
        self.route_scale = route_scale
        self.base = base
        self.initial_context_len = initial_context_len
        self.max_context_len = max_context_len
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.scaling_factor = scaling_factor
        self.auto_map = {
            "AutoConfig": "modeling_gpt.GPTConfig",
            "AutoModelForCausalLM": "modeling_gpt.GPTForCausalLM",
        }


def _prepare_mask(attention_mask: Optional[torch.Tensor], input_ids: torch.Tensor) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        if attention_mask.all():
            return None
        batch_size, seq_len = input_ids.shape
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
        key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        return causal.unsqueeze(0).unsqueeze(0) & key_mask
    if attention_mask.dtype in (torch.int64, torch.int32):
        return attention_mask.bool()
    return attention_mask


class GPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GPTConfig
    main_input_name = "input_ids"
    _supports_cache_class = False

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.model = GPT_FLASH(config)
        ignore_idx = config.pad_token_id if config.pad_token_id is not None else -100
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.post_init()

    def _init_weights(self, module):
        pass

    def tie_weights(self, **kwargs):
        return {}

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.model.unembedding

    def set_output_embeddings(self, value):
        self.model.unembedding = value

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        kwargs.pop("past_key_values", None)
        kwargs.pop("cache_position", None)
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if not getattr(self, '_rope_ready', False):
            self._init_rope(input_ids.device)
        mask = _prepare_mask(attention_mask, input_ids)
        logits = self.model(input_ids, attn_mask=mask)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

    def _init_rope(self, device):
        """Recompute RoPE cos/sin buffers on the correct device.
        Non-persistent buffers are not saved in safetensors and may be empty
        after loading with device_map (accelerate creates meta tensors during init)."""
        for layer in self.model.layers:
            rope = layer.attention.rope
            rope.device = device
            cos, sin = rope.compute_cos_sin(rope.cos.shape[0])
            rope.cos = cos.to(device)
            rope.sin = sin.to(device)
        self._rope_ready = True


# Register for local Auto classes
AutoConfig.register("custom_gpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, GPTForCausalLM)


def write_standalone_modeling_file(export_dir: str):
    """Reads model_flash_attn.py, transforms it for self-contained Hub loading,
    and writes modeling_gpt.py to export_dir.

    Uses AST parsing for reliable stripping of training-only code, then applies
    text-level transformations for imports, dtype removal, and config replacement.
    """
    import ast
    import re
    import textwrap

    # ── Read the source model file ────────────────────────────
    source_path = os.path.join(os.path.dirname(__file__), "src", "models", "model_flash_attn.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.split('\n')

    # ── AST analysis: find line ranges to remove ──────────────
    tree = ast.parse(source)
    remove_ranges = []  # list of (start_line, end_line) 1-indexed inclusive

    # Methods to strip from specific classes
    moe_strip = {'get_expert_utilization', 'get_wandb_metrics'}
    gpt_strip = {'step_qk_scale_anneal', 'get_attention_diagnostics', 'get_telemetry_diagnostics'}

    for node in ast.walk(tree):
        # Remove _make_softcap_score_mod function
        if isinstance(node, ast.FunctionDef) and node.name == '_make_softcap_score_mod':
            remove_ranges.append((node.lineno, node.end_lineno))

        # Remove training-only methods from classes
        if isinstance(node, ast.ClassDef):
            strip_set = set()
            if node.name == 'MoE':
                strip_set = moe_strip
            elif node.name == 'GPT_FLASH':
                strip_set = gpt_strip

            if strip_set:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in strip_set:
                        # Include any decorator lines above the def
                        start = item.lineno
                        for dec in item.decorator_list:
                            start = min(start, dec.lineno)
                        # Also grab comment lines immediately before (# ── ... ──)
                        check_line = start - 2  # 0-indexed
                        while check_line >= 0 and lines[check_line].strip().startswith('# ──'):
                            start = check_line + 1  # back to 1-indexed
                            check_line -= 1
                        remove_ranges.append((start, item.end_lineno))

    # Sort and merge overlapping ranges
    remove_ranges.sort()

    # Build set of lines to remove (1-indexed)
    remove_lines = set()
    for start, end in remove_ranges:
        for i in range(start, end + 1):
            remove_lines.add(i)

    # ── Find import block end ─────────────────────────────────
    import_block_end = 0  # 0-indexed exclusive
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Try, ast.If)):
            # Check if this is part of the import block (at top level, before classes/functions)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_block_end = max(import_block_end, node.end_lineno)
            elif isinstance(node, ast.Try):
                # try/except import blocks
                is_import_try = all(
                    isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.If))
                    for handler in node.handlers
                    for stmt in handler.body
                )
                if is_import_try:
                    import_block_end = max(import_block_end, node.end_lineno)
            elif isinstance(node, ast.If):
                # if flash_attn_func is not None: ...
                import_block_end = max(import_block_end, node.end_lineno)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            break  # Past the import block
        elif isinstance(node, ast.Assign):
            # _FLEX_AVAILABLE = True/False
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '_FLEX_AVAILABLE':
                    import_block_end = max(import_block_end, node.end_lineno)

    # ── Build filtered body (skip imports + removed methods) ──
    kept_lines = []
    for i, line in enumerate(lines):
        lineno = i + 1  # 1-indexed
        if lineno <= import_block_end:
            continue
        if lineno in remove_lines:
            continue
        kept_lines.append(line)

    body = '\n'.join(kept_lines)

    # ── Text transformations ──────────────────────────────────

    # Replace ModelConfig → GPTConfig
    body = body.replace('ModelConfig', 'GPTConfig')

    # Remove dtype=config.dtype and dtype=torch.float32 from nn.Linear/nn.Embedding
    body = re.sub(r',\s*dtype\s*=\s*config\.dtype', '', body)
    body = re.sub(r',\s*dtype\s*=\s*torch\.float32', '', body)

    # Remove the dtype param from RotaryEmbedding.__init__ signature
    body = re.sub(r',\s*\n\s*dtype\s*:\s*torch\.dtype,', ',', body)

    # Remove torch.float32 positional arg in Attention's RotaryEmbedding() call
    body = re.sub(
        r'(self\.rope\s*=\s*RotaryEmbedding\(\s*\n\s*config\.head_dim,\s*\n\s*config\.base,)\s*\n\s*torch\.float32,',
        r'\1',
        body
    )

    # Remove assert end_pos <= self.max_cache_len block (multi-line)
    body = re.sub(
        r' *assert end_pos <= self\.max_cache_len,\s*\(\s*\n(?:\s*f".*\n)*\s*\)\s*\n',
        '',
        body
    )

    # Remove self.last_routing_probs attribute and assignments
    body = re.sub(r' *self\.last_routing_probs[^\n]*\n', '', body)

    # Simplify Gate.forward: remove original_scores indirection
    body = re.sub(r' *original_scores = scores\n', '', body)
    body = body.replace('weights = original_scores.gather(1, indices)', 'weights = scores.gather(1, indices)')

    # Collapse two-line weight normalization into one line
    body = re.sub(
        r'( *)weights /= weights\.sum\(dim=-1, keepdim=True\) *\n\s*weights = weights \* self\.route_scale',
        r'\1weights = (weights / weights.sum(dim=-1, keepdim=True)) * self.route_scale',
        body
    )

    # Remove dtype=torch.float32 from register_buffer bias (already done by general dtype removal, but catch leftovers)
    body = re.sub(r',\s*dtype=torch\.float32', '', body)

    # Replace flash_attn else branch with SDPA
    flash_block = re.compile(
        r'( +)else:\n'
        r'\s*# ── Training:.*\n'
        r'\s*attn_out = flash_attn_func\(\n'
        r'(?:.*\n)*?'
        r'\s*\)\n',
        re.MULTILINE
    )
    sdpa_else = (
        '{indent}else:\n'
        '{indent}    Q = Q.transpose(1, 2)\n'
        '{indent}    K = K.transpose(1, 2)\n'
        '{indent}    V = V.transpose(1, 2)\n'
        '{indent}    is_causal = attn_mask is None\n'
        '{indent}    attn_out = F.scaled_dot_product_attention(\n'
        '{indent}        Q, K, V,\n'
        '{indent}        attn_mask=attn_mask,\n'
        '{indent}        is_causal=is_causal,\n'
        '{indent}        enable_gqa=(self.n_heads != self.n_kv_heads),\n'
        '{indent}    )\n'
        '{indent}    attn_out = attn_out.transpose(1, 2)\n'
    )
    body = flash_block.sub(lambda m: sdpa_else.format(indent=m.group(1)), body)

    # ── Assemble final file ───────────────────────────────────
    hub_imports = textwrap.dedent("""\
        import math
        from typing import Tuple, Optional
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        try:
            from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
        except ImportError:
            from transformers.configuration_utils import PretrainedConfig
            from transformers.modeling_utils import PreTrainedModel
            from transformers.generation.utils import GenerationMixin
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    """)

    gpt_config_class = textwrap.dedent("""\

        class GPTConfig(PretrainedConfig):
            model_type = "custom_gpt"

            def __init__(
                self,
                vocab_size=49152,
                num_attn_heads=12,
                num_key_value_heads=6,
                hidden_dim=768,
                intermediate_size=760,
                ffn_dropout=0.0,
                num_hidden_layers=24,
                num_experts=4,
                num_experts_per_tok=2,
                update_param=2e-3,
                route_scale=1.0,
                base=10000,
                initial_context_len=2048,
                max_context_len=2048,
                ntk_alpha=1.0,
                ntk_beta=32.0,
                scaling_factor=1.0,
                pad_token_id=4,
                bos_token_id=0,
                eos_token_id=0,
                tie_word_embeddings=False,
                is_decoder=True,
                **kwargs,
            ):
                super().__init__(
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    tie_word_embeddings=tie_word_embeddings,
                    is_decoder=is_decoder,
                    **kwargs,
                )
                self.vocab_size = vocab_size
                self.num_attn_heads = num_attn_heads
                self.num_key_value_heads = num_key_value_heads
                self.hidden_dim = hidden_dim
                self.intermediate_size = intermediate_size
                self.ffn_dropout = ffn_dropout
                self.head_dim = hidden_dim // num_attn_heads
                self.num_hidden_layers = num_hidden_layers
                self.num_experts = num_experts
                self.num_experts_per_tok = num_experts_per_tok
                self.update_param = update_param
                self.route_scale = route_scale
                self.base = base
                self.initial_context_len = initial_context_len
                self.max_context_len = max_context_len
                self.ntk_alpha = ntk_alpha
                self.ntk_beta = ntk_beta
                self.scaling_factor = scaling_factor
                self.auto_map = {
                    "AutoConfig": "modeling_gpt.GPTConfig",
                    "AutoModelForCausalLM": "modeling_gpt.GPTForCausalLM",
                }

    """)

    hf_wrapper = textwrap.dedent("""\


        def _prepare_mask(attention_mask, input_ids):
            if attention_mask is None:
                return None
            if attention_mask.dim() == 2:
                if attention_mask.all():
                    return None
                batch_size, seq_len = input_ids.shape
                causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
                key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
                return causal.unsqueeze(0).unsqueeze(0) & key_mask
            if attention_mask.dtype in (torch.int64, torch.int32):
                return attention_mask.bool()
            return attention_mask


        class GPTForCausalLM(PreTrainedModel, GenerationMixin):
            config_class = GPTConfig
            main_input_name = "input_ids"
            _supports_cache_class = False

            @classmethod
            def _supports_default_dynamic_cache(cls):
                return False

            def __init__(self, config):
                super().__init__(config)
                self.model = GPT_FLASH(config)
                ignore_idx = config.pad_token_id if config.pad_token_id is not None else -100
                self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
                self.post_init()

            def _init_weights(self, module):
                pass

            def tie_weights(self, **kwargs):
                return {}

            def get_input_embeddings(self):
                return self.model.embeddings

            def set_input_embeddings(self, value):
                self.model.embeddings = value

            def get_output_embeddings(self):
                return self.model.unembedding

            def set_output_embeddings(self, value):
                self.model.unembedding = value

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                kwargs.pop("past_key_values", None)
                kwargs.pop("cache_position", None)
                return {
                    "input_ids": input_ids,
                    "attention_mask": kwargs.get("attention_mask", None),
                }

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                if not getattr(self, '_rope_ready', False):
                    self._init_rope(input_ids.device)
                mask = _prepare_mask(attention_mask, input_ids)
                logits = self.model(input_ids, attn_mask=mask)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

            def _init_rope(self, device):
                # Recompute RoPE cos/sin buffers on the correct device.
                # Non-persistent buffers are not in safetensors and may be empty
                # after loading with device_map (accelerate creates meta tensors during init).
                for layer in self.model.layers:
                    rope = layer.attention.rope
                    rope.device = device
                    cos, sin = rope.compute_cos_sin(rope.cos.shape[0])
                    rope.cos = cos.to(device)
                    rope.sin = sin.to(device)
                self._rope_ready = True
    """)

    code = hub_imports + "\n" + gpt_config_class + body + hf_wrapper

    # Clean up excessive blank lines
    code = re.sub(r'\n{4,}', '\n\n\n', code)

    with open(os.path.join(export_dir, "modeling_gpt.py"), "w", encoding="utf-8") as f:
        f.write(code)


def create_model_card(save_path: str, repo_id: str, model_config: GPTConfig) -> None:
    """Generate a standard Hugging Face model card README.md with YAML metadata."""
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        library_name="transformers",
        tags=["code", "moe", "mixture-of-experts", "flash-attention", "custom-architecture"],
        model_name=repo_id.split("/")[-1] if "/" in repo_id else repo_id,
        pipeline_tag="text-generation",
    )

    content = f"""---
{card_data.to_yaml()}
---

# {card_data.model_name}

A **398.7M parameter Mixture-of-Experts (MoE)** causal language model optimized for code generation and technical reasoning, pretrained on ~60B tokens.

> **398.7M total parameters, ~286M active per token** — 4 routed experts + 1 shared expert with top-2 routing per token.

## Model Architecture

| Property | Value |
|:---|:---|
| **Architecture** | Decoder-only Transformer with MoE FFN layers |
| **Total Parameters** | **398.7M** |
| **Active Parameters / Token** | **~286M** |
| **Hidden Dimension ($d_{{\\text{{model}}}}$)** | {model_config.hidden_dim} |
| **Intermediate Size ($d_{{\\text{{ff}}}}$)** | {model_config.intermediate_size} |
| **Hidden Layers** | {model_config.num_hidden_layers} |
| **Attention Heads / KV Heads** | {model_config.num_attn_heads} / {model_config.num_key_value_heads} (GQA 2:1) |
| **Head Dimension** | {model_config.hidden_dim // model_config.num_attn_heads} |
| **Routed Experts** | {model_config.num_experts} |
| **Active Experts / Token** | {model_config.num_experts_per_tok} (Top-{model_config.num_experts_per_tok}) |
| **Shared Experts** | 1 |
| **Context Length** | {model_config.max_context_len} (extensible to 8192 via YaRN) |
| **Vocabulary Size** | {model_config.vocab_size} |
| **Precision** | BFloat16 Mixed Precision |
| **Positional Encoding** | RoPE with YaRN scaling support |

### Parameter Breakdown

| Component | Parameters |
|:---|---:|
| Embeddings | ~37.7M |
| Unembedding | ~37.8M |
| Attention (×{model_config.num_hidden_layers} layers) | ~1.8M each |
| MoE FFN (×{model_config.num_hidden_layers} layers) | ~11.7M each |
| Layer Norms + Misc | ~0.07M |
| **Total** | **398.7M** |

### Key Architectural Features

- **Grouped Query Attention (GQA)**: {model_config.num_attn_heads} query heads with {model_config.num_key_value_heads} key-value heads (2:1 ratio) for memory-efficient attention
- **QK-Norm**: RMSNorm applied to query and key projections before RoPE for attention stability
- **Auxiliary-Loss-Free MoE Routing**: Sigmoid gating with dynamic bias adjustment ([DeepSeek-V3 paper](https://arxiv.org/abs/2408.15664)) — achieves near-perfect ~25% utilization per expert without auxiliary losses
- **SwiGLU Activation with Soft-Clamping**: `limit=7.0` prevents activation explosions during long training runs
- **Batched Expert Dispatch**: Sort-and-slice dispatch with `searchsorted` boundaries for contiguous memory access
- **RoPE with YaRN Extension**: Base context of {model_config.initial_context_len} tokens, extensible to 8192 via YaRN scaling

## Training Details

### Phase 1 — Pretraining (~60B tokens)

| Training Config | Value |
|:---|:---|
| **Hardware** | H200 GPU |
| **Peak Learning Rate** | 3e-4 |
| **Min Learning Rate** | 3e-5 |
| **Scheduler** | WSD (Warmup-Stable-Decay) |
| **Warmup Steps** | 500 |
| **Total Steps** | 101,726 |
| **Effective Batch Size** | 37 × 8 = 296 sequences |
| **Tokens per Step** | ~0.61M |
| **Gradient Clipping** | 1.0 |

### Training Data Mix

| Dataset | Weight | Category |
|:---|:---:|:---|
| `starcoderdata` — Python | 14 | Source Code |
| `starcoderdata` — JavaScript | 8 | Source Code |
| `starcoderdata` — Java | 6 | Source Code |
| `starcoderdata` — TypeScript | 4 | Source Code |
| `starcoderdata` — C++ | 6 | Source Code |
| `starcoderdata` — C | 4 | Source Code |
| `starcoderdata` — C# | 3 | Source Code |
| `starcoderdata` — Go | 4 | Source Code |
| `starcoderdata` — Rust | 3 | Source Code |
| `starcoderdata` — PHP | 3 | Source Code |
| `fineweb-edu-dedup` | 20 | General Knowledge |
| `cosmopedia-v2` | 7 | General Knowledge |
| `wikipedia-en` | 3 | General Knowledge |
| `finemath-4plus` | 8 | Math / Reasoning |
| `stackexchange` (programming/CS) | 7 | CS / Engineering |

**Category breakdown**: Source Code 55% · General Knowledge 30% · Math/Reasoning 8% · CS/Engineering 7%

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True
)

prompt = "def binary_search(arr, target):"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    use_cache=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{{project828,
  author = {{AkshithAI}},
  title = {{Project 828: MoE Transformer with Training Pipeline}},
  year = {{2025}},
  publisher = {{GitHub}},
  url = {{https://github.com/AkshithAI/project_828}}
}}
```
"""
    with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(content)


def export_and_upload(
    repo_id: str,
    checkpoint_path: str | None = None,
    export_dir: str = "./hf_export",
    push: bool = False,
    private: bool = False,
):
    """
    1. Instantiates the HF model & config using GPT_FLASH.
    2. Loads weights from checkpoint (if provided).
    3. Saves model, tokenizer, standalone modeling file, and Model Card to `export_dir`.
    4. Uploads only relevant files to Hugging Face Hub (if push=True).
    """
    os.makedirs(export_dir, exist_ok=True)
    print(f"[1/4] Initializing model & configuration...")
    config = GPTConfig()
    model = GPTForCausalLM(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[2/4] Loading checkpoint weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if any(k.startswith("layers.") or k.startswith("embeddings.") for k in state_dict.keys()):
            model.model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        print("[2/4] No checkpoint specified. Using initialized weights.")

    print(f"[3/4] Saving model, tokenizer, generation config, modeling file, and model card to: {export_dir}")
    model.save_pretrained(export_dir)
    tokenizer_v1.save_pretrained(export_dir)
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        max_new_tokens=200,
        use_cache=False,
    )
    gen_config.save_pretrained(export_dir)
    write_standalone_modeling_file(export_dir)
    create_model_card(save_path=export_dir, repo_id=repo_id, model_config=config)

    if push:
        print(f"[4/4] Uploading to Hugging Face Hub ({repo_id})...")
        api = HfApi()
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        api.upload_folder(
            folder_path=export_dir,
            repo_id=repo_id,
            repo_type="model",
            # Only upload specific files
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "*.json",
                "*.py",
                "README.md",
                "tokenizer*",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
            # Exclude temp files and internal artifacts
            ignore_patterns=[
                "__pycache__/*",
                "*.pt",
                "*.tmp",
                ".DS_Store",
            ],
            commit_message="Upload custom model, tokenizer, and model card",
        )
        print(f"Successfully published model to: https://huggingface.co/{repo_id}")
    else:
        print(f"Export complete in '{export_dir}'. Pass push=True or `--push` to upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and push custom model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="your-username/project-828-gpt-base", help="HF repository ID (e.g. username/model-name)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--export_dir", type=str, default="./hf_export", help="Local directory to export files")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()
    export_and_upload(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint,
        export_dir=args.export_dir,
        push=args.push,
        private=args.private,
    )
