"""
Training Telemetry — Hidden Failure-Mode Detection
====================================================

Provides lightweight metric computation functions for monitoring 4 categories
of silent model degradation:

1. **Routing Entropy & Token Affinity** — Detects MoE router collapse
2. **Hidden State Dimensionality Collapse** — Detects representation collapse
3. **Weight-to-Update Norm Ratios** — Detects frozen or runaway layers
4. **Domain PPL Divergence** — Computed in validate_domains.py (see there)

All functions are designed to be called from the training loop with minimal
overhead. They read from cached state (routing probs, optimizer moments)
rather than running extra forward passes, except for hidden state telemetry
which requires a no-grad forward at val_interval cadence.

Usage in train.py:
    from .telemetry import (
        compute_routing_telemetry,
        compute_weight_update_ratios,
        compute_hidden_state_telemetry,
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

try:
    from flash_attn import flash_attn_func as _orig_flash_attn_func
except ImportError:
    _orig_flash_attn_func = None


def _safe_flash_attn_func(q, k, v, causal=False, softcap=None, **kwargs):
    if _orig_flash_attn_func is None:
        raise ImportError("flash_attn is not installed")
    try:
        if softcap is not None and softcap > 0:
            return _orig_flash_attn_func(q, k, v, causal=causal, softcap=softcap, **kwargs)
        else:
            return _orig_flash_attn_func(q, k, v, causal=causal, **kwargs)
    except TypeError as e:
        if "softcap" in str(e) or "unexpected keyword argument" in str(e):
            if softcap is not None and softcap > 0:
                # Emulate soft-capped attention manually (CPU tests/mocks)
                Q_t = q.transpose(1, 2)
                K_t = k.transpose(1, 2)
                V_t = v.transpose(1, 2)
                nq, nkv = Q_t.shape[1], K_t.shape[1]
                if nq != nkv:
                    K_t = K_t.repeat_interleave(nq // nkv, dim=1)
                    V_t = V_t.repeat_interleave(nq // nkv, dim=1)
                scale = 1.0 / math.sqrt(Q_t.shape[-1])
                scores = (Q_t @ K_t.transpose(-2, -1)) * scale
                scores = softcap * torch.tanh(scores / softcap)
                if causal:
                    seq_len = Q_t.shape[-2]
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                        diagonal=1
                    )
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_w = F.softmax(scores, dim=-1)
                return (attn_w @ V_t).transpose(1, 2)
            else:
                return _orig_flash_attn_func(q, k, v, causal=causal, **kwargs)
        raise e


# ═══════════════════════════════════════════════════════════════
# 1. MoE Routing Entropy & Router Weight Cosine Similarity
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_routing_telemetry(model: nn.Module) -> Dict[str, float]:
    """
    Compute per-layer routing entropy and router weight cosine similarity.

    Reads ``gate.last_routing_probs`` (cached during the training forward)
    and ``gate.router.weight`` (small matrix, E×D).

    Metrics returned:
        telemetry/routing_entropy/layer_{i}       — mean entropy of routing probs
        telemetry/routing_entropy_ratio/layer_{i}  — entropy / max_entropy
        telemetry/router_cos_sim/layer_{i}         — mean pairwise cosine sim of router rows
        telemetry/routing_entropy_min              — min entropy ratio across layers
        telemetry/router_cos_sim_max               — max cosine sim across layers

    Red flags:
        - routing_entropy_ratio dropping below 0.5 → hard gating / collapse
        - routing_entropy_ratio above 0.95 with perfect load balance → bypass collapse
        - router_cos_sim above 0.7 → expert projections converging
    """
    metrics: Dict[str, float] = {}
    entropy_ratios = []
    cos_sims = []
    num_experts = None

    for i, layer in enumerate(model.layers):
        gate = layer.mlp.gate
        num_experts = gate.num_experts
        max_entropy = math.log(num_experts)

        # ── Routing entropy from cached sigmoid probs ──
        probs = getattr(gate, "last_routing_probs", None)  # (N_tokens, E) or None
        if probs is not None:
            # Normalize sigmoid outputs to a probability distribution
            probs_float = probs.float()
            prob_dist = probs_float / probs_float.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            # Shannon entropy: H = -sum(p_i * log(p_i))
            ent = -(prob_dist * prob_dist.clamp(min=1e-12).log()).sum(dim=-1)  # (N,)
            mean_ent = ent.mean()  # keep as tensor

            # Bulk transfer: single .item() instead of keeping on GPU
            mean_ent_val = mean_ent.item()
            ent_ratio = mean_ent_val / max_entropy if max_entropy > 0 else 0.0

            metrics[f"telemetry/routing_entropy/layer_{i}"] = mean_ent_val
            metrics[f"telemetry/routing_entropy_ratio/layer_{i}"] = ent_ratio
            entropy_ratios.append(ent_ratio)

        # ── Router weight cosine similarity ──
        W = gate.router.weight.data.float()  # (E, D)
        W_norm = F.normalize(W, dim=1)
        cos_matrix = W_norm @ W_norm.T  # (E, E)
        # Mask diagonal (self-similarity = 1.0 always)
        mask = ~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
        off_diag = cos_matrix[mask]
        # Bulk transfer: 2 values at once via a stacked tensor
        cos_stats = torch.stack([off_diag.mean(), off_diag.max()]).cpu().tolist()
        mean_cos, max_cos = cos_stats[0], cos_stats[1]

        metrics[f"telemetry/router_cos_sim/layer_{i}"] = mean_cos
        metrics[f"telemetry/router_cos_sim_max/layer_{i}"] = max_cos
        cos_sims.append(mean_cos)

    # Global aggregates
    if entropy_ratios:
        metrics["telemetry/routing_entropy_min"] = min(entropy_ratios)
        metrics["telemetry/routing_entropy_mean"] = sum(entropy_ratios) / len(entropy_ratios)
    if cos_sims:
        metrics["telemetry/router_cos_sim_max"] = max(cos_sims)
        metrics["telemetry/router_cos_sim_mean"] = sum(cos_sims) / len(cos_sims)

    return metrics


# ═══════════════════════════════════════════════════════════════
# 2. Hidden State Dimensionality Collapse
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_hidden_state_telemetry(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Dict[str, float]:
    """
    Probe hidden state health at layers L//2 and L-2 via a no-grad forward.

    Runs a single forward pass through the unwrapped model with hooks to
    capture intermediate activations. Computes:
        - Average pairwise cosine similarity of token representations
        - Singular value spectrum ratio (SV_1 / SV_10) for rank collapse detection

    Metrics returned:
        telemetry/hidden_cos_sim/layer_{idx}   — avg cosine sim between tokens
        telemetry/hidden_sv_ratio/layer_{idx}   — SV_1 / SV_10 (or SV_last if <10)
        telemetry/hidden_sv_top1_frac/layer_{idx} — SV_1 / sum(SV) (energy concentration)

    Red flags:
        - hidden_cos_sim > 0.85 → tokens losing distinctness (dimensionality collapse)
        - hidden_sv_ratio > 100 → effective rank has collapsed to a few dimensions

    Note: This function runs an EXTRA no-grad forward pass on ``input_ids``.
          Call only at val_interval to avoid performance impact.
    """
    metrics: Dict[str, float] = {}
    n_layers = len(model.layers)

    # Probe points: middle layer and near-output layer
    probe_indices = sorted(set([n_layers // 2, max(n_layers - 2, 0)]))

    # Hook storage
    captured: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output is the residual stream tensor (B, S, D)
            captured[layer_idx] = output.detach().float()
        return hook_fn

    # Register hooks on the target decoder blocks
    handles = []
    for idx in probe_indices:
        h = model.layers[idx].register_forward_hook(_make_hook(idx))
        handles.append(h)

    try:
        # Run a no-grad forward pass (model must be in eval mode temporarily)
        was_training = model.training
        model.eval()
        model(input_ids)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    # Analyze captured activations
    for idx, hidden in captured.items():
        # hidden: (B, S, D) — flatten batch and sequence
        H = hidden.reshape(-1, hidden.shape[-1])  # (N, D)
        N, D = H.shape

        if N < 2:
            continue

        # ── Cosine similarity ──
        # Sample up to 512 tokens to keep SVD tractable
        max_tokens = min(N, 512)
        if N > max_tokens:
            perm = torch.randperm(N, device=H.device)[:max_tokens]
            H_sample = H[perm]
        else:
            H_sample = H

        H_normed = F.normalize(H_sample, dim=-1)
        cos_matrix = H_normed @ H_normed.T  # (max_tokens, max_tokens)
        # Mask diagonal
        n_sample = H_sample.shape[0]
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=H.device)
        avg_cos_sim = cos_matrix[mask].mean().item()

        metrics[f"telemetry/hidden_cos_sim/layer_{idx}"] = avg_cos_sim

        # ── Singular value spectrum ──
        # SVD on the sampled hidden states
        try:
            svs = torch.linalg.svdvals(H_sample)  # (min(N, D),)
            sv1 = svs[0].item()
            sv10_idx = min(9, len(svs) - 1)
            sv10 = svs[sv10_idx].item()
            sv_ratio = sv1 / max(sv10, 1e-12)
            sv_total = svs.sum().item()
            sv_top1_frac = sv1 / max(sv_total, 1e-12)

            metrics[f"telemetry/hidden_sv_ratio/layer_{idx}"] = sv_ratio
            metrics[f"telemetry/hidden_sv_top1_frac/layer_{idx}"] = sv_top1_frac
        except Exception:
            # SVD can fail on degenerate inputs; skip gracefully
            pass

    return metrics


# ═══════════════════════════════════════════════════════════════
# 3. Layer-wise Weight-to-Update Norm Ratios
# ═══════════════════════════════════════════════════════════════

# Weight category classification by parameter name substring
_WEIGHT_CATEGORIES = {
    "attn_qkvo": ["attention.wq.", "attention.wk.", "attention.wv.", "attention.wo."],
    "router_gate": [".gate.router."],
    "expert_down": [".experts.", ".w2."],
    "expert_up": [".experts.", ".w1.", ".w3."],
    "shared_expert": [".shared_experts."],
    "embeddings": ["embeddings.", "unembedding."],
    "norms": ["norm", ".scale"],
}


def _categorize_param(name: str) -> str:
    """Classify a parameter name into a weight category."""
    # Router gate is very specific — check first
    if ".gate.router." in name:
        return "router_gate"
    # Expert down-projection
    if ".experts." in name and ".w2." in name:
        return "expert_down"
    # Expert up-projection (w1, w3)
    if ".experts." in name and (".w1." in name or ".w3." in name):
        return "expert_up"
    # Shared expert
    if ".shared_experts." in name:
        return "shared_expert"
    # Attention projections
    if any(k in name for k in ["attention.wq.", "attention.wk.", "attention.wv.", "attention.wo."]):
        return "attn_qkvo"
    # Embeddings
    if "embeddings." in name or "unembedding." in name:
        return "embeddings"
    # Norms (q_norm, k_norm, layer norms)
    if "norm" in name or ".scale" in name:
        return "norms"
    return "other"


@torch.no_grad()
def compute_weight_update_ratios(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> Dict[str, float]:
    """
    Compute the effective update-to-weight ratio per weight category.

    For AdamW, the effective update for parameter p at step t is approximately:
        Δp ≈ lr × m_t / (√v_t + ε)
    where m_t = exp_avg (first moment), v_t = exp_avg_sq (second moment).

    We compute:
        ratio = ‖Δp‖ / ‖p‖ = lr × ‖m / (√v + ε)‖ / ‖p‖

    Then aggregate by weight category (mean ratio).

    Metrics returned:
        telemetry/update_ratio/{category}    — mean update/weight ratio
        telemetry/weight_norm/{category}     — mean weight norm
        telemetry/update_ratio_min           — min ratio across categories
        telemetry/update_ratio_max           — max ratio across categories

    Red flags:
        - ratio < 1e-5 → layer is "frozen" (weights too large for optimizer to move)
        - ratio > 1e-2 → layer is unstable (updates too large relative to weights)
    """
    metrics: Dict[str, float] = {}

    # Collect per-category norms
    category_update_norms: Dict[str, list] = {}
    category_weight_norms: Dict[str, list] = {}

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None and p not in optimizer.state:
                continue
            state = optimizer.state.get(p, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue  # No Adam state yet (e.g., step 0)

            # Find parameter name
            param_name = None
            for name, param in model.named_parameters():
                if param is p:
                    param_name = name
                    break
            if param_name is None:
                continue

            category = _categorize_param(param_name)

            # Compute effective update norm
            m = state["exp_avg"]       # first moment
            v = state["exp_avg_sq"]    # second moment
            eps = group.get("eps", 1e-8)

            # AdamW update: lr * m / (sqrt(v) + eps)
            update = m / (v.sqrt() + eps)
            update_norm = (lr * update.float().norm()).item()
            weight_norm = p.data.float().norm().item()

            if category not in category_update_norms:
                category_update_norms[category] = []
                category_weight_norms[category] = []
            category_update_norms[category].append(update_norm)
            category_weight_norms[category].append(weight_norm)

    # Aggregate per category
    all_ratios = []
    for cat in sorted(category_update_norms.keys()):
        u_norms = category_update_norms[cat]
        w_norms = category_weight_norms[cat]
        mean_update = sum(u_norms) / len(u_norms)
        mean_weight = sum(w_norms) / len(w_norms)
        ratio = mean_update / max(mean_weight, 1e-12)

        metrics[f"telemetry/update_ratio/{cat}"] = ratio
        metrics[f"telemetry/weight_norm/{cat}"] = mean_weight
        metrics[f"telemetry/update_norm/{cat}"] = mean_update
        all_ratios.append(ratio)

    if all_ratios:
        metrics["telemetry/update_ratio_min"] = min(all_ratios)
        metrics["telemetry/update_ratio_max"] = max(all_ratios)

    return metrics


# ═══════════════════════════════════════════════════════════════
# 4. Async Non-Blocking Telemetry Logger
# ═══════════════════════════════════════════════════════════════

import queue
import threading

class AsyncTelemetryLogger:
    """
    Non-blocking async telemetry logger.
    Offloads wandb.log() network I/O and dictionary logging to a background
    thread queue, ensuring 0 ms blocking overhead on the GPU training loop.
    """

    def __init__(self, wandb_run=None, queue_size: int = 1000):
        self.wandb_run = wandb_run
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._logging_worker, daemon=True, name="async-telemetry-logger"
        )
        self._worker_thread.start()

    def log(self, metrics: Dict[str, Any], step: int):
        """Enqueue metrics for non-blocking background logging."""
        if not self.wandb_run:
            return
        try:
            self.queue.put_nowait((metrics.copy(), step))
        except queue.Full:
            pass  # Drop metric frame if queue overflows to protect training loop

    def _logging_worker(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:
                    break
                metrics, step = item
                if self.wandb_run:
                    self.wandb_run.log(metrics, step=step)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncTelemetryLogger] Warning: failed to log metrics: {e}")

    def flush_and_shutdown(self):
        """Flush remaining metrics and stop worker thread."""
        self._stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
