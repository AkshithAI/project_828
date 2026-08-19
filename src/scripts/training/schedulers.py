"""
Learning-rate schedulers for multi-phase pre-training.

Provides:
    - WSDScheduler : Warmup → Stable → (Cosine) Decay
    - create_phase_scheduler : factory that builds the right scheduler from a PhaseConfig
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from ..configs.model_config import PhaseConfig


class WSDScheduler(LambdaLR):
    """
    Warmup-Stable-Decay scheduler (as used in MiniCPM / DeepSeek-V2).

    Three piecewise-linear / cosine segments:
        1. warmup   : linear ramp  0 → peak_lr
        2. stable   : constant      peak_lr
        3. decay    : cosine decay  peak_lr → min_lr

    All LR values are expressed as *multipliers* of the optimizer's base lr
    (which should be set to ``peak_lr``).

    Two scheduling domains are supported:

    * **Step-based** (default): boundaries are optimizer steps. Simple, but the
      effective token budget shifts if the global batch size or context length
      changes mid-run (e.g. a context-extension curriculum).

    * **Token-based**: pass ``*_tokens`` boundaries and drive the schedule with
      :meth:`step_tokens` using the *cumulative non-padding token count*. This
      keeps warmup/stable/decay aligned to a fixed token budget regardless of
      padding efficiency or batch-size changes — recommended per
      ``private/pretraining_corrections.md`` §6.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        stable_frac: float = 0.76,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
        *,
        warmup_tokens: Optional[int] = None,
        decay_start_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        start_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio
        # Warmup start multiplier. 0.0 (default) ramps from zero , the standard
        # from scratch behavior. For continuation/extension runs, set this to
        # start_lr / peak_lr so warmup rewarms from the base run's final LR
        # (e.g. 3e-5 -> 6e-5) instead of collapsing the LR to zero first.
        self.start_lr_ratio = start_lr_ratio

        # ── Token-based boundaries (optional) ──
        # When ``total_tokens`` is provided the schedule is driven by cumulative
        # non-padding tokens via ``step_tokens``; ``_current_tokens`` tracks the
        # running count and is what ``_lr_lambda`` consults.
        self.token_based = total_tokens is not None
        self.total_tokens = total_tokens
        self.warmup_tokens = warmup_tokens if warmup_tokens is not None else 0
        # Default decay to the same stable fraction if not explicitly given.
        if self.token_based:
            if decay_start_tokens is not None:
                self.decay_start_tokens = decay_start_tokens
            else:
                self.decay_start_tokens = self.warmup_tokens + int(
                    (self.total_tokens - self.warmup_tokens) * stable_frac
                )
            self.decay_tokens = max(self.total_tokens - self.decay_start_tokens, 1)
        self._current_tokens = 0

        # Step-based boundaries
        self.stable_end = warmup_steps + int(
            (total_steps - warmup_steps) * stable_frac
        )
        self.decay_steps = max(total_steps - self.stable_end, 1)

        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _token_lr_multiplier(self, tokens: int) -> float:
        # 1. Warmup — linear ramp start_lr_ratio → 1.0
        if tokens < self.warmup_tokens:
            frac = max(tokens / max(self.warmup_tokens, 1), 0.0)
            return self.start_lr_ratio + (1.0 - self.start_lr_ratio) * frac
        # 2. Stable
        if tokens < self.decay_start_tokens:
            return 1.0
        # 3. Cosine decay → min_lr_ratio
        progress = (tokens - self.decay_start_tokens) / self.decay_tokens
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _lr_lambda(self, step: int) -> float:
        # Token-based schedule ignores the optimizer-step counter and consults
        # the externally-tracked cumulative token count instead.
        if self.token_based:
            return self._token_lr_multiplier(self._current_tokens)

        # 1. Warmup — linear ramp start_lr_ratio → 1.0
        if step < self.warmup_steps:
            frac = max(step / max(self.warmup_steps, 1), 0.0)
            return self.start_lr_ratio + (1.0 - self.start_lr_ratio) * frac

        # 2. Stable
        if step < self.stable_end:
            return 1.0

        # 3. Cosine decay → min_lr_ratio
        progress = (step - self.stable_end) / self.decay_steps
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step_tokens(self, num_new_tokens: int) -> None:
        """Advance the token-based schedule by *num_new_tokens* and update LR.

        Call this once per optimizer update (after ``optimizer.step()``) with the
        number of *non-padding* tokens consumed since the previous call. Only
        valid when the scheduler was constructed with token-based boundaries.
        """
        if not self.token_based:
            raise RuntimeError(
                "step_tokens() requires a token-based WSDScheduler "
                "(construct with total_tokens=...)."
            )
        self._current_tokens += int(num_new_tokens)
        multiplier = self._token_lr_multiplier(self._current_tokens)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["_current_tokens"] = self._current_tokens
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self._current_tokens = state_dict.pop("_current_tokens", 0)
        super().load_state_dict(state_dict)


def create_phase_scheduler(
    optimizer: Optimizer,
    phase_config: PhaseConfig,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Build the correct LR scheduler for a given phase.

    Args:
        optimizer:    The optimizer (base lr should already be set to phase_config.peak_lr).
        phase_config: A PhaseConfig instance.
        last_epoch:   Passed through to the scheduler for resumption.

    Returns:
        A ``LambdaLR``-compatible scheduler.
    """
    if phase_config.scheduler_type == "wsd":
        # Continuation warmup: rewarm from start_lr → peak_lr instead of 0 → peak.
        start_lr = getattr(phase_config, "start_lr", None)
        start_lr_ratio = (start_lr / phase_config.peak_lr) if start_lr else 0.0
        return WSDScheduler(
            optimizer,
            warmup_steps=phase_config.warmup_steps,
            total_steps=phase_config.total_steps,
            stable_frac=phase_config.wsd_stable_frac,
            min_lr_ratio=phase_config.min_lr / phase_config.peak_lr,
            last_epoch=last_epoch,
            # Optional token-based scheduling; inert unless total_tokens is set.
            warmup_tokens=getattr(phase_config, "warmup_tokens", None),
            decay_start_tokens=getattr(phase_config, "decay_start_tokens", None),
            total_tokens=getattr(phase_config, "total_tokens", None),
            start_lr_ratio=start_lr_ratio,
        )

    elif phase_config.scheduler_type == "cosine":
        warmup = phase_config.warmup_steps
        total = phase_config.total_steps
        min_ratio = phase_config.min_lr / phase_config.peak_lr

        def _cosine_lambda(step: int) -> float:
            if step < warmup:
                return max(step / max(warmup, 1), 0.0)
            progress = (step - warmup) / max(total - warmup, 1)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(optimizer, _cosine_lambda, last_epoch=last_epoch)

    else:
        raise ValueError(
            f"Unknown scheduler_type={phase_config.scheduler_type!r}. "
            f"Expected 'wsd' or 'cosine'."
        )