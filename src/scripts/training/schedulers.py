"""
Learning-rate schedulers for multi-phase pre-training.

Provides:
    - WSDScheduler : Warmup → Stable → (Cosine) Decay
    - create_phase_scheduler : factory that builds the right scheduler from a PhaseConfig
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from ..configs.model_config import PhaseConfig


class WSDScheduler(LambdaLR):
    """
    Warmup-Stable-Decay scheduler (as used in MiniCPM / DeepSeek-V2).

    Three piecewise-linear / cosine segments:
        1. [0, warmup_steps)              : linear ramp  0 → peak_lr
        2. [warmup_steps, stable_end)     : constant      peak_lr
        3. [stable_end, total_steps)      : cosine decay  peak_lr → min_lr

    All LR values are expressed as *multipliers* of the optimizer's base lr
    (which should be set to ``peak_lr``).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        stable_frac: float = 0.76,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.stable_frac = stable_frac
        self.min_lr_ratio = min_lr_ratio

        # Boundaries
        self.stable_end = warmup_steps + int(
            (total_steps - warmup_steps) * stable_frac
        )
        self.decay_steps = max(total_steps - self.stable_end, 1)

        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step: int) -> float:
        # 1. Warmup
        if step < self.warmup_steps:
            return max(step / max(self.warmup_steps, 1), 0.0)

        # 2. Stable
        if step < self.stable_end:
            return 1.0

        # 3. Cosine decay → min_lr_ratio
        progress = (step - self.stable_end) / self.decay_steps
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


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
        return WSDScheduler(
            optimizer,
            warmup_steps=phase_config.warmup_steps,
            total_steps=phase_config.total_steps,
            stable_frac=phase_config.wsd_stable_frac,
            min_lr_ratio=phase_config.min_lr / phase_config.peak_lr,
            last_epoch=last_epoch,
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