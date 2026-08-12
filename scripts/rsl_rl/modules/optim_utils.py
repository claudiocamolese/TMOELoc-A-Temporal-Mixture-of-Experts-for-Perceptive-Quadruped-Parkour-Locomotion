"""Optimisation stabilisers shared by both distillation students.

Motivated by an observed failure: a run that was training cleanly for 1000
iterations (action loss 5.97 -> 3.50, routing perplexity ~3.1, episode length
397 -> 760) hit a single loss spike at iteration 1020, peaked at 392, and never
recovered -- it settled at 36 with a collapsed router. Adam's second-moment
estimate absorbs a spike like that and the run is effectively dead afterwards.

Gradient clipping alone does not prevent it: Adam normalises by sqrt(v), so the
step size is not proportional to the gradient norm.
"""

from __future__ import annotations

import math
from collections import deque


def warmup_cosine_lr(
    iteration: int, peak_lr: float, warmup_iters: int, total_iters: int, min_ratio: float = 0.1
) -> float:
    """Linear warmup, then cosine decay to ``min_ratio * peak_lr``.

    The upstream student inherits the teacher's actor weights, so it can take a
    constant 1e-3 from step zero. A Transformer trained from scratch cannot.
    """
    if warmup_iters > 0 and iteration < warmup_iters:
        return peak_lr * (iteration + 1) / warmup_iters
    if total_iters <= warmup_iters:
        return peak_lr
    progress = (iteration - warmup_iters) / max(1, total_iters - warmup_iters)
    progress = min(1.0, max(0.0, progress))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class SpikeGuard:
    """Drop optimiser steps whose loss dwarfs the recent history.

    A single bad batch -- one environment tumbling into a state no expert has
    seen, say -- should not be allowed to poison the optimiser state for the
    rest of the run. Skipping the step costs one batch; not skipping it cost a
    whole run.
    """

    def __init__(
        self,
        window: int = 100,
        factor: float = 5.0,
        min_history: int = 20,
        max_consecutive: int = 5,
    ):
        self.history: deque[float] = deque(maxlen=window)
        self.factor = factor
        self.min_history = min_history
        self.max_consecutive = max_consecutive
        self.num_skipped = 0
        self.consecutive = 0

    def should_skip(self, loss: float) -> bool:
        if not math.isfinite(loss):
            self.num_skipped += 1
            self.consecutive += 1
            return True

        # The loss always enters the history, even when the step is skipped.
        # An earlier version only recorded accepted losses, so once the loss
        # level shifted upwards the median stayed pinned at its pre-spike value
        # and every subsequent step was skipped: training froze permanently
        # (observed from iteration 900 onwards, 50 skips in 50 iterations).
        self.history.append(loss)
        if len(self.history) < self.min_history:
            return False

        ordered = sorted(self.history)
        median = ordered[len(ordered) // 2]
        if loss > self.factor * median and self.consecutive < self.max_consecutive:
            self.num_skipped += 1
            self.consecutive += 1
            return True
        self.consecutive = 0
        return False
