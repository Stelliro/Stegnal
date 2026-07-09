# difficulty.py

"""A self-learning controller for smooth difficulty transitions.

Instead of nudging difficulty by a fixed step, this controller *learns* the
local relationship between channel difficulty and reconstruction SSIM (the
slope ``dSSIM/dDifficulty``, normally negative — harder means lower SSIM) from
the run's own history. From that it proposes how much difficulty *should* change
to move SSIM toward a target band, damped and capped so transitions stay smooth.

It is intentionally tiny and dependency-light (an online slope estimate via an
exponential moving average), so it adapts within a handful of generations
without any training data or heavy model.
"""

from __future__ import annotations

import numpy as np


class DifficultyController:
    """Online learner that proposes smooth difficulty changes.

    Parameters
    ----------
    target_ssim:
        The SSIM the controller steers toward (the "competence frontier").
    max_step:
        Hard cap on a single proposed change — the smoothness knob.
    learning_rate:
        Fraction of the model-predicted move actually applied each step (damping).
    slope_smoothing:
        EMA weight for the learned slope (closer to 1 = slower, steadier learning).
    """

    def __init__(self, *, target_ssim: float = 0.5, max_step: float = 0.05,
                 learning_rate: float = 0.35, slope_smoothing: float = 0.7):
        self.target_ssim = float(target_ssim)
        self.max_step = float(max_step)
        self.learning_rate = float(learning_rate)
        self.slope_smoothing = float(np.clip(slope_smoothing, 0.0, 0.999))
        # Prior: SSIM drops ~0.5 per unit difficulty. Refined online.
        self._slope = -0.5
        self._last: tuple[float, float] | None = None
        self.samples = 0

    # -- learning -------------------------------------------------------
    def observe(self, difficulty: float, ssim: float) -> None:
        """Update the learned difficulty->SSIM slope from a new observation."""
        difficulty = float(difficulty)
        ssim = float(ssim)
        if self._last is not None:
            d0, s0 = self._last
            dd = difficulty - d0
            if abs(dd) > 1e-4:
                slope = (ssim - s0) / dd
                # keep the slope sensible (negative, bounded)
                slope = float(np.clip(slope, -5.0, -1e-3))
                a = self.slope_smoothing
                self._slope = a * self._slope + (1.0 - a) * slope
                self.samples += 1
        self._last = (difficulty, ssim)

    @property
    def learned_slope(self) -> float:
        return self._slope

    # -- proposing ------------------------------------------------------
    def propose_step(self, ssim: float) -> float:
        """Signed difficulty change to move ``ssim`` toward the target band.

        Positive => make it harder (there's SSIM slack); negative => ease off.
        Scaled by the learned slope, damped by ``learning_rate``, capped by
        ``max_step`` for smoothness.
        """
        slope = self._slope if abs(self._slope) > 1e-3 else -1e-3
        # difficulty change predicted to bring SSIM to target: (target - ssim)/slope
        raw = (self.target_ssim - float(ssim)) / slope
        step = raw * self.learning_rate
        return float(np.clip(step, -self.max_step, self.max_step))

    def next_difficulty(self, current: float, ssim: float) -> float:
        """Convenience: clamp(current + proposed step) into a valid difficulty."""
        return float(np.clip(current + self.propose_step(ssim), 0.01, 1.0))

    # -- persistence ----------------------------------------------------
    def to_state(self) -> dict:
        return {
            "target_ssim": self.target_ssim, "max_step": self.max_step,
            "learning_rate": self.learning_rate, "slope_smoothing": self.slope_smoothing,
            "slope": self._slope, "samples": self.samples,
        }

    @classmethod
    def from_state(cls, state: dict) -> DifficultyController:
        ctrl = cls(
            target_ssim=state.get("target_ssim", 0.5),
            max_step=state.get("max_step", 0.05),
            learning_rate=state.get("learning_rate", 0.35),
            slope_smoothing=state.get("slope_smoothing", 0.7),
        )
        ctrl._slope = float(state.get("slope", ctrl._slope))
        ctrl.samples = int(state.get("samples", 0))
        return ctrl


__all__ = ["DifficultyController"]
