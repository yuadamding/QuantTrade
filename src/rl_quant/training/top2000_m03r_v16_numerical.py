"""Typed numerical-failure boundary for M03R-v16 training."""

from __future__ import annotations


class M03RV16NumericalTrainingError(FloatingPointError):
    """A finite check failed during score training or validation."""


__all__ = ["M03RV16NumericalTrainingError"]
