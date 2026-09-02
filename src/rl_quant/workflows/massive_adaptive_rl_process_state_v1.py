"""Scoped preservation of process-global RNG state during adaptive-RL replay."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Any, cast, Iterator

import numpy as np
import torch


def _copied_numpy_rng_state() -> tuple[Any, ...]:
    state = cast(tuple[Any, Any, Any, Any, Any], np.random.get_state())
    return (
        state[0],
        state[1].copy(),
        state[2],
        state[3],
        state[4],
    )


@contextmanager
def preserve_massive_adaptive_rl_process_rng_state_v1(
    *, include_cuda: bool = False
) -> Iterator[None]:
    """Restore Python, NumPy, CPU Torch, and visible CUDA RNGs after replay."""

    python_state = random.getstate()
    numpy_state = _copied_numpy_rng_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if include_cuda
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_states))


__all__ = ["preserve_massive_adaptive_rl_process_rng_state_v1"]
