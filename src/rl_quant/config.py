"""Central runtime configuration shared by every training/eval workflow.

Defining the device / AMP / memory-guard / seed options in one place keeps them consistent across
entry points -- the failure mode this prevents is a shared option (e.g. ``--amp-dtype`` or
``--min-free-vram-gb``) reaching one trainer but silently missing from another.

``add_runtime_args`` is torch-free so it can build a parser (and serve ``--help``) without importing
torch; ``resolve_runtime`` imports torch lazily, matching the scripts' "torch optional until run"
pattern.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


@dataclass(frozen=True)
class RuntimeConfig:
    device: "torch.device"
    use_amp: bool
    amp_dtype: str
    seed: int


def add_runtime_args(parser: argparse.ArgumentParser, *, seed_default: int = 17) -> None:
    """Add the shared runtime CLI options to ``parser`` (device, AMP, dtype, memory guard, seed).

    ``seed_default`` is parameterized because workflows historically defaulted to different seeds;
    everything else is identical across workflows by design. No torch import required.
    """
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP autocast precision when --amp is set: fp16 (default) or bf16 (no GradScaler).",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=0.0,
        help="Fail fast before training if free CUDA memory is below this many GiB (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=seed_default)


def resolve_runtime(args: Any, *, phase: str = "training") -> RuntimeConfig:
    """Resolve device, configure the torch runtime, enforce the free-VRAM guard, seed RNGs, and
    return the immutable :class:`RuntimeConfig`. Centralizes the setup each script did inline."""
    import torch

    from rl_quant.core import configure_torch_runtime, require_min_free_vram, resolve_torch_device

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    require_min_free_vram(device, float(getattr(args, "min_free_vram_gb", 0.0) or 0.0), phase=phase)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    return RuntimeConfig(
        device=device,
        use_amp=bool(getattr(args, "amp", False)),
        amp_dtype=str(getattr(args, "amp_dtype", "fp16")),
        seed=int(args.seed),
    )
