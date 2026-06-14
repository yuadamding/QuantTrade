#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local torch/CUDA runtime with a tensor smoke test.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amp", action="store_true", help="Run the matmul inside CUDA autocast")
    parser.add_argument("--json", action="store_true", help="Emit the runtime summary as JSON")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    try:
        import torch

        from rl_quant.core import (
            autocast_context,
            configure_torch_runtime,
            resolve_torch_device,
            torch_runtime_summary,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Torch is not installed in this Python environment. "
                "Use the ml1 conda environment, for example: conda run -n ml1 python scripts/check_torch_cuda.py"
            ) from exc
        raise

    device = resolve_torch_device(args.device)
    configure_torch_runtime(device)
    runtime = torch_runtime_summary(device)
    torch.manual_seed(args.seed)
    cuda_index = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    if args.json:
        print(json.dumps(runtime, indent=2))
    else:
        print(f"Torch: {runtime['torch_version']}")
        print(f"Device: {runtime['device']}")
        if device.type == "cuda":
            print(f"CUDA: {runtime['cuda_version']} | {runtime['cuda_device_name']}")
            print(f"Capability: {runtime['cuda_capability']} | Memory: {runtime['cuda_total_memory_gb']} GB")
            print(f"TF32 matmul: {runtime['cuda_tf32_matmul']} | cuDNN benchmark: {runtime['cudnn_benchmark']}")

    size = int(args.matrix_size)
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with autocast_context(device, args.amp):
            result = None
            for _ in range(args.repeats):
                result = a @ b
        end_event.record()
        torch.cuda.synchronize(cuda_index)
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        start = time.perf_counter()
        with autocast_context(device, args.amp):
            result = None
            for _ in range(args.repeats):
                result = a @ b
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    if result is None:
        raise RuntimeError("matmul smoke test did not run")
    finite = bool(torch.isfinite(result.float()).all().item())
    print(
        f"Matmul smoke: size={size} repeats={args.repeats} "
        f"amp={args.amp and device.type == 'cuda'} finite={finite} elapsed_ms={elapsed_ms:.2f}"
    )
    if device.type == "cuda":
        print(f"CUDA peak memory: {torch.cuda.max_memory_allocated(cuda_index) / 1024**2:.1f} MiB")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
