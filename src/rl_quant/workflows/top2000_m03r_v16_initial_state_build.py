"""Create the V16 common initial state from package-owned source bytes."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
)
from rl_quant.training.top2000_m03r_v16_initial_state import (
    write_m03r_v16_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)

M03R_V16_INITIAL_STATE_BUILD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-package-owned-initial-state-v1"
)


def build_package_owned_m03r_v16_initial_state(
    output_state_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, object]:
    random.seed(M03R_V16_PREDICTIVE_SPEC.seed)
    torch.manual_seed(M03R_V16_PREDICTIVE_SPEC.seed)
    policy = Top2000M03RV16PredictivePolicy(0)
    state_sha, state_file_sha, architecture_sha = (
        write_m03r_v16_initial_parameter_state(output_state_path, policy)
    )
    unsigned: dict[str, object] = {
        "schema": M03R_V16_INITIAL_STATE_BUILD_SCHEMA,
        "protocol_sha256": M03R_V16_PROTOCOL_SHA256,
        "initial_parameter_state_file_sha256": state_file_sha,
        "initial_parameter_state_sha256": state_sha,
        "initial_parameter_architecture_sha256": architecture_sha,
        "policy_source_sha256": file_sha256(
            Path(__file__).resolve().parents[1]
            / "training/top2000_m03r_v16_policy.py"
        ),
        "builder_source_sha256": file_sha256(__file__),
        "setting_index": 0,
        "seed": M03R_V16_PREDICTIVE_SPEC.seed,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    target = Path(output_receipt_path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_file_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-state", required=True)
    parser.add_argument("--output-receipt", required=True)
    args = parser.parse_args(argv)
    build_package_owned_m03r_v16_initial_state(
        args.output_state,
        args.output_receipt,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_INITIAL_STATE_BUILD_SCHEMA",
    "build_package_owned_m03r_v16_initial_state",
    "main",
]
