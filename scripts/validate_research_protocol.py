#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.research_protocol import (  # noqa: E402
    DatasetManifest,
    ExperimentRegistry,
    ModelManifest,
    ResearchProtocolError,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate QuantTrade dataset/model manifests against the research protocol contract.",
    )
    parser.add_argument("--dataset-manifest", type=Path, action="append", default=[])
    parser.add_argument("--model-manifest", type=Path, action="append", default=[])
    parser.add_argument("--registry", type=Path, help="Optional JSONL registry file to append validation records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_manifest and not args.model_manifest:
        raise SystemExit("Provide at least one --dataset-manifest or --model-manifest.")

    registry = ExperimentRegistry(args.registry) if args.registry else None
    failures: list[str] = []
    for path in args.dataset_manifest:
        try:
            manifest = DatasetManifest.from_dict(json.loads(path.read_text()))
            manifest.validate()
            print(f"OK dataset manifest: {path} ({manifest.dataset_id})")
            if registry:
                registry.append(
                    {
                        "record_type": "dataset_manifest_validation",
                        "validated_at_utc": utc_now_iso(),
                        "path": str(path),
                        "dataset_id": manifest.dataset_id,
                        "status": "ok",
                    }
                )
        except (OSError, json.JSONDecodeError, TypeError, ResearchProtocolError) as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")

    for path in args.model_manifest:
        try:
            manifest = ModelManifest.from_dict(json.loads(path.read_text()))
            manifest.validate_reportable()
            print(f"OK model manifest: {path} ({manifest.model_id})")
            if registry:
                registry.append(
                    {
                        "record_type": "model_manifest_validation",
                        "validated_at_utc": utc_now_iso(),
                        "path": str(path),
                        "model_id": manifest.model_id,
                        "status": "ok",
                    }
                )
        except (OSError, json.JSONDecodeError, TypeError, ResearchProtocolError) as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")

    if failures:
        for failure in failures:
            print(f"FAILED {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
