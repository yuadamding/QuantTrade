"""`validate` command: check dataset/model manifests against the research-protocol contract.

Migrated verbatim from scripts/validate_research_protocol.py (the script is now a thin wrapper). The package
owns the logic; the orchestration here uses canonical package imports. ``main(argv)`` is argv-parameterizable so
it is testable and callable from the unified CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

from rl_quant.evaluation.research_protocol import (
    DatasetManifest,
    EvaluationProtocol,
    ExperimentRegistry,
    ModelManifest,
    ResearchProtocolError,
    utc_now_iso,
)


def adapt_trainer_model_manifest(payload: dict) -> dict:
    """Best-effort mapping from a trainer-written model_manifest.json to the ModelManifest schema.

    Trainers (e.g. train_hourly_causal_transformer_rl) write a richer/differently-keyed manifest
    (training_dataset, hyperparameters_hash, nested baseline/stress entries, a validation_protocol
    without name/benchmark_names/train_start). This adapter renames the known keys, coerces the
    baseline/stress entries to the dataclass shapes, synthesizes the missing EvaluationProtocol
    fields, and drops out-of-schema keys so ModelManifest.from_dict can validate the run.
    """
    raw = dict(payload)
    raw.setdefault("training_dataset_id", str(raw.get("training_dataset", "")))
    raw.setdefault("hyperparameter_search_space_hash", str(raw.get("hyperparameters_hash", "")))
    raw.setdefault("hyperparameter_trials", 1)
    raw.setdefault("feature_names_hash", str(raw.get("feature_names_hash", "")))
    raw.setdefault("action_names_hash", str(raw.get("action_names_hash", "")))

    def _coerce_baseline(entry: dict) -> dict:
        return {
            "name": str(entry.get("name", "")),
            "total_return": float(entry.get("total_return") or 0.0),
            "sharpe": entry.get("sharpe", entry.get("annualized_sharpe")),
            "max_drawdown": float(entry.get("max_drawdown") or 0.0),
            "turnover": entry.get("turnover", entry.get("total_switches")),
            "notes": str(entry.get("notes", "")),
        }

    def _coerce_stress(entry: dict) -> dict:
        return {
            "name": str(entry.get("name", "")),
            "kind": str(entry.get("kind", "")),
            "parameter": str(entry.get("parameter", "cost_bps")),
            "value": entry.get("value", entry.get("cost_bps", "")),
            "total_return": float(entry.get("total_return") or 0.0),
            "sharpe": entry.get("sharpe", entry.get("annualized_sharpe")),
            "max_drawdown": float(entry.get("max_drawdown") or 0.0),
        }

    baselines = [_coerce_baseline(e) for e in (raw.get("baseline_results") or []) if isinstance(e, dict)]
    raw["baseline_results"] = baselines
    raw["cost_stress_results"] = [_coerce_stress(e) for e in (raw.get("cost_stress_results") or []) if isinstance(e, dict)]
    raw["frequency_stress_results"] = [
        _coerce_stress(e) for e in (raw.get("frequency_stress_results") or []) if isinstance(e, dict)
    ]

    protocol = dict(raw.get("validation_protocol") or {})
    protocol.setdefault("name", str(raw.get("selected_by", "trainer_holdout")))
    protocol.setdefault("train_start", None)
    if not protocol.get("benchmark_names"):
        protocol["benchmark_names"] = [b["name"] for b in baselines]
    protocol_fields = {f.name for f in dataclass_fields(EvaluationProtocol)}
    raw["validation_protocol"] = {k: v for k, v in protocol.items() if k in protocol_fields}

    manifest_fields = {f.name for f in dataclass_fields(ModelManifest)}
    return {k: v for k, v in raw.items() if k in manifest_fields}


def load_model_manifest(payload: dict) -> ModelManifest:
    """Load a ModelManifest, falling back to the trainer-schema adapter on a strict-schema mismatch."""
    try:
        return ModelManifest.from_dict(payload)
    except (TypeError, KeyError):
        return ModelManifest.from_dict(adapt_trainer_model_manifest(payload))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate QuantTrade dataset/model manifests against the research protocol contract.",
    )
    parser.add_argument("--dataset-manifest", type=Path, action="append", default=[])
    parser.add_argument("--model-manifest", type=Path, action="append", default=[])
    parser.add_argument("--registry", type=Path, help="Optional JSONL registry file to append validation records.")
    parser.add_argument(
        "--allow-legacy-selection",
        action="store_true",
        help=(
            "Validate model manifests in legacy mode (strict=False): skip the structured "
            "selection_split=='validation' anti-leakage gate and fall back to the brittle 'test' in "
            "selected_by heuristic. For re-validating historical manifests written before selection_split "
            "existed; new trainer-written manifests should pass strict validation without this flag."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
            manifest = load_model_manifest(json.loads(path.read_text()))
            manifest.validate_reportable(strict=not args.allow_legacy_selection)
            selection_mode = "legacy" if args.allow_legacy_selection else "strict"
            print(f"OK model manifest ({selection_mode}): {path} ({manifest.model_id})")
            if registry:
                registry.append(
                    {
                        "record_type": "model_manifest_validation",
                        "validated_at_utc": utc_now_iso(),
                        "path": str(path),
                        "model_id": manifest.model_id,
                        "selection_validation_mode": selection_mode,
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
