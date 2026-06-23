#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
QUANT_ROOT = PACKAGE_ROOT.parent
QUANT_ROOT_TEXT = QUANT_ROOT.resolve().as_posix()
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.features.stock_covariates import (  # noqa: E402
    COVARIATE_FLAT_PROTOCOL_VERSION,
    build_action_covariate_tensor,
    load_silver_rows_by_symbol,
    read_covariate_coverage_manifest,
    tensor_content_hashes,
    validate_action_covariate_feature_schema,
)
from rl_quant.partition_protocol import strict_latest_partition_violations  # noqa: E402
from rl_quant.research_protocol import stable_json_hash, utc_now_iso  # noqa: E402


def default_data_root() -> Path:
    shared_data = PROJECT_ROOT.parent / "data"
    if PROJECT_ROOT.name in {"QuantTrade", "rl_quant"} and shared_data.exists():
        return shared_data
    return PROJECT_ROOT / "data"


DATA_ROOT = default_data_root()
DEFAULT_PARTITIONS_ROOT = (
    DATA_ROOT
    / "protocol"
    / "polygon_second_top500_2025_to_2026-06-15"
    / "hour_from_second_1s_top50"
    / "partitions"
)
DEFAULT_COVARIATE_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "silver" / "top500_2023_to_present"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach compact action-level stock covariate sidecars to existing hour-from-second partitions."
    )
    parser.add_argument("--partitions-root", type=Path, default=DEFAULT_PARTITIONS_ROOT)
    parser.add_argument("--dataset-file-name", default="hour_from_second_dataset.pt")
    parser.add_argument("--covariates-root", type=Path, default=DEFAULT_COVARIATE_ROOT)
    parser.add_argument("--covariate-feature-schema", type=Path)
    parser.add_argument("--output-file-name", default="action_covariates.pt")
    parser.add_argument("--max-age-days", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-partitions", type=int, default=0)
    parser.add_argument(
        "--partition-selection",
        choices=["latest", "earliest"],
        default="latest",
        help="When --max-partitions is set, choose latest partitions by default; earliest is diagnostic only.",
    )
    parser.add_argument(
        "--allow-truncated-training-history",
        action="store_true",
        help=(
            "Permit a manifest whose selected partitions omit earlier available history. Off by "
            "default: restricted integration outputs are marked non-reportable."
        ),
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relativize_local_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if text == QUANT_ROOT_TEXT:
        return "."
    prefix = f"{QUANT_ROOT_TEXT}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def resolve_recorded_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (QUANT_ROOT / path).resolve()


def timestamp_ms(value: str) -> int:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def partition_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
    if args.max_partitions > 0:
        if args.partition_selection == "latest":
            paths = paths[-args.max_partitions :]
        elif args.partition_selection == "earliest":
            paths = paths[: args.max_partitions]
        else:
            raise ValueError(f"Unsupported partition selection: {args.partition_selection!r}")
    if not paths:
        raise ValueError(f"No partition datasets found below {args.partitions_root}")
    return paths


def partition_selection_reportability_errors(
    args: argparse.Namespace,
    *,
    selected_labels: list[str] | None = None,
    all_available_labels: list[str] | None = None,
) -> list[str]:
    if all_available_labels is None:
        all_available_labels = [
            path.parent.name for path in sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
        ]
    if selected_labels is None:
        selected_labels = [path.parent.name for path in partition_paths(args)]
    return strict_latest_partition_violations(
        selected_labels=selected_labels,
        all_available_labels=all_available_labels,
        allow_truncated_training_history=bool(args.allow_truncated_training_history),
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _payload_action_schema_hash(payload: dict[str, Any]) -> str | None:
    value = payload.get("action_schema_hash", payload.get("action_metadata_hash"))
    return None if value is None else str(value)


def base_dataset_identity(dataset_path: Path, payload: dict[str, Any]) -> dict[str, str | None]:
    return {
        "base_dataset_file_name": dataset_path.name,
        "base_dataset_sha256": file_sha256(dataset_path),
        "base_dataset_payload_hash": None if payload.get("payload_hash") is None else str(payload.get("payload_hash")),
        "base_dataset_feature_schema_hash": (
            None if payload.get("feature_schema_hash") is None else str(payload.get("feature_schema_hash"))
        ),
        "base_dataset_action_schema_hash": _payload_action_schema_hash(payload),
    }


def validate_existing_sidecar(
    *,
    sidecar: dict[str, Any],
    dataset_path: Path,
    payload: dict[str, Any],
    covariates_root: Path,
    source_manifest_hash: str,
    schema_file_hash: str,
) -> None:
    if not isinstance(sidecar, dict):
        raise ValueError("sidecar payload is not a dictionary")
    expected_identity = base_dataset_identity(dataset_path, payload)
    for key, expected in expected_identity.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    if list(sidecar.get("action_names", [])) != list(payload.get("action_names", [])):
        raise ValueError("action_names mismatch")
    if list(sidecar.get("decision_timestamps", [])) != list(payload.get("decision_timestamps", [])):
        raise ValueError("decision_timestamps mismatch")
    if sidecar.get("action_covariate_source_manifest_hash") != source_manifest_hash:
        raise ValueError("action_covariate_source_manifest_hash mismatch")
    if sidecar.get("action_covariate_feature_schema_file_hash") != schema_file_hash:
        raise ValueError("action_covariate_feature_schema_file_hash mismatch")
    try:
        stored_root = resolve_recorded_path(sidecar.get("covariates_root", ""))
        expected_root = covariates_root.resolve()
    except OSError as exc:
        raise ValueError("covariates_root is invalid") from exc
    if stored_root != expected_root:
        raise ValueError("covariates_root mismatch")
    if sidecar.get("covariate_protocol_version") != COVARIATE_FLAT_PROTOCOL_VERSION:
        raise ValueError("covariate_protocol_version mismatch")
    action_features = sidecar.get("action_features")
    feature_names = list(sidecar.get("action_feature_names", []))
    action_available = sidecar.get("action_feature_available_timestamps_ms")
    if not torch.is_tensor(action_features) or action_features.ndim != 3:
        raise ValueError("action_features missing or invalid")
    expected_rows = len(payload.get("decision_timestamps", []))
    expected_actions = len(payload.get("action_names", []))
    if tuple(action_features.shape[:2]) != (expected_rows, expected_actions):
        raise ValueError("action_features row/action shape mismatch")
    if len(feature_names) != int(action_features.shape[-1]):
        raise ValueError("action_feature_names length mismatch")
    if not torch.is_tensor(action_available) or tuple(action_available.shape) != tuple(action_features.shape):
        raise ValueError("action_feature_available_timestamps_ms shape mismatch")
    hash_keys = [
        "action_features",
        "action_feature_available_timestamps_ms",
        "action_covariates",
        "action_covariate_mask",
        "action_covariate_available_timestamps_ms",
        "action_source_coverage",
        "action_covariate_action_type_features",
    ]
    expected_hashes = tensor_content_hashes(sidecar, hash_keys)
    recorded_hashes = sidecar.get("tensor_content_hashes")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("tensor_content_hashes missing or invalid")
    if dict(recorded_hashes) != expected_hashes:
        raise ValueError("tensor_content_hashes mismatch")
    for key, expected in expected_hashes.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"{key} mismatch")


def load_action_names(paths: list[Path]) -> list[str]:
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    action_names = list(payload["action_names"])
    if not action_names or action_names[0] != "CASH":
        raise ValueError("Expected hour-from-second action_names to begin with CASH.")
    return action_names


def build_sidecar(
    *,
    dataset_path: Path,
    output_file_name: str,
    silver_rows_by_symbol: dict[str, list[dict[str, Any]]],
    coverage_by_symbol: dict[str, dict[str, bool]],
    source_manifest_hash: str,
    schema_file_hash: str,
    covariates_root: Path,
    max_age_days: int,
    force: bool,
) -> dict[str, Any]:
    output = dataset_path.with_name(output_file_name)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    stale_existing_error: str | None = None
    if output.exists() and not force:
        sidecar = torch.load(output, map_location="cpu", weights_only=True)
        try:
            validate_existing_sidecar(
                sidecar=sidecar,
                dataset_path=dataset_path,
                payload=payload,
                covariates_root=covariates_root,
                source_manifest_hash=source_manifest_hash,
                schema_file_hash=schema_file_hash,
            )
        except ValueError as exc:
            stale_existing_error = str(exc)
        else:
            return {
                "partition": dataset_path.parent.name,
                "status": "skipped_existing_validated",
                "output": relativize_local_path(output),
                "rows": len(payload["decision_timestamps"]),
                "actions": len(payload["action_names"]),
                "features": len(sidecar.get("action_feature_names", [])),
                "reportability_errors": list(sidecar.get("action_covariate_reportability_errors", [])),
            }

    action_names = list(payload["action_names"])
    decisions = list(payload["decision_timestamps"])
    decision_ms = [timestamp_ms(value) for value in decisions]
    covariates = build_action_covariate_tensor(
        silver_rows_by_symbol=silver_rows_by_symbol,
        action_names=action_names,
        decision_timestamps_ms=decision_ms,
        source_coverage_by_symbol=coverage_by_symbol,
        source_manifest_hash=source_manifest_hash,
        max_age_days=max_age_days,
    )
    covariates["action_covariate_feature_schema_file_hash"] = schema_file_hash

    action_covariates = covariates["action_covariates"].float()
    action_mask = covariates["action_covariate_mask"].bool()
    action_available = covariates["action_covariate_available_timestamps_ms"].long()
    action_type_features = covariates["action_covariate_action_type_features"].float()
    decision_tensor = torch.tensor(decision_ms, dtype=torch.long).view(-1, 1, 1)
    mask_available = decision_tensor.expand_as(action_available)
    action_type_available = decision_tensor.expand(
        decision_tensor.shape[0],
        action_covariates.shape[1],
        action_type_features.shape[-1],
    )
    action_features = torch.cat([action_covariates, action_mask.float(), action_type_features], dim=-1)
    action_feature_available = torch.cat([action_available, mask_available, action_type_available], dim=-1)
    known = action_feature_available >= 0
    row_available = torch.where(known, action_feature_available, torch.full_like(action_feature_available, -1)).amax(dim=-1)
    known_covariate_values = action_available >= 0
    value_row_available = torch.where(
        known_covariate_values,
        action_available,
        torch.full_like(action_available, -1),
    ).amax(dim=-1)
    feature_names = [
        *[f"stock_covariates_v1.{name}" for name in covariates["action_covariate_feature_names"]],
        *[f"stock_covariates_v1_mask.{name}" for name in covariates["action_covariate_feature_names"]],
        *[
            f"stock_covariates_v1_type.{name}"
            for name in covariates["action_covariate_action_type_feature_names"]
        ],
    ]
    feature_count = int(action_covariates.shape[-1])
    action_type_count = int(action_type_features.shape[-1])
    sidecar = {
        "integration_schema_version": "hour_from_second_action_covariates_v1",
        "created_at_utc": utc_now_iso(),
        **base_dataset_identity(dataset_path, payload),
        "decision_timestamps": decisions,
        "decision_timestamps_ms": decision_ms,
        "action_names": action_names,
        "action_features": action_features,
        "action_feature_names": feature_names,
        "action_feature_available_timestamps_ms": action_feature_available,
        "action_features_available_timestamps_ms": row_available,
        "action_features_any_available_timestamps_ms": row_available,
        "action_covariate_value_latest_available_timestamps_ms": value_row_available,
        "action_feature_groups": {
            "stock_covariates_v1": [0, feature_count],
            "stock_covariates_v1_mask": [feature_count, 2 * feature_count],
            "stock_covariates_v1_type": [2 * feature_count, 2 * feature_count + action_type_count],
        },
        "covariate_mode": "flat_append_baseline",
        "covariate_protocol_version": COVARIATE_FLAT_PROTOCOL_VERSION,
        "action_features_augmented_with_covariates": False,
        "action_features_are_covariate_sidecar_only": True,
        "action_covariate_mask_appended_to_action_features": True,
        "action_covariate_cash_semantics": (
            "CASH covariate values are zero-imputed and mask=false; explicit "
            "stock_covariates_v1_type action-type channels disambiguate CASH from true zero values."
        ),
        "covariates_root": relativize_local_path(covariates_root),
        "action_covariate_feature_schema_file_hash": schema_file_hash,
        **covariates,
    }
    sidecar["tensor_content_hashes"] = tensor_content_hashes(
        sidecar,
        [
            "action_features",
            "action_feature_available_timestamps_ms",
            "action_covariates",
            "action_covariate_mask",
            "action_covariate_available_timestamps_ms",
            "action_source_coverage",
            "action_covariate_action_type_features",
        ],
    )
    sidecar.update(sidecar["tensor_content_hashes"])
    sidecar["action_covariate_sidecar_hash"] = stable_json_hash(
        {
            "integration_schema_version": sidecar["integration_schema_version"],
            "decision_timestamps": decisions,
            "action_names": action_names,
            "action_feature_names": feature_names,
            "action_covariate_schema_hash": sidecar["action_covariate_schema_hash"],
            "action_covariate_source_manifest_hash": source_manifest_hash,
            "action_covariate_feature_schema_file_hash": schema_file_hash,
            "base_dataset_sha256": sidecar["base_dataset_sha256"],
            "base_dataset_payload_hash": sidecar["base_dataset_payload_hash"],
            "base_dataset_feature_schema_hash": sidecar["base_dataset_feature_schema_hash"],
            "base_dataset_action_schema_hash": sidecar["base_dataset_action_schema_hash"],
            "action_features_are_covariate_sidecar_only": sidecar["action_features_are_covariate_sidecar_only"],
            "covariate_reportability_errors": sidecar["action_covariate_reportability_errors"],
            "tensor_content_hashes": sidecar["tensor_content_hashes"],
        }
    )
    validate_existing_sidecar(
        sidecar=sidecar,
        dataset_path=dataset_path,
        payload=payload,
        covariates_root=covariates_root,
        source_manifest_hash=source_manifest_hash,
        schema_file_hash=schema_file_hash,
    )
    atomic_torch_save(sidecar, output)
    return {
        "partition": dataset_path.parent.name,
        "status": "written" if stale_existing_error is None else "rewritten_stale_existing",
        "output": relativize_local_path(output),
        "rows": len(decisions),
        "actions": len(action_names),
        "features": len(feature_names),
        "stale_existing_error": stale_existing_error,
        "reportability_errors": list(sidecar["action_covariate_reportability_errors"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    paths = partition_paths(args)
    all_available_labels = [
        path.parent.name for path in sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
    ]
    selected_labels = [path.parent.name for path in paths]
    selection_errors = partition_selection_reportability_errors(
        args,
        selected_labels=selected_labels,
        all_available_labels=all_available_labels,
    )
    schema_path = args.covariate_feature_schema or args.covariates_root / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Covariate feature schema does not exist: {schema_path}")
    validate_action_covariate_feature_schema(schema_path)
    manifest_path = args.covariates_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Covariate manifest does not exist: {manifest_path}")
    action_names = load_action_names(paths)
    silver_rows_by_symbol = load_silver_rows_by_symbol(args.covariates_root, action_names)
    coverage_by_symbol = read_covariate_coverage_manifest(manifest_path)
    source_manifest_hash = file_sha256(manifest_path)
    schema_file_hash = file_sha256(schema_path)

    records: list[dict[str, Any]] = []
    print(f"Integrating action covariates for {len(paths)} partitions with workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                build_sidecar,
                dataset_path=path,
                output_file_name=args.output_file_name,
                silver_rows_by_symbol=silver_rows_by_symbol,
                coverage_by_symbol=coverage_by_symbol,
                source_manifest_hash=source_manifest_hash,
                schema_file_hash=schema_file_hash,
                covariates_root=args.covariates_root,
                max_age_days=args.max_age_days,
                force=args.force,
            )
            for path in paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(paths)}] {record['status']} {record['partition']}", flush=True)
    records.sort(key=lambda item: str(item["partition"]))
    record_reportability_errors = list(
        dict.fromkeys(
            error
            for item in records
            for error in item.get("reportability_errors", [])
        )
    )
    reportability_errors = list(dict.fromkeys([*selection_errors, *record_reportability_errors]))
    non_reportable_partition_count = sum(1 for item in records if item.get("reportability_errors"))
    summary = {
        "created_at_utc": utc_now_iso(),
        "partitions_root": relativize_local_path(args.partitions_root),
        "dataset_file_name": args.dataset_file_name,
        "covariates_root": relativize_local_path(args.covariates_root),
        "covariate_manifest_hash": source_manifest_hash,
        "covariate_feature_schema_file_hash": schema_file_hash,
        "partition_selection": args.partition_selection,
        "allow_truncated_training_history": bool(args.allow_truncated_training_history),
        "partition_selection_reportability_errors": selection_errors,
        "partition_count": len(paths),
        "written_count": sum(1 for item in records if item["status"] == "written"),
        "rewritten_stale_existing_count": sum(1 for item in records if item["status"] == "rewritten_stale_existing"),
        "skipped_count": sum(1 for item in records if item["status"].startswith("skipped")),
        "non_reportable_partition_count": non_reportable_partition_count,
        "reportable": not reportability_errors,
        "reportability_errors": reportability_errors,
        "records": records,
    }
    summary_path = args.partitions_root.parent / "action_covariate_integration_manifest.json"
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Integration summary -> {relativize_local_path(summary_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
