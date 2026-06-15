#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "rl_quant" else PACKAGE_ROOT
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.features.stock_covariates import (  # noqa: E402
    build_action_covariate_tensor,
    load_silver_rows_by_symbol,
    read_covariate_coverage_manifest,
    tensor_content_hashes,
    validate_action_covariate_feature_schema,
)
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
    / "hour_from_second_1s"
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
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_ms(value: str) -> int:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def partition_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.partitions_root.glob(f"*/{args.dataset_file_name}"))
    if args.max_partitions > 0:
        paths = paths[: args.max_partitions]
    if not paths:
        raise ValueError(f"No partition datasets found below {args.partitions_root}")
    return paths


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
    if output.exists() and not force:
        return {"partition": dataset_path.parent.name, "status": "skipped_existing", "output": str(output)}

    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
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
        "base_dataset_file_name": dataset_path.name,
        "decision_timestamps": decisions,
        "decision_timestamps_ms": decision_ms,
        "action_names": action_names,
        "action_features": action_features,
        "action_feature_names": feature_names,
        "action_feature_available_timestamps_ms": action_feature_available,
        "action_features_available_timestamps_ms": row_available,
        "action_feature_groups": {
            "stock_covariates_v1": [0, feature_count],
            "stock_covariates_v1_mask": [feature_count, 2 * feature_count],
            "stock_covariates_v1_type": [2 * feature_count, 2 * feature_count + action_type_count],
        },
        "covariate_mode": "flat_append_baseline",
        "covariate_protocol_version": "stock_covariates_flat_append_v2",
        "action_features_augmented_with_covariates": True,
        "action_covariate_mask_appended_to_action_features": True,
        "action_covariate_cash_semantics": (
            "CASH covariate values are zero-imputed and mask=false; explicit "
            "stock_covariates_v1_type action-type channels disambiguate CASH from true zero values."
        ),
        "covariates_root": str(covariates_root),
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
            "covariate_reportability_errors": sidecar["action_covariate_reportability_errors"],
            "tensor_content_hashes": sidecar["tensor_content_hashes"],
        }
    )
    torch.save(sidecar, output)
    return {
        "partition": dataset_path.parent.name,
        "status": "written",
        "output": str(output),
        "rows": len(decisions),
        "actions": len(action_names),
        "features": len(feature_names),
        "reportability_errors": list(sidecar["action_covariate_reportability_errors"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    paths = partition_paths(args)
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
    summary = {
        "created_at_utc": utc_now_iso(),
        "partitions_root": str(args.partitions_root),
        "dataset_file_name": args.dataset_file_name,
        "covariates_root": str(args.covariates_root),
        "covariate_manifest_hash": source_manifest_hash,
        "covariate_feature_schema_file_hash": schema_file_hash,
        "partition_count": len(paths),
        "written_count": sum(1 for item in records if item["status"] == "written"),
        "skipped_count": sum(1 for item in records if item["status"].startswith("skipped")),
        "records": records,
    }
    summary_path = args.partitions_root.parent / "action_covariate_integration_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Integration summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
