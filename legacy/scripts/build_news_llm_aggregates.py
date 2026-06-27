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
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_quant.features.news_llm import (  # noqa: E402
    NEWS_LLM_ACTION_SIDECAR_SCHEMA_VERSION,
    NEWS_LLM_AGGREGATE_FEATURE_NAMES,
    NEWS_LLM_FLAT_APPEND_MODE,
    NEWS_LLM_PROTOCOL_VERSION,
    build_action_news_llm_tensor,
    load_news_llm_rows_by_symbol,
    read_manifest,
)
from rl_quant.features.stock_covariates import tensor_content_hashes  # noqa: E402
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
DEFAULT_NEWS_LLM_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_llm_v1" / "top500_2023_to_present"
DEFAULT_ARTICLE_ROOT = DATA_ROOT / "polygon" / "stock_covariates" / "news_articles_v1" / "top500_2023_to_present"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build decision-aligned action news_llm_v1 sidecars for hour-from-second partitions."
    )
    parser.add_argument("--partitions-root", type=Path, default=DEFAULT_PARTITIONS_ROOT)
    parser.add_argument("--dataset-file-name", default="hour_from_second_dataset.pt")
    parser.add_argument("--news-llm-root", type=Path, default=DEFAULT_NEWS_LLM_ROOT)
    parser.add_argument("--article-root", type=Path, default=DEFAULT_ARTICLE_ROOT)
    parser.add_argument("--output-file-name", default="action_news_llm_covariates.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--expected-action-count",
        type=int,
        default=0,
        help="Expected action dimension, including CASH. Use 501 for TOP500 and 1501 for TOP501-2000.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild every selected sidecar, even if an existing one validates.")
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume an interrupted build by validating and skipping existing sidecars. "
            "Stale or unreadable sidecars are rebuilt. Enabled by default; --force disables reuse."
        ),
    )
    parser.add_argument("--max-partitions", type=int, default=0)
    parser.add_argument(
        "--partition-selection",
        choices=["latest", "earliest"],
        default="latest",
        help="When --max-partitions is set, choose latest partitions by default; earliest is diagnostic only.",
    )
    parser.add_argument(
        "--reportability-policy",
        choices=["diagnostic", "strict"],
        default="diagnostic",
        help="strict fails (nonzero exit) if any sidecar is non-reportable; diagnostic writes them and continues.",
    )
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
        if args.partition_selection == "latest":
            paths = paths[-args.max_partitions :]
        elif args.partition_selection == "earliest":
            paths = paths[: args.max_partitions]
        else:
            raise ValueError(f"Unsupported partition selection: {args.partition_selection!r}")
    if not paths:
        raise ValueError(f"No partition datasets found below {args.partitions_root}")
    return paths


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
    news_llm_manifest_hash: str,
    article_manifest_hash: str | None = None,
) -> None:
    if not isinstance(sidecar, dict):
        raise ValueError("news LLM sidecar payload is not a dictionary")
    expected_identity = base_dataset_identity(dataset_path, payload)
    for key, expected in expected_identity.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    if sidecar.get("integration_schema_version") != NEWS_LLM_ACTION_SIDECAR_SCHEMA_VERSION:
        raise ValueError("integration_schema_version mismatch")
    if sidecar.get("news_llm_feature_manifest_hash") != news_llm_manifest_hash:
        raise ValueError("news_llm_feature_manifest_hash mismatch")
    # The source-coverage masks (missing vs known-zero news) depend on the article manifest /
    # source symbol set, which is derived from --article-root independently of --news-llm-root.
    # A changed article manifest must force a rebuild so a stale sidecar is not reused with the
    # wrong source-coverage masks.
    if sidecar.get("news_article_manifest_hash") != article_manifest_hash:
        raise ValueError("news_article_manifest_hash mismatch")
    if list(sidecar.get("action_names", [])) != list(payload.get("action_names", [])):
        raise ValueError("action_names mismatch")
    if list(sidecar.get("decision_timestamps", [])) != list(payload.get("decision_timestamps", [])):
        raise ValueError("decision_timestamps mismatch")
    features = sidecar.get("action_features")
    available = sidecar.get("action_feature_available_timestamps_ms")
    names = list(sidecar.get("action_feature_names", []))
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("action_features missing or invalid")
    expected_rows = len(payload.get("decision_timestamps", []))
    expected_actions = len(payload.get("action_names", []))
    if tuple(features.shape[:2]) != (expected_rows, expected_actions):
        raise ValueError("action_features row/action shape mismatch")
    if len(names) != int(features.shape[-1]):
        raise ValueError("action_feature_names length mismatch")
    if not torch.is_tensor(available) or tuple(available.shape) != tuple(features.shape):
        raise ValueError("action_feature_available_timestamps_ms shape mismatch")
    decision_ms = torch.tensor([timestamp_ms(value) for value in payload["decision_timestamps"]], dtype=torch.long)
    decision_tensor = decision_ms.view(-1, 1, 1).expand_as(available)
    known = available >= 0
    if bool((available[known] > decision_tensor[known]).any().item()):
        raise ValueError("news LLM sidecar contains future action feature availability")
    expected_hashes = tensor_content_hashes(
        sidecar,
        [
            "action_features",
            "action_feature_available_timestamps_ms",
            "action_news_llm_features",
            "action_news_llm_mask",
            "action_news_llm_available_timestamps_ms",
        ],
    )
    if dict(sidecar.get("tensor_content_hashes", {})) != expected_hashes:
        raise ValueError("tensor_content_hashes mismatch")


def load_action_names(paths: list[Path], *, expected_action_count: int = 0) -> list[str]:
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    action_names = list(payload["action_names"])
    if not action_names or action_names[0] != "CASH":
        raise ValueError("Expected hour-from-second action_names to begin with CASH.")
    if expected_action_count > 0 and len(action_names) != expected_action_count:
        raise ValueError(
            f"Action universe mismatch in {paths[0]}: got {len(action_names)} actions, "
            f"expected {expected_action_count}. First actions={action_names[:12]}"
        )
    mismatches: list[dict[str, Any]] = []
    for path in paths[1:]:
        other = torch.load(path, map_location="cpu", weights_only=True)
        other_names = list(other.get("action_names", []))
        if other_names != action_names:
            mismatches.append(
                {
                    "partition": path.parent.name,
                    "action_count": len(other_names),
                    "action_names_head": other_names[:12],
                }
            )
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise ValueError(
            "Partition action_names are not identical across the selected build; "
            f"first mismatches={mismatches}"
        )
    return action_names


def sidecar_action_features(bundle: dict[str, Any], decision_ms: list[int]) -> dict[str, Any]:
    features = bundle["action_news_llm_features"].float()
    mask = bundle["action_news_llm_mask"].bool()
    available = bundle["action_news_llm_available_timestamps_ms"].long()
    decision_tensor = torch.tensor(decision_ms, dtype=torch.long).view(-1, 1, 1)
    mask_available = decision_tensor.expand_as(available)
    action_features = torch.cat([features, mask.float()], dim=-1)
    action_feature_available = torch.cat([available, mask_available], dim=-1)
    # Row-level freshness is the max availability over the VALUE channels only. The mask channels
    # are pinned to decision_ms, so including them would make every row look perfectly fresh-at-T
    # regardless of how stale the underlying news is, defeating any downstream staleness gating.
    value_known = available >= 0
    row_available = torch.where(
        value_known,
        available,
        torch.full_like(available, -1),
    ).amax(dim=-1)
    feature_names = [
        *[f"stock_news_llm_v1.{name}" for name in NEWS_LLM_AGGREGATE_FEATURE_NAMES],
        *[f"stock_news_llm_v1_mask.{name}" for name in NEWS_LLM_AGGREGATE_FEATURE_NAMES],
    ]
    width = len(NEWS_LLM_AGGREGATE_FEATURE_NAMES)
    return {
        "action_features": action_features,
        "action_feature_names": feature_names,
        "action_feature_available_timestamps_ms": action_feature_available,
        "action_features_available_timestamps_ms": row_available,
        "action_features_any_available_timestamps_ms": row_available,
        "action_feature_groups": {
            "stock_news_llm_v1": [0, width],
            "stock_news_llm_v1_mask": [width, 2 * width],
        },
    }


def build_sidecar(
    *,
    dataset_path: Path,
    output_file_name: str,
    news_llm_rows_by_symbol: dict[str, list[dict[str, Any]]],
    source_symbols: list[str],
    news_llm_manifest: dict[str, Any],
    news_llm_manifest_hash: str,
    article_manifest_hash: str | None,
    feature_manifest_reportability_errors: list[str],
    force: bool,
    resume_existing: bool,
) -> dict[str, Any]:
    output = dataset_path.with_name(output_file_name)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    stale_existing_error: str | None = None
    if output.exists() and resume_existing and not force:
        try:
            sidecar = torch.load(output, map_location="cpu", weights_only=True)
            validate_existing_sidecar(
                sidecar=sidecar,
                dataset_path=dataset_path,
                payload=payload,
                news_llm_manifest_hash=news_llm_manifest_hash,
                article_manifest_hash=article_manifest_hash,
            )
        except Exception as exc:
            stale_existing_error = f"{type(exc).__name__}: {exc}"
        else:
            return {
                "partition": dataset_path.parent.name,
                "status": "skipped_existing_validated",
                "output": str(output),
                "rows": len(payload["decision_timestamps"]),
                "actions": len(payload["action_names"]),
                "features": len(sidecar.get("action_feature_names", [])),
                "reportability_errors": list(sidecar.get("action_news_llm_reportability_errors", [])),
            }

    action_names = list(payload["action_names"])
    decisions = list(payload["decision_timestamps"])
    decision_ms = [timestamp_ms(value) for value in decisions]
    bundle = build_action_news_llm_tensor(
        news_llm_rows_by_symbol=news_llm_rows_by_symbol,
        action_names=action_names,
        decision_timestamps_ms=decision_ms,
        source_symbols=source_symbols,
        source_manifest_hash=news_llm_manifest_hash,
    )
    errors = list(bundle["action_news_llm_reportability_errors"])
    errors.extend(feature_manifest_reportability_errors)
    bundle["action_news_llm_reportability_errors"] = list(dict.fromkeys(errors))
    feature_payload = sidecar_action_features(bundle, decision_ms)
    sidecar = {
        "integration_schema_version": NEWS_LLM_ACTION_SIDECAR_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        **base_dataset_identity(dataset_path, payload),
        "decision_timestamps": decisions,
        "decision_timestamps_ms": decision_ms,
        "action_names": action_names,
        "news_llm_covariate_mode": NEWS_LLM_FLAT_APPEND_MODE,
        "news_llm_protocol_version": NEWS_LLM_PROTOCOL_VERSION,
        "news_llm_feature_manifest_hash": news_llm_manifest_hash,
        "news_article_manifest_hash": article_manifest_hash,
        "news_llm_manifest": news_llm_manifest,
        "action_features_augmented_with_news_llm": False,
        "action_features_are_news_llm_sidecar_only": True,
        **bundle,
        **feature_payload,
    }
    sidecar["tensor_content_hashes"] = tensor_content_hashes(
        sidecar,
        [
            "action_features",
            "action_feature_available_timestamps_ms",
            "action_news_llm_features",
            "action_news_llm_mask",
            "action_news_llm_available_timestamps_ms",
        ],
    )
    sidecar.update(sidecar["tensor_content_hashes"])
    sidecar["action_news_llm_sidecar_hash"] = stable_json_hash(
        {
            "integration_schema_version": sidecar["integration_schema_version"],
            "decision_timestamps": decisions,
            "action_names": action_names,
            "action_feature_names": sidecar["action_feature_names"],
            "action_news_llm_schema_hash": sidecar["action_news_llm_schema_hash"],
            "news_llm_feature_manifest_hash": news_llm_manifest_hash,
            # Source coverage (missing vs known-zero news) derives from the article manifest, so
            # bind it directly into the sidecar identity rather than relying on indirect coverage
            # via tensor content hashes.
            "news_article_manifest_hash": sidecar.get("news_article_manifest_hash"),
            "base_dataset_sha256": sidecar["base_dataset_sha256"],
            "base_dataset_payload_hash": sidecar["base_dataset_payload_hash"],
            "base_dataset_feature_schema_hash": sidecar["base_dataset_feature_schema_hash"],
            "base_dataset_action_schema_hash": sidecar["base_dataset_action_schema_hash"],
            "action_features_are_news_llm_sidecar_only": sidecar["action_features_are_news_llm_sidecar_only"],
            "reportability_errors": sidecar["action_news_llm_reportability_errors"],
            "tensor_content_hashes": sidecar["tensor_content_hashes"],
        }
    )
    validate_existing_sidecar(
        sidecar=sidecar,
        dataset_path=dataset_path,
        payload=payload,
        news_llm_manifest_hash=news_llm_manifest_hash,
        article_manifest_hash=article_manifest_hash,
    )
    atomic_torch_save(sidecar, output)
    return {
        "partition": dataset_path.parent.name,
        "status": "written" if stale_existing_error is None else "rewritten_stale_existing",
        "output": str(output),
        "rows": len(decisions),
        "actions": len(action_names),
        "features": len(sidecar["action_feature_names"]),
        "stale_existing_error": stale_existing_error,
        "reportability_errors": list(sidecar["action_news_llm_reportability_errors"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    paths = partition_paths(args)
    news_manifest_path = args.news_llm_root / "manifest.json"
    if not news_manifest_path.exists():
        raise FileNotFoundError(f"News LLM manifest does not exist: {news_manifest_path}")
    news_llm_manifest = read_manifest(news_manifest_path)
    news_llm_manifest_hash = file_sha256(news_manifest_path)
    article_manifest_path = args.article_root / "manifest.json"
    feature_manifest_reportability_errors = list(news_llm_manifest.get("reportability_errors", []))
    # Read the article manifest only after confirming it exists, so the missing-manifest branch is
    # unambiguous regardless of read_manifest's missing-path behavior.
    if article_manifest_path.exists():
        article_manifest = read_manifest(article_manifest_path)
        article_manifest_hash = file_sha256(article_manifest_path)
        # Source coverage comes ONLY from the article manifest. Do not fall back to the LLM-row
        # universe, which would conflate "explicit source coverage exists" with "we merely have
        # LLM rows for this symbol" and confuse missing-source vs known-zero-news downstream.
        source_symbols = list(article_manifest.get("symbols_with_source_news", []))
    else:
        article_manifest = {}
        article_manifest_hash = None
        source_symbols = []
        feature_manifest_reportability_errors.append("news_article_manifest_missing")
    action_names = load_action_names(paths, expected_action_count=int(args.expected_action_count))
    # Read diagnostically: the news-LLM feature manifest's reportability errors are already merged
    # into feature_manifest_reportability_errors and propagate into each sidecar, so a non-reportable
    # feature table yields non-reportable sidecars rather than aborting the build here.
    rows_by_symbol = load_news_llm_rows_by_symbol(
        args.news_llm_root, action_names, allow_nonreportable=True
    )
    records: list[dict[str, Any]] = []
    print(
        "Building news LLM action sidecars for "
        f"{len(paths)} partitions with workers={args.workers} "
        f"resume_existing={bool(args.resume_existing and not args.force)} force={bool(args.force)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                build_sidecar,
                dataset_path=path,
                output_file_name=args.output_file_name,
                news_llm_rows_by_symbol=rows_by_symbol,
                source_symbols=source_symbols,
                news_llm_manifest=news_llm_manifest,
                news_llm_manifest_hash=news_llm_manifest_hash,
                article_manifest_hash=article_manifest_hash,
                feature_manifest_reportability_errors=feature_manifest_reportability_errors,
                force=args.force,
                resume_existing=args.resume_existing,
            )
            for path in paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(paths)}] {record['status']} {record['partition']}", flush=True)
    records.sort(key=lambda item: str(item["partition"]))
    reportability_errors = list(
        dict.fromkeys(
            error
            for item in records
            for error in item.get("reportability_errors", [])
        )
    )
    summary = {
        "created_at_utc": utc_now_iso(),
        "partitions_root": str(args.partitions_root),
        "dataset_file_name": args.dataset_file_name,
        "news_llm_root": str(args.news_llm_root),
        "article_root": str(args.article_root),
        "news_llm_manifest_hash": news_llm_manifest_hash,
        "news_article_manifest_hash": article_manifest_hash,
        "partition_selection": args.partition_selection,
        "resume_existing": bool(args.resume_existing and not args.force),
        "force": bool(args.force),
        "partition_count": len(paths),
        "written_count": sum(1 for item in records if item["status"] == "written"),
        "rewritten_stale_existing_count": sum(1 for item in records if item["status"] == "rewritten_stale_existing"),
        "skipped_count": sum(1 for item in records if item["status"].startswith("skipped")),
        "non_reportable_partition_count": sum(1 for item in records if item.get("reportability_errors")),
        "reportable": not reportability_errors,
        "reportability_errors": reportability_errors,
        "records": records,
    }
    summary_path = args.partitions_root.parent / "action_news_llm_integration_manifest.json"
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(f"News LLM integration summary -> {summary_path}")
    if args.reportability_policy == "strict" and reportability_errors:
        preview = "; ".join(str(error) for error in reportability_errors[:20])
        raise SystemExit(f"non-reportable news LLM sidecar build under strict policy: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
