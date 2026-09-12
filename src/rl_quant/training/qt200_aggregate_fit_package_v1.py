"""Persisted cutoff-masked forecast inputs for bars-only engineering checks.

This package does not select the study's final splits or train a model. Each
explicit expanding-history view ends at its own cutoff, uses only fit feature
statistics, and keeps every fit row regardless of future target availability.
"""

from __future__ import annotations

import itertools
import math
import os
from datetime import date
from pathlib import Path

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_features_v1 import (
    FEATURE_NAMES, SCHEMA as FEATURE_SCHEMA, _parquet_rows,
)
from rl_quant.data_sources.massive.qt200_aggregate_targets_v1 import (
    BUCKET_NAMES, SCHEMA as TARGET_SCHEMA, read_completion, verify_artifact,
)
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.training.qt200_aggregate_forecast_data_v1 import prepare_fit_inputs

SCHEMA = "rl-quant.qt200-bars-forecast-fit-package-v1"
PLAN_SCHEMA = "rl-quant.qt200-bars-engineering-fit-plan-v1"
FILES = ("plan.json", "normalizers.json", "fit-inputs.parquet")


def build_fit_plan(*, sessions: list[str], fit_lengths: list[int]) -> dict:
    """Choose support-check cutoffs by calendar length, never outcome value.

    Lengths are explicit engineering inputs, not validation/outer assignments.
    The final study still needs its own frozen selection and access contract.
    """
    if (not sessions or sessions != sorted(set(sessions))
            or any(date.fromisoformat(day).isoformat() != day for day in sessions)):
        raise ValueError("Canonical fit-plan calendar required")
    if (not fit_lengths or any(type(n) is not int or not 1 <= n < len(sessions) for n in fit_lengths)
            or fit_lengths != sorted(set(fit_lengths))):
        raise ValueError("Strictly increasing bounded fit lengths required")
    return dict(schema=PLAN_SCHEMA, research_track="QT200-AGG-DEV-01",
        purpose="expanding-history-input-engineering-not-final-study-split",
        calendar_sha256=transport.digest(transport.canonical(sessions)),
        feature_names=list(FEATURE_NAMES), bucket_names=list(BUCKET_NAMES),
        views=[dict(view_id=f"history-{n:04d}", fit_sessions=sessions[:n],
                    cutoff_session=sessions[n - 1], heldout_start=sessions[n]) for n in fit_lengths],
        normalization="fit-features-only-per-view", target_access="mature-at-cutoff-only",
        missing_encoding="null-with-explicit-mask", model_fitted=False,
        study_selection_frozen=False, training_authorized=False, outer_access_authorized=False)


def _validate_plan(plan: dict, days: list[str]) -> None:
    try:
        lengths = [len(view["fit_sessions"]) for view in plan["views"]]
        expected = build_fit_plan(sessions=days, fit_lengths=lengths)
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed engineering fit plan") from exc
    if plan != expected:
        raise ValueError("Engineering fit plan/calendar/cutoff differs")


def _roots(feature_root: Path, target_root: Path, output: Path) -> None:
    roots = (feature_root, target_root, output)
    if (any(p.resolve() != p for p in roots)
            or any(a == b or a in b.parents or b in a.parents for a, b in itertools.combinations(roots, 2))):
        raise ValueError("Separate canonical input/output generations required")


def _view_inputs(view: dict, feature_root: Path, feature_sha256: str,
                 target_root: Path, target_sha256: str) -> dict:
    return prepare_fit_inputs(feature_root=feature_root, feature_sha256=feature_sha256,
        target_root=target_root, target_sha256=target_sha256,
        **{k: view[k] for k in ("fit_sessions", "cutoff_session", "heldout_start")})


def _reverify_parents(feature_root: Path, feature_sha256: str, target_root: Path, target_sha256: str) -> None:
    for root, digest, schema, artifact in (
        (feature_root, feature_sha256, FEATURE_SCHEMA, "features.parquet"),
        (target_root, target_sha256, TARGET_SCHEMA, "targets.parquet"),
    ):
        _, parent = read_completion(root, digest, schema)
        for name in (artifact, "specification.json"):
            verify_artifact(root, parent, name)
        read_completion(root, digest, schema)


def _rows(view: dict, prepared: dict):
    normalizer = prepared["normalizer"]
    for sample in prepared["samples"]:
        normalized = [None if x is None else (x - mu) / scale for x, mu, scale in
            zip(sample["features"], normalizer["mean"], normalizer["scale"], strict=True)]
        if any(value is not None and not math.isfinite(value) for value in normalized):
            raise ValueError("Nonfinite normalized fit feature")
        yield dict(view_id=view["view_id"], **sample, normalized_features=normalized,
                   training_authorized=False)


def _summary(view: dict, prepared: dict) -> dict:
    samples = prepared["samples"]
    return dict(view_id=view["view_id"], fit_sessions=len(view["fit_sessions"]),
        fit_start=view["fit_sessions"][0], fit_stop=view["fit_sessions"][-1],
        cutoff_session=view["cutoff_session"], heldout_start=view["heldout_start"], rows=len(samples),
        rows_with_any_mature_target=sum(any(row["target_loss_mask"]) for row in samples),
        rows_with_all_features_and_any_mature_target=sum(
            all(row["feature_mask"]) and any(row["target_loss_mask"]) for row in samples),
        rows_without_mature_targets_retained=sum(not any(row["target_loss_mask"]) for row in samples),
        mature_labels_by_bucket=prepared["mature_labels_by_bucket"],
        feature_observation_counts=prepared["normalizer"]["counts"])


def _expected_report(*, feature_sha256: str, target_sha256: str, plan: dict,
                     files: dict, summaries: list[dict]) -> dict:
    return dict(schema=SCHEMA, feature_sha256=feature_sha256, target_sha256=target_sha256,
        plan_sha256=transport.digest(transport.canonical(plan)), files=files, views=summaries,
        rows=sum(view["rows"] for view in summaries), ordered_tickers=list(SYMBOLS),
        feature_names=list(FEATURE_NAMES), bucket_names=list(BUCKET_NAMES),
        numeric_fit_input_preparation_complete=True, source_reverified_before_and_after=True,
        normalization_fitted=True, normalization_scope="separate-fit-only-view-statistics",
        missingness_retained=True, label_maturity_is_assumed=True, forecast_fitted=False,
        historical_identity_qualified=False, corporate_action_accounting_qualified=False,
        joint_h100_runtime_qualified=False, native_v5_qualified=False,
        study_selection_frozen=False, outer_access_authorized=False, training_ready=False)


def materialize_fit_package(*, feature_root: Path, feature_sha256: str, target_root: Path,
                            target_sha256: str, plan: dict, output: Path) -> dict:
    """Persist exact fit tensors into a new directory; never repair old output."""
    _roots(feature_root, target_root, output)
    _, target = read_completion(target_root, target_sha256, TARGET_SCHEMA)
    _validate_plan(plan, target["sessions"])
    if output.exists():
        raise FileExistsError(output)
    # Validate/reconstruct the first complete view before any publication.
    first = _view_inputs(plan["views"][0], feature_root, feature_sha256, target_root, target_sha256)
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("view_id", pa.string()), ("ticker", pa.string()), ("decision_session", pa.string()),
        ("feature_row_index", pa.int64()), ("features", pa.list_(pa.float64(), len(FEATURE_NAMES))),
        ("feature_mask", pa.list_(pa.bool_(), len(FEATURE_NAMES))),
        ("normalized_features", pa.list_(pa.float64(), len(FEATURE_NAMES))),
        ("targets", pa.list_(pa.float64(), len(BUCKET_NAMES))),
        ("target_loss_mask", pa.list_(pa.bool_(), len(BUCKET_NAMES))),
        ("training_authorized", pa.bool_()),
    ])
    output.mkdir(mode=0o700)
    transport.write_once(output / "plan.json", transport.canonical(plan))
    summaries, normalizers = [], {}
    with pq.ParquetWriter(output / "fit-inputs.parquet", schema, compression="zstd", compression_level=9) as writer:
        for index, view in enumerate(plan["views"]):
            prepared = first if index == 0 else _view_inputs(
                view, feature_root, feature_sha256, target_root, target_sha256)
            if index == 0:
                first = None
            normalizers[view["view_id"]] = prepared["normalizer"]
            summaries.append(_summary(view, prepared))
            pending = []
            for row in _rows(view, prepared):
                pending.append(row)
                if len(pending) == 4096:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            del prepared
    transport.write_once(output / "normalizers.json", transport.canonical(normalizers))
    files = {}
    for name in FILES:
        with (output / name).open("rb") as stream:
            os.fsync(stream.fileno())
        body = transport.read_regular(output / name, 512 * 1024 * 1024)
        files[name] = dict(path=name, bytes=len(body), sha256=transport.digest(body))
    report = _expected_report(feature_sha256=feature_sha256, target_sha256=target_sha256,
                              plan=plan, files=files, summaries=summaries)
    _reverify_parents(feature_root, feature_sha256, target_root, target_sha256)
    transport.write_once(output / "COMPLETE.json", transport.canonical(report))
    return report


def verify_fit_package(*, feature_root: Path, feature_sha256: str, target_root: Path,
                       target_sha256: str, output: Path, complete_sha256: str, plan_sha256: str) -> dict:
    """Reconstruct every row/scaler from parents, read-only; no caller proof flags."""
    _roots(feature_root, target_root, output)
    before, complete = read_completion(output, complete_sha256, SCHEMA)
    _, target = read_completion(target_root, target_sha256, TARGET_SCHEMA)
    if (complete.get("feature_sha256") != feature_sha256 or complete.get("target_sha256") != target_sha256
            or complete.get("plan_sha256") != plan_sha256 or set(complete.get("files", {})) != set(FILES)):
        raise ValueError("Fit package parent/plan/files binding differs")
    bodies = {name: verify_artifact(output, complete, name) for name in FILES}
    if transport.digest(bodies["plan.json"]) != plan_sha256:
        raise ValueError("Fit package plan hash differs")
    plan = transport.parse_json(bodies["plan.json"])
    normalizers = transport.parse_json(bodies["normalizers.json"])
    _validate_plan(plan, target["sessions"])
    if set(normalizers) != {view["view_id"] for view in plan["views"]}:
        raise ValueError("Fit normalizer view inventory differs")
    actual = iter(_parquet_rows(output / "fit-inputs.parquet"))
    summaries, checked = [], 0
    for view in plan["views"]:
        prepared = _view_inputs(view, feature_root, feature_sha256, target_root, target_sha256)
        if prepared["normalizer"] != normalizers[view["view_id"]]:
            raise ValueError("Fit-only normalizer reconstruction differs")
        summaries.append(_summary(view, prepared))
        for expected in _rows(view, prepared):
            if next(actual, None) != expected:
                raise ValueError("Fit row/mask/cutoff reconstruction differs")
            checked += 1
        del prepared
    if next(actual, None) is not None:
        raise ValueError("Extra fit rows")
    expected_report = _expected_report(feature_sha256=feature_sha256, target_sha256=target_sha256,
                                       plan=plan, files=complete["files"], summaries=summaries)
    if complete != expected_report:
        raise ValueError("Fit package summary or authorization differs")
    for name in FILES:
        if verify_artifact(output, complete, name) != bodies[name]:
            raise ValueError("Fit package changed during verification")
    if transport.read_regular(output / "COMPLETE.json", 4 * 1024 * 1024) != before:
        raise ValueError("Fit completion changed during verification")
    _reverify_parents(feature_root, feature_sha256, target_root, target_sha256)
    return dict(schema="rl-quant.qt200-bars-fit-input-replay-v1", complete_sha256=complete_sha256,
        plan_sha256=plan_sha256, feature_sha256=feature_sha256, target_sha256=target_sha256,
        rows_reconstructed=checked, views_reconstructed=len(summaries), nonmaterializing=True,
        exact_rows_and_normalizers_reproduced=True, model_fitted=False, training_authorized=False,
        native_v5_qualified=False)
