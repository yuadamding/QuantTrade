from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    TOP2000_HOLD30_SOURCE_BAR_SECONDS,
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03R_V9_PROJECTOR_EXPOSURE_NAMES,
    M03R_V9_SECTOR_EXPOSURE_NAMES,
    M03RV9PolygonRiskInputs,
    M03RV9PolygonRiskSlice,
    M03RV9RiskMaterializationError,
    load_top2000_m03r_v9_risk_source,
    materialize_top2000_m03r_v9_risk_source,
    write_top2000_m03r_v9_risk_source,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                list(values),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _weekdays(start: date, count: int) -> tuple[str, ...]:
    result: list[str] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(result)


def _cache() -> Top2000VerifiedDevelopmentCache:
    dates = _weekdays(date(2022, 1, 3), 80)
    actions = ("CASH", "A", "B", "C", "EQUITY:CASH")
    row = torch.arange(len(dates), dtype=torch.float64)
    daily = torch.zeros((len(dates), len(actions), 5), dtype=torch.float64)
    daily[:, 0, :4] = 1.0
    for asset in range(1, len(actions)):
        close = (
            20.0 + 3.0 * asset + (0.02 * asset * row) + torch.sin(row / (2.0 + asset))
        )
        daily[:, asset, 0] = close * 0.997
        daily[:, asset, 1] = close * 1.005
        daily[:, asset, 2] = close * 0.995
        daily[:, asset, 3] = close
        daily[:, asset, 4] = (100_000.0 * asset) + 1_000.0 * row
    available = torch.ones(daily.shape[:2], dtype=torch.bool)
    return Top2000VerifiedDevelopmentCache(
        daily_ohlcv=daily,
        availability=available,
        exchange_dates=dates,
        action_ids=actions,
        cache_sha256="1" * 64,
        cache_identity="2" * 64,
        search_identity="3" * 64,
        action_hash=_axis_sha256(actions),
        bar_seconds=TOP2000_HOLD30_SOURCE_BAR_SECONDS,
        acknowledgement=DEVELOPMENT_ACK,
        development_only=True,
        bars_only=True,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _polygon_inputs(tmp_path: Path) -> M03RV9PolygonRiskInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    slices = (("first", ("A", "B")), ("second", ("C", "CASH", "D")))
    input_slices: list[M03RV9PolygonRiskSlice] = []
    root_receipts: list[dict[str, Any]] = []
    for label, symbols in slices:
        universe = tmp_path / f"{label}.txt"
        universe.write_text("\n".join(symbols) + "\n")
        base = tmp_path / f"{label}-base"
        delta = tmp_path / f"{label}-delta"
        base.mkdir()
        delta.mkdir()
        (base / "manifest.csv").write_text("base\n")
        (base / "dataset_manifest.json").write_text('{"layer":"base"}\n')
        (delta / "manifest.csv").write_text("delta\n")
        (delta / "dataset_manifest.json").write_text('{"layer":"delta"}\n')
        for symbol in symbols:
            base_rows: list[dict[str, Any]] = [
                {
                    "asof_date": "2022-01-01",
                    "record_available": True,
                    "ticker": symbol,
                    "sic_code": {
                        "A": "3571",
                        "B": "6021",
                        "C": None,
                        "CASH": "6021",
                        "D": "1000",
                    }[symbol],
                }
            ]
            if symbol == "A":
                base_rows.append(
                    {
                        "asof_date": "2022-02-01",
                        "record_available": True,
                        "ticker": symbol,
                        "sic_code": "7372",
                    }
                )
            _write_jsonl(base / symbol / "overview_snapshots.jsonl", base_rows)
            _write_jsonl(
                delta / symbol / "overview_snapshots.jsonl",
                [
                    {
                        "asof_date": "2026-08-11",
                        "record_available": True,
                        "ticker": symbol,
                        "sic_code": "9999",
                    }
                ],
            )
        input_slices.append(
            M03RV9PolygonRiskSlice(
                label=label,
                universe_path=universe,
                base_root=base,
                delta_root=delta,
            )
        )
        root_receipts.append(
            {
                "slice": label,
                "symbols": len(symbols),
                "universe_sha256": _file_sha256(universe),
                "base_root": str(base.resolve()),
                "delta_root": str(delta.resolve()),
                "base_dataset_manifest_sha256": _file_sha256(
                    base / "dataset_manifest.json"
                ),
                "base_manifest_sha256": _file_sha256(base / "manifest.csv"),
                "delta_dataset_manifest_sha256": _file_sha256(
                    delta / "dataset_manifest.json"
                ),
                "delta_manifest_sha256": _file_sha256(delta / "manifest.csv"),
            }
        )
    receipt: dict[str, Any] = {
        "schema": "rl-quant.polygon-stock-covariate-delta-validation-v1",
        "checks": {
            "manifest_key_coverage": True,
            "file_sha256_match": True,
            "row_counts_match": True,
            "regular_non_symlink_files": True,
            "event_dates_within_requested_window": True,
            "download_failures": 0,
            "transient_api_failures": 0,
        },
        "combined_symbols": 5,
        "combined_unique_symbols": 5,
        "roots": root_receipts,
        "research_only": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt["canonical_payload_sha256"] = _canonical_sha256(receipt)
    receipt_path = tmp_path / "validation.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return M03RV9PolygonRiskInputs(
        slices=tuple(input_slices),
        validation_receipt_path=receipt_path,
        validation_receipt_file_sha256=_file_sha256(receipt_path),
    )


def test_materializes_causal_sector_and_cache_risk_surface(tmp_path: Path) -> None:
    cache = _cache()
    result = materialize_top2000_m03r_v9_risk_source(cache, _polygon_inputs(tmp_path))

    assert result.exposures.exposure_loadings.shape == (
        80,
        5,
        1 + len(M03R_V9_PROJECTOR_EXPOSURE_NAMES),
    )
    assert result.exposures.exposure_loadings.dtype == torch.float32
    assert result.exposures.exposure_available_timestamp_ms.shape == (80, 5, 3)
    assert result.mapped_risky_action_count == 4
    assert result.unused_raw_symbols == ("D",)
    assert result.future_delta_rows_consumed == 0
    assert result.readiness.blocker_codes == (
        "missing-projector-manifest",
        "target-projector-exposure-name-mismatch",
    )
    assert result.readiness.predictive_worker_authorized is False

    sector_start = 1
    manufacturing = M03R_V9_SECTOR_EXPOSURE_NAMES.index("sector-manufacturing")
    services = M03R_V9_SECTOR_EXPOSURE_NAMES.index("sector-services")
    unknown = M03R_V9_SECTOR_EXPOSURE_NAMES.index("sector-unknown-other")
    a_sectors = result.exposures.exposure_loadings[
        :, 1, sector_start : sector_start + len(M03R_V9_SECTOR_EXPOSURE_NAMES)
    ]
    c_sectors = result.exposures.exposure_loadings[
        :, 3, sector_start : sector_start + len(M03R_V9_SECTOR_EXPOSURE_NAMES)
    ]
    jan31 = cache.exchange_dates.index("2022-01-31")
    feb2 = cache.exchange_dates.index("2022-02-02")
    assert a_sectors[jan31, manufacturing].item() == 1.0
    assert a_sectors[feb2, services].item() == 1.0
    assert bool((c_sectors[:, unknown] == 1.0).all())
    assert bool(
        (
            result.exposures.exposure_available_timestamp_ms
            <= result.exposures.decision_timestamp_ms[:, None, None]
        ).all()
    )


def test_no_clobber_artifact_roundtrip_and_external_hash_gate(tmp_path: Path) -> None:
    result = materialize_top2000_m03r_v9_risk_source(
        _cache(), _polygon_inputs(tmp_path / "source")
    )
    written = write_top2000_m03r_v9_risk_source(result, tmp_path / "artifact")
    loaded, loaded_files = load_top2000_m03r_v9_risk_source(
        written.manifest_path,
        expected_manifest_file_sha256=written.manifest_file_sha256,
    )
    assert loaded.receipt_sha256 == result.receipt_sha256
    assert loaded.exposures.tensor_sha256 == result.exposures.tensor_sha256
    assert loaded_files.artifact_file_sha256 == written.artifact_file_sha256
    with pytest.raises(FileExistsError):
        write_top2000_m03r_v9_risk_source(result, tmp_path / "artifact")
    with pytest.raises(M03RV9RiskMaterializationError, match="manifest file hash"):
        load_top2000_m03r_v9_risk_source(
            written.manifest_path,
            expected_manifest_file_sha256="f" * 64,
        )


def test_cache_axis_and_polygon_receipt_drift_fail_before_materialization(
    tmp_path: Path,
) -> None:
    cache = _cache()
    inputs = _polygon_inputs(tmp_path)
    with pytest.raises(M03RV9RiskMaterializationError, match="action axis"):
        materialize_top2000_m03r_v9_risk_source(
            replace(cache, action_hash="e" * 64), inputs
        )
    inputs.validation_receipt_path.write_text("{}\n")
    with pytest.raises(M03RV9RiskMaterializationError, match="receipt hash"):
        materialize_top2000_m03r_v9_risk_source(cache, inputs)
