from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
import pytest

from rl_quant.training.top2000_m03r_v15_preflight import (
    load_m03r_v15_structural_preflight,
    run_m03r_v15_structural_preflight,
    write_m03r_v15_structural_preflight,
)


class _Validated(SimpleNamespace):
    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def test_v15_real_preflight_covers_every_origin_and_round_trips(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    import rl_quant.training.top2000_m03r_v15_preflight as preflight

    availability = torch.ones((1001, 4), dtype=torch.bool)
    availability[253, 1] = False
    cache = _Validated(
        daily_ohlcv=torch.ones((1001, 4, 5)),
        availability=availability,
        action_ids=("CASH", "A", "B", "C"),
        cache_sha256="a" * 64,
        action_hash="b" * 64,
    )
    exposures = _Validated(
        asset_axis_sha256="b" * 64,
        state_start_index=0,
        exposure_loadings=torch.zeros((1001, 4, 2), dtype=torch.float64),
        regression_weights=torch.ones((1001, 4), dtype=torch.float64),
        cash_index=0,
        projector_exposure_names=("x", "y"),
        projector_exposure_families=("sector", "style-risk"),
        receipt_sha256="c" * 64,
    )
    risk = _Validated(
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        exposures=exposures,
        receipt_sha256="d" * 64,
    )
    observed: dict[tuple[int, int], torch.Tensor] = {}
    calls: dict[int, int] = {}

    def _operator(**kwargs: Any) -> Any:
        origin = int(kwargs["origin_state_index"])
        occurrence = calls.get(origin, 0)
        calls[origin] = occurrence + 1
        mask = kwargs["available_mask"].clone()
        observed[(origin, occurrence)] = mask
        support = int(mask.sum())
        return SimpleNamespace(
            receipt_sha256=f"{origin * 2 + occurrence + 1:064x}",
            factor_qualified_risky_asset_count=support,
            weighted_residual_degrees_of_freedom=max(1, support - 1),
            qualified_asset_mask=mask,
        )

    monkeypatch.setattr(preflight, "build_m03r_v15_residual_operator", _operator)
    package_identities = {
        "cache_manifest_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "operator_source_sha256": "0" * 64,
        "risk_artifact_file_sha256": "1" * 64,
        "risk_source_manifest_file_sha256": "2" * 64,
        "projector_manifest_file_sha256": "3" * 64,
        "projector_manifest_sha256": "4" * 64,
        "projector_binding_sha256": "5" * 64,
    }
    receipt = run_m03r_v15_structural_preflight(  # type: ignore[arg-type]
        cache, risk, **package_identities
    )
    assert receipt.scheduled_origin_count == 716
    assert receipt.first_scheduled_origin == 251
    assert receipt.last_scheduled_origin == 996
    assert all(value == 2 for value in calls.values())
    assert bool(observed[(251, 1)][1])
    assert not bool(observed[(251, 0)][1])
    assert receipt.target_action_mask_difference_origin_count > 0

    path = tmp_path / "v15-structural-preflight.json"
    file_sha = write_m03r_v15_structural_preflight(path, receipt)
    loaded = load_m03r_v15_structural_preflight(
        path,
        expected_file_sha256=file_sha,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert loaded == receipt
    loaded.validate_for_package(
        cache_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_source_receipt_sha256="d" * 64,
        exposure_receipt_sha256="c" * 64,
        **package_identities,
    )
    for field in (
        "source_manifest_sha256",
        "operator_source_sha256",
        "risk_artifact_file_sha256",
        "risk_source_manifest_file_sha256",
        "projector_manifest_file_sha256",
        "projector_manifest_sha256",
        "projector_binding_sha256",
    ):
        with pytest.raises(Exception, match="exact package"):
            replace(loaded, **{field: "9" * 64}).validate_for_package(
                cache_sha256="a" * 64,
                asset_axis_sha256="b" * 64,
                risk_source_receipt_sha256="d" * 64,
                exposure_receipt_sha256="c" * 64,
                **package_identities,
            )
