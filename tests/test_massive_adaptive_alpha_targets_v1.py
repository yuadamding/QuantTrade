from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rl_quant.alpha.targets import OriginExposurePanel
from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MassiveAdaptiveAlphaTargetsV1Error,
    MassiveAdaptiveEconomicPathV1,
    build_massive_adaptive_alpha_targets_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
)


_ASSETS = ("SEC-A", "SEC-B", "SEC-C", "SEC-D", "SEC-E")
_DECISION_AT_MS = 1_000
_FILL_AT_MS = 2_000


def _path(
    security_id: str,
    *,
    scale: float,
    missing_offset: int | None = None,
    fallback_offset: int | None = None,
) -> MassiveAdaptiveEconomicPathV1:
    economic = tuple(_FILL_AT_MS + 1_000 * offset for offset in range(127))
    available = tuple(value + 50 for value in economic)
    values = [scale * (1.0 + 0.001 * offset) for offset in range(127)]
    valid = [True] * 127
    terminal = [False] * 127
    kinds = ["market"] * 127
    if missing_offset is not None:
        values[missing_offset] = 0.0
        valid[missing_offset] = False
        kinds[missing_offset] = "missing"
    if fallback_offset is not None:
        for offset in range(fallback_offset, 127):
            values[offset] = 0.0
            valid[offset] = True
            terminal[offset] = True
            kinds[offset] = "terminal-disposition"
    body = {
        "schema": "rl-quant.massive-adaptive-economic-path-v1",
        "security_id": security_id,
        "decision_at_ms": _DECISION_AT_MS,
        "fill_at_ms": _FILL_AT_MS,
        "economic_at_ms": economic,
        "available_at_ms": available,
        "values": tuple(values),
        "valid": tuple(valid),
        "terminal": tuple(terminal),
        "mark_kinds": tuple(kinds),
        "mark_receipts": tuple(
            semantic_sha256((security_id, "mark", offset)) for offset in range(127)
        ),
        "unresolved_terminal_fallback_session_offset": fallback_offset,
        "conservative_total_loss_fallback": fallback_offset is not None,
        "source_economic_path_receipt_sha256": semantic_sha256(
            (security_id, "source-economic-path")
        ),
    }
    result = MassiveAdaptiveEconomicPathV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _exposures(*, qualified: tuple[bool, ...] = (True,) * 5) -> OriginExposurePanel:
    return OriginExposurePanel(
        origin_at_ms=_DECISION_AT_MS,
        available_at_ms=_DECISION_AT_MS,
        asset_ids=_ASSETS,
        exposure_names=("intercept", "style"),
        exposures=tuple((1.0, float(index - 2)) for index in range(5)),
        regression_weights=(1.0, 2.0, 3.0, 2.0, 1.0),
        qualified_asset_mask=qualified,
        source_receipt_sha256=semantic_sha256("source-time-exposures"),
    )


def _targets(
    *,
    paths: tuple[MassiveAdaptiveEconomicPathV1, ...] | None = None,
):
    selected = paths or tuple(
        _path(security_id, scale=100.0 + 10.0 * index)
        for index, security_id in enumerate(_ASSETS)
    )
    return build_massive_adaptive_alpha_targets_v1(
        decision_session_date="2024-01-02",
        built_at_ms=max(max(path.available_at_ms) for path in selected),
        paths=selected,
        exposure_panel=_exposures(),
        origin_receipt_sha256="a" * 64,
        economic_accounting_receipt_sha256="b" * 64,
        fill_source_receipt_sha256="c" * 64,
        terminal_authority_receipt_sha256="d" * 64,
        economic_coverage_receipt_sha256="e" * 64,
    )


def test_builds_exact_nonoverlapping_bucket_returns_and_residuals() -> None:
    targets = _targets()

    assert targets.bucket_ids == tuple(
        bucket.bucket_id for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
    )
    expected = tuple(
        (1.0 + 0.001 * bucket.end_offset_sessions)
        / (1.0 + 0.001 * bucket.start_offset_sessions)
        - 1.0
        for bucket in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS
    )
    assert targets.rows[0].raw_bucket_returns == pytest.approx(expected)
    assert targets.valid_counts_by_bucket == (5,) * 7
    assert targets.factor_valid == (True,) * 7
    assert not targets.source_paths_replayed
    assert not targets.predictive_training_authorized
    assert not targets.profitability_reporting_authorized
    assert not targets.lockbox_access_authorized
    assert not targets.reinforcement_learning_authorized

    residual = np.asarray(
        [row.residual_bucket_returns for row in targets.rows], dtype=np.float64
    )
    factor = np.asarray(
        [row.factor_component_returns for row in targets.rows], dtype=np.float64
    )
    raw = np.asarray(
        [row.raw_bucket_returns for row in targets.rows], dtype=np.float64
    )
    assert np.allclose(raw, factor + residual, rtol=1.0e-10, atol=1.0e-12)
    design = np.asarray(targets.residual_operator.qualified_design)
    weights = np.asarray(targets.residual_operator.qualified_weights)
    assert np.max(np.abs(design.T @ (weights[:, None] * residual))) <= 2.0e-10


def test_missing_interior_mark_invalidates_complete_common_support() -> None:
    paths = tuple(
        _path(
            security_id,
            scale=100.0 + 10.0 * index,
            missing_offset=3 if security_id == "SEC-E" else None,
        )
        for index, security_id in enumerate(_ASSETS)
    )
    targets = _targets(paths=paths)
    affected = targets.rows[-1]

    assert affected.economic_valid_by_bucket == (
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    )
    assert affected.training_valid_by_bucket == (False,) * 7
    assert targets.common_training_asset_mask == (True, True, True, True, False)
    assert targets.valid_counts_by_bucket == (4,) * 7
    assert affected.factor_component_returns == (0.0,) * 7
    assert affected.residual_bucket_returns == (0.0,) * 7


def test_conservative_terminal_loss_is_carried_without_duration_semantics() -> None:
    paths = tuple(
        _path(
            security_id,
            scale=100.0 + 10.0 * index,
            fallback_offset=3 if security_id == "SEC-E" else None,
        )
        for index, security_id in enumerate(_ASSETS)
    )
    targets = _targets(paths=paths)
    terminal = targets.rows[-1]

    assert terminal.raw_bucket_returns[0] > 0.0
    assert terminal.raw_bucket_returns[1] == -1.0
    assert terminal.raw_bucket_returns[2:] == (0.0,) * 5
    assert terminal.conservative_fallback_by_bucket == (
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    assert terminal.terminal_by_bucket == (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    assert all(terminal.training_valid_by_bucket)


def test_path_and_target_chronology_fail_closed() -> None:
    path = _path("SEC-A", scale=100.0)
    with pytest.raises(
        MassiveAdaptiveAlphaTargetsV1Error, match="path receipt differs"
    ):
        replace(path, values=(101.0, *path.values[1:])).validate()

    paths = tuple(
        _path(security_id, scale=100.0 + 10.0 * index)
        for index, security_id in enumerate(_ASSETS)
    )
    with pytest.raises(
        MassiveAdaptiveAlphaTargetsV1Error, match="build chronology"
    ):
        build_massive_adaptive_alpha_targets_v1(
            decision_session_date="2024-01-02",
            built_at_ms=max(max(path.available_at_ms) for path in paths) - 1,
            paths=paths,
            exposure_panel=_exposures(),
            origin_receipt_sha256="a" * 64,
            economic_accounting_receipt_sha256="b" * 64,
            fill_source_receipt_sha256="c" * 64,
            terminal_authority_receipt_sha256="d" * 64,
            economic_coverage_receipt_sha256="e" * 64,
        )


def test_target_artifact_rejects_receipt_and_operator_corruption() -> None:
    targets = _targets()
    with pytest.raises(
        MassiveAdaptiveAlphaTargetsV1Error, match="factor replay differs"
    ):
        replace(
            targets,
            factor_return_target=(9.0, *targets.factor_return_target[1:]),
        ).validate()
    with pytest.raises(
        MassiveAdaptiveAlphaTargetsV1Error, match="operator differs"
    ):
        changed = replace(
            targets.rows[0], residual_operator_receipt_sha256="f" * 64
        )
        changed_row = replace(
            changed,
            receipt_sha256=semantic_sha256(changed.unsigned()),
        )
        replace(targets, rows=(changed_row, *targets.rows[1:])).validate()
