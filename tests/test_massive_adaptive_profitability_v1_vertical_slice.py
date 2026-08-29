from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    build_massive_adaptive_forecast_calibration_v1,
    materialize_massive_adaptive_forecast_calibration_v1,
    parse_massive_adaptive_forecast_calibration_v1,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    build_massive_adaptive_profit_trace_v1,
    materialize_massive_adaptive_profit_trace_v1,
    parse_massive_adaptive_profit_trace_v1,
)
from rl_quant.evaluation.massive_adaptive_profitability_authority_v1 import (
    MassiveAdaptiveProfitabilityAuthorityV1Error,
    build_massive_adaptive_profitability_authority_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.protocol.canonical_artifact import semantic_sha256


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _calibration(security_ids: tuple[str, ...], *, root=None):
    window_receipt = _digest("training-window-plan")
    valid = torch.ones(len(security_ids), dtype=torch.bool)
    prediction = torch.zeros((len(security_ids), 7), dtype=torch.float32)
    scale = torch.full_like(prediction, 0.01)
    forecast_row = SimpleNamespace(
        decision_session_date="2023-12-29",
        security_ids=security_ids,
        residual_mean=prediction,
        residual_scale=scale,
        valid=valid,
        receipt_sha256=_digest("training-forecast-row"),
    )
    target_rows = tuple(
        SimpleNamespace(
            security_id=security_id,
            residual_bucket_returns=tuple(
                0.001 * (bucket + 1) * (1.0 + index / len(security_ids))
                for bucket in range(7)
            ),
            training_valid_by_bucket=(True,) * 7,
            receipt_sha256=_digest(("training-target", security_id)),
        )
        for index, security_id in enumerate(security_ids)
    )
    source_target = SimpleNamespace(
        decision_session_date="2023-12-29",
        targets=SimpleNamespace(
            security_ids=security_ids,
            rows=target_rows,
        ),
    )
    training_forecasts = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=(forecast_row,),
        runtime_forecasts_replayed=True,
        origin_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-forecast-archive"),
        committed_source_data_qualified=False,
        window_plan_receipt_sha256=window_receipt,
    )
    training_targets = SimpleNamespace(
        validate=lambda: None,
        runtime_source_targets=(source_target,),
        runtime_roots_replayed=True,
        decision_session_dates=("2023-12-29",),
        semantic_receipt_sha256=_digest("training-target-archive"),
        committed_source_data_qualified=False,
    )
    training_window_plan = SimpleNamespace(
        validate=lambda: None,
        split_role="training",
        semantic_receipt_sha256=window_receipt,
        rows=(SimpleNamespace(origin_session_date="2023-12-29"),),
    )
    if root is not None:
        return materialize_massive_adaptive_forecast_calibration_v1(
            root=root,
            artifact_id="training-calibration",
            training_forecasts=training_forecasts,
            training_targets=training_targets,
            training_window_plan=training_window_plan,
            committed_at_ms=9_000,
        )
    return build_massive_adaptive_forecast_calibration_v1(
        training_forecasts=training_forecasts,
        training_targets=training_targets,
        training_window_plan=training_window_plan,
    )


def _fixture():
    dates = ("2024-01-02", "2024-01-03", "2024-01-04")
    security_ids = tuple(f"SEC-{index:03d}" for index in range(100))
    identity_receipt = _digest("pit-identity")
    masters = tuple(
        SimpleNamespace(
            security_id=security_id,
            issuer_id=f"ISSUER-{index:03d}",
            identity_source_receipt_sha256=_digest(("identity", security_id)),
        )
        for index, security_id in enumerate(security_ids)
    )
    identity = SimpleNamespace(
        validate=lambda: None,
        receipt_sha256=identity_receipt,
        security_master=masters,
    )
    bars_rows = {}
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    for date_index, session_date in enumerate(dates):
        for asset_index, security_id in enumerate(security_ids):
            values = [0.0] * len(MASSIVE_DAILY_BARS_V0_FIELDS)
            price = (
                100.0
                + asset_index * 0.1
                + date_index * (0.2 if asset_index < 50 else -0.1)
            )
            values[close_index] = price
            values[dollar_index] = 100_000_000.0
            bars_rows[(session_date, security_id)] = SimpleNamespace(
                bars_values=tuple(values),
                bars_valid=(True,) * len(values),
                receipt_sha256=_digest(("daily", session_date, security_id)),
            )
    daily_receipt = _digest("daily-input")
    daily = SimpleNamespace(
        validate=lambda: None,
        sessions=tuple(SimpleNamespace(source_session_date=value) for value in dates),
        row=lambda *, session_date, security_id: bars_rows[(session_date, security_id)],
        semantic_receipt_sha256=daily_receipt,
        daily_input_data_qualified=False,
    )
    fill_rows = {}
    for session_date in dates[1:]:
        for asset_index, security_id in enumerate(security_ids):
            price = 100.05 + asset_index * 0.1
            fill_rows[(session_date, security_id)] = SimpleNamespace(
                valid=True,
                fill_vwap=price,
                qualifying_share_volume=1_000_000.0,
                receipt_sha256=_digest(("fill", session_date, security_id)),
            )
    fill_receipt = _digest("fill-source")
    fill_source = SimpleNamespace(
        validate=lambda: None,
        row=lambda *, session_date, security_id: fill_rows[(session_date, security_id)],
        daily_input_authority_semantic_receipt_sha256=daily_receipt,
        semantic_receipt_sha256=fill_receipt,
        source_data_qualified=False,
    )
    forecast_rows = []
    roots = []
    contexts = []
    plan_rows = []
    for date_index, session_date in enumerate(dates[:2]):
        next_session = dates[date_index + 1]
        root_receipt = _digest(("decision-root", session_date))
        context_receipt = _digest(("context", session_date))
        inference_receipt = _digest(("inference-row", session_date))
        means = torch.empty((len(security_ids), 7), dtype=torch.float32)
        for asset_index in range(len(security_ids)):
            means[asset_index] = (0.004 if asset_index < 50 else -0.002) * torch.arange(
                1, 8, dtype=torch.float32
            )
        valid = torch.ones(len(security_ids), dtype=torch.bool)
        forecast_rows.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                security_ids=security_ids,
                residual_mean=means,
                residual_scale=torch.full_like(means, 0.001),
                valid=valid,
                decision_root_receipt_sha256=root_receipt,
                inference_row_receipt_sha256=inference_receipt,
                array_receipts=(*(_digest((session_date, i)) for i in range(18)),),
                receipt_sha256=_digest(("forecast-row", session_date)),
            )
        )
        roots.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                context_security_ids=security_ids,
                semantic_receipt_sha256=root_receipt,
                context_origin_receipt_sha256=context_receipt,
                source_data_qualified=False,
            )
        )
        contexts.append(
            SimpleNamespace(
                validate=lambda: None,
                decision_session_date=session_date,
                semantic_receipt_sha256=context_receipt,
                identity_authority_receipt_sha256=identity_receipt,
            )
        )
        plan_rows.append(
            SimpleNamespace(
                validate=lambda **_: None,
                decision_session_date=session_date,
                next_session_date=next_session,
                context_session_dates=(session_date,),
                receipt_sha256=inference_receipt,
            )
        )
    forecast_archive = SimpleNamespace(
        validate=lambda: None,
        runtime_rows=tuple(forecast_rows),
        runtime_forecasts_replayed=True,
        row_receipts=tuple(row.receipt_sha256 for row in forecast_rows),
        semantic_receipt_sha256=_digest("validation-forecast-archive"),
        development_forecast_authorized=False,
    )
    inference_plan = SimpleNamespace(
        validate=lambda: None,
        rows=tuple(plan_rows),
        semantic_receipt_sha256=_digest("inference-plan"),
    )
    return SimpleNamespace(
        forecast_archive=forecast_archive,
        calibration=_calibration(security_ids),
        inference_plan=inference_plan,
        roots=tuple(roots),
        contexts=tuple(contexts),
        fill_source=fill_source,
        fill_rows=fill_rows,
        daily=daily,
        identity=identity,
    )


def _trace(fixture, *, cost: float = 20.0):
    return build_massive_adaptive_profit_trace_v1(
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        initial_capital=10_000_000.0,
        transaction_cost_basis_points=cost,
    )


def test_forecast_to_compiler_to_fill_to_continuous_book_replays(tmp_path) -> None:
    fixture = _fixture()
    trace = _trace(fixture)

    assert len(trace.rows) == 2
    assert trace.rows[1].pretrade_book_receipt_sha256 == (
        trace.rows[0].posttrade_book_receipt_sha256
    )
    assert trace.final_equity > 0.0
    assert trace.final_benchmark_equity > 0.0
    assert trace.deterministic_profitability_replayed
    assert not trace.source_data_qualified
    assert not trace.profitability_reporting_authorized
    assert not trace.reinforcement_learning_authorized

    committed = materialize_massive_adaptive_profit_trace_v1(
        root=tmp_path,
        artifact_id="inner-validation-profit",
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
        initial_capital=10_000_000.0,
        committed_at_ms=10_000,
    )
    assert committed.deterministic_profitability_replayed
    assert committed.loaded_source is not None
    generic = parse_massive_adaptive_profit_trace_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )
    assert not generic.deterministic_profitability_replayed
    assert generic.semantic_receipt_sha256 == committed.semantic_receipt_sha256

    authority = build_massive_adaptive_profitability_authority_v1(
        trace=committed,
        forecast_archive=fixture.forecast_archive,
        calibration=fixture.calibration,
        inference_plan=fixture.inference_plan,
        decision_roots=fixture.roots,
        context_origins=fixture.contexts,
        fill_source=fixture.fill_source,
        daily_input_authority=fixture.daily,
        identity_authority=fixture.identity,
    )
    assert authority.deterministic_trace_replayed
    assert not authority.development_profitability_authorized
    assert not authority.profitability_reporting_authorized
    assert not authority.reinforcement_learning_authorized


def test_cost_ladder_and_fill_mutation_are_economically_visible() -> None:
    fixture = _fixture()
    low = _trace(fixture, cost=10.0)
    primary = _trace(fixture, cost=20.0)
    high = _trace(fixture, cost=40.0)
    assert low.final_equity > primary.final_equity > high.final_equity

    original = primary
    key = next(iter(fixture.fill_rows))
    changed_rows = dict(fixture.fill_rows)
    changed_rows[key] = SimpleNamespace(
        **{
            **changed_rows[key].__dict__,
            "fill_vwap": changed_rows[key].fill_vwap * 1.01,
            "receipt_sha256": _digest(("mutated-fill", key)),
        }
    )
    changed_fill = SimpleNamespace(
        **{
            **fixture.fill_source.__dict__,
            "semantic_receipt_sha256": _digest("mutated-fill-source"),
            "row": lambda *, session_date, security_id: changed_rows[
                (session_date, security_id)
            ],
        }
    )
    with pytest.raises(MassiveAdaptiveProfitabilityAuthorityV1Error):
        build_massive_adaptive_profitability_authority_v1(
            trace=original,
            forecast_archive=fixture.forecast_archive,
            calibration=fixture.calibration,
            inference_plan=fixture.inference_plan,
            decision_roots=fixture.roots,
            context_origins=fixture.contexts,
            fill_source=changed_fill,
            daily_input_authority=fixture.daily,
            identity_authority=fixture.identity,
        )


def test_compiler_authority_has_no_future_fill_input() -> None:
    import inspect

    from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v1 import (
        build_massive_adaptive_compiler_input_authority_v1,
    )

    assert (
        "fill_source"
        not in inspect.signature(
            build_massive_adaptive_compiler_input_authority_v1
        ).parameters
    )


def test_training_calibration_generic_reload_is_nonauthorizing(tmp_path) -> None:
    value = _calibration(
        tuple(f"SEC-{index:03d}" for index in range(100)), root=tmp_path
    )
    assert value.runtime_calibration_replayed
    assert value.loaded_source is not None
    generic = parse_massive_adaptive_forecast_calibration_v1(
        root=tmp_path, loaded_source=value.loaded_source
    )
    assert not generic.runtime_calibration_replayed
    assert not generic.development_calibration_authorized
