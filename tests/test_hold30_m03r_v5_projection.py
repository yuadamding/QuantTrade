"""Qualification for the M03R-only ensemble and constrained execution path."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from rl_quant.execution.hold30_m03r_projection_v5 import (
    M03R_ACTIVE_BETA_EXPOSURE_NAME,
    M03R_RISK_MANIFEST_SCHEMA,
    M03RAssetAlignedBook,
    M03RProjectionError,
    M03RProjectionNumerics,
    M03RQualifiedRiskManifest,
    M03RRiskManifest,
    bind_m03r_risk_manifest,
    execute_m03r_post_seed_ensemble,
    qualify_m03r_risk_manifest,
    validate_m03r_qualified_risk_manifest,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_m03r_ensemble_v5 import (
    M03REnsembleError,
    M03REnsembleMember,
    M03RSeedCheckpointBinding,
    aggregate_m03r_alpha_intents,
    bind_m03r_seed_checkpoint_ensemble_manifest,
    compute_m03r_asset_order_sha256,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    bind_m03r_confidence_calibration,
)

ASSETS = 121
CASH_ID = "CASH"
DATA_MANIFEST_SHA256 = "d" * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _asset_ids(permutation: torch.Tensor | None = None) -> tuple[str, ...]:
    values = (CASH_ID, *(f"S{index:03d}" for index in range(1, ASSETS)))
    if permutation is None:
        return values
    return tuple(values[index] for index in permutation.tolist())


def _members(
    *,
    permutation: torch.Tensor | None = None,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
    unavailable_indices: tuple[int, ...] = (),
    omit_exact_hold: bool = False,
    signal_confidence: float | None = None,
    force_exact_hold: bool = False,
) -> tuple[M03REnsembleMember, ...]:
    base = torch.linspace(-4.0, 4.0, ASSETS, dtype=torch.float64)
    base[0] = 0.0
    members: list[M03REnsembleMember] = []
    order_sha256 = compute_m03r_asset_order_sha256(_asset_ids(permutation))
    for seed in (19, 3, 41, 11, 29):
        confidence = (
            0.50 + seed / 1_000.0
            if signal_confidence is None
            else float(signal_confidence)
        )
        entry = (base + seed / 100.0).unsqueeze(0)
        mean = (0.01 * base + seed / 10_000.0).unsqueeze(0)
        downside = torch.full((1, ASSETS), 0.02 + seed / 100_000.0, dtype=torch.float64)
        raw_hazard = torch.full((1, ASSETS), 4.0 + seed / 100.0, dtype=torch.float64)
        if setting_id == "A08-fixed-exit-hazard":
            raw_hazard.zero_()
        exact_hold = torch.zeros((1, ASSETS), dtype=torch.float64)
        if force_exact_hold:
            exact_hold.fill_(1.0)
        exact_hold_logit = torch.where(
            exact_hold.bool(),
            torch.full_like(exact_hold, 20.0),
            torch.full_like(exact_hold, -20.0),
        )
        exact_hold_soft = torch.sigmoid(exact_hold_logit)
        ineligible = (0, *unavailable_indices)
        raw_hazard[:, ineligible] = 0.0
        exact_hold[:, ineligible] = 1.0
        exact_hold_logit[:, ineligible] = 0.0
        exact_hold_soft[:, ineligible] = 1.0
        hazard = 12.0 * torch.tanh(raw_hazard / 12.0)
        hazard[:, ineligible] = -12.0
        auxiliary = torch.stack(
            (mean[0], mean[0] * 2, mean[0] * 3, mean[0] * 4), dim=-1
        ).unsqueeze(0)
        checkpoint_sha256 = _digest(f"checkpoint-{setting_id}-{seed}")
        model_state_sha256 = _digest(f"model-state-{setting_id}-{seed}")
        raw_confidence_logit = torch.tensor(
            [
                -1_000.0
                if confidence <= 0.0
                else 1_000.0
                if confidence >= 1.0
                else float(torch.logit(torch.tensor(confidence, dtype=torch.float64)))
            ],
            dtype=torch.float64,
        )
        calibration = bind_m03r_confidence_calibration(
            setting_id=setting_id,
            seed=seed,
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=model_state_sha256,
            source_score_array_sha256=_digest(
                f"confidence-source-scores-{setting_id}-{seed}"
            ),
            source_target_array_sha256=_digest(
                f"confidence-source-targets-{setting_id}-{seed}"
            ),
            fit_fold_ids=("inner-00", "inner-01"),
            fit_start_trading_session="2021-01-04",
            fit_end_trading_session="2024-12-31",
            temperature=1.0,
            intercept=0.0,
            fit_observation_count=756,
            brier_score=0.20,
            expected_calibration_error=0.03,
            observed_target_rate=0.51,
        )
        if permutation is not None:
            entry = entry[:, permutation]
            mean = mean[:, permutation]
            downside = downside[:, permutation]
            raw_hazard = raw_hazard[:, permutation]
            hazard = hazard[:, permutation]
            exact_hold = exact_hold[:, permutation]
            exact_hold_logit = exact_hold_logit[:, permutation]
            exact_hold_soft = exact_hold_soft[:, permutation]
            auxiliary = auxiliary[:, permutation]
        members.append(
            M03REnsembleMember(
                protocol_generation=M03R_PROTOCOL_GENERATION,
                design_id=M03R_DESIGN_ID,
                setting_id=setting_id,
                seed=seed,
                checkpoint_sha256=checkpoint_sha256,
                model_state_sha256=model_state_sha256,
                asset_order_sha256=order_sha256,
                confidence_calibration_manifest_sha256=calibration.manifest_sha256,
                confidence_calibration_manifest=calibration,
                data_manifest_sha256=DATA_MANIFEST_SHA256,
                intent=Hold30Intent(
                    entry_scores=entry,
                    hazard_residual=hazard,
                    raw_hazard_residual=raw_hazard,
                    exact_hold_logit=None if omit_exact_hold else exact_hold_logit,
                    exact_hold_soft_probability=(
                        None if omit_exact_hold else exact_hold_soft
                    ),
                    exact_hold_decision_st=(None if omit_exact_hold else exact_hold),
                    exposure_residual=torch.zeros(1, dtype=torch.float64),
                    alpha_mean_30d=mean,
                    alpha_downside_30d=downside,
                    active_risk_scale=torch.tensor(
                        [0.04 * confidence], dtype=torch.float64
                    ),
                    signal_confidence=torch.tensor([confidence], dtype=torch.float64),
                    uncalibrated_signal_confidence_logit=raw_confidence_logit,
                    benchmark_derisk_request=torch.zeros(1, dtype=torch.float64),
                    auxiliary_alpha_mean=auxiliary,
                ),
            )
        )
    return tuple(members)


def _seed_manifest(
    members: tuple[M03REnsembleMember, ...],
    asset_ids: tuple[str, ...],
    *,
    setting_id: str,
):
    return bind_m03r_seed_checkpoint_ensemble_manifest(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=setting_id,
        asset_order_sha256=compute_m03r_asset_order_sha256(asset_ids),
        data_manifest_sha256=DATA_MANIFEST_SHA256,
        members=tuple(
            M03RSeedCheckpointBinding(
                seed=member.seed,
                checkpoint_sha256=member.checkpoint_sha256,
                model_state_sha256=member.model_state_sha256,
                confidence_calibration_manifest_sha256=(
                    member.confidence_calibration_manifest_sha256
                ),
            )
            for member in sorted(members, key=lambda item: item.seed)
        ),
    )


def _aggregate(
    members: tuple[M03REnsembleMember, ...],
    available: torch.Tensor,
    *,
    asset_ids: tuple[str, ...] | None = None,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
):
    axis = _asset_ids() if asset_ids is None else asset_ids
    manifest = _seed_manifest(members, axis, setting_id=setting_id)
    return aggregate_m03r_alpha_intents(
        members,
        available,
        axis,
        manifest,
        expected_seed_checkpoint_manifest_sha256=manifest.manifest_sha256,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=setting_id,
    )


def _call_aggregate(
    members: tuple[M03REnsembleMember, ...],
    available: torch.Tensor,
    *,
    protocol_generation: str = M03R_PROTOCOL_GENERATION,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
):
    axis = _asset_ids()
    manifest = _seed_manifest(members, axis, setting_id=setting_id)
    return aggregate_m03r_alpha_intents(
        members,
        available,
        axis,
        manifest,
        expected_seed_checkpoint_manifest_sha256=manifest.manifest_sha256,
        protocol_generation=protocol_generation,
        design_id=M03R_DESIGN_ID,
        setting_id=setting_id,
    )


def _risk_manifest(
    *,
    permutation: torch.Tensor | None = None,
    exposure_bound: float = 0.01,
    covariance_variance: float = 0.04,
    covariance_override: torch.Tensor | None = None,
) -> M03RRiskManifest:
    asset_ids = _asset_ids()
    risky_index = torch.arange(ASSETS - 1, dtype=torch.float64)
    rows = [
        torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                torch.ones(ASSETS - 1, dtype=torch.float64),
            )
        ),
        torch.cat(
            (
                torch.zeros(1, dtype=torch.float64),
                torch.where(risky_index < 60, 1.0, -1.0),
            )
        ),
        torch.cat(
            (torch.zeros(1, dtype=torch.float64), torch.linspace(-1.0, 1.0, ASSETS - 1))
        ),
        torch.cat((torch.zeros(1, dtype=torch.float64), torch.sin(risky_index + 0.3))),
        torch.cat((torch.zeros(1, dtype=torch.float64), torch.cos(risky_index + 0.7))),
        torch.cat(
            (torch.zeros(1, dtype=torch.float64), torch.sin(0.3 * risky_index + 0.2))
        ),
        torch.cat(
            (torch.zeros(1, dtype=torch.float64), torch.cos(0.2 * risky_index + 0.4))
        ),
    ]
    loadings = torch.stack(rows)
    covariance = torch.eye(ASSETS, dtype=torch.float64) * covariance_variance
    covariance[0, 0] = 0.0
    if covariance_override is not None:
        covariance = covariance_override.detach().clone().to(dtype=torch.float64)
    names = (
        M03R_ACTIVE_BETA_EXPOSURE_NAME,
        "sector:technology",
        "factor:size",
        "factor:momentum",
        "factor:value",
        "factor:volatility",
        "factor:liquidity",
    )
    families = (
        "market",
        "sector",
        "size",
        "momentum",
        "value",
        "volatility",
        "liquidity",
    )
    lower = torch.full((len(names),), -exposure_bound, dtype=torch.float64)
    upper = torch.full((len(names),), exposure_bound, dtype=torch.float64)
    lower[0], upper[0] = -0.10, 0.10
    if permutation is not None:
        asset_ids = tuple(asset_ids[index] for index in permutation.tolist())
        loadings = loadings[:, permutation]
        covariance = covariance[permutation][:, permutation]
    return bind_m03r_risk_manifest(
        schema=M03R_RISK_MANIFEST_SCHEMA,
        as_of_trading_session="2025-12-31",
        asset_ids=asset_ids,
        exposure_names=names,
        exposure_families=families,
        exposure_units=("unit-beta", *("normalized-loading" for _ in names[1:])),
        exposure_normalization_ids=tuple("pit-zscore-v1" for _ in names),
        exposure_estimation_window_trading_sessions=252,
        missing_value_policy="fail-closed",
        covariance_estimator_id="sample-covariance-v1",
        covariance_shrinkage_id="none",
        covariance_return_convention="daily-simple-return",
        stale_loading_policy="same-session-required",
        infeasibility_policy="fail-closed-no-artifact",
        exposure_loadings=loadings,
        exposure_lower_bounds=lower,
        exposure_upper_bounds=upper,
        daily_return_covariance=covariance,
        annual_tracking_error_ceiling=0.06,
        maximum_risky_asset_weight=0.01,
    )


def _book(permutation: torch.Tensor | None = None) -> M03RAssetAlignedBook:
    benchmark = torch.zeros(ASSETS, dtype=torch.float64)
    benchmark[1:] = 1.0 / (ASSETS - 1)
    current = benchmark.clone()
    available = torch.ones(ASSETS, dtype=torch.bool)
    ledger = torch.zeros((ASSETS, 61), dtype=torch.float64)
    ledger[1:, 35] = current[1:]
    axis = _asset_ids(permutation)
    if permutation is not None:
        current = current[permutation]
        benchmark = benchmark[permutation]
        available = available[permutation]
        ledger = ledger[permutation]
    return M03RAssetAlignedBook(
        decision_trading_session="2025-12-31",
        asset_ids=axis,
        asset_order_sha256=compute_m03r_asset_order_sha256(axis),
        current_weights=current,
        benchmark_weights=benchmark,
        decision_available=available,
        age_notional=ledger,
    )


def _execute(
    *,
    members: tuple[M03REnsembleMember, ...] | None = None,
    manifest: M03RRiskManifest | None = None,
    book: M03RAssetAlignedBook | None = None,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
    turnover: float = 0.10,
    numerics: M03RProjectionNumerics | None = None,
):
    risk = _risk_manifest() if manifest is None else manifest
    aligned = _book() if book is None else book
    seeded = _members(setting_id=setting_id) if members is None else members
    seed_manifest = _seed_manifest(seeded, aligned.asset_ids, setting_id=setting_id)
    qualified_risk = qualify_m03r_risk_manifest(
        risk,
        expected_manifest_sha256=risk.manifest_sha256,
    )
    return execute_m03r_post_seed_ensemble(
        seeded,
        aligned,
        qualified_risk,
        seed_manifest,
        expected_risk_manifest_sha256=risk.manifest_sha256,
        expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=setting_id,
        cash_asset_id=CASH_ID,
        maximum_one_way_turnover=turnover,
        numerics=M03RProjectionNumerics() if numerics is None else numerics,
    )


def test_m03r_alpha_ensemble_is_explicit_seed_order_independent_and_exact_identity() -> (
    None
):
    available = torch.ones((1, ASSETS), dtype=torch.bool)
    members = _members()
    first = _aggregate(members, available)
    second = _aggregate(tuple(reversed(members)), available)
    assert first.ordered_seeds == (3, 11, 19, 29, 41)
    assert torch.equal(first.intent.entry_scores, second.intent.entry_scores)
    assert torch.equal(first.intent.hazard_residual, second.intent.hazard_residual)
    assert first.intent.entry_scores is not None
    assert first.intent.entry_scores[0, 0].item() == 0.0
    assert first.intent.signal_confidence is not None
    assert first.intent.signal_confidence.item() == pytest.approx(0.519)

    with pytest.raises(ValueError, match="V3 remains"):
        _call_aggregate(
            members,
            available,
            protocol_generation="prelockbox-hold30-alpha-mech8-v3",
        )
    with pytest.raises(M03REnsembleError, match="no residual-alpha heads"):
        _call_aggregate(
            members,
            available,
            setting_id="M02-active-risk-no-alpha-heads",
        )
    mislabeled = (
        replace(members[0], setting_id="A04-no-downside-score-adjustment"),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="exact requested setting identity"):
        _call_aggregate(
            mislabeled,
            available,
        )


def test_m03r_ensemble_matches_actual_sentinels_binary_hold_and_confidence_scale() -> (
    None
):
    available = torch.ones((1, ASSETS), dtype=torch.bool)
    available[:, -1] = False
    members = _members(unavailable_indices=(ASSETS - 1,))
    aggregate = _aggregate(members, available).intent
    assert aggregate.raw_hazard_residual is not None
    assert aggregate.hazard_residual is not None
    assert aggregate.exact_hold_decision_st is not None
    assert torch.equal(
        aggregate.raw_hazard_residual[0, [0, ASSETS - 1]],
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.hazard_residual[0, [0, ASSETS - 1]],
        torch.full((2,), -12.0, dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.exact_hold_decision_st[0, [0, ASSETS - 1]],
        torch.ones(2, dtype=torch.float64),
    )
    assert aggregate.signal_confidence is not None
    assert aggregate.active_risk_scale is not None
    assert torch.allclose(
        aggregate.active_risk_scale,
        0.04 * aggregate.signal_confidence,
        atol=1e-12,
        rtol=1e-12,
    )

    missing_hold = tuple(
        replace(
            member,
            intent=replace(
                member.intent,
                exact_hold_logit=None,
                exact_hold_soft_probability=None,
                exact_hold_decision_st=None,
            ),
        )
        for member in members
    )
    with pytest.raises(M03REnsembleError, match="hard ST decision"):
        _aggregate(
            missing_hold,
            available,
        )
    decision = members[0].intent.exact_hold_decision_st.clone()
    decision[0, 1] = 0.5
    soft_hold = (
        replace(
            members[0],
            intent=replace(members[0].intent, exact_hold_decision_st=decision),
        ),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="hard binary"):
        _aggregate(
            soft_hold,
            available,
        )
    bad_scale = (
        replace(
            members[0],
            intent=replace(
                members[0].intent,
                active_risk_scale=members[0].intent.active_risk_scale + 0.001,
            ),
        ),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match=r"0.04 \* signal_confidence"):
        _aggregate(
            bad_scale,
            available,
        )
    forged_calibrated_output = (
        replace(
            members[0],
            intent=replace(
                members[0].intent,
                signal_confidence=members[0].intent.signal_confidence + 0.01,
                active_risk_scale=members[0].intent.active_risk_scale + 0.0004,
            ),
        ),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="bound calibrator"):
        _aggregate(forged_calibrated_output, available)

    a08 = _aggregate(
        _members(setting_id="A08-fixed-exit-hazard", omit_exact_hold=True),
        torch.ones((1, ASSETS), dtype=torch.bool),
        setting_id="A08-fixed-exit-hazard",
    )
    assert a08.intent.exact_hold_decision_st is None


def test_projection_enforces_conservation_caps_factor_beta_and_tracking_error() -> None:
    result = _execute(
        manifest=_risk_manifest(exposure_bound=0.002, covariance_variance=0.04)
    )
    projected = result.projected_weights
    executed = result.executed_weights
    assert projected.sum().item() == pytest.approx(1.0, abs=2e-9)
    assert executed.sum().item() == pytest.approx(1.0, abs=2e-9)
    assert float(projected.min()) >= -2e-9
    assert float(executed.min()) >= -2e-9
    assert float(projected[1:].max()) <= 0.01 + 2e-9
    assert float(executed[1:].max()) <= 0.01 + 2e-9
    diagnostics = result.diagnostics
    assert diagnostics.projection_application_count == 2
    assert diagnostics.solver_converged
    assert diagnostics.maximum_final_violation <= 2e-9
    assert abs(float(diagnostics.final_active_exposures[0])) <= 0.10 + 2e-9
    assert bool((diagnostics.final_active_exposures[1:].abs() <= 0.002 + 2e-9).all())
    assert diagnostics.projected_annual_tracking_error <= 0.06 + 2e-9
    assert diagnostics.final_annual_tracking_error <= 0.06 + 2e-9
    assert (
        diagnostics.requested_annual_tracking_error
        > diagnostics.projected_annual_tracking_error
    )


def test_strong_learned_hazard_executes_at_zero_confidence_without_replacement() -> None:
    result = _execute(members=_members(signal_confidence=0.0))
    diagnostics = result.diagnostics
    assert diagnostics.preferred_annual_tracking_error_cap == 0.0
    assert diagnostics.confidence_tracking_error_scale == 0.0
    assert diagnostics.confidence_limited_incremental_active_risk == 0.0
    assert diagnostics.confidence_limited_incremental_one_way_turnover == 0.0
    assert result.executed_learned_hazard_sell_notional.sum() > 0.0
    assert result.executed_entry_buy_notional.sum() == 0.0
    torch.testing.assert_close(
        result.projected_weights,
        result.projected_hazard_anchor_weights,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        result.executed_learned_hazard_sale_age_notional.sum(dim=-1),
        result.executed_learned_hazard_sell_notional,
        atol=2e-9,
        rtol=2e-9,
    )
    assert diagnostics.projected_annual_tracking_error <= 0.06 + 2e-9
    assert diagnostics.maximum_final_violation <= 2e-9


def test_zero_confidence_carries_existing_feasible_active_book() -> None:
    base = _book()
    current = base.current_weights.clone()
    current[1] += 0.001
    current[2] -= 0.001
    ledger = torch.zeros_like(base.age_notional)
    ledger[1:, 35] = current[1:]
    active_book = replace(
        base,
        current_weights=current,
        age_notional=ledger,
    )
    result = _execute(
        members=_members(signal_confidence=0.0, force_exact_hold=True),
        manifest=_risk_manifest(exposure_bound=0.10, covariance_variance=0.001),
        book=active_book,
    )
    assert result.diagnostics.preferred_annual_tracking_error_cap == 0.0
    assert result.diagnostics.effective_annual_tracking_error_cap == pytest.approx(
        result.diagnostics.repaired_current_annual_tracking_error
    )
    assert result.executed_learned_hazard_sell_notional.sum() == 0.0
    assert result.executed_entry_buy_notional.sum() == 0.0
    assert torch.allclose(result.requested_weights, current, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result.projected_weights, current, atol=1e-10, rtol=1e-10)
    assert torch.allclose(result.executed_weights, current, atol=1e-10, rtol=1e-10)


def test_intermediate_confidence_caps_projected_te_below_hard_safety_ceiling() -> None:
    result = _execute(
        members=_members(signal_confidence=0.25),
        manifest=_risk_manifest(exposure_bound=0.01, covariance_variance=0.04),
    )
    diagnostics = result.diagnostics
    assert diagnostics.preferred_annual_tracking_error_cap == pytest.approx(0.01)
    assert diagnostics.effective_annual_tracking_error_cap == pytest.approx(
        min(0.06, diagnostics.hazard_anchor_annual_tracking_error + 0.01)
    )
    assert 0 < diagnostics.confidence_tracking_error_scale < 1
    assert diagnostics.confidence_limited_incremental_active_risk <= 0.01 + 2e-9
    assert diagnostics.projected_annual_tracking_error <= 0.06 + 2e-9


def test_confidence_changes_risk_budget_but_not_cross_sectional_entry_scores() -> None:
    low = _execute(members=_members(signal_confidence=0.25))
    high = _execute(members=_members(signal_confidence=1.0))
    assert torch.allclose(
        low.requested_weights,
        high.requested_weights,
        atol=1e-12,
        rtol=1e-12,
    )
    assert (
        low.diagnostics.effective_annual_tracking_error_cap
        < high.diagnostics.effective_annual_tracking_error_cap
    )


def test_turnover_interpolation_preserves_feasibility() -> None:
    result = _execute(turnover=0.001)
    diagnostics = result.diagnostics
    assert diagnostics.requested_one_way_turnover > 0.001
    assert diagnostics.executed_one_way_turnover == pytest.approx(0.001, abs=2e-10)
    assert 0 < diagnostics.turnover_interpolation_scale < 1
    assert diagnostics.maximum_final_violation <= 2e-9
    assert diagnostics.final_annual_tracking_error <= 0.06 + 2e-9


def test_drifted_cap_and_factor_book_is_repaired_before_discretionary_intent() -> None:
    base_book = _book()
    current = base_book.current_weights.clone()
    benchmark = base_book.benchmark_weights
    current[1] += 0.01
    current[61] = 0.0
    current[62] -= 0.01 - float(benchmark[61])
    assert current.sum().item() == pytest.approx(1.0)
    assert current[1].item() > 0.01
    ledger = torch.zeros((ASSETS, 61), dtype=torch.float64)
    ledger[1:, 35] = current[1:]
    result = _execute(
        manifest=_risk_manifest(exposure_bound=0.002),
        book=replace(base_book, current_weights=current, age_notional=ledger),
    )
    diagnostics = result.diagnostics
    assert diagnostics.risk_forced_repair_application_count == 1
    assert diagnostics.risk_forced_repair_solver_converged
    assert diagnostics.risk_forced_repair_solver_iterations > 0
    assert diagnostics.pre_repair_maximum_linear_violation > 0
    assert diagnostics.risk_forced_repair_maximum_violation <= 2e-9
    assert diagnostics.risk_forced_repair_one_way_turnover > 0
    assert diagnostics.risk_forced_repair_sell_notional == pytest.approx(
        diagnostics.risk_forced_repair_buy_notional, abs=2e-9
    )
    assert diagnostics.projection_application_count == 2
    repaired = result.repaired_current_weights
    assert repaired.sum().item() == pytest.approx(1.0, abs=2e-9)
    assert float(repaired[1:].max()) <= 0.01 + 2e-9
    assert bool((diagnostics.repaired_active_exposures[1:].abs() <= 0.002 + 2e-9).all())
    assert diagnostics.repaired_current_annual_tracking_error <= 0.06 + 2e-9
    assert torch.allclose(
        result.repaired_age_notional.sum(dim=-1)[1:],
        repaired[1:].clamp_min(0.0),
        atol=2e-9,
        rtol=2e-9,
    )
    assert float(result.repaired_age_notional[1:, 0].sum()) > 0
    assert float(result.repaired_age_notional[1:, 35].sum()) < 1.0
    assert diagnostics.executed_one_way_turnover <= 0.10 + 2e-9


def test_execution_is_deterministic_and_asset_permutation_equivariant() -> None:
    first = _execute()
    repeated = _execute()
    assert torch.equal(first.requested_weights, repeated.requested_weights)
    assert torch.equal(first.projected_weights, repeated.projected_weights)
    assert torch.equal(first.executed_weights, repeated.executed_weights)

    generator = torch.Generator().manual_seed(812)
    permutation = torch.randperm(ASSETS, generator=generator)
    permuted = _execute(
        members=_members(permutation=permutation),
        manifest=_risk_manifest(permutation=permutation),
        book=_book(permutation=permutation),
    )
    assert torch.allclose(
        permuted.requested_weights,
        first.requested_weights[permutation],
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.allclose(
        permuted.projected_weights,
        first.projected_weights[permutation],
        atol=2e-10,
        rtol=2e-10,
    )
    assert torch.allclose(
        permuted.executed_weights,
        first.executed_weights[permutation],
        atol=2e-10,
        rtol=2e-10,
    )


def test_execution_cause_deltas_telescope_and_final_age_ledger_conserves() -> None:
    base = _book()
    current = base.current_weights.clone()
    benchmark = base.benchmark_weights.clone()
    available = base.decision_available.clone()
    ledger = base.age_notional.clone()
    available[-1] = False
    benchmark[0] += benchmark[-1]
    benchmark[-1] = 0.0
    stale_book = replace(
        base,
        current_weights=current,
        benchmark_weights=benchmark,
        decision_available=available,
        age_notional=ledger,
    )
    result = _execute(
        members=_members(unavailable_indices=(ASSETS - 1,)),
        book=stale_book,
    )
    attributed = sum(
        (
            result.unavailable_delta,
            result.risk_repair_delta,
            result.hazard_release_delta,
            result.benchmark_derisk_delta,
            result.hazard_anchor_factor_projection_delta,
            result.hazard_anchor_tracking_error_projection_delta,
            result.entry_reallocation_delta,
            result.factor_projection_delta,
            result.tracking_error_projection_delta,
            result.confidence_budget_delta,
            result.turnover_truncation_delta,
        ),
        torch.zeros_like(current),
    )
    assert torch.allclose(
        current + attributed,
        result.executed_weights,
        atol=2e-8,
        rtol=2e-8,
    )
    assert result.unavailable_delta[-1].item() == pytest.approx(-current[-1].item())
    assert result.unavailable_delta[0].item() == pytest.approx(current[-1].item())
    assert torch.allclose(
        result.final_age_notional.sum(dim=-1)[1:],
        result.executed_weights[1:].clamp_min(0.0),
        atol=2e-8,
        rtol=2e-8,
    )
    assert result.final_age_notional[0].abs().sum().item() == 0.0
    assert torch.allclose(
        result.executed_learned_hazard_sell_notional
        + result.executed_benchmark_derisk_sell_notional
        + result.executed_projection_sell_notional,
        (result.repaired_current_weights - result.executed_weights).clamp_min(0.0),
        atol=2e-9,
        rtol=2e-9,
    )
    assert torch.allclose(
        result.executed_entry_buy_notional + result.executed_projection_buy_notional,
        (result.executed_weights - result.repaired_current_weights).clamp_min(0.0),
        atol=2e-9,
        rtol=2e-9,
    )
    assert torch.allclose(
        result.executed_learned_hazard_sale_age_notional.sum(dim=-1),
        result.executed_learned_hazard_sell_notional,
        atol=2e-9,
        rtol=2e-9,
    )


def test_asset_order_and_seed_checkpoint_lineage_fail_closed() -> None:
    book = _book()
    risk = _risk_manifest()
    members = _members()
    seed_manifest = _seed_manifest(
        members,
        book.asset_ids,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    with pytest.raises(M03REnsembleError, match="distinct calibrator"):
        bind_m03r_seed_checkpoint_ensemble_manifest(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            asset_order_sha256=seed_manifest.asset_order_sha256,
            data_manifest_sha256=seed_manifest.data_manifest_sha256,
            members=tuple(
                replace(
                    binding,
                    confidence_calibration_manifest_sha256="c" * 64,
                )
                for binding in seed_manifest.members
            ),
        )
    generator = torch.Generator().manual_seed(19)
    permutation = torch.randperm(ASSETS, generator=generator)
    with pytest.raises(M03RProjectionError, match="book axis"):
        _execute(
            book=_book(permutation),
            manifest=risk,
            members=_members(permutation=permutation),
        )

    mutated = (
        replace(members[0], checkpoint_sha256=_digest("different-checkpoint")),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="do not match"):
        execute_m03r_post_seed_ensemble(
            mutated,
            book,
            qualify_m03r_risk_manifest(
                risk,
                expected_manifest_sha256=risk.manifest_sha256,
            ),
            seed_manifest,
            expected_risk_manifest_sha256=risk.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )

    wrong_calibrator = (
        replace(
            members[0],
            confidence_calibration_manifest_sha256=_digest(
                "calibrator-from-another-seed"
            ),
        ),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="do not match"):
        execute_m03r_post_seed_ensemble(
            wrong_calibrator,
            book,
            qualify_m03r_risk_manifest(
                risk,
                expected_manifest_sha256=risk.manifest_sha256,
            ),
            seed_manifest,
            expected_risk_manifest_sha256=risk.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )


def test_risk_manifest_hash_binds_estimation_and_missing_value_semantics() -> None:
    manifest = _risk_manifest()
    changed = bind_m03r_risk_manifest(
        **{
            **{
                field: getattr(manifest, field)
                for field in M03RRiskManifest.__dataclass_fields__
                if field != "manifest_sha256"
            },
            "covariance_shrinkage_id": "ledoit-wolf-v1",
        }
    )
    assert changed.manifest_sha256 != manifest.manifest_sha256
    with pytest.raises(M03RProjectionError, match="content hash"):
        qualify_m03r_risk_manifest(
            changed,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_qualified_risk_manifest_rejects_post_qualification_tensor_mutation() -> None:
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    qualified.manifest.exposure_loadings[0, 1] += 1.0
    members = _members()
    book = _book()
    seed_manifest = _seed_manifest(
        members,
        book.asset_ids,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    with pytest.raises(M03RProjectionError, match="changed after qualification"):
        execute_m03r_post_seed_ensemble(
            members,
            book,
            qualified,
            seed_manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )


def test_qualified_risk_manifest_rejects_data_mutation_without_version_change() -> (
    None
):
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    tensor = qualified.manifest.exposure_loadings
    version = int(tensor._version)
    tensor.data[0, 1] += 1.0
    assert int(tensor._version) == version

    with pytest.raises(M03RProjectionError, match="tensor content changed"):
        validate_m03r_qualified_risk_manifest(
            qualified,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_qualified_risk_manifest_rejects_shared_numpy_view_mutation() -> None:
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    tensor = qualified.manifest.exposure_loadings
    version = int(tensor._version)
    shared = tensor.numpy()
    shared[0, 1] += 1.0
    assert int(tensor._version) == version

    with pytest.raises(M03RProjectionError, match="tensor content changed"):
        validate_m03r_qualified_risk_manifest(
            qualified,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_qualified_risk_manifest_rejects_cached_factor_data_mutation() -> None:
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    version = int(qualified.covariance_factor._version)
    qualified.covariance_factor.data[1, 1] += 1.0
    assert int(qualified.covariance_factor._version) == version

    with pytest.raises(M03RProjectionError, match="tensor content changed"):
        validate_m03r_qualified_risk_manifest(
            qualified,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_qualified_risk_revalidation_does_not_repeat_eigendecomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_eigh = torch.linalg.eigh

    def counted_eigh(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counted_eigh)
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    validate_m03r_qualified_risk_manifest(
        qualified,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    validate_m03r_qualified_risk_manifest(
        qualified,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    assert calls == 1


def test_fabricated_or_replaced_qualified_wrapper_cannot_reuse_capability() -> None:
    manifest = _risk_manifest()
    qualified = qualify_m03r_risk_manifest(
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    forged = replace(
        qualified,
        sorted_asset_indices=tuple(reversed(qualified.sorted_asset_indices)),
    )
    members = _members()
    book = _book()
    seed_manifest = _seed_manifest(
        members,
        book.asset_ids,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    with pytest.raises(M03RProjectionError, match="qualification capability"):
        execute_m03r_post_seed_ensemble(
            members,
            book,
            forged,
            seed_manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )
    copied_wrapper = replace(qualified)
    with pytest.raises(M03RProjectionError, match="qualification capability"):
        validate_m03r_qualified_risk_manifest(
            copied_wrapper,
            expected_manifest_sha256=manifest.manifest_sha256,
        )
    replaced_inner = replace(qualified, manifest=replace(qualified.manifest))
    with pytest.raises(M03RProjectionError, match="qualification capability"):
        execute_m03r_post_seed_ensemble(
            members,
            book,
            replaced_inner,
            seed_manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )
    manually_constructed = M03RQualifiedRiskManifest(
        manifest=qualified.manifest,
        manifest_sha256=qualified.manifest_sha256,
        asset_order_sha256=qualified.asset_order_sha256,
        sorted_asset_indices=qualified.sorted_asset_indices,
        covariance_factor=qualified.covariance_factor,
        tensor_versions=qualified.tensor_versions,
        _qualification_capability=None,
    )
    with pytest.raises(M03RProjectionError, match="qualification capability"):
        execute_m03r_post_seed_ensemble(
            members,
            book,
            manually_constructed,
            seed_manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )


def test_projection_fails_closed_for_invalid_hash_covariance_book_and_identity() -> (
    None
):
    manifest = _risk_manifest()
    book = _book()
    current = book.current_weights
    members = _members()
    seed_manifest = _seed_manifest(
        members,
        book.asset_ids,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    with pytest.raises(M03RProjectionError, match="content hash"):
        qualify_m03r_risk_manifest(
            manifest,
            expected_manifest_sha256="f" * 64,
        )

    bad_covariance = manifest.daily_return_covariance.clone()
    bad_covariance[1, 1] = -0.5
    invalid_covariance = bind_m03r_risk_manifest(
        **{
            **{
                field: getattr(manifest, field)
                for field in M03RRiskManifest.__dataclass_fields__
                if field != "manifest_sha256"
            },
            "daily_return_covariance": bad_covariance,
        }
    )
    with pytest.raises(M03RProjectionError, match="positive semidefinite"):
        _execute(manifest=invalid_covariance)

    infeasible_current = current.clone()
    infeasible_current[1] += 0.02
    infeasible_current[2] -= 0.02
    with pytest.raises(M03RProjectionError, match="long-only unit portfolio"):
        _execute(book=replace(book, current_weights=infeasible_current))
    with pytest.raises(ValueError, match="A10"):
        _execute(setting_id="A10-no-factor-neutral-projection")
    with pytest.raises(ValueError, match="A05"):
        _execute(setting_id="A05-fixed-te-floor")
    with pytest.raises(ValueError, match="V3 remains"):
        execute_m03r_post_seed_ensemble(
            members,
            book,
            qualify_m03r_risk_manifest(
                manifest,
                expected_manifest_sha256=manifest.manifest_sha256,
            ),
            seed_manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
            expected_seed_checkpoint_manifest_sha256=seed_manifest.manifest_sha256,
            protocol_generation="prelockbox-hold30-alpha-mech8-v3",
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
        )


def test_projection_fails_closed_when_solver_does_not_converge() -> None:
    with pytest.raises(M03RProjectionError, match="did not converge"):
        _execute(
            manifest=_risk_manifest(exposure_bound=1e-5),
            numerics=M03RProjectionNumerics(tolerance=1e-12, maximum_iterations=1),
        )


def test_low_confidence_continuously_limits_rotation_of_an_active_book() -> None:
    base = _book()
    current = base.current_weights.clone()
    current[1] += 0.001
    current[2] -= 0.001
    ledger = torch.zeros_like(base.age_notional)
    ledger[1:, 35] = current[1:]
    active_book = replace(base, current_weights=current, age_notional=ledger)
    risk = _risk_manifest(exposure_bound=0.10, covariance_variance=0.001)

    zero = _execute(
        members=_members(signal_confidence=0.0),
        manifest=risk,
        book=active_book,
    )
    epsilon = _execute(
        members=_members(signal_confidence=1e-6),
        manifest=risk,
        book=active_book,
    )
    high = _execute(
        members=_members(signal_confidence=1.0),
        manifest=risk,
        book=active_book,
    )

    torch.testing.assert_close(
        zero.projected_weights,
        zero.projected_hazard_anchor_weights,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        epsilon.requested_weights,
        high.requested_weights,
        atol=1e-12,
        rtol=1e-12,
    )
    assert zero.diagnostics.confidence_incremental_risk_scale == 0.0
    assert zero.diagnostics.confidence_limited_incremental_active_risk == 0.0
    assert epsilon.diagnostics.confidence_limited_incremental_active_risk <= (
        0.04e-6 + 2e-9
    )
    epsilon_move = torch.linalg.vector_norm(
        epsilon.projected_weights - zero.projected_weights
    )
    high_move = torch.linalg.vector_norm(high.projected_weights - zero.projected_weights)
    assert 0 < float(epsilon_move) < float(high_move) * 1e-3
    assert (
        abs(
            float(epsilon.executed_learned_hazard_sell_notional.sum())
            - float(zero.executed_learned_hazard_sell_notional.sum())
        )
        <= 1e-6 + 2e-9
    )


def test_low_confidence_l1_cap_blocks_rotation_in_covariance_nullspace() -> None:
    common_factor = torch.ones(ASSETS, dtype=torch.float64)
    covariance = 0.001 * torch.outer(common_factor, common_factor)
    singular = _risk_manifest(
        exposure_bound=0.10,
        covariance_override=covariance,
    )
    epsilon = _execute(
        members=_members(signal_confidence=1e-6),
        manifest=singular,
        turnover=1.0,
    )
    assert epsilon.diagnostics.requested_incremental_active_risk <= 1e-10
    assert epsilon.diagnostics.requested_incremental_one_way_turnover > 1e-3
    assert epsilon.diagnostics.confidence_one_way_turnover_scale < 1e-3
    assert epsilon.diagnostics.confidence_limited_incremental_one_way_turnover <= (
        1e-6 + 2e-9
    )


def test_unavailable_and_risk_repair_accounting_are_disjoint() -> None:
    base = _book()
    current = base.current_weights.clone()
    benchmark = base.benchmark_weights.clone()
    available = base.decision_available.clone()
    available[-1] = False
    benchmark[0] += benchmark[-1]
    benchmark[-1] = 0.0
    unavailable_only = _execute(
        members=_members(unavailable_indices=(ASSETS - 1,), force_exact_hold=True),
        book=replace(
            base,
            benchmark_weights=benchmark,
            decision_available=available,
        ),
    )
    diagnostics = unavailable_only.diagnostics
    assert diagnostics.unavailable_forced_one_way_turnover == pytest.approx(current[-1])
    assert diagnostics.unavailable_forced_sell_notional == pytest.approx(current[-1])
    assert diagnostics.unavailable_forced_buy_notional == pytest.approx(current[-1])
    assert diagnostics.risk_forced_repair_one_way_turnover == 0.0
    assert diagnostics.risk_forced_repair_sell_notional == 0.0
    assert diagnostics.risk_forced_repair_buy_notional == 0.0
    torch.testing.assert_close(
        unavailable_only.executed_unavailable_sale_age_notional.sum(dim=-1),
        unavailable_only.executed_unavailable_sell_notional,
    )
    assert unavailable_only.executed_risk_repair_sale_age_notional.sum() == 0.0


def test_cause_specific_sale_age_tensors_partition_every_executed_sale() -> None:
    base = _book()
    current = base.current_weights.clone()
    current[1] += 0.01
    current[61] = 0.0
    current[62] -= 0.01 - float(base.benchmark_weights[61])
    ledger = torch.zeros_like(base.age_notional)
    ledger[1:, 35] = current[1:]
    result = _execute(
        manifest=_risk_manifest(exposure_bound=0.002),
        book=replace(base, current_weights=current, age_notional=ledger),
    )
    pairs = (
        (
            result.executed_unavailable_sale_age_notional,
            result.executed_unavailable_sell_notional,
        ),
        (
            result.executed_risk_repair_sale_age_notional,
            result.executed_risk_repair_sell_notional,
        ),
        (
            result.executed_learned_hazard_sale_age_notional,
            result.executed_learned_hazard_sell_notional,
        ),
        (
            result.executed_benchmark_derisk_sale_age_notional,
            result.executed_benchmark_derisk_sell_notional,
        ),
        (
            result.executed_projection_sale_age_notional,
            result.executed_projection_sell_notional,
        ),
    )
    for sale_age, sell_notional in pairs:
        torch.testing.assert_close(
            sale_age.sum(dim=-1),
            sell_notional,
            atol=1e-8,
            rtol=1e-8,
        )
    torch.testing.assert_close(
        result.executed_total_sale_age_notional,
        sum((sale_age for sale_age, _sell in pairs), torch.zeros_like(pairs[0][0])),
        atol=1e-10,
        rtol=1e-10,
    )
    assert result.executed_risk_repair_sale_age_notional.sum() > 0.0
