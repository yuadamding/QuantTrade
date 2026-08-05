"""Qualification for the M03R-only ensemble and constrained execution path."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution.hold30_m03r_projection import (
    M03R_ACTIVE_BETA_EXPOSURE_NAME,
    M03R_RISK_MANIFEST_SCHEMA,
    M03RProjectionError,
    M03RProjectionNumerics,
    M03RRiskManifest,
    bind_m03r_risk_manifest,
    execute_m03r_post_seed_ensemble,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_m03r_ensemble import (
    M03REnsembleError,
    M03REnsembleMember,
    aggregate_m03r_alpha_intents,
)
from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
)

ASSETS = 121
CASH_ID = "CASH"


def _members(
    *,
    permutation: torch.Tensor | None = None,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
    unavailable_indices: tuple[int, ...] = (),
    omit_exact_hold: bool = False,
    signal_confidence: float | None = None,
) -> tuple[M03REnsembleMember, ...]:
    base = torch.linspace(-4.0, 4.0, ASSETS, dtype=torch.float64)
    base[0] = 0.0
    members: list[M03REnsembleMember] = []
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
        ineligible = (0, *unavailable_indices)
        raw_hazard[:, ineligible] = 0.0
        exact_hold[:, ineligible] = 1.0
        hazard = 12.0 * torch.tanh(raw_hazard / 12.0)
        hazard[:, ineligible] = -12.0
        auxiliary = torch.stack(
            (mean[0], mean[0] * 2, mean[0] * 3, mean[0] * 4), dim=-1
        ).unsqueeze(0)
        if permutation is not None:
            entry = entry[:, permutation]
            mean = mean[:, permutation]
            downside = downside[:, permutation]
            raw_hazard = raw_hazard[:, permutation]
            hazard = hazard[:, permutation]
            exact_hold = exact_hold[:, permutation]
            auxiliary = auxiliary[:, permutation]
        members.append(
            M03REnsembleMember(
                protocol_generation=M03R_PROTOCOL_GENERATION,
                design_id=M03R_DESIGN_ID,
                setting_id=setting_id,
                seed=seed,
                intent=Hold30Intent(
                    entry_scores=entry,
                    hazard_residual=hazard,
                    raw_hazard_residual=raw_hazard,
                    exact_hold_probability=None if omit_exact_hold else exact_hold,
                    exposure_residual=torch.zeros(1, dtype=torch.float64),
                    alpha_mean_30d=mean,
                    alpha_downside_30d=downside,
                    active_risk_scale=torch.tensor(
                        [0.04 * confidence], dtype=torch.float64
                    ),
                    signal_confidence=torch.tensor([confidence], dtype=torch.float64),
                    auxiliary_alpha_mean=auxiliary,
                ),
            )
        )
    return tuple(members)


def _risk_manifest(
    *,
    permutation: torch.Tensor | None = None,
    exposure_bound: float = 0.01,
    covariance_variance: float = 0.04,
) -> M03RRiskManifest:
    asset_ids = (CASH_ID, *(f"S{index:03d}" for index in range(1, ASSETS)))
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
        exposure_loadings=loadings,
        exposure_lower_bounds=lower,
        exposure_upper_bounds=upper,
        daily_return_covariance=covariance,
        annual_tracking_error_ceiling=0.06,
        maximum_risky_asset_weight=0.01,
    )


def _books(permutation: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    benchmark = torch.zeros(ASSETS, dtype=torch.float64)
    benchmark[1:] = 1.0 / (ASSETS - 1)
    current = benchmark.clone()
    available = torch.ones(ASSETS, dtype=torch.bool)
    ledger = torch.zeros((ASSETS, 61), dtype=torch.float64)
    ledger[1:, 35] = current[1:]
    if permutation is not None:
        return (
            current[permutation],
            benchmark[permutation],
            available[permutation],
            ledger[permutation],
        )
    return current, benchmark, available, ledger


def _execute(
    *,
    members: tuple[M03REnsembleMember, ...] | None = None,
    manifest: M03RRiskManifest | None = None,
    books: tuple[torch.Tensor, ...] | None = None,
    setting_id: str = M03R_CANONICAL_SETTING_ID,
    turnover: float = 0.10,
    numerics: M03RProjectionNumerics | None = None,
):
    risk = _risk_manifest() if manifest is None else manifest
    current, benchmark, available, ledger = _books() if books is None else books
    return execute_m03r_post_seed_ensemble(
        _members(setting_id=setting_id) if members is None else members,
        current,
        benchmark,
        available,
        ledger,
        risk,
        expected_risk_manifest_sha256=risk.manifest_sha256,
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
    first = aggregate_m03r_alpha_intents(
        members,
        available,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    second = aggregate_m03r_alpha_intents(
        tuple(reversed(members)),
        available,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
    )
    assert first.ordered_seeds == (3, 11, 19, 29, 41)
    assert torch.equal(first.intent.entry_scores, second.intent.entry_scores)
    assert torch.equal(first.intent.hazard_residual, second.intent.hazard_residual)
    assert first.intent.entry_scores is not None
    assert first.intent.entry_scores[0, 0].item() == 0.0
    assert first.intent.signal_confidence is not None
    assert first.intent.signal_confidence.item() == pytest.approx(0.519)

    with pytest.raises(ValueError, match="V3 remains"):
        aggregate_m03r_alpha_intents(
            members,
            available,
            protocol_generation="prelockbox-hold30-alpha-mech8-v3",
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )
    with pytest.raises(M03REnsembleError, match="no residual-alpha heads"):
        aggregate_m03r_alpha_intents(
            members,
            available,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id="M02-active-risk-no-alpha-heads",
        )
    mislabeled = (
        replace(members[0], setting_id="A04-no-uncertainty-scaling"),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="exact requested setting identity"):
        aggregate_m03r_alpha_intents(
            mislabeled,
            available,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )


def test_m03r_ensemble_matches_actual_sentinels_binary_hold_and_confidence_scale() -> (
    None
):
    available = torch.ones((1, ASSETS), dtype=torch.bool)
    available[:, -1] = False
    members = _members(unavailable_indices=(ASSETS - 1,))
    aggregate = aggregate_m03r_alpha_intents(
        members,
        available,
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
    ).intent
    assert aggregate.raw_hazard_residual is not None
    assert aggregate.hazard_residual is not None
    assert aggregate.exact_hold_probability is not None
    assert torch.equal(
        aggregate.raw_hazard_residual[0, [0, ASSETS - 1]],
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.hazard_residual[0, [0, ASSETS - 1]],
        torch.full((2,), -12.0, dtype=torch.float64),
    )
    assert torch.equal(
        aggregate.exact_hold_probability[0, [0, ASSETS - 1]],
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
        replace(member, intent=replace(member.intent, exact_hold_probability=None))
        for member in members
    )
    with pytest.raises(M03REnsembleError, match="hard exact-hold"):
        aggregate_m03r_alpha_intents(
            missing_hold,
            available,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )
    soft = members[0].intent.exact_hold_probability.clone()
    soft[0, 1] = 0.5
    soft_hold = (
        replace(
            members[0], intent=replace(members[0].intent, exact_hold_probability=soft)
        ),
        *members[1:],
    )
    with pytest.raises(M03REnsembleError, match="hard binary"):
        aggregate_m03r_alpha_intents(
            soft_hold,
            available,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
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
        aggregate_m03r_alpha_intents(
            bad_scale,
            available,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
        )

    a08 = aggregate_m03r_alpha_intents(
        _members(setting_id="A08-fixed-exit-hazard", omit_exact_hold=True),
        torch.ones((1, ASSETS), dtype=torch.bool),
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id="A08-fixed-exit-hazard",
    )
    assert a08.intent.exact_hold_probability is None


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
    assert diagnostics.projection_application_count == 1
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


def test_zero_confidence_projects_exactly_to_benchmark_without_te_floor() -> None:
    _current, benchmark, _available, _ledger = _books()
    result = _execute(members=_members(signal_confidence=0.0))
    diagnostics = result.diagnostics
    assert diagnostics.preferred_annual_tracking_error_cap == 0.0
    assert diagnostics.effective_annual_tracking_error_cap == 0.0
    assert diagnostics.confidence_tracking_error_scale == 0.0
    assert diagnostics.projected_annual_tracking_error == 0.0
    assert torch.equal(result.projected_weights, benchmark)


def test_intermediate_confidence_caps_projected_te_below_hard_safety_ceiling() -> None:
    result = _execute(
        members=_members(signal_confidence=0.25),
        manifest=_risk_manifest(exposure_bound=0.01, covariance_variance=0.04),
    )
    diagnostics = result.diagnostics
    assert diagnostics.preferred_annual_tracking_error_cap == pytest.approx(0.01)
    assert diagnostics.effective_annual_tracking_error_cap == pytest.approx(0.01)
    assert 0 < diagnostics.confidence_tracking_error_scale < 1
    assert diagnostics.projected_annual_tracking_error == pytest.approx(0.01, abs=2e-9)
    assert diagnostics.projected_annual_tracking_error < 0.06


def test_turnover_interpolation_preserves_feasibility() -> None:
    result = _execute(turnover=0.001)
    diagnostics = result.diagnostics
    assert diagnostics.requested_one_way_turnover > 0.001
    assert diagnostics.executed_one_way_turnover == pytest.approx(0.001, abs=2e-10)
    assert 0 < diagnostics.turnover_interpolation_scale < 1
    assert diagnostics.maximum_final_violation <= 2e-9
    assert diagnostics.final_annual_tracking_error <= 0.06 + 2e-9


def test_drifted_cap_and_factor_book_is_repaired_before_discretionary_intent() -> None:
    current, benchmark, available, _ledger = _books()
    current[1] += 0.01
    current[61] = 0.0
    current[62] -= 0.01 - float(benchmark[61])
    assert current.sum().item() == pytest.approx(1.0)
    assert current[1].item() > 0.01
    ledger = torch.zeros((ASSETS, 61), dtype=torch.float64)
    ledger[1:, 35] = current[1:]
    result = _execute(
        manifest=_risk_manifest(exposure_bound=0.002),
        books=(current, benchmark, available, ledger),
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
    assert diagnostics.projection_application_count == 1
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
        books=_books(permutation=permutation),
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


def test_projection_fails_closed_for_invalid_hash_covariance_book_and_identity() -> (
    None
):
    manifest = _risk_manifest()
    current, benchmark, available, ledger = _books()
    with pytest.raises(M03RProjectionError, match="content hash"):
        execute_m03r_post_seed_ensemble(
            _members(),
            current,
            benchmark,
            available,
            ledger,
            manifest,
            expected_risk_manifest_sha256="f" * 64,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id=M03R_CANONICAL_SETTING_ID,
            cash_asset_id=CASH_ID,
            maximum_one_way_turnover=0.1,
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
        _execute(books=(infeasible_current, benchmark, available, ledger))
    with pytest.raises(ValueError, match="A10"):
        _execute(setting_id="A10-no-factor-neutral-projection")
    with pytest.raises(ValueError, match="A05"):
        _execute(setting_id="A05-fixed-te-floor")
    with pytest.raises(ValueError, match="V3 remains"):
        execute_m03r_post_seed_ensemble(
            _members(),
            current,
            benchmark,
            available,
            ledger,
            manifest,
            expected_risk_manifest_sha256=manifest.manifest_sha256,
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
