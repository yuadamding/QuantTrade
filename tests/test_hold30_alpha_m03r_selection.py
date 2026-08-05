"""Content binding, eligibility, and LCB ranking for M03R checkpoints."""

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    M03R_SUPERSEDED_PROTOCOL_GENERATION,
)
from rl_quant.training.hold30_alpha_m03r_selection import (
    M03RCheckpointCandidate,
    M03RCheckpointSelectionContract,
    M03RFoldSeed,
    M03RSelectionError,
    M03RValidationMetrics,
    build_m03r_checkpoint_candidate,
    select_m03r_checkpoint,
)


def _inventory() -> tuple[M03RFoldSeed, ...]:
    return tuple(
        M03RFoldSeed(f"fold-{fold:02d}", seed) for fold in range(6) for seed in range(5)
    )


def _contract(**changes: object) -> M03RCheckpointSelectionContract:
    fields = {
        "setting_id": M03R_CANONICAL_SETTING_ID,
        "expected_fold_seed_inventory": _inventory(),
        "inference_contract_sha256": "1" * 64,
        "source_arrays_sha256": "2" * 64,
        "minimum_notional_survival_at_20_sessions": 0.40,
        "minimum_notional_survival_at_30_sessions": 0.20,
        "minimum_restricted_mean_holding_time_through_60_sessions": 20.0,
        "maximum_restricted_mean_holding_time_through_60_sessions": 40.0,
        "minimum_discretionary_sold_notional": 1.0,
        "maximum_fold_censored_notional_fraction": 0.50,
        "maximum_requested_executed_projection_distance": 0.05,
        "maximum_forced_turnover_fraction": 0.10,
    }
    fields.update(changes)
    return M03RCheckpointSelectionContract(**fields)


def _metrics(update: int = 8, **changes: object) -> M03RValidationMetrics:
    row = M03RValidationMetrics(
        update=update,
        net_active_return_20bp=0.02,
        net_active_return_40bp=0.01,
        block_bootstrap_lcb95_net_active_return_20bp=0.001,
        annual_tracking_error=0.0,
        active_market_beta=0.0,
        notional_survival_at_20_sessions=0.60,
        notional_survival_at_30_sessions=0.30,
        restricted_mean_holding_time_through_60_sessions=30.0,
        discretionary_sold_notional=2.0,
        fold_censored_notional_fraction=0.20,
        requested_executed_projection_distance=0.01,
        forced_turnover_fraction=0.02,
        information_ratio_20bp=0.5,
        total_portfolio_sharpe_20bp=1.0,
        maximum_drawdown_20bp=0.1,
        turnover_cost_20bp=0.01,
    )
    return replace(row, **changes)


def _candidate(
    update: int = 8,
    *,
    contract: M03RCheckpointSelectionContract | None = None,
    inventory: tuple[M03RFoldSeed, ...] | None = None,
    bundle_character: str = "a",
    **metric_changes: object,
) -> M03RCheckpointCandidate:
    resolved = _contract() if contract is None else contract
    return build_m03r_checkpoint_candidate(
        contract=resolved,
        observed_fold_seed_inventory=(
            resolved.expected_fold_seed_inventory if inventory is None else inventory
        ),
        checkpoint_bundle_sha256=bundle_character * 64,
        metrics=_metrics(update, **metric_changes),
    )


def test_zero_tracking_error_is_eligible_when_every_other_gate_passes() -> None:
    contract = _contract()
    assert _candidate(contract=contract, annual_tracking_error=0.0).eligible(contract)


def test_survival_and_censoring_replace_sale_median_as_holding_gates() -> None:
    contract = _contract()
    assert not _candidate(
        contract=contract, notional_survival_at_30_sessions=0.19
    ).eligible(contract)
    assert not _candidate(
        contract=contract, fold_censored_notional_fraction=0.51
    ).eligible(contract)


def test_selection_ranks_by_active_return_lcb_before_point_return() -> None:
    contract = _contract()
    high_point_low_lcb = _candidate(
        8,
        contract=contract,
        bundle_character="a",
        net_active_return_20bp=0.10,
        block_bootstrap_lcb95_net_active_return_20bp=0.001,
    )
    lower_point_high_lcb = _candidate(
        16,
        contract=contract,
        bundle_character="b",
        net_active_return_20bp=0.03,
        block_bootstrap_lcb95_net_active_return_20bp=0.01,
    )
    selected = select_m03r_checkpoint(
        M03R_CANONICAL_SETTING_ID,
        (high_point_low_lcb, lower_point_high_lcb),
        contract=contract,
    )
    assert selected.checkpoint_bundle_sha256 == "b" * 64


def test_unresolved_result_moving_gates_fail_closed() -> None:
    unresolved = M03RCheckpointSelectionContract(
        setting_id=M03R_CANONICAL_SETTING_ID,
        expected_fold_seed_inventory=_inventory(),
        inference_contract_sha256="1" * 64,
        source_arrays_sha256="2" * 64,
    )
    with pytest.raises(M03RSelectionError, match="remain unresolved"):
        select_m03r_checkpoint(
            M03R_CANONICAL_SETTING_ID,
            (),
            contract=unresolved,
        )


def test_inventory_is_exact_not_a_caller_claimed_count_or_boolean() -> None:
    contract = _contract()
    incomplete = _candidate(
        contract=contract,
        inventory=contract.expected_fold_seed_inventory[:-1],
    )
    with pytest.raises(M03RSelectionError, match="fold_seed_inventory"):
        select_m03r_checkpoint(
            M03R_CANONICAL_SETTING_ID,
            (incomplete,),
            contract=contract,
        )
    assert "coverage_complete" not in M03RValidationMetrics.__dataclass_fields__
    assert (
        "complete_fold_seed_count" not in M03RCheckpointCandidate.__dataclass_fields__
    )


def test_evidence_receipt_is_deterministic_and_detects_metric_tampering() -> None:
    contract = _contract()
    first = _candidate(contract=contract)
    second = _candidate(contract=contract)
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.receipt_sha256 == first.recompute_receipt_sha256()
    with pytest.raises(M03RSelectionError, match="does not match canonical payload"):
        replace(first, metrics=replace(first.metrics, net_active_return_20bp=0.021))
    with pytest.raises(M03RSelectionError, match="does not match canonical payload"):
        replace(first, receipt_sha256="f" * 64)


@pytest.mark.parametrize(
    ("contract_change", "match"),
    [
        ({"inference_contract_sha256": "3" * 64}, "inference_contract_sha256"),
        ({"source_arrays_sha256": "4" * 64}, "source_arrays_sha256"),
        (
            {"minimum_notional_survival_at_20_sessions": 0.41},
            "selection_contract_sha256",
        ),
    ],
)
def test_evidence_fails_against_changed_contract(
    contract_change: dict[str, object], match: str
) -> None:
    original = _contract()
    evidence = _candidate(contract=original)
    changed = _contract(**contract_change)
    with pytest.raises(M03RSelectionError, match=match):
        evidence.validate_against(changed)


def test_bootstrap_confidence_is_exactly_95_percent() -> None:
    with pytest.raises(M03RSelectionError, match="exactly 95%"):
        _contract(bootstrap_confidence_level=0.90)
    candidate = _candidate()
    with pytest.raises(M03RSelectionError, match="exactly 95%"):
        replace(candidate, bootstrap_confidence_level=0.90)


def test_cross_generation_and_cross_setting_identities_fail_closed() -> None:
    with pytest.raises(M03RSelectionError, match="cannot identify M03R"):
        _contract(protocol_generation=M03R_SUPERSEDED_PROTOCOL_GENERATION)
    contract = _contract()
    evidence = _candidate(contract=contract)
    with pytest.raises(M03RSelectionError, match="requested setting_id"):
        select_m03r_checkpoint(
            "A04-no-uncertainty-scaling",
            (evidence,),
            contract=contract,
        )
    with pytest.raises(M03RSelectionError, match="design_id"):
        replace(
            evidence,
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id="daily_raw_pit300_hold30_v3",
        )
    assert evidence.design_id == M03R_DESIGN_ID


def test_inventory_requires_unique_canonical_cells() -> None:
    inventory = _inventory()
    with pytest.raises(M03RSelectionError, match="canonical fold/seed order"):
        _contract(expected_fold_seed_inventory=tuple(reversed(inventory)))
    with pytest.raises(M03RSelectionError, match="duplicate"):
        _contract(expected_fold_seed_inventory=(*inventory, inventory[-1]))
