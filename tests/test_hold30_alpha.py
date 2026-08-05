from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    HOLD30_BENCHMARK_ID,
    HOLD30_CASH_ASSET_ID,
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaDataError,
    Hold30AlphaEvaluationPanel,
    Hold30AlphaEvaluationProvenance,
    Hold30AlphaLabelDomain,
    bind_hold30_alpha_evaluation_panel,
    build_hold30_residual_alpha_labels,
    verify_hold30_residual_alpha_labels,
)
from rl_quant.protocol.hold30_alpha_v3_freeze import (
    bind_hold30_alpha_v3_data_contract,
)
from rl_quant.training.hold30_alpha import (
    Hold30AlphaObjectiveDomainBinding,
    Hold30AlphaTrainingError,
    bind_hold30_alpha_objective_inputs,
)

DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _provenance() -> Hold30PointInTimeProvenance:
    return Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("snapshot"),
        raw_market_data_sha256=_digest("raw"),
        universe_events_sha256=_digest("universe"),
        tradability_events_sha256=_digest("tradability"),
        corporate_actions_sha256=_digest("corporate-actions"),
        identifier_events_sha256=_digest("identifiers"),
        c1_benchmark_trace_sha256=_digest("c1"),
        risk_limits_sha256=_digest("risk"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="fixture-pit-universe",
        stable_asset_id_namespace="fixture-perm-id",
        benchmark_id=HOLD30_BENCHMARK_ID,
        cash_asset_id=HOLD30_CASH_ASSET_ID,
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )


def _sequence(
    *,
    positions: int = 160,
    fill_tradability: torch.Tensor | None = None,
) -> Hold30DatasetSequence:
    dtype = torch.float64
    batch, assets = 1, 3
    first = 1_735_776_000_000
    decisions = first + torch.arange(positions, dtype=torch.int64) * DAY_MS
    fills = decisions - 6 * HOUR_MS
    fills[0] = decisions[0] - HOUR_MS
    shape = (positions, batch, assets)
    membership = torch.ones(shape, dtype=torch.bool)
    decision_tradability = torch.ones(shape, dtype=torch.bool)
    if fill_tradability is None:
        fill_tradability = torch.ones(shape, dtype=torch.bool)

    decision_state = torch.arange(
        positions * batch * assets * 2, dtype=dtype
    ).reshape(positions, batch, assets, 2)
    returns = torch.empty((positions - 1, batch, assets), dtype=dtype)
    returns[..., 0] = 0.0001
    returns[..., 1] = 0.0010
    returns[..., 2] = 0.0020
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False

    c1_weights = torch.zeros(shape, dtype=dtype)
    c1_weights[..., 0] = 1.0
    c1_returns = returns[..., 0].clone()
    caps = torch.ones(shape, dtype=dtype)
    caps[..., 0] = 1.0
    fill_trade = membership & fill_tradability
    caps[..., 1:] = torch.where(
        fill_trade[..., 1:], caps[..., 1:], torch.zeros_like(caps[..., 1:])
    )
    gross = torch.ones((positions, batch), dtype=dtype)
    cost = torch.full((positions - 1, batch), 0.002, dtype=dtype)

    decision_known = decisions.view(-1, 1, 1).expand(shape).clone()
    fill_known = fills.view(-1, 1, 1).expand(shape).clone()
    versions = torch.zeros(shape, dtype=torch.int64)
    absent = torch.full(shape, -1, dtype=torch.int64)
    evidence = Hold30AsOfEvidence(
        decision_membership_known_at_ms=decision_known.clone(),
        decision_tradability_known_at_ms=decision_known.clone(),
        fill_membership_known_at_ms=fill_known.clone(),
        fill_tradability_known_at_ms=fill_known.clone(),
        corporate_action_factor=torch.ones(shape, dtype=dtype),
        corporate_action_version=versions.clone(),
        corporate_action_known_at_ms=absent.clone(),
        identifier_version=versions.clone(),
        identifier_known_at_ms=absent.clone(),
    )
    return Hold30DatasetSequence(
        decision_timestamps_ms=decisions,
        fill_timestamps_ms=fills,
        asset_ids=("CASH", "PERM-1", "PERM-2"),
        decision_state=decision_state,
        decision_membership=membership,
        decision_tradability=decision_tradability,
        fill_membership=membership.clone(),
        fill_tradability=fill_tradability,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=c1_weights,
        c1_benchmark_net_returns=c1_returns,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=cost,
        asof_evidence=evidence,
        provenance=_provenance(),
    )


def _evaluation_panel(sequence: Hold30DatasetSequence) -> Hold30AlphaEvaluationPanel:
    rows, batch = sequence.n_positions - 1, sequence.batch_size
    provenance = Hold30AlphaEvaluationProvenance(
        risk_free_id="pit-cash-total-return",
        market_benchmark_id="pit-cap-weight-market",
        factor_model_id="declared-market-size-model",
        factor_names=("MKT", "SIZE"),
        factor_return_conventions=("excess-over-risk-free", "zero-investment"),
        risk_free_artifact_sha256=_digest("risk-free-artifact"),
        market_artifact_sha256=_digest("market-artifact"),
        factor_artifact_sha256=_digest("factor-artifact"),
        factor_plan_sha256=_digest("factor-plan"),
    )
    return Hold30AlphaEvaluationPanel(
        source_axis_id=sequence.axis_id,
        risk_free_returns=sequence.asset_returns[..., sequence.cash_index].clone(),
        risk_free_valid=torch.ones((rows, batch), dtype=torch.bool),
        market_total_returns=torch.full((rows, batch), 0.0015, dtype=torch.float64),
        market_valid=torch.ones((rows, batch), dtype=torch.bool),
        factor_returns=torch.zeros((rows, batch, 2), dtype=torch.float64),
        factor_valid=torch.ones((rows, batch, 2), dtype=torch.bool),
        provenance=provenance,
    )


def test_evaluator_data_is_content_bound_and_never_actor_visible() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    receipt = bind_hold30_alpha_evaluation_panel(sequence, panel)

    assert receipt.source_axis_id == sequence.axis_id
    assert receipt.c1_trace_sha256 == sequence.provenance.c1_benchmark_trace_sha256
    assert panel.provenance.factor_usage == ("evaluation-only",)
    assert panel.provenance.policy_feature_access is False
    assert "beta-objective" in panel.provenance.market_usage
    assert "a06-a07-total-sharpe-objective" in panel.provenance.risk_free_usage
    assert "checkpoint-ranking" in panel.provenance.risk_free_usage
    repeated = bind_hold30_alpha_evaluation_panel(sequence, panel)
    assert receipt.receipt_id == repeated.receipt_id

    binding_mutations = (
        {"source_axis_id": _digest("another-axis")},
        {"c1_trace_sha256": _digest("another-c1-trace")},
        {"cash_returns_sha256": _digest("another-cash-series")},
        {"evaluation_panel_id": _digest("another-evaluation-panel")},
        {"evaluation_provenance_id": _digest("another-provenance")},
        {"global_path_ids": (1,)},
    )
    mutated_binding_ids = {
        replace(receipt, **mutation).receipt_id for mutation in binding_mutations
    }
    assert receipt.receipt_id not in mutated_binding_ids
    assert len(mutated_binding_ids) == len(binding_mutations)

    provenance = panel.provenance
    assert provenance.receipt_id == replace(provenance).receipt_id
    provenance_mutations = (
        {"risk_free_id": "pit-cash-total-return-v2"},
        {"market_benchmark_id": "pit-cap-weight-market-v2"},
        {"factor_model_id": "declared-market-size-model-v2"},
        {"factor_names": ("MKT2", "SIZE")},
        {"factor_return_conventions": ("total-return", "zero-investment")},
        {"risk_free_artifact_sha256": _digest("another-risk-free-artifact")},
        {"market_artifact_sha256": _digest("another-market-artifact")},
        {"factor_artifact_sha256": _digest("another-factor-artifact")},
        {"factor_plan_sha256": _digest("another-factor-plan")},
    )
    mutated_provenance_ids = {
        replace(provenance, **mutation).receipt_id
        for mutation in provenance_mutations
    }
    assert provenance.receipt_id not in mutated_provenance_ids
    assert len(mutated_provenance_ids) == len(provenance_mutations)

    changed_cash = panel.risk_free_returns.clone()
    changed_cash[0, 0] += 1e-12
    with pytest.raises(Hold30AlphaDataError, match="bitwise"):
        bind_hold30_alpha_evaluation_panel(
            sequence, replace(panel, risk_free_returns=changed_cash)
        )


def test_evaluator_data_rejects_implicit_missing_values_or_actor_access() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    invalid = panel.factor_valid.clone()
    invalid[0, 0, 0] = False
    with pytest.raises(Hold30AlphaDataError, match="exact zero"):
        replace(panel, factor_valid=invalid, factor_returns=panel.factor_returns + 0.1)

    with pytest.raises(Hold30AlphaDataError, match="policy features"):
        replace(panel.provenance, policy_feature_access=True)

    with pytest.raises(Hold30AlphaDataError, match="checkpoint-ranking"):
        replace(
            panel.provenance,
            risk_free_usage=(
                "portfolio-accounting",
                "a06-a07-total-sharpe-objective",
                "evaluation",
            ),
        )

    incomplete_factors = panel.factor_valid.clone()
    incomplete_factors[0, 0, 0] = False
    incomplete_values = panel.factor_returns.clone()
    incomplete_values[0, 0, 0] = 0.0
    incomplete_panel = replace(
        panel,
        factor_valid=incomplete_factors,
        factor_returns=incomplete_values,
    )
    with pytest.raises(Hold30AlphaDataError, match="factor panel must be complete"):
        bind_hold30_alpha_evaluation_panel(sequence, incomplete_panel)


def test_residual_labels_use_post_fill_returns_and_censor_horizon_63() -> None:
    sequence = _sequence()
    domains = (Hold30AlphaLabelDomain("development", 0, 159),)
    labels = build_hold30_residual_alpha_labels(sequence, domains=domains)

    assert labels.horizons == (5, 21, 30, 63)
    assert torch.equal(labels.origin_rows, sequence.roles.score_indices)
    first_origin = 0
    expected = 5 * (
        torch.log1p(torch.tensor(0.0010, dtype=torch.float64))
        - torch.log1p(torch.tensor(0.0001, dtype=torch.float64))
    )
    assert labels.valid[0, first_origin, 0, 1]
    torch.testing.assert_close(
        labels.values[0, first_origin, 0, 1],
        expected.to(torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert not bool(labels.valid[..., sequence.cash_index].any())
    assert not bool(labels.censored[..., sequence.cash_index].any())

    origin_95 = int((labels.origin_rows == 95).nonzero().item())
    origin_96 = int((labels.origin_rows == 96).nonzero().item())
    assert labels.valid[3, origin_95, 0, 1]
    assert labels.censored[3, origin_96, 0, 1]
    assert labels.values[3, origin_96, 0, 1].item() == 0.0
    verify_hold30_residual_alpha_labels(sequence, labels)


def test_labels_never_cross_a_declared_split() -> None:
    sequence = _sequence()
    labels = build_hold30_residual_alpha_labels(
        sequence,
        domains=(
            Hold30AlphaLabelDomain("train", 0, 100),
            Hold30AlphaLabelDomain("validation", 100, 159),
        ),
    )
    origin_95 = int((labels.origin_rows == 95).nonzero().item())
    origin_100 = int((labels.origin_rows == 100).nonzero().item())
    assert labels.censored[0, origin_95, 0, 1]
    assert not labels.valid[0, origin_95, 0, 1]
    assert labels.valid[0, origin_100, 0, 1]


def test_forced_exit_uses_cash_after_the_inbound_stock_return() -> None:
    tradable = torch.ones((160, 1, 3), dtype=torch.bool)
    tradable[66:, 0, 1] = False
    sequence = _sequence(fill_tradability=tradable)
    labels = build_hold30_residual_alpha_labels(
        sequence, domains=(Hold30AlphaLabelDomain("development", 0, 159),)
    )
    # Origin 63 fills at 64. It earns stock rows 64 and 65; the failed fill at
    # position 66 moves the notional to CASH for rows 66, 67, and 68.
    expected_stock = 2 * torch.log1p(torch.tensor(0.0010, dtype=torch.float64))
    expected_cash = 3 * torch.log1p(torch.tensor(0.0001, dtype=torch.float64))
    expected_c1 = 5 * torch.log1p(torch.tensor(0.0001, dtype=torch.float64))
    torch.testing.assert_close(
        labels.values[0, 0, 0, 1],
        expected_stock + expected_cash - expected_c1,
        atol=1e-12,
        rtol=1e-12,
    )


def test_label_receipt_must_recompute_from_source_tensors() -> None:
    sequence = _sequence()
    labels = build_hold30_residual_alpha_labels(
        sequence, domains=(Hold30AlphaLabelDomain("development", 0, 159),)
    )
    changed = labels.values.clone()
    changed[0, 0, 0, 1] += 1e-9
    tampered = replace(labels, values=changed)
    with pytest.raises(Hold30AlphaDataError, match="does not recompute"):
        verify_hold30_residual_alpha_labels(sequence, tampered)


def test_typed_data_contract_binds_exact_external_return_roles() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    binding = bind_hold30_alpha_evaluation_panel(sequence, panel)
    labels = build_hold30_residual_alpha_labels(
        sequence, domains=(Hold30AlphaLabelDomain("development", 0, 159),)
    )
    contract = bind_hold30_alpha_v3_data_contract(
        panel=panel,
        binding=binding,
        labels=labels,
    )

    assert contract.risk_free_usage == (
        "portfolio-accounting",
        "a06-a07-total-sharpe-objective",
        "checkpoint-ranking",
        "evaluation",
    )
    assert contract.market_usage == (
        "beta-objective",
        "checkpoint-eligibility",
        "evaluation",
    )
    assert contract.factor_usage == ("evaluation-only",)
    assert contract.policy_feature_access is False
    assert contract.manifest_payload()["training_benchmark"]["usage"] == [
        "action-anchor",
        "active-objective-and-label-benchmark",
    ]


def test_objective_adapter_uses_only_receipt_bound_score_rows() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    binding = bind_hold30_alpha_evaluation_panel(sequence, panel)
    train_domain = Hold30AlphaLabelDomain("train", 0, 159)
    labels = build_hold30_residual_alpha_labels(
        sequence,
        domains=(train_domain,),
    )
    bound = bind_hold30_alpha_objective_inputs(
        sequence,
        labels,
        panel,
        binding,
        Hold30AlphaObjectiveDomainBinding(
            role="training",
            domain=train_domain,
        ),
    )
    score_shape = tuple(bound.benchmark_net_return.shape)
    auxiliary_shape = tuple(bound.auxiliary_target.shape)
    policy = torch.full(score_shape, 0.001, dtype=torch.float64, requires_grad=True)
    batch = bound.build_batch(
        "hold30a-m03-alpha-core",
        policy_net_return=policy,
        discretionary_turnover=torch.zeros_like(policy),
        early_exit_mass=torch.zeros_like(policy),
        evaluation_point_id=_digest("evaluation-point"),
        auxiliary_prediction=torch.zeros(auxiliary_shape, dtype=torch.float64),
        downside_30d=torch.ones(auxiliary_shape[:-1], dtype=torch.float64),
    )

    assert batch.binding_kind == "receipt-bound"
    assert batch.source_axis_id == sequence.axis_id
    assert batch.objective_inputs_id == bound.objective_inputs_id
    assert batch.role == "training"
    assert batch.evaluation_point_id == _digest("evaluation-point")
    assert torch.equal(batch.origin_row_ids, bound.score_origin_rows)
    assert torch.equal(batch.global_path_ids, torch.zeros_like(batch.origin_row_ids))
    assert batch.policy_net_return.requires_grad
    assert not batch.benchmark_net_return.requires_grad
    assert not batch.market_return.requires_grad
    assert not batch.risk_free_return.requires_grad
    assert torch.equal(
        batch.benchmark_net_return.reshape(score_shape),
        sequence.c1_benchmark_net_returns.index_select(
            0, bound.score_origin_rows
        ),
    )
    assert torch.equal(
        batch.market_return.reshape(score_shape),
        panel.market_total_returns.index_select(0, bound.score_origin_rows),
    )
    assert torch.equal(
        batch.risk_free_return.reshape(score_shape),
        panel.risk_free_returns.index_select(0, bound.score_origin_rows),
    )
    with pytest.raises(Hold30AlphaTrainingError, match="m02 cannot receive"):
        bound.build_batch(
            "hold30a-m02-active-te",
            policy_net_return=policy,
            discretionary_turnover=torch.zeros_like(policy),
            early_exit_mass=torch.zeros_like(policy),
            evaluation_point_id=_digest("evaluation-point"),
            auxiliary_prediction=torch.zeros(auxiliary_shape, dtype=torch.float64),
        )


def test_objective_adapter_rejects_axis_and_binding_receipt_mismatches() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    binding = bind_hold30_alpha_evaluation_panel(sequence, panel)
    train_domain = Hold30AlphaLabelDomain("train", 0, 159)
    labels = build_hold30_residual_alpha_labels(
        sequence,
        domains=(train_domain,),
    )
    domain_binding = Hold30AlphaObjectiveDomainBinding(
        role="training",
        domain=train_domain,
    )
    with pytest.raises(Hold30AlphaTrainingError, match="do not share one source axis"):
        bind_hold30_alpha_objective_inputs(
            sequence,
            replace(labels, source_axis_id=_digest("other-axis")),
            panel,
            binding,
            domain_binding,
        )
    with pytest.raises(Hold30AlphaTrainingError, match="does not match"):
        bind_hold30_alpha_objective_inputs(
            sequence,
            labels,
            panel,
            replace(binding, evaluation_panel_id=_digest("other-panel")),
            domain_binding,
        )


def test_objective_adapter_cannot_mix_train_validation_or_outer_rows() -> None:
    sequence = _sequence()
    panel = _evaluation_panel(sequence)
    receipt = bind_hold30_alpha_evaluation_panel(sequence, panel)
    train = Hold30AlphaLabelDomain("train", 0, 100)
    validation = Hold30AlphaLabelDomain("validation", 100, 130)
    outer = Hold30AlphaLabelDomain("outer", 130, 159)
    labels = build_hold30_residual_alpha_labels(
        sequence,
        domains=(train, validation, outer),
    )

    train_bound = bind_hold30_alpha_objective_inputs(
        sequence,
        labels,
        panel,
        receipt,
        Hold30AlphaObjectiveDomainBinding(role="training", domain=train),
    )
    validation_bound = bind_hold30_alpha_objective_inputs(
        sequence,
        labels,
        panel,
        receipt,
        Hold30AlphaObjectiveDomainBinding(
            role="inner-validation",
            domain=validation,
        ),
    )
    validation_shape = tuple(validation_bound.benchmark_net_return.shape)
    validation_policy = torch.zeros(
        validation_shape,
        dtype=torch.float64,
        requires_grad=True,
    )
    validation_batch = validation_bound.build_batch(
        "hold30a-m02-active-te",
        policy_net_return=validation_policy,
        discretionary_turnover=torch.zeros_like(validation_policy),
        early_exit_mass=torch.zeros_like(validation_policy),
        evaluation_point_id=_digest("validation-evaluation-point"),
    )

    score_rows = sequence.roles.score_indices
    expected_train = score_rows[(score_rows >= train.start) & (score_rows < train.stop)]
    expected_validation = score_rows[
        (score_rows >= validation.start) & (score_rows < validation.stop)
    ]
    assert torch.equal(train_bound.score_origin_rows, expected_train)
    assert torch.equal(validation_bound.score_origin_rows, expected_validation)
    assert set(train_bound.score_origin_rows.tolist()).isdisjoint(
        validation_bound.score_origin_rows.tolist()
    )
    assert train_bound.objective_inputs_id != validation_bound.objective_inputs_id
    assert validation_batch.role == "inner-validation"
    assert validation_batch.objective_inputs_id == validation_bound.objective_inputs_id
    assert torch.equal(
        validation_batch.origin_row_ids,
        validation_bound.score_origin_rows,
    )
    assert torch.equal(
        validation_batch.global_path_ids,
        torch.zeros_like(validation_batch.origin_row_ids),
    )

    with pytest.raises(Hold30AlphaTrainingError, match="requires label domain"):
        Hold30AlphaObjectiveDomainBinding(role="training", domain=validation)
    with pytest.raises(Hold30AlphaTrainingError, match="requires label domain"):
        Hold30AlphaObjectiveDomainBinding(role="inner-validation", domain=train)
    with pytest.raises(Hold30AlphaTrainingError, match="requires label domain"):
        Hold30AlphaObjectiveDomainBinding(role="training", domain=outer)
