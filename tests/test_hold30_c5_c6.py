from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.evaluation.hold30_c5_c6 import (
    HOLD30_C5_DATES_PER_UPDATE,
    HOLD30_C5_PAIRS_PER_DATE,
    HOLD30_C6_OUTER_SCORE_DOMAIN,
    HOLD30_C6_REPLICATES,
    Hold30C5C6Error,
    Hold30C5CheckpointReference,
    Hold30C5CohortIdentity,
    Hold30C5FitBinding,
    Hold30C5ScheduleKeyBinding,
    Hold30C5ValidationScore,
    Hold30C6PermutationDomain,
    Hold30EmpiricalIntentTrace,
    bind_c6_source_inventory,
    build_c5_labels,
    build_c5_optimizer,
    c5_model_state_sha256,
    capture_empirical_intents,
    construct_c6_controls,
    construct_selected_c5_control,
    coordinate_c5_seed_cohort,
    derive_c5_schedule_key_binding,
    materialize_c5_date_schedule,
    materialize_c5_pair_schedule,
    materialize_c6_permutation_schedule,
    train_c5_update,
    verify_c5_date_schedule,
    verify_c5_pair_schedule,
    verify_c5_selection_receipt,
    verify_c6_permutation_schedule,
)
from rl_quant.evaluation.hold30_controls import price_hold30_cost_ladder
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30 import HOLD30_MECH8_SETTINGS
from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime, Hold30Sequence

POSITIONS = 157
ASSETS = 48
DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _provenance() -> Hold30PointInTimeProvenance:
    return Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("data"),
        raw_market_data_sha256=_digest("raw"),
        universe_events_sha256=_digest("universe"),
        tradability_events_sha256=_digest("tradability"),
        corporate_actions_sha256=_digest("corporate"),
        identifier_events_sha256=_digest("identifier"),
        c1_benchmark_trace_sha256=_digest("c1"),
        risk_limits_sha256=_digest("risk"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="pit-c5-c6-test-v1",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )


def _sequence() -> Hold30DatasetSequence:
    dtype = torch.float64
    batch = 1
    first = 1_704_067_200_000
    decision_ts = first + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (POSITIONS, batch, ASSETS)
    membership = torch.ones(shape, dtype=torch.bool)
    decision_tradability = torch.ones(shape, dtype=torch.bool)
    fill_tradability = torch.ones(shape, dtype=torch.bool)
    # Origin 63 can buy asset one. It earns rows 64..69, is forced out at
    # fill 70, then earns the frozen cash series for the remainder of C5.
    decision_tradability[69:75, 0, 1] = False
    fill_tradability[70:75, 0, 1] = False

    rows = POSITIONS - 1
    returns = torch.zeros((rows, batch, ASSETS), dtype=dtype)
    returns[..., 0] = 0.0001
    time = torch.arange(rows, dtype=dtype).view(-1, 1, 1) * 1e-7
    cross = torch.arange(1, ASSETS, dtype=dtype).view(1, 1, -1) * 1e-5
    returns[..., 1:] = time + cross
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False

    c1_weights = torch.zeros(shape, dtype=dtype)
    c1_weights[..., 0] = 1.0
    c1_net = returns[..., 0].clone()
    caps = torch.full(shape, 0.01, dtype=dtype)
    caps[..., 0] = 1.0
    caps[~fill_tradability] = 0.0
    gross = torch.ones((POSITIONS, batch), dtype=dtype)
    costs = torch.full((rows, batch), 0.002, dtype=dtype)
    decision_state = torch.zeros((*shape, 2), dtype=dtype)
    decision_state[..., 0] = torch.arange(ASSETS, dtype=dtype).view(1, 1, -1)
    decision_state[..., 1] = torch.arange(POSITIONS, dtype=dtype).view(-1, 1, 1)

    decision_known = decision_ts.view(-1, 1, 1).expand(shape).clone()
    fill_known = fill_ts.view(-1, 1, 1).expand(shape).clone()
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
        decision_timestamps_ms=decision_ts,
        fill_timestamps_ms=fill_ts,
        asset_ids=("CASH", *(f"PERM-{index:03d}" for index in range(1, ASSETS))),
        decision_state=decision_state,
        decision_membership=membership.clone(),
        decision_tradability=decision_tradability,
        fill_membership=membership.clone(),
        fill_tradability=fill_tradability,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=c1_weights,
        c1_benchmark_net_returns=c1_net,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=costs,
        asof_evidence=evidence,
        provenance=_provenance(),
    )


def _runtime_sequence(sequence: Hold30DatasetSequence) -> Hold30Sequence:
    return Hold30Sequence(
        decision_state=sequence.decision_state,
        asset_returns=sequence.asset_returns,
        decision_available=sequence.decision_trade,
        fill_membership=sequence.fill_membership,
        fill_availability=sequence.fill_tradability,
        benchmark_weights=sequence.c1_benchmark_weights,
        risk_asset_caps=sequence.risk_asset_caps,
        risk_gross_max=sequence.risk_gross_max,
        benchmark_net_returns=sequence.c1_benchmark_net_returns,
        initial_ledger=CohortLedger.from_staggered_endowment(
            sequence.c1_benchmark_weights[0],
            cash_index=0,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        cost_rate=0.002,
        track_entry_units=sequence.roles.score[:-1],
        axis_id=sequence.axis_id,
    )


def _schedules(sequence: Hold30DatasetSequence):
    rows = tuple(int(value) for value in sequence.roles.score_indices.tolist())
    binding = derive_c5_schedule_key_binding(
        control_root_seed_hex=_digest("control-root"),
        executable_manifest_sha256=_digest("manifest"),
        fold_index=0,
    )
    pair = materialize_c5_pair_schedule(
        sequence,
        date_rows=rows,
        key_binding=binding,
    )
    dates = materialize_c5_date_schedule(
        sequence,
        permitted_rows=rows,
        key_binding=binding,
    )
    return pair, dates


class _ToyPolicy(torch.nn.Module):
    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01 + bias, dtype=torch.float64))

    def c5_entry_scores(
        self,
        state: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        value = state[..., 0] * self.scale
        return torch.where(available, value, torch.zeros_like(value))


class _RuntimeH2Policy(torch.nn.Module):
    def hold30_intent(
        self,
        state: torch.Tensor,
        _weights: torch.Tensor,
        _available: torch.Tensor,
        _age: torch.Tensor,
    ) -> Hold30Intent:
        entry = state[..., 0] * 0.01
        return Hold30Intent(
            entry_scores=entry,
            hazard_residual=torch.zeros_like(entry),
            exposure_residual=entry.new_zeros((entry.shape[0],)),
        )


class _StateProvider:
    trains_upstream_encoder = True

    def __init__(self, sequence: Hold30DatasetSequence) -> None:
        self.states = sequence.decision_state[:-1]
        self.decision_available = sequence.decision_trade[:-1].permute(1, 0, 2).clone()
        self.replay_calls = 0
        payload = {
            "schema_version": 1,
            "provider": "tests._StateProvider",
            "source_axis_id": sequence.axis_id,
            "raw_bars_sha256": _digest("bars"),
            "frozen_context_sha256": _digest("context"),
            "batch_size": sequence.batch_size,
            "decision_count": sequence.n_positions - 1,
            "asset_count": sequence.num_assets,
        }
        payload["binding_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.binding_config = payload

    def replay_origin_states(
        self,
        _policy: torch.nn.Module,
        _sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        self.replay_calls += 1
        return self.states.index_select(0, origins.to(dtype=torch.int64))

    def canonical_states(
        self,
        _policy: torch.nn.Module,
        _sequence: Hold30Sequence,
    ) -> torch.Tensor:
        return self.states


def _identity(sequence: Hold30DatasetSequence, pair_receipt: str, date_receipt: str):
    binding = derive_c5_schedule_key_binding(
        control_root_seed_hex=_digest("control-root"),
        executable_manifest_sha256=_digest("manifest"),
        fold_index=0,
    )
    return Hold30C5CohortIdentity(
        fold_index=0,
        executable_manifest_sha256=_digest("manifest"),
        development_receipt_sha256=_digest("development"),
        fold_sha256=_digest("fold"),
        training_axis_id=sequence.axis_id,
        inner_validation_axis_id=sequence.axis_id,
        training_labels_sha256=_digest("training-labels"),
        validation_labels_sha256=_digest("validation-labels"),
        control_schedule_binding_sha256=binding.receipt_sha256,
        stage1_normalization_receipt_sha256=_digest("stage1-normalization-receipt"),
        pair_schedule_sha256=pair_receipt,
        date_schedule_sha256=date_receipt,
    )


def _fit_binding(sequence: Hold30DatasetSequence) -> Hold30C5FitBinding:
    payload = {
        "schema": "rl-quant.hold30.c5-fit-binding",
        "schema_version": 1,
        "fold_index": 0,
        "development_receipt_sha256": _digest("development"),
        "fold_sha256": _digest("fold"),
        "training_axis_id": sequence.axis_id,
        "inner_validation_axis_id": _digest("inner-validation-axis"),
        "training_absolute_range": [0, POSITIONS],
        "validation_absolute_range": [POSITIONS, POSITIONS + 95],
        "outer_absolute_range": [POSITIONS + 95, POSITIONS + 190],
        "role": "expanding_training",
        "outer_access": False,
    }
    return Hold30C5FitBinding(
        fold_index=0,
        development_receipt_sha256=payload["development_receipt_sha256"],
        fold_sha256=payload["fold_sha256"],
        training_axis_id=payload["training_axis_id"],
        inner_validation_axis_id=payload["inner_validation_axis_id"],
        training_absolute_range=(0, POSITIONS),
        validation_absolute_range=(POSITIONS, POSITIONS + 95),
        outer_absolute_range=(POSITIONS + 95, POSITIONS + 190),
        receipt_sha256=_digest(payload),
    )


def _refs(
    identity: Hold30C5CohortIdentity,
    update: int,
    *,
    models: tuple[_ToyPolicy, ...] | None = None,
) -> tuple[Hold30C5CheckpointReference, ...]:
    return tuple(
        Hold30C5CheckpointReference(
            seed=seed,
            update=update,
            checkpoint_id=f"c5-seed-{seed}-update-{update:03d}",
            model_state_sha256=(
                c5_model_state_sha256(models[index])
                if models is not None
                else _digest(("model", seed, update))
            ),
            checkpoint_receipt_sha256=_digest(("checkpoint", seed, update)),
            stage1_normalization_receipt_sha256=(
                identity.stage1_normalization_receipt_sha256
            ),
            training_labels_sha256=identity.training_labels_sha256,
            control_schedule_binding_sha256=(identity.control_schedule_binding_sha256),
            pair_schedule_sha256=identity.pair_schedule_sha256,
            date_schedule_sha256=identity.date_schedule_sha256,
        )
        for index, seed in enumerate(HOLD30_SEEDS)
    )


def test_c5_labels_use_exact_post_fill_path_and_never_cross_roles() -> None:
    sequence = _sequence()
    labels = build_c5_labels(sequence)
    origin = 63
    stock_log = torch.log1p(sequence.asset_returns[64:70, 0, 1]).sum()
    cash_log = torch.log1p(sequence.asset_returns[70:94, 0, 0]).sum()
    benchmark_log = torch.log1p(sequence.c1_benchmark_net_returns[64:94, 0]).sum()

    assert labels.valid[origin, 0, 1]
    torch.testing.assert_close(
        labels.values[origin, 0, 1],
        stock_log + cash_log - benchmark_log,
        rtol=0.0,
        atol=1e-15,
    )
    assert not bool(labels.valid[~labels.score_rows].any())
    assert not bool(labels.censored.any())
    assert labels.receipt_sha256 == build_c5_labels(sequence).receipt_sha256


def test_c5_hash_schedules_are_deterministic_outcome_blind_and_tamper_detected() -> (
    None
):
    sequence = _sequence()
    pair, dates = _schedules(sequence)
    second_pair, second_dates = _schedules(sequence)

    assert pair.receipt_sha256 == second_pair.receipt_sha256
    assert dates.receipt_sha256 == second_dates.receipt_sha256
    assert pair.key_binding.receipt_sha256 == dates.key_binding.receipt_sha256
    assert pair.key_binding.executable_manifest_sha256 == _digest("manifest")
    assert pair.hash_domain.endswith("/c5-pair-schedule")
    assert dates.hash_domain.endswith("/c5-date-schedule")
    assert pair.hash_key_sha256 != dates.hash_key_sha256
    assert torch.equal(pair.pairs, second_pair.pairs)
    assert torch.equal(dates.update_rows, second_dates.update_rows)
    assert pair.pairs.shape[2] == HOLD30_C5_PAIRS_PER_DATE
    assert all(
        len(set(row.tolist())) == HOLD30_C5_DATES_PER_UPDATE
        for row in dates.update_rows
    )
    verify_c5_pair_schedule(sequence, pair)
    verify_c5_date_schedule(sequence, dates)

    other_pair = materialize_c5_pair_schedule(
        sequence,
        date_rows=pair.date_rows,
        key_binding=derive_c5_schedule_key_binding(
            control_root_seed_hex=_digest("other-control-root"),
            executable_manifest_sha256=_digest("manifest"),
            fold_index=0,
        ),
    )
    with pytest.raises(Hold30C5C6Error, match="self-hash"):
        verify_c5_pair_schedule(sequence, replace(pair, pairs=other_pair.pairs))
    other_dates = materialize_c5_date_schedule(
        sequence,
        permitted_rows=dates.permitted_rows,
        key_binding=derive_c5_schedule_key_binding(
            control_root_seed_hex=_digest("other-control-root"),
            executable_manifest_sha256=_digest("manifest"),
            fold_index=0,
        ),
    )
    with pytest.raises(Hold30C5C6Error, match="self-hash"):
        verify_c5_date_schedule(
            sequence, replace(dates, update_rows=other_dates.update_rows)
        )

    with pytest.raises(Hold30C5C6Error, match="do not derive"):
        Hold30C5ScheduleKeyBinding(
            control_root_seed_hex=_digest("control-root"),
            executable_manifest_sha256=_digest("manifest"),
            fold_index=0,
            pair_key_sha256=_digest("invented-pair-key"),
            date_key_sha256=dates.hash_key_sha256,
        )


def test_c5_update_uses_four_microbatches_one_step_and_exact_global_denominator() -> (
    None
):
    sequence = _sequence()
    labels = build_c5_labels(sequence)
    pair, dates = _schedules(sequence)
    policy = _ToyPolicy()
    provider = _StateProvider(sequence)
    optimizer = build_c5_optimizer(policy)
    before = policy.scale.detach().clone()

    result = train_c5_update(
        policy,
        provider,
        _runtime_sequence(sequence),
        labels,
        pair,
        dates,
        optimizer,
        update=1,
        fit_binding=_fit_binding(sequence),
    )

    assert provider.replay_calls == 4
    assert result.valid_pair_count > 0
    assert len(set(result.date_rows)) == 16
    assert result.model_state_sha256 == c5_model_state_sha256(policy)
    assert policy.scale.detach() != before
    assert optimizer.state[policy.scale]["step"] == 1

    other_sequence = replace(
        sequence,
        provenance=replace(
            sequence.provenance,
            data_snapshot_sha256=_digest("outer-or-other-data"),
        ),
    )
    with pytest.raises(Hold30C5C6Error, match="validation or outer data"):
        train_c5_update(
            policy,
            provider,
            _runtime_sequence(sequence),
            labels,
            pair,
            dates,
            optimizer,
            update=2,
            fit_binding=_fit_binding(other_sequence),
        )


def test_c5_selection_is_synchronous_shared_and_receipt_tamper_evident() -> None:
    sequence = _sequence()
    pair, dates = _schedules(sequence)
    identity = _identity(sequence, pair.receipt_sha256, dates.receipt_sha256)
    wealth = {8: 0.0012, 16: 0.0011, 24: 0.0011, 32: 0.0010, 40: 0.0010}

    outcome = coordinate_c5_seed_cohort(
        identity,
        _refs(identity, 0),
        advance_cohort=lambda update: _refs(identity, update),
        validate_ensemble=lambda update, _checkpoints: Hold30C5ValidationScore(
            update=update,
            active_log_wealth=wealth[update],
            discretionary_turnover=0.02,
            trace_sha256=_digest(("validation-trace", update)),
            inner_validation_axis_id=identity.inner_validation_axis_id,
            validation_labels_sha256=identity.validation_labels_sha256,
        ),
    )

    assert outcome.stopped_update == 40
    assert outcome.selected.update == 32
    receipt = outcome.receipt()
    assert verify_c5_selection_receipt(receipt) == outcome
    tampered = dict(receipt)
    tampered["selected_update"] = 40
    with pytest.raises(Hold30C5C6Error, match="self-hash"):
        verify_c5_selection_receipt(tampered)
    with pytest.raises(Hold30C5C6Error, match="five references"):
        coordinate_c5_seed_cohort(
            identity,
            _refs(identity, 0)[:-1],
            advance_cohort=lambda update: _refs(identity, update),
            validate_ensemble=lambda *_args: pytest.fail("must not validate"),
        )
    bad_refs = list(_refs(identity, 0))
    bad_refs[0] = replace(
        bad_refs[0],
        stage1_normalization_receipt_sha256=_digest("c5-outcome-normalization"),
    )
    with pytest.raises(Hold30C5C6Error, match="shared training evidence"):
        coordinate_c5_seed_cohort(
            identity,
            bad_refs,
            advance_cohort=lambda update: _refs(identity, update),
            validate_ensemble=lambda *_args: pytest.fail("must not validate"),
        )


def test_c5_selected_models_execute_once_through_canonical_h2() -> None:
    sequence = _sequence()
    pair, dates = _schedules(sequence)
    identity = _identity(sequence, pair.receipt_sha256, dates.receipt_sha256)
    models = tuple(_ToyPolicy(index * 0.001) for index in range(5))
    providers = tuple(_StateProvider(sequence) for _ in range(5))
    wealth = {8: 0.0012, 16: 0.0011, 24: 0.0011, 32: 0.0010, 40: 0.0010}
    outcome = coordinate_c5_seed_cohort(
        identity,
        _refs(identity, 0, models=models),
        advance_cohort=lambda update: _refs(identity, update, models=models),
        validate_ensemble=lambda update, _checkpoints: Hold30C5ValidationScore(
            update=update,
            active_log_wealth=wealth[update],
            discretionary_turnover=0.02,
            trace_sha256=_digest(("model-validation-trace", update)),
            inner_validation_axis_id=identity.inner_validation_axis_id,
            validation_labels_sha256=identity.validation_labels_sha256,
        ),
    )
    trace = construct_selected_c5_control(
        sequence,
        models,
        providers,
        outcome.receipt(),
    )

    assert trace.control_id == "C5"
    assert trace.outer_start == 63
    assert not bool((trace.turnover_by_cause[TurnoverCause.STARTUP] != 0).any())
    assert not bool((trace.turnover_by_cause[TurnoverCause.TERMINAL] != 0).any())
    assert bool(
        (trace.turnover_by_cause[TurnoverCause.DISCRETIONARY] <= 0.10 + 1e-12).all()
    )
    ladder = price_hold30_cost_ladder(trace)
    assert [rung.cost_bps for rung in ladder.rungs] == [10, 20, 40]


def _empirical_source(
    sequence: Hold30DatasetSequence,
    *,
    setting_index: int,
    fold_index: int,
) -> Hold30EmpiricalIntentTrace:
    setting = HOLD30_MECH8_SETTINGS[setting_index]
    decisions = sequence.n_positions - 1
    entry = sequence.asset_returns.new_zeros(
        (decisions, sequence.batch_size, sequence.num_assets)
    )
    dates = torch.arange(decisions, dtype=entry.dtype).view(-1, 1, 1)
    assets = torch.arange(sequence.num_assets, dtype=entry.dtype).view(1, 1, -1)
    entry.copy_(dates * 0.001 + assets * 0.01)
    entry[..., 0] = 0.0
    common = {
        "mechanism": setting.mechanism,
        "setting_id": setting.setting_id,
        "fold_index": fold_index,
        "ensemble_member_count": 5,
        "source_id": f"{setting.setting_id}/fold-{fold_index}",
        "source_axis_id": sequence.axis_id,
        "source_trace_sha256": _digest(
            ("canonical-source-trace", setting.setting_id, fold_index)
        ),
        "outer_score_rows": tuple(
            int(value) for value in sequence.roles.score_indices.tolist()
        ),
        "decision_available": sequence.decision_trade[:-1].clone(),
    }
    if setting.mechanism in {"H0", "H1"}:
        return Hold30EmpiricalIntentTrace(
            **common,
            target_logits=entry,
            gate=entry.new_full((decisions, sequence.batch_size), 0.03),
        )
    if setting.mechanism == "H2":
        return Hold30EmpiricalIntentTrace(
            **common,
            entry_scores=entry,
            hazard_residual=torch.zeros_like(entry),
            exposure_residual=entry.new_zeros((decisions, sequence.batch_size)),
        )
    return Hold30EmpiricalIntentTrace(**common, entry_scores=entry)


def _empirical_h2(sequence: Hold30DatasetSequence) -> Hold30EmpiricalIntentTrace:
    return _empirical_source(sequence, setting_index=2, fold_index=0)


def _c6_inventory(sequence: Hold30DatasetSequence):
    return bind_c6_source_inventory(
        tuple(
            _empirical_source(
                sequence,
                setting_index=setting.setting_index,
                fold_index=fold_index,
            )
            for setting in HOLD30_MECH8_SETTINGS
            for fold_index in range(6)
        )
    )


def test_c6_capture_uses_raw_pending_intents_from_the_canonical_trace() -> None:
    sequence = _sequence()
    runtime_sequence = _runtime_sequence(sequence)
    runtime = Hold30ChronologicalRuntime("H2")
    with torch.no_grad():
        canonical, _rows = runtime.canonical_pass(
            _RuntimeH2Policy(),
            runtime_sequence,
            Hold30ReplayGeometry().roles(POSITIONS),
        )
    source = capture_empirical_intents(
        canonical,
        mechanism="H2",
        setting_id="hold30-m02-age-hazard",
        fold_index=0,
        source_id="hold30-m02-age-hazard/fold-0",
        source_axis_id=sequence.axis_id,
        source_trace_sha256=_digest("source-canonical-trace"),
        outer_score_rows=sequence.roles.score_indices.tolist(),
    )

    assert source.n_decisions == POSITIONS - 1
    assert source.entry_scores is not None
    torch.testing.assert_close(
        source.entry_scores[63],
        canonical.transitions[63].raw_intent.entry_scores,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        source.receipt_sha256
        == capture_empirical_intents(
            canonical,
            mechanism="H2",
            setting_id="hold30-m02-age-hazard",
            fold_index=0,
            source_id="hold30-m02-age-hazard/fold-0",
            source_axis_id=sequence.axis_id,
            source_trace_sha256=_digest("source-canonical-trace"),
            outer_score_rows=sequence.roles.score_indices.tolist(),
        ).receipt_sha256
    )


def test_c6_requires_explicit_domains_replays_64_raw_intent_permutations_and_detects_tamper() -> (
    None
):
    sequence = _sequence()
    source = _empirical_h2(sequence)
    with pytest.raises(Hold30C5C6Error, match="every stable setting/fold"):
        bind_c6_source_inventory((source,))
    with pytest.raises(Hold30C5C6Error, match="five-seed ensemble"):
        replace(source, ensemble_member_count=4)
    with pytest.raises(Hold30C5C6Error, match="exact 63 outer-score"):
        materialize_c6_permutation_schedule(
            source,
            domains=(),
            hash_key_sha256=_digest("c6-key"),
            hash_domain="prelockbox-hold30-c6-test",
        )
    with pytest.raises(Hold30C5C6Error, match="exact 63 outer-score"):
        materialize_c6_permutation_schedule(
            source,
            domains=(Hold30C6PermutationDomain("too-small", (63, 64)),),
            hash_key_sha256=_digest("c6-key"),
            hash_domain="prelockbox-hold30-c6-test",
        )
    domain = Hold30C6PermutationDomain(
        HOLD30_C6_OUTER_SCORE_DOMAIN,
        tuple(int(value) for value in sequence.roles.score_indices.tolist()),
    )
    schedule = materialize_c6_permutation_schedule(
        source,
        domains=(domain,),
        hash_key_sha256=_digest("c6-key"),
        hash_domain="prelockbox-hold30-c6-test",
    )
    verify_c6_permutation_schedule(source, schedule)
    assert schedule.mappings.shape == (HOLD30_C6_REPLICATES, POSITIONS - 1)
    assert all(
        not torch.equal(mapping, torch.arange(POSITIONS - 1))
        for mapping in schedule.mappings
    )
    fixed_rows = sorted(set(range(POSITIONS - 1)) - set(domain.rows))
    assert torch.equal(
        schedule.mappings[:, fixed_rows],
        torch.tensor(fixed_rows, dtype=torch.int64).expand(HOLD30_C6_REPLICATES, -1),
    )

    inventory = _c6_inventory(sequence)
    traces = construct_c6_controls(sequence, source, schedule, inventory)
    assert len(traces) == HOLD30_C6_REPLICATES
    assert all(trace.control_id == "C6" for trace in traces)
    assert len({trace.trace_sha256 for trace in traces}) == HOLD30_C6_REPLICATES
    assert all(
        not bool((trace.turnover_by_cause[TurnoverCause.TERMINAL] != 0).any())
        for trace in traces
    )

    tampered_mapping = schedule.mappings.clone()
    rows = torch.tensor(domain.rows[:2])
    tampered_mapping[0, rows] = tampered_mapping[0, rows.flip(0)]
    with pytest.raises(Hold30C5C6Error, match="self-hash"):
        tampered = replace(schedule, mappings=tampered_mapping)
        verify_c6_permutation_schedule(source, tampered)
