from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_PRELOCKBOX_CUTOFF_MS,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetError,
    Hold30DatasetSequence,
    Hold30NullDomain,
    Hold30PointInTimeProvenance,
)
from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence


DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _digest(character: str) -> str:
    return character * 64


def _provenance(**updates) -> Hold30PointInTimeProvenance:
    values = {
        "data_snapshot_sha256": _digest("a"),
        "raw_market_data_sha256": _digest("b"),
        "universe_events_sha256": _digest("c"),
        "tradability_events_sha256": _digest("d"),
        "corporate_actions_sha256": _digest("e"),
        "identifier_events_sha256": _digest("f"),
        "c1_benchmark_trace_sha256": _digest("1"),
        "risk_limits_sha256": _digest("2"),
        "universe_mode": HOLD30_UNIVERSE_MODE,
        "universe_rule_id": "pit-dollar-volume-v1",
        "stable_asset_id_namespace": "perm-id-v1",
        "benchmark_id": "C1",
        "cash_asset_id": "CASH",
        "cash_return_rule": HOLD30_CASH_RETURN_RULE,
    }
    values.update(updates)
    return Hold30PointInTimeProvenance(**values)


def _sequence(
    *,
    positions: int = 98,
    decision_membership: torch.Tensor | None = None,
    decision_tradability: torch.Tensor | None = None,
    fill_membership: torch.Tensor | None = None,
    fill_tradability: torch.Tensor | None = None,
    provenance: Hold30PointInTimeProvenance | None = None,
) -> Hold30DatasetSequence:
    dtype = torch.float64
    batch, assets = 1, 4
    first_decision = 1_735_776_000_000  # 2025-01-02T00:00:00Z
    decision_ts = first_decision + torch.arange(positions, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (positions, batch, assets)
    masks = torch.ones(shape, dtype=torch.bool)
    decision_membership = masks.clone() if decision_membership is None else decision_membership
    decision_tradability = masks.clone() if decision_tradability is None else decision_tradability
    fill_membership = masks.clone() if fill_membership is None else fill_membership
    fill_tradability = masks.clone() if fill_tradability is None else fill_tradability

    decision_state = torch.empty((*shape, 2), dtype=dtype)
    for position in range(positions):
        for asset in range(assets):
            decision_state[position, 0, asset] = torch.tensor(
                [1000.0 * position + asset, -1000.0 * position - asset], dtype=dtype
            )

    returns = torch.zeros((positions - 1, batch, assets), dtype=dtype)
    returns[..., 0] = 0.0001  # CASH is an explicit return series, not an implicit zero.
    c1 = torch.zeros(shape, dtype=dtype)
    c1[..., 0] = 1.0
    c1_return = returns[..., 0].clone()
    fill_trade = fill_membership & fill_tradability
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    ordinary = fill_membership[:-1].clone()
    ordinary[..., 0] = False
    caps = torch.zeros(shape, dtype=dtype)
    caps[..., 0] = 1.0
    caps[..., 1:] = torch.where(
        fill_trade[..., 1:], torch.full_like(caps[..., 1:], 0.01), 0.0
    )
    gross = torch.ones((positions, batch), dtype=dtype)
    costs = torch.full((positions - 1, batch), 0.002, dtype=dtype)

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
        asset_ids=("CASH", "PERM-1", "PERM-2", "PERM-3"),
        decision_state=decision_state,
        decision_membership=decision_membership,
        decision_tradability=decision_tradability,
        fill_membership=fill_membership,
        fill_tradability=fill_tradability,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=c1,
        c1_benchmark_net_returns=c1_return,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=costs,
        asof_evidence=evidence,
        provenance=_provenance() if provenance is None else provenance,
    )


def test_materializes_terminal_score_support_and_runtime_shape_contract() -> None:
    sequence = _sequence(positions=98)

    assert sequence.n_positions == 98
    assert sequence.roles.warmup.sum().item() == 63
    assert sequence.roles.score_indices.tolist() == [63, 64, 65, 66]
    assert sequence.roles.support.sum().item() == 30
    assert sequence.roles.terminal.nonzero().flatten().tolist() == [97]
    assert sequence.roles.utility_rows.shape == (4, 31)
    assert sequence.roles.utility_rows[0].tolist() == list(range(63, 94))
    assert sequence.roles.replay_terminal_rows.tolist() == [94, 95, 96, 97]

    marker = object()
    runtime = sequence.runtime_kwargs(initial_ledger=marker)
    assert runtime["initial_ledger"] is marker
    assert runtime["axis_id"] == sequence.axis_id
    assert set(runtime) == {
        "decision_state",
        "asset_returns",
        "decision_available",
        "fill_membership",
        "fill_availability",
        "benchmark_weights",
        "risk_asset_caps",
        "risk_gross_max",
        "benchmark_net_returns",
        "initial_ledger",
        "cost_rate",
        "initial_equity",
        "track_entry_units",
        "axis_id",
    }
    assert torch.equal(runtime["track_entry_units"], sequence.roles.score[:-1])
    assert runtime["track_entry_units"].sum().item() == 4
    assert not bool(runtime["track_entry_units"][:63].any())
    assert not bool(runtime["track_entry_units"][-30:].any())

    ledger = CohortLedger.from_weights(
        sequence.c1_benchmark_weights[0],
        cash_index=sequence.cash_index,
        track_initial_units=True,
    )
    runtime_sequence = Hold30Sequence(
        **sequence.runtime_kwargs(initial_ledger=ledger)
    )
    assert runtime_sequence.n_positions == sequence.n_positions
    assert runtime_sequence.axis_id == sequence.axis_id
    torch.testing.assert_close(
        runtime_sequence.initial_ledger.weights,
        sequence.c1_benchmark_weights[0],
    )


def test_no_lookahead_a_trade_handles_deletion_and_unknown_addition() -> None:
    positions = 98
    shape = (positions, 1, 4)
    decision_member = torch.ones(shape, dtype=torch.bool)
    fill_member = torch.ones(shape, dtype=torch.bool)

    # At t=10 PERM-1 is decision-visible, but it is deleted at the t+1 fill.
    fill_member[11:, 0, 1] = False
    # PERM-2 is first added at the t+1 fill; the t=10 decision cannot buy it.
    decision_member[:11, 0, 2] = False
    fill_member[:11, 0, 2] = False
    sequence = _sequence(
        positions=positions,
        decision_membership=decision_member,
        fill_membership=fill_member,
    )

    assert bool(sequence.decision_trade[10, 0, 1])
    assert not bool(sequence.fill_trade[11, 0, 1])
    assert not bool(sequence.a_trade[10, 0, 1])
    assert not bool(sequence.decision_trade[10, 0, 2])
    assert bool(sequence.fill_trade[11, 0, 2])
    assert not bool(sequence.a_trade[10, 0, 2])
    assert bool(sequence.a_trade[11, 0, 2])

    # The runtime sees only the decision intersection at t; it receives the
    # fill facts separately and cannot leak them into the actor observation.
    runtime = sequence.runtime_kwargs(initial_ledger=object())
    assert bool(runtime["decision_available"][10, 0, 1])
    assert not bool(runtime["fill_membership"][11, 0, 1])


def test_decision_axis_fails_closed_on_duplicate_or_2026_position() -> None:
    sequence = _sequence()
    duplicate = sequence.decision_timestamps_ms.clone()
    duplicate[2] = duplicate[1]
    with pytest.raises(Hold30DatasetError, match="strictly increasing"):
        replace(sequence, decision_timestamps_ms=duplicate)

    after_lockbox = sequence.decision_timestamps_ms.clone()
    after_lockbox[-1] = HOLD30_PRELOCKBOX_CUTOFF_MS
    fill = sequence.fill_timestamps_ms.clone()
    fill[-1] = after_lockbox[-1] - 6 * HOUR_MS
    with pytest.raises(Hold30DatasetError, match="precede"):
        replace(sequence, decision_timestamps_ms=after_lockbox, fill_timestamps_ms=fill)


def _null_domains() -> tuple[Hold30NullDomain, ...]:
    return (
        Hold30NullDomain("train", 0, 62),
        Hold30NullDomain("validation", 62, 124),
        Hold30NullDomain("outer", 124, 186),
    )


def _outcome_sequence() -> Hold30DatasetSequence:
    sequence = _sequence(positions=187)
    returns = sequence.asset_returns.clone()
    rows = torch.arange(186, dtype=returns.dtype).view(-1, 1)
    returns[:, 0, 1:] = rows / 10_000.0 + torch.tensor(
        [0.00001, 0.00002, 0.00003], dtype=returns.dtype
    )
    return replace(sequence, asset_returns=returns)


def test_outcome_nulls_are_deterministic_domain_bound_and_receipt_complete() -> None:
    sequence = _outcome_sequence()
    domains = _null_domains()
    n_time_1 = sequence.n_time(731, domains=domains)
    n_time_2 = sequence.n_time(731, domains=domains)
    n_xs_1 = sequence.n_xs(991, domains=domains)
    n_xs_2 = sequence.n_xs(991, domains=domains)

    assert torch.equal(n_time_1.asset_returns, n_time_2.asset_returns)
    assert torch.equal(n_time_1.source_index, n_time_2.source_index)
    assert n_time_1.receipt == n_time_2.receipt
    assert torch.equal(n_xs_1.asset_returns, n_xs_2.asset_returns)
    assert torch.equal(n_xs_1.source_index, n_xs_2.source_index)
    assert n_xs_1.receipt == n_xs_2.receipt
    assert n_time_1.receipt.kind == "N_time"
    assert n_xs_1.receipt.kind == "N_xs"
    assert n_time_1.receipt.transform_id != n_xs_1.receipt.transform_id
    assert n_time_1.receipt.domains == tuple(
        (domain.name, domain.start, domain.stop) for domain in domains
    )
    assert n_time_1.receipt.output_outcomes_sha256 != n_time_1.receipt.input_outcomes_sha256
    assert n_xs_1.receipt.output_outcomes_sha256 != n_xs_1.receipt.input_outcomes_sha256

    for domain in domains:
        sources = n_time_1.source_index[domain.start : domain.stop]
        assert sorted(sources.tolist()) == list(range(domain.start, domain.stop))
        destinations = torch.arange(domain.start, domain.stop)
        assert bool(((sources - destinations).abs() >= 31).all())
        for destination, source in zip(destinations.tolist(), sources.tolist(), strict=True):
            required = sequence.ordinary_return_valid[destination]
            assert bool(sequence.ordinary_return_valid[source][required].all())

    # Cross-sectional maps are nonidentity cycles over all three risky assets.
    risky_map = n_xs_1.source_index[..., 1:]
    identity = torch.arange(1, 4).view(1, 1, 3).expand_as(risky_map)
    assert not bool((risky_map == identity).any())

    # Actor state is not an input or output of either outcome transform.
    assert torch.equal(sequence.decision_state, _outcome_sequence().decision_state)
    with pytest.raises(TypeError, match="null_view"):
        sequence.runtime_kwargs(initial_ledger=object(), null_view=n_time_1)  # type: ignore[call-arg]


def test_outcome_nulls_fix_cash_and_mandatory_outcomes_and_hash_outputs() -> None:
    sequence = _outcome_sequence()
    mandatory = sequence.mandatory_return_mask.clone()
    ordinary = sequence.ordinary_return_valid.clone()
    # With a 62-row domain and 31-row minimum distance, rows 70 and 101 are
    # the only legal partners for each other. Marking both keeps N_time
    # feasible while proving that mandatory coordinates never move.
    mandatory[[70, 101], 0, 1] = True
    ordinary[[70, 101], 0, 1] = False
    returns = sequence.asset_returns.clone()
    returns[70, 0, 1] = -0.75
    returns[101, 0, 1] = -0.50
    sequence = replace(
        sequence,
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
    )
    domains = _null_domains()
    time_view = sequence.n_time(731, domains=domains)
    xs_view = sequence.n_xs(991, domains=domains)

    torch.testing.assert_close(time_view.asset_returns[..., 0], sequence.asset_returns[..., 0])
    torch.testing.assert_close(xs_view.asset_returns[..., 0], sequence.asset_returns[..., 0])
    assert time_view.asset_returns[70, 0, 1].item() == -0.75
    assert xs_view.asset_returns[70, 0, 1].item() == -0.75

    changed = replace(sequence, asset_returns=sequence.asset_returns + ordinary * 0.0005)
    changed_view = changed.n_time(731, domains=domains)
    assert torch.equal(changed_view.source_index, time_view.source_index)
    assert changed_view.receipt.input_outcomes_sha256 != time_view.receipt.input_outcomes_sha256
    assert changed_view.receipt.output_outcomes_sha256 != time_view.receipt.output_outcomes_sha256
    assert changed_view.receipt.transform_id != time_view.receipt.transform_id


def test_outcome_nulls_fail_without_perfect_match_or_two_xs_coordinates() -> None:
    too_short = _sequence(positions=184)
    impossible_domains = (
        Hold30NullDomain("train", 0, 61),
        Hold30NullDomain("validation", 61, 122),
        Hold30NullDomain("outer", 122, 183),
    )
    with pytest.raises(Hold30DatasetError, match="no perfect"):
        too_short.n_time(1, domains=impossible_domains)

    membership = torch.zeros((187, 1, 4), dtype=torch.bool)
    membership[..., :2] = True
    one_risky = _sequence(
        positions=187,
        decision_membership=membership,
        fill_membership=membership,
    )
    with pytest.raises(Hold30DatasetError, match="at least two"):
        one_risky.n_xs(1, domains=_null_domains())


def test_invalid_or_missing_point_in_time_provenance_fails_closed() -> None:
    with pytest.raises(Hold30DatasetError, match="future-selected TOP2000"):
        _provenance(universe_mode="future_selected_top2000")
    with pytest.raises(Hold30DatasetError, match="lowercase SHA-256"):
        _provenance(corporate_actions_sha256="E" * 64)
    with pytest.raises(Hold30DatasetError, match="explicit one-step"):
        _provenance(cash_return_rule="implicit_zero")
    with pytest.raises(Hold30DatasetError, match="identify C1"):
        _provenance(benchmark_id="equal_weight_inferred")

    sequence = _sequence()
    with pytest.raises(Hold30DatasetError, match="complete Hold30PointInTimeProvenance"):
        replace(sequence, provenance=None)  # type: ignore[arg-type]
    with pytest.raises(Hold30DatasetError, match="ordered tuple of unique"):
        replace(sequence, asset_ids=("CASH", "PERM-1", "PERM-1", "PERM-3"))


def test_future_event_receipt_and_unversioned_corporate_change_fail_closed() -> None:
    sequence = _sequence()
    future_known = sequence.asof_evidence.decision_membership_known_at_ms.clone()
    future_known[12, 0, 2] = sequence.decision_timestamps_ms[12] + 1
    with pytest.raises(Hold30DatasetError, match="no later than its information time"):
        replace(
            sequence,
            asof_evidence=replace(
                sequence.asof_evidence,
                decision_membership_known_at_ms=future_known,
            ),
        )

    factors = sequence.asof_evidence.corporate_action_factor.clone()
    factors[20:, 0, 1] = 2.0
    with pytest.raises(Hold30DatasetError, match="without a new corporate-action event version"):
        replace(
            sequence,
            asof_evidence=replace(
                sequence.asof_evidence,
                corporate_action_factor=factors,
            ),
        )


def test_c1_cannot_use_a_fill_addition_unknown_to_preceding_decision() -> None:
    positions = 98
    shape = (positions, 1, 4)
    decision_member = torch.ones(shape, dtype=torch.bool)
    fill_member = torch.ones(shape, dtype=torch.bool)
    decision_member[:11, 0, 2] = False
    fill_member[:11, 0, 2] = False
    sequence = _sequence(
        positions=positions,
        decision_membership=decision_member,
        fill_membership=fill_member,
    )
    invalid_c1 = sequence.c1_benchmark_weights.clone()
    invalid_c1[11, 0, 0] = 0.99
    invalid_c1[11, 0, 2] = 0.01

    with pytest.raises(Hold30DatasetError, match="preceding decision/fill"):
        replace(sequence, c1_benchmark_weights=invalid_c1)
