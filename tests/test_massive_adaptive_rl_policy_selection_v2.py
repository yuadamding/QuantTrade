from __future__ import annotations

from dataclasses import asdict, replace
from io import BytesIO
import inspect
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.training import massive_adaptive_rl_policy_selection_v2 as selection_v2
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256,
    MassiveAdaptiveRLPolicyCandidateV2,
    MassiveAdaptiveRLPolicySelectionV2Error,
    build_massive_adaptive_rl_policy_candidate_v2,
    authorize_massive_adaptive_rl_policy_selection_authority_v2,
    materialize_massive_adaptive_rl_policy_selection_authority_v2,
    parse_massive_adaptive_rl_policy_selection_authority_v2,
    select_massive_adaptive_rl_policy_v2,
    validation_rank_key_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from test_massive_adaptive_rl_policy_selection_v1 import (
    _checkpoint,
    _fixed_control_authority,
    _fixed_validation_trace,
    _trace,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _failures(
    *,
    primary_incremental: float,
    ppo_minus_fc06: float,
    active: float,
    low: float,
    primary_terminal: float,
    high: float,
    drawdown: float,
) -> tuple[str, ...]:
    criteria = MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
    values: list[str] = []
    if primary_incremental <= 0.0:
        values.append(criteria[0])
    if ppo_minus_fc06 <= 0.0:
        values.append(criteria[1])
    if active <= 0.0:
        values.append(criteria[2])
    if high < 0.0:
        values.append(criteria[3])
    if not low >= primary_terminal >= high:
        values.append(criteria[4])
    if drawdown > 0.25:
        values.append(criteria[5])
    return tuple(sorted(values))


def _candidate(
    *,
    manifest,
    ordinal: int,
    fold_index: int = 1,
    primary_incremental: float = 0.03,
    ppo_minus_fc06: float = 0.01,
    active: float = 0.02,
    low: float = 0.04,
    primary_terminal: float = 0.03,
    high: float = 0.01,
    drawdown: float = 0.05,
    update_index: int | None = None,
    checkpoint_receipt: str | None = None,
    checkpoint_authority_receipt: str | None = None,
    model_state_receipt: str | None = None,
    training_forecast_authority_receipt: str | None = None,
    fixed_control_selection_authority_receipt: str | None = None,
    selected_fc06_action_receipt: str | None = None,
    source_data_qualified: bool = True,
) -> MassiveAdaptiveRLPolicyCandidateV2:
    update = (
        manifest.base_manifest.base_manifest.schedule(
            fold_index
        ).candidate_update_indices[ordinal]
        if update_index is None
        else update_index
    )
    checkpoint = checkpoint_receipt or _digest(("checkpoint", fold_index, ordinal))
    checkpoint_authority = checkpoint_authority_receipt or _digest(
        ("checkpoint-authority", fold_index, ordinal)
    )
    fc06_incremental = float(primary_incremental - ppo_minus_fc06)
    derived_ppo_minus_fc06 = float(primary_incremental - fc06_incremental)
    failures = _failures(
        primary_incremental=primary_incremental,
        ppo_minus_fc06=derived_ppo_minus_fc06,
        active=active,
        low=low,
        primary_terminal=primary_terminal,
        high=high,
        drawdown=drawdown,
    )
    body = {
        "schema": selection_v2.MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SCHEMA,
        "manifest_v4_receipt_sha256": manifest.semantic_receipt_sha256,
        "training_manifest_v3_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "fold_index": fold_index,
        "checkpoint_authority_receipt_sha256": checkpoint_authority,
        "checkpoint_receipt_sha256": checkpoint,
        "model_state_receipt_sha256": (
            model_state_receipt or _digest(("model", fold_index, ordinal))
        ),
        "update_index": update,
        "training_forecast_authority_receipt_sha256": (
            training_forecast_authority_receipt or _digest(("forecast", fold_index))
        ),
        "primary_trace_receipt_sha256": _digest(("primary", fold_index, ordinal)),
        "low_cost_trace_receipt_sha256": _digest(("low", fold_index, ordinal)),
        "high_cost_trace_receipt_sha256": _digest(("high", fold_index, ordinal)),
        "decision_target_inventory_sha256": _digest(("targets", fold_index, ordinal)),
        "fixed_control_selection_authority_receipt_sha256": (
            fixed_control_selection_authority_receipt
            or _digest(("fixed-selection", fold_index))
        ),
        "selected_fc06_action_receipt_sha256": (
            selected_fc06_action_receipt or _digest(("fc06", fold_index))
        ),
        "fc06_validation_trace_receipt_sha256": _digest(
            ("fc06-validation", fold_index)
        ),
        "legacy_candidate_v1_receipt_sha256": _digest(
            ("legacy", fold_index, ordinal)
        ),
        "fc06_primary_incremental_log_wealth": fc06_incremental,
        "ppo_minus_fc06_log_wealth": derived_ppo_minus_fc06,
        "primary_incremental_rl_log_wealth": float(primary_incremental),
        "primary_strategy_active_log_wealth": float(active),
        "low_cost_terminal_liquidation_adjusted_return": float(low),
        "primary_cost_terminal_liquidation_adjusted_return": float(
            primary_terminal
        ),
        "high_cost_terminal_liquidation_adjusted_return": float(high),
        "maximum_drawdown": float(drawdown),
        "validation_eligibility_failures": failures,
        "economically_eligible": not failures,
        "source_data_qualified": source_data_qualified,
        "validation_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_tie_breaking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "numerical_comparison_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_NUMERICAL_COMPARISON_V1_SPEC_SHA256
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_CANDIDATE_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLPolicyCandidateV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _select(manifest, *candidates: MassiveAdaptiveRLPolicyCandidateV2):
    ordered = tuple(sorted(candidates, key=lambda row: row.update_index))
    return select_massive_adaptive_rl_policy_v2(
        manifest=manifest,
        fold_fit_authority_receipt_sha256=_digest("fold-fit"),
        expected_candidate_checkpoint_authority_receipts=tuple(
            row.checkpoint_authority_receipt_sha256 for row in ordered
        ),
        candidates=tuple(reversed(ordered)),
    )


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (
            {"primary_incremental": 0.04, "ppo_minus_fc06": 0.005},
            {"primary_incremental": 0.03, "ppo_minus_fc06": 0.02},
        ),
        (
            {"ppo_minus_fc06": 0.02, "active": 0.005},
            {"ppo_minus_fc06": 0.01, "active": 0.03},
        ),
        ({"active": 0.03, "high": 0.005}, {"active": 0.02, "high": 0.02}),
        ({"high": 0.02, "drawdown": 0.20}, {"high": 0.01, "drawdown": 0.01}),
        ({"drawdown": 0.04}, {"drawdown": 0.05}),
    ),
)
def test_v4_metric_precedence_is_lexicographic(
    first: dict[str, float],
    second: dict[str, float],
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-metric-order"
    )
    candidate_a = _candidate(manifest=manifest, ordinal=0, **first)
    candidate_b = _candidate(manifest=manifest, ordinal=1, **second)

    selected = _select(manifest, candidate_a, candidate_b)

    assert selected.selected_candidate_receipt_sha256 == (
        candidate_a.semantic_receipt_sha256
    )
    assert selected.ranked_candidate_receipts[0] == candidate_a.semantic_receipt_sha256


def test_v4_update_and_checkpoint_receipt_complete_the_total_order() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-total-order"
    )
    earlier = _candidate(manifest=manifest, ordinal=0)
    later = _candidate(manifest=manifest, ordinal=1)
    assert _select(manifest, later, earlier).selected_update_index == earlier.update_index

    lexical_high = _candidate(
        manifest=manifest,
        ordinal=0,
        update_index=earlier.update_index,
        checkpoint_receipt="f" * 64,
    )
    lexical_low = _candidate(
        manifest=manifest,
        ordinal=1,
        update_index=earlier.update_index,
        checkpoint_receipt="0" * 64,
    )
    assert min((lexical_high, lexical_low), key=validation_rank_key_v1) is lexical_low


def test_v4_eligible_pool_beats_higher_ranked_ineligible_candidate() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-eligible-pool"
    )
    higher_but_ineligible = _candidate(
        manifest=manifest,
        ordinal=0,
        primary_incremental=0.05,
        high=-0.01,
    )
    lower_but_eligible = _candidate(
        manifest=manifest,
        ordinal=1,
        primary_incremental=0.03,
    )

    selected = _select(manifest, higher_but_ineligible, lower_but_eligible)

    assert selected.selection_pool_kind == "eligible"
    assert selected.selected_candidate_validation_eligible
    assert selected.selected_candidate_receipt_sha256 == (
        lower_but_eligible.semantic_receipt_sha256
    )


def test_v4_no_eligible_pool_selects_diagnostic_candidate_without_exception() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-no-eligible"
    )
    higher = _candidate(
        manifest=manifest,
        ordinal=0,
        primary_incremental=0.05,
        drawdown=0.30,
    )
    lower = _candidate(
        manifest=manifest,
        ordinal=1,
        primary_incremental=0.03,
        drawdown=0.40,
    )

    selected = _select(manifest, higher, lower)

    assert selected.selection_pool_kind == "all-no-eligible"
    assert selected.selected_candidate_receipt_sha256 == higher.semantic_receipt_sha256
    assert not selected.selected_candidate_validation_eligible
    assert not selected.positive_profitability_authorization_eligible
    assert selected.validation_eligibility_failures == (
        "maximum-drawdown-at-most-0.25",
    )
    assert not selected.profitability_reporting_authorized
    assert not selected.outer_evaluation_authorized


def test_v4_numeric_semantics_are_exact_and_canonical() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-binary64"
    )
    positive = _candidate(
        manifest=manifest,
        ordinal=0,
        primary_incremental=1.0e-15,
        ppo_minus_fc06=1.0e-15,
    )
    zero = _candidate(
        manifest=manifest,
        ordinal=1,
        primary_incremental=0.0,
        ppo_minus_fc06=0.0,
    )
    assert positive.economically_eligible
    assert not zero.economically_eligible
    assert {
        "primary-incremental-rl-log-wealth-strictly-positive",
        "ppo-minus-fc06-log-wealth-strictly-positive",
    }.issubset(zero.validation_eligibility_failures)

    negative_zero = replace(
        positive,
        primary_incremental_rl_log_wealth=-0.0,
        semantic_receipt_sha256="0" * 64,
    )
    with pytest.raises(MassiveAdaptiveRLPolicySelectionV2Error, match="differs"):
        negative_zero.validate()
    nonfinite = replace(
        positive,
        maximum_drawdown=float("nan"),
    )
    with pytest.raises(MassiveAdaptiveRLPolicySelectionV2Error, match="differs"):
        nonfinite.validate()


def test_unqualified_candidate_cannot_claim_positive_authorization_eligibility() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-unqualified"
    )
    candidate = _candidate(
        manifest=manifest,
        ordinal=0,
        fold_index=0,
        source_data_qualified=False,
    )

    selection = _select(manifest, candidate)

    assert selection.selected_candidate_validation_eligible
    assert not selection.source_data_qualified
    assert not selection.positive_profitability_authorization_eligible


def test_authorizing_selection_api_accepts_only_replayed_fold_validation() -> None:
    for operation in (
        materialize_massive_adaptive_rl_policy_selection_authority_v2,
        authorize_massive_adaptive_rl_policy_selection_authority_v2,
    ):
        parameters = inspect.signature(operation).parameters
        assert "validation_authority" in parameters
        assert "candidates" not in parameters
        assert "fold_fit_authority" not in parameters


def test_v4_candidate_coverage_schedule_and_lineage_are_exact() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-coverage"
    )
    first = _candidate(manifest=manifest, ordinal=0)
    second = _candidate(manifest=manifest, ordinal=1)

    with pytest.raises(MassiveAdaptiveRLPolicySelectionV2Error, match="coverage"):
        select_massive_adaptive_rl_policy_v2(
            manifest=manifest,
            fold_fit_authority_receipt_sha256=_digest("fold-fit"),
            expected_candidate_checkpoint_authority_receipts=(
                first.checkpoint_authority_receipt_sha256,
                second.checkpoint_authority_receipt_sha256,
            ),
            candidates=(first,),
        )

    wrong_update = _candidate(
        manifest=manifest,
        ordinal=1,
        update_index=999,
    )
    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV2Error,
        match="lineage or schedule",
    ):
        _select(manifest, first, wrong_update)

    another_manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-other-manifest"
    )
    with pytest.raises(
        MassiveAdaptiveRLPolicySelectionV2Error,
        match="lineage or schedule",
    ):
        _select(another_manifest, first, second)


def test_v4_candidate_failure_inventory_cannot_be_promoted() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-failure-tamper"
    )
    candidate = _candidate(
        manifest=manifest,
        ordinal=0,
        fold_index=0,
        drawdown=0.30,
    )
    promoted = replace(
        candidate,
        validation_eligibility_failures=(),
        economically_eligible=True,
        semantic_receipt_sha256="0" * 64,
    )
    promoted = replace(
        promoted,
        semantic_receipt_sha256=semantic_sha256(promoted.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLPolicySelectionV2Error, match="differs"):
        promoted.validate()


def test_candidate_v2_derives_exact_v4_failures_from_validated_v1_evidence(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-evidence-builder"
    )
    candidate = build_massive_adaptive_rl_policy_candidate_v2(
        manifest=manifest,
        checkpoint_authority_receipt_sha256=_digest("checkpoint-authority"),
        checkpoint=_checkpoint(),  # type: ignore[arg-type]
        low_cost_trace=_trace(
            cost=10.0,
            terminal_return=0.03,
            incremental=0.004,
            active=0.02,
            frozen=True,
        ),
        primary_trace=_trace(
            cost=20.0,
            terminal_return=0.02,
            incremental=0.003,
            active=0.015,
            frozen=False,
        ),
        high_cost_trace=_trace(
            cost=40.0,
            terminal_return=0.001,
            incremental=0.001,
            active=0.005,
            frozen=True,
        ),
        fixed_control_selection_authority=_fixed_control_authority(tmp_path),
        fixed_control_validation_trace=_fixed_validation_trace(),
    )

    assert candidate.manifest_v4_receipt_sha256 == manifest.semantic_receipt_sha256
    assert candidate.ppo_minus_fc06_log_wealth < 0.0
    assert candidate.validation_eligibility_failures == (
        "ppo-minus-fc06-log-wealth-strictly-positive",
    )
    assert not candidate.economically_eligible


def test_policy_selection_v2_generic_authority_is_persisted_but_nonauthorizing(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="selection-v2-persisted-generic"
    )
    candidate = _candidate(manifest=manifest, ordinal=0, fold_index=0)
    selection = _select(manifest, candidate)
    relative = "massive-adaptive/rl-policy-selection-v2/generic.json"
    publish_massive_source_object(
        stream=BytesIO(
            canonical_json_file_bytes(
                {
                    "fold_validation_authority_receipt_sha256": _digest(
                        "persisted-fold-validation"
                    ),
                    "selection": asdict(selection),
                    "candidates": (asdict(candidate),),
                }
            )
        ),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=(
            selection_v2.MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_DATASET
        ),
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=(
            selection_v2.MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=selection.semantic_receipt_sha256,
        committed_at_ms=1,
        request_id="ADAPTIVE-RL-POLICY-SELECTION-V2-GENERIC",
    )
    loaded = load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path=relative,
        verified_at_ms=2,
    )

    generic = parse_massive_adaptive_rl_policy_selection_authority_v2(
        root=tmp_path,
        loaded_source=loaded,
    )

    assert generic.selection_receipt_sha256 == selection.semantic_receipt_sha256
    assert generic.fold_validation_authority_receipt_sha256 == _digest(
        "persisted-fold-validation"
    )
    assert generic.source_data_qualified
    assert generic.selected_candidate_validation_eligible
    assert generic.positive_profitability_authorization_eligible
    assert not generic.runtime_selection_replayed
    assert not generic.development_policy_selection_authorized
    assert not generic.policy_freezing_authorized
    assert not generic.outer_diagnostic_preparation_authorized
    assert not generic.profitability_reporting_authorized
