"""Replay and circularity tests for package-owned M03R confidence fitting."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import replace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_CANONICAL_SETTING_ID as V6_SETTING_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN_ID as V6_DESIGN_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_PROTOCOL_GENERATION as V6_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationError,
    apply_m03r_confidence_calibration,
)
from rl_quant.training.hold30_m03r_confidence_fit import (
    M03R_CONFIDENCE_ECE_BIN_COUNT,
    M03R_CONFIDENCE_ECE_BINNING_RULE_ID,
    M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID,
    M03R_CONFIDENCE_V6_TARGET_PATH_ID,
    M03RConfidenceCalibrationFitEvidence,
    M03RConfidenceFitError,
    M03RV6ConfidenceOutcomeEvidence,
    build_m03r_v6_confidence_outcome_evidence,
    compute_m03r_confidence_target_construction_sha256,
    fit_and_bind_m03r_confidence_calibration,
    replay_m03r_confidence_calibration_fit,
    validate_m03r_confidence_calibration_fit_evidence,
    validate_m03r_v6_confidence_outcome_evidence,
)

SETTING_ID = "M03R-active-alpha-hold30"
SEED = 19


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _arrays(
    permutation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    logits = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    folds = tuple("inner-00" if index < 6 else "inner-01" for index in range(12))
    dates = tuple(f"2024-01-{index + 2:02d}" for index in range(12))
    if permutation is None:
        return logits, targets, folds, dates
    order = tuple(int(index) for index in permutation.tolist())
    return (
        logits[permutation],
        targets[permutation],
        tuple(folds[index] for index in order),
        tuple(dates[index] for index in order),
    )


def _fit(
    arrays: tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]
    | None = None,
) -> M03RConfidenceCalibrationFitEvidence:
    logits, targets, folds, dates = _arrays() if arrays is None else arrays
    return fit_and_bind_m03r_confidence_calibration(
        setting_id=SETTING_ID,
        seed=SEED,
        checkpoint_sha256=_digest("frozen-checkpoint"),
        model_state_sha256=_digest("frozen-model-state"),
        raw_logits=logits,
        binary_targets=targets,
        fold_ids=folds,
        trading_sessions=dates,
        checkpoint_frozen_before_calibration=True,
    )


def _unit_risk_outcomes() -> torch.Tensor:
    return torch.tensor(
        [-0.06, -0.02, 0.01, -0.04, 0.03, -0.01, 0.08, 0.02, -0.03, 0.04, 0.07, 0.05],
        dtype=torch.float64,
    )


def _v6_daily_return_paths(
    outcomes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_outcomes = _unit_risk_outcomes() if outcomes is None else outcomes
    benchmark = torch.full((active_outcomes.numel(), 30), 0.0002, dtype=torch.float64)
    policy = torch.expm1(torch.log1p(benchmark) + active_outcomes.unsqueeze(1) / 30.0)
    return policy, benchmark


def _v6_outcome_evidence(
    outcomes: torch.Tensor | None = None,
    *,
    proposal_path_manifest_sha256: str | None = None,
    fold_ids: tuple[str, ...] | None = None,
    trading_sessions: tuple[str, ...] | None = None,
) -> M03RV6ConfidenceOutcomeEvidence:
    policy, benchmark = _v6_daily_return_paths(outcomes)
    _, _, default_folds, default_dates = _arrays()
    return build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=policy,
        c1_net_simple_returns=benchmark,
        fold_ids=default_folds if fold_ids is None else fold_ids,
        trading_sessions=(
            default_dates if trading_sessions is None else trading_sessions
        ),
        proposal_path_manifest_sha256=(
            _digest("authoritative-unit-risk-proposal-path")
            if proposal_path_manifest_sha256 is None
            else proposal_path_manifest_sha256
        ),
    )


def _fit_v6(
    outcome_evidence: M03RV6ConfidenceOutcomeEvidence | None = None,
) -> M03RConfidenceCalibrationFitEvidence:
    logits, _, folds, dates = _arrays()
    return fit_and_bind_m03r_confidence_calibration(
        setting_id=V6_SETTING_ID,
        seed=SEED,
        checkpoint_sha256=_digest("frozen-v6-checkpoint"),
        model_state_sha256=_digest("frozen-v6-model-state"),
        raw_logits=logits,
        binary_targets=None,
        fold_ids=folds,
        trading_sessions=dates,
        checkpoint_frozen_before_calibration=True,
        protocol_generation=V6_PROTOCOL_GENERATION,
        design_id=V6_DESIGN_ID,
        v6_outcome_evidence=(
            _v6_outcome_evidence() if outcome_evidence is None else outcome_evidence
        ),
    )


def test_fit_is_deterministic_permutation_invariant_and_exactly_replayable() -> None:
    first = _fit()
    repeated = _fit()
    permutation = torch.tensor([7, 1, 10, 3, 0, 11, 5, 4, 2, 9, 8, 6])
    permuted = _fit(_arrays(permutation))

    assert first == repeated == permuted
    assert first.evidence_sha256 == repeated.evidence_sha256
    logits, targets, folds, dates = _arrays(permutation)
    replay_m03r_confidence_calibration_fit(
        first,
        raw_logits=logits,
        binary_targets=targets,
        fold_ids=folds,
        trading_sessions=dates,
    )


def test_fit_recomputes_brier_ece_and_content_binds_fixed_bins() -> None:
    evidence = _fit()
    manifest = evidence.calibration_manifest
    logits, targets, _, _ = _arrays()
    probabilities = apply_m03r_confidence_calibration(
        logits,
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_setting_id=SETTING_ID,
        expected_seed=SEED,
        expected_checkpoint_sha256=manifest.checkpoint_sha256,
        expected_model_state_sha256=manifest.model_state_sha256,
        expected_source_score_array_sha256=manifest.source_score_array_sha256,
        expected_source_target_array_sha256=manifest.source_target_array_sha256,
    )
    expected_brier = float((probabilities - targets).square().mean())
    expected_ece = math.fsum(
        (row.observation_count / targets.numel()) * row.absolute_calibration_gap
        for row in evidence.ece_bins
    )

    assert manifest.brier_score == pytest.approx(expected_brier, abs=1e-15)
    assert manifest.expected_calibration_error == pytest.approx(
        expected_ece,
        abs=1e-15,
    )
    assert evidence.ece_binning_rule_id == M03R_CONFIDENCE_ECE_BINNING_RULE_ID
    assert evidence.ece_bin_count == M03R_CONFIDENCE_ECE_BIN_COUNT
    assert len(evidence.ece_bins) == M03R_CONFIDENCE_ECE_BIN_COUNT
    assert sum(row.observation_count for row in evidence.ece_bins) == targets.numel()
    assert evidence.ece_bins[-1].upper_edge_inclusive
    assert not any(row.upper_edge_inclusive for row in evidence.ece_bins[:-1])


def test_two_stage_protocol_freezes_policy_before_fit_and_forbids_feedback() -> None:
    evidence = _fit()
    assert evidence.two_stage_protocol_id == M03R_CONFIDENCE_TWO_STAGE_PROTOCOL_ID
    assert evidence.checkpoint_frozen_before_calibration
    assert not evidence.post_calibration_policy_updates_permitted
    signature = inspect.signature(fit_and_bind_m03r_confidence_calibration)
    assert "temperature" not in signature.parameters
    assert "intercept" not in signature.parameters
    assert "brier_score" not in signature.parameters
    assert "expected_calibration_error" not in signature.parameters
    assert "target_definition" not in signature.parameters
    assert "standardized_unit_risk_30_session_active_log_returns" not in (
        signature.parameters
    )
    assert "v6_outcome_evidence" in signature.parameters

    logits, targets, folds, dates = _arrays()
    with pytest.raises(M03RConfidenceFitError, match="frozen before calibration"):
        fit_and_bind_m03r_confidence_calibration(
            setting_id=SETTING_ID,
            seed=SEED,
            checkpoint_sha256=_digest("frozen-checkpoint"),
            model_state_sha256=_digest("frozen-model-state"),
            raw_logits=logits,
            binary_targets=targets,
            fold_ids=folds,
            trading_sessions=dates,
            checkpoint_frozen_before_calibration=False,
        )
    with pytest.raises(M03RConfidenceFitError, match="forbids later policy updates"):
        validate_m03r_confidence_calibration_fit_evidence(
            replace(evidence, post_calibration_policy_updates_permitted=True)
        )


def test_v6_fit_binds_unit_risk_proposal_target_before_confidence_sizing() -> None:
    evidence = _fit_v6()
    contract = evidence.target_construction_contract
    manifest = evidence.calibration_manifest
    logits, _, folds, dates = _arrays()
    outcome_evidence = _v6_outcome_evidence()
    outcomes = outcome_evidence.active_log_return_outcomes

    assert contract.protocol_generation == V6_PROTOCOL_GENERATION
    assert contract.design_id == V6_DESIGN_ID
    assert contract.proposal_path_id == M03R_CONFIDENCE_V6_TARGET_PATH_ID
    assert contract.standardized_unit_risk_proposal_required
    assert contract.final_confidence_sized_policy_path_prohibited
    assert contract.confidence_sizing_relationship == (
        "confidence-does-not-enter-target-outcome-path-v1"
    )
    assert "standardized-unit-risk" in contract.target_definition
    assert "confidence-sized" not in contract.target_definition
    assert contract.contract_sha256 == (
        compute_m03r_confidence_target_construction_sha256(contract)
    )
    assert manifest.protocol_generation == V6_PROTOCOL_GENERATION
    assert manifest.design_id == V6_DESIGN_ID
    assert manifest.target_definition == contract.target_definition
    assert evidence.source_standardized_unit_risk_active_log_return_array_sha256
    assert evidence.v6_outcome_receipt == outcome_evidence.receipt
    assert evidence.v6_outcome_receipt.proposal_path_manifest_sha256 == _digest(
        "authoritative-unit-risk-proposal-path"
    )
    assert manifest.observed_target_rate == pytest.approx(
        float((outcomes > 0.0).to(dtype=torch.float64).mean())
    )
    assert manifest.source_target_array_sha256 != (
        _fit().calibration_manifest.source_target_array_sha256
    )

    calibrated = apply_m03r_confidence_calibration(
        logits,
        manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_setting_id=V6_SETTING_ID,
        expected_seed=SEED,
        expected_checkpoint_sha256=manifest.checkpoint_sha256,
        expected_model_state_sha256=manifest.model_state_sha256,
        expected_source_score_array_sha256=manifest.source_score_array_sha256,
        expected_source_target_array_sha256=manifest.source_target_array_sha256,
        expected_protocol_generation=V6_PROTOCOL_GENERATION,
        expected_design_id=V6_DESIGN_ID,
    )
    assert calibrated.shape == logits.shape
    replay_m03r_confidence_calibration_fit(
        evidence,
        raw_logits=logits,
        binary_targets=None,
        fold_ids=folds,
        trading_sessions=dates,
        v6_outcome_evidence=outcome_evidence,
    )
    with pytest.raises(M03RConfidenceCalibrationError, match="setting"):
        apply_m03r_confidence_calibration(
            logits,
            manifest,
            expected_manifest_sha256=manifest.manifest_sha256,
            expected_setting_id=V6_SETTING_ID,
            expected_seed=SEED,
            expected_checkpoint_sha256=manifest.checkpoint_sha256,
            expected_model_state_sha256=manifest.model_state_sha256,
            expected_source_score_array_sha256=manifest.source_score_array_sha256,
            expected_source_target_array_sha256=manifest.source_target_array_sha256,
        )


def test_v6_outcome_constructor_computes_exact_path_and_rejects_invalid_returns() -> (
    None
):
    policy, benchmark = _v6_daily_return_paths()
    _, _, folds, dates = _arrays()
    evidence = _v6_outcome_evidence()
    expected = torch.log1p(policy).sum(dim=1) - torch.log1p(benchmark).sum(dim=1)

    assert torch.equal(evidence.active_log_return_outcomes, expected)
    assert evidence.receipt.observation_count == policy.shape[0]
    assert evidence.receipt.post_fill_return_count == 30
    validate_m03r_v6_confidence_outcome_evidence(evidence)

    invalid_policy = policy.clone()
    invalid_policy[0, 0] = -1.0
    with pytest.raises(M03RConfidenceFitError, match="strictly greater than -1"):
        build_m03r_v6_confidence_outcome_evidence(
            standardized_unit_risk_policy_net_simple_returns=invalid_policy,
            c1_net_simple_returns=benchmark,
            fold_ids=folds,
            trading_sessions=dates,
            proposal_path_manifest_sha256=_digest("proposal-path"),
        )
    with pytest.raises(M03RConfidenceFitError, match=r"\[observation,30\]"):
        build_m03r_v6_confidence_outcome_evidence(
            standardized_unit_risk_policy_net_simple_returns=policy[:, :-1],
            c1_net_simple_returns=benchmark[:, :-1],
            fold_ids=folds,
            trading_sessions=dates,
            proposal_path_manifest_sha256=_digest("proposal-path"),
        )
    tampered_policy = evidence.standardized_unit_risk_policy_net_simple_returns.clone()
    tampered_policy[0, 0] += 0.001
    with pytest.raises(M03RConfidenceFitError, match="do not reproduce"):
        validate_m03r_v6_confidence_outcome_evidence(
            replace(
                evidence,
                standardized_unit_risk_policy_net_simple_returns=tampered_policy,
            )
        )


def test_v6_outcome_row_order_is_bound_to_the_exact_logit_rows() -> None:
    logits, _, folds, dates = _arrays()
    policy, benchmark = _v6_daily_return_paths()
    base = _v6_outcome_evidence()
    permutation = torch.tensor([7, 1, 10, 3, 0, 11, 5, 4, 2, 9, 8, 6])
    order = tuple(int(index) for index in permutation.tolist())
    reordered = build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=policy[permutation],
        c1_net_simple_returns=benchmark[permutation],
        fold_ids=tuple(folds[index] for index in order),
        trading_sessions=tuple(dates[index] for index in order),
        proposal_path_manifest_sha256=(base.receipt.proposal_path_manifest_sha256),
    )

    assert reordered.receipt.ordered_row_identity_sha256 != (
        base.receipt.ordered_row_identity_sha256
    )
    with pytest.raises(M03RConfidenceFitError, match="exact logit/fitter row order"):
        fit_and_bind_m03r_confidence_calibration(
            setting_id=V6_SETTING_ID,
            seed=SEED,
            checkpoint_sha256=_digest("frozen-v6-checkpoint"),
            model_state_sha256=_digest("frozen-v6-model-state"),
            raw_logits=logits,
            binary_targets=None,
            fold_ids=folds,
            trading_sessions=dates,
            checkpoint_frozen_before_calibration=True,
            protocol_generation=V6_PROTOCOL_GENERATION,
            design_id=V6_DESIGN_ID,
            v6_outcome_evidence=reordered,
        )

    tampered_rows = list(base.trading_sessions)
    tampered_rows[0], tampered_rows[1] = tampered_rows[1], tampered_rows[0]
    with pytest.raises(M03RConfidenceFitError, match="row identities"):
        validate_m03r_v6_confidence_outcome_evidence(
            replace(base, trading_sessions=tuple(tampered_rows))
        )


def test_v6_target_definition_is_package_owned_and_substitution_fails_closed() -> None:
    evidence = _fit_v6()
    contract = evidence.target_construction_contract
    substituted = replace(
        contract,
        target_definition=(
            "probability-final-confidence-sized-policy-path-is-positive"
        ),
    )
    with pytest.raises(M03RConfidenceFitError, match="frozen generation contract"):
        validate_m03r_confidence_calibration_fit_evidence(
            replace(evidence, target_construction_contract=substituted)
        )

    logits, targets, folds, dates = _arrays()
    with pytest.raises(M03RConfidenceFitError, match="caller-authored binary_targets"):
        fit_and_bind_m03r_confidence_calibration(
            setting_id=V6_SETTING_ID,
            seed=SEED,
            checkpoint_sha256=_digest("frozen-v6-checkpoint"),
            model_state_sha256=_digest("frozen-v6-model-state"),
            raw_logits=logits,
            binary_targets=targets,
            fold_ids=folds,
            trading_sessions=dates,
            checkpoint_frozen_before_calibration=True,
            protocol_generation=V6_PROTOCOL_GENERATION,
            design_id=V6_DESIGN_ID,
            v6_outcome_evidence=_v6_outcome_evidence(),
        )
    with pytest.raises(M03RConfidenceFitError, match="typed economic-path"):
        fit_and_bind_m03r_confidence_calibration(
            setting_id=V6_SETTING_ID,
            seed=SEED,
            checkpoint_sha256=_digest("frozen-v6-checkpoint"),
            model_state_sha256=_digest("frozen-v6-model-state"),
            raw_logits=logits,
            binary_targets=None,
            fold_ids=folds,
            trading_sessions=dates,
            checkpoint_frozen_before_calibration=True,
            protocol_generation=V6_PROTOCOL_GENERATION,
            design_id=V6_DESIGN_ID,
        )
    policy, benchmark = _v6_daily_return_paths()
    with pytest.raises(M03RConfidenceFitError, match="detached finite floating"):
        build_m03r_v6_confidence_outcome_evidence(
            standardized_unit_risk_policy_net_simple_returns=(
                policy.requires_grad_(True)
            ),
            c1_net_simple_returns=benchmark,
            fold_ids=folds,
            trading_sessions=dates,
            proposal_path_manifest_sha256=_digest("proposal-path"),
        )
    with pytest.raises(M03RConfidenceFitError, match="immutable v6 design ID"):
        fit_and_bind_m03r_confidence_calibration(
            setting_id=V6_SETTING_ID,
            seed=SEED,
            checkpoint_sha256=_digest("frozen-v6-checkpoint"),
            model_state_sha256=_digest("frozen-v6-model-state"),
            raw_logits=logits,
            binary_targets=targets,
            fold_ids=folds,
            trading_sessions=dates,
            checkpoint_frozen_before_calibration=True,
            protocol_generation=V6_PROTOCOL_GENERATION,
            design_id="caller-substituted-design",
        )


def test_v6_outcome_array_is_content_bound_and_replayed_before_sign_target() -> None:
    base_outcome_evidence = _v6_outcome_evidence()
    base = _fit_v6(base_outcome_evidence)
    policy = base_outcome_evidence.standardized_unit_risk_policy_net_simple_returns
    benchmark = base_outcome_evidence.c1_net_simple_returns
    mutated_policy = policy.clone()
    mutated_policy[0, 0] -= 0.0001  # Same sign, different economic path/outcome.
    mutated_outcome_evidence = build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=mutated_policy,
        c1_net_simple_returns=benchmark,
        fold_ids=base_outcome_evidence.fold_ids,
        trading_sessions=base_outcome_evidence.trading_sessions,
        proposal_path_manifest_sha256=(
            base_outcome_evidence.receipt.proposal_path_manifest_sha256
        ),
    )
    mutated = _fit_v6(mutated_outcome_evidence)
    base_receipt = base.v6_outcome_receipt
    mutated_receipt = mutated.v6_outcome_receipt
    assert base_receipt is not None
    assert mutated_receipt is not None

    assert (
        mutated_receipt.policy_daily_net_simple_return_array_sha256
        != base_receipt.policy_daily_net_simple_return_array_sha256
    )
    assert (
        mutated_receipt.computed_active_log_return_outcome_array_sha256
        != base_receipt.computed_active_log_return_outcome_array_sha256
    )
    assert (
        mutated.source_standardized_unit_risk_active_log_return_array_sha256
        != base.source_standardized_unit_risk_active_log_return_array_sha256
    )
    assert (
        mutated.calibration_manifest.source_target_array_sha256
        != base.calibration_manifest.source_target_array_sha256
    )
    assert mutated.evidence_sha256 != base.evidence_sha256

    logits, _, folds, dates = _arrays()
    with pytest.raises(M03RConfidenceFitError, match="do not replay"):
        replay_m03r_confidence_calibration_fit(
            base,
            raw_logits=logits,
            binary_targets=None,
            fold_ids=folds,
            trading_sessions=dates,
            v6_outcome_evidence=mutated_outcome_evidence,
        )

    changed_path = _v6_outcome_evidence(
        proposal_path_manifest_sha256=_digest("different-authoritative-path")
    )
    assert changed_path.receipt.evidence_sha256 != (
        base_outcome_evidence.receipt.evidence_sha256
    )
    assert _fit_v6(changed_path).evidence_sha256 != base.evidence_sha256

    mutated_benchmark = benchmark.clone()
    mutated_benchmark[1, 0] += 0.0001
    changed_c1 = build_m03r_v6_confidence_outcome_evidence(
        standardized_unit_risk_policy_net_simple_returns=policy,
        c1_net_simple_returns=mutated_benchmark,
        fold_ids=base_outcome_evidence.fold_ids,
        trading_sessions=base_outcome_evidence.trading_sessions,
        proposal_path_manifest_sha256=(
            base_outcome_evidence.receipt.proposal_path_manifest_sha256
        ),
    )
    assert changed_c1.receipt.c1_daily_net_simple_return_array_sha256 != (
        base_outcome_evidence.receipt.c1_daily_net_simple_return_array_sha256
    )
    assert changed_c1.receipt.computed_active_log_return_outcome_array_sha256 != (
        base_outcome_evidence.receipt.computed_active_log_return_outcome_array_sha256
    )


def test_actual_source_arrays_folds_dates_and_checkpoint_identity_move_evidence() -> (
    None
):
    base = _fit()
    logits, targets, folds, dates = _arrays()
    changed_logits = logits.clone()
    changed_logits[3] += 0.25
    changed_score = _fit((changed_logits, targets, folds, dates))
    assert (
        changed_score.calibration_manifest.source_score_array_sha256
        != base.calibration_manifest.source_score_array_sha256
    )
    assert changed_score.evidence_sha256 != base.evidence_sha256

    changed_folds = list(folds)
    changed_folds[3] = "inner-02"
    changed_fold = _fit((logits, targets, tuple(changed_folds), dates))
    assert changed_fold.source_fold_array_sha256 != base.source_fold_array_sha256
    assert changed_fold.evidence_sha256 != base.evidence_sha256

    changed_dates = list(dates)
    changed_dates[3] = "2024-02-20"
    changed_date = _fit((logits, targets, folds, tuple(changed_dates)))
    assert changed_date.source_date_array_sha256 != base.source_date_array_sha256
    assert changed_date.evidence_sha256 != base.evidence_sha256

    changed_checkpoint = fit_and_bind_m03r_confidence_calibration(
        setting_id=SETTING_ID,
        seed=SEED,
        checkpoint_sha256=_digest("another-frozen-checkpoint"),
        model_state_sha256=_digest("another-frozen-model-state"),
        raw_logits=logits,
        binary_targets=targets,
        fold_ids=folds,
        trading_sessions=dates,
        checkpoint_frozen_before_calibration=True,
    )
    assert (
        changed_checkpoint.calibration_manifest.checkpoint_sha256
        != base.calibration_manifest.checkpoint_sha256
    )
    assert changed_checkpoint.evidence_sha256 != base.evidence_sha256


def test_replay_and_validation_fail_closed_for_tampering() -> None:
    evidence = _fit()
    logits, targets, folds, dates = _arrays()
    mutated = logits.clone()
    mutated[0] += 0.01
    with pytest.raises(M03RConfidenceFitError, match="do not replay"):
        replay_m03r_confidence_calibration_fit(
            evidence,
            raw_logits=mutated,
            binary_targets=targets,
            fold_ids=folds,
            trading_sessions=dates,
        )
    with pytest.raises(M03RConfidenceFitError, match="binning_rule_id drifted"):
        validate_m03r_confidence_calibration_fit_evidence(
            replace(evidence, ece_binning_rule_id="caller-selected-bins")
        )
    with pytest.raises(M03RConfidenceFitError, match="payload"):
        validate_m03r_confidence_calibration_fit_evidence(
            replace(
                evidence, final_binary_log_loss=evidence.final_binary_log_loss + 1.0
            )
        )


def test_fit_rejects_non_evidence_inputs_and_duplicate_row_identity() -> None:
    logits, targets, folds, dates = _arrays()
    cases = (
        (logits.clone().requires_grad_(True), targets, folds, dates, "detached"),
        (logits, torch.zeros_like(targets), folds, dates, "both binary"),
        (
            logits,
            targets,
            folds,
            (dates[0], dates[0], *dates[2:]),
            "must be unique",
        ),
    )
    for case_logits, case_targets, case_folds, case_dates, message in cases:
        with pytest.raises(M03RConfidenceFitError, match=message):
            fit_and_bind_m03r_confidence_calibration(
                setting_id=SETTING_ID,
                seed=SEED,
                checkpoint_sha256=_digest("frozen-checkpoint"),
                model_state_sha256=_digest("frozen-model-state"),
                raw_logits=case_logits,
                binary_targets=case_targets,
                fold_ids=case_folds,
                trading_sessions=case_dates,
                checkpoint_frozen_before_calibration=True,
            )
