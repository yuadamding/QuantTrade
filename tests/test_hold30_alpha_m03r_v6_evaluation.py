"""Generation-qualified public numerical evaluation tests for M03R v6."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from rl_quant.evaluation.hold30_alpha_m03r_v6 import (
    M03R_V6_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS,
    M03R_V6_PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    M03RV6EvaluationError,
    M03RV6FactorManifest,
    M03RV6InferenceManifest,
    build_m03r_v6_factor_manifest,
    build_m03r_v6_inference_manifest,
    evaluate_m03r_v6_inference,
    m03r_v6_candidate_policy_returns_sha256,
    m03r_v6_common_evaluator_inputs_sha256,
    validate_m03r_v6_evaluation_receipt,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_PROTOCOL_GENERATION as M03R_V5_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_CANONICAL_SETTING_ID,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
)
from rl_quant.workflows.hold30_prelockbox import HOLD30_COMPONENT_TESTS


def _digest(character: str) -> str:
    return character * 64


def _factor_manifest() -> M03RV6FactorManifest:
    return build_m03r_v6_factor_manifest(
        factor_names=("SIZE", "VALUE"),
        factor_return_conventions=(
            "daily-simple-long-short-return",
            "daily-simple-long-short-return",
        ),
        point_in_time_source_manifest_sha256=_digest("1"),
    )


def _inference_manifest(
    factor_manifest: M03RV6FactorManifest,
) -> M03RV6InferenceManifest:
    return build_m03r_v6_inference_manifest(
        factor_manifest=factor_manifest,
        bootstrap_replicates=1_000,
        bootstrap_seed_sha256=_digest("2"),
    )


def _panel() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(91)
    shape = (6, 63)
    risk_free = np.full(shape, 0.0001)
    market = rng.normal(0.0004, 0.008, size=shape)
    market_excess = market - risk_free
    factors = rng.normal(0.0, 0.003, size=(*shape, 2))
    benchmark = (
        risk_free
        + 0.0001
        + 0.90 * market_excess
        + 0.20 * factors[..., 0]
        - 0.10 * factors[..., 1]
    )
    active = (
        0.0002 + 0.05 * market_excess + 0.10 * factors[..., 0] + 0.05 * factors[..., 1]
    )
    return benchmark + active, benchmark, risk_free, market, factors


def _metadata() -> tuple[np.ndarray, np.ndarray]:
    start = date(2020, 1, 1)
    dates = np.asarray(
        [(start + timedelta(days=index)).isoformat() for index in range(6 * 63)]
    ).reshape(6, 63)
    folds = np.asarray(
        [[f"outer-{fold}"] * 63 for fold in range(6)],
        dtype=object,
    )
    return dates, folds


def _input_hashes(
    panel: tuple[np.ndarray, ...],
    factor_manifest: M03RV6FactorManifest,
    inference_manifest: M03RV6InferenceManifest,
) -> tuple[str, str]:
    policy, benchmark, risk_free, market, factors = panel
    dates, folds = _metadata()
    common = m03r_v6_common_evaluator_inputs_sha256(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        score_dates=dates,
        fold_ids=folds,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    candidate = m03r_v6_candidate_policy_returns_sha256(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
        policy_net_returns=policy,
        common_evaluator_inputs_sha256=common,
    )
    return common, candidate


def _evaluate(
    panel: tuple[np.ndarray, ...] | None = None,
    *,
    factor_manifest: M03RV6FactorManifest | None = None,
    inference_manifest: M03RV6InferenceManifest | None = None,
    common_evaluator_inputs_sha256: str | None = None,
    candidate_policy_returns_sha256: str | None = None,
) -> tuple[dict[str, object], M03RV6FactorManifest, M03RV6InferenceManifest]:
    resolved_panel = _panel() if panel is None else panel
    resolved_factor = _factor_manifest() if factor_manifest is None else factor_manifest
    resolved_inference = (
        _inference_manifest(resolved_factor)
        if inference_manifest is None
        else inference_manifest
    )
    policy, benchmark, risk_free, market, factors = resolved_panel
    dates, folds = _metadata()
    computed_common, computed_candidate = _input_hashes(
        resolved_panel,
        resolved_factor,
        resolved_inference,
    )
    receipt = evaluate_m03r_v6_inference(
        protocol_generation=M03R_PROTOCOL_GENERATION,
        design_id=M03R_DESIGN_ID,
        setting_id=M03R_CANONICAL_SETTING_ID,
        score_dates=dates,
        fold_ids=folds,
        policy_net_returns=policy,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        factor_manifest=resolved_factor,
        inference_manifest=resolved_inference,
        common_evaluator_inputs_sha256=(
            computed_common
            if common_evaluator_inputs_sha256 is None
            else common_evaluator_inputs_sha256
        ),
        candidate_policy_returns_sha256=(
            computed_candidate
            if candidate_policy_returns_sha256 is None
            else candidate_policy_returns_sha256
        ),
    )
    return receipt, resolved_factor, resolved_inference


def test_v6_reports_portfolio_benchmark_and_active_multifactor_alpha() -> None:
    receipt, factor_manifest, inference_manifest = _evaluate()
    validate_m03r_v6_evaluation_receipt(
        receipt,
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    portfolio = receipt["portfolio_multifactor_regression"]
    benchmark = receipt["benchmark_multifactor_regression"]
    active = receipt["active_multifactor_regression"]
    assert active["alpha_daily"] == pytest.approx(0.0002)
    assert benchmark["alpha_daily"] == pytest.approx(0.0001)
    assert portfolio["alpha_daily"] == pytest.approx(0.0003)
    assert active["alpha_daily"] == pytest.approx(
        portfolio["alpha_daily"] - benchmark["alpha_daily"]
    )
    for factor_name in ("PIT_CAP_MARKET_EXCESS", "SIZE", "VALUE"):
        assert active["loadings"][factor_name] == pytest.approx(
            portfolio["loadings"][factor_name] - benchmark["loadings"][factor_name]
        )

    assert set(receipt["bootstrap"]) == {"10", "21", "30"}
    assert M03R_V6_PRIMARY_BOOTSTRAP_BLOCK_LENGTH == 21
    assert M03R_V6_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS == (10, 30)
    for block in ("10", "21", "30"):
        result = receipt["bootstrap"][block]
        assert result["active_multifactor_alpha_daily_lcb"] == pytest.approx(0.0002)
        assert result["active_multifactor_alpha_annualized_lcb"] == pytest.approx(
            252.0 * result["active_multifactor_alpha_daily_lcb"]
        )
    assert receipt["promotion_authorized"] is False
    assert receipt["evaluation_scope"] == (
        "pure-numerical-public-surface-not-production-driver"
    )


def test_v6_fails_closed_without_typed_factor_or_inference_manifest() -> None:
    panel = _panel()
    policy, benchmark, risk_free, market, factors = panel
    dates, folds = _metadata()
    factor_manifest = _factor_manifest()
    inference_manifest = _inference_manifest(factor_manifest)
    common = {
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
        "setting_id": M03R_CANONICAL_SETTING_ID,
        "score_dates": dates,
        "fold_ids": folds,
        "policy_net_returns": policy,
        "benchmark_net_returns": benchmark,
        "risk_free_returns": risk_free,
        "market_total_returns": market,
        "factor_returns": factors,
        "common_evaluator_inputs_sha256": _digest("0"),
        "candidate_policy_returns_sha256": _digest("0"),
    }
    with pytest.raises(M03RV6EvaluationError, match="factor manifest is required"):
        evaluate_m03r_v6_inference(
            **common,
            factor_manifest=None,
            inference_manifest=inference_manifest,
        )
    with pytest.raises(M03RV6EvaluationError, match="inference manifest is required"):
        evaluate_m03r_v6_inference(
            **common,
            factor_manifest=factor_manifest,
            inference_manifest=None,
        )
    with pytest.raises(M03RV6EvaluationError, match="factor manifest is required"):
        build_m03r_v6_inference_manifest(
            factor_manifest=None,
            bootstrap_replicates=1_000,
            bootstrap_seed_sha256=_digest("2"),
        )


def test_v6_entry_rejects_v5_or_aliased_identity() -> None:
    panel = _panel()
    factor_manifest = _factor_manifest()
    inference_manifest = _inference_manifest(factor_manifest)
    policy, benchmark, risk_free, market, factors = panel
    dates, folds = _metadata()
    with pytest.raises(M03RV6EvaluationError, match="v5 remains immutable"):
        m03r_v6_common_evaluator_inputs_sha256(
            protocol_generation=M03R_V5_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            score_dates=dates,
            fold_ids=folds,
            benchmark_net_returns=benchmark,
            risk_free_returns=risk_free,
            market_total_returns=market,
            factor_returns=factors,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
        )
    common, _candidate = _input_hashes(
        panel,
        factor_manifest,
        inference_manifest,
    )
    with pytest.raises(M03RV6EvaluationError, match="unknown M03R v6 setting"):
        m03r_v6_candidate_policy_returns_sha256(
            protocol_generation=M03R_PROTOCOL_GENERATION,
            design_id=M03R_DESIGN_ID,
            setting_id="M03R-active-alpha-hold30",
            policy_net_returns=policy,
            common_evaluator_inputs_sha256=common,
        )


def test_v6_candidate_hash_rejects_mutated_policy_path() -> None:
    panel = _panel()
    factor_manifest = _factor_manifest()
    inference_manifest = _inference_manifest(factor_manifest)
    original_common, original_candidate = _input_hashes(
        panel,
        factor_manifest,
        inference_manifest,
    )
    mutated = tuple(value.copy() for value in panel)
    mutated[0][0, 0] += 1e-12
    mutated_common, mutated_candidate = _input_hashes(
        mutated,
        factor_manifest,
        inference_manifest,
    )
    assert mutated_common == original_common
    assert mutated_candidate != original_candidate
    with pytest.raises(M03RV6EvaluationError, match="does not match"):
        _evaluate(
            mutated,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
            common_evaluator_inputs_sha256=original_common,
            candidate_policy_returns_sha256=original_candidate,
        )


def test_v6_multiple_candidates_share_only_common_evaluator_identity() -> None:
    first = _panel()
    second = tuple(value.copy() for value in first)
    second[0][0, 0] += 1e-6
    factor_manifest = _factor_manifest()
    inference_manifest = _inference_manifest(factor_manifest)

    first_common, first_candidate = _input_hashes(
        first,
        factor_manifest,
        inference_manifest,
    )
    second_common, second_candidate = _input_hashes(
        second,
        factor_manifest,
        inference_manifest,
    )
    assert first_common == second_common
    assert first_candidate != second_candidate

    first_receipt, _, _ = _evaluate(
        first,
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    second_receipt, _, _ = _evaluate(
        second,
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    assert (
        first_receipt["common_evaluator_inputs_sha256"]
        == second_receipt["common_evaluator_inputs_sha256"]
        == first_common
    )
    assert (
        first_receipt["candidate_policy_returns_sha256"]
        != second_receipt["candidate_policy_returns_sha256"]
    )
    assert first_receipt["receipt_sha256"] != second_receipt["receipt_sha256"]


def test_v6_common_hash_rejects_mutated_benchmark_path() -> None:
    panel = _panel()
    factor_manifest = _factor_manifest()
    inference_manifest = _inference_manifest(factor_manifest)
    original_common, _original_candidate = _input_hashes(
        panel,
        factor_manifest,
        inference_manifest,
    )
    mutated = tuple(value.copy() for value in panel)
    mutated[1][0, 0] += 1e-12
    mutated_common, _mutated_candidate = _input_hashes(
        mutated,
        factor_manifest,
        inference_manifest,
    )
    assert mutated_common != original_common
    with pytest.raises(M03RV6EvaluationError, match="common.*does not match"):
        _evaluate(
            mutated,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
            common_evaluator_inputs_sha256=original_common,
        )


def test_v6_manifest_hashes_bind_exact_factor_and_inference_semantics() -> None:
    factor_manifest = _factor_manifest()
    with pytest.raises(M03RV6EvaluationError, match="factor manifest hash mismatch"):
        replace(factor_manifest, factor_names=("SIZE", "QUALITY"))

    inference_manifest = _inference_manifest(factor_manifest)
    assert inference_manifest.primary_bootstrap_block_length_trading_sessions == 21
    assert inference_manifest.sensitivity_bootstrap_block_lengths_trading_sessions == (
        10,
        30,
    )
    with pytest.raises(M03RV6EvaluationError, match="inference manifest hash mismatch"):
        replace(inference_manifest, bootstrap_replicates=1_001)


def test_v6_receipt_boundary_rejects_identity_or_manifest_drift() -> None:
    receipt, factor_manifest, inference_manifest = _evaluate()
    altered = deepcopy(receipt)
    altered["protocol_generation"] = M03R_V5_PROTOCOL_GENERATION
    with pytest.raises(M03RV6EvaluationError, match="v5 remains immutable"):
        validate_m03r_v6_evaluation_receipt(
            altered,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
        )

    alternative_factor = build_m03r_v6_factor_manifest(
        factor_names=("SIZE", "QUALITY"),
        factor_return_conventions=(
            "daily-simple-long-short-return",
            "daily-simple-long-short-return",
        ),
        point_in_time_source_manifest_sha256=_digest("1"),
    )
    with pytest.raises(
        M03RV6EvaluationError,
        match="inference and factor manifests are not bound",
    ):
        validate_m03r_v6_evaluation_receipt(
            receipt,
            factor_manifest=alternative_factor,
            inference_manifest=inference_manifest,
        )


def test_v6_receipt_hash_binds_active_alpha_lcb() -> None:
    receipt, factor_manifest, inference_manifest = _evaluate()
    altered = deepcopy(receipt)
    altered["bootstrap"]["21"]["active_multifactor_alpha_daily_lcb"] += 1e-12
    with pytest.raises(M03RV6EvaluationError, match="values drifted|hash mismatch"):
        validate_m03r_v6_evaluation_receipt(
            altered,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
        )

    altered = deepcopy(receipt)
    altered["candidate_policy_returns_sha256"] = _digest("f")
    with pytest.raises(M03RV6EvaluationError, match="hash mismatch"):
        validate_m03r_v6_evaluation_receipt(
            altered,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
        )

    altered = deepcopy(receipt)
    altered["active_multifactor_regression"]["alpha_daily"] += 1e-6
    altered["active_multifactor_regression"]["alpha_annualized_arithmetic"] += 252e-6
    with pytest.raises(M03RV6EvaluationError, match="portfolio minus benchmark"):
        validate_m03r_v6_evaluation_receipt(
            altered,
            factor_manifest=factor_manifest,
            inference_manifest=inference_manifest,
        )


def test_v6_evaluator_is_registered_for_software_qualification() -> None:
    assert (
        "src/rl_quant/evaluation/hold30_alpha_m03r_v6.py",
        "tests/test_hold30_alpha_m03r_v6_evaluation.py",
    ) in HOLD30_COMPONENT_TESTS
