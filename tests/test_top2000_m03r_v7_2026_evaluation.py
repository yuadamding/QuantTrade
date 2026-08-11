from __future__ import annotations

import copy
import hashlib
import io
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation import top2000_m03r_v7_2026 as evaluation_module
from rl_quant.evaluation import top2000_m03r_v7_2026_cohort_survival as cohort_module
from rl_quant.evaluation import top2000_m03r_v7_2026_factor_data as factor_data_module
from rl_quant.evaluation.top2000_m03r_v7_2026 import (
    TOP2000_M03R_V7_2026_DECISION_COUNT,
    TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS,
    TOP2000_M03R_V7_2026_SCORE_DATE_AXIS,
    Top2000M03RV72026EvaluationError,
    Top2000M03RV72026Telemetry,
    build_top2000_m03r_v7_2026_inference_plan,
    evaluate_top2000_m03r_v7_2026_panel,
    load_top2000_m03r_v7_2026_receipt,
    validate_top2000_m03r_v7_2026_receipt,
    write_top2000_m03r_v7_2026_receipt,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
    Top2000M03RV72026CohortTrajectories,
    Top2000M03RV72026CohortTrajectoryReceipt,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
    TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
    TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
    Top2000M03RV72026FactorData,
    build_top2000_m03r_v7_2026_factor_data,
    retrieve_top2000_m03r_v7_2026_official_factor_archives,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)


def _business_dates() -> tuple[str, ...]:
    assert len(TOP2000_M03R_V7_2026_SCORE_DATE_AXIS) == (
        TOP2000_M03R_V7_2026_DECISION_COUNT
    )
    return TOP2000_M03R_V7_2026_SCORE_DATE_AXIS


def _turnover(
    shape: tuple[int, ...],
    *,
    discretionary: float,
) -> dict[str, np.ndarray]:
    result = {
        cause.value: np.zeros(shape, dtype=np.float64) for cause in TURNOVER_CAUSES
    }
    result[TurnoverCause.DISCRETIONARY.value].fill(discretionary)
    result[TurnoverCause.STARTUP.value][(..., 0)] = 0.01
    return result


def _telemetry(settings: int, rows: int) -> Top2000M03RV72026Telemetry:
    at_risk = np.ones((settings, rows, 61), dtype=np.float64)
    discretionary = np.zeros_like(at_risk)
    discretionary[..., 2] = 0.01
    forced = {
        cause: np.zeros_like(at_risk)
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED.value,
            TurnoverCause.AVAILABILITY_FORCED.value,
            TurnoverCause.RISK_FORCED.value,
            TurnoverCause.TERMINAL.value,
        )
    }
    forced[TurnoverCause.RISK_FORCED.value][..., 4] = 0.005
    actions = {
        "HOLD": np.full((settings, rows), 10.0),
        "CONTINUOUS": np.full((settings, rows), 3.0),
        "EXIT": np.ones((settings, rows)),
    }
    hazard = np.broadcast_to(
        np.asarray([0.0, 0.25, 1.0], dtype=np.float64),
        (settings, rows, 3),
    ).copy()
    return Top2000M03RV72026Telemetry(
        requested_to_executed_projection_distance=np.full(
            (settings, rows), 0.002
        ),
        age_notional_at_risk=at_risk,
        discretionary_exit_notional_by_age=discretionary,
        forced_exit_notional_by_cause_and_age=forced,
        action_counts_by_type=actions,
        continuous_hazard=hazard,
        continuous_hazard_observed=np.ones_like(hazard, dtype=np.bool_),
    )


def _cohort_panel(
    checkpoint_hashes: dict[str, str],
) -> tuple[Top2000M03RV72026CohortTrajectories, ...]:
    dates = _business_dates()
    rows = len(dates)
    result: list[Top2000M03RV72026CohortTrajectories] = []
    for setting_id in M03R_SEED17_TOP2000_SETTING_IDS:
        entry = np.ones(rows, dtype=np.float64)
        events = np.zeros((rows, 61), dtype=np.float64)
        events[:, 30] = 1.0
        terminal = np.zeros_like(events)
        forced = {
            cause.value: np.zeros_like(events)
            for cause in (
                TurnoverCause.MEMBERSHIP_FORCED,
                TurnoverCause.AVAILABILITY_FORCED,
                TurnoverCause.RISK_FORCED,
                TurnoverCause.TERMINAL,
            )
        }
        forced_hashes = tuple(
            (
                cause,
                cohort_module._array_sha256(
                    f"forced_censor/{cause}", value
                ),
            )
            for cause, value in forced.items()
        )
        trajectory_receipt = Top2000M03RV72026CohortTrajectoryReceipt(
            setting_id=setting_id,
            checkpoint_sha256=checkpoint_hashes[setting_id],
            checkpoint_fold_index=5,
            chronology_receipt_sha256="c" * 64,
            economic_execution_receipt_sha256=cohort_module._sha256(
                {"setting_id": setting_id, "artifact": "economic-execution"}
            ),
            score_dates_sha256=cohort_module._sha256(list(dates)),
            entry_units_sha256=cohort_module._array_sha256(
                "entry_units", entry
            ),
            discretionary_event_units_by_age_sha256=(
                cohort_module._array_sha256(
                    "discretionary_event_units_by_age", events
                )
            ),
            forced_censor_units_by_cause_and_age_sha256=forced_hashes,
            terminal_censor_units_by_age_sha256=cohort_module._array_sha256(
                "terminal_censor_units_by_age", terminal
            ),
            origin_rows=rows,
        )
        result.append(
            Top2000M03RV72026CohortTrajectories(
                origin_dates=dates,
                entry_units=entry,
                discretionary_event_units_by_age=events,
                forced_censor_units_by_cause_and_age=forced,
                terminal_censor_units_by_age=terminal,
                receipt=trajectory_receipt,
            )
        )
    return tuple(result)


def _factor_data(
    root: Path,
    *,
    dates: tuple[str, ...],
    risk_free: np.ndarray,
    market_excess: np.ndarray,
    factors: np.ndarray,
) -> Top2000M03RV72026FactorData:
    five_factor_rows = [",Mkt-RF,SMB,HML,RMW,CMA,RF"]
    momentum_rows = [",Mom"]
    for index, score_date in enumerate(dates):
        source_date = score_date.replace("-", "")
        five_factor_values = (
            market_excess[index],
            *factors[index, :4],
            risk_free[index],
        )
        five_factor_rows.append(
            ",".join(
                (
                    source_date,
                    *(f"{100.0 * float(value):.15g}" for value in five_factor_values),
                )
            )
        )
        momentum_rows.append(
            f"{source_date},{100.0 * float(factors[index, 4]):.15g}"
        )
    five_factor_zip = io.BytesIO()
    momentum_zip = io.BytesIO()
    with zipfile.ZipFile(five_factor_zip, "w") as archive:
        archive.writestr(
            TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
            "\n".join(five_factor_rows),
        )
    with zipfile.ZipFile(momentum_zip, "w") as archive:
        archive.writestr(
            TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
            "\n".join(momentum_rows),
        )
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    payloads = {
        contract.five_factor_download_url: five_factor_zip.getvalue(),
        contract.momentum_download_url: momentum_zip.getvalue(),
    }

    class _Response:
        def __init__(self, raw: bytes, url: str) -> None:
            self._raw = io.BytesIO(raw)
            self._url = url
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return self._url

        def read(self, size: int) -> bytes:
            return self._raw.read(size)

    def _urlopen(request, *, timeout: int):
        assert timeout == 30
        return _Response(payloads[request.full_url], request.full_url)

    with (
        patch.object(factor_data_module, "urlopen", _urlopen),
        patch.object(
            factor_data_module,
            "_utc_now",
            lambda: "2026-06-25T12:00:00Z",
        ),
    ):
        evidence, _receipt_file_sha256 = (
            retrieve_top2000_m03r_v7_2026_official_factor_archives(
                output_directory=root / "official-factor-archives",
                output_receipt_path=root / "official-factor-retrieval.json",
                frozen_plan_file_sha256="a" * 64,
                frozen_plan_receipt_sha256="b" * 64,
            )
        )
    return build_top2000_m03r_v7_2026_factor_data(
        retrieval_evidence=evidence,
        score_dates=dates,
    )


def _inputs(
    *,
    with_factors: bool,
    factor_root: Path | None = None,
    with_telemetry: bool = True,
) -> dict[str, object]:
    dates = _business_dates()
    rows = len(dates)
    settings = len(M03R_SEED17_TOP2000_SETTING_IDS)
    rng = np.random.default_rng(20260810)
    risk_free = np.full(rows, 0.0001)
    market_excess = rng.normal(0.0002, 0.006, rows)
    factors = rng.normal(0.0, 0.003, (rows, 5))
    benchmark = (
        risk_free
        + 0.00002
        + 0.9 * market_excess
        + factors @ np.asarray([0.1, -0.05, 0.03, 0.02, -0.04])
    )
    policy_gross = np.stack(
        [
            risk_free
            + (0.00030 - index * 0.000015)
            + (0.10 + 0.005 * index) * market_excess
            + factors @ np.asarray([0.03, 0.02, -0.01, 0.01, 0.015])
            for index in range(settings)
        ]
    )
    policy_turnover = _turnover((settings, rows), discretionary=0.01)
    benchmark_turnover = _turnover((rows,), discretionary=0.002)
    total_policy_turnover = sum(policy_turnover.values())
    total_benchmark_turnover = sum(benchmark_turnover.values())
    values: dict[str, object] = {
        "score_dates": dates,
        "setting_ids": M03R_SEED17_TOP2000_SETTING_IDS,
        "portfolio_net_returns_20bp": (
            policy_gross - 0.002 * total_policy_turnover
        ),
        "benchmark_net_returns_20bp": (
            benchmark - 0.002 * total_benchmark_turnover
        ),
        "portfolio_turnover_by_cause": policy_turnover,
        "benchmark_turnover_by_cause": benchmark_turnover,
        "checkpoint_sha256_by_setting": {
            setting_id: f"{index + 1:064x}"
            for index, setting_id in enumerate(M03R_SEED17_TOP2000_SETTING_IDS)
        },
        "checkpoint_fold_index": 5,
        "checkpoint_role": "headline",
        "training_completion_receipt_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
        "chronology_receipt_sha256": "c" * 64,
        "execution_receipt_sha256": "d" * 64,
        "inference_plan": build_top2000_m03r_v7_2026_inference_plan(),
        "telemetry": _telemetry(settings, rows) if with_telemetry else None,
    }
    if with_factors:
        if factor_root is None:
            raise AssertionError("factor_root is required for factor-backed tests")
        values["factor_data"] = _factor_data(
            factor_root,
            dates=dates,
            risk_free=risk_free,
            market_excess=market_excess,
            factors=factors,
        )
    return values


def _rehash_receipt(value: dict[str, object]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = evaluation_module._sha256(unsigned)


def _assert_rehashed_receipt_rejected(
    value: dict[str, object],
    *,
    match: str,
) -> None:
    _rehash_receipt(value)
    with pytest.raises(Top2000M03RV72026EvaluationError, match=match):
        validate_top2000_m03r_v7_2026_receipt(
            value,
            expected_receipt_sha256=value["receipt_sha256"],
        )


def test_frozen_2026_score_axis_excludes_exchange_holidays() -> None:
    assert TOP2000_M03R_V7_2026_DECISION_COUNT == 118
    assert TOP2000_M03R_V7_2026_SCORE_DATE_AXIS[0] == "2026-01-02"
    assert TOP2000_M03R_V7_2026_SCORE_DATE_AXIS[-1] == "2026-06-23"
    assert {
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
        "2026-05-25",
        "2026-06-19",
    }.isdisjoint(TOP2000_M03R_V7_2026_SCORE_DATE_AXIS)


def test_full_retrospective_reprices_one_path_and_reports_bound_evidence(
    tmp_path: Path,
) -> None:
    inputs = _inputs(with_factors=True, factor_root=tmp_path)
    checkpoint_hashes = inputs["checkpoint_sha256_by_setting"]
    assert isinstance(checkpoint_hashes, dict)
    inputs["cohort_trajectories"] = _cohort_panel(checkpoint_hashes)
    receipt = evaluate_top2000_m03r_v7_2026_panel(**inputs)

    validate_top2000_m03r_v7_2026_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert receipt["development_only"] is True
    assert receipt["scientific_reporting_authorized"] is False
    assert receipt["promotion_authorized"] is False
    assert receipt["point_in_time_evidence"]["universe"]["status"] == "unavailable"
    assert receipt["point_in_time_evidence"]["official_factors"]["status"] == "available"
    factor_data = inputs["factor_data"]
    assert isinstance(factor_data, Top2000M03RV72026FactorData)
    assert receipt["point_in_time_evidence"]["official_factors"][
        "factor_data_receipt_sha256"
    ] == factor_data.receipt_sha256
    assert receipt["chronology"] == {
        "single_continuous_path": True,
        "fold_resets": False,
        "start_date": _business_dates()[0],
        "end_date": _business_dates()[-1],
        "decision_count": len(_business_dates()),
    }
    assert receipt["checkpoint_fold_index"] == 5
    assert receipt["checkpoint_role"] == "headline"
    assert receipt["cost_ladder"]["decision_cost_basis_points"] == 20
    assert receipt["cost_ladder"]["policy_reexecuted_for_sensitivity_rungs"] is False
    assert receipt["dispersion_estimator"]["standard_deviation"] == "sample-ddof-1"
    assert receipt["cohort_survival_evidence"]["status"] == "available"

    canonical = receipt["settings"][0]
    metrics_10 = canonical["cost_ladder"]["10"]
    metrics_20 = canonical["cost_ladder"]["20"]
    metrics_40 = canonical["cost_ladder"]["40"]
    primary_returns = np.asarray(inputs["portfolio_net_returns_20bp"])[0]
    assert metrics_20["annualized_portfolio_arithmetic_mean_return"] == (
        pytest.approx(252.0 * primary_returns.mean())
    )
    total_turnover = sum(
        np.asarray(value)[0]
        for value in inputs["portfolio_turnover_by_cause"].values()
    )
    assert (
        metrics_10["annualized_portfolio_arithmetic_mean_return"]
        - metrics_20["annualized_portfolio_arithmetic_mean_return"]
    ) == pytest.approx(252.0 * 0.001 * total_turnover.mean())
    assert (
        metrics_10["annualized_portfolio_arithmetic_mean_return"]
        > metrics_40["annualized_portfolio_arithmetic_mean_return"]
    )
    factor = canonical["factor_attribution"]
    assert factor["status"] == "available"
    assert factor["active_multifactor_regression"]["alpha_daily"] == pytest.approx(
        factor["portfolio_multifactor_regression"]["alpha_daily"]
        - factor["benchmark_multifactor_regression"]["alpha_daily"],
        abs=1.0e-12,
    )
    assert canonical["active_beta"]["status"] == "available"
    assert canonical["bootstrap"]["21"]["sharpe_difference_lcb"] is not None
    assert (
        canonical["bootstrap"]["21"][
            "active_multifactor_alpha_annualized_lcb"
        ]
        is not None
    )
    assert canonical["telemetry"]["status"] == "available"
    assert canonical["telemetry"]["actions"]["frequencies"]["HOLD"] == (
        pytest.approx(10.0 / 14.0)
    )
    assert canonical["telemetry"]["continuous_hazard"]["near_zero_fraction"] == (
        pytest.approx(1.0 / 3.0)
    )
    assert canonical["telemetry"]["holding_snapshot_descriptive"][
        "snapshot_notional_survival"
    ]["10"] < 1.0
    assert canonical["telemetry"]["holding_snapshot_descriptive"][
        "eligible_for_required_cohort_rmst"
    ] is False
    assert canonical["cohort_rmst60"]["status"] == "available"
    assert canonical["cohort_rmst60"]["rmst60_sessions"] == pytest.approx(30.0)
    assert canonical["reversal_episode_performance"] == {
        "status": "unavailable",
        "reason": "frozen-typed-pre-outcome-reversal-episode-artifact-unavailable-v1",
    }
    assert receipt["reversal_episode_evidence"] == {
        "status": "unavailable",
        "reason": "frozen-typed-pre-outcome-reversal-episode-artifact-unavailable-v1",
        "receipt_sha256": None,
    }
    assert set(canonical["turnover_by_cause"]) == {
        cause.value for cause in TURNOVER_CAUSES
    }

    contrasts = receipt["paired_contrasts"]
    assert contrasts["joint_primary_block_draws"] is True
    assert len(contrasts["rows"]) == len(TOP2000_M03R_V7_2026_PRIMARY_CONTRASTS)
    assert contrasts["multiplicity_method"] == (
        "joint-max-absolute-centered-contrast-fwer-0.05"
    )
    assert contrasts["raw_one_sided_p_value_method"] == (
        "null-centered-paired-bootstrap-upper-tail"
    )
    assert contrasts["joint_max_absolute_centered_critical_value"] >= 0.0
    assert all(
        0.0 <= row["multiplicity_adjusted_p_value"] <= 1.0
        for row in contrasts["rows"]
    )

    wrong_manifest = copy.deepcopy(receipt)
    manifest = wrong_manifest["point_in_time_evidence"]["official_factors"][
        "manifest"
    ]
    manifest["frequency"] = "weekly"
    manifest["manifest_sha256"] = evaluation_module._sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _assert_rehashed_receipt_rejected(
        wrong_manifest,
        match="factor semantics",
    )

    malformed_regression = copy.deepcopy(receipt)
    del malformed_regression["settings"][0]["factor_attribution"][
        "active_multifactor_regression"
    ]["loadings"]["Mom"]
    _assert_rehashed_receipt_rejected(
        malformed_regression,
        match="regression identity",
    )

    contradictory_beta = copy.deepcopy(receipt)
    beta = contradictory_beta["settings"][0]["active_beta"]
    beta["constraint_satisfied"] = not beta["constraint_satisfied"]
    _assert_rehashed_receipt_rejected(
        contradictory_beta,
        match="available-state invariants",
    )


def test_missing_factor_and_telemetry_are_unavailable_not_fabricated() -> None:
    inputs = _inputs(with_factors=False, with_telemetry=False)
    receipt = evaluate_top2000_m03r_v7_2026_panel(**inputs)

    assert receipt["point_in_time_evidence"]["official_factors"] == {
        "status": "unavailable",
        "reason": "official-daily-ff5-momentum-receipt-not-supplied",
        "factor_data_receipt_sha256": None,
        "factor_arrays_sha256": None,
        "manifest": None,
    }
    assert receipt["cohort_survival_evidence"] == {
        "status": "unavailable",
        "reason": "complete-score-origin-cohort-trajectory-panel-not-supplied",
        "receipt": None,
    }
    for row in receipt["settings"]:
        assert row["factor_attribution"]["status"] == "unavailable"
        assert row["factor_attribution"]["active_multifactor_regression"] is None
        assert row["active_beta"]["status"] == "unavailable"
        assert row["cost_ladder"]["20"]["portfolio_sharpe"] is None
        assert row["bootstrap"]["21"]["sharpe_difference_lcb"] is None
        assert row["bootstrap"]["21"]["active_multifactor_alpha_annualized_lcb"] is None
        assert row["telemetry"]["status"] == "unavailable"
        assert row["cohort_rmst60"]["status"] == "unavailable"

    partial = dict(inputs)
    partial["factor_data"] = object()
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="exact typed official-factor contract",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**partial)


def test_factor_data_array_mutation_breaks_its_bound_receipt(tmp_path: Path) -> None:
    inputs = _inputs(
        with_factors=True,
        factor_root=tmp_path,
        with_telemetry=False,
    )
    factor_data = inputs["factor_data"]
    assert isinstance(factor_data, Top2000M03RV72026FactorData)
    changed_rf = list(factor_data.risk_free_returns)
    changed_rf[0] += 0.0001
    object.__setattr__(factor_data, "risk_free_returns", tuple(changed_rf))

    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="factor_data failed receipt and array validation",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


def test_factor_dates_must_match_the_bound_artifact(tmp_path: Path) -> None:
    inputs = _inputs(
        with_factors=True,
        factor_root=tmp_path,
        with_telemetry=False,
    )
    factor_data = inputs["factor_data"]
    assert isinstance(factor_data, Top2000M03RV72026FactorData)
    shifted = list(_business_dates())
    shifted[1] = "2026-01-03"
    object.__setattr__(factor_data, "score_dates", tuple(shifted))

    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="factor_data failed receipt and array validation",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("reversal_episode_mask", np.ones(len(_business_dates()), dtype=np.bool_)),
        ("reversal_episode_receipt_sha256", "5" * 64),
    ],
)
def test_v1_rejects_caller_authored_reversal_episode_inputs(
    name: str,
    value: object,
) -> None:
    inputs = _inputs(with_factors=False, with_telemetry=False)
    inputs[name] = value

    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="frozen typed pre-outcome artifact",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


def test_inference_and_checkpoint_fold_cannot_weaken_frozen_family() -> None:
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="10,000-draw joint",
    ):
        build_top2000_m03r_v7_2026_inference_plan(
            bootstrap_replicates=1_000,
        )

    inputs = _inputs(with_factors=False, with_telemetry=False)
    inputs["checkpoint_fold_index"] = 4
    inputs["checkpoint_role"] = "headline"
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="fold 5 as the sole headline",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


@pytest.mark.parametrize(
    "bad_dates, message",
    [
        (
            (*_business_dates()[:-1], _business_dates()[-2]),
            "strictly increasing",
        ),
        (
            ("2025-12-31", *_business_dates()[1:]),
            "canonical 2026 date",
        ),
    ],
)
def test_chronology_rejects_resets_and_non_2026_dates(
    bad_dates: tuple[str, ...],
    message: str,
) -> None:
    inputs = _inputs(with_factors=False, with_telemetry=False)
    inputs["score_dates"] = bad_dates
    with pytest.raises(Top2000M03RV72026EvaluationError, match=message):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


def test_candidate_hash_is_setting_specific_and_receipt_tamper_fails() -> None:
    inputs = _inputs(with_factors=False, with_telemetry=False)
    first = evaluate_top2000_m03r_v7_2026_panel(**inputs)
    changed = dict(inputs)
    policies = np.asarray(inputs["portfolio_net_returns_20bp"]).copy()
    policies[3, 0] += 0.0001
    changed["portfolio_net_returns_20bp"] = policies
    second = evaluate_top2000_m03r_v7_2026_panel(**changed)

    assert first["common_inputs_sha256"] == second["common_inputs_sha256"]
    for index, (left, right) in enumerate(
        zip(first["settings"], second["settings"], strict=True)
    ):
        if index == 3:
            assert left["setting_inputs_sha256"] != right["setting_inputs_sha256"]
        else:
            assert left["setting_inputs_sha256"] == right["setting_inputs_sha256"]

    expected_receipt_sha256 = first["receipt_sha256"]
    first["promotion_authorized"] = True
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="authorization",
    ):
        validate_top2000_m03r_v7_2026_receipt(
            first,
            expected_receipt_sha256=expected_receipt_sha256,
        )


def test_receipt_file_and_rehashed_nested_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = evaluate_top2000_m03r_v7_2026_panel(
        **_inputs(with_factors=False, with_telemetry=False)
    )
    path = tmp_path / "evaluation.json"
    file_sha256 = write_top2000_m03r_v7_2026_receipt(path, receipt)
    loaded = load_top2000_m03r_v7_2026_receipt(
        path,
        expected_file_sha256=file_sha256,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert loaded == receipt
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_top2000_m03r_v7_2026_receipt(path, receipt)

    symlink_path = tmp_path / "evaluation-link.json"
    symlink_path.symlink_to(path)
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="file identity or size",
    ):
        load_top2000_m03r_v7_2026_receipt(
            symlink_path,
            expected_file_sha256=file_sha256,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )

    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_bytes = evaluation_module._canonical_json(receipt) + b"\n"
    noncanonical_path.write_bytes(noncanonical_bytes)
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="canonical JSON",
    ):
        load_top2000_m03r_v7_2026_receipt(
            noncanonical_path,
            expected_file_sha256=hashlib.sha256(noncanonical_bytes).hexdigest(),
            expected_receipt_sha256=receipt["receipt_sha256"],
        )

    tampered = copy.deepcopy(receipt)
    tampered["settings"][0]["cost_ladder"]["20"][
        "annualized_active_log_return"
    ] += 0.01
    unsigned = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = evaluation_module._sha256(unsigned)
    tampered_path = tmp_path / "tampered.json"
    tampered_bytes = evaluation_module._canonical_json(tampered)
    tampered_path.write_bytes(tampered_bytes)
    tampered_file_sha256 = hashlib.sha256(tampered_bytes).hexdigest()
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="file identity|hash mismatch",
    ):
        load_top2000_m03r_v7_2026_receipt(
            tampered_path,
            expected_file_sha256=file_sha256,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="hash mismatch",
    ):
        load_top2000_m03r_v7_2026_receipt(
            tampered_path,
            expected_file_sha256=tampered_file_sha256,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="hash mismatch",
    ):
        validate_top2000_m03r_v7_2026_receipt(
            tampered,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )

    missing_metric = copy.deepcopy(receipt)
    del missing_metric["settings"][0]["cost_ladder"]["20"][
        "active_maximum_drawdown"
    ]
    missing_unsigned = {
        key: value
        for key, value in missing_metric.items()
        if key != "receipt_sha256"
    }
    missing_metric["receipt_sha256"] = evaluation_module._sha256(
        missing_unsigned
    )
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="metric inventory",
    ):
        validate_top2000_m03r_v7_2026_receipt(
            missing_metric,
            expected_receipt_sha256=missing_metric["receipt_sha256"],
        )

    wrong_window = copy.deepcopy(receipt)
    wrong_window["chronology"]["start_date"] = "2026-01-05"
    wrong_unsigned = {
        key: value for key, value in wrong_window.items() if key != "receipt_sha256"
    }
    wrong_window["receipt_sha256"] = evaluation_module._sha256(wrong_unsigned)
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="chronology semantics",
    ):
        validate_top2000_m03r_v7_2026_receipt(
            wrong_window,
            expected_receipt_sha256=wrong_window["receipt_sha256"],
        )

    partial_path = tmp_path / "partial-preserved.json"
    real_write = os.write
    write_calls = 0

    def _fail_after_one_byte(descriptor: int, payload: object) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, memoryview(payload)[:1])
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(evaluation_module.os, "write", _fail_after_one_byte)
    with pytest.raises(OSError, match="synthetic interrupted write"):
        write_top2000_m03r_v7_2026_receipt(partial_path, receipt)
    assert partial_path.is_file()
    assert partial_path.stat().st_size == 1


def test_rehashed_nested_and_cross_field_inconsistencies_fail_closed(
    tmp_path: Path,
) -> None:
    receipt = evaluate_top2000_m03r_v7_2026_panel(
        **_inputs(with_factors=False, with_telemetry=True)
    )

    malformed_factor = copy.deepcopy(receipt)
    malformed_factor["settings"][0]["factor_attribution"][
        "active_multifactor_regression"
    ] = {"malformed": True}
    _assert_rehashed_receipt_rejected(
        malformed_factor,
        match="unavailable-state invariants",
    )
    malformed_path = tmp_path / "malformed-factor.json"
    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="unavailable-state invariants",
    ):
        write_top2000_m03r_v7_2026_receipt(
            malformed_path,
            malformed_factor,
        )
    assert not malformed_path.exists()

    empty_holding = copy.deepcopy(receipt)
    empty_holding["settings"][0]["telemetry"][
        "holding_snapshot_descriptive"
    ] = {}
    _assert_rehashed_receipt_rejected(
        empty_holding,
        match="holding_snapshot_descriptive fields",
    )

    wrong_count = copy.deepcopy(receipt)
    wrong_count["chronology"]["decision_count"] = 31
    _assert_rehashed_receipt_rejected(
        wrong_count,
        match="chronology semantics",
    )

    wrong_turnover_cost = copy.deepcopy(receipt)
    wrong_turnover_cost["settings"][0]["turnover_by_cause"]["discretionary"][
        "cost_return_units"
    ]["20"] = 999.0
    _assert_rehashed_receipt_rejected(
        wrong_turnover_cost,
        match="cost values",
    )

    wrong_turnover_mean = copy.deepcopy(receipt)
    wrong_turnover_mean["settings"][0]["turnover_by_cause"]["discretionary"][
        "mean_daily_one_way_turnover"
    ] = 999.0
    _assert_rehashed_receipt_rejected(
        wrong_turnover_mean,
        match="mean turnover",
    )

    contradictory_contrast = copy.deepcopy(receipt)
    contradictory_contrast["paired_contrasts"]["rows"][0][
        "simultaneous_fwer95_ucb"
    ] = -999.0
    _assert_rehashed_receipt_rejected(
        contradictory_contrast,
        match="bounds or rejection",
    )

    contradictory_rejection = copy.deepcopy(receipt)
    contrast = contradictory_rejection["paired_contrasts"]["rows"][0]
    contrast["multiplicity_reject_at_family_alpha_0_05"] = not contrast[
        "multiplicity_reject_at_family_alpha_0_05"
    ]
    _assert_rehashed_receipt_rejected(
        contradictory_rejection,
        match="bounds or rejection",
    )

    contradictory_beta = copy.deepcopy(receipt)
    contradictory_beta["settings"][0]["active_beta"].update(
        {
            "status": "available",
            "reason": "nonsense",
            "active_market_beta": None,
            "bootstrap_standard_error": None,
            "equivalence_absolute_upper_bound": None,
            "constraint_satisfied": True,
        }
    )
    _assert_rehashed_receipt_rejected(
        contradictory_beta,
        match="finite number",
    )


def test_evaluator_revalidates_mutable_cohort_arrays() -> None:
    inputs = _inputs(with_factors=False, with_telemetry=False)
    checkpoint_hashes = inputs["checkpoint_sha256_by_setting"]
    assert isinstance(checkpoint_hashes, dict)
    panel = _cohort_panel(checkpoint_hashes)
    panel[0].discretionary_event_units_by_age[0, 30] += 0.01
    inputs["cohort_trajectories"] = panel

    with pytest.raises(
        Top2000M03RV72026EvaluationError,
        match="complete-origin validation",
    ):
        evaluate_top2000_m03r_v7_2026_panel(**inputs)


def test_sample_dispersion_and_null_centered_p_value_are_hand_computable() -> None:
    active = np.asarray([0.01, 0.03], dtype=np.float64)
    benchmark = np.zeros_like(active)
    sample_standard_deviation = np.sqrt(0.0002)
    expected_ratio = 0.02 / sample_standard_deviation * np.sqrt(252.0)

    metrics = evaluation_module._return_metrics(active, benchmark)
    assert metrics["annualized_tracking_error"] == pytest.approx(
        np.sqrt(252.0) * sample_standard_deviation
    )
    assert metrics["annualized_information_ratio"] == pytest.approx(
        expected_ratio
    )
    assert evaluation_module._annualized_sharpe(active) == pytest.approx(
        expected_ratio
    )

    skewed_draws = np.asarray([-100.0, 2.5, 2.5, 2.5])
    assert evaluation_module._null_centered_one_sided_bootstrap_p_value(
        2.0,
        skewed_draws,
    ) == pytest.approx(0.2)
    assert evaluation_module._null_centered_one_sided_bootstrap_p_value(
        -0.1,
        skewed_draws,
    ) == 1.0
