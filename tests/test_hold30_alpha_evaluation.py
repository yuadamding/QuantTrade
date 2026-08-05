from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaDataBindingReceipt,
    Hold30AlphaDataError,
    Hold30AlphaEvaluationProvenance,
)
from rl_quant.evaluation.hold30_alpha_evaluation import (
    HOLD30_A06_OVERLAY_ID,
    HOLD30_ALPHA_CORE_ID,
    HOLD30_ALPHA_GENERATION,
    HOLD30_ALPHA_IDS,
    HOLD30_ALPHA_STREAM_BY_ID,
    Hold30AlphaAuxiliaryPaths,
    Hold30AlphaEvaluationError,
    Hold30AlphaEvaluationPlan,
    Hold30AlphaFoldPanel,
    build_hold30_alpha_artifact_inventory,
    build_hold30_alpha_mech8_summary,
    build_hold30_alpha_tranche,
    evaluate_hold30_alpha_auxiliary_paths,
    evaluate_hold30_alpha_stream,
    evaluate_hold30_matched_controls,
    manifest_binding_sha256s,
    publish_hold30_alpha_lockbox_marker,
    verify_hold30_alpha_artifact_inventory,
    verify_hold30_alpha_terminal_inventory,
    verify_hold30_c6_ownership,
)
from rl_quant.protocol.hold30_alpha_v3 import HOLD30_ALPHA_C1_BENCHMARK_ID


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update("torch.float64".encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _endpoint(
    setting: str,
    stream: str,
    fold: int,
    policy,
    c1,
    *,
    risk_free,
    market,
    factors,
    cross,
    dates,
    source_row_indices,
    policy_weights,
    c1_weights,
    provenance: Hold30AlphaEvaluationProvenance,
    binding: Hold30AlphaDataBindingReceipt,
) -> dict:
    receipt = {
        "schema": "rl-quant.hold30.alpha-endpoint-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "setting_id": setting,
        "stream_id": stream,
        "fold_index": fold,
        "risk_free_receipt_sha256": provenance.risk_free_artifact_sha256,
        "factor_receipt_sha256": provenance.factor_artifact_sha256,
        "cross_section_receipt_sha256": _digest(("xs", fold, stream)),
        "risk_free_returns_sha256": _array_digest(risk_free),
        "market_benchmark_id": provenance.market_benchmark_id,
        "market_artifact_sha256": provenance.market_artifact_sha256,
        "market_total_returns_sha256": _array_digest(market),
        "evaluation_provenance_id": provenance.receipt_id,
        "data_binding_receipt_id": binding.receipt_id,
        "evaluation_panel_id": binding.evaluation_panel_id,
        "source_axis_id": binding.source_axis_id,
        "factor_returns_sha256": _digest(
            {name: _array_digest(factors[name]) for name in sorted(factors)}
        ),
        "cross_section_inputs_sha256": _digest(cross),
        "dates_sha256": _digest(list(dates)),
        "source_row_indices_sha256": _digest(list(source_row_indices)),
        "policy_weights_sha256": _array_digest(policy_weights),
        "C1_weights_sha256": _array_digest(c1_weights),
        "tensor_receipts": {},
    }
    for cost in (10, 20, 40):
        active_log = np.log1p(policy[cost]) - np.log1p(c1[cost])
        receipt["tensor_receipts"][str(cost)] = {
            "policy_net_returns_sha256": _tensor_digest(policy[cost]),
            "C1_net_returns_sha256": _tensor_digest(c1[cost]),
            "active_log_returns_sha256": _tensor_digest(active_log),
        }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def _panels(
    setting: str,
    stream: str,
    *,
    overlay: float = 0.0,
    factor_conventions: tuple[str, str] = ("zero-investment", "zero-investment"),
):
    result = []
    assets = 20
    for fold in range(6):
        phase = fold * 0.11
        day = np.arange(63, dtype=np.float64)
        market = 0.0003 + 0.006 * np.sin(day * 0.31 + phase)
        alpha = 0.00025 + 0.0008 * np.cos(day * 0.17 + phase) + overlay
        c1 = {
            cost: (market - (cost - 20) * 1e-6).astype(np.float64)
            for cost in (10, 20, 40)
        }
        policy = {
            cost: (market + alpha - (cost - 20) * 2e-6).astype(np.float64)
            for cost in (10, 20, 40)
        }
        weights = np.full((63, assets), 1.0 / assets, dtype=np.float64)
        tilted = weights.copy()
        tilted[:, 1] += 0.02
        tilted[:, 2] -= 0.02
        scores = {}
        outcomes = {}
        valid = {}
        base_score = np.linspace(-1.0, 1.0, assets, dtype=np.float64)
        for horizon in (5, 21, 30, 63):
            score = np.tile(base_score, (63, 1)) + day[:, None] * 1e-4
            scores[horizon] = score
            outcomes[horizon] = 0.001 * score + 0.0001 * np.sin(day[:, None])
            valid[horizon] = np.ones((63, assets), dtype=np.bool_)
            valid[horizon][:, 0] = False
        risk_free = np.full(63, 0.0001, dtype=np.float64)
        factors = {
            "SIZE": (0.001 * np.cos(day * 0.21 + phase)).astype(np.float64),
            "VALUE": (0.001 * np.sin(day * 0.13 + phase)).astype(np.float64),
        }
        uncertainty = np.tile(np.linspace(0.0, 1.0, assets), (63, 1))
        age_pnl = np.tile(np.linspace(0.0001, 0.0, 61), (63, 1))
        cross = {
            "scores": {str(h): _array_digest(scores[h]) for h in (5, 21, 30, 63)},
            "future_excess_returns": {
                str(h): _array_digest(outcomes[h]) for h in (5, 21, 30, 63)
            },
            "future_valid": {str(h): _array_digest(valid[h]) for h in (5, 21, 30, 63)},
            "uncertainty": _array_digest(uncertainty),
            "alpha_pnl_by_age": _array_digest(age_pnl),
        }
        risk_free_artifact = _digest(("rf", fold, stream))
        factor_artifact = _digest(("factors", fold, stream))
        source_axis_id = _digest(("axis", fold, stream))
        evaluation_panel_id = _digest(("evaluation-panel", fold, stream))
        provenance = Hold30AlphaEvaluationProvenance(
            risk_free_id="PIT-CASH",
            market_benchmark_id="PIT-CAP-MARKET",
            factor_model_id="PIT-EVALUATOR-FACTORS",
            factor_names=("SIZE", "VALUE"),
            factor_return_conventions=factor_conventions,
            risk_free_artifact_sha256=risk_free_artifact,
            market_artifact_sha256=_digest(("market", fold, stream)),
            factor_artifact_sha256=factor_artifact,
            factor_plan_sha256=_digest("factor-plan"),
        )
        binding = Hold30AlphaDataBindingReceipt(
            protocol_generation=HOLD30_ALPHA_GENERATION,
            source_axis_id=source_axis_id,
            c1_benchmark_id=HOLD30_ALPHA_C1_BENCHMARK_ID,
            c1_trace_sha256=_digest(("c1", fold, stream)),
            cash_returns_sha256=_tensor_digest(risk_free),
            evaluation_panel_id=evaluation_panel_id,
            evaluation_provenance_id=provenance.receipt_id,
            global_path_ids=(0,),
        )
        dates = tuple(f"2025-{fold + 1:02d}-{index:02d}" for index in range(1, 64))
        source_row_indices = tuple(range(fold * 63, (fold + 1) * 63))
        endpoint = _endpoint(
            setting,
            stream,
            fold,
            policy,
            c1,
            risk_free=risk_free,
            market=market,
            factors=factors,
            cross=cross,
            dates=dates,
            source_row_indices=source_row_indices,
            policy_weights=tilted,
            c1_weights=weights,
            provenance=provenance,
            binding=binding,
        )
        result.append(
            Hold30AlphaFoldPanel(
                protocol_generation=HOLD30_ALPHA_GENERATION,
                setting_id=setting,
                stream_id=stream,
                fold_index=fold,
                dates=dates,
                source_row_indices=source_row_indices,
                policy_net_returns=policy,
                c1_net_returns=c1,
                pit_risk_free_returns=risk_free,
                pit_market_total_returns=market,
                factor_returns=factors,
                cash_index=0,
                policy_weights=tilted,
                c1_weights=weights,
                scores=scores,
                future_excess_returns=outcomes,
                future_valid=valid,
                uncertainty=uncertainty,
                alpha_pnl_by_age=age_pnl,
                endpoint_receipt=endpoint,
                evaluation_provenance=provenance,
                data_binding_receipt=binding,
                risk_free_receipt_sha256=endpoint["risk_free_receipt_sha256"],
                factor_receipt_sha256=endpoint["factor_receipt_sha256"],
                cross_section_receipt_sha256=endpoint["cross_section_receipt_sha256"],
            )
        )
    return tuple(result)


def _plan() -> Hold30AlphaEvaluationPlan:
    return Hold30AlphaEvaluationPlan(factor_names=("SIZE", "VALUE"))


def test_typed_stream_recomputes_full_metric_families_and_binds_endpoints() -> None:
    panels = _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")
    assert panels[0].evaluation_provenance.risk_free_usage == (
        "portfolio-accounting",
        "a06-a07-total-sharpe-objective",
        "checkpoint-ranking",
        "evaluation",
    )
    assert panels[0].evaluation_provenance.factor_usage == ("evaluation-only",)
    receipt = evaluate_hold30_alpha_stream(panels, plan=_plan())
    assert receipt["sessions"] == 378
    assert set(receipt["cost_ladder"]) == {"10", "20", "40"}
    primary = receipt["cost_ladder"]["20"]
    assert primary["policy_total"]["total_net_return"] > 0.0
    assert primary["active"]["information_ratio_annualized"] > 0.5
    assert set(receipt["regression"]["market_only"]["hac"]) == {"10", "21", "30", "42"}
    assert set(receipt["cross_sectional"]) == {"5", "21", "30", "63", "alpha_decay_by_age"}
    assert receipt["promotion_authorized"] is False

    panels[0].policy_net_returns[20][0] += 0.01
    with pytest.raises(Hold30AlphaEvaluationError, match="endpoint tensor"):
        evaluate_hold30_alpha_stream(panels, plan=_plan())


def test_active_statistics_use_log_returns_and_multifactor_always_includes_market() -> None:
    panels = _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")
    receipt = evaluate_hold30_alpha_stream(panels, plan=_plan())
    policy = np.concatenate([panel.policy_net_returns[20] for panel in panels])
    c1 = np.concatenate([panel.c1_net_returns[20] for panel in panels])
    active_log = np.log1p(policy) - np.log1p(c1)
    active = receipt["cost_ladder"]["20"]["active"]
    expected_te = np.std(active_log, ddof=1) * np.sqrt(252.0)
    expected_ir = np.sqrt(252.0) * np.mean(active_log) / np.std(active_log, ddof=1)
    assert active["sum_active_log_return"] == pytest.approx(np.sum(active_log))
    assert active["tracking_error_annualized"] == pytest.approx(expected_te)
    assert active["information_ratio_annualized"] == pytest.approx(expected_ir)
    assert active["tracking_error_annualized"] != pytest.approx(
        np.std(policy - c1, ddof=1) * np.sqrt(252.0), rel=1e-12, abs=1e-12
    )
    multifactor = receipt["regression"]["declared_multifactor"]
    assert tuple(multifactor["loadings"]) == (
        "PIT_CAP_MARKET_EXCESS",
        "SIZE",
        "VALUE",
    )


def test_endpoint_binds_dates_and_both_weight_paths() -> None:
    panels = _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")
    changed_dates = list(panels)
    changed_dates[0] = replace(changed_dates[0], dates=tuple(reversed(changed_dates[0].dates)))
    with pytest.raises(Hold30AlphaEvaluationError, match="dates_sha256"):
        evaluate_hold30_alpha_stream(changed_dates, plan=_plan())

    for field, match in (
        ("policy_weights", "policy_weights_sha256"),
        ("c1_weights", "C1_weights_sha256"),
    ):
        changed = list(_panels(HOLD30_ALPHA_CORE_ID, "alpha_core"))
        weights = np.array(getattr(changed[0], field), copy=True)
        weights[0, 1] += 0.001
        weights[0, 2] -= 0.001
        changed[0] = replace(changed[0], **{field: weights})
        with pytest.raises(Hold30AlphaEvaluationError, match=match):
            evaluate_hold30_alpha_stream(changed, plan=_plan())


def test_source_row_indices_are_required_strict_and_endpoint_bound() -> None:
    panel = _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")[0]
    with pytest.raises(Hold30AlphaEvaluationError, match="63 nonnegative integer"):
        replace(panel, source_row_indices=tuple(range(62)))
    with pytest.raises(Hold30AlphaEvaluationError, match="63 nonnegative integer"):
        replace(panel, source_row_indices=(-1, *range(1, 63)))
    with pytest.raises(Hold30AlphaEvaluationError, match="strictly increasing"):
        replace(panel, source_row_indices=(*range(62), 61))

    changed = list(_panels(HOLD30_ALPHA_CORE_ID, "alpha_core"))
    changed[0] = replace(changed[0], source_row_indices=tuple(range(1_000, 1_063)))
    with pytest.raises(Hold30AlphaEvaluationError, match="source_row_indices_sha256"):
        evaluate_hold30_alpha_stream(changed, plan=_plan())


def test_data_binding_cash_digest_is_recomputed_from_pit_risk_free_returns() -> None:
    panels = list(_panels(HOLD30_ALPHA_CORE_ID, "alpha_core"))
    panel = panels[0]
    assert panel.data_binding_receipt.cash_returns_sha256 == _tensor_digest(
        panel.pit_risk_free_returns
    )

    bad_binding = replace(
        panel.data_binding_receipt,
        cash_returns_sha256="f" * 64,
    )
    endpoint = dict(panel.endpoint_receipt)
    endpoint["data_binding_receipt_id"] = bad_binding.receipt_id
    endpoint.pop("receipt_sha256")
    endpoint["receipt_sha256"] = _digest(endpoint)
    panels[0] = replace(
        panel,
        data_binding_receipt=bad_binding,
        endpoint_receipt=endpoint,
    )
    with pytest.raises(Hold30AlphaEvaluationError, match="cash_returns_sha256"):
        evaluate_hold30_alpha_stream(panels, plan=_plan())


def test_uncertainty_must_be_nonnegative() -> None:
    panel = _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")[0]
    uncertainty = np.array(panel.uncertainty, copy=True)
    uncertainty[0, 1] = -1e-12
    with pytest.raises(Hold30AlphaEvaluationError, match="uncertainty must be nonnegative"):
        replace(panel, uncertainty=uncertainty)


def test_manifest_bound_moving_block_intervals_are_recomputed() -> None:
    plan = Hold30AlphaEvaluationPlan(
        factor_names=("SIZE", "VALUE"),
        bootstrap_seed_sha256="a" * 64,
        bootstrap_replicates=1_000,
        bootstrap_block_lengths=(10,),
        interval_alpha=0.05,
    )
    receipt = evaluate_hold30_alpha_stream(
        _panels(HOLD30_ALPHA_CORE_ID, "alpha_core"),
        plan=plan,
    )
    intervals = receipt["regression"]["moving_block_intervals"]
    assert set(intervals) == {"10"}
    assert len(intervals["10"]["market_alpha_daily_interval"]) == 2
    assert len(intervals["10"]["multifactor_alpha_daily_interval"]) == 2


def test_factor_conventions_are_transformed_and_must_match_across_folds() -> None:
    zero = evaluate_hold30_alpha_stream(
        _panels(HOLD30_ALPHA_CORE_ID, "alpha_core"), plan=_plan()
    )
    total_panels = _panels(
        HOLD30_ALPHA_CORE_ID,
        "alpha_core",
        factor_conventions=("total-return", "total-return"),
    )
    total = evaluate_hold30_alpha_stream(total_panels, plan=_plan())
    assert total["factor_return_conventions"] == {
        "SIZE": "total-return",
        "VALUE": "total-return",
    }
    assert (
        total["regression"]["declared_multifactor"]["alpha_daily"]
        != zero["regression"]["declared_multifactor"]["alpha_daily"]
    )

    mixed = list(_panels(HOLD30_ALPHA_CORE_ID, "alpha_core"))
    mixed[-1] = total_panels[-1]
    with pytest.raises(Hold30AlphaEvaluationError, match="conventions differ"):
        evaluate_hold30_alpha_stream(mixed, plan=_plan())

    with pytest.raises(Hold30AlphaDataError, match="supported return convention"):
        Hold30AlphaEvaluationProvenance(
            risk_free_id="PIT-CASH",
            market_benchmark_id="PIT-CAP-MARKET",
            factor_model_id="PIT-EVALUATOR-FACTORS",
            factor_names=("SIZE",),
            factor_return_conventions=("undeclared",),
            risk_free_artifact_sha256="a" * 64,
            market_artifact_sha256="b" * 64,
            factor_artifact_sha256="c" * 64,
            factor_plan_sha256="d" * 64,
        )


def test_alpha_core_and_a06_are_separate_and_tranche_stays_closed() -> None:
    streams = {
        setting: evaluate_hold30_alpha_stream(
            _panels(setting, HOLD30_ALPHA_STREAM_BY_ID[setting], overlay=index * 1e-6),
            plan=_plan(),
        )
        for index, setting in enumerate(HOLD30_ALPHA_IDS)
    }
    core = streams[HOLD30_ALPHA_CORE_ID]
    overlay = streams[HOLD30_A06_OVERLAY_ID]
    summary = build_hold30_alpha_mech8_summary(streams)
    assert summary["setting_order"] == list(HOLD30_ALPHA_IDS)
    assert set(summary["contrasts"]) == {
        "m01_minus_m00_persistence",
        "m02_minus_m01_active_objective",
        "m03_minus_m02_alpha_heads",
        "m03_minus_a04_uncertainty",
        "m03_minus_a05_te_floor",
        "a06_minus_m03_sharpe_overlay",
        "a07_minus_m03_direct_sharpe",
    }
    active = np.concatenate(
        [
            np.log1p(panel.policy_net_returns[20])
            - np.log1p(panel.c1_net_returns[20])
            for panel in _panels(HOLD30_ALPHA_CORE_ID, "alpha_core")
        ]
    )
    controls = np.stack([active - 1e-5 * (index + 1) for index in range(64)])
    matched = evaluate_hold30_matched_controls(
        alpha_core_active_log_returns=active,
        control_active_log_returns=controls,
        target_profile={
            "turnover": 0.03,
            "risky_exposure": 0.98,
            "median_sale_age": 30.0,
            "survival_30": 0.50,
        },
        control_profiles={
            "turnover": np.full(64, 0.03),
            "risky_exposure": np.full(64, 0.98),
            "median_sale_age": np.full(64, 30.0),
            "survival_30": np.full(64, 0.50),
        },
    )
    tranche = build_hold30_alpha_tranche(
        mech8_summary=summary,
        alpha_core=core,
        a06_overlay=overlay,
        matched_controls=matched,
        plan=_plan(),
    )
    assert tranche["promotion_authorized"] is False
    assert "factor_alpha_multiplicity_procedure_not_frozen" in tranche["promotion_blockers"]
    assert "confirmatory_factor_alpha_hypothesis_family_not_frozen" in tranche[
        "promotion_blockers"
    ]

    endpoint_sources = tuple(core["endpoint_receipt_sha256s"])
    auxiliary = Hold30AlphaAuxiliaryPaths(
        protocol_generation=HOLD30_ALPHA_GENERATION,
        seed_active_log_returns=np.full((5, 6, 63), 1e-5, dtype=np.float64),
        seed_run_receipt_sha256s=tuple(
            tuple(_digest(("seed", seed, fold)) for fold in range(6))
            for seed in (17, 29, 43, 71, 101)
        ),
        initialization_active_log_returns=np.full((6, 63), 1e-5, dtype=np.float64),
        initialization_source_endpoint_sha256s=endpoint_sources,
        c8_active_log_returns=np.zeros((64, 6, 63), dtype=np.float64),
        c8_cross_fold_mapping_receipt_sha256=_digest("c8-map"),
        c8_selection_receipt_sha256s=tuple(_digest(("c8", fold)) for fold in range(6)),
    )
    auxiliary_receipt = evaluate_hold30_alpha_auxiliary_paths(
        auxiliary,
        alpha_core_receipt=core,
    )
    assert auxiliary_receipt["positive_seed_count"] == 5
    assert auxiliary_receipt["candidate_exceeds_61st_C8"] is True
    assert auxiliary_receipt["promotion_authorized"] is False

    reused = dict(overlay)
    reused["endpoint_receipt_sha256s"] = core["endpoint_receipt_sha256s"]
    with pytest.raises(Hold30AlphaEvaluationError, match="reuse endpoint"):
        build_hold30_alpha_tranche(
            mech8_summary=summary,
            alpha_core=core,
            a06_overlay=reused,
            matched_controls=matched,
            plan=_plan(),
        )


def test_c6_ownership_is_exact_8_by_6_and_nonduplicated() -> None:
    rows = []
    for setting in HOLD30_ALPHA_IDS:
        for fold in range(6):
            rows.append(
                {
                    "setting_id": setting,
                    "fold_index": fold,
                    "replicates": 64,
                    "outer_score_rows": 63,
                    "other_rows_fixed": True,
                    "canonical_five_seed_intent_receipt_sha256": _digest(
                        ("intent", setting, fold)
                    ),
                    "permutation_receipt_sha256": _digest(("permutation", setting, fold)),
                }
            )
    receipt = {
        "schema": "rl-quant.hold30.alpha-c6-ownership-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "rows": rows,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    verify_hold30_c6_ownership(receipt)
    broken = json.loads(json.dumps(receipt))
    broken["rows"].pop()
    broken["receipt_sha256"] = _digest({key: value for key, value in broken.items() if key != "receipt_sha256"})
    with pytest.raises(Hold30AlphaEvaluationError, match="8 x 6"):
        verify_hold30_c6_ownership(broken)


def test_terminal_inventory_is_exact_240_without_selective_retry() -> None:
    receipts = []
    for setting in HOLD30_ALPHA_IDS:
        for fold in range(6):
            for seed in (17, 29, 43, 71, 101):
                receipt = {
                    "schema": "rl-quant.hold30.alpha-terminal-trial-v1",
                    "protocol_generation": HOLD30_ALPHA_GENERATION,
                    "setting_id": setting,
                    "fold_index": fold,
                    "seed": seed,
                    "terminal_status": "success",
                    "selective_retry": False,
                    "run_receipt_sha256": _digest(("run", setting, fold, seed)),
                    "artifact_graph_sha256": _digest(("graph", setting, fold, seed)),
                }
                receipt["receipt_sha256"] = _digest(receipt)
                receipts.append(receipt)
    verify_hold30_alpha_terminal_inventory(receipts)
    with pytest.raises(Hold30AlphaEvaluationError, match="8 x 6 x 5"):
        verify_hold30_alpha_terminal_inventory(receipts[:-1])


def test_inventory_hashes_pretty_json_and_closes_manifest_bindings(tmp_path: Path) -> None:
    payload = {"z": [1, 2], "a": {"value": True}}
    pretty = json.dumps(payload, indent=2).encode() + b"\n"
    (tmp_path / "payload.json").write_bytes(pretty)
    binding = b"retained source archive"
    (tmp_path / "source.tar").write_bytes(binding)
    entries = {
        "payload": ("payload.json", hashlib.sha256(pretty).hexdigest(), _digest(payload)),
        "source": ("source.tar", hashlib.sha256(binding).hexdigest()),
    }
    inventory = build_hold30_alpha_artifact_inventory(entries)
    live = verify_hold30_alpha_artifact_inventory(
        inventory,
        root=tmp_path,
        expected_json_payloads={"payload": payload},
        required_manifest_sha256s=(hashlib.sha256(binding).hexdigest(),),
    )
    assert set(live) == {"payload", "source"}
    manifest = {
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "bindings": {
            "source_archive_sha256": hashlib.sha256(binding).hexdigest(),
            "git_commit": "a" * 40,
        },
    }
    assert manifest_binding_sha256s(manifest) == (hashlib.sha256(binding).hexdigest(),)


def test_atomic_lockbox_marker_is_idempotent_only_for_identical_reveal(tmp_path: Path) -> None:
    marker = tmp_path / "consumed.json"
    evaluation = {
        "schema": "rl-quant.hold30.alpha-final-evaluation-v1",
        "protocol_generation": HOLD30_ALPHA_GENERATION,
        "manifest_sha256": "a" * 64,
        "lockbox_id_sha256": "b" * 64,
        "all_required_live_artifacts_verified": True,
        "evaluation_complete": True,
        "scientific_qualification": False,
        "promotion_authorized": False,
        "launch_authorized": False,
    }
    evaluation["receipt_sha256"] = _digest(evaluation)
    first = publish_hold30_alpha_lockbox_marker(
        marker,
        verified_evaluation_receipt=evaluation,
    )
    second = publish_hold30_alpha_lockbox_marker(
        marker,
        verified_evaluation_receipt=evaluation,
    )
    assert first == second
    with pytest.raises(Hold30AlphaEvaluationError, match="different consumption"):
        changed = dict(evaluation)
        changed["lockbox_id_sha256"] = "d" * 64
        changed["receipt_sha256"] = _digest(
            {key: value for key, value in changed.items() if key != "receipt_sha256"}
        )
        publish_hold30_alpha_lockbox_marker(marker, verified_evaluation_receipt=changed)


def test_v2_generation_is_rejected() -> None:
    with pytest.raises(Hold30AlphaEvaluationError, match="v2"):
        Hold30AlphaEvaluationPlan(
            factor_names=("SIZE",),
            protocol_generation="prelockbox-hold30-mech8-v2",
        )
    with pytest.raises(Hold30AlphaEvaluationError, match="cannot shadow"):
        Hold30AlphaEvaluationPlan(factor_names=("PIT_CAP_MARKET_EXCESS",))
