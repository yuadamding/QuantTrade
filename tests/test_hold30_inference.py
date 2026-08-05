from __future__ import annotations

import copy
import math

import pytest

from rl_quant.evaluation.hold30_inference import (
    HOLD30_INFERENCE_BLOCK_LENGTHS,
    HOLD30_INFERENCE_REPLICATES,
    HOLD30_MAX_T_FAMILY,
    HOLD30_PLANNED_CONTRASTS,
    HOLD30_WRC_SPA_FAMILY,
    Hold30InferenceError,
    Hold30InferencePlan,
    compute_hold30_inference,
    verify_hold30_inference_receipt,
)
from rl_quant.protocol.hold30 import HOLD30_MECH8_IDS
from rl_quant.protocol.hold30_freeze import sha256_payload


def _digest(value: object) -> str:
    return sha256_payload(value)


def _traces() -> dict[str, tuple[tuple[float, ...], ...]]:
    offsets = {
        name: 0.0001 + 0.00005 * index
        for index, name in enumerate(HOLD30_WRC_SPA_FAMILY)
    }
    offsets["hold30-m00-legacy-gate"] = 0.0001
    offsets["hold30-m01-slow-gate"] = 0.0005
    offsets["hold30-m02-age-hazard"] = 0.0025
    offsets["hold30-m03-sleeve30"] = 0.0003
    result: dict[str, tuple[tuple[float, ...], ...]] = {}
    for model_index, name in enumerate(HOLD30_WRC_SPA_FAMILY):
        result[name] = tuple(
            tuple(
                offsets[name]
                + 0.0002 * math.sin((time + 1) * 0.31 + fold)
                + 0.00003 * math.cos((model_index + 1) * (time + 1) * 0.17)
                for time in range(63)
            )
            for fold in range(6)
        )
    return result


def _sources() -> dict[str, str]:
    return {name: _digest(("trace-receipt", name)) for name in HOLD30_WRC_SPA_FAMILY}


@pytest.fixture(scope="module")
def computed_receipt():
    traces = _traces()
    sources = _sources()
    receipt = compute_hold30_inference(
        traces,
        source_receipt_sha256=sources,
        plan=Hold30InferencePlan(_digest("frozen-bootstrap-seed")),
    )
    return traces, sources, receipt


def test_v2_families_and_plan_are_exact() -> None:
    assert HOLD30_INFERENCE_REPLICATES == 10_000
    assert HOLD30_INFERENCE_BLOCK_LENGTHS == (5, 10, 30)
    assert HOLD30_WRC_SPA_FAMILY == (*HOLD30_MECH8_IDS, "C2", "C3", "C4", "C5")
    assert HOLD30_MAX_T_FAMILY == (
        "hold30-m01-slow-gate",
        "hold30-m02-age-hazard",
        "hold30-a04-no-age-input",
        "hold30-a05-no-early-penalty",
        "hold30-a06-no-turn-penalty",
        "hold30-a07-no-exp-timing",
    )
    assert len(HOLD30_PLANNED_CONTRASTS) == 8

    plan = Hold30InferencePlan(_digest("seed"))
    evidence = plan.receipt()
    assert evidence["resampling"]["blocks_never_cross_folds"] is True
    assert evidence["resampling"]["indices_shared_jointly_across_all_series"] is True
    assert evidence["rng"]["integer_encoding"] == "unsigned-big-endian"


def test_joint_inference_is_receipt_bound_and_positive_fixture_passes_statistics(
    computed_receipt,
) -> None:
    traces, sources, receipt = computed_receipt
    verify_hold30_inference_receipt(
        receipt,
        active_log_returns=traces,
        source_receipt_sha256=sources,
    )
    assert receipt["statistical_sensitivity_pass"] is True
    for block in ("5", "10", "30"):
        result = receipt["results_by_block_length"][block]
        assert result["candidate"]["one_sided_95pct_lower"] > 0.0
        assert result["white_reality_check"]["one_sided_pvalue"] <= 0.10
        assert result["hansen_spa"]["one_sided_pvalue"] <= 0.10
        assert (
            result["max_t"]["hold30-m02-age-hazard"]["adjusted_one_sided_pvalue"]
            <= 0.10
        )

    tampered = copy.deepcopy(receipt)
    tampered["results_by_block_length"]["10"]["candidate"][
        "one_sided_95pct_lower"
    ] = -1.0
    with pytest.raises(Hold30InferenceError, match="self-hash"):
        verify_hold30_inference_receipt(tampered)


def test_live_trace_mutation_cannot_reuse_inference_receipt(computed_receipt) -> None:
    traces, sources, receipt = computed_receipt
    changed = dict(traces)
    folds = [list(fold) for fold in changed["hold30-m02-age-hazard"]]
    folds[0][0] += 0.001
    changed["hold30-m02-age-hazard"] = tuple(tuple(fold) for fold in folds)

    with pytest.raises(Hold30InferenceError, match="live active traces"):
        verify_hold30_inference_receipt(
            receipt,
            active_log_returns=changed,
            source_receipt_sha256=sources,
        )


def test_missing_or_malformed_family_fails_closed() -> None:
    traces = _traces()
    traces.pop("C5")
    with pytest.raises(Hold30InferenceError, match="family mismatch"):
        compute_hold30_inference(
            traces,
            source_receipt_sha256=_sources(),
            plan=Hold30InferencePlan(_digest("seed")),
        )

    with pytest.raises(Hold30InferenceError, match="10,000"):
        Hold30InferencePlan(_digest("seed"), replicates=100)
