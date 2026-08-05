"""Frozen joint inference for ``prelockbox-hold30-mech8-v2``.

The evaluator consumes aligned, already sealed 20 bp active log-return traces.
It never receives portfolio actions or raw outcomes and therefore cannot alter
the trading experiment.  Moving blocks are sampled independently inside each
of the six outer folds while the same sampled indices are reused for every
setting, control, and planned contrast.

The byte encoding, null centering, studentization, percentile convention, tie
handling, and named statistical families are part of the emitted inference
plan.  A caller must supply a manifest-bound SHA-256 seed; there is no implicit
or wall-clock RNG seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from rl_quant.protocol.hold30 import HOLD30_MECH8_IDS, HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import HOLD30_FOLDS, sha256_payload


HOLD30_INFERENCE_SCHEMA = "rl-quant.hold30.inference-v1"
HOLD30_INFERENCE_REPLICATES = 10_000
HOLD30_INFERENCE_BLOCK_LENGTHS = (5, 10, 30)
HOLD30_OUTER_SCORE_DECISIONS = 63
HOLD30_INFERENCE_PRIMARY_BLOCK = 10
HOLD30_INFERENCE_ALPHA = 0.05
HOLD30_FAMILY_ALPHA = 0.10

HOLD30_WRC_SPA_FAMILY = (*HOLD30_MECH8_IDS, "C2", "C3", "C4", "C5")
HOLD30_MAX_T_FAMILY = (
    "hold30-m01-slow-gate",
    "hold30-m02-age-hazard",
    "hold30-a04-no-age-input",
    "hold30-a05-no-early-penalty",
    "hold30-a06-no-turn-penalty",
    "hold30-a07-no-exp-timing",
)
HOLD30_PLANNED_CONTRASTS = (
    ("h1_minus_h0", "hold30-m01-slow-gate", "hold30-m00-legacy-gate"),
    ("h2_minus_h1", "hold30-m02-age-hazard", "hold30-m01-slow-gate"),
    ("h3_minus_h1", "hold30-m03-sleeve30", "hold30-m01-slow-gate"),
    ("h2_minus_h3", "hold30-m02-age-hazard", "hold30-m03-sleeve30"),
    ("h2_minus_a04", "hold30-m02-age-hazard", "hold30-a04-no-age-input"),
    (
        "h2_minus_a05",
        "hold30-m02-age-hazard",
        "hold30-a05-no-early-penalty",
    ),
    (
        "h2_minus_a06",
        "hold30-m02-age-hazard",
        "hold30-a06-no-turn-penalty",
    ),
    (
        "h2_minus_a07",
        "hold30-m02-age-hazard",
        "hold30-a07-no-exp-timing",
    ),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE = "hold30-m02-age-hazard"
_RNG_DOMAIN = b"rl-quant.hold30.joint-moving-block-v1\x00"


class Hold30InferenceError(ValueError):
    """Sealed traces or inference evidence violate the frozen v2 contract."""


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise Hold30InferenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True, slots=True)
class Hold30InferencePlan:
    """Normative statistical choices that must be bound before outer access."""

    bootstrap_seed_sha256: str
    replicates: int = HOLD30_INFERENCE_REPLICATES
    block_lengths: tuple[int, ...] = HOLD30_INFERENCE_BLOCK_LENGTHS
    primary_block_length: int = HOLD30_INFERENCE_PRIMARY_BLOCK
    one_sided_alpha: float = HOLD30_INFERENCE_ALPHA
    family_alpha: float = HOLD30_FAMILY_ALPHA
    protocol_generation: str = HOLD30_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
        _require_digest("bootstrap_seed_sha256", self.bootstrap_seed_sha256)
        if self.protocol_generation != HOLD30_PROTOCOL_GENERATION:
            raise Hold30InferenceError("inference protocol generation mismatch")
        if self.replicates != HOLD30_INFERENCE_REPLICATES:
            raise Hold30InferenceError("production inference requires exactly 10,000 replicates")
        if tuple(self.block_lengths) != HOLD30_INFERENCE_BLOCK_LENGTHS:
            raise Hold30InferenceError("inference block lengths must be exactly (5, 10, 30)")
        if self.primary_block_length != HOLD30_INFERENCE_PRIMARY_BLOCK:
            raise Hold30InferenceError("primary inference block length must be 10")
        if self.one_sided_alpha != HOLD30_INFERENCE_ALPHA:
            raise Hold30InferenceError("one-sided inference alpha must be 0.05")
        if self.family_alpha != HOLD30_FAMILY_ALPHA:
            raise Hold30InferenceError("family-test alpha must be 0.10")

    def receipt(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rng": {
                "algorithm": "SHA-256 counter modulo",
                "domain_hex": _RNG_DOMAIN.hex(),
                "integer_encoding": "unsigned-big-endian",
                "counter_fields": [
                    "bootstrap_seed_bytes[32]",
                    "block_length:u16",
                    "replicate:u32",
                    "fold:u16",
                    "block:u16",
                ],
            },
            "resampling": {
                "method": "noncircular-moving-block-bootstrap",
                "block_start_domain": "0..fold_length-block_length inclusive",
                "truncate_last_block": True,
                "blocks_never_cross_folds": True,
                "indices_shared_jointly_across_all_series": True,
                "fold_pooling": "decision-weighted",
            },
            "confidence_intervals": {
                "candidate_lower": "uncentered-bootstrap-nearest-rank-5pct",
                "planned_contrasts": "uncentered-bootstrap-nearest-rank-2.5pct/97.5pct",
            },
            "nulls": {
                "white": "all-column empirical-mean centered, unstudentized max",
                "spa": "Hansen-consistent recentering, bootstrap-SE studentized max",
                "max_t": "all-column empirical-mean centered, bootstrap-SE studentized max",
                "contrast": "empirical-mean centered, bootstrap-SE studentized one-sided",
            },
            "ties": "add-one p-value counts bootstrap statistic >= observed",
            "holm": {
                "alternative": "left-minus-right greater than zero",
                "tie_break": "lexical contrast id",
                "alpha": HOLD30_INFERENCE_ALPHA,
            },
            "families": {
                "white_spa": list(HOLD30_WRC_SPA_FAMILY),
                "max_t": list(HOLD30_MAX_T_FAMILY),
                "planned_contrasts": [list(value) for value in HOLD30_PLANNED_CONTRASTS],
            },
        }


def _coerce_traces(
    active_log_returns: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[tuple[str, ...], np.ndarray]:
    if not isinstance(active_log_returns, Mapping):
        raise Hold30InferenceError("active_log_returns must be a named mapping")
    expected = set(HOLD30_WRC_SPA_FAMILY)
    supplied = set(active_log_returns)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        raise Hold30InferenceError(
            f"inference trace family mismatch; missing={missing}, unknown={unknown}"
        )
    names = tuple(HOLD30_WRC_SPA_FAMILY)
    rows: list[np.ndarray] = []
    for name in names:
        folds = active_log_returns[name]
        if len(folds) != HOLD30_FOLDS:
            raise Hold30InferenceError(f"{name} must contain exactly six folds")
        fold_rows: list[np.ndarray] = []
        for fold_index, fold in enumerate(folds):
            values = np.asarray(tuple(fold), dtype=np.float64)
            if values.shape != (HOLD30_OUTER_SCORE_DECISIONS,):
                raise Hold30InferenceError(
                    f"{name} fold {fold_index} must have exactly 63 scored decisions"
                )
            if not bool(np.isfinite(values).all()):
                raise Hold30InferenceError(f"{name} fold {fold_index} is non-finite")
            fold_rows.append(values)
        rows.append(np.stack(fold_rows, axis=0))
    return names, np.stack(rows, axis=0)


def _validate_source_receipts(
    source_receipt_sha256: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(source_receipt_sha256, Mapping) or set(source_receipt_sha256) != set(
        HOLD30_WRC_SPA_FAMILY
    ):
        raise Hold30InferenceError("source receipt mapping must exactly match the WRC/SPA family")
    return {
        name: _require_digest(f"source receipt {name}", source_receipt_sha256[name])
        for name in HOLD30_WRC_SPA_FAMILY
    }


def _counter_start(
    seed: bytes,
    *,
    block_length: int,
    replicate: int,
    fold: int,
    block: int,
    choices: int,
) -> int:
    material = b"".join(
        (
            _RNG_DOMAIN,
            seed,
            block_length.to_bytes(2, "big", signed=False),
            replicate.to_bytes(4, "big", signed=False),
            fold.to_bytes(2, "big", signed=False),
            block.to_bytes(2, "big", signed=False),
        )
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % choices


def _moving_block_indices(
    plan: Hold30InferencePlan,
    *,
    block_length: int,
    fold_length: int,
) -> np.ndarray:
    if block_length > fold_length:
        raise Hold30InferenceError("block length exceeds a scored fold")
    blocks = math.ceil(fold_length / block_length)
    choices = fold_length - block_length + 1
    seed = bytes.fromhex(plan.bootstrap_seed_sha256)
    output = np.empty(
        (plan.replicates, HOLD30_FOLDS, fold_length),
        dtype=np.int16,
    )
    offsets = np.arange(block_length, dtype=np.int16)
    for replicate in range(plan.replicates):
        for fold in range(HOLD30_FOLDS):
            cursor = 0
            for block in range(blocks):
                start = _counter_start(
                    seed,
                    block_length=block_length,
                    replicate=replicate,
                    fold=fold,
                    block=block,
                    choices=choices,
                )
                take = min(block_length, fold_length - cursor)
                output[replicate, fold, cursor : cursor + take] = start + offsets[:take]
                cursor += take
    return output


def _bootstrap_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # values: [series, fold, time]; indices: [replicate, fold, time].
    replicates = indices.shape[0]
    result = np.zeros((replicates, values.shape[0]), dtype=np.float64)
    for fold in range(HOLD30_FOLDS):
        sampled = values[:, fold, :][:, indices[:, fold, :]]
        # advanced indexing yields [series, replicate, time]
        result += sampled.sum(axis=2).T
    result /= float(HOLD30_FOLDS * values.shape[2])
    return result


def _nearest_rank(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0 or not 0.0 < probability < 1.0:
        raise Hold30InferenceError("invalid nearest-rank request")
    index = max(0, min(ordered.size - 1, math.ceil(probability * ordered.size) - 1))
    return float(ordered[index])


def _add_one_pvalue(null_statistics: np.ndarray, observed: float) -> float:
    exceed = int(np.count_nonzero(null_statistics >= observed))
    return float((1 + exceed) / (null_statistics.size + 1))


def _standard_errors(bootstrap: np.ndarray) -> np.ndarray:
    result = bootstrap.std(axis=0, ddof=1)
    if not bool(np.isfinite(result).all()) or bool((result <= 0.0).any()):
        raise Hold30InferenceError(
            "studentized inference requires positive finite bootstrap standard errors"
        )
    return result


def _holm(raw: Mapping[str, float]) -> dict[str, dict[str, float | bool | int]]:
    ordered = sorted(raw, key=lambda name: (float(raw[name]), name))
    count = len(ordered)
    prior = 0.0
    result: dict[str, dict[str, float | bool | int]] = {}
    for rank, name in enumerate(ordered, start=1):
        adjusted = min(1.0, max(prior, (count - rank + 1) * float(raw[name])))
        prior = adjusted
        result[name] = {
            "raw_pvalue": float(raw[name]),
            "adjusted_pvalue": adjusted,
            "rank": rank,
            "reject_at_0_05": adjusted <= HOLD30_INFERENCE_ALPHA,
        }
    return {name: result[name] for name, *_ in HOLD30_PLANNED_CONTRASTS}


def _block_result(
    names: tuple[str, ...],
    values: np.ndarray,
    bootstrap: np.ndarray,
    *,
    block_length: int,
) -> dict[str, Any]:
    name_to_index = {name: index for index, name in enumerate(names)}
    means = values.mean(axis=(1, 2))
    total_observations = HOLD30_FOLDS * HOLD30_OUTER_SCORE_DECISIONS

    candidate_index = name_to_index[_CANDIDATE]
    candidate_bootstrap = bootstrap[:, candidate_index]
    candidate = {
        "mean_active_log_return": float(means[candidate_index]),
        "one_sided_95pct_lower": _nearest_rank(
            candidate_bootstrap,
            HOLD30_INFERENCE_ALPHA,
        ),
    }

    contrast_series: list[np.ndarray] = []
    contrast_points: dict[str, float] = {}
    contrast_bootstrap: dict[str, np.ndarray] = {}
    intervals: dict[str, dict[str, float]] = {}
    for contrast_id, left, right in HOLD30_PLANNED_CONTRASTS:
        series = values[name_to_index[left]] - values[name_to_index[right]]
        draws = bootstrap[:, name_to_index[left]] - bootstrap[:, name_to_index[right]]
        contrast_series.append(series)
        contrast_points[contrast_id] = float(series.mean())
        contrast_bootstrap[contrast_id] = draws
        intervals[contrast_id] = {
            "point_mean": contrast_points[contrast_id],
            "lower_95pct": _nearest_rank(draws, 0.025),
            "upper_95pct": _nearest_rank(draws, 0.975),
        }

    contrast_matrix = np.stack(contrast_series, axis=0)
    contrast_draw_matrix = np.stack(
        [contrast_bootstrap[name] for name, *_ in HOLD30_PLANNED_CONTRASTS],
        axis=1,
    )
    contrast_se = _standard_errors(contrast_draw_matrix)
    contrast_observed_t = contrast_matrix.mean(axis=(1, 2)) / contrast_se
    contrast_null = (
        contrast_draw_matrix - contrast_matrix.mean(axis=(1, 2))[None, :]
    ) / contrast_se[None, :]
    contrast_raw_p = {
        contrast_id: _add_one_pvalue(contrast_null[:, index], contrast_observed_t[index])
        for index, (contrast_id, *_rest) in enumerate(HOLD30_PLANNED_CONTRASTS)
    }

    wrc_indices = [name_to_index[name] for name in HOLD30_WRC_SPA_FAMILY]
    wrc_means = means[wrc_indices]
    wrc_bootstrap = bootstrap[:, wrc_indices]
    white_observed = max(0.0, float(wrc_means.max()))
    white_null = np.maximum(0.0, (wrc_bootstrap - wrc_means[None, :]).max(axis=1))

    spa_se = _standard_errors(wrc_bootstrap)
    spa_observed_columns = wrc_means / spa_se
    spa_observed = max(0.0, float(spa_observed_columns.max()))
    threshold = -math.sqrt(2.0 * math.log(math.log(total_observations)))
    spa_center = np.where(spa_observed_columns >= threshold, wrc_means, 0.0)
    spa_null = np.maximum(
        0.0,
        ((wrc_bootstrap - spa_center[None, :]) / spa_se[None, :]).max(axis=1),
    )

    max_t_indices = [name_to_index[name] for name in HOLD30_MAX_T_FAMILY]
    max_t_means = means[max_t_indices]
    max_t_bootstrap = bootstrap[:, max_t_indices]
    max_t_se = _standard_errors(max_t_bootstrap)
    max_t_observed = max_t_means / max_t_se
    max_t_null = ((max_t_bootstrap - max_t_means[None, :]) / max_t_se[None, :]).max(
        axis=1
    )
    max_t = {
        name: {
            "observed_t": float(max_t_observed[index]),
            "adjusted_one_sided_pvalue": _add_one_pvalue(
                max_t_null,
                float(max_t_observed[index]),
            ),
        }
        for index, name in enumerate(HOLD30_MAX_T_FAMILY)
    }

    return {
        "block_length": block_length,
        "observations": total_observations,
        "candidate": candidate,
        "planned_contrast_intervals": intervals,
        "planned_contrast_holm": _holm(contrast_raw_p),
        "white_reality_check": {
            "observed_max_mean": white_observed,
            "one_sided_pvalue": _add_one_pvalue(white_null, white_observed),
        },
        "hansen_spa": {
            "observed_max_t": spa_observed,
            "consistent_recenter_threshold": threshold,
            "one_sided_pvalue": _add_one_pvalue(spa_null, spa_observed),
        },
        "max_t": max_t,
        "statistical_gate_diagnostics": {
            "candidate_lower_positive": candidate["one_sided_95pct_lower"] > 0.0,
            "white_p_at_most_0_10": _add_one_pvalue(white_null, white_observed)
            <= HOLD30_FAMILY_ALPHA,
            "spa_p_at_most_0_10": _add_one_pvalue(spa_null, spa_observed)
            <= HOLD30_FAMILY_ALPHA,
            "candidate_max_t_p_at_most_0_10": max_t[_CANDIDATE][
                "adjusted_one_sided_pvalue"
            ]
            <= HOLD30_FAMILY_ALPHA,
        },
    }


def compute_hold30_inference(
    active_log_returns: Mapping[str, Sequence[Sequence[float]]],
    *,
    source_receipt_sha256: Mapping[str, str],
    plan: Hold30InferencePlan,
) -> dict[str, Any]:
    """Compute the exact v2 joint inference artifact.

    Inputs must be the canonical continuing 20 bp active log-return series
    versus C1, with six 63-decision folds for every named setting/control.
    """

    if not isinstance(plan, Hold30InferencePlan):
        raise TypeError("plan must be Hold30InferencePlan")
    names, values = _coerce_traces(active_log_returns)
    sources = _validate_source_receipts(source_receipt_sha256)
    trace_payload = {
        name: [[float(value) for value in fold] for fold in active_log_returns[name]]
        for name in names
    }
    trace_sha = sha256_payload(trace_payload)
    results: dict[str, Any] = {}
    for block_length in plan.block_lengths:
        indices = _moving_block_indices(
            plan,
            block_length=block_length,
            fold_length=HOLD30_OUTER_SCORE_DECISIONS,
        )
        bootstrap = _bootstrap_means(values, indices)
        results[str(block_length)] = _block_result(
            names,
            values,
            bootstrap,
            block_length=block_length,
        )

    all_lengths = tuple(results[str(length)] for length in plan.block_lengths)
    statistical_sensitivity_pass = all(
        all(row["statistical_gate_diagnostics"].values()) for row in all_lengths
    )
    payload: dict[str, Any] = {
        "schema_version": HOLD30_INFERENCE_SCHEMA,
        "receipt_type": "prelockbox-hold30-joint-inference",
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "economic_input": {
            "cost_basis_points": 20,
            "continuing_wealth": True,
            "estimand": "daily_active_log_return_versus_C1",
            "folds": HOLD30_FOLDS,
            "decisions_per_fold": HOLD30_OUTER_SCORE_DECISIONS,
        },
        "inference_plan": plan.receipt(),
        "inference_plan_sha256": sha256_payload(plan.receipt()),
        "source_receipt_sha256": sources,
        "active_trace_sha256": trace_sha,
        "results_by_block_length": results,
        "statistical_sensitivity_pass": statistical_sensitivity_pass,
        "primary_block_length": HOLD30_INFERENCE_PRIMARY_BLOCK,
        "scientific_qualification": False,
        "promotion_authorized": False,
    }
    payload["receipt_sha256"] = _receipt_hash(payload)
    return payload


def verify_hold30_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    active_log_returns: Mapping[str, Sequence[Sequence[float]]] | None = None,
    source_receipt_sha256: Mapping[str, str] | None = None,
) -> None:
    """Verify shape/self-hash and optionally bind the live sealed traces."""

    if not isinstance(receipt, Mapping):
        raise Hold30InferenceError("inference receipt must be a mapping")
    required = {
        "schema_version",
        "receipt_type",
        "protocol_generation",
        "economic_input",
        "inference_plan",
        "inference_plan_sha256",
        "source_receipt_sha256",
        "active_trace_sha256",
        "results_by_block_length",
        "statistical_sensitivity_pass",
        "primary_block_length",
        "scientific_qualification",
        "promotion_authorized",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise Hold30InferenceError("inference receipt has partial or unknown fields")
    if (
        receipt["schema_version"] != HOLD30_INFERENCE_SCHEMA
        or receipt["receipt_type"] != "prelockbox-hold30-joint-inference"
        or receipt["protocol_generation"] != HOLD30_PROTOCOL_GENERATION
        or receipt["primary_block_length"] != HOLD30_INFERENCE_PRIMARY_BLOCK
        or receipt["scientific_qualification"] is not False
        or receipt["promotion_authorized"] is not False
    ):
        raise Hold30InferenceError("inference receipt identity/authority fields are invalid")
    if receipt["economic_input"] != {
        "cost_basis_points": 20,
        "continuing_wealth": True,
        "estimand": "daily_active_log_return_versus_C1",
        "folds": HOLD30_FOLDS,
        "decisions_per_fold": HOLD30_OUTER_SCORE_DECISIONS,
    }:
        raise Hold30InferenceError("inference economic input contract is invalid")
    inference_plan = receipt["inference_plan"]
    if not isinstance(inference_plan, Mapping):
        raise Hold30InferenceError("inference plan must be a mapping")
    try:
        expected_plan = Hold30InferencePlan(
            bootstrap_seed_sha256=inference_plan["bootstrap_seed_sha256"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Hold30InferenceError("inference plan is not the frozen v2 plan") from exc
    if inference_plan != expected_plan.receipt():
        raise Hold30InferenceError("inference plan differs from the frozen v2 plan")
    _require_digest("inference_plan_sha256", receipt["inference_plan_sha256"])
    if sha256_payload(receipt["inference_plan"]) != receipt["inference_plan_sha256"]:
        raise Hold30InferenceError("inference plan digest mismatch")
    _validate_source_receipts(receipt["source_receipt_sha256"])
    _require_digest("active_trace_sha256", receipt["active_trace_sha256"])
    if set(receipt["results_by_block_length"]) != {"5", "10", "30"}:
        raise Hold30InferenceError("inference result block lengths are incomplete")
    claimed = _require_digest("receipt_sha256", receipt["receipt_sha256"])
    if _receipt_hash(receipt) != claimed:
        raise Hold30InferenceError("inference receipt self-hash mismatch")
    if active_log_returns is not None or source_receipt_sha256 is not None:
        if active_log_returns is None or source_receipt_sha256 is None:
            raise Hold30InferenceError("live verification requires traces and source receipts")
        _names, _values = _coerce_traces(active_log_returns)
        del _names, _values
        sources = _validate_source_receipts(source_receipt_sha256)
        if sources != receipt["source_receipt_sha256"]:
            raise Hold30InferenceError("live source receipts differ from inference receipt")
        trace_payload = {
            name: [[float(value) for value in fold] for fold in active_log_returns[name]]
            for name in HOLD30_WRC_SPA_FAMILY
        }
        if sha256_payload(trace_payload) != receipt["active_trace_sha256"]:
            raise Hold30InferenceError("live active traces differ from inference receipt")
        recomputed = compute_hold30_inference(
            active_log_returns,
            source_receipt_sha256=source_receipt_sha256,
            plan=expected_plan,
        )
        if _canonical_json(recomputed) != _canonical_json(receipt):
            raise Hold30InferenceError("live inference result differs from receipt")


__all__ = [
    "HOLD30_INFERENCE_BLOCK_LENGTHS",
    "HOLD30_INFERENCE_REPLICATES",
    "HOLD30_INFERENCE_SCHEMA",
    "HOLD30_MAX_T_FAMILY",
    "HOLD30_PLANNED_CONTRASTS",
    "HOLD30_WRC_SPA_FAMILY",
    "Hold30InferenceError",
    "Hold30InferencePlan",
    "compute_hold30_inference",
    "verify_hold30_inference_receipt",
]
