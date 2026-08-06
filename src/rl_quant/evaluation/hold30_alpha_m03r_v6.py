"""Generation-qualified pure numerical evaluation surface for M03R v6.

The module evaluates already-produced chronological return arrays.  It is not a
production driver and cannot authorize promotion.  V6 protocol, design,
setting, factor, inference, and source-array identities are validated at both
entry and receipt boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np

from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    M03REvaluationError as _V5NumericalError,
)
from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    _fold_fixed_effect_coefficients as _numerical_fold_fixed_effect_coefficients,
)
from rl_quant.evaluation.hold30_alpha_m03r_v5 import (
    _regression as _numerical_regression,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    validate_m03r_v6_artifact_identity,
)

M03R_V6_EVALUATION_SCHEMA = "rl-quant.hold30.m03r-v6-evaluation-v1"
M03R_V6_FACTOR_MANIFEST_SCHEMA = "rl-quant.hold30.m03r-v6-factor-manifest-v1"
M03R_V6_INFERENCE_MANIFEST_SCHEMA = "rl-quant.hold30.m03r-v6-inference-manifest-v1"
M03R_V6_COMMON_EVALUATOR_INPUT_SCHEMA = (
    "rl-quant.hold30.m03r-v6-common-evaluator-inputs-v1"
)
M03R_V6_CANDIDATE_POLICY_RETURNS_SCHEMA = (
    "rl-quant.hold30.m03r-v6-candidate-policy-returns-v1"
)
M03R_V6_OUTER_FOLDS = 6
M03R_V6_SCORE_SESSIONS_PER_FOLD = 63
M03R_V6_PRIMARY_BOOTSTRAP_BLOCK_LENGTH = 21
M03R_V6_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS = (10, 30)
M03R_V6_PRIMARY_HAC_LAG = 30
M03R_V6_MARKET_FACTOR_NAME = "PIT_CAP_MARKET_EXCESS"
M03R_V6_PROMOTION_BLOCKERS = (
    "public-production-evaluator-driver-not-implemented",
    "multiplicity-adjusted-factor-alpha-family-not-bound",
    "outer-data-role-and-access-receipts-not-bound",
)
_DIGEST_CHARS = frozenset("0123456789abcdef")
_COMMON_EVALUATOR_INPUT_DOMAIN = (
    b"rl-quant.hold30.m03r-v6-common-evaluator-inputs-v1\x00"
)
_CANDIDATE_POLICY_RETURNS_DOMAIN = (
    b"rl-quant.hold30.m03r-v6-candidate-policy-returns-v1\x00"
)
_BOOTSTRAP_DOMAIN = b"rl-quant.hold30.m03r-v6-moving-block-v1\x00"
_SUPPORTED_FACTOR_RETURN_CONVENTIONS = frozenset(
    {
        "daily-simple-long-short-return",
        "daily-simple-excess-return",
    }
)


class M03RV6EvaluationError(ValueError):
    """A v6 evaluator input, manifest, or receipt is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M03RV6EvaluationError(
            "v6 evaluation payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise M03RV6EvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise M03RV6EvaluationError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True, slots=True)
class M03RV6FactorManifest:
    """Point-in-time factor identity frozen before evaluation."""

    factor_names: tuple[str, ...]
    factor_return_conventions: tuple[str, ...]
    point_in_time_source_manifest_sha256: str
    manifest_sha256: str
    point_in_time: bool = True
    outer_data_used_to_define_factor_set: bool = False
    schema: str = M03R_V6_FACTOR_MANIFEST_SCHEMA
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    def semantics(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "factor_names": list(self.factor_names),
            "factor_return_conventions": list(self.factor_return_conventions),
            "point_in_time_source_manifest_sha256": (
                self.point_in_time_source_manifest_sha256
            ),
            "point_in_time": self.point_in_time,
            "outer_data_used_to_define_factor_set": (
                self.outer_data_used_to_define_factor_set
            ),
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_V6_FACTOR_MANIFEST_SCHEMA
            or self.protocol_generation != M03R_PROTOCOL_GENERATION
            or self.design_id != M03R_DESIGN_ID
        ):
            raise M03RV6EvaluationError("v6 factor-manifest identity drifted")
        if (
            type(self.factor_names) is not tuple
            or not self.factor_names
            or any(
                type(name) is not str or not name or name.strip() != name
                for name in self.factor_names
            )
            or len(set(self.factor_names)) != len(self.factor_names)
            or M03R_V6_MARKET_FACTOR_NAME in self.factor_names
        ):
            raise M03RV6EvaluationError(
                "factor_names must be unique canonical non-market names"
            )
        if (
            type(self.factor_return_conventions) is not tuple
            or len(self.factor_return_conventions) != len(self.factor_names)
            or any(
                convention not in _SUPPORTED_FACTOR_RETURN_CONVENTIONS
                for convention in self.factor_return_conventions
            )
        ):
            raise M03RV6EvaluationError(
                "every factor requires one supported return convention"
            )
        _require_digest(
            "point_in_time_source_manifest_sha256",
            self.point_in_time_source_manifest_sha256,
        )
        if not self.point_in_time or self.outer_data_used_to_define_factor_set:
            raise M03RV6EvaluationError(
                "factor manifest must be point-in-time and defined without outer data"
            )
        if _require_digest("manifest_sha256", self.manifest_sha256) != _sha256(
            self.semantics()
        ):
            raise M03RV6EvaluationError("factor manifest hash mismatch")


def build_m03r_v6_factor_manifest(
    *,
    factor_names: tuple[str, ...],
    factor_return_conventions: tuple[str, ...],
    point_in_time_source_manifest_sha256: str,
) -> M03RV6FactorManifest:
    """Build an immutable factor manifest from its exact semantics."""

    fields: dict[str, Any] = {
        "factor_names": factor_names,
        "factor_return_conventions": factor_return_conventions,
        "point_in_time_source_manifest_sha256": (point_in_time_source_manifest_sha256),
        "point_in_time": True,
        "outer_data_used_to_define_factor_set": False,
        "schema": M03R_V6_FACTOR_MANIFEST_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
    }
    unsigned = M03RV6FactorManifest.__new__(M03RV6FactorManifest)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV6FactorManifest(
        **fields,
        manifest_sha256=_sha256(unsigned.semantics()),
    )


@dataclass(frozen=True, slots=True)
class M03RV6InferenceManifest:
    """Uncertainty plan frozen before outer evaluation."""

    factor_manifest_sha256: str
    bootstrap_replicates: int
    bootstrap_seed_sha256: str
    inference_manifest_sha256: str
    primary_bootstrap_block_length_trading_sessions: int = 21
    sensitivity_bootstrap_block_lengths_trading_sessions: tuple[int, ...] = (10, 30)
    primary_hac_lag_trading_sessions: int = 30
    one_sided_alpha: float = 0.05
    schema: str = M03R_V6_INFERENCE_MANIFEST_SCHEMA
    protocol_generation: str = M03R_PROTOCOL_GENERATION
    design_id: str = M03R_DESIGN_ID

    @property
    def bootstrap_block_lengths_trading_sessions(self) -> tuple[int, ...]:
        return (
            self.primary_bootstrap_block_length_trading_sessions,
            *self.sensitivity_bootstrap_block_lengths_trading_sessions,
        )

    def semantics(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "factor_manifest_sha256": self.factor_manifest_sha256,
            "outer_fold_count": M03R_V6_OUTER_FOLDS,
            "score_sessions_per_fold": M03R_V6_SCORE_SESSIONS_PER_FOLD,
            "primary_bootstrap_block_length_trading_sessions": (
                self.primary_bootstrap_block_length_trading_sessions
            ),
            "sensitivity_bootstrap_block_lengths_trading_sessions": list(
                self.sensitivity_bootstrap_block_lengths_trading_sessions
            ),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed_sha256": self.bootstrap_seed_sha256,
            "bootstrap_resampling": "within-fold-circular-moving-block",
            "primary_hac_lag_trading_sessions": (self.primary_hac_lag_trading_sessions),
            "one_sided_alpha": self.one_sided_alpha,
            "quantile_method": "inverted_cdf",
        }

    def __post_init__(self) -> None:
        if (
            self.schema != M03R_V6_INFERENCE_MANIFEST_SCHEMA
            or self.protocol_generation != M03R_PROTOCOL_GENERATION
            or self.design_id != M03R_DESIGN_ID
        ):
            raise M03RV6EvaluationError("v6 inference-manifest identity drifted")
        _require_digest("factor_manifest_sha256", self.factor_manifest_sha256)
        _require_digest("bootstrap_seed_sha256", self.bootstrap_seed_sha256)
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates < 1_000
        ):
            raise M03RV6EvaluationError("bootstrap_replicates must be at least 1,000")
        if (
            self.primary_bootstrap_block_length_trading_sessions
            != M03R_V6_PRIMARY_BOOTSTRAP_BLOCK_LENGTH
            or self.sensitivity_bootstrap_block_lengths_trading_sessions
            != M03R_V6_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS
        ):
            raise M03RV6EvaluationError(
                "v6 bootstrap blocks must be primary 21 and sensitivities 10/30"
            )
        if self.primary_hac_lag_trading_sessions != M03R_V6_PRIMARY_HAC_LAG:
            raise M03RV6EvaluationError("v6 primary HAC lag must be 30 sessions")
        if self.one_sided_alpha != 0.05:
            raise M03RV6EvaluationError("v6 one-sided alpha must be exactly 0.05")
        if _require_digest(
            "inference_manifest_sha256", self.inference_manifest_sha256
        ) != _sha256(self.semantics()):
            raise M03RV6EvaluationError("inference manifest hash mismatch")


def build_m03r_v6_inference_manifest(
    *,
    factor_manifest: M03RV6FactorManifest | None,
    bootstrap_replicates: int,
    bootstrap_seed_sha256: str,
) -> M03RV6InferenceManifest:
    """Build the frozen 21/(10,30) within-fold bootstrap plan."""

    if not isinstance(factor_manifest, M03RV6FactorManifest):
        raise M03RV6EvaluationError("typed v6 factor manifest is required")
    factor_manifest.__post_init__()
    fields: dict[str, Any] = {
        "factor_manifest_sha256": factor_manifest.manifest_sha256,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed_sha256": bootstrap_seed_sha256,
        "primary_bootstrap_block_length_trading_sessions": 21,
        "sensitivity_bootstrap_block_lengths_trading_sessions": (10, 30),
        "primary_hac_lag_trading_sessions": 30,
        "one_sided_alpha": 0.05,
        "schema": M03R_V6_INFERENCE_MANIFEST_SCHEMA,
        "protocol_generation": M03R_PROTOCOL_GENERATION,
        "design_id": M03R_DESIGN_ID,
    }
    unsigned = M03RV6InferenceManifest.__new__(M03RV6InferenceManifest)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV6InferenceManifest(
        **fields,
        inference_manifest_sha256=_sha256(unsigned.semantics()),
    )


def _canonical_string_array(
    name: str,
    value: object,
    shape: tuple[int, ...],
    *,
    iso_dates: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != shape:
        raise M03RV6EvaluationError(f"{name} must have shape {shape}")
    result = np.empty(shape, dtype=object)
    for index, item in np.ndenumerate(array):
        if type(item) is not str or not item or item.strip() != item or "\x00" in item:
            raise M03RV6EvaluationError(f"{name} must contain canonical strings")
        if iso_dates:
            try:
                parsed = date.fromisoformat(item)
            except ValueError as exc:
                raise M03RV6EvaluationError(
                    f"{name} must contain canonical ISO-8601 dates"
                ) from exc
            if parsed.isoformat() != item:
                raise M03RV6EvaluationError(
                    f"{name} must contain canonical ISO-8601 dates"
                )
        result[index] = item
    return result


def _normalized_common_inputs(
    *,
    score_dates: object,
    fold_ids: object,
    benchmark_net_returns: object,
    risk_free_returns: object,
    market_total_returns: object,
    factor_returns: object,
    factor_count: int,
) -> tuple[np.ndarray, ...]:
    shape = (M03R_V6_OUTER_FOLDS, M03R_V6_SCORE_SESSIONS_PER_FOLD)
    dates = _canonical_string_array("score_dates", score_dates, shape, iso_dates=True)
    folds = _canonical_string_array("fold_ids", fold_ids, shape)
    row_fold_ids = tuple(str(folds[row, 0]) for row in range(M03R_V6_OUTER_FOLDS))
    if (
        any(len(set(folds[row].tolist())) != 1 for row in range(M03R_V6_OUTER_FOLDS))
        or len(set(row_fold_ids)) != M03R_V6_OUTER_FOLDS
    ):
        raise M03RV6EvaluationError("fold_ids must identify six distinct row folds")
    parsed_dates = [date.fromisoformat(str(item)) for item in dates.reshape(-1)]
    if any(current <= previous for previous, current in pairwise(parsed_dates)):
        raise M03RV6EvaluationError("score_dates must be globally strictly increasing")
    benchmark = _finite_array("benchmark_net_returns", benchmark_net_returns, shape)
    risk_free = _finite_array("risk_free_returns", risk_free_returns, shape)
    market = _finite_array("market_total_returns", market_total_returns, shape)
    factors = _finite_array(
        "factor_returns",
        factor_returns,
        (*shape, factor_count),
    )
    if np.any(benchmark <= -1.0):
        raise M03RV6EvaluationError("benchmark returns must exceed -1")
    return dates, folds, benchmark, risk_free, market, factors


def _normalized_inputs(
    *,
    score_dates: object,
    fold_ids: object,
    policy_net_returns: object,
    benchmark_net_returns: object,
    risk_free_returns: object,
    market_total_returns: object,
    factor_returns: object,
    factor_count: int,
) -> tuple[np.ndarray, ...]:
    common = _normalized_common_inputs(
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark_net_returns,
        risk_free_returns=risk_free_returns,
        market_total_returns=market_total_returns,
        factor_returns=factor_returns,
        factor_count=factor_count,
    )
    shape = (M03R_V6_OUTER_FOLDS, M03R_V6_SCORE_SESSIONS_PER_FOLD)
    policy = _finite_array("policy_net_returns", policy_net_returns, shape)
    if np.any(policy <= -1.0):
        raise M03RV6EvaluationError("policy returns must exceed -1")
    dates, folds, benchmark, risk_free, market, factors = common
    return dates, folds, policy, benchmark, risk_free, market, factors


def _update_array_hash(digest: Any, name: str, value: np.ndarray) -> None:
    if value.dtype == object:
        payload = _canonical_json(value.tolist())
        dtype = "canonical-utf8-string"
    else:
        normalized = np.ascontiguousarray(value, dtype=">f8")
        payload = normalized.tobytes(order="C")
        dtype = "big-endian-float64"
    metadata = _canonical_json(
        {"name": name, "shape": list(value.shape), "normalized_dtype": dtype}
    )
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _validate_v6_common_identity(
    *,
    protocol_generation: str,
    design_id: str,
) -> None:
    """Validate generation/design without making common evidence setting-specific."""

    if protocol_generation != M03R_PROTOCOL_GENERATION:
        raise M03RV6EvaluationError(
            "M03R v6 evaluator evidence cannot use another protocol generation; "
            "v5 remains immutable"
        )
    if design_id != M03R_DESIGN_ID:
        raise M03RV6EvaluationError("M03R v6 evaluator design identity drifted")


def _validated_manifests(
    *,
    factor_manifest: M03RV6FactorManifest | None,
    inference_manifest: M03RV6InferenceManifest | None,
) -> tuple[M03RV6FactorManifest, M03RV6InferenceManifest]:
    if not isinstance(factor_manifest, M03RV6FactorManifest):
        raise M03RV6EvaluationError("typed v6 factor manifest is required")
    if not isinstance(inference_manifest, M03RV6InferenceManifest):
        raise M03RV6EvaluationError("typed v6 inference manifest is required")
    factor_manifest.__post_init__()
    inference_manifest.__post_init__()
    if inference_manifest.factor_manifest_sha256 != factor_manifest.manifest_sha256:
        raise M03RV6EvaluationError("inference and factor manifests are not bound")
    return factor_manifest, inference_manifest


def m03r_v6_common_evaluator_inputs_sha256(
    *,
    protocol_generation: str,
    design_id: str,
    score_dates: object,
    fold_ids: object,
    benchmark_net_returns: object,
    risk_free_returns: object,
    market_total_returns: object,
    factor_returns: object,
    factor_manifest: M03RV6FactorManifest | None,
    inference_manifest: M03RV6InferenceManifest | None,
) -> str:
    """Bind candidate-independent chronology, benchmark, factors, and inference."""

    _validate_v6_common_identity(
        protocol_generation=protocol_generation,
        design_id=design_id,
    )
    factor_manifest, inference_manifest = _validated_manifests(
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    arrays = _normalized_common_inputs(
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark_net_returns,
        risk_free_returns=risk_free_returns,
        market_total_returns=market_total_returns,
        factor_returns=factor_returns,
        factor_count=len(factor_manifest.factor_names),
    )
    digest = hashlib.sha256()
    digest.update(_COMMON_EVALUATOR_INPUT_DOMAIN)
    header = _canonical_json(
        {
            "schema": M03R_V6_COMMON_EVALUATOR_INPUT_SCHEMA,
            "protocol_generation": protocol_generation,
            "design_id": design_id,
            "factor_manifest_sha256": factor_manifest.manifest_sha256,
            "inference_manifest_sha256": (inference_manifest.inference_manifest_sha256),
        }
    )
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name, value in zip(
        (
            "score_dates",
            "fold_ids",
            "benchmark_net_returns",
            "risk_free_returns",
            "market_total_returns",
            "factor_returns",
        ),
        arrays,
        strict=True,
    ):
        _update_array_hash(digest, name, value)
    return digest.hexdigest()


def m03r_v6_candidate_policy_returns_sha256(
    *,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
    policy_net_returns: object,
    common_evaluator_inputs_sha256: str,
) -> str:
    """Bind one checkpoint's policy path to one frozen common input panel."""

    try:
        validate_m03r_v6_artifact_identity(
            protocol_generation=protocol_generation,
            design_id=design_id,
            setting_id=setting_id,
        )
    except ValueError as exc:
        raise M03RV6EvaluationError(str(exc)) from exc
    shape = (M03R_V6_OUTER_FOLDS, M03R_V6_SCORE_SESSIONS_PER_FOLD)
    policy = _finite_array("policy_net_returns", policy_net_returns, shape)
    if np.any(policy <= -1.0):
        raise M03RV6EvaluationError("policy returns must exceed -1")
    common_digest = _require_digest(
        "common_evaluator_inputs_sha256", common_evaluator_inputs_sha256
    )
    digest = hashlib.sha256()
    digest.update(_CANDIDATE_POLICY_RETURNS_DOMAIN)
    header = _canonical_json(
        {
            "schema": M03R_V6_CANDIDATE_POLICY_RETURNS_SCHEMA,
            "protocol_generation": protocol_generation,
            "design_id": design_id,
            "setting_id": setting_id,
            "common_evaluator_inputs_sha256": common_digest,
        }
    )
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    _update_array_hash(digest, "policy_net_returns", policy)
    return digest.hexdigest()


def _block_indices(
    seed: bytes,
    replicate: int,
    fold: int,
    block_length: int,
) -> np.ndarray:
    needed = M03R_V6_SCORE_SESSIONS_PER_FOLD
    material = (
        _BOOTSTRAP_DOMAIN
        + seed
        + replicate.to_bytes(8, "big")
        + fold.to_bytes(2, "big")
        + block_length.to_bytes(2, "big")
    )
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    )
    starts = rng.integers(0, needed, size=math.ceil(needed / block_length))
    return np.concatenate(
        [(start + np.arange(block_length, dtype=np.int64)) % needed for start in starts]
    )[:needed]


def _regression(
    dependent: np.ndarray,
    regressors: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    try:
        return _numerical_regression(dependent, regressors, names)
    except _V5NumericalError as exc:
        raise M03RV6EvaluationError(str(exc)) from exc


def _active_alpha(
    dependent: np.ndarray,
    regressors: np.ndarray,
    *,
    context: str,
) -> float:
    try:
        coefficients, _design = _numerical_fold_fixed_effect_coefficients(
            dependent,
            regressors,
            context=context,
        )
    except _V5NumericalError as exc:
        raise M03RV6EvaluationError(str(exc)) from exc
    return float(coefficients[0])


def evaluate_m03r_v6_inference(
    *,
    protocol_generation: str,
    design_id: str,
    setting_id: str,
    score_dates: object,
    fold_ids: object,
    policy_net_returns: object,
    benchmark_net_returns: object,
    risk_free_returns: object,
    market_total_returns: object,
    factor_returns: object,
    factor_manifest: M03RV6FactorManifest | None,
    inference_manifest: M03RV6InferenceManifest | None,
    common_evaluator_inputs_sha256: str,
    candidate_policy_returns_sha256: str,
) -> dict[str, Any]:
    """Evaluate v6 portfolio, benchmark, and active factor alpha."""

    try:
        setting = validate_m03r_v6_artifact_identity(
            protocol_generation=protocol_generation,
            design_id=design_id,
            setting_id=setting_id,
        )
    except ValueError as exc:
        raise M03RV6EvaluationError(str(exc)) from exc
    factor_manifest, inference_manifest = _validated_manifests(
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    arrays = _normalized_inputs(
        score_dates=score_dates,
        fold_ids=fold_ids,
        policy_net_returns=policy_net_returns,
        benchmark_net_returns=benchmark_net_returns,
        risk_free_returns=risk_free_returns,
        market_total_returns=market_total_returns,
        factor_returns=factor_returns,
        factor_count=len(factor_manifest.factor_names),
    )
    _dates, _folds, policy, benchmark, risk_free, market, factors = arrays
    computed_common_sha256 = m03r_v6_common_evaluator_inputs_sha256(
        protocol_generation=protocol_generation,
        design_id=design_id,
        score_dates=score_dates,
        fold_ids=fold_ids,
        benchmark_net_returns=benchmark,
        risk_free_returns=risk_free,
        market_total_returns=market,
        factor_returns=factors,
        factor_manifest=factor_manifest,
        inference_manifest=inference_manifest,
    )
    if _require_digest(
        "common_evaluator_inputs_sha256", common_evaluator_inputs_sha256
    ) != computed_common_sha256:
        raise M03RV6EvaluationError(
            "common_evaluator_inputs_sha256 does not match the supplied common arrays"
        )
    computed_candidate_sha256 = m03r_v6_candidate_policy_returns_sha256(
        protocol_generation=protocol_generation,
        design_id=design_id,
        setting_id=setting_id,
        policy_net_returns=policy,
        common_evaluator_inputs_sha256=computed_common_sha256,
    )
    if _require_digest(
        "candidate_policy_returns_sha256", candidate_policy_returns_sha256
    ) != computed_candidate_sha256:
        raise M03RV6EvaluationError(
            "candidate_policy_returns_sha256 does not match the supplied policy path"
        )

    market_excess = market - risk_free
    regressors = np.concatenate((market_excess[..., None], factors), axis=-1)
    names = (M03R_V6_MARKET_FACTOR_NAME, *factor_manifest.factor_names)
    portfolio = _regression(policy - risk_free, regressors, names)
    benchmark_regression = _regression(benchmark - risk_free, regressors, names)
    active = _regression(policy - benchmark, regressors, names)

    bootstrap: dict[str, Any] = {}
    seed = bytes.fromhex(inference_manifest.bootstrap_seed_sha256)
    for block_length in inference_manifest.bootstrap_block_lengths_trading_sessions:
        alphas = np.empty(inference_manifest.bootstrap_replicates, dtype=np.float64)
        for replicate in range(inference_manifest.bootstrap_replicates):
            indexes = tuple(
                _block_indices(seed, replicate, fold, block_length)
                for fold in range(M03R_V6_OUTER_FOLDS)
            )
            sampled_active = np.stack(
                [
                    (policy - benchmark)[fold, index]
                    for fold, index in enumerate(indexes)
                ]
            )
            sampled_market = np.stack(
                [market_excess[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_factors = np.stack(
                [factors[fold, index] for fold, index in enumerate(indexes)]
            )
            sampled_regressors = np.concatenate(
                (sampled_market[..., None], sampled_factors),
                axis=-1,
            )
            alphas[replicate] = _active_alpha(
                sampled_active,
                sampled_regressors,
                context=f"v6 bootstrap active alpha replicate {replicate}",
            )
        daily_lcb = float(
            np.quantile(
                alphas,
                inference_manifest.one_sided_alpha,
                method="inverted_cdf",
            )
        )
        bootstrap[str(block_length)] = {
            "one_sided_confidence_level": float(
                1.0 - inference_manifest.one_sided_alpha
            ),
            "active_multifactor_alpha_daily_lcb": daily_lcb,
            "active_multifactor_alpha_annualized_lcb": 252.0 * daily_lcb,
        }

    unsigned: dict[str, Any] = {
        "schema": M03R_V6_EVALUATION_SCHEMA,
        "protocol_generation": protocol_generation,
        "design_id": design_id,
        "setting_id": setting.setting_id,
        "promotion_eligible_setting": setting.promotion_eligible,
        "evaluation_scope": "pure-numerical-public-surface-not-production-driver",
        "promotion_authorized": False,
        "promotion_blockers": list(M03R_V6_PROMOTION_BLOCKERS),
        "factor_manifest": {
            **factor_manifest.semantics(),
            "manifest_sha256": factor_manifest.manifest_sha256,
        },
        "inference_manifest": {
            **inference_manifest.semantics(),
            "inference_manifest_sha256": (inference_manifest.inference_manifest_sha256),
        },
        "common_evaluator_inputs_schema": M03R_V6_COMMON_EVALUATOR_INPUT_SCHEMA,
        "common_evaluator_inputs_sha256": computed_common_sha256,
        "candidate_policy_returns_schema": (
            M03R_V6_CANDIDATE_POLICY_RETURNS_SCHEMA
        ),
        "candidate_policy_returns_sha256": computed_candidate_sha256,
        "portfolio_multifactor_regression": portfolio,
        "benchmark_multifactor_regression": benchmark_regression,
        "active_multifactor_regression": active,
        "bootstrap": bootstrap,
    }
    return {**unsigned, "receipt_sha256": _sha256(unsigned)}


def validate_m03r_v6_evaluation_receipt(
    receipt: object,
    *,
    factor_manifest: M03RV6FactorManifest | None,
    inference_manifest: M03RV6InferenceManifest | None,
) -> None:
    """Validate v6 identity/manifests and all content-bound receipt fields."""

    if not isinstance(receipt, dict):
        raise M03RV6EvaluationError("v6 evaluation receipt must be a dict")
    if not isinstance(factor_manifest, M03RV6FactorManifest):
        raise M03RV6EvaluationError("typed v6 factor manifest is required")
    if not isinstance(inference_manifest, M03RV6InferenceManifest):
        raise M03RV6EvaluationError("typed v6 inference manifest is required")
    factor_manifest.__post_init__()
    inference_manifest.__post_init__()
    if inference_manifest.factor_manifest_sha256 != factor_manifest.manifest_sha256:
        raise M03RV6EvaluationError("inference and factor manifests are not bound")
    required = {
        "schema",
        "protocol_generation",
        "design_id",
        "setting_id",
        "promotion_eligible_setting",
        "evaluation_scope",
        "promotion_authorized",
        "promotion_blockers",
        "factor_manifest",
        "inference_manifest",
        "common_evaluator_inputs_schema",
        "common_evaluator_inputs_sha256",
        "candidate_policy_returns_schema",
        "candidate_policy_returns_sha256",
        "portfolio_multifactor_regression",
        "benchmark_multifactor_regression",
        "active_multifactor_regression",
        "bootstrap",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise M03RV6EvaluationError("v6 evaluation receipt keys drifted")
    try:
        setting = validate_m03r_v6_artifact_identity(
            protocol_generation=receipt["protocol_generation"],
            design_id=receipt["design_id"],
            setting_id=receipt["setting_id"],
        )
    except (TypeError, ValueError) as exc:
        raise M03RV6EvaluationError(str(exc)) from exc
    if receipt["schema"] != M03R_V6_EVALUATION_SCHEMA:
        raise M03RV6EvaluationError("v6 evaluation schema drifted")
    expected_factor = {
        **factor_manifest.semantics(),
        "manifest_sha256": factor_manifest.manifest_sha256,
    }
    expected_inference = {
        **inference_manifest.semantics(),
        "inference_manifest_sha256": inference_manifest.inference_manifest_sha256,
    }
    if receipt["factor_manifest"] != expected_factor:
        raise M03RV6EvaluationError("receipt factor manifest mismatch")
    if receipt["inference_manifest"] != expected_inference:
        raise M03RV6EvaluationError("receipt inference manifest mismatch")
    if (
        receipt["promotion_eligible_setting"] is not setting.promotion_eligible
        or receipt["evaluation_scope"]
        != "pure-numerical-public-surface-not-production-driver"
        or receipt["promotion_authorized"] is not False
        or receipt["promotion_blockers"] != list(M03R_V6_PROMOTION_BLOCKERS)
    ):
        raise M03RV6EvaluationError("v6 evaluation authorization semantics drifted")
    if (
        receipt["common_evaluator_inputs_schema"]
        != M03R_V6_COMMON_EVALUATOR_INPUT_SCHEMA
    ):
        raise M03RV6EvaluationError("v6 common-evaluator-input schema drifted")
    _require_digest(
        "common_evaluator_inputs_sha256",
        receipt["common_evaluator_inputs_sha256"],
    )
    if (
        receipt["candidate_policy_returns_schema"]
        != M03R_V6_CANDIDATE_POLICY_RETURNS_SCHEMA
    ):
        raise M03RV6EvaluationError("v6 candidate-policy-return schema drifted")
    _require_digest(
        "candidate_policy_returns_sha256",
        receipt["candidate_policy_returns_sha256"],
    )
    regression_names = (
        "portfolio_multifactor_regression",
        "benchmark_multifactor_regression",
        "active_multifactor_regression",
    )
    regressions: list[dict[str, Any]] = []
    expected_loading_names = {
        M03R_V6_MARKET_FACTOR_NAME,
        *factor_manifest.factor_names,
    }
    for name in regression_names:
        row = receipt[name]
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("loadings"), dict)
            or set(row["loadings"]) != expected_loading_names
        ):
            raise M03RV6EvaluationError(f"{name} structure drifted")
        alpha_daily = float(row.get("alpha_daily", math.nan))
        alpha_annualized = float(row.get("alpha_annualized_arithmetic", math.nan))
        if (
            not math.isfinite(alpha_daily)
            or not math.isfinite(alpha_annualized)
            or not math.isclose(
                alpha_annualized,
                252.0 * alpha_daily,
                abs_tol=1e-15,
            )
            or any(
                not math.isfinite(float(value)) for value in row["loadings"].values()
            )
        ):
            raise M03RV6EvaluationError(f"{name} values drifted")
        regressions.append(row)
    portfolio_regression, benchmark_regression, active_regression = regressions
    if not math.isclose(
        float(active_regression["alpha_daily"]),
        float(portfolio_regression["alpha_daily"])
        - float(benchmark_regression["alpha_daily"]),
        abs_tol=1e-12,
    ) or any(
        not math.isclose(
            float(active_regression["loadings"][name]),
            float(portfolio_regression["loadings"][name])
            - float(benchmark_regression["loadings"][name]),
            abs_tol=1e-12,
        )
        for name in expected_loading_names
    ):
        raise M03RV6EvaluationError(
            "active multifactor regression must equal portfolio minus benchmark"
        )
    bootstrap = receipt["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"10", "21", "30"}:
        raise M03RV6EvaluationError("v6 bootstrap blocks must be exactly 10/21/30")
    for block in ("10", "21", "30"):
        row = bootstrap[block]
        if not isinstance(row, dict) or set(row) != {
            "one_sided_confidence_level",
            "active_multifactor_alpha_daily_lcb",
            "active_multifactor_alpha_annualized_lcb",
        }:
            raise M03RV6EvaluationError("v6 bootstrap receipt row drifted")
        daily = float(row["active_multifactor_alpha_daily_lcb"])
        annualized = float(row["active_multifactor_alpha_annualized_lcb"])
        if (
            not math.isfinite(daily)
            or not math.isfinite(annualized)
            or not math.isclose(annualized, 252.0 * daily, abs_tol=1e-15)
            or row["one_sided_confidence_level"] != 0.95
        ):
            raise M03RV6EvaluationError("v6 active-alpha bootstrap values drifted")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _require_digest("receipt_sha256", receipt["receipt_sha256"]) != _sha256(
        unsigned
    ):
        raise M03RV6EvaluationError("v6 evaluation receipt hash mismatch")


__all__ = [
    "M03R_V6_BOOTSTRAP_SENSITIVITY_BLOCK_LENGTHS",
    "M03R_V6_CANDIDATE_POLICY_RETURNS_SCHEMA",
    "M03R_V6_COMMON_EVALUATOR_INPUT_SCHEMA",
    "M03R_V6_EVALUATION_SCHEMA",
    "M03R_V6_FACTOR_MANIFEST_SCHEMA",
    "M03R_V6_INFERENCE_MANIFEST_SCHEMA",
    "M03R_V6_PRIMARY_BOOTSTRAP_BLOCK_LENGTH",
    "M03RV6EvaluationError",
    "M03RV6FactorManifest",
    "M03RV6InferenceManifest",
    "build_m03r_v6_factor_manifest",
    "build_m03r_v6_inference_manifest",
    "evaluate_m03r_v6_inference",
    "m03r_v6_candidate_policy_returns_sha256",
    "m03r_v6_common_evaluator_inputs_sha256",
    "validate_m03r_v6_evaluation_receipt",
]
