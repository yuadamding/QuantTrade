"""Deterministic C7/C8 random controls for the Hold-30 pre-lockbox protocol.

The random intent generator is deliberately return-blind: its only market
input is the decision-time trade mask.  Economic returns enter later, through
the package-owned chronological runtime, and the matcher may consume them only
through the two explicitly named covariance statistics (beta and tracking
error).  Every otherwise underspecified byte or numeric convention is bound in
the bank identity and receipts.

Only the superseding mechanism-8 v2 generation is accepted.  A bank always
contains IDs ``0..8191``.  C7 is exactly ``0..63``; C8 is filtered and ranked
only from ``64..8191``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from rl_quant.envs.hold30 import TurnoverCause
from rl_quant.execution.hold30 import (
    HOLD30_MAX_DISCRETIONARY_TURNOVER,
    HOLD30_MAX_STOCK_WEIGHT,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30 import (
    HOLD30_PROTOCOL_GENERATION,
    resolve_hold30_setting,
)
from rl_quant.training.hold30 import Hold30CreditRoles
from rl_quant.training.hold30_runtime import (
    HOLD30_RECONCILIATION_TOLERANCE,
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30Sequence,
)


HOLD30_RANDOM_BANK_SIZE = 8_192
HOLD30_RANDOM_CONTROL_COUNT = 64
HOLD30_C7_IDS = tuple(range(HOLD30_RANDOM_CONTROL_COUNT))
HOLD30_C8_CANDIDATE_IDS = tuple(
    range(HOLD30_RANDOM_CONTROL_COUNT, HOLD30_RANDOM_BANK_SIZE)
)

HOLD30_RANDOM_RHOS = (0.90, 0.95, 0.98, 0.99)
HOLD30_RANDOM_SCORE_SCALES = (0.25, 0.50, 1.00, 2.00)
HOLD30_RANDOM_H1_GATE_MEANS = (0.01, 0.02, 1.0 / 30.0, 0.05, 0.08)
HOLD30_RANDOM_H1_GATE_LOGIT_NOISES = (0.10, 0.30, 0.60)
HOLD30_RANDOM_H2_HAZARD_CENTERS = (-4.0, -2.0, 0.0, 2.0, 4.0)
HOLD30_RANDOM_H2_HAZARD_SCALES = (0.50, 1.00, 2.00, 4.00)
HOLD30_RANDOM_H2_EXPOSURE_SCALES = (0.00, 0.25, 0.50, 1.00)

HOLD30_RANDOM_KEY_ENCODING = "utf8-nul-separated-decimal-integers-v1"
HOLD30_RANDOM_STREAM_CONTRACT = (
    "numpy-philox4x64-10-direct-key-counter-box-muller-float64-v1"
)
HOLD30_RANDOM_AR1_INITIAL_STATE = "all_latents_exact_zero_before_first_decision"
HOLD30_RANDOM_CENTERING = "available_risky_arithmetic_mean_exact_zero"
HOLD30_RANDOM_BANK_SCHEMA = "rl-quant.hold30.random-control-bank"
HOLD30_C7_RECEIPT_SCHEMA = "rl-quant.hold30.c7-controls"
HOLD30_C8_SELECTION_SCHEMA = "rl-quant.hold30.c8-selection"
HOLD30_CROSS_FOLD_MAPPING_SCHEMA = "rl-quant.hold30.random-control-cross-fold-map"
HOLD30_HARD_FEASIBILITY_SCHEMA = "rl-quant.hold30.random-hard-feasibility"
HOLD30_C8_PROFILE_SCHEMA = "rl-quant.hold30.c8-profile"

HOLD30_C8_TOLERANCES = {
    "discretionary_turnover_relative": 0.05,
    "risky_exposure_absolute": 0.01,
    "beta_absolute": 0.05,
    "tracking_error_relative": 0.10,
    "hhi_relative": 0.10,
    "median_sale_age_absolute_sessions": 3.0,
    "survival_30_absolute": 0.05,
}
HOLD30_C8_MIN_SOLD_NOTIONAL = 0.10
_DIGEST_LENGTH = 64
_FEASIBILITY_TOLERANCE = HOLD30_RECONCILIATION_TOLERANCE
_METRIC_ORDER = (
    "discretionary_turnover",
    "risky_exposure",
    "beta",
    "tracking_error",
    "hhi",
    "median_sale_age",
    "survival_30",
)


class Hold30RandomControlError(ValueError):
    """A random-control artifact is invalid, incomplete, or contaminated."""


class Hold30C8GateError(Hold30RandomControlError):
    """The target or candidate bank cannot produce the required 64 controls."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    if "receipt_sha256" in receipt:
        raise Hold30RandomControlError("unsealed payload already contains receipt_sha256")
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    return receipt


def _verify_self_hash(receipt: Mapping[str, Any], *, schema: str) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != schema:
        raise Hold30RandomControlError(f"expected {schema} receipt")
    supplied = _require_digest("receipt_sha256", receipt.get("receipt_sha256"))
    payload = dict(receipt)
    del payload["receipt_sha256"]
    if supplied != _payload_sha256(payload):
        raise Hold30RandomControlError(f"{schema} receipt hash mismatch")


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30RandomControlError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_real(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Hold30RandomControlError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise Hold30RandomControlError(f"{name} must be finite and valid")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _stack_sha256(values: Sequence[torch.Tensor]) -> str:
    if not values:
        return hashlib.sha256(b"").hexdigest()
    return _tensor_sha256(torch.stack(tuple(values)))


def _require_replicate_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Hold30RandomControlError("replicate_id must be an integer")
    if not 0 <= value < HOLD30_RANDOM_BANK_SIZE:
        raise Hold30RandomControlError("replicate_id must lie in [0, 8191]")
    return value


def _protocol_variant_mechanism(
    protocol_generation: str,
    variant_id: str,
    mechanism: str,
) -> None:
    if protocol_generation != HOLD30_PROTOCOL_GENERATION:
        raise Hold30RandomControlError(
            "artifact-producing random controls accept only the superseding v2 generation"
        )
    try:
        setting = resolve_hold30_setting(variant_id)
    except ValueError as exc:
        raise Hold30RandomControlError(str(exc)) from exc
    if setting.mechanism not in {"H1", "H2"} or mechanism != setting.mechanism:
        raise Hold30RandomControlError(
            "v2 C7/C8 variants must be registered H1/H2 settings with matching mechanism"
        )


@dataclass(frozen=True, slots=True)
class Hold30RandomBankIdentity:
    """Return-blind identity for one 8,192-trace variant/fold bank."""

    protocol_generation: str
    variant_id: str
    mechanism: Literal["H1", "H2"]
    fold_index: int
    axis_id: str
    decision_count: int
    batch_size: int
    asset_count: int
    cash_index: int
    dtype: Literal["float32", "float64"]
    decision_mask_sha256: str
    source_archive_sha256: str
    generator_implementation_sha256: str
    builder_receipt_sha256: str
    constraints_receipt_sha256: str
    chronology_receipt_sha256: str
    classification_dimensions: tuple[Literal["liquidity", "sector"], ...]
    classification_availability_receipt_sha256: str

    def __post_init__(self) -> None:
        _protocol_variant_mechanism(
            self.protocol_generation,
            self.variant_id,
            self.mechanism,
        )
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < 6
        ):
            raise Hold30RandomControlError("fold_index must be an integer in [0, 5]")
        if not isinstance(self.axis_id, str) or not self.axis_id or "\x00" in self.axis_id:
            raise Hold30RandomControlError("axis_id must be a nonempty NUL-free string")
        for name in ("protocol_generation", "variant_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise Hold30RandomControlError(f"{name} must be a nonempty NUL-free string")
        for name in ("decision_count", "batch_size", "asset_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Hold30RandomControlError(f"{name} must be a positive integer")
        if self.asset_count < 2:
            raise Hold30RandomControlError("a random-control bank needs CASH and a risky asset")
        if (
            isinstance(self.cash_index, bool)
            or not isinstance(self.cash_index, int)
            or not 0 <= self.cash_index < self.asset_count
        ):
            raise Hold30RandomControlError("cash_index lies outside the asset axis")
        if self.dtype not in {"float32", "float64"}:
            raise Hold30RandomControlError("dtype must be float32 or float64")
        for name in (
            "decision_mask_sha256",
            "source_archive_sha256",
            "generator_implementation_sha256",
            "builder_receipt_sha256",
            "constraints_receipt_sha256",
            "chronology_receipt_sha256",
            "classification_availability_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.classification_dimensions != tuple(
            sorted(set(self.classification_dimensions))
        ) or any(
            value not in {"sector", "liquidity"}
            for value in self.classification_dimensions
        ):
            raise Hold30RandomControlError(
                "classification_dimensions must be sorted unique sector/liquidity names"
            )

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.random-control-bank-identity",
            "schema_version": 1,
            "protocol_generation": self.protocol_generation,
            "variant_id": self.variant_id,
            "mechanism": self.mechanism,
            "fold_index": self.fold_index,
            "axis_id": self.axis_id,
            "shape": {
                "decision_count": self.decision_count,
                "batch_size": self.batch_size,
                "asset_count": self.asset_count,
                "cash_index": self.cash_index,
                "dtype": self.dtype,
            },
            "decision_mask_sha256": self.decision_mask_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "generator_implementation_sha256": self.generator_implementation_sha256,
            "builder_receipt_sha256": self.builder_receipt_sha256,
            "constraints_receipt_sha256": self.constraints_receipt_sha256,
            "chronology_receipt_sha256": self.chronology_receipt_sha256,
            "classification_contract": {
                "dimensions": list(self.classification_dimensions),
                "availability_receipt_sha256": (
                    self.classification_availability_receipt_sha256
                ),
                "missing_declared_dimension_policy": "fail_closed_no_drop_or_impute",
            },
            "rng_contract": {
                "key": "SHA256(protocol || variant || fold || replicate_id)",
                "key_encoding": HOLD30_RANDOM_KEY_ENCODING,
                "stream_algorithm": HOLD30_RANDOM_STREAM_CONTRACT,
                "numpy_version": np.__version__,
                "byte_order": "big",
                "uniform_mapping": "top53_plus_half_over_2pow53",
                "normal_mapping": "box_muller_log_sqrt_cos_sin",
                "ar1_initial_state": HOLD30_RANDOM_AR1_INITIAL_STATE,
                "centering": HOLD30_RANDOM_CENTERING,
            },
            "bank_contract": {
                "bank_ids": [0, HOLD30_RANDOM_BANK_SIZE - 1],
                "c7_ids": [0, HOLD30_RANDOM_CONTROL_COUNT - 1],
                "c8_candidate_ids": [
                    HOLD30_RANDOM_CONTROL_COUNT,
                    HOLD30_RANDOM_BANK_SIZE - 1,
                ],
                "parameter_arrays": {
                    "rho": list(HOLD30_RANDOM_RHOS),
                    "score_scale": list(HOLD30_RANDOM_SCORE_SCALES),
                    "h1_gate_mean": list(HOLD30_RANDOM_H1_GATE_MEANS),
                    "h1_gate_logit_noise": list(
                        HOLD30_RANDOM_H1_GATE_LOGIT_NOISES
                    ),
                    "h2_hazard_center": list(HOLD30_RANDOM_H2_HAZARD_CENTERS),
                    "h2_hazard_scale": list(HOLD30_RANDOM_H2_HAZARD_SCALES),
                    "h2_exposure_scale": list(HOLD30_RANDOM_H2_EXPOSURE_SCALES),
                },
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.payload)


def hold30_random_key(identity: Hold30RandomBankIdentity, replicate_id: int) -> bytes:
    """Return the frozen SHA-256 key for one bank ID."""

    replicate_id = _require_replicate_id(replicate_id)
    encoded = b"\x00".join(
        (
            identity.protocol_generation.encode("utf-8"),
            identity.variant_id.encode("utf-8"),
            str(identity.fold_index).encode("ascii"),
            str(replicate_id).encode("ascii"),
        )
    )
    return hashlib.sha256(encoded).digest()


def _philox_raw(base_key: bytes, stream: str, count: int) -> np.ndarray:
    if not isinstance(stream, str) or not stream or "\x00" in stream:
        raise Hold30RandomControlError("RNG stream names must be nonempty and NUL-free")
    if count < 0:
        raise Hold30RandomControlError("RNG draw count cannot be negative")
    material = hashlib.sha256(
        b"hold30-random-stream-v1\x00" + base_key + b"\x00" + stream.encode("ascii")
    ).digest()
    key = int.from_bytes(material[:16], "big")
    counter = int.from_bytes(material[16:], "big")
    bit_generator = np.random.Philox(key=key, counter=counter)
    return bit_generator.random_raw(count)


def _choice(base_key: bytes, stream: str, values: Sequence[float]) -> float:
    raw = int(_philox_raw(base_key, f"choice:{stream}", 1)[0])
    return float(values[raw % len(values)])


def _normal_array(base_key: bytes, stream: str, count: int) -> np.ndarray:
    pair_count = (count + 1) // 2
    raw = _philox_raw(base_key, f"normal:{stream}", pair_count * 2)
    mantissa = (raw >> np.uint64(11)).astype(np.float64)
    uniforms = (mantissa + 0.5) / float(1 << 53)
    radius = np.sqrt(-2.0 * np.log(uniforms[0::2]))
    angle = 2.0 * np.pi * uniforms[1::2]
    output = np.empty(pair_count * 2, dtype=np.float64)
    output[0::2] = radius * np.cos(angle)
    output[1::2] = radius * np.sin(angle)
    return output[:count]


def _ar1(
    base_key: bytes,
    stream: str,
    shape: tuple[int, ...],
    rho: float,
) -> np.ndarray:
    draws = _normal_array(base_key, stream, math.prod(shape)).reshape(shape)
    output = np.empty_like(draws)
    previous = np.zeros(shape[1:], dtype=np.float64)
    innovation = math.sqrt(1.0 - rho * rho)
    for position in range(shape[0]):
        previous = rho * previous + innovation * draws[position]
        output[position] = previous
    return output


@dataclass(frozen=True, slots=True)
class Hold30RandomParameters:
    rho: float
    score_scale: float
    gate_mean: float | None = None
    gate_logit_noise: float | None = None
    hazard_center: float | None = None
    hazard_scale: float | None = None
    exposure_scale: float | None = None

    @property
    def payload(self) -> dict[str, float | None]:
        return {
            "rho": self.rho,
            "score_scale": self.score_scale,
            "gate_mean": self.gate_mean,
            "gate_logit_noise": self.gate_logit_noise,
            "hazard_center": self.hazard_center,
            "hazard_scale": self.hazard_scale,
            "exposure_scale": self.exposure_scale,
        }


@dataclass(frozen=True, slots=True)
class Hold30RandomIntentTrace:
    identity_sha256: str
    replicate_id: int
    rng_key_sha256: str
    parameters: Hold30RandomParameters
    decision_mask_sha256: str
    target_logits: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    entry_scores: torch.Tensor | None = None
    hazard_residual: torch.Tensor | None = None
    exposure_residual: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_digest("identity_sha256", self.identity_sha256)
        _require_replicate_id(self.replicate_id)
        _require_digest("rng_key_sha256", self.rng_key_sha256)
        _require_digest("decision_mask_sha256", self.decision_mask_sha256)
        populated = {
            name: value
            for name in (
                "target_logits",
                "gate",
                "entry_scores",
                "hazard_residual",
                "exposure_residual",
            )
            if (value := getattr(self, name)) is not None
        }
        for name, value in populated.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise Hold30RandomControlError(f"{name} must be a floating tensor")
            if not bool(torch.isfinite(value).all()):
                raise Hold30RandomControlError(f"{name} contains non-finite values")
        h1 = set(populated) == {"target_logits", "gate"}
        h2 = set(populated) == {
            "entry_scores",
            "hazard_residual",
            "exposure_residual",
        }
        if not (h1 or h2):
            raise Hold30RandomControlError("random intent fields do not form H1 or H2")

    @property
    def mechanism(self) -> str:
        return "H1" if self.target_logits is not None else "H2"

    @property
    def decision_count(self) -> int:
        reference = self.target_logits if self.target_logits is not None else self.entry_scores
        assert reference is not None
        return int(reference.shape[0])

    def intent_at(self, position: int) -> Hold30Intent:
        if not 0 <= position < self.decision_count:
            raise Hold30RandomControlError("intent position lies outside the trace")
        return Hold30Intent(
            target_logits=(None if self.target_logits is None else self.target_logits[position]),
            gate=None if self.gate is None else self.gate[position],
            entry_scores=(None if self.entry_scores is None else self.entry_scores[position]),
            hazard_residual=(
                None if self.hazard_residual is None else self.hazard_residual[position]
            ),
            exposure_residual=(
                None if self.exposure_residual is None else self.exposure_residual[position]
            ),
        )

    @property
    def receipt_payload(self) -> dict[str, Any]:
        tensors = {
            name: _tensor_sha256(value)
            for name in (
                "target_logits",
                "gate",
                "entry_scores",
                "hazard_residual",
                "exposure_residual",
            )
            if (value := getattr(self, name)) is not None
        }
        return {
            "schema": "rl-quant.hold30.random-intent-trace",
            "schema_version": 1,
            "identity_sha256": self.identity_sha256,
            "replicate_id": self.replicate_id,
            "rng_key_sha256": self.rng_key_sha256,
            "decision_mask_sha256": self.decision_mask_sha256,
            "mechanism": self.mechanism,
            "parameters": self.parameters.payload,
            "tensor_sha256s": tensors,
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.receipt_payload)


def _center_available_risky(
    values: np.ndarray,
    decision_mask: np.ndarray,
    cash_index: int,
) -> np.ndarray:
    eligible = decision_mask.copy()
    eligible[..., cash_index] = False
    count = eligible.sum(axis=-1, keepdims=True)
    total = np.where(eligible, values, 0.0).sum(axis=-1, keepdims=True)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = np.where(eligible, values - mean, 0.0)
    centered[..., cash_index] = 0.0
    return centered


def generate_hold30_random_intents(
    identity: Hold30RandomBankIdentity,
    replicate_id: int,
    decision_mask: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> Hold30RandomIntentTrace:
    """Generate one return-blind H1/H2 intent chronology from its bank ID."""

    replicate_id = _require_replicate_id(replicate_id)
    expected_shape = (
        identity.decision_count,
        identity.batch_size,
        identity.asset_count,
    )
    if (
        not isinstance(decision_mask, torch.Tensor)
        or decision_mask.dtype != torch.bool
        or tuple(decision_mask.shape) != expected_shape
    ):
        raise Hold30RandomControlError(
            f"decision_mask must be boolean with shape {expected_shape}"
        )
    mask_cpu = decision_mask.detach().to(device="cpu").contiguous()
    if not bool(mask_cpu[..., identity.cash_index].all()):
        raise Hold30RandomControlError("CASH must always be decision-available")
    if _tensor_sha256(mask_cpu) != identity.decision_mask_sha256:
        raise Hold30RandomControlError("decision mask does not match the bank identity")
    base_key = hold30_random_key(identity, replicate_id)
    rho = _choice(base_key, "rho", HOLD30_RANDOM_RHOS)
    score_scale = _choice(base_key, "score_scale", HOLD30_RANDOM_SCORE_SCALES)
    shape = expected_shape
    mask_numpy = mask_cpu.numpy()
    dtype = torch.float32 if identity.dtype == "float32" else torch.float64
    common = {
        "identity_sha256": identity.receipt_sha256,
        "replicate_id": replicate_id,
        "rng_key_sha256": base_key.hex(),
        "decision_mask_sha256": identity.decision_mask_sha256,
    }
    if identity.mechanism == "H1":
        gate_mean = _choice(base_key, "h1_gate_mean", HOLD30_RANDOM_H1_GATE_MEANS)
        gate_noise = _choice(
            base_key,
            "h1_gate_logit_noise",
            HOLD30_RANDOM_H1_GATE_LOGIT_NOISES,
        )
        scores = score_scale * _ar1(base_key, "h1_stock", shape, rho)
        scores = _center_available_risky(scores, mask_numpy, identity.cash_index)
        gate_latent = _ar1(
            base_key,
            "h1_gate",
            (identity.decision_count, identity.batch_size),
            rho,
        )
        logit_mean = math.log(gate_mean / (1.0 - gate_mean))
        gates = 1.0 / (1.0 + np.exp(-(logit_mean + gate_noise * gate_latent)))
        return Hold30RandomIntentTrace(
            **common,
            parameters=Hold30RandomParameters(
                rho,
                score_scale,
                gate_mean=gate_mean,
                gate_logit_noise=gate_noise,
            ),
            target_logits=torch.from_numpy(scores).to(device=device, dtype=dtype),
            gate=torch.from_numpy(gates).to(device=device, dtype=dtype),
        )

    hazard_center = _choice(
        base_key,
        "h2_hazard_center",
        HOLD30_RANDOM_H2_HAZARD_CENTERS,
    )
    hazard_scale = _choice(
        base_key,
        "h2_hazard_scale",
        HOLD30_RANDOM_H2_HAZARD_SCALES,
    )
    exposure_scale = _choice(
        base_key,
        "h2_exposure_scale",
        HOLD30_RANDOM_H2_EXPOSURE_SCALES,
    )
    entries = score_scale * _ar1(base_key, "h2_entry", shape, rho)
    entries = _center_available_risky(entries, mask_numpy, identity.cash_index)
    hazards = hazard_center + hazard_scale * _ar1(base_key, "h2_hazard", shape, rho)
    hazards = np.clip(hazards, -12.0, 12.0)
    hazards[..., identity.cash_index] = -12.0
    exposures = exposure_scale * _ar1(
        base_key,
        "h2_exposure",
        (identity.decision_count, identity.batch_size),
        rho,
    )
    return Hold30RandomIntentTrace(
        **common,
        parameters=Hold30RandomParameters(
            rho,
            score_scale,
            hazard_center=hazard_center,
            hazard_scale=hazard_scale,
            exposure_scale=exposure_scale,
        ),
        entry_scores=torch.from_numpy(entries).to(device=device, dtype=dtype),
        hazard_residual=torch.from_numpy(hazards).to(device=device, dtype=dtype),
        exposure_residual=torch.from_numpy(exposures).to(device=device, dtype=dtype),
    )


def verify_hold30_random_intents(
    identity: Hold30RandomBankIdentity,
    trace: Hold30RandomIntentTrace,
    decision_mask: torch.Tensor,
) -> None:
    """Regenerate one intent trace and require exact tensor/receipt equality."""

    if trace.identity_sha256 != identity.receipt_sha256:
        raise Hold30RandomControlError("intent trace binds a different bank identity")
    expected = generate_hold30_random_intents(
        identity,
        trace.replicate_id,
        decision_mask,
        device="cpu",
    )
    if expected.receipt_payload != trace.receipt_payload:
        raise Hold30RandomControlError("random intent trace does not reconstruct exactly")
    for name in (
        "target_logits",
        "gate",
        "entry_scores",
        "hazard_residual",
        "exposure_residual",
    ):
        left = getattr(expected, name)
        right = getattr(trace, name)
        if (left is None) != (right is None):
            raise Hold30RandomControlError("random intent field presence changed")
        if left is not None and not torch.equal(left, right.detach().to(device="cpu")):
            raise Hold30RandomControlError(f"random intent field {name} changed")


class _PrecomputedIntentPolicy:
    def __init__(
        self,
        intent_trace: Hold30RandomIntentTrace,
        decision_mask: torch.Tensor,
    ) -> None:
        self.intent_trace = intent_trace
        self.decision_mask = decision_mask
        self.position = 0

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del state_t, prev_weights, age_summaries
        if self.position >= self.intent_trace.decision_count:
            raise Hold30RandomControlError("runtime requested too many random intents")
        if not torch.equal(available, self.decision_mask[self.position]):
            raise Hold30RandomControlError("runtime decision mask changed after generation")
        intent = self.intent_trace.intent_at(self.position)
        self.position += 1
        return intent


def _intent_payload(intent: Hold30Intent) -> dict[str, str]:
    return {
        name: _tensor_sha256(value)
        for name in (
            "target_logits",
            "gate",
            "entry_scores",
            "hazard_residual",
            "exposure_residual",
        )
        if (value := getattr(intent, name)) is not None
    }


def _trace_economic_sha256(trace: Hold30CanonicalTrace) -> str:
    payload = {
        "boundary_weights": _stack_sha256(
            [state.ledger.weights for state in trace.boundary_states]
        ),
        "boundary_economic_age": _stack_sha256(
            [state.ledger.economic_value for state in trace.boundary_states]
        ),
        "boundary_retention_age": _stack_sha256(
            [state.ledger.retention_units for state in trace.boundary_states]
        ),
        "raw_intents": [_intent_payload(row.raw_intent) for row in trace.transitions],
        "pre_cost_weights": _stack_sha256(
            [row.pre_cost_weights for row in trace.transitions]
        ),
        "filled_deltas": _stack_sha256([row.filled_delta for row in trace.transitions]),
        "net_returns": _stack_sha256([row.net_return for row in trace.transitions]),
        "benchmark_net_returns": _stack_sha256(
            [row.benchmark_net_return for row in trace.transitions]
        ),
        "turnover": {
            cause.value: _stack_sha256(
                [row.turnover_by_cause[cause] for row in trace.transitions]
            )
            for cause in TurnoverCause
        },
    }
    return _payload_sha256(payload)


def _require_same_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: float = _FEASIBILITY_TOLERANCE,
) -> None:
    if tuple(actual.shape) != tuple(expected.shape) or not bool(
        torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
    ):
        raise Hold30RandomControlError(f"hard feasibility failed: {name}")


def audit_hold30_random_trace(
    identity: Hold30RandomBankIdentity,
    intent_trace: Hold30RandomIntentTrace,
    sequence: Hold30Sequence,
    trace: Hold30CanonicalTrace,
) -> dict[str, Any]:
    """Verify exact intent binding and every runtime hard portfolio constraint."""

    decision_mask = sequence.decision_available[:-1]
    verify_hold30_random_intents(identity, intent_trace, decision_mask)
    expected_decisions = identity.decision_count
    if len(trace.transitions) != expected_decisions:
        raise Hold30RandomControlError("random trace decision count changed")
    if len(trace.boundary_states) != expected_decisions + 1:
        raise Hold30RandomControlError("random trace boundary count changed")
    if sequence.axis_id != identity.axis_id:
        raise Hold30RandomControlError("random trace axis differs from its bank")
    expected_shape = (
        identity.decision_count,
        identity.batch_size,
        identity.asset_count,
    )
    if tuple(decision_mask.shape) != expected_shape:
        raise Hold30RandomControlError("runtime sequence shape differs from its bank")
    if _tensor_sha256(decision_mask) != identity.decision_mask_sha256:
        raise Hold30RandomControlError("runtime decision mask differs from its bank")
    expected_dtype = torch.float32 if identity.dtype == "float32" else torch.float64
    if sequence.asset_returns.dtype != expected_dtype:
        raise Hold30RandomControlError("runtime dtype differs from its bank")
    if not bool(
        torch.equal(
            sequence.cost_rates,
            torch.full_like(sequence.cost_rates, 20.0 / 10_000.0),
        )
    ):
        raise Hold30RandomControlError("random controls require the canonical exact 20-bp trace")

    cash = identity.cash_index
    risky = torch.ones(identity.asset_count, dtype=torch.bool)
    risky[cash] = False
    for row_index, transition in enumerate(trace.transitions):
        if _intent_payload(transition.raw_intent) != _intent_payload(
            intent_trace.intent_at(row_index)
        ):
            raise Hold30RandomControlError("runtime raw intent differs from bank intent")
        fill = row_index + 1
        state_before = trace.boundary_states[row_index]
        state_after = trace.boundary_states[fill]
        state_before.ledger.assert_reconciles(state_before.ledger.weights)
        state_after.ledger.assert_reconciles(state_after.ledger.weights)
        _require_same_tensor(
            "decision book",
            transition.decision_weights,
            state_before.ledger.weights,
        )
        _require_same_tensor(
            "filled delta",
            transition.filled_delta,
            transition.pre_cost_weights - transition.risk_repaired_weights,
        )
        _require_same_tensor(
            "post-cost normalized book",
            transition.post_cost_weights,
            transition.pre_cost_weights,
        )
        weights = transition.pre_cost_weights
        if bool((weights < -_FEASIBILITY_TOLERANCE).any()) or not bool(
            torch.allclose(
                weights.sum(-1),
                torch.ones_like(weights.sum(-1)),
                atol=_FEASIBILITY_TOLERANCE,
                rtol=_FEASIBILITY_TOLERANCE,
            )
        ):
            raise Hold30RandomControlError("hard feasibility failed: long-only simplex")
        permitted = sequence.fill_membership[fill] & sequence.fill_availability[fill]
        prohibited_risky = (~permitted) & risky.to(device=permitted.device).view(1, -1)
        if bool((weights.masked_select(prohibited_risky).abs() > _FEASIBILITY_TOLERANCE).any()):
            raise Hold30RandomControlError("hard feasibility failed: fill availability")
        cap = torch.minimum(
            sequence.risk_asset_caps[fill],
            weights.new_full(weights.shape, HOLD30_MAX_STOCK_WEIGHT),
        )
        if bool(
            (
                weights[:, risky.to(device=weights.device)]
                - cap[:, risky.to(device=weights.device)]
                > _FEASIBILITY_TOLERANCE
            ).any()
        ):
            raise Hold30RandomControlError("hard feasibility failed: per-name cap")
        risky_gross = weights[:, risky.to(device=weights.device)].sum(-1)
        gross_limit = torch.minimum(
            sequence.risk_gross_max[fill],
            torch.ones_like(sequence.risk_gross_max[fill]),
        )
        if bool((risky_gross - gross_limit > _FEASIBILITY_TOLERANCE).any()):
            raise Hold30RandomControlError("hard feasibility failed: risky gross")
        discretionary = transition.turnover_by_cause[TurnoverCause.DISCRETIONARY]
        if bool(
            (
                discretionary
                > HOLD30_MAX_DISCRETIONARY_TURNOVER + _FEASIBILITY_TOLERANCE
            ).any()
        ):
            raise Hold30RandomControlError("hard feasibility failed: turnover ceiling")
        if bool((transition.projection_distance > _FEASIBILITY_TOLERANCE).any()):
            raise Hold30RandomControlError("hard feasibility failed: safety projection")
        for cause in (TurnoverCause.STARTUP, TurnoverCause.TERMINAL):
            if bool(
                (
                    transition.turnover_by_cause[cause].abs()
                    > _FEASIBILITY_TOLERANCE
                ).any()
            ):
                raise Hold30RandomControlError(
                    f"hard feasibility failed: forbidden {cause.value} turnover"
                )

    payload = {
        "schema": HOLD30_HARD_FEASIBILITY_SCHEMA,
        "schema_version": 1,
        "bank_identity_sha256": identity.receipt_sha256,
        "replicate_id": intent_trace.replicate_id,
        "intent_receipt_sha256": intent_trace.receipt_sha256,
        "trace_sha256": _trace_economic_sha256(trace),
        "sequence_hard_input_sha256s": {
            "decision_available": _tensor_sha256(sequence.decision_available),
            "fill_membership": _tensor_sha256(sequence.fill_membership),
            "fill_availability": _tensor_sha256(sequence.fill_availability),
            "risk_asset_caps": _tensor_sha256(sequence.risk_asset_caps),
            "risk_gross_max": _tensor_sha256(sequence.risk_gross_max),
            "cost_rates": _tensor_sha256(sequence.cost_rates),
            "entry_tracking_mask": _tensor_sha256(sequence.entry_tracking_mask),
        },
        "hard_contract": {
            "canonical_cost_bps": 20,
            "max_stock_weight": HOLD30_MAX_STOCK_WEIGHT,
            "max_discretionary_turnover": HOLD30_MAX_DISCRETIONARY_TURNOVER,
            "reconciliation_tolerance": _FEASIBILITY_TOLERANCE,
            "continuing_no_terminal_liquidation": True,
            "package_owned_builder": identity.builder_receipt_sha256,
            "checks": [
                "intent_exact_reconstruction",
                "ledger_conservation",
                "long_only_simplex",
                "fill_membership_and_availability",
                "per_name_cap",
                "risky_gross_limit",
                "discretionary_turnover_ceiling",
                "safety_projection_identity",
                "no_startup_or_terminal_turnover",
            ],
        },
        "metric_scope": {
            "hard_audit_is_non_gating_for_holding_metrics": True,
            "holding_and_matching_gate_owner": "build_hold30_c8_profile",
            "score_and_holding_masks_required_there": True,
            "entry_tracking_mask_is_bound_above": True,
        },
        "hard_feasible": True,
        "failures": [],
    }
    return _seal_receipt(payload)


@dataclass(frozen=True, slots=True)
class Hold30RandomControlRun:
    intent_trace: Hold30RandomIntentTrace
    trace: Hold30CanonicalTrace
    hard_feasibility_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        _verify_self_hash(
            self.hard_feasibility_receipt,
            schema=HOLD30_HARD_FEASIBILITY_SCHEMA,
        )
        if (
            self.hard_feasibility_receipt.get("replicate_id")
            != self.intent_trace.replicate_id
            or self.hard_feasibility_receipt.get("intent_receipt_sha256")
            != self.intent_trace.receipt_sha256
            or self.hard_feasibility_receipt.get("trace_sha256")
            != _trace_economic_sha256(self.trace)
            or self.hard_feasibility_receipt.get("hard_feasible") is not True
            or self.hard_feasibility_receipt.get("failures") != []
        ):
            raise Hold30RandomControlError(
                "random-control run does not match its hard-feasibility receipt"
            )

    @property
    def trace_sha256(self) -> str:
        return str(self.hard_feasibility_receipt["trace_sha256"])


def run_hold30_random_control(
    identity: Hold30RandomBankIdentity,
    replicate_id: int,
    sequence: Hold30Sequence,
    roles: Hold30CreditRoles,
) -> Hold30RandomControlRun:
    """Run one random intent path through the exact H1/H2 closed-loop runtime."""

    if roles.n_positions != sequence.n_positions:
        raise Hold30RandomControlError("credit roles do not match the control chronology")
    mask = sequence.decision_available[:-1]
    intent_trace = generate_hold30_random_intents(
        identity,
        replicate_id,
        mask,
        device=sequence.asset_returns.device,
    )
    policy = _PrecomputedIntentPolicy(intent_trace, mask)
    runtime = Hold30ChronologicalRuntime(identity.mechanism)
    with torch.no_grad():
        trace, _ = runtime.canonical_pass(policy, sequence, roles)
    if policy.position != identity.decision_count:
        raise Hold30RandomControlError("runtime did not consume the complete intent trace")
    receipt = audit_hold30_random_trace(identity, intent_trace, sequence, trace)
    return Hold30RandomControlRun(intent_trace, trace, receipt)


@dataclass(frozen=True, slots=True)
class Hold30PointInTimeAllocation:
    """One optional sector/liquidity allocation included in C8 matching."""

    name: Literal["sector", "liquidity"]
    group_ids: tuple[str, ...]
    values: tuple[float, ...]
    tolerance: float
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.name not in {"sector", "liquidity"}:
            raise Hold30RandomControlError("allocation name must be sector or liquidity")
        if (
            not isinstance(self.group_ids, tuple)
            or not self.group_ids
            or len(set(self.group_ids)) != len(self.group_ids)
            or any(not isinstance(value, str) or not value for value in self.group_ids)
        ):
            raise Hold30RandomControlError("allocation group_ids must be unique strings")
        if len(self.values) != len(self.group_ids):
            raise Hold30RandomControlError("allocation values do not match group_ids")
        for index, value in enumerate(self.values):
            _require_real(f"allocation[{index}]", value, nonnegative=True)
        if not 0.0 < _require_real("allocation tolerance", self.tolerance) <= 1.0:
            raise Hold30RandomControlError("allocation tolerance must lie in (0, 1]")
        _require_digest("allocation source_receipt_sha256", self.source_receipt_sha256)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group_ids": list(self.group_ids),
            "values": list(self.values),
            "tolerance_absolute": self.tolerance,
            "source_receipt_sha256": self.source_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class Hold30ClassificationInput:
    """Receipt-bound point-in-time one-hot classifications for profile building."""

    name: Literal["sector", "liquidity"]
    group_ids: tuple[str, ...]
    assignment: torch.Tensor
    tolerance: float
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.name not in {"sector", "liquidity"}:
            raise Hold30RandomControlError("classification name must be sector or liquidity")
        if not self.group_ids or len(set(self.group_ids)) != len(self.group_ids):
            raise Hold30RandomControlError("classification group_ids must be unique")
        if (
            not isinstance(self.assignment, torch.Tensor)
            or self.assignment.dtype != torch.bool
            or self.assignment.ndim != 4
            or self.assignment.shape[-1] != len(self.group_ids)
        ):
            raise Hold30RandomControlError(
                "classification assignment must be boolean [decision,batch,asset,group]"
            )
        if not 0.0 < _require_real("classification tolerance", self.tolerance) <= 1.0:
            raise Hold30RandomControlError("classification tolerance must lie in (0, 1]")
        _require_digest("classification source_receipt_sha256", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class Hold30C8Profile:
    """Allowlisted matching statistics; performance levels cannot enter this type."""

    bank_identity_sha256: str
    source_trace_sha256: str
    hard_feasibility_receipt_sha256: str
    return_covariance_receipt_sha256: str
    classification_availability_receipt_sha256: str
    score_mask_sha256: str
    holding_mask_sha256: str
    bank_id: int | None
    discretionary_turnover: float
    risky_exposure: float
    beta: float | None
    tracking_error: float | None
    hhi: float
    median_sale_age: float | None
    survival_30: float | None
    sold_notional: float
    age30_mass_at_risk: float
    allocations: tuple[Hold30PointInTimeAllocation, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "bank_identity_sha256",
            "source_trace_sha256",
            "hard_feasibility_receipt_sha256",
            "return_covariance_receipt_sha256",
            "classification_availability_receipt_sha256",
            "score_mask_sha256",
            "holding_mask_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.bank_id is not None:
            _require_replicate_id(self.bank_id)
        nullable = {"beta", "tracking_error", "median_sale_age", "survival_30"}
        for name in _METRIC_ORDER:
            raw = getattr(self, name)
            if raw is None and name in nullable:
                continue
            value = _require_real(name, raw)
            if name != "beta" and value < 0.0:
                raise Hold30RandomControlError(f"{name} cannot be negative")
        for name in ("sold_notional", "age30_mass_at_risk"):
            _require_real(name, getattr(self, name), nonnegative=True)
        if not 0.0 <= self.risky_exposure <= 1.0 + _FEASIBILITY_TOLERANCE:
            raise Hold30RandomControlError("risky_exposure lies outside [0, 1]")
        if self.survival_30 is not None and not (
            0.0 <= self.survival_30 <= 1.0 + _FEASIBILITY_TOLERANCE
        ):
            raise Hold30RandomControlError("survival_30 lies outside [0, 1]")
        names = tuple(value.name for value in self.allocations)
        if names != tuple(sorted(set(names))):
            raise Hold30RandomControlError("allocations must be unique and sorted by name")

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": HOLD30_C8_PROFILE_SCHEMA,
            "schema_version": 1,
            "bank_identity_sha256": self.bank_identity_sha256,
            "source_trace_sha256": self.source_trace_sha256,
            "hard_feasibility_receipt_sha256": self.hard_feasibility_receipt_sha256,
            "return_covariance_receipt_sha256": self.return_covariance_receipt_sha256,
            "classification_availability_receipt_sha256": (
                self.classification_availability_receipt_sha256
            ),
            "score_mask_sha256": self.score_mask_sha256,
            "holding_mask_sha256": self.holding_mask_sha256,
            "bank_id": self.bank_id,
            "allowlisted_matching_fields": {
                name: getattr(self, name) for name in _METRIC_ORDER
            },
            "estimability_fields": {
                "sold_notional": self.sold_notional,
                "age30_mass_at_risk": self.age30_mass_at_risk,
            },
            "point_in_time_allocations": [value.payload for value in self.allocations],
            "forbidden_fields_absent": [
                "mean_return",
                "terminal_wealth",
                "active_return",
                "information_ratio",
                "drawdown",
                "p_values",
                "performance_rank",
            ],
            "metric_conventions": {
                "turnover": "mean_scored_discretionary_one_way",
                "risky_exposure": "mean_scored_postfill_risky_weight",
                "beta": "sample_covariance_over_sample_benchmark_variance",
                "tracking_error": "annualized_sample_std_active_daily_return_sqrt252",
                "hhi": "mean_scored_sum_squared_absolute_risky_weights",
                "median_sale_age": "return_neutral_discretionary_sold_entry_units",
                "survival_30": (
                    "product_age1_through_30_discretionary_hazards_"
                    "with_forced_exit_competing_risk_censoring"
                ),
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.payload)


def _masked_transition_tensors(
    trace: Hold30CanonicalTrace,
    mask: torch.Tensor,
    field: str,
) -> torch.Tensor:
    values = torch.stack([getattr(row, field) for row in trace.transitions])
    return values[mask.to(device=values.device)]


def build_hold30_c8_profile(
    identity: Hold30RandomBankIdentity,
    trace: Hold30CanonicalTrace,
    *,
    hard_feasibility_receipt_sha256: str,
    return_covariance_receipt_sha256: str,
    classification_availability_receipt_sha256: str,
    score_mask: torch.Tensor,
    holding_mask: torch.Tensor,
    bank_id: int | None = None,
    classifications: Sequence[Hold30ClassificationInput] = (),
) -> Hold30C8Profile:
    """Compute only the fields permitted by the frozen C8 matcher.

    ``score_mask`` owns risk/covariance fields. ``holding_mask`` explicitly
    binds the score-origin support rows used for sale/survival evidence.  Their
    separation prevents an evaluator from silently dropping right-support.
    """

    decisions = len(trace.transitions)
    if decisions != identity.decision_count:
        raise Hold30RandomControlError("profile trace differs from its bank chronology")
    for name, mask in (("score_mask", score_mask), ("holding_mask", holding_mask)):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (decisions,)
            or not bool(mask.any())
        ):
            raise Hold30RandomControlError(f"{name} must select at least one decision")
    _require_digest(
        "hard_feasibility_receipt_sha256",
        hard_feasibility_receipt_sha256,
    )
    _require_digest(
        "return_covariance_receipt_sha256",
        return_covariance_receipt_sha256,
    )
    _require_digest(
        "classification_availability_receipt_sha256",
        classification_availability_receipt_sha256,
    )
    if (
        classification_availability_receipt_sha256
        != identity.classification_availability_receipt_sha256
    ):
        raise Hold30RandomControlError(
            "classification availability does not match the bank identity"
        )
    if bank_id is not None:
        _require_replicate_id(bank_id)

    scored_net = _masked_transition_tensors(trace, score_mask, "net_return")
    scored_benchmark = _masked_transition_tensors(
        trace,
        score_mask,
        "benchmark_net_return",
    )
    flat_policy = scored_net.reshape(-1).to(dtype=torch.float64)
    flat_benchmark = scored_benchmark.reshape(-1).to(dtype=torch.float64)
    beta: float | None = None
    tracking_error: float | None = None
    if flat_policy.numel() >= 2:
        centered_policy = flat_policy - flat_policy.mean()
        centered_benchmark = flat_benchmark - flat_benchmark.mean()
        benchmark_ss = (centered_benchmark * centered_benchmark).sum()
        if bool(benchmark_ss > 0.0):
            beta = float((centered_policy * centered_benchmark).sum() / benchmark_ss)
        active = flat_policy - flat_benchmark
        tracking_error = float(active.std(unbiased=True) * math.sqrt(252.0))

    scored_books = _masked_transition_tensors(trace, score_mask, "pre_cost_weights")
    cash = trace.boundary_states[0].ledger.cash_index
    risky_books = scored_books.clone()
    risky_books[..., cash] = 0.0
    risky_exposure = float(risky_books.sum(-1).mean())
    hhi = float((risky_books * risky_books).sum(-1).mean())
    discretionary_turnover = float(
        torch.stack(
            [
                row.turnover_by_cause[TurnoverCause.DISCRETIONARY]
                for row in trace.transitions
            ]
        )[score_mask.to(device=trace.transitions[0].decision_weights.device)]
        .mean()
        .to(dtype=torch.float64)
    )

    sold_value_by_age = torch.zeros(61, dtype=torch.float64)
    sold_units_by_age = torch.zeros(61, dtype=torch.float64)
    at_risk_by_age = torch.zeros(61, dtype=torch.float64)
    for selected, transition in zip(holding_mask.tolist(), trace.transitions, strict=True):
        if not selected:
            continue
        accounting = transition.accounting_by_cause[TurnoverCause.DISCRETIONARY]
        sold_value = accounting.sold_value_by_age.detach().to(device="cpu", dtype=torch.float64)
        sold_units = accounting.sold_units_by_age.detach().to(device="cpu", dtype=torch.float64)
        risk_units = transition.retention_units_after_forced.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        sold_value[:, cash] = 0.0
        sold_units[:, cash] = 0.0
        risk_units[:, cash] = 0.0
        sold_value_by_age += sold_value.sum(dim=(0, 1))
        sold_units_by_age += sold_units.sum(dim=(0, 1))
        at_risk_by_age += risk_units.sum(dim=(0, 1))
    sold_notional = float(sold_value_by_age.sum() / identity.batch_size)
    total_sold_units = sold_units_by_age.sum()
    median_sale_age: float | None = None
    if bool(total_sold_units > 0.0):
        threshold = total_sold_units / 2.0
        median_sale_age = float(
            torch.nonzero(
                sold_units_by_age.cumsum(0) >= threshold,
                as_tuple=False,
            )[0, 0]
        )
    age30_mass_at_risk = float(at_risk_by_age[30] / identity.batch_size)
    survival_30: float | None = None
    # New fills enter age zero after the discretionary stage.  Their first
    # genuine risk set is age one at the next fill, so S(30) is the product of
    # hazards for ages 1..30 (not an imputed age-zero denominator).
    risk = at_risk_by_age[1:31]
    deaths = sold_units_by_age[1:31]
    if bool((risk > 0.0).all()) and bool((deaths <= risk + 1e-10).all()):
        hazards = (deaths / risk).clamp(0.0, 1.0)
        survival_30 = float(torch.prod(1.0 - hazards))

    allocations: list[Hold30PointInTimeAllocation] = []
    seen: set[str] = set()
    for classification in sorted(classifications, key=lambda value: value.name):
        if classification.name in seen:
            raise Hold30RandomControlError("classification inputs contain a duplicate name")
        seen.add(classification.name)
        expected_shape = (
            decisions,
            identity.batch_size,
            identity.asset_count,
            len(classification.group_ids),
        )
        if tuple(classification.assignment.shape) != expected_shape:
            raise Hold30RandomControlError(
                f"{classification.name} assignment does not match the trace"
            )
        assignment = classification.assignment[
            score_mask.to(device=classification.assignment.device)
        ].to(device=scored_books.device)
        if bool(assignment[..., cash, :].any()):
            raise Hold30RandomControlError("CASH cannot carry a sector/liquidity class")
        class_count = assignment.sum(-1)
        held = risky_books > _FEASIBILITY_TOLERANCE
        held[..., cash] = False
        if bool((class_count.masked_select(held) != 1).any()):
            raise Hold30RandomControlError(
                f"held risky assets lack unique PIT {classification.name} classes"
            )
        group_allocations = (
            risky_books.unsqueeze(-1) * assignment.to(dtype=risky_books.dtype)
        ).sum(dim=-2)
        mean_allocations = group_allocations.mean(dim=(0, 1))
        allocations.append(
            Hold30PointInTimeAllocation(
                classification.name,
                classification.group_ids,
                tuple(float(value) for value in mean_allocations),
                classification.tolerance,
                classification.source_receipt_sha256,
            )
        )
    if tuple(value.name for value in allocations) != identity.classification_dimensions:
        raise Hold30RandomControlError(
            "declared PIT classification dimension was dropped or added"
        )

    return Hold30C8Profile(
        bank_identity_sha256=identity.receipt_sha256,
        source_trace_sha256=_trace_economic_sha256(trace),
        hard_feasibility_receipt_sha256=hard_feasibility_receipt_sha256,
        return_covariance_receipt_sha256=return_covariance_receipt_sha256,
        classification_availability_receipt_sha256=(
            classification_availability_receipt_sha256
        ),
        score_mask_sha256=_tensor_sha256(score_mask),
        holding_mask_sha256=_tensor_sha256(holding_mask),
        bank_id=bank_id,
        discretionary_turnover=discretionary_turnover,
        risky_exposure=risky_exposure,
        beta=beta,
        tracking_error=tracking_error,
        hhi=hhi,
        median_sale_age=median_sale_age,
        survival_30=survival_30,
        sold_notional=sold_notional,
        age30_mass_at_risk=age30_mass_at_risk,
        allocations=tuple(allocations),
    )


@dataclass(frozen=True, slots=True)
class Hold30RandomBankEntry:
    """Receipt references for one generated, closed-loop bank trace."""

    replicate_id: int
    rng_key_sha256: str
    intent_receipt_sha256: str
    trace_sha256: str
    hard_feasibility_receipt_sha256: str
    hard_feasibility_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_replicate_id(self.replicate_id)
        for name in (
            "rng_key_sha256",
            "intent_receipt_sha256",
            "trace_sha256",
            "hard_feasibility_receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not isinstance(self.hard_feasibility_failures, tuple)
            or tuple(sorted(set(self.hard_feasibility_failures)))
            != self.hard_feasibility_failures
            or any(not isinstance(value, str) or not value for value in self.hard_feasibility_failures)
        ):
            raise Hold30RandomControlError(
                "hard_feasibility_failures must be sorted unique nonempty strings"
            )

    @classmethod
    def from_run(cls, run: Hold30RandomControlRun) -> "Hold30RandomBankEntry":
        receipt = run.hard_feasibility_receipt
        _verify_self_hash(receipt, schema=HOLD30_HARD_FEASIBILITY_SCHEMA)
        return cls(
            run.intent_trace.replicate_id,
            run.intent_trace.rng_key_sha256,
            run.intent_trace.receipt_sha256,
            run.trace_sha256,
            str(receipt["receipt_sha256"]),
        )

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "replicate_id": self.replicate_id,
            "rng_key_sha256": self.rng_key_sha256,
            "intent_receipt_sha256": self.intent_receipt_sha256,
            "trace_sha256": self.trace_sha256,
            "hard_feasibility_receipt_sha256": self.hard_feasibility_receipt_sha256,
            "hard_feasibility_failures": list(self.hard_feasibility_failures),
        }


def build_hold30_random_bank_receipt(
    identity: Hold30RandomBankIdentity,
    entries: Sequence[Hold30RandomBankEntry],
) -> dict[str, Any]:
    """Bind the exact complete 8,192-ID closed-loop bank."""

    ordered = tuple(entries)
    ids = tuple(value.replicate_id for value in ordered)
    if ids != tuple(range(HOLD30_RANDOM_BANK_SIZE)):
        raise Hold30RandomControlError(
            "random-control bank entries must be exactly ordered IDs 0..8191"
        )
    for entry in ordered:
        if entry.rng_key_sha256 != hold30_random_key(identity, entry.replicate_id).hex():
            raise Hold30RandomControlError("bank entry RNG key does not reconstruct")
    payload = {
        "schema": HOLD30_RANDOM_BANK_SCHEMA,
        "schema_version": 1,
        "bank_identity": identity.payload,
        "bank_identity_sha256": identity.receipt_sha256,
        "bank_size": HOLD30_RANDOM_BANK_SIZE,
        "entries": [entry.payload for entry in ordered],
        "complete_id_range": [0, HOLD30_RANDOM_BANK_SIZE - 1],
    }
    return _seal_receipt(payload)


def verify_hold30_random_bank_receipt(
    identity: Hold30RandomBankIdentity,
    entries: Sequence[Hold30RandomBankEntry],
    receipt: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, schema=HOLD30_RANDOM_BANK_SCHEMA)
    expected = build_hold30_random_bank_receipt(identity, entries)
    if dict(receipt) != expected:
        raise Hold30RandomControlError("random-control bank receipt does not reconstruct")


def build_hold30_c7_receipt(
    identity: Hold30RandomBankIdentity,
    entries: Sequence[Hold30RandomBankEntry],
    bank_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exactly the first 64 bank IDs without outcome-dependent filtering."""

    verify_hold30_random_bank_receipt(identity, entries, bank_receipt)
    c7 = tuple(entries[:HOLD30_RANDOM_CONTROL_COUNT])
    if tuple(value.replicate_id for value in c7) != HOLD30_C7_IDS:
        raise Hold30RandomControlError("C7 is not exactly bank IDs 0..63")
    failures = [
        value.replicate_id for value in c7 if value.hard_feasibility_failures
    ]
    if failures:
        raise Hold30RandomControlError(
            f"C7 contains hard-infeasible bank IDs: {failures}"
        )
    return _seal_receipt(
        {
            "schema": HOLD30_C7_RECEIPT_SCHEMA,
            "schema_version": 1,
            "bank_identity_sha256": identity.receipt_sha256,
            "bank_receipt_sha256": bank_receipt["receipt_sha256"],
            "selection_rule": "first_64_stable_bank_ids_no_return_filter",
            "selected_ids": list(HOLD30_C7_IDS),
            "selected_entries": [value.payload for value in c7],
        }
    )


def verify_hold30_c7_receipt(
    identity: Hold30RandomBankIdentity,
    entries: Sequence[Hold30RandomBankEntry],
    bank_receipt: Mapping[str, Any],
    c7_receipt: Mapping[str, Any],
) -> None:
    _verify_self_hash(c7_receipt, schema=HOLD30_C7_RECEIPT_SCHEMA)
    expected = build_hold30_c7_receipt(identity, entries, bank_receipt)
    if dict(c7_receipt) != expected:
        raise Hold30RandomControlError("C7 receipt does not reconstruct")


def _estimability_reasons(profile: Hold30C8Profile) -> list[str]:
    reasons: list[str] = []
    if profile.sold_notional < HOLD30_C8_MIN_SOLD_NOTIONAL:
        reasons.append("estimability:sold_notional_below_0.10")
    if profile.age30_mass_at_risk <= 0.0:
        reasons.append("estimability:age30_mass_at_risk_nonpositive")
    for name in ("beta", "tracking_error", "median_sale_age", "survival_30"):
        if getattr(profile, name) is None:
            reasons.append(f"estimability:{name}_undefined")
    return reasons


def _relative_gap(
    candidate: float,
    target: float,
    tolerance: float,
) -> tuple[float, bool]:
    absolute = abs(candidate - target)
    if target == 0.0:
        return (0.0, True) if candidate == 0.0 else (math.inf, False)
    normalized = absolute / abs(target) / tolerance
    return normalized, normalized <= 1.0


def _absolute_gap(
    candidate: float,
    target: float,
    tolerance: float,
) -> tuple[float, bool]:
    normalized = abs(candidate - target) / tolerance
    return normalized, normalized <= 1.0


def _profile_match(
    target: Hold30C8Profile,
    candidate: Hold30C8Profile,
) -> tuple[list[str], float | None, dict[str, float | None]]:
    reasons: list[str] = []
    normalized: dict[str, float | None] = {}
    definitions = (
        (
            "discretionary_turnover",
            "relative",
            HOLD30_C8_TOLERANCES["discretionary_turnover_relative"],
        ),
        (
            "risky_exposure",
            "absolute",
            HOLD30_C8_TOLERANCES["risky_exposure_absolute"],
        ),
        ("beta", "absolute", HOLD30_C8_TOLERANCES["beta_absolute"]),
        (
            "tracking_error",
            "relative",
            HOLD30_C8_TOLERANCES["tracking_error_relative"],
        ),
        ("hhi", "relative", HOLD30_C8_TOLERANCES["hhi_relative"]),
        (
            "median_sale_age",
            "absolute",
            HOLD30_C8_TOLERANCES["median_sale_age_absolute_sessions"],
        ),
        (
            "survival_30",
            "absolute",
            HOLD30_C8_TOLERANCES["survival_30_absolute"],
        ),
    )
    for name, mode, tolerance in definitions:
        target_value = getattr(target, name)
        candidate_value = getattr(candidate, name)
        if target_value is None or candidate_value is None:
            normalized[name] = None
            reasons.append(f"match:{name}_undefined")
            continue
        if mode == "relative":
            gap, passed = _relative_gap(candidate_value, target_value, tolerance)
        else:
            gap, passed = _absolute_gap(candidate_value, target_value, tolerance)
        normalized[name] = None if not math.isfinite(gap) else gap
        if not passed:
            reasons.append(f"match:{name}")

    if (
        candidate.classification_availability_receipt_sha256
        != target.classification_availability_receipt_sha256
    ):
        reasons.append("match:classification_availability_receipt")
    if candidate.score_mask_sha256 != target.score_mask_sha256:
        reasons.append("match:score_mask")
    if candidate.holding_mask_sha256 != target.holding_mask_sha256:
        reasons.append("match:holding_mask")
    target_allocations = {value.name: value for value in target.allocations}
    candidate_allocations = {value.name: value for value in candidate.allocations}
    if set(candidate_allocations) != set(target_allocations):
        reasons.append("match:classification_dimensions")
    for name in sorted(set(target_allocations) | set(candidate_allocations)):
        left = target_allocations.get(name)
        right = candidate_allocations.get(name)
        if left is None or right is None:
            continue
        if (
            right.group_ids != left.group_ids
            or right.source_receipt_sha256 != left.source_receipt_sha256
            or right.tolerance != left.tolerance
        ):
            reasons.append(f"match:{name}_classification_contract")
            continue
        for group_id, target_value, candidate_value in zip(
            left.group_ids,
            left.values,
            right.values,
            strict=True,
        ):
            key = f"{name}:{group_id}"
            gap, passed = _absolute_gap(
                candidate_value,
                target_value,
                left.tolerance,
            )
            normalized[key] = gap
            if not passed:
                reasons.append(f"match:{key}")

    finite = [value for value in normalized.values() if value is not None]
    distance = None
    if len(finite) == len(normalized) and all(math.isfinite(value) for value in finite):
        distance = math.fsum(value * value for value in finite)
    return reasons, distance, normalized


def _validate_profile_binding(
    identity: Hold30RandomBankIdentity,
    profile: Hold30C8Profile,
    *,
    entry: Hold30RandomBankEntry | None,
) -> None:
    if profile.bank_identity_sha256 != identity.receipt_sha256:
        raise Hold30RandomControlError("C8 profile binds a different bank identity")
    if (
        profile.classification_availability_receipt_sha256
        != identity.classification_availability_receipt_sha256
        or tuple(value.name for value in profile.allocations)
        != identity.classification_dimensions
    ):
        raise Hold30RandomControlError(
            "C8 profile dropped or changed a declared PIT classification dimension"
        )
    if entry is None:
        if profile.bank_id is not None:
            raise Hold30RandomControlError("learned target profile cannot have a bank ID")
        return
    if profile.bank_id != entry.replicate_id:
        raise Hold30RandomControlError("candidate profile binds a different bank ID")
    if profile.source_trace_sha256 != entry.trace_sha256:
        raise Hold30RandomControlError("candidate profile binds a different trace")
    if (
        profile.hard_feasibility_receipt_sha256
        != entry.hard_feasibility_receipt_sha256
    ):
        raise Hold30RandomControlError(
            "candidate profile binds a different hard-feasibility receipt"
        )


def _c8_receipt_payload(
    identity: Hold30RandomBankIdentity,
    learned_profile: Hold30C8Profile,
    bank_receipt: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    status: str,
    target_reasons: Sequence[str],
    selected_ids: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": HOLD30_C8_SELECTION_SCHEMA,
        "schema_version": 1,
        "status": status,
        "bank_identity_sha256": identity.receipt_sha256,
        "bank_receipt_sha256": bank_receipt["receipt_sha256"],
        "learned_profile": learned_profile.payload,
        "learned_profile_sha256": learned_profile.receipt_sha256,
        "target_precondition_reasons": list(target_reasons),
        "candidate_id_range": [
            HOLD30_RANDOM_CONTROL_COUNT,
            HOLD30_RANDOM_BANK_SIZE - 1,
        ],
        "matching_field_allowlist": list(_METRIC_ORDER),
        "forbidden_selection_fields": [
            "mean_return",
            "terminal_wealth",
            "active_return",
            "information_ratio",
            "drawdown",
            "p_values",
            "performance_rank",
        ],
        "tolerances": dict(HOLD30_C8_TOLERANCES),
        "minimum_sold_notional": HOLD30_C8_MIN_SOLD_NOTIONAL,
        "zero_target_relative_rule": "candidate_must_equal_exact_zero_else_reject",
        "distance_rule": "sum_squared_tolerance_normalized_gaps",
        "ranking_rule": "filter_all_tolerances_then_ascending_distance_then_bank_id",
        "candidate_rows": list(candidate_rows),
        "selected_ids": list(selected_ids),
        "selected_count": len(selected_ids),
    }


def _select_hold30_c8(
    identity: Hold30RandomBankIdentity,
    learned_profile: Hold30C8Profile,
    entries: Sequence[Hold30RandomBankEntry],
    bank_receipt: Mapping[str, Any],
    candidate_profiles: Mapping[int, Hold30C8Profile],
) -> dict[str, Any]:
    verify_hold30_random_bank_receipt(identity, entries, bank_receipt)
    _validate_profile_binding(identity, learned_profile, entry=None)
    if set(candidate_profiles) - set(HOLD30_C8_CANDIDATE_IDS):
        raise Hold30RandomControlError("C8 profiles contain non-candidate bank IDs")
    target_reasons = _estimability_reasons(learned_profile)
    if target_reasons:
        return _seal_receipt(
            _c8_receipt_payload(
                identity,
                learned_profile,
                bank_receipt,
                (),
                status="target_precondition_failed",
                target_reasons=target_reasons,
                selected_ids=(),
            )
        )

    entry_by_id = {value.replicate_id: value for value in entries}
    rows: list[dict[str, Any]] = []
    feasible: list[tuple[float, int, int]] = []
    for replicate_id in HOLD30_C8_CANDIDATE_IDS:
        entry = entry_by_id[replicate_id]
        profile = candidate_profiles.get(replicate_id)
        reasons = [
            f"hard_feasibility:{reason}"
            for reason in entry.hard_feasibility_failures
        ]
        distance: float | None = None
        normalized: dict[str, float | None] = {}
        if profile is None:
            reasons.append("profile_missing")
        else:
            _validate_profile_binding(identity, profile, entry=entry)
            if (
                profile.return_covariance_receipt_sha256
                != learned_profile.return_covariance_receipt_sha256
            ):
                reasons.append("match:return_covariance_receipt")
            reasons.extend(_estimability_reasons(profile))
            match_reasons, distance, normalized = _profile_match(
                learned_profile,
                profile,
            )
            reasons.extend(match_reasons)
        reasons = list(dict.fromkeys(reasons))
        row = {
            "replicate_id": replicate_id,
            "entry": entry.payload,
            "profile": None if profile is None else profile.payload,
            "profile_sha256": None if profile is None else profile.receipt_sha256,
            "rejected_reasons": reasons,
            "normalized_gaps": normalized,
            "distance": distance,
            "distance_rank": None,
            "selected": False,
        }
        row_index = len(rows)
        rows.append(row)
        if not reasons and distance is not None:
            feasible.append((distance, replicate_id, row_index))
    feasible.sort(key=lambda value: (value[0], value[1]))
    for rank, (_, _, row_index) in enumerate(feasible):
        rows[row_index]["distance_rank"] = rank
    selected = feasible[:HOLD30_RANDOM_CONTROL_COUNT]
    selected_ids = [replicate_id for _, replicate_id, _ in selected]
    for _, _, row_index in selected:
        rows[row_index]["selected"] = True
    status = (
        "passed"
        if len(selected_ids) == HOLD30_RANDOM_CONTROL_COUNT
        else "insufficient_feasible_controls"
    )
    return _seal_receipt(
        _c8_receipt_payload(
            identity,
            learned_profile,
            bank_receipt,
            rows,
            status=status,
            target_reasons=(),
            selected_ids=selected_ids,
        )
    )


def select_hold30_c8_controls(
    identity: Hold30RandomBankIdentity,
    learned_profile: Hold30C8Profile,
    entries: Sequence[Hold30RandomBankEntry],
    bank_receipt: Mapping[str, Any],
    candidate_profiles: Mapping[int, Hold30C8Profile],
) -> dict[str, Any]:
    """Filter and deterministically rank C8, or raise with a sealed failure receipt."""

    receipt = _select_hold30_c8(
        identity,
        learned_profile,
        entries,
        bank_receipt,
        candidate_profiles,
    )
    if receipt["status"] != "passed":
        raise Hold30C8GateError(
            f"C8 matched-control gate failed: {receipt['status']}",
            receipt,
        )
    return receipt


def verify_hold30_c8_selection_receipt(
    identity: Hold30RandomBankIdentity,
    learned_profile: Hold30C8Profile,
    entries: Sequence[Hold30RandomBankEntry],
    bank_receipt: Mapping[str, Any],
    candidate_profiles: Mapping[int, Hold30C8Profile],
    receipt: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, schema=HOLD30_C8_SELECTION_SCHEMA)
    expected = _select_hold30_c8(
        identity,
        learned_profile,
        entries,
        bank_receipt,
        candidate_profiles,
    )
    if dict(receipt) != expected:
        raise Hold30RandomControlError("C8 selection receipt does not reconstruct")


@dataclass(frozen=True, slots=True)
class Hold30FoldRandomControlReceipts:
    identity: Hold30RandomBankIdentity
    bank_receipt: Mapping[str, Any]
    c7_receipt: Mapping[str, Any]
    c8_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        _verify_self_hash(self.bank_receipt, schema=HOLD30_RANDOM_BANK_SCHEMA)
        _verify_self_hash(self.c7_receipt, schema=HOLD30_C7_RECEIPT_SCHEMA)
        _verify_self_hash(self.c8_receipt, schema=HOLD30_C8_SELECTION_SCHEMA)
        identity_sha = self.identity.receipt_sha256
        for name, receipt in (
            ("bank", self.bank_receipt),
            ("C7", self.c7_receipt),
            ("C8", self.c8_receipt),
        ):
            if receipt.get("bank_identity_sha256") != identity_sha:
                raise Hold30RandomControlError(
                    f"fold {name} receipt binds a different bank identity"
                )
        if self.bank_receipt.get("bank_identity") != self.identity.payload:
            raise Hold30RandomControlError("fold bank embeds a different identity payload")
        bank_size = self.bank_receipt.get("bank_size")
        bank_entries = self.bank_receipt.get("entries")
        if bank_size != HOLD30_RANDOM_BANK_SIZE:
            raise Hold30RandomControlError("fold bank size is not 8192")
        if (
            not isinstance(bank_entries, list)
            or len(bank_entries) != HOLD30_RANDOM_BANK_SIZE
            or [row.get("replicate_id") for row in bank_entries]
            != list(range(HOLD30_RANDOM_BANK_SIZE))
        ):
            raise Hold30RandomControlError("fold bank entries are not complete IDs 0..8191")
        if self.c7_receipt.get("bank_receipt_sha256") != self.bank_receipt.get(
            "receipt_sha256"
        ) or self.c8_receipt.get("bank_receipt_sha256") != self.bank_receipt.get(
            "receipt_sha256"
        ):
            raise Hold30RandomControlError("fold control receipt binds a different bank")
        if self.c7_receipt.get("selected_ids") != list(HOLD30_C7_IDS):
            raise Hold30RandomControlError("fold C7 receipt is not IDs 0..63")
        c7_entries = self.c7_receipt.get("selected_entries")
        if (
            not isinstance(c7_entries, list)
            or [row.get("replicate_id") for row in c7_entries] != list(HOLD30_C7_IDS)
        ):
            raise Hold30RandomControlError("fold C7 selected entries are incomplete")
        if self.c8_receipt.get("status") != "passed":
            raise Hold30RandomControlError("fold C8 selection did not pass")
        selected = self.c8_receipt.get("selected_ids")
        if (
            not isinstance(selected, list)
            or len(selected) != HOLD30_RANDOM_CONTROL_COUNT
            or len(set(selected)) != HOLD30_RANDOM_CONTROL_COUNT
            or any(value not in HOLD30_C8_CANDIDATE_IDS for value in selected)
        ):
            raise Hold30RandomControlError("fold C8 selected IDs are incomplete")
        rows = self.c8_receipt.get("candidate_rows")
        if (
            not isinstance(rows, list)
            or len(rows) != len(HOLD30_C8_CANDIDATE_IDS)
            or [row.get("replicate_id") for row in rows]
            != list(HOLD30_C8_CANDIDATE_IDS)
        ):
            raise Hold30RandomControlError("fold C8 candidate rows are incomplete")
        selected_rows = [row for row in rows if row.get("selected") is True]
        if any(
            isinstance(row.get("distance_rank"), bool)
            or not isinstance(row.get("distance_rank"), int)
            for row in selected_rows
        ):
            raise Hold30RandomControlError("fold C8 selected row lacks an integer rank")
        ranked = sorted(
            (
                (row["distance_rank"], row.get("replicate_id"))
                for row in selected_rows
            ),
            key=lambda value: value[0],
        )
        if ranked != list(enumerate(selected)):
            raise Hold30RandomControlError(
                "fold C8 selected IDs do not follow frozen distance ranks"
            )


def build_hold30_cross_fold_control_mapping(
    fold_receipts: Sequence[Hold30FoldRandomControlReceipts],
) -> dict[str, Any]:
    """Hash the complete C7/C8 aggregate-path map before wealth is computed."""

    ordered = tuple(sorted(fold_receipts, key=lambda value: value.identity.fold_index))
    if tuple(value.identity.fold_index for value in ordered) != tuple(range(6)):
        raise Hold30RandomControlError("cross-fold controls require folds exactly 0..5")
    protocols = {value.identity.protocol_generation for value in ordered}
    variants = {value.identity.variant_id for value in ordered}
    mechanisms = {value.identity.mechanism for value in ordered}
    if len(protocols) != 1 or len(variants) != 1 or len(mechanisms) != 1:
        raise Hold30RandomControlError(
            "cross-fold controls must share protocol, variant, and mechanism"
        )
    mapping: list[dict[str, Any]] = []
    for aggregate_path in range(HOLD30_RANDOM_CONTROL_COUNT):
        for fold in ordered:
            mapping.append(
                {
                    "control_id": "C7",
                    "aggregate_path": aggregate_path,
                    "fold": fold.identity.fold_index,
                    "bank_id": aggregate_path,
                    "distance_rank": None,
                }
            )
    for aggregate_path in range(HOLD30_RANDOM_CONTROL_COUNT):
        for fold in ordered:
            selected = fold.c8_receipt["selected_ids"]
            mapping.append(
                {
                    "control_id": "C8",
                    "aggregate_path": aggregate_path,
                    "fold": fold.identity.fold_index,
                    "bank_id": selected[aggregate_path],
                    "distance_rank": aggregate_path,
                }
            )
    payload = {
        "schema": HOLD30_CROSS_FOLD_MAPPING_SCHEMA,
        "schema_version": 1,
        "protocol_generation": ordered[0].identity.protocol_generation,
        "variant_id": ordered[0].identity.variant_id,
        "mechanism": ordered[0].identity.mechanism,
        "folds": list(range(6)),
        "fold_receipt_sha256s": [
            {
                "fold": value.identity.fold_index,
                "bank": value.bank_receipt["receipt_sha256"],
                "c7": value.c7_receipt["receipt_sha256"],
                "c8": value.c8_receipt["receipt_sha256"],
            }
            for value in ordered
        ],
        "mapping_rule": {
            "c7": "path_r_concatenates_bank_id_r_from_folds_0_through_5",
            "c8": "path_j_concatenates_fold_jth_distance_ranked_selected_id",
            "return_reranking": False,
        },
        "mapping": mapping,
        "mapping_sha256": _payload_sha256(mapping),
        "mapping_frozen_before_aggregate_wealth": True,
    }
    return _seal_receipt(payload)


def verify_hold30_cross_fold_control_mapping(
    fold_receipts: Sequence[Hold30FoldRandomControlReceipts],
    receipt: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, schema=HOLD30_CROSS_FOLD_MAPPING_SCHEMA)
    expected = build_hold30_cross_fold_control_mapping(fold_receipts)
    if dict(receipt) != expected:
        raise Hold30RandomControlError("cross-fold control mapping does not reconstruct")


__all__ = [
    "HOLD30_C7_IDS",
    "HOLD30_C8_CANDIDATE_IDS",
    "HOLD30_C8_MIN_SOLD_NOTIONAL",
    "HOLD30_C8_TOLERANCES",
    "HOLD30_RANDOM_BANK_SIZE",
    "HOLD30_RANDOM_CONTROL_COUNT",
    "HOLD30_RANDOM_H1_GATE_LOGIT_NOISES",
    "HOLD30_RANDOM_H1_GATE_MEANS",
    "HOLD30_RANDOM_H2_EXPOSURE_SCALES",
    "HOLD30_RANDOM_H2_HAZARD_CENTERS",
    "HOLD30_RANDOM_H2_HAZARD_SCALES",
    "HOLD30_RANDOM_RHOS",
    "HOLD30_RANDOM_SCORE_SCALES",
    "Hold30C8GateError",
    "Hold30C8Profile",
    "Hold30ClassificationInput",
    "Hold30FoldRandomControlReceipts",
    "Hold30PointInTimeAllocation",
    "Hold30RandomBankEntry",
    "Hold30RandomBankIdentity",
    "Hold30RandomControlError",
    "Hold30RandomControlRun",
    "Hold30RandomIntentTrace",
    "Hold30RandomParameters",
    "audit_hold30_random_trace",
    "build_hold30_c7_receipt",
    "build_hold30_c8_profile",
    "build_hold30_cross_fold_control_mapping",
    "build_hold30_random_bank_receipt",
    "generate_hold30_random_intents",
    "hold30_random_key",
    "run_hold30_random_control",
    "select_hold30_c8_controls",
    "verify_hold30_c7_receipt",
    "verify_hold30_c8_selection_receipt",
    "verify_hold30_cross_fold_control_mapping",
    "verify_hold30_random_bank_receipt",
    "verify_hold30_random_intents",
]
