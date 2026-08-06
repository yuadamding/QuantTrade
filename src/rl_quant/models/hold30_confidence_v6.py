"""Standalone M03R v6 confidence head for the no-alpha-head M02 control.

M02 intentionally keeps the common entry/hazard policy and therefore must not
instantiate the residual-alpha head bundle.  It nevertheless retains the v6
confidence-scaled active-risk mechanism.  This module implements only that
scalar mechanism and its governed two-stage lifecycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Literal

import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_DESIGN,
    M03R_DESIGN_ID,
    M03R_PROTOCOL_GENERATION,
    resolve_m03r_v6_setting,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationError,
    M03RConfidenceCalibrationManifest,
    apply_m03r_confidence_calibration,
    validate_m03r_confidence_calibration_manifest,
)

M03R_V6_STANDALONE_CONFIDENCE_HEAD_SCHEMA: Final[str] = (
    "rl-quant.m03r-v6-standalone-confidence-head-v1"
)
M03R_V6_FROZEN_POLICY_CONFIDENCE_BINDING_SCHEMA: Final[str] = (
    "rl-quant.m03r-v6-frozen-policy-confidence-binding-v1"
)
M03RV6ConfidenceLifecycleStage = Literal[
    "v6-training-uncalibrated",
    "v6-post-freeze-calibrated",
]
M03R_V6_CONFIDENCE_LIFECYCLE_STAGES: Final[tuple[str, str]] = (
    "v6-training-uncalibrated",
    "v6-post-freeze-calibrated",
)


class M03RV6StandaloneConfidenceError(ValueError):
    """The standalone M02 confidence route is unbound or malformed."""


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M03RV6StandaloneConfidenceError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def m03r_v6_policy_state_sha256(policy: nn.Module) -> str:
    """Content-address the exact loaded full-policy state without pickle bytes."""

    if not isinstance(policy, nn.Module):
        raise M03RV6StandaloneConfidenceError("policy must be a torch module")
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV6FrozenPolicyConfidenceBinding:
    """Receipt proving post-load state verification before confidence execution."""

    loaded_checkpoint_sha256: str
    loaded_policy_state_sha256: str
    calibration_manifest_sha256s: tuple[str, ...]
    calibration_fit_evidence_sha256s: tuple[str, ...]
    bound_confidence_head_count: int
    receipt_sha256: str
    schema: str = M03R_V6_FROZEN_POLICY_CONFIDENCE_BINDING_SCHEMA

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "loaded_checkpoint_sha256": self.loaded_checkpoint_sha256,
            "loaded_policy_state_sha256": self.loaded_policy_state_sha256,
            "calibration_manifest_sha256s": list(
                self.calibration_manifest_sha256s
            ),
            "calibration_fit_evidence_sha256s": list(
                self.calibration_fit_evidence_sha256s
            ),
            "bound_confidence_head_count": self.bound_confidence_head_count,
        }

    def __post_init__(self) -> None:
        if self.schema != M03R_V6_FROZEN_POLICY_CONFIDENCE_BINDING_SCHEMA:
            raise M03RV6StandaloneConfidenceError(
                "frozen-policy confidence binding schema drifted"
            )
        _require_sha256("loaded_checkpoint_sha256", self.loaded_checkpoint_sha256)
        _require_sha256(
            "loaded_policy_state_sha256", self.loaded_policy_state_sha256
        )
        if (
            not self.calibration_manifest_sha256s
            or len(self.calibration_manifest_sha256s)
            != self.bound_confidence_head_count
            or len(self.calibration_fit_evidence_sha256s)
            != self.bound_confidence_head_count
        ):
            raise M03RV6StandaloneConfidenceError(
                "every bound confidence head requires one manifest and fit receipt"
            )
        for name, values in (
            ("calibration_manifest_sha256s", self.calibration_manifest_sha256s),
            (
                "calibration_fit_evidence_sha256s",
                self.calibration_fit_evidence_sha256s,
            ),
        ):
            for value in values:
                _require_sha256(name, value)
        if (
            isinstance(self.bound_confidence_head_count, bool)
            or not isinstance(self.bound_confidence_head_count, int)
            or self.bound_confidence_head_count <= 0
        ):
            raise M03RV6StandaloneConfidenceError(
                "bound_confidence_head_count must be positive"
            )
        expected = hashlib.sha256(
            json.dumps(
                self.canonical_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if _require_sha256("receipt_sha256", self.receipt_sha256) != expected:
            raise M03RV6StandaloneConfidenceError(
                "frozen-policy confidence binding receipt hash mismatch"
            )


def bind_m03r_v6_frozen_policy_confidence(
    policy: nn.Module,
    *,
    loaded_checkpoint_sha256: str,
) -> M03RV6FrozenPolicyConfidenceBinding:
    """Verify the full post-load policy, then freeze its calibrated heads.

    The root must be the containing policy, not a confidence child module. This
    makes constructor/config claims insufficient: forward execution remains
    blocked until the exact loaded policy state matches the fit evidence.
    """

    if not isinstance(policy, nn.Module):
        raise M03RV6StandaloneConfidenceError("policy must be a torch module")
    if callable(
        getattr(policy, "_bind_m03r_v6_post_freeze_confidence_state", None)
    ):
        raise M03RV6StandaloneConfidenceError(
            "binding requires the containing full-policy module, not a child head"
        )
    checkpoint = _require_sha256(
        "loaded_checkpoint_sha256", loaded_checkpoint_sha256
    )
    state_digest = m03r_v6_policy_state_sha256(policy)
    identities: list[tuple[str, str]] = []
    bound_children: list[Any] = []
    for child in policy.modules():
        binder: Any = getattr(
            child,
            "_bind_m03r_v6_post_freeze_confidence_state",
            None,
        )
        if binder is None:
            continue
        identity = binder(
            loaded_checkpoint_sha256=checkpoint,
            loaded_policy_state_sha256=state_digest,
        )
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(type(value) is not str for value in identity)
        ):
            raise M03RV6StandaloneConfidenceError(
                "post-freeze confidence head returned malformed binding identity"
            )
        identities.append(identity)
        bound_children.append(child)
    if not identities:
        raise M03RV6StandaloneConfidenceError(
            "full policy contains no post-freeze M03R v6 confidence head"
        )
    # The fit contract forbids every post-calibration policy update. Freeze only
    # after every child has accepted the same full-policy/checkpoint identity so
    # a later validation failure cannot leave a partially frozen policy.
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    policy.eval()
    for child in bound_children:
        activator: Any = getattr(
            child,
            "_activate_m03r_v6_post_freeze_confidence_state",
            None,
        )
        if activator is None:
            raise M03RV6StandaloneConfidenceError(
                "post-freeze confidence head lacks its activation boundary"
            )
        activator()
    manifest_sha256s = tuple(value[0] for value in identities)
    fit_evidence_sha256s = tuple(value[1] for value in identities)
    payload: dict[str, object] = {
        "schema": M03R_V6_FROZEN_POLICY_CONFIDENCE_BINDING_SCHEMA,
        "loaded_checkpoint_sha256": checkpoint,
        "loaded_policy_state_sha256": state_digest,
        "calibration_manifest_sha256s": list(manifest_sha256s),
        "calibration_fit_evidence_sha256s": list(fit_evidence_sha256s),
        "bound_confidence_head_count": len(identities),
    }
    receipt_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt = M03RV6FrozenPolicyConfidenceBinding(
        loaded_checkpoint_sha256=checkpoint,
        loaded_policy_state_sha256=state_digest,
        calibration_manifest_sha256s=manifest_sha256s,
        calibration_fit_evidence_sha256s=fit_evidence_sha256s,
        bound_confidence_head_count=len(identities),
        receipt_sha256=receipt_sha256,
    )
    return receipt


@dataclass(frozen=True, slots=True)
class M03RV6StandaloneConfidenceConfig:
    """Exact two-stage identity for a v6 confidence-only model branch."""

    setting_id: str
    hidden_dim: int
    lifecycle_stage: M03RV6ConfidenceLifecycleStage
    calibration_manifest_sha256: str | None = None
    calibration_manifest: M03RConfidenceCalibrationManifest | None = None
    calibration_seed: int | None = None
    calibration_checkpoint_sha256: str | None = None
    calibration_model_state_sha256: str | None = None
    calibration_source_score_array_sha256: str | None = None
    calibration_source_target_array_sha256: str | None = None
    calibration_fit_evidence: object | None = None

    def __post_init__(self) -> None:
        setting = resolve_m03r_v6_setting(self.setting_id)
        if (
            setting.residual_alpha_heads
            or not setting.use_confidence_scaled_active_risk_budget
        ):
            raise M03RV6StandaloneConfidenceError(
                "the standalone v6 confidence head is exclusive to a registered "
                "confidence-sized setting without residual-alpha heads"
            )
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
        ):
            raise M03RV6StandaloneConfidenceError(
                "hidden_dim must be a positive integer"
            )
        if self.lifecycle_stage not in M03R_V6_CONFIDENCE_LIFECYCLE_STAGES:
            raise M03RV6StandaloneConfidenceError(
                "standalone v6 confidence requires an explicit training or "
                "post-freeze lifecycle stage"
            )

        calibration_identity = (
            self.calibration_seed,
            self.calibration_checkpoint_sha256,
            self.calibration_model_state_sha256,
            self.calibration_source_score_array_sha256,
            self.calibration_source_target_array_sha256,
        )
        calibration_bound = (
            self.calibration_manifest_sha256 is not None
            or self.calibration_manifest is not None
            or any(value is not None for value in calibration_identity)
        )
        if self.lifecycle_stage == "v6-training-uncalibrated":
            if calibration_bound or self.calibration_fit_evidence is not None:
                raise M03RV6StandaloneConfidenceError(
                    "v6 uncalibrated training forbids calibration manifests, "
                    "checkpoint identity, and fit evidence"
                )
            return

        if (
            not isinstance(self.calibration_manifest_sha256, str)
            or self.calibration_manifest is None
            or any(value is None for value in calibration_identity)
            or self.calibration_fit_evidence is None
        ):
            raise M03RV6StandaloneConfidenceError(
                "v6 post-freeze confidence execution requires typed calibration-fit "
                "evidence and its exact manifest/checkpoint identity"
            )
        from rl_quant.training.hold30_m03r_confidence_fit import (
            M03RConfidenceCalibrationFitEvidence,
            M03RConfidenceFitError,
            validate_m03r_confidence_calibration_fit_evidence,
        )

        evidence = self.calibration_fit_evidence
        if not isinstance(evidence, M03RConfidenceCalibrationFitEvidence):
            raise M03RV6StandaloneConfidenceError(
                "v6 confidence-fit evidence must use the typed governed artifact"
            )
        try:
            validate_m03r_confidence_calibration_fit_evidence(evidence)
        except M03RConfidenceFitError as error:
            raise M03RV6StandaloneConfidenceError(
                f"invalid standalone v6 confidence-fit evidence: {error}"
            ) from error
        if evidence.calibration_manifest != self.calibration_manifest:
            raise M03RV6StandaloneConfidenceError(
                "v6 confidence-fit evidence manifest does not match config"
            )
        if (
            evidence.target_construction_contract.protocol_generation
            != M03R_PROTOCOL_GENERATION
            or evidence.target_construction_contract.design_id != M03R_DESIGN_ID
        ):
            raise M03RV6StandaloneConfidenceError(
                "v6 confidence-fit evidence belongs to another generation"
            )

        assert self.calibration_seed is not None
        assert self.calibration_checkpoint_sha256 is not None
        assert self.calibration_model_state_sha256 is not None
        assert self.calibration_source_score_array_sha256 is not None
        assert self.calibration_source_target_array_sha256 is not None
        try:
            validate_m03r_confidence_calibration_manifest(
                self.calibration_manifest,
                expected_manifest_sha256=self.calibration_manifest_sha256,
                expected_setting_id=self.setting_id,
                expected_seed=self.calibration_seed,
                expected_checkpoint_sha256=self.calibration_checkpoint_sha256,
                expected_model_state_sha256=self.calibration_model_state_sha256,
                expected_source_score_array_sha256=(
                    self.calibration_source_score_array_sha256
                ),
                expected_source_target_array_sha256=(
                    self.calibration_source_target_array_sha256
                ),
                expected_protocol_generation=M03R_PROTOCOL_GENERATION,
                expected_design_id=M03R_DESIGN_ID,
            )
        except M03RConfidenceCalibrationError as error:
            raise M03RV6StandaloneConfidenceError(
                f"invalid standalone M03R v6 confidence calibration: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class M03RV6StandaloneConfidenceOutput:
    """Scalar confidence intent emitted independently of residual-alpha heads."""

    uncalibrated_logit: torch.Tensor
    signal_confidence: torch.Tensor | None
    active_risk_scale: torch.Tensor
    benchmark_derisk_request: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.uncalibrated_logit, torch.Tensor)
            or self.uncalibrated_logit.ndim != 1
            or not self.uncalibrated_logit.is_floating_point()
            or not bool(torch.isfinite(self.uncalibrated_logit).all())
        ):
            raise M03RV6StandaloneConfidenceError(
                "uncalibrated_logit must be finite floating [batch]"
            )
        expected = tuple(self.uncalibrated_logit.shape)
        for name in ("active_risk_scale", "benchmark_derisk_request"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected
                or value.dtype != self.uncalibrated_logit.dtype
                or value.device != self.uncalibrated_logit.device
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RV6StandaloneConfidenceError(
                    f"{name} must align with uncalibrated_logit"
                )
        if self.signal_confidence is not None and (
            not isinstance(self.signal_confidence, torch.Tensor)
            or tuple(self.signal_confidence.shape) != expected
            or self.signal_confidence.dtype != self.uncalibrated_logit.dtype
            or self.signal_confidence.device != self.uncalibrated_logit.device
            or not bool(torch.isfinite(self.signal_confidence).all())
            or bool(
                ((self.signal_confidence < 0.0) | (self.signal_confidence > 1.0)).any()
            )
        ):
            raise M03RV6StandaloneConfidenceError(
                "signal_confidence must be aligned and lie in [0,1]"
            )
        if bool((self.active_risk_scale < 0.0).any()) or bool(
            (self.benchmark_derisk_request != 0.0).any()
        ):
            raise M03RV6StandaloneConfidenceError(
                "active risk must be nonnegative and canonical benchmark derisk zero"
            )


class M03RV6StandaloneConfidenceHead(nn.Module):
    """Market-only scalar confidence route for M02.

    The pooled market state is detached so the separately optimized confidence
    objective cannot update the common entry/hazard policy.  During training,
    confidence does not size its own standardized-unit-risk outcome path.
    """

    def __init__(self, config: M03RV6StandaloneConfidenceConfig) -> None:
        super().__init__()
        self.config = config
        self._post_freeze_state_bound = False
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        output = self.confidence_head[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("standalone confidence output must be Linear")
        nn.init.orthogonal_(output.weight, gain=1e-3)
        nn.init.zeros_(output.bias)

    def _bind_m03r_v6_post_freeze_confidence_state(
        self,
        *,
        loaded_checkpoint_sha256: str,
        loaded_policy_state_sha256: str,
    ) -> tuple[str, str]:
        if self.config.lifecycle_stage != "v6-post-freeze-calibrated":
            raise M03RV6StandaloneConfidenceError(
                "only post-freeze confidence heads may bind loaded policy state"
            )
        checkpoint = _require_sha256(
            "loaded_checkpoint_sha256", loaded_checkpoint_sha256
        )
        state_digest = _require_sha256(
            "loaded_policy_state_sha256", loaded_policy_state_sha256
        )
        if checkpoint != self.config.calibration_checkpoint_sha256:
            raise M03RV6StandaloneConfidenceError(
                "loaded checkpoint does not match confidence-fit evidence"
            )
        if state_digest != self.config.calibration_model_state_sha256:
            raise M03RV6StandaloneConfidenceError(
                "loaded policy state does not match confidence-fit evidence"
            )
        from rl_quant.training.hold30_m03r_confidence_fit import (
            M03RConfidenceCalibrationFitEvidence,
        )

        evidence = self.config.calibration_fit_evidence
        if not isinstance(evidence, M03RConfidenceCalibrationFitEvidence):
            raise M03RV6StandaloneConfidenceError(
                "post-freeze confidence-fit evidence is no longer typed"
            )
        manifest = self.config.calibration_manifest
        if manifest is None:
            raise M03RV6StandaloneConfidenceError(
                "post-freeze confidence calibration manifest is missing"
            )
        return manifest.manifest_sha256, evidence.evidence_sha256

    def _activate_m03r_v6_post_freeze_confidence_state(self) -> None:
        if self.config.lifecycle_stage != "v6-post-freeze-calibrated":
            raise M03RV6StandaloneConfidenceError(
                "training confidence heads cannot enter post-freeze execution"
            )
        self.confidence_head.eval()
        self._post_freeze_state_bound = True

    def train(self, mode: bool = True) -> M03RV6StandaloneConfidenceHead:
        super().train(mode)
        if self._post_freeze_state_bound:
            self.confidence_head.eval()
        return self

    def forward(
        self,
        market_hidden: torch.Tensor,
        available: torch.Tensor,
        *,
        cash_index: int = 0,
    ) -> M03RV6StandaloneConfidenceOutput:
        if (
            self.config.lifecycle_stage == "v6-post-freeze-calibrated"
            and not self._post_freeze_state_bound
        ):
            raise M03RV6StandaloneConfidenceError(
                "post-freeze confidence forward requires package-owned full-policy "
                "state binding after checkpoint load"
            )
        if (
            not isinstance(market_hidden, torch.Tensor)
            or market_hidden.ndim != 3
            or market_hidden.shape[-1] != self.config.hidden_dim
            or not market_hidden.is_floating_point()
            or not bool(torch.isfinite(market_hidden).all())
        ):
            raise M03RV6StandaloneConfidenceError(
                "market_hidden must be finite floating [batch,asset,hidden_dim]"
            )
        batch, assets, _width = market_hidden.shape
        if (
            not isinstance(available, torch.Tensor)
            or available.dtype != torch.bool
            or tuple(available.shape) != (batch, assets)
            or available.device != market_hidden.device
        ):
            raise M03RV6StandaloneConfidenceError(
                "available must be boolean [batch,asset] on the model device"
            )
        if (
            isinstance(cash_index, bool)
            or not isinstance(cash_index, int)
            or not 0 <= cash_index < assets
        ):
            raise M03RV6StandaloneConfidenceError(
                "cash_index is outside the asset axis"
            )
        risky = available.clone()
        risky[:, cash_index] = False
        mask = risky.to(dtype=market_hidden.dtype).unsqueeze(-1)
        pooled = (market_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        raw_logit = self.confidence_head(pooled.detach()).squeeze(-1)

        signal_confidence: torch.Tensor | None = None
        maximum = float(
            M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
        )
        if self.config.lifecycle_stage == "v6-training-uncalibrated":
            active_risk = torch.full_like(raw_logit, maximum)
        else:
            manifest = self.config.calibration_manifest
            digest = self.config.calibration_manifest_sha256
            seed = self.config.calibration_seed
            checkpoint = self.config.calibration_checkpoint_sha256
            model_state = self.config.calibration_model_state_sha256
            score_digest = self.config.calibration_source_score_array_sha256
            target_digest = self.config.calibration_source_target_array_sha256
            assert manifest is not None
            assert digest is not None
            assert seed is not None
            assert checkpoint is not None
            assert model_state is not None
            assert score_digest is not None
            assert target_digest is not None
            signal_confidence = apply_m03r_confidence_calibration(
                raw_logit,
                manifest,
                expected_manifest_sha256=digest,
                expected_setting_id=self.config.setting_id,
                expected_seed=seed,
                expected_checkpoint_sha256=checkpoint,
                expected_model_state_sha256=model_state,
                expected_source_score_array_sha256=score_digest,
                expected_source_target_array_sha256=target_digest,
                expected_protocol_generation=M03R_PROTOCOL_GENERATION,
                expected_design_id=M03R_DESIGN_ID,
            )
            active_risk = maximum * signal_confidence

        return M03RV6StandaloneConfidenceOutput(
            uncalibrated_logit=raw_logit,
            signal_confidence=signal_confidence,
            active_risk_scale=active_risk,
            benchmark_derisk_request=torch.zeros_like(active_risk),
        )


__all__ = [
    "M03R_V6_CONFIDENCE_LIFECYCLE_STAGES",
    "M03R_V6_FROZEN_POLICY_CONFIDENCE_BINDING_SCHEMA",
    "M03R_V6_STANDALONE_CONFIDENCE_HEAD_SCHEMA",
    "M03RV6ConfidenceLifecycleStage",
    "M03RV6FrozenPolicyConfidenceBinding",
    "M03RV6StandaloneConfidenceConfig",
    "M03RV6StandaloneConfidenceError",
    "M03RV6StandaloneConfidenceHead",
    "M03RV6StandaloneConfidenceOutput",
    "bind_m03r_v6_frozen_policy_confidence",
    "m03r_v6_policy_state_sha256",
]
