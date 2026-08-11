"""Chronological action construction for TOP2000 M03R-v8 development.

The runtime applies learned exits first, then proposes a bounded incremental
reallocation from that hazard anchor, and only then applies the currently
qualified TOP2000 factor projector.  The pure ``build_with_trace`` boundary
retains each economic book separately so canonical and replay callers can
content-address the stage that first collapses a causal policy difference.

Unlike the v7 compatibility projector, the v8 projector consumes nonzero,
content-bound exposure slabs. The 1.5x setting therefore changes a real
feasible region rather than multiplying exact zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.execution.cost_aware_active_policy import (
    M03RV8CostAwareActiveProposal,
    build_cost_aware_active_proposal,
)
from rl_quant.execution.hold30 import Hold30BuiltAction
from rl_quant.execution.hold30_exit_v6 import build_m03r_v6_exit_release
from rl_quant.execution.top2000_m03r_v8_projection import (
    M03RV8ProjectionResult,
    M03RV8QualifiedRiskManifest,
    project_m03r_v8_active_book,
)
from rl_quant.models.daily_policy import Hold30Intent, hold30_release_hazard
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_ACTIVE_POLICY,
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03RV8Top2000DevSetting,
)
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)

M03R_V8_ACTION_TRACE_SCHEMA = "rl-quant.top2000-dev.m03r-v8-action-trace-v1"


class M03RV8RuntimeError(ValueError):
    """The v8 chronological action is malformed or unsupported."""


class M03RV8RuntimeBlocker(M03RV8RuntimeError):
    """A result-moving runtime input has not yet been frozen."""


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _one_way(delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * delta.abs().sum(dim=-1)


@dataclass(frozen=True, slots=True)
class M03RV8ActionConstructionTrace:
    """All separately governed books for one v8 fill-time construction."""

    setting_id: str
    repaired_weights: torch.Tensor
    proposed_release_by_age: torch.Tensor
    proposed_release: torch.Tensor
    raw_hazard_anchor_weights: torch.Tensor
    hazard_anchor_weights: torch.Tensor
    gated_proposal_weights: torch.Tensor
    projected_weights: torch.Tensor
    executed_weights: torch.Tensor
    proposal: M03RV8CostAwareActiveProposal
    hazard_projection: M03RV8ProjectionResult
    proposal_projection: M03RV8ProjectionResult
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_ACTION_TRACE_SCHEMA

    def validate(self) -> None:
        reference = self.repaired_weights
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 2
            or not reference.is_floating_point()
            or not bool(torch.isfinite(reference).all())
        ):
            raise M03RV8RuntimeError(
                "repaired book must be finite floating [batch,asset]"
            )
        for name in (
            "proposed_release",
            "raw_hazard_anchor_weights",
            "hazard_anchor_weights",
            "gated_proposal_weights",
            "projected_weights",
            "executed_weights",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RV8RuntimeError(f"{name} must align with the repaired book")
        if (
            not isinstance(self.proposed_release_by_age, torch.Tensor)
            or tuple(self.proposed_release_by_age.shape) != (*reference.shape, 61)
            or self.proposed_release_by_age.dtype != reference.dtype
            or self.proposed_release_by_age.device != reference.device
            or bool((self.proposed_release_by_age < 0.0).any())
            or not bool(torch.isfinite(self.proposed_release_by_age).all())
        ):
            raise M03RV8RuntimeError(
                "proposed_release_by_age must be finite nonnegative [batch,asset,61]"
            )
        self.proposal.validate()
        if (
            self.proposal.hazard_anchor_weights.data_ptr()
            != self.hazard_anchor_weights.data_ptr()
            and not torch.equal(
                self.proposal.hazard_anchor_weights,
                self.hazard_anchor_weights,
            )
        ):
            raise M03RV8RuntimeError(
                "proposal is not bound to the recorded hazard anchor"
            )
        if not torch.equal(
            self.proposal.requested_weights, self.gated_proposal_weights
        ):
            raise M03RV8RuntimeError(
                "gated proposal book drifted from its typed proposal"
            )
        if not torch.equal(self.executed_weights, self.projected_weights):
            raise M03RV8RuntimeError(
                "builder-level executed book must equal the projected book"
            )
        if (
            not torch.equal(
                self.hazard_projection.projected_weights,
                self.hazard_anchor_weights,
            )
            or not torch.equal(
                self.proposal_projection.projected_weights,
                self.projected_weights,
            )
            or self.hazard_projection.risk_manifest_sha256
            != self.proposal_projection.risk_manifest_sha256
        ):
            raise M03RV8RuntimeError(
                "projection evidence drifted from the action books"
            )
        if not torch.allclose(
            self.proposed_release_by_age.sum(dim=-1),
            self.proposed_release,
            atol=2.0e-6,
            rtol=2.0e-6,
        ) or bool((self.proposed_release - reference.clamp_min(0.0) > 2.0e-6).any()):
            raise M03RV8RuntimeError("hazard release does not reconcile by age")
        for name in (
            "repaired_weights",
            "raw_hazard_anchor_weights",
            "hazard_anchor_weights",
            "gated_proposal_weights",
            "projected_weights",
            "executed_weights",
        ):
            value = getattr(self, name)
            if bool((value < -2.0e-7).any()) or not torch.allclose(
                value.sum(dim=-1),
                torch.ones(value.shape[0], device=value.device, dtype=value.dtype),
                atol=2.0e-6,
                rtol=2.0e-6,
            ):
                raise M03RV8RuntimeError(f"{name} is not a long-only simplex")
        if (
            not self.setting_id.startswith("V8-")
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != M03R_V8_ACTION_TRACE_SCHEMA
        ):
            raise M03RV8RuntimeError("v8 action trace identity drifted")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "setting_id": self.setting_id,
            "book_sha256": {
                name: _tensor_sha256(getattr(self, name))
                for name in (
                    "repaired_weights",
                    "raw_hazard_anchor_weights",
                    "hazard_anchor_weights",
                    "gated_proposal_weights",
                    "projected_weights",
                    "executed_weights",
                    "proposed_release_by_age",
                    "proposed_release",
                )
            },
            "requested_incremental_one_way_turnover_sha256": _tensor_sha256(
                self.proposal.requested_incremental_one_way_turnover
            ),
            "risk_manifest_sha256": self.proposal_projection.risk_manifest_sha256,
            "hazard_projection_scale_sha256": _tensor_sha256(
                self.hazard_projection.radial_scale
            ),
            "proposal_projection_scale_sha256": _tensor_sha256(
                self.proposal_projection.radial_scale
            ),
        }
        return {**payload, "receipt_sha256": _payload_sha256(payload)}


def _hazard_anchor(
    intent: Hold30Intent,
    ledger: CohortLedger,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repaired = ledger.weights
    hazard = intent.hazard_residual
    if hazard is None or tuple(hazard.shape) != tuple(repaired.shape):
        raise M03RV8RuntimeError("v8 action requires aligned hazard residuals")
    risky = torch.ones_like(repaired, dtype=torch.bool)
    risky[:, ledger.cash_index] = False
    if intent.exit_action_v6 is not None:
        if any(
            value is not None
            for value in (
                intent.exact_hold_probability,
                intent.exact_hold_decision_st,
            )
        ):
            raise M03RV8RuntimeError(
                "three-way exit action and legacy exact-HOLD fields are exclusive"
            )
        release = build_m03r_v6_exit_release(
            ledger.economic_value,
            hazard,
            intent.exit_action_v6,
        )
        release_by_age = release.discretionary_release_by_age
        proposed_release = release.discretionary_release
    else:
        exact_hold = (
            intent.exact_hold_decision_st
            if intent.exact_hold_decision_st is not None
            else intent.exact_hold_probability
        )
        ages = torch.arange(
            61,
            device=repaired.device,
            dtype=ledger.economic_value.dtype,
        )
        hazards = hold30_release_hazard(
            ages,
            hazard.to(dtype=ledger.economic_value.dtype).unsqueeze(-1),
            exact_hold_probability=(
                None
                if exact_hold is None
                else exact_hold.to(dtype=ledger.economic_value.dtype).unsqueeze(-1)
            ),
        )
        release_by_age = ledger.economic_value * hazards
        release_by_age = torch.where(
            risky.unsqueeze(-1),
            release_by_age,
            torch.zeros_like(release_by_age),
        )
        proposed_release = release_by_age.sum(dim=-1)
    proposed_release = torch.where(
        risky,
        torch.minimum(proposed_release, repaired.clamp_min(0.0)),
        torch.zeros_like(proposed_release),
    )
    retained = torch.where(
        risky,
        (repaired - proposed_release).clamp_min(0.0),
        torch.zeros_like(repaired),
    )
    anchor = retained.clone()
    anchor[:, ledger.cash_index] = 1.0 - retained.sum(dim=-1)
    return anchor, release_by_age, proposed_release


class Top2000M03RV8ActionBuilder:
    """Pure v8 action builder suitable for the existing delayed-fill runtime."""

    def __init__(
        self,
        setting: M03RV8Top2000DevSetting,
        risk_manifest: M03RV8QualifiedRiskManifest,
        *,
        training_one_way_cost_basis_points: int = 20,
    ) -> None:
        if not isinstance(setting, M03RV8Top2000DevSetting):
            raise M03RV8RuntimeError("setting must use the frozen v8 protocol type")
        if not isinstance(risk_manifest, M03RV8QualifiedRiskManifest):
            raise M03RV8RuntimeError("risk_manifest must use the qualified v8 type")
        risk_manifest.validate()
        if (
            isinstance(training_one_way_cost_basis_points, bool)
            or not isinstance(training_one_way_cost_basis_points, int)
            or training_one_way_cost_basis_points != 20
        ):
            raise M03RV8RuntimeError("v8 training cost must remain exactly 20 bp")
        self.setting = setting
        self.risk_manifest = risk_manifest
        self.training_one_way_cost_basis_points = training_one_way_cost_basis_points

    def build_with_trace(
        self,
        intent: Hold30Intent,
        repaired_ledger: CohortLedger,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> tuple[Hold30BuiltAction, M03RV8ActionConstructionTrace]:
        """Construct and retain the hazard, gated, and projected books."""

        repaired = repaired_ledger.weights
        if self.risk_manifest.asset_count != repaired.shape[1]:
            raise M03RV8RuntimeError("risk manifest does not match the execution axis")
        alpha = intent.alpha_mean_30d
        uncertainty = intent.alpha_downside_30d
        confidence = intent.signal_confidence
        if (
            alpha is None
            or uncertainty is None
            or confidence is None
            or tuple(alpha.shape) != tuple(repaired.shape)
            or tuple(uncertainty.shape) != tuple(repaired.shape)
            or tuple(confidence.shape) != (repaired.shape[0],)
        ):
            raise M03RV8RuntimeError(
                "v8 cost-aware execution requires mean, uncertainty, and confidence"
            )
        raw_hazard_anchor, release_by_age, proposed_release = _hazard_anchor(
            intent,
            repaired_ledger,
        )
        # Qualify the exit/de-risk anchor independently. Confidence and the
        # cost gate can therefore control only later new or expanding risk.
        hazard_projection = project_m03r_v8_active_book(
            raw_hazard_anchor,
            benchmark_weights,
            trade_mask,
            risk_asset_caps,
            risk_gross_max,
            self.risk_manifest,
            factor_sector_bound_multiplier=(
                self.setting.factor_sector_bound_multiplier
            ),
        )
        hazard_anchor = hazard_projection.projected_weights
        gate_trade_mask = trade_mask.bool().clone()
        if intent.exit_action_v6 is not None:
            gate_trade_mask &= (
                intent.exit_action_v6.continuous_decision_st.detach() == 1.0
            )
        gate_trade_mask[:, repaired_ledger.cash_index] = False
        held_mask = repaired_ledger.economic_value.sum(dim=-1) > 0.0
        held_mask[:, repaired_ledger.cash_index] = False

        base_cost = repaired.new_full(
            repaired.shape,
            self.training_one_way_cost_basis_points * 1.0e-4,
        )
        if self.setting.cost_gate_mode == "disabled":
            cost = torch.zeros_like(base_cost)
            uncertainty_multiplier = 0.0
            entry_hurdle = 0.0
            retention_hurdle = 0.0
        elif self.setting.cost_gate_mode == "strong":
            cost = M03R_V8_ACTIVE_POLICY.strong_cost_gate_cost_multiplier * base_cost
            uncertainty_multiplier = (
                M03R_V8_ACTIVE_POLICY.strong_cost_gate_uncertainty_multiplier
            )
            entry_hurdle = M03R_V8_ACTIVE_POLICY.entry_hurdle_multiplier
            retention_hurdle = M03R_V8_ACTIVE_POLICY.retention_hurdle_multiplier
        else:
            cost = base_cost
            uncertainty_multiplier = M03R_V8_ACTIVE_POLICY.uncertainty_multiplier
            entry_hurdle = M03R_V8_ACTIVE_POLICY.entry_hurdle_multiplier
            retention_hurdle = M03R_V8_ACTIVE_POLICY.retention_hurdle_multiplier
        proposal = build_cost_aware_active_proposal(
            hazard_anchor,
            benchmark_weights,
            alpha,
            uncertainty,
            cost,
            confidence,
            held_mask,
            gate_trade_mask,
            risk_asset_caps,
            maximum_incremental_one_way_turnover=(
                M03R_V8_ACTIVE_POLICY.maximum_incremental_one_way_turnover
            ),
            uncertainty_multiplier=uncertainty_multiplier,
            entry_hurdle_multiplier=entry_hurdle,
            retention_hurdle_multiplier=retention_hurdle,
            cash_index=repaired_ledger.cash_index,
        )
        proposal_projection = project_m03r_v8_active_book(
            proposal.requested_weights,
            benchmark_weights,
            trade_mask,
            risk_asset_caps,
            risk_gross_max,
            self.risk_manifest,
            factor_sector_bound_multiplier=(
                self.setting.factor_sector_bound_multiplier
            ),
        )
        projected = proposal_projection.projected_weights
        requested_delta = proposal.requested_weights - repaired
        constructed_delta = projected - repaired
        built = Hold30BuiltAction(
            target_weights=projected,
            requested_delta=requested_delta,
            constructed_delta=constructed_delta,
            requested_turnover=_one_way(requested_delta),
            constructed_turnover=_one_way(constructed_delta),
            desired_risky_exposure=(
                projected.sum(dim=-1) - projected[:, repaired_ledger.cash_index]
            ),
            proposed_release_by_age=release_by_age,
            proposed_release=proposed_release,
            capacity_shortfall=(
                proposal.allowed_incremental_one_way_turnover
                - proposal.requested_incremental_one_way_turnover
            ).clamp_min(0.0),
        )
        trace = M03RV8ActionConstructionTrace(
            setting_id=self.setting.setting_id,
            repaired_weights=repaired,
            proposed_release_by_age=release_by_age,
            proposed_release=proposed_release,
            raw_hazard_anchor_weights=raw_hazard_anchor,
            hazard_anchor_weights=hazard_anchor,
            gated_proposal_weights=proposal.requested_weights,
            projected_weights=projected,
            executed_weights=projected,
            proposal=proposal,
            hazard_projection=hazard_projection,
            proposal_projection=proposal_projection,
        )
        trace.validate()
        return built, trace

    def __call__(
        self,
        intent: Hold30Intent,
        repaired_ledger: CohortLedger,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> Hold30BuiltAction:
        built, _trace = self.build_with_trace(
            intent,
            repaired_ledger,
            benchmark_weights,
            trade_mask,
            risk_asset_caps,
            risk_gross_max,
        )
        return built


def build_top2000_m03r_v8_chronological_runtime(
    policy: Top2000M03RV8DevelopmentPolicy,
    risk_manifest: M03RV8QualifiedRiskManifest,
) -> Hold30ChronologicalRuntime:
    """Bind one policy setting to its action builder and delayed-fill runtime."""

    if not isinstance(policy, Top2000M03RV8DevelopmentPolicy):
        raise M03RV8RuntimeError(
            "v8 chronological runtime requires the generation-qualified policy"
        )
    if policy.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256:
        raise M03RV8RuntimeError("v8 policy protocol binding drifted")
    return Hold30ChronologicalRuntime(
        "H2",
        action_builder=Top2000M03RV8ActionBuilder(policy.setting, risk_manifest),
    )


__all__ = [
    "M03R_V8_ACTION_TRACE_SCHEMA",
    "M03RV8ActionConstructionTrace",
    "M03RV8RuntimeBlocker",
    "M03RV8RuntimeError",
    "Top2000M03RV8ActionBuilder",
    "build_top2000_m03r_v8_chronological_runtime",
]
