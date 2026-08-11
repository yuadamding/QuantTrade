"""Deterministic eight-row distinct-policy sentinel for M03R-v8.

This CPU-only sentinel is a prelaunch structural gate.  It proves that when
each frozen causal row receives a deliberately different causal input, the
runtime preserves a distinct gated proposal and final executed book.  It is
not predictive, capacity, performance, or GPU evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.execution.top2000_m03r_v8_projection import (
    M03RV8QualifiedRiskManifest,
    qualify_m03r_v8_risk_manifest,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_V8_TOP2000_DEV_SETTING_IDS,
    resolve_m03r_v8_top2000_dev_setting,
)
from rl_quant.training.top2000_m03r_v8_runtime import Top2000M03RV8ActionBuilder

M03R_V8_DISTINCT_POLICY_SENTINEL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v8-distinct-policy-sentinel-v1"
)


class M03RV8DistinctPolicySentinelError(ValueError):
    """Synthetic sentinel inputs or evidence are malformed."""


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


def _intent_sha256(intent: Hold30Intent) -> str:
    fields = {}
    for name in (
        "entry_scores",
        "hazard_residual",
        "exact_hold_probability",
        "alpha_mean_30d",
        "alpha_downside_30d",
        "signal_confidence",
        "auxiliary_alpha_mean",
    ):
        value = getattr(intent, name)
        fields[name] = None if value is None else _tensor_sha256(value)
    return _payload_sha256(fields)


@dataclass(frozen=True, slots=True)
class M03RV8SyntheticSettingCase:
    """One exact setting and its deliberately distinct causal inputs."""

    setting_id: str
    intent: Hold30Intent
    repaired_ledger: CohortLedger
    benchmark_weights: torch.Tensor
    trade_mask: torch.Tensor
    risk_asset_caps: torch.Tensor
    risk_gross_max: torch.Tensor
    risk_manifest: M03RV8QualifiedRiskManifest


@dataclass(frozen=True, slots=True)
class M03RV8DistinctPolicySentinelRow:
    setting_index: int
    setting_id: str
    causal_input_sha256: str
    raw_hazard_anchor_sha256: str
    hazard_anchor_sha256: str
    gated_proposal_sha256: str
    projected_weights_sha256: str
    executed_weights_sha256: str
    action_trace_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class M03RV8DistinctPolicySentinelResult:
    rows: tuple[M03RV8DistinctPolicySentinelRow, ...]
    unique_causal_input_count: int
    unique_gated_proposal_count: int
    unique_projected_weights_count: int
    unique_executed_weights_count: int
    passed: bool
    cpu_only: bool = True
    gpu_capacity_evidence: bool = False
    performance_evidence: bool = False
    protocol_sha256: str = M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = M03R_V8_DISTINCT_POLICY_SENTINEL_SCHEMA

    def validate(self) -> None:
        if (
            len(self.rows) != 8
            or tuple(row.setting_index for row in self.rows) != tuple(range(8))
            or tuple(row.setting_id for row in self.rows)
            != M03R_V8_TOP2000_DEV_SETTING_IDS
        ):
            raise M03RV8DistinctPolicySentinelError(
                "sentinel requires one ordered row for every v8 setting"
            )
        hash_fields = (
            "causal_input_sha256",
            "raw_hazard_anchor_sha256",
            "hazard_anchor_sha256",
            "gated_proposal_sha256",
            "projected_weights_sha256",
            "executed_weights_sha256",
            "action_trace_receipt_sha256",
        )
        for row in self.rows:
            for name in hash_fields:
                value = getattr(row, name)
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise M03RV8DistinctPolicySentinelError(
                        f"sentinel row {name} is not a SHA-256 digest"
                    )
        observed_counts = (
            len({row.causal_input_sha256 for row in self.rows}),
            len({row.gated_proposal_sha256 for row in self.rows}),
            len({row.projected_weights_sha256 for row in self.rows}),
            len({row.executed_weights_sha256 for row in self.rows}),
        )
        declared_counts = (
            self.unique_causal_input_count,
            self.unique_gated_proposal_count,
            self.unique_projected_weights_count,
            self.unique_executed_weights_count,
        )
        if observed_counts != declared_counts:
            raise M03RV8DistinctPolicySentinelError(
                "sentinel uniqueness counts do not match row evidence"
            )
        reference = self.rows[0]
        # Settings 4/5 change only the gate and setting 7 changes only the
        # downstream factor slab. Their raw causal tensors intentionally equal
        # the reference. Setting 7 must also equal the reference through the
        # gate, then diverge at projection. Every final book must be distinct.
        expected_stage_map = (
            self.rows[4].causal_input_sha256 == reference.causal_input_sha256
            and self.rows[5].causal_input_sha256 == reference.causal_input_sha256
            and self.rows[7].causal_input_sha256 == reference.causal_input_sha256
            and self.rows[7].gated_proposal_sha256 == reference.gated_proposal_sha256
            and self.rows[4].gated_proposal_sha256 != reference.gated_proposal_sha256
            and self.rows[5].gated_proposal_sha256 != reference.gated_proposal_sha256
            and self.rows[7].projected_weights_sha256
            != reference.projected_weights_sha256
        )
        expected_passed = observed_counts == (5, 7, 8, 8) and expected_stage_map
        if (
            self.passed is not expected_passed
            or not self.cpu_only
            or self.gpu_capacity_evidence
            or self.performance_evidence
            or self.protocol_sha256 != M03R_V8_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != M03R_V8_DISTINCT_POLICY_SENTINEL_SCHEMA
        ):
            raise M03RV8DistinctPolicySentinelError(
                "sentinel status or research scope drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _payload_sha256(asdict(self))


def build_m03r_v8_synthetic_setting_cases() -> tuple[M03RV8SyntheticSettingCase, ...]:
    """Build the small deterministic CPU fixture used by the sentinel."""

    weights = torch.tensor(
        [[0.96, 0.008, 0.008, 0.008, 0.008, 0.008]],
        dtype=torch.float64,
    )
    ledger = CohortLedger.from_weights(
        weights,
        cash_index=0,
        initial_age=20,
        track_initial_units=True,
    )
    trade_mask = torch.ones_like(weights, dtype=torch.bool)
    caps = torch.tensor(
        [[1.0, 0.01, 0.01, 0.01, 0.01, 0.01]],
        dtype=torch.float64,
    )
    covariance = torch.eye(6, dtype=torch.float64) * 1.0e-6
    covariance[0, 0] = 0.0
    risk_manifest = qualify_m03r_v8_risk_manifest(
        exposure_names=("synthetic-factor",),
        asset_axis_sha256="3" * 64,
        source_receipt_sha256="4" * 64,
        exposure_loadings=torch.tensor(
            [[0.0], [1.0], [-1.0], [0.5], [-0.5], [0.0]],
            dtype=torch.float64,
        ),
        exposure_lower_bounds=torch.tensor([-0.0002], dtype=torch.float64),
        exposure_upper_bounds=torch.tensor([0.0002], dtype=torch.float64),
        active_beta_loadings=torch.zeros(6, dtype=torch.float64),
        daily_return_covariance=covariance,
        cash_index=0,
    )
    alpha_rows = (
        (0.000, 0.008, 0.004, -0.004, -0.008, 0.002),
        (0.000, 0.004, 0.009, -0.003, -0.009, 0.001),
        (0.000, 0.007, 0.001, -0.006, -0.002, 0.004),
        (0.000, 0.006, 0.003, -0.007, -0.001, 0.005),
        (0.000, 0.008, 0.004, -0.004, -0.008, 0.002),
        (0.000, 0.008, 0.004, -0.004, -0.008, 0.002),
        (0.000, 0.005, 0.002, -0.008, -0.003, 0.006),
        (0.000, 0.008, 0.004, -0.004, -0.008, 0.002),
    )
    cases: list[M03RV8SyntheticSettingCase] = []
    for setting_index, values in enumerate(alpha_rows):
        alpha = torch.tensor([values], dtype=torch.float64)
        hazard = -2.0 if setting_index != 6 else 0.0
        exact_hold = (
            torch.ones_like(alpha) if setting_index == 3 else torch.zeros_like(alpha)
        )
        intent = Hold30Intent(
            entry_scores=alpha,
            hazard_residual=torch.full_like(alpha, hazard),
            raw_hazard_residual=torch.full_like(alpha, hazard),
            exact_hold_probability=exact_hold,
            exposure_residual=torch.zeros(1, dtype=torch.float64),
            alpha_mean_30d=alpha,
            alpha_downside_30d=torch.full_like(alpha, 0.002),
            active_risk_scale=torch.tensor([0.04], dtype=torch.float64),
            signal_confidence=torch.tensor([0.8], dtype=torch.float64),
            uncalibrated_signal_confidence_logit=torch.zeros(1, dtype=torch.float64),
            benchmark_derisk_request=torch.zeros(1, dtype=torch.float64),
            auxiliary_alpha_mean=torch.stack((alpha, alpha, alpha, alpha), dim=-1),
        )
        cases.append(
            M03RV8SyntheticSettingCase(
                setting_id=M03R_V8_TOP2000_DEV_SETTING_IDS[setting_index],
                intent=intent,
                repaired_ledger=ledger,
                benchmark_weights=weights,
                trade_mask=trade_mask,
                risk_asset_caps=caps,
                risk_gross_max=torch.ones(1, dtype=torch.float64),
                risk_manifest=risk_manifest,
            )
        )
    return tuple(cases)


def run_m03r_v8_distinct_policy_sentinel(
    cases: tuple[M03RV8SyntheticSettingCase, ...] | None = None,
) -> M03RV8DistinctPolicySentinelResult:
    """Run all eight rows once and report stage-by-stage uniqueness."""

    selected = build_m03r_v8_synthetic_setting_cases() if cases is None else cases
    if len(selected) != 8 or tuple(row.setting_id for row in selected) != (
        M03R_V8_TOP2000_DEV_SETTING_IDS
    ):
        raise M03RV8DistinctPolicySentinelError(
            "sentinel cases must follow the exact eight-setting inventory"
        )
    rows: list[M03RV8DistinctPolicySentinelRow] = []
    for setting_index, case in enumerate(selected):
        setting = resolve_m03r_v8_top2000_dev_setting(case.setting_id)
        builder = Top2000M03RV8ActionBuilder(setting, case.risk_manifest)
        _built, trace = builder.build_with_trace(
            case.intent,
            case.repaired_ledger,
            case.benchmark_weights,
            case.trade_mask,
            case.risk_asset_caps,
            case.risk_gross_max,
        )
        rows.append(
            M03RV8DistinctPolicySentinelRow(
                setting_index=setting_index,
                setting_id=case.setting_id,
                causal_input_sha256=_intent_sha256(case.intent),
                raw_hazard_anchor_sha256=_tensor_sha256(
                    trace.raw_hazard_anchor_weights
                ),
                hazard_anchor_sha256=_tensor_sha256(trace.hazard_anchor_weights),
                gated_proposal_sha256=_tensor_sha256(trace.gated_proposal_weights),
                projected_weights_sha256=_tensor_sha256(trace.projected_weights),
                executed_weights_sha256=_tensor_sha256(trace.executed_weights),
                action_trace_receipt_sha256=trace.receipt_payload["receipt_sha256"],
            )
        )
    result = M03RV8DistinctPolicySentinelResult(
        rows=tuple(rows),
        unique_causal_input_count=len({row.causal_input_sha256 for row in rows}),
        unique_gated_proposal_count=len({row.gated_proposal_sha256 for row in rows}),
        unique_projected_weights_count=len(
            {row.projected_weights_sha256 for row in rows}
        ),
        unique_executed_weights_count=len(
            {row.executed_weights_sha256 for row in rows}
        ),
        passed=(
            len({row.causal_input_sha256 for row in rows}) == 5
            and len({row.gated_proposal_sha256 for row in rows}) == 7
            and len({row.projected_weights_sha256 for row in rows}) == 8
            and len({row.executed_weights_sha256 for row in rows}) == 8
            and rows[4].causal_input_sha256 == rows[0].causal_input_sha256
            and rows[5].causal_input_sha256 == rows[0].causal_input_sha256
            and rows[7].causal_input_sha256 == rows[0].causal_input_sha256
            and rows[7].gated_proposal_sha256 == rows[0].gated_proposal_sha256
            and rows[4].gated_proposal_sha256 != rows[0].gated_proposal_sha256
            and rows[5].gated_proposal_sha256 != rows[0].gated_proposal_sha256
            and rows[7].projected_weights_sha256 != rows[0].projected_weights_sha256
        ),
    )
    result.validate()
    return result


def collapsed_m03r_v8_sentinel_fixture() -> tuple[M03RV8SyntheticSettingCase, ...]:
    """Return a deliberate duplicate-input fixture for regression testing."""

    cases = list(build_m03r_v8_synthetic_setting_cases())
    cases[1] = replace(cases[1], intent=cases[0].intent)
    return tuple(cases)


__all__ = [
    "M03R_V8_DISTINCT_POLICY_SENTINEL_SCHEMA",
    "M03RV8DistinctPolicySentinelError",
    "M03RV8DistinctPolicySentinelResult",
    "M03RV8DistinctPolicySentinelRow",
    "M03RV8SyntheticSettingCase",
    "build_m03r_v8_synthetic_setting_cases",
    "collapsed_m03r_v8_sentinel_fixture",
    "run_m03r_v8_distinct_policy_sentinel",
]
