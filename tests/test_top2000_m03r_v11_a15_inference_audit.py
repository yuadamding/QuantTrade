from __future__ import annotations

import hashlib
import json
import copy
import os
import stat
from dataclasses import asdict, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.top2000_m03r_v11_a15_inference_audit import (
    M03RV11A15InferenceAuditError,
    build_m03r_v11_a15_audit_fold_evidence,
    build_m03r_v11_a15_audit_panel_report,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_PANEL_SCHEMA,
    M03R_V11_A15_AUDIT_VARIANTS,
    M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA,
    M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
    M03R_V11_A15_INFERENCE_AUDIT_SPEC,
    M03RV11A15AuditVariant,
    M03RV11A15InferenceAuditProtocolError,
    resolve_m03r_v11_a15_audit_variant,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import M03RV9AlphaDistribution
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_runtime import (
    M03RV11A15InferenceAuditRuntimeError,
    run_m03r_v11_a15_inference_audit_replay,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_fold import (
    m03r_v11_a15_audit_risk_semantic_lineage_sha256,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
    M03RV11A15AuditOneH100Capacity,
    M03RV11A15AuditTemplateConfig,
    M03RV11A15InferenceAuditKubernetesError,
    build_m03r_v11_a15_audit_live_evidence,
    render_m03r_v11_a15_inference_audit_suspended_job,
)
from rl_quant.training.top2000_m03r_v11_seadragon_operator import (
    M03RV11CreateOperatorConfig,
    M03RV11SeadragonOperatorError,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesRBACEvidence,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_lifecycle import (
    M03RV11A15AuditAttachConfig,
    run_m03r_v11_a15_audit_attach_lifecycle,
)
from rl_quant.workflows.top2000_m03r_v11_a15_inference_audit import (
    M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA,
    M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
)
from rl_quant.workflows import (
    top2000_m03r_v11_a15_inference_audit_seadragon_prepare as audit_prepare,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan import (
    M03R_V11_A15_PARENT_JOB_NAME,
    M03R_V11_A15_PARENT_RUN_ID,
    M03RV11A15InferenceAuditPlan,
    M03RV11A15InferenceAuditPlanError,
    M03RV11A15ParentCheckpointBinding,
    M03RV11A15ParentWorkerBinding,
    load_m03r_v11_a15_inference_audit_plan,
    write_m03r_v11_a15_inference_audit_plan,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_package import (
    M03RV11A15InferenceAuditPackageArtifacts,
    M03RV11A15InferenceAuditPackageError,
    build_m03r_v11_a15_inference_audit_authorization,
    build_m03r_v11_a15_inference_audit_package_plan,
    load_m03r_v11_a15_inference_audit_authorization,
    load_m03r_v11_a15_inference_audit_package_plan,
    write_m03r_v11_a15_inference_audit_authorization,
    write_m03r_v11_a15_inference_audit_package_plan,
)
from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_PROTOCOL_SHA256,
)


class _RiskState:
    def __init__(self, transitions: int, assets: int, start: int) -> None:
        self.asset_axis_sha256 = "a" * 64
        self.origin_state_indices = tuple(start + index for index in range(transitions))
        self.manifest_sha256 = "b" * 64
        self.state_sha256 = "c" * 64
        self.schema = "risk-state-v1"
        self.asset_count = assets
        self.source_binding_sha256 = "d" * 64
        self.daily_returns_receipt_sha256 = "e" * 64
        self.source_exposure_receipt_sha256 = "f" * 64
        self.cash_index = 0
        self.covariance_factor = torch.zeros(
            (transitions, assets, 2), dtype=torch.float64
        )
        self.specific_variance = torch.zeros((transitions, assets), dtype=torch.float64)

    def validate(self) -> None:
        return None

    def require_fast_identity(self, **kwargs: object) -> None:
        assert kwargs["sequence_asset_axis_sha256"] == self.asset_axis_sha256
        assert kwargs["checkpoint_asset_axis_sha256"] == self.asset_axis_sha256
        assert kwargs["expected_manifest_sha256"] == self.manifest_sha256


class _Operator:
    def __init__(self, origin: int, assets: int) -> None:
        self.origin_state_index = origin
        self.asset_axis_sha256 = "a" * 64
        self.source_exposure_receipt_sha256 = "f" * 64
        self.qualified_asset_mask = torch.ones(assets, dtype=torch.bool)
        self.qualified_asset_mask[0] = False
        self.receipt_sha256 = f"{origin:064x}"

    def validate(self) -> None:
        return None


def _sequence(transitions: int = 6, assets: int = 21) -> Hold30Sequence:
    risky_weight = 0.95 / (assets - 1)
    benchmark = torch.full(
        (transitions + 1, 1, assets), risky_weight, dtype=torch.float64
    )
    benchmark[:, :, 0] = 0.05
    base = torch.linspace(-0.01, 0.01, assets, dtype=torch.float64)
    returns = torch.stack(
        [base.roll(index).unsqueeze(0) for index in range(transitions)]
    )
    returns[:, :, 0] = 0.0
    available = torch.ones((transitions + 1, 1, assets), dtype=torch.bool)
    return Hold30Sequence(
        decision_state=torch.zeros((transitions + 1, 1, assets, 1)),
        asset_returns=returns,
        decision_available=available,
        fill_membership=available,
        fill_availability=available,
        benchmark_weights=benchmark,
        risk_asset_caps=torch.ones_like(benchmark),
        risk_gross_max=torch.ones((transitions + 1, 1), dtype=torch.float64),
        benchmark_net_returns=torch.zeros((transitions, 1), dtype=torch.float64),
        initial_ledger=CohortLedger.from_staggered_endowment(
            benchmark[0],
            cash_index=0,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        axis_id="a" * 64,
    )


def _distributions(
    transitions: int = 6, assets: int = 21
) -> tuple[M03RV9AlphaDistribution, ...]:
    rows = []
    for index in range(transitions):
        means = torch.zeros((1, assets, 4), dtype=torch.float64)
        means[0, 1:, 2] = torch.linspace(
            -0.02, 0.02, assets - 1, dtype=torch.float64
        ).roll(index)
        log_scale = torch.full_like(means, torch.log(torch.tensor(0.02)))
        rows.append(
            M03RV9AlphaDistribution(
                mean_by_horizon=means,
                log_scale_by_horizon=log_scale,
                selected_horizon_sessions=30,
                selected_mean=means[..., 2],
                selected_scale=torch.exp(log_scale[..., 2]),
            )
        )
    return tuple(rows)


def test_a15_risk_lineage_binds_semantics_not_cross_node_tensor_bytes() -> None:
    risk = _RiskState(transitions=6, assets=21, start=70)
    expected = m03r_v11_a15_audit_risk_semantic_lineage_sha256(risk)  # type: ignore[arg-type]
    risk.state_sha256 = "9" * 64
    assert (
        m03r_v11_a15_audit_risk_semantic_lineage_sha256(risk)  # type: ignore[arg-type]
        == expected
    )
    risk.manifest_sha256 = "8" * 64
    assert (
        m03r_v11_a15_audit_risk_semantic_lineage_sha256(risk)  # type: ignore[arg-type]
        != expected
    )


def _audit_plan() -> M03RV11A15InferenceAuditPlan:
    workers = tuple(
        M03RV11A15ParentWorkerBinding(
            setting_index=setting,
            terminal_relative_path=(
                f"completion-{setting:02d}-setting-{setting:02d}/"
                "predictive-terminal.json"
            ),
            terminal_file_sha256=f"{100 + setting:064x}",
            terminal_receipt_sha256=f"{110 + setting:064x}",
            worker_plan_sha256=f"{120 + setting:064x}",
            fold_terminal_file_sha256=tuple(
                f"{200 + setting * 10 + fold:064x}" for fold in range(6)
            ),
        )
        for setting in range(2)
    )
    checkpoints = tuple(
        M03RV11A15ParentCheckpointBinding(
            setting_index=setting,
            fold_index=fold,
            horizon_sessions=horizon,
            checkpoint_relative_path=(
                f"completion-{setting:02d}-setting-{setting:02d}/checkpoints/"
                f"fold-{fold:02d}-horizon-{horizon:02d}-update-0064.pt"
            ),
            checkpoint_file_sha256=f"{300 + setting * 100 + fold * 2 + offset:064x}",
            model_state_sha256=f"{400 + setting * 10 + fold:064x}",
            fold_terminal_relative_path=(
                f"completion-{setting:02d}-setting-{setting:02d}/receipts/"
                f"fold-{fold:02d}-terminal.json"
            ),
            fold_terminal_file_sha256=workers[setting].fold_terminal_file_sha256[fold],
            fold_terminal_receipt_sha256=f"{500 + setting * 10 + fold:064x}",
            worker_plan_sha256=workers[setting].worker_plan_sha256,
            training_source_array_sha256=f"{600 + setting * 10 + fold:064x}",
            training_residual_operator_root_sha256=(
                f"{700 + setting * 10 + fold:064x}"
            ),
            qualification_source_array_sha256=(f"{800 + setting * 10 + fold:064x}"),
            qualification_residual_operator_root_sha256=(
                f"{900 + setting * 10 + fold:064x}"
            ),
            fold_risk_state_sha256=f"{1000 + setting * 10 + fold:064x}",
        )
        for setting in range(2)
        for fold in range(6)
        for offset, horizon in enumerate((21, 30))
    )
    provisional = M03RV11A15InferenceAuditPlan(
        parent_run_id=M03R_V11_A15_PARENT_RUN_ID,
        parent_job_name=M03R_V11_A15_PARENT_JOB_NAME,
        parent_protocol_sha256=M03R_V11_PROTOCOL_SHA256,
        parent_package_plan_file_sha256="1" * 64,
        parent_package_plan_sha256="2" * 64,
        parent_execution_authorization_file_sha256="3" * 64,
        parent_execution_authorization_receipt_sha256="4" * 64,
        parent_source_archive_sha256="5" * 64,
        parent_image_reference="registry/research@sha256:" + "6" * 64,
        parent_terminal_evidence_relative_path=(
            "predictive-evidence/terminal-evidence.json"
        ),
        parent_terminal_evidence_file_sha256="7" * 64,
        parent_cleanup_receipt_relative_path=(
            "predictive-evidence/cleanup-receipt.json"
        ),
        parent_cleanup_receipt_file_sha256="8" * 64,
        parent_cleanup_receipt_sha256="9" * 64,
        workers=workers,
        checkpoints=checkpoints,
        receipt_sha256="0" * 64,
    )
    encoded = json.dumps(
        provisional.unsigned_payload(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    plan = replace(
        provisional,
        receipt_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    plan.validate()
    return plan


def _run(
    monkeypatch: pytest.MonkeyPatch,
    variant_id: str,
    *,
    fold_index: int = 0,
) -> object:
    import rl_quant.training.top2000_m03r_v11_a15_inference_audit_runtime as runtime

    transitions, assets, start = 6, 21, 70 + fold_index * 10
    sequence = _sequence(transitions, assets)

    def _apply(value: torch.Tensor, operator: _Operator) -> object:
        output = value.clone()
        output[~operator.qualified_asset_mask.to(output.device)] = 0.0
        output[operator.qualified_asset_mask.to(output.device)] -= output[
            operator.qualified_asset_mask.to(output.device)
        ].mean()
        return SimpleNamespace(residual=output)

    def _project(requested: torch.Tensor, *args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            projected_weights=requested,
            requested_to_executed_retention=torch.ones(
                requested.shape[0], dtype=requested.dtype
            ),
        )

    def _proposal(
        anchor: torch.Tensor,
        benchmark: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        cost: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        del benchmark, cost, args
        maximum = float(kwargs["maximum_incremental_one_way_turnover"])
        delta = mean.clone()
        delta[:, 0] = 0.0
        delta[:, 1:] -= delta[:, 1:].mean(-1, keepdim=True)
        turnover = 0.5 * delta.abs().sum(-1)
        multiplier = torch.minimum(
            torch.ones_like(turnover),
            torch.full_like(turnover, maximum) / turnover.clamp_min(1.0e-12),
        )
        delta *= multiplier.unsqueeze(-1)
        requested_turnover = 0.5 * delta.abs().sum(-1)
        probability = torch.sigmoid(mean / scale)
        allowed = torch.full_like(requested_turnover, maximum)
        return SimpleNamespace(
            requested_weights=anchor + delta,
            entry_probability=probability,
            exit_probability=1.0 - probability,
            buy_gate=probability,
            sell_gate=1.0 - probability,
            requested_incremental_one_way_turnover=requested_turnover,
            allowed_incremental_one_way_turnover=allowed,
        )

    monkeypatch.setattr(runtime, "apply_m03r_v11_residual_operator", _apply)
    monkeypatch.setattr(runtime, "project_m03r_v9_active_book", _project)
    monkeypatch.setattr(runtime, "build_cost_aware_active_proposal_v3", _proposal)
    benchmark_gross = (
        sequence.benchmark_weights[:-1, 0] * sequence.asset_returns[:, 0]
    ).sum(-1)
    operators = tuple(_Operator(start + index, assets) for index in range(transitions))
    return run_m03r_v11_a15_inference_audit_replay(
        sequence,
        _distributions(transitions, assets),
        operators,  # type: ignore[arg-type]
        _RiskState(transitions, assets, start),  # type: ignore[arg-type]
        resolve_m03r_v11_a15_audit_variant(variant_id),
        setting_index=0,
        fold_index=fold_index,
        selected_horizon_sessions=30,
        state_start_index=start,
        checkpoint_file_sha256="d" * 64,
        checkpoint_model_state_sha256="e" * 64,
        checkpoint_asset_axis_sha256="a" * 64,
        source_receipt_sha256="f" * 64,
        benchmark_gross_returns=benchmark_gross,
        benchmark_one_way_turnover=torch.zeros(transitions, dtype=torch.float64),
    )


def test_a15_audit_protocol_is_inference_only_and_predeclared() -> None:
    M03R_V11_A15_INFERENCE_AUDIT_SPEC.validate()
    assert len(M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256) == 64
    assert not M03R_V11_A15_INFERENCE_AUDIT_SPEC.training_authorized
    assert not M03R_V11_A15_INFERENCE_AUDIT_SPEC.checkpoint_selection_authorized
    assert not M03R_V11_A15_INFERENCE_AUDIT_SPEC.economic_generation_may_be_minted
    assert not M03R_V11_A15_INFERENCE_AUDIT_SPEC.outer_2026_access_authorized
    assert tuple(row.variant_id for row in M03R_V11_A15_AUDIT_VARIANTS) == (
        "original-cap-200bp",
        "original-cap-150bp",
        "original-cap-100bp",
        "original-cap-050bp",
        "zero-signal-cap-200bp",
        "sign-flipped-cap-200bp",
        "shuffled-cap-200bp",
    )
    with pytest.raises(M03RV11A15InferenceAuditProtocolError, match="drifted"):
        M03RV11A15AuditVariant("original-cap-200bp", "original", 0.01).validate()


def test_a15_parent_plan_round_trip_is_exact_and_no_clobber(tmp_path: Path) -> None:
    plan = _audit_plan()
    target = tmp_path / "audit-plan.json"
    file_sha256 = write_m03r_v11_a15_inference_audit_plan(target, plan)
    loaded = load_m03r_v11_a15_inference_audit_plan(
        target,
        expected_file_sha256=file_sha256,
    )
    assert loaded == plan
    assert isinstance(loaded.workers[0].fold_terminal_file_sha256, tuple)
    assert len(loaded.checkpoints) == 24
    with pytest.raises(M03RV11A15InferenceAuditPlanError, match="already exists"):
        write_m03r_v11_a15_inference_audit_plan(target, plan)


def test_a15_parent_plan_rejects_rehashed_semantic_drift() -> None:
    plan = _audit_plan()
    drifted = replace(plan, training_authorized=True, receipt_sha256="0" * 64)
    encoded = json.dumps(
        drifted.unsigned_payload(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    drifted = replace(drifted, receipt_sha256=hashlib.sha256(encoded).hexdigest())
    with pytest.raises(M03RV11A15InferenceAuditPlanError, match="drifted"):
        drifted.validate()


def test_a15_parent_receipts_retain_the_parent_newline_hash_contract() -> None:
    import rl_quant.training.top2000_m03r_v11_a15_inference_audit_plan as plan_module

    unsigned = {"schema": "parent-receipt", "completed": True}
    encoded = (
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    payload = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    assert (
        plan_module._validate_receipt(payload, name="parent_test")
        == payload["receipt_sha256"]
    )
    payload["receipt_sha256"] = hashlib.sha256(encoded.rstrip(b"\n")).hexdigest()
    with pytest.raises(M03RV11A15InferenceAuditPlanError, match="drifted"):
        plan_module._validate_receipt(payload, name="parent_test")


def test_a15_audit_package_authorizes_only_exact_inference(tmp_path: Path) -> None:
    audit = _audit_plan()
    artifacts = M03RV11A15InferenceAuditPackageArtifacts(
        source_archive_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        dependency_lock_sha256="c" * 64,
        worker_source_sha256="d" * 64,
        audit_plan_file_sha256="e" * 64,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        parent_package_plan_file_sha256=(audit.parent_package_plan_file_sha256),
        parent_package_plan_sha256=audit.parent_package_plan_sha256,
        parent_execution_authorization_file_sha256=(
            audit.parent_execution_authorization_file_sha256
        ),
        parent_execution_authorization_receipt_sha256=(
            audit.parent_execution_authorization_receipt_sha256
        ),
        parent_source_archive_sha256=audit.parent_source_archive_sha256,
        parent_terminal_evidence_file_sha256=(
            audit.parent_terminal_evidence_file_sha256
        ),
        parent_cleanup_receipt_file_sha256=(audit.parent_cleanup_receipt_file_sha256),
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,
        image_reference=audit.parent_image_reference,
        image_digest_sha256=audit.parent_image_reference.rsplit("@sha256:", 1)[-1],
    )
    package = build_m03r_v11_a15_inference_audit_package_plan(artifacts, audit)
    plan_path = tmp_path / "package-plan.json"
    plan_file_sha = write_m03r_v11_a15_inference_audit_package_plan(
        plan_path,
        package,
        audit,
    )
    authorization = build_m03r_v11_a15_inference_audit_authorization(
        package,
        audit,
        package_plan_file_sha256=plan_file_sha,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_file_sha = write_m03r_v11_a15_inference_audit_authorization(
        authorization_path,
        authorization,
        package,
        audit,
    )
    assert (
        load_m03r_v11_a15_inference_audit_package_plan(
            plan_path,
            expected_file_sha256=plan_file_sha,
            audit=audit,
        )
        == package
    )
    assert (
        load_m03r_v11_a15_inference_audit_authorization(
            authorization_path,
            expected_file_sha256=authorization_file_sha,
            package=package,
            audit=audit,
        )
        == authorization
    )
    assert authorization.inference_audit_authorized
    assert not authorization.training_authorized
    assert not authorization.checkpoint_selection_authorized
    assert not authorization.economic_training_authorized
    assert not authorization.outer_2026_access_authorized
    with pytest.raises(M03RV11A15InferenceAuditPackageError, match="drifted"):
        replace(package, training_authorized=True).validate(audit)


def _audit_package_bundle() -> tuple[object, object, object]:
    audit = _audit_plan()
    artifacts = M03RV11A15InferenceAuditPackageArtifacts(
        source_archive_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        dependency_lock_sha256="c" * 64,
        worker_source_sha256="d" * 64,
        audit_plan_file_sha256="e" * 64,
        audit_plan_receipt_sha256=audit.receipt_sha256,
        parent_package_plan_file_sha256=audit.parent_package_plan_file_sha256,
        parent_package_plan_sha256=audit.parent_package_plan_sha256,
        parent_execution_authorization_file_sha256=(
            audit.parent_execution_authorization_file_sha256
        ),
        parent_execution_authorization_receipt_sha256=(
            audit.parent_execution_authorization_receipt_sha256
        ),
        parent_source_archive_sha256=audit.parent_source_archive_sha256,
        parent_terminal_evidence_file_sha256=(
            audit.parent_terminal_evidence_file_sha256
        ),
        parent_cleanup_receipt_file_sha256=audit.parent_cleanup_receipt_file_sha256,
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,
        image_reference=audit.parent_image_reference,
        image_digest_sha256=audit.parent_image_reference.rsplit("@sha256:", 1)[-1],
    )
    package = build_m03r_v11_a15_inference_audit_package_plan(artifacts, audit)
    authorization = build_m03r_v11_a15_inference_audit_authorization(
        package,
        audit,
        package_plan_file_sha256="f" * 64,
    )
    return audit, package, authorization


def _capacity(
    *,
    package_plan_sha256: str = "6" * 64,
    authorization_receipt_sha256: str = "7" * 64,
    audit_plan_receipt_sha256: str = "8" * 64,
    parent_cleanup_receipt_sha256: str = "9" * 64,
    source_archive_sha256: str = "9" * 64,
) -> M03RV11A15AuditOneH100Capacity:
    provisional = M03RV11A15AuditOneH100Capacity(
        static_gate_receipt_sha256="1" * 64,
        capacity_terminal_file_sha256="2" * 64,
        capacity_terminal_receipt_sha256="3" * 64,
        startup_file_sha256="4" * 64,
        cursor_artifact_file_sha256="b" * 64,
        job_uid="job-uid",
        pod_uid="pod-uid",
        image_id="registry/research@sha256:" + "5" * 64,
        job_name="qt-m03r-v11-a15-audit-a05",
        run_id="qt-m03r-v11-a15-inference-audit-s17-20260813-a05",
        package_plan_sha256=package_plan_sha256,
        authorization_receipt_sha256=authorization_receipt_sha256,
        audit_plan_receipt_sha256=audit_plan_receipt_sha256,
        parent_cleanup_receipt_sha256=parent_cleanup_receipt_sha256,
        source_archive_sha256=source_archive_sha256,
        cleanup_receipt_file_sha256="a" * 64,
        receipt_sha256="0" * 64,
    )
    payload = {key: value for key, value in provisional.unsigned_payload().items()}
    receipt = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    value = replace(provisional, receipt_sha256=receipt)
    value.validate()
    return value


def test_a15_kubernetes_renders_zero_one_and_two_h100_boundaries() -> None:
    audit, package, authorization = _audit_package_bundle()
    rbac = M03RV7KubernetesRBACEvidence(
        jobs_get=True,
        jobs_list=True,
        jobs_create=True,
        jobs_patch=True,
        jobs_delete=True,
        pods_get=True,
        pods_list=True,
        pods_watch=True,
        pod_logs_get=True,
    )
    live = build_m03r_v11_a15_audit_live_evidence(
        observed_at_utc="2026-08-13T18:00:00+00:00",
        rbac=rbac,
        protected_or_other_committed_h100_count=0,
        live_schedulable_free_h100_count=None,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )
    template = M03RV11A15AuditTemplateConfig(
        job_name=package.job_name,  # type: ignore[attr-defined]
        run_id=package.run_id,  # type: ignore[attr-defined]
        service_account_name="default",
        pvc_claim_name="yding4-gpu-home",
    )
    common = {
        "audit": audit,
        "package": package,
        "authorization": authorization,
        "package_plan_file_sha256": "f" * 64,
        "authorization_file_sha256": "6" * 64,
        "live": live,
        "template": template,
        "now_utc": datetime(2026, 8, 13, 18, 1, tzinfo=UTC),
    }
    static = render_m03r_v11_a15_inference_audit_suspended_job(
        **common,  # type: ignore[arg-type]
        mode="static",
    )
    capacity = render_m03r_v11_a15_inference_audit_suspended_job(
        **common,  # type: ignore[arg-type]
        mode="capacity",
    )
    audit_job = render_m03r_v11_a15_inference_audit_suspended_job(
        **common,  # type: ignore[arg-type]
        mode="audit",
        capacity=_capacity(
            package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
            authorization_receipt_sha256=authorization.receipt_sha256,  # type: ignore[attr-defined]
            audit_plan_receipt_sha256=audit.receipt_sha256,  # type: ignore[attr-defined]
            source_archive_sha256=package.artifacts.source_archive_sha256,  # type: ignore[attr-defined]
        ),
    )
    assert (static.completions, static.parallelism, static.gpus_per_completion) == (
        1,
        1,
        0,
    )
    assert (
        capacity.completions,
        capacity.parallelism,
        capacity.gpus_per_completion,
    ) == (1, 1, 1)
    assert (
        audit_job.completions,
        audit_job.parallelism,
        audit_job.gpus_per_completion,
    ) == (2, 2, 1)
    for rendered in (static, capacity, audit_job):
        rendered.validate()
        manifest = rendered.manifest
        assert manifest["spec"]["suspend"] is True
        assert manifest["metadata"]["namespace"] == "yn-gpu-workload"
        annotations = manifest["metadata"]["annotations"]
        assert annotations["rl-quant/training-authorized"] == "false"
        assert annotations["rl-quant/outer-2026-access-authorized"] == "false"
        pod = manifest["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert len(container["volumeMounts"]) == 6
    static_container = static.manifest["spec"]["template"]["spec"]["containers"][0]
    assert {row["name"]: row.get("value") for row in static_container["env"]}[
        "NVIDIA_VISIBLE_DEVICES"
    ] == "none"
    assert "nodeSelector" not in static.manifest["spec"]["template"]["spec"]
    audit_pod = audit_job.manifest["spec"]["template"]["spec"]
    assert audit_pod["nodeSelector"] == {"gpu-type": "H100"}
    assert audit_job.manifest["spec"]["parallelism"] == 2
    output_root, run_as_user, run_as_group = audit_prepare._worker_output_identity(
        static
    )
    assert output_root == Path(
        "/rsrch8/home/bcb/yding4/quant/training/runs/"
        "qt-m03r-v11-a15-inference-audit-s17-20260813-a05/phases/static"
    )
    assert (run_as_user, run_as_group) == (307469, 600815)

    from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
        bind_m03r_v11_a15_audit_admitted_suspended_job,
    )

    static_first = _admitted_audit_job(static, resource_version="41")
    static_second = _admitted_audit_job(static, resource_version="42")
    static_binding = bind_m03r_v11_a15_audit_admitted_suspended_job(
        rendered=static,
        first_read=static_first,  # type: ignore[arg-type]
        second_read=static_second,  # type: ignore[arg-type]
        attached_owned_pod_uids=(),
    )
    assert static_binding.parallelism == 1
    assert static_binding.job_uid == "audit-job-uid"


def test_a15_prepare_creates_empty_worker_owned_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    (training_root / "runs").mkdir(parents=True)
    monkeypatch.setattr(audit_prepare, "SEADRAGON_QUANTTRADE_ROOT", training_root)
    output = training_root / "runs" / "run-a05" / "phases" / "static"
    result = audit_prepare._prepare_worker_output_root(
        output,
        run_as_user=os.geteuid(),
        run_as_group=os.getegid(),
    )
    metadata = result.stat()
    assert result == output
    assert stat.S_IMODE(metadata.st_mode) & stat.S_IRWXU == stat.S_IRWXU
    assert not stat.S_IMODE(metadata.st_mode) & stat.S_IRWXO

    (output / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        audit_prepare.M03RV11A15InferenceAuditPrepareError,
        match="must be empty",
    ):
        audit_prepare._prepare_worker_output_root(
            output,
            run_as_user=os.geteuid(),
            run_as_group=os.getegid(),
        )


def test_a15_prepare_rejects_controller_worker_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    (training_root / "runs").mkdir(parents=True)
    monkeypatch.setattr(audit_prepare, "SEADRAGON_QUANTTRADE_ROOT", training_root)
    with pytest.raises(
        audit_prepare.M03RV11A15InferenceAuditPrepareError,
        match="UID/GID",
    ):
        audit_prepare._prepare_worker_output_root(
            training_root / "runs" / "run-a05" / "phases" / "static",
            run_as_user=os.geteuid() + 1,
            run_as_group=os.getegid(),
        )


def test_a15_kubernetes_requires_capacity_and_respects_user_cap() -> None:
    audit, package, authorization = _audit_package_bundle()
    rbac = M03RV7KubernetesRBACEvidence(
        **dict.fromkeys(
            (
                "jobs_get",
                "jobs_list",
                "jobs_create",
                "jobs_patch",
                "jobs_delete",
                "pods_get",
                "pods_list",
                "pods_watch",
                "pod_logs_get",
            ),
            True,
        )
    )
    live = build_m03r_v11_a15_audit_live_evidence(
        observed_at_utc="2026-08-13T18:00:00+00:00",
        rbac=rbac,
        protected_or_other_committed_h100_count=15,
        live_schedulable_free_h100_count=None,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )
    template = M03RV11A15AuditTemplateConfig(
        job_name=package.job_name,  # type: ignore[attr-defined]
        run_id=package.run_id,  # type: ignore[attr-defined]
        service_account_name="default",
        pvc_claim_name="yding4-gpu-home",
    )
    inputs = {
        "audit": audit,
        "package": package,
        "authorization": authorization,
        "package_plan_file_sha256": "f" * 64,
        "authorization_file_sha256": "6" * 64,
        "live": live,
        "template": template,
        "now_utc": datetime(2026, 8, 13, 18, 1, tzinfo=UTC),
        "mode": "audit",
    }
    with pytest.raises(M03RV11A15InferenceAuditKubernetesError, match="requires"):
        render_m03r_v11_a15_inference_audit_suspended_job(
            **inputs,  # type: ignore[arg-type]
        )
    with pytest.raises(M03RV11A15InferenceAuditKubernetesError, match="cap"):
        render_m03r_v11_a15_inference_audit_suspended_job(
            **inputs,  # type: ignore[arg-type]
            capacity=_capacity(
                package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
                authorization_receipt_sha256=authorization.receipt_sha256,  # type: ignore[attr-defined]
                audit_plan_receipt_sha256=audit.receipt_sha256,  # type: ignore[attr-defined]
                source_archive_sha256=package.artifacts.source_archive_sha256,  # type: ignore[attr-defined]
            ),
        )


def test_a15_create_operator_accepts_only_two_completion_audit_geometry() -> None:
    root = "/rsrch8/home/bcb/yding4/quant/training/launches/audit"
    value = M03RV11CreateOperatorConfig(
        mode="audit",
        job_name="qt-m03r-v11-a15-audit-a05",
        run_id="qt-m03r-v11-a15-inference-audit-s17-20260813-a05",
        rendered_path=f"{root}/rendered.json",
        rendered_file_sha256="1" * 64,
        manifest_path=f"{root}/manifest.json",
        manifest_file_sha256="2" * 64,
        evidence_root=f"{root}/create-evidence",
        binding_output_path=f"{root}/binding.json",
        activation_output_path=f"{root}/activation.json",
        package_plan_sha256="3" * 64,
        execution_authorization_receipt_sha256="4" * 64,
        source_archive_sha256="5" * 64,
        capacity_receipt_sha256="6" * 64,
        operator_source_sha256="7" * 64,
        completions=2,
        parallelism=2,
    )
    assert value.mode == "audit"
    with pytest.raises(M03RV11SeadragonOperatorError, match="drifted"):
        replace(value, completions=3)


def _admitted_audit_job(
    rendered: object,
    *,
    resource_version: str,
    uid: str = "audit-job-uid",
) -> dict[str, object]:
    value = copy.deepcopy(rendered.manifest)  # type: ignore[attr-defined]
    value["metadata"].update({"uid": uid, "resourceVersion": resource_version})
    value["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": uid}
    }
    metadata = value["spec"]["template"]["metadata"]
    metadata["creationTimestamp"] = None
    metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": value["metadata"]["name"],
            "controller-uid": uid,
            "job-name": value["metadata"]["name"],
        }
    )
    if rendered.mode == "static":  # type: ignore[attr-defined]
        metadata["labels"]["runai/queue"] = "yding4-yn-gpu-workload-queue"
    return value


def _plain_json(path: Path, value: object) -> str:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_a15_capacity_attach_validates_h100_output_and_exact_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v11_a15_inference_audit_lifecycle as lifecycle
    import rl_quant.training.top2000_m03r_v7_seadragon_lifecycle as common_lifecycle
    from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
        bind_m03r_v11_a15_audit_admitted_suspended_job,
    )

    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    monkeypatch.setattr(common_lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    audit, package, authorization = _audit_package_bundle()
    rbac = M03RV7KubernetesRBACEvidence(
        **dict.fromkeys(
            (
                "jobs_get",
                "jobs_list",
                "jobs_create",
                "jobs_patch",
                "jobs_delete",
                "pods_get",
                "pods_list",
                "pods_watch",
                "pod_logs_get",
            ),
            True,
        )
    )
    live = build_m03r_v11_a15_audit_live_evidence(
        observed_at_utc="2026-08-13T18:00:00+00:00",
        rbac=rbac,
        protected_or_other_committed_h100_count=0,
        live_schedulable_free_h100_count=None,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )
    template = M03RV11A15AuditTemplateConfig(
        job_name=package.job_name,  # type: ignore[attr-defined]
        run_id=package.run_id,  # type: ignore[attr-defined]
        service_account_name="default",
        pvc_claim_name="yding4-gpu-home",
    )
    rendered = render_m03r_v11_a15_inference_audit_suspended_job(
        audit=audit,  # type: ignore[arg-type]
        package=package,  # type: ignore[arg-type]
        authorization=authorization,  # type: ignore[arg-type]
        package_plan_file_sha256="f" * 64,
        authorization_file_sha256="6" * 64,
        live=live,
        template=template,
        now_utc=datetime(2026, 8, 13, 18, 1, tzinfo=UTC),
        mode="capacity",
    )
    first = _admitted_audit_job(rendered, resource_version="17")
    second = _admitted_audit_job(rendered, resource_version="18")
    binding = bind_m03r_v11_a15_audit_admitted_suspended_job(
        rendered=rendered,
        first_read=first,  # type: ignore[arg-type]
        second_read=second,  # type: ignore[arg-type]
        attached_owned_pod_uids=(),
    )
    activation = build_m03r_v7_exact_job_activation_request(
        binding,
        second,  # type: ignore[arg-type]
    )
    rendered_path = tmp_path / "rendered.json"
    binding_path = tmp_path / "binding.json"
    activation_path = tmp_path / "activation.json"
    rendered_sha = _plain_json(rendered_path, asdict(rendered))
    binding_sha = _plain_json(binding_path, asdict(binding))
    activation_sha = _plain_json(activation_path, asdict(activation))

    output = tmp_path / "output"
    startup_unsigned = {
        "schema": M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
        "protocol_sha256": package.protocol_sha256,  # type: ignore[attr-defined]
        "audit_plan_file_sha256": package.artifacts.audit_plan_file_sha256,  # type: ignore[attr-defined]
        "audit_plan_receipt_sha256": audit.receipt_sha256,  # type: ignore[attr-defined]
        "audit_package_plan_file_sha256": "f" * 64,
        "audit_package_plan_sha256": package.package_plan_sha256,  # type: ignore[attr-defined]
        "audit_authorization_file_sha256": "6" * 64,
        "audit_authorization_receipt_sha256": authorization.receipt_sha256,  # type: ignore[attr-defined]
        "parent_package_plan_file_sha256": audit.parent_package_plan_file_sha256,  # type: ignore[attr-defined]
        "parent_package_plan_sha256": audit.parent_package_plan_sha256,  # type: ignore[attr-defined]
        "parent_execution_authorization_receipt_sha256": audit.parent_execution_authorization_receipt_sha256,  # type: ignore[attr-defined]
        "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,  # type: ignore[attr-defined]
        "setting_index": 0,
        "mode": "capacity",
        "hardware": {
            "visible_device_count": 1,
            "device_name": "NVIDIA H100 80GB HBM3",
            "device_total_memory": 80 * 1024**3,
            "compute_capability": [9, 0],
            "torch_cuda_version": "12.4",
            "exact_h100_80gb": True,
        },
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    startup = {
        **startup_unsigned,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                startup_unsigned,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }
    startup_path = output / "capacity-sentinel" / "startup.json"
    startup_file_sha = _plain_json(startup_path, startup)
    cursor_path = (
        output / "capacity-sentinel" / "fold-artifacts" / "fold-00-horizon-30.pt"
    )
    cursor_path.parent.mkdir(parents=True)
    cursor_path.write_bytes(b"immutable-capacity-cursor")
    cursor_file_sha = hashlib.sha256(cursor_path.read_bytes()).hexdigest()
    terminal_unsigned = {
        "schema": M03R_V11_A15_AUDIT_CAPACITY_TERMINAL_SCHEMA,
        "protocol_sha256": package.protocol_sha256,  # type: ignore[attr-defined]
        "audit_plan_file_sha256": package.artifacts.audit_plan_file_sha256,  # type: ignore[attr-defined]
        "audit_plan_receipt_sha256": audit.receipt_sha256,  # type: ignore[attr-defined]
        "startup_file_sha256": startup_file_sha,
        "setting_index": 0,
        "fold_index": 0,
        "horizon_sessions": 30,
        "variant_count": 7,
        "cursor_artifact_file_sha256": cursor_file_sha,
        "exact_h100_80gb": True,
        "full_execution_path_proven": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "economic_optimizer_updates": 0,
        "outer_2026_accessed": False,
        "h100_capacity_evidence": True,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    terminal = {
        **terminal_unsigned,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                terminal_unsigned,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }
    _plain_json(output / "capacity-sentinel" / "capacity-terminal.json", terminal)

    lifecycle_source_sha = hashlib.sha256(
        Path(lifecycle.__file__).read_bytes()
    ).hexdigest()
    config = M03RV11A15AuditAttachConfig(
        mode="capacity",
        job_name=package.job_name,  # type: ignore[attr-defined]
        run_id=package.run_id,  # type: ignore[attr-defined]
        job_uid=binding.job_uid,
        rendered_path=str(rendered_path),
        rendered_file_sha256=rendered_sha,
        binding_path=str(binding_path),
        binding_file_sha256=binding_sha,
        activation_request_path=str(activation_path),
        activation_request_file_sha256=activation_sha,
        output_root=str(output),
        evidence_root=str(tmp_path / "evidence"),
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        authorization_receipt_sha256=authorization.receipt_sha256,  # type: ignore[attr-defined]
        audit_plan_receipt_sha256=audit.receipt_sha256,  # type: ignore[attr-defined]
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,  # type: ignore[attr-defined]
        source_archive_sha256=package.artifacts.source_archive_sha256,  # type: ignore[attr-defined]
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
        lifecycle_source_sha256=lifecycle_source_sha,
        completions=1,
        parallelism=1,
        gpus_per_completion=1,
        static_gate_receipt_sha256="7" * 64,
        capacity_receipt_sha256="not-yet-created",
        phase_receipt_output_path=str(tmp_path / "capacity-receipt.json"),
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path),
    )
    config_path = tmp_path / "config.json"
    config_sha = _plain_json(config_path, asdict(config))

    active = copy.deepcopy(second)
    active["metadata"]["resourceVersion"] = "19"
    active["spec"]["suspend"] = False
    terminal_job = copy.deepcopy(active)
    terminal_job["metadata"]["resourceVersion"] = "20"
    terminal_job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    pod = {
        "metadata": {
            "name": "audit-pod-0",
            "uid": "audit-pod-uid",
            "annotations": {"batch.kubernetes.io/job-completion-index": "0"},
            "ownerReferences": [{"uid": binding.job_uid, "controller": True}],
        },
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "imageID": (
                        "containerd://registry/research@sha256:"
                        + package.artifacts.image_digest_sha256  # type: ignore[attr-defined]
                    ),
                    "state": {"terminated": {"exitCode": 0}},
                }
            ],
        },
    }

    class _Transport:
        def __init__(self) -> None:
            self.activated = False
            self.deleted = False

        def get_job(self, *, allow_absent: bool = False):
            del allow_absent
            if self.deleted:
                return None
            return terminal_job if self.activated else second

        def get_owned_pods(self):
            return () if not self.activated or self.deleted else (pod,)

        def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
            assert pod_name == "audit-pod-0"
            assert limit_bytes > 0
            return b"capacity complete\n"

        def activate(self, request):
            assert request.job_uid == binding.job_uid
            self.activated = True
            return active

        def delete(self, request, options_path: Path) -> None:
            assert request.job_uid == binding.job_uid
            assert options_path.is_file()
            self.deleted = True

    run_m03r_v11_a15_audit_attach_lifecycle(
        config_path,
        config_sha,
        transport=_Transport(),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    capacity_value = json.loads((tmp_path / "capacity-receipt.json").read_bytes())
    typed = M03RV11A15AuditOneH100Capacity(**capacity_value)
    typed.validate()
    assert typed.pod_uid == "audit-pod-uid"
    cleanup = json.loads((tmp_path / "evidence" / "cleanup-receipt.json").read_bytes())
    assert cleanup["first_job_absent"] is True
    assert cleanup["second_job_absent"] is True


def test_a15_replay_controls_are_target_blind_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _run(monkeypatch, "original-cap-200bp")
    zero = _run(monkeypatch, "zero-signal-cap-200bp")
    flipped = _run(monkeypatch, "sign-flipped-cap-200bp")
    shuffled_first = _run(monkeypatch, "shuffled-cap-200bp")
    shuffled_second = _run(monkeypatch, "shuffled-cap-200bp")
    assert not original.targets_or_outcomes_used_to_construct_actions
    assert torch.count_nonzero(zero.feasible_signal_trace) == 0
    assert torch.allclose(
        flipped.feasible_signal_trace, -original.feasible_signal_trace
    )
    assert torch.equal(
        shuffled_first.feasible_signal_trace, shuffled_second.feasible_signal_trace
    )
    assert shuffled_first.trace_sha256 == shuffled_second.trace_sha256
    assert not torch.equal(
        shuffled_first.feasible_signal_trace, original.feasible_signal_trace
    )
    assert torch.allclose(
        original.policy_gross_returns - original.benchmark_gross_returns,
        original.carry_active_return
        + original.anchor_repair_active_return
        + original.alpha_signal_active_return,
    )
    assert zero.requested_incremental_turnover.max().item() == 0.0
    assert original.requested_incremental_turnover.max().item() > 0.0


def test_a15_replay_rejects_posthoc_array_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _run(monkeypatch, "original-cap-200bp")
    trace.policy_gross_returns[0] += 0.01
    with pytest.raises(M03RV11A15InferenceAuditRuntimeError, match="drifted"):
        trace.validate()


def test_a15_fold_and_panel_report_controls_and_block_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = []
    for fold_index in range(6):
        trace = _run(
            monkeypatch,
            "original-cap-100bp",
            fold_index=fold_index,
        )
        target = trace.feasible_signal_trace * 0.25
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:, 0] = False
        evidence = build_m03r_v11_a15_audit_fold_evidence(
            trace,
            score_session_index=torch.arange(
                100 * fold_index,
                100 * fold_index + target.shape[0],
                dtype=torch.int64,
            ),
            target_log_return=target,
            valid=valid,
            target_source_array_sha256=f"{fold_index + 1:064x}",
        )
        evidence.validate()
        assert evidence.score_to_requested_delta_spearman.mean().item() > 0.99
        assert evidence.quantile_target_return[0][:, -1].mean() > 0.0
        assert evidence.quantile_target_return[0][:, 0].mean() < 0.0
        folds.append(evidence)
    report = build_m03r_v11_a15_audit_panel_report(tuple(folds))
    report.validate()
    assert report.variant_id == "original-cap-100bp"
    assert tuple(row[0] for row in report.annualized_net_active_return_by_cost) == (
        0,
        10,
        20,
        40,
    )
    assert tuple(row[0] for row in report.annualized_lcb_by_block_and_cost) == (
        10,
        21,
        30,
    )
    assert not report.training_performed
    assert not report.checkpoint_selection_performed
    assert not report.economic_generation_may_be_minted
    assert not report.outer_2026_accessed


def test_a15_fold_rejects_outcome_or_axis_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _run(monkeypatch, "original-cap-200bp")
    target = trace.feasible_signal_trace.clone()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, 0] = True
    with pytest.raises(M03RV11A15InferenceAuditError, match="axes"):
        build_m03r_v11_a15_audit_fold_evidence(
            trace,
            score_session_index=torch.arange(target.shape[0], dtype=torch.int64),
            target_log_return=target,
            valid=valid,
            target_source_array_sha256="1" * 64,
        )


def test_a15_lifecycle_revalidates_every_worker_artifact_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v11_a15_inference_audit_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    audit, package, authorization = _audit_package_bundle()
    output = tmp_path / "output"
    config = M03RV11A15AuditAttachConfig(
        mode="audit",
        job_name=package.job_name,  # type: ignore[attr-defined]
        run_id=package.run_id,  # type: ignore[attr-defined]
        job_uid="audit-job-uid",
        rendered_path=str(tmp_path / "rendered.json"),
        rendered_file_sha256="1" * 64,
        binding_path=str(tmp_path / "binding.json"),
        binding_file_sha256="2" * 64,
        activation_request_path=str(tmp_path / "activation.json"),
        activation_request_file_sha256="3" * 64,
        output_root=str(output),
        evidence_root=str(tmp_path / "evidence"),
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[attr-defined]
        authorization_receipt_sha256=authorization.receipt_sha256,  # type: ignore[attr-defined]
        audit_plan_receipt_sha256=audit.receipt_sha256,  # type: ignore[attr-defined]
        parent_cleanup_receipt_sha256=audit.parent_cleanup_receipt_sha256,  # type: ignore[attr-defined]
        source_archive_sha256=package.artifacts.source_archive_sha256,  # type: ignore[attr-defined]
        image_digest_sha256=package.artifacts.image_digest_sha256,  # type: ignore[attr-defined]
        lifecycle_source_sha256=hashlib.sha256(
            Path(lifecycle.__file__).read_bytes()
        ).hexdigest(),
        completions=2,
        parallelism=2,
        gpus_per_completion=1,
        static_gate_receipt_sha256="4" * 64,
        capacity_receipt_sha256="5" * 64,
        phase_receipt_output_path=str(tmp_path / "final.json"),
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path),
    )

    def _signed(unsigned: dict[str, object]) -> dict[str, object]:
        return {
            **unsigned,
            "receipt_sha256": hashlib.sha256(
                json.dumps(
                    unsigned,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
        }

    report_paths: dict[tuple[int, int, str], Path] = {}
    terminal_paths: dict[int, Path] = {}
    for setting_index in range(2):
        root = output / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
        startup = _signed(
            {
                "schema": M03R_V11_A15_AUDIT_STARTUP_SCHEMA,
                "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
                "audit_plan_receipt_sha256": audit.receipt_sha256,  # type: ignore[attr-defined]
                "audit_package_plan_sha256": package.package_plan_sha256,  # type: ignore[attr-defined]
                "audit_authorization_receipt_sha256": authorization.receipt_sha256,  # type: ignore[attr-defined]
                "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,  # type: ignore[attr-defined]
                "setting_index": setting_index,
                "mode": "audit",
                "hardware": {
                    "visible_device_count": 1,
                    "device_name": "NVIDIA H100 80GB HBM3",
                    "device_total_memory": 80 * 1024**3,
                    "exact_h100_80gb": True,
                },
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "economic_optimizer_updates": 0,
                "outer_2026_accessed": False,
                "promotion_eligible": False,
            }
        )
        startup_sha = _plain_json(root / "startup.json", startup)
        artifact_rows = []
        for fold in range(6):
            for horizon in (21, 30):
                artifact_path = (
                    root
                    / "fold-artifacts"
                    / f"fold-{fold:02d}-horizon-{horizon:02d}.pt"
                )
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_bytes(
                    f"setting={setting_index};fold={fold};horizon={horizon}".encode()
                )
                artifact_rows.append(
                    {
                        "fold_index": fold,
                        "horizon_sessions": horizon,
                        "file_sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                    }
                )
        report_rows = []
        for horizon in (21, 30):
            for variant in M03R_V11_A15_AUDIT_VARIANTS:
                report = _signed(
                    {
                        "schema": M03R_V11_A15_AUDIT_PANEL_SCHEMA,
                        "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
                        "setting_index": setting_index,
                        "setting_id": f"setting-{setting_index}",
                        "horizon_sessions": horizon,
                        "variant_id": variant.variant_id,
                        "fold_receipt_sha256": [
                            f"{fold + 1:064x}" for fold in range(6)
                        ],
                        "annualized_gross_active_return": 0.0,
                        "annualized_net_active_return_by_cost": [
                            [cost, 0.0] for cost in (0, 10, 20, 40)
                        ],
                        "annualized_lcb_by_block_and_cost": [
                            [block, [[cost, 0.0] for cost in (0, 10, 20, 40)]]
                            for block in (10, 21, 30)
                        ],
                        "top_bottom_lcb_by_block_and_quantiles": [
                            [block, [[quantiles, 0.0] for quantiles in (10, 20)]]
                            for block in (10, 21, 30)
                        ],
                        "aggregate_break_even_one_way_cost_basis_points": None,
                        "break_even_category": "no-positive-break-even",
                        "mean_action_cap_hit_fraction": 0.0,
                        "mean_score_to_action_spearman": 0.0,
                        "mean_brier_probability_beats_10bp": 0.0,
                        "mean_ece_probability_beats_10bp": 0.0,
                        "annualized_carry_active_return": 0.0,
                        "annualized_anchor_repair_active_return": 0.0,
                        "annualized_alpha_signal_active_return": 0.0,
                        "training_performed": False,
                        "checkpoint_selection_performed": False,
                        "economic_generation_may_be_minted": False,
                        "outer_2026_accessed": False,
                    }
                )
                report_path = (
                    root
                    / "panel-reports"
                    / f"horizon-{horizon:02d}-{variant.variant_id}.json"
                )
                report_sha = _plain_json(report_path, report)
                report_paths[(setting_index, horizon, variant.variant_id)] = report_path
                report_rows.append(
                    {
                        "horizon_sessions": horizon,
                        "variant_id": variant.variant_id,
                        "receipt_sha256": report["receipt_sha256"],
                        "file_sha256": report_sha,
                    }
                )
        terminal = _signed(
            {
                "schema": M03R_V11_A15_AUDIT_WORKER_TERMINAL_SCHEMA,
                "protocol_sha256": M03R_V11_A15_INFERENCE_AUDIT_PROTOCOL_SHA256,
                "audit_plan_receipt_sha256": audit.receipt_sha256,  # type: ignore[attr-defined]
                "parent_cleanup_receipt_sha256": audit.parent_cleanup_receipt_sha256,  # type: ignore[attr-defined]
                "setting_index": setting_index,
                "startup_file_sha256": startup_sha,
                "cursor_artifacts": artifact_rows,
                "panel_reports": report_rows,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "economic_optimizer_updates": 0,
                "economic_generation_may_be_minted": False,
                "outer_2026_accessed": False,
                "posthoc_exploratory": True,
                "promotion_eligible": False,
            }
        )
        terminal_path = root / "audit-terminal.json"
        _plain_json(terminal_path, terminal)
        terminal_paths[setting_index] = terminal_path

    outputs, file_hashes = lifecycle._validate_audit_output(config)
    assert len(outputs) == 2
    assert len(file_hashes) == 56

    bad_path = report_paths[(0, 21, "original-cap-200bp")]
    bad_report = json.loads(bad_path.read_bytes())
    bad_report["mean_action_cap_hit_fraction"] = 2.0
    bad_report = _signed(
        {key: value for key, value in bad_report.items() if key != "receipt_sha256"}
    )
    bad_file_sha = _plain_json(bad_path, bad_report)
    terminal_path = terminal_paths[0]
    terminal = json.loads(terminal_path.read_bytes())
    for row in terminal["panel_reports"]:
        if row["horizon_sessions"] == 21 and row["variant_id"] == "original-cap-200bp":
            row["receipt_sha256"] = bad_report["receipt_sha256"]
            row["file_sha256"] = bad_file_sha
    terminal = _signed(
        {key: value for key, value in terminal.items() if key != "receipt_sha256"}
    )
    _plain_json(terminal_path, terminal)
    with pytest.raises(
        lifecycle.M03RV11A15InferenceAuditLifecycleError,
        match="panel report semantics drifted",
    ):
        lifecycle._validate_audit_output(config)
