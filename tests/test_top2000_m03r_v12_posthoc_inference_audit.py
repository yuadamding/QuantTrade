from __future__ import annotations

import copy
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rl_quant.evaluation.top2000_m03r_v12_posthoc_inference_audit import (
    M03RV12PosthocAuditInputs,
    M03RV12PosthocAuditPanelReport,
    M03RV12PosthocInferenceAuditError,
    _average_ranks,
    build_m03r_v12_posthoc_audit_fold_evidence,
    build_m03r_v12_posthoc_audit_inputs,
    build_m03r_v12_posthoc_audit_panel_report,
    build_m03r_v12_posthoc_causal_action_mask,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_posthoc_inference_audit import (
    M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
    M03R_V12_POSTHOC_AUDIT_SPEC,
    M03R_V12_POSTHOC_AUDIT_VARIANTS,
    M03RV12PosthocAuditVariant,
)
from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03RV7KubernetesRBACEvidence,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v11_a15_inference_audit_kubernetes import (
    M03RV11A15AuditTemplateConfig,
    build_m03r_v11_a15_audit_live_evidence,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_kubernetes import (
    M03RV12PosthocAuditKubernetesError,
    M03RV12PosthocAuditOneH100Capacity,
    bind_m03r_v12_posthoc_audit_admitted_suspended_job,
    render_m03r_v12_posthoc_audit_suspended_job,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_lifecycle import (
    M03RV12PosthocAuditAttachConfig,
    run_m03r_v12_posthoc_audit_attach_lifecycle,
)
from rl_quant.workflows.top2000_m03r_v12_posthoc_inference_audit import (
    _validate_semantic_receipt,
)
from rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_package import (
    M03RV12PosthocAuditCheckpointBinding,
    M03RV12PosthocAuditPackageError,
    M03RV12PosthocAuditPackagePlan,
    M03RV12PosthocAuditParentBinding,
    M03RV12PosthocAuditSourceArtifacts,
    build_m03r_v12_posthoc_audit_package_plan,
    load_m03r_v12_posthoc_audit_package_plan,
    write_m03r_v12_posthoc_audit_package_plan,
)


def _digest(character: str) -> str:
    return character * 64


def _inputs(
    *, fold_index: int = 0, setting_index: int = 0
) -> M03RV12PosthocAuditInputs:
    dates, assets = 3, 6
    origins = torch.tensor([700, 701, 702], dtype=torch.int64)
    base = torch.tensor([0.0, -0.03, -0.01, 0.01, 0.02, 0.04])
    economic = torch.stack((base, base * 0.8, -base))
    rank = torch.stack((base.flip(0), base.flip(0) * 0.8, base))
    target = torch.stack((base * 0.5, base * 0.4, -base * 0.5))
    decision_available = torch.ones((dates, assets), dtype=torch.bool)
    regression_weight = torch.ones((dates, assets), dtype=torch.float64)
    action_mask = build_m03r_v12_posthoc_causal_action_mask(
        decision_available,
        regression_weight,
    )
    label_valid = action_mask.clone()
    post_fill_returns = torch.tensor(
        [
            [0.0, -0.01, -0.005, 0.004, 0.008, 0.012],
            [0.0, -0.008, -0.004, 0.003, 0.006, 0.010],
            [0.0, 0.02, 0.01, -0.01, -0.015, -0.025],
        ],
        dtype=torch.float64,
    )
    benchmark = torch.full((dates, assets), 1.0 / assets, dtype=torch.float64)
    caps = torch.ones_like(benchmark)
    return build_m03r_v12_posthoc_audit_inputs(
        setting_index=setting_index,
        fold_index=fold_index,
        checkpoint_file_sha256=f"{fold_index + 1:x}" * 64,
        checkpoint_model_state_sha256=_digest("2"),
        source_array_sha256=_digest("3"),
        asset_axis_sha256=_digest("4"),
        action_mask_source_sha256=_digest("5"),
        post_fill_return_source_sha256=_digest("6"),
        origin_indices=origins,
        raw_economic_mean=economic,
        raw_rank_score=rank,
        economic_mean=economic,
        rank_score=rank,
        selected_scale=torch.full((dates, assets), 0.02),
        target_log_return=target,
        label_valid=label_valid,
        causal_action_mask=action_mask,
        fill_execution_mask=action_mask,
        post_fill_asset_returns=post_fill_returns,
        benchmark_target_weights=benchmark,
        asset_weight_caps=caps,
    )


def _variant(
    channel: str = "economic-mean",
    transform: str = "original",
    cap: float = 0.02,
) -> M03RV12PosthocAuditVariant:
    return M03RV12PosthocAuditVariant(channel, transform, cap)  # type: ignore[arg-type]


def _package_plan() -> M03RV12PosthocAuditPackagePlan:
    checkpoints = tuple(
        M03RV12PosthocAuditCheckpointBinding(
            setting_index=setting,
            setting_id=(
                "V12-P0-separate-listwise-rank-economic-scale",
                "V12-P1-separate-rank-gaussian-economic-scale",
                "V12-P2-economic-scale-no-rank-control",
            )[setting],
            fold_index=fold,
            checkpoint_relative_path=(
                f"completion-{setting:02d}-setting-{setting:02d}/checkpoints/"
                f"fold-{fold:02d}-horizon-03-update-0064.pt"
            ),
            checkpoint_file_sha256=f"{setting + 1:x}" * 64,
            model_state_sha256=f"{fold + 4:x}" * 64,
            training_residual_operator_root_sha256="a" * 64,
            training_source_array_sha256="b" * 64,
            parent_fold_terminal_relative_path=(
                f"completion-{setting:02d}-setting-{setting:02d}/receipts/"
                f"fold-{fold:02d}-terminal.json"
            ),
            parent_fold_terminal_file_sha256="c" * 64,
            parent_fold_terminal_receipt_sha256="d" * 64,
        )
        for setting in range(3)
        for fold in range(6)
    )
    parent = M03RV12PosthocAuditParentBinding(
        run_id="qt-m03r-v12-h3-predictive-s17-20260813-a05",
        package_plan_sha256="e" * 64,
        package_plan_file_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        parent_protocol_sha256=M03R_V12_PROTOCOL_SHA256,
        checkpoint_bindings=checkpoints,
        predictive_terminal_relative_paths=tuple(
            f"completion-{setting:02d}-setting-{setting:02d}/predictive-terminal.json"
            for setting in range(3)
        ),
        predictive_terminal_file_sha256=("2" * 64, "3" * 64, "4" * 64),
        predictive_terminal_receipt_sha256=("5" * 64, "6" * 64, "7" * 64),
    )
    artifacts = M03RV12PosthocAuditSourceArtifacts(
        source_archive_sha256="8" * 64,
        source_manifest_file_sha256="9" * 64,
        source_inventory_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
        worker_source_sha256="c" * 64,
        image_reference="registry/research@sha256:" + "d" * 64,
        image_digest_sha256="d" * 64,
    )
    return build_m03r_v12_posthoc_audit_package_plan(artifacts, parent)


def _live(now: datetime):
    return build_m03r_v11_a15_audit_live_evidence(
        observed_at_utc=now.isoformat(),
        rbac=M03RV7KubernetesRBACEvidence(
            jobs_get=True,
            jobs_list=True,
            jobs_create=True,
            jobs_patch=True,
            jobs_delete=True,
            pods_get=True,
            pods_list=True,
            pods_watch=True,
            pod_logs_get=True,
        ),
        protected_or_other_committed_h100_count=0,
        live_schedulable_free_h100_count=16,
        live_h100_cap_verified=True,
        gpu_selector_observed_live=True,
    )


def _capacity(
    package: M03RV12PosthocAuditPackagePlan,
) -> M03RV12PosthocAuditOneH100Capacity:
    provisional = M03RV12PosthocAuditOneH100Capacity(
        audit_package_plan_sha256=package.package_plan_sha256,
        audit_package_plan_file_sha256="e" * 64,
        static_receipt_sha256="1" * 64,
        startup_file_sha256="2" * 64,
        startup_receipt_sha256="3" * 64,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        job_uid="job-uid",
        pod_uid="pod-uid",
        image_id="registry/research@sha256:" + "d" * 64,
        cleanup_receipt_file_sha256="4" * 64,
        receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        receipt_sha256=hashlib.sha256(
            json.dumps(
                provisional.unsigned_payload(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    )


def test_protocol_is_posthoc_only_and_forbids_2026() -> None:
    M03R_V12_POSTHOC_AUDIT_SPEC.validate()
    assert len(M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256) == 64
    assert not M03R_V12_POSTHOC_AUDIT_SPEC.training_authorized
    assert not M03R_V12_POSTHOC_AUDIT_SPEC.checkpoint_selection_authorized
    assert not M03R_V12_POSTHOC_AUDIT_SPEC.economic_generation_may_be_minted
    assert not M03R_V12_POSTHOC_AUDIT_SPEC.outer_2026_access_authorized


def test_vectorized_average_ranks_preserves_stable_tie_semantics() -> None:
    values = torch.tensor([4.0, 1.0, 1.0, 3.0, 4.0, 2.0])
    assert torch.equal(
        _average_ranks(values),
        torch.tensor([4.5, 0.5, 0.5, 3.0, 4.5, 2.0], dtype=torch.float64),
    )


def test_parent_v12_receipt_uses_its_exact_newline_hash_convention() -> None:
    unsigned = {"schema": "parent-v12", "outer_2026_accessed": False}
    raw = (
        json.dumps(unsigned, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    receipt = {**unsigned, "receipt_sha256": hashlib.sha256(raw).hexdigest()}
    _validate_semantic_receipt(receipt)
    receipt["outer_2026_accessed"] = True
    with pytest.raises(RuntimeError, match="semantic receipt drifted"):
        _validate_semantic_receipt(receipt)


def test_source_only_audit_package_round_trips_and_binds_18_checkpoints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit-plan.json"
    plan = _package_plan()
    file_sha = write_m03r_v12_posthoc_audit_package_plan(path, plan)
    loaded = load_m03r_v12_posthoc_audit_package_plan(
        path, expected_file_sha256=file_sha
    )
    assert loaded.receipt_sha256 == plan.receipt_sha256
    assert len(loaded.parent.checkpoint_bindings) == 18
    assert loaded.maximum_h100_requests == 3
    assert not loaded.training_authorized


def test_audit_package_rejects_a_parent_checkpoint_permutation() -> None:
    plan = _package_plan()
    rows = list(plan.parent.checkpoint_bindings)
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(M03RV12PosthocAuditPackageError, match="parent binding"):
        replace(plan.parent, checkpoint_bindings=tuple(rows)).validate()


def test_suspended_jobs_are_zero_one_and_three_h100_inference_only() -> None:
    package = _package_plan()
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    common = {
        "package": package,
        "audit_plan_file_sha256": "e" * 64,
        "live": _live(now),
        "template": M03RV11A15AuditTemplateConfig(
            job_name="qt-v12-audit-a01",
            run_id="qt-v12-audit-a01",
            service_account_name="default",
            pvc_claim_name="research-pvc",
            parent_package_mount_path="/mnt/parent-package",
        ),
        "now_utc": now,
    }
    static = render_m03r_v12_posthoc_audit_suspended_job(
        **common, mode="static"
    )
    capacity = render_m03r_v12_posthoc_audit_suspended_job(
        **common, mode="capacity"
    )
    audit = render_m03r_v12_posthoc_audit_suspended_job(
        **common, mode="audit", capacity=_capacity(package)
    )
    assert static.gpus_per_completion == 0
    assert capacity.completions == capacity.gpus_per_completion == 1
    assert audit.completions == audit.parallelism == 3
    assert audit.gpus_per_completion == 1
    assert audit.manifest["spec"]["suspend"] is True
    assert (
        static.manifest["spec"]["template"]["spec"]["dnsPolicy"]
        == "ClusterFirst"
    )
    static_resources = static.manifest["spec"]["template"]["spec"]["containers"][
        0
    ]["resources"]
    assert static_resources["requests"]["ephemeral-storage"] == "1Gi"
    assert static_resources["limits"]["ephemeral-storage"] == "4Gi"
    args = audit.manifest["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--indexed-output-root" in args
    assert "--setting-index" not in args
    mounts = audit.manifest["spec"]["template"]["spec"]["containers"][0][
        "volumeMounts"
    ]
    assert sum(row.get("readOnly") is True for row in mounts) == 3
    annotations = audit.manifest["metadata"]["annotations"]
    assert annotations["rl-quant/training-authorized"] == "false"
    assert annotations["rl-quant/outer-2026-access-authorized"] == "false"


def test_full_audit_rejects_unbound_capacity_evidence() -> None:
    package = _package_plan()
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    with pytest.raises(M03RV12PosthocAuditKubernetesError, match="another plan"):
        render_m03r_v12_posthoc_audit_suspended_job(
            package=package,
            audit_plan_file_sha256="f" * 64,
            live=_live(now),
            template=M03RV11A15AuditTemplateConfig(
                job_name="qt-v12-audit-a01",
                run_id="qt-v12-audit-a01",
                service_account_name="default",
                pvc_claim_name="research-pvc",
                parent_package_mount_path="/mnt/parent-package",
            ),
            now_utc=now,
            mode="audit",
            capacity=_capacity(package),
        )


def _admitted_audit_job(
    rendered: object,
    *,
    resource_version: str,
    uid: str = "v12-posthoc-job-uid",
) -> dict[str, object]:
    value = copy.deepcopy(rendered.manifest)  # type: ignore[attr-defined]
    value["metadata"].update({"uid": uid, "resourceVersion": resource_version})
    value["spec"]["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": uid}
    }
    metadata = value["spec"]["template"]["metadata"]
    metadata["creationTimestamp"] = None
    metadata["labels"] = dict(metadata["labels"])
    metadata["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "batch.kubernetes.io/job-name": value["metadata"]["name"],
            "controller-uid": uid,
            "job-name": value["metadata"]["name"],
        }
    )
    return value


def _plain_json(path: Path, value: object) -> str:
    raw = (
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
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_lifecycle_outputs(
    output: Path,
    *,
    package_plan_sha256: str,
    package_plan_file_sha256: str,
) -> None:
    for setting_index in range(3):
        root = output / f"completion-{setting_index:02d}-setting-{setting_index:02d}"
        fold_artifact_hashes: list[str] = []
        for fold_index in range(6):
            artifact = root / "fold-artifacts" / f"fold-{fold_index:02d}.pt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{setting_index}:{fold_index}".encode())
            fold_artifact_hashes.append(hashlib.sha256(artifact.read_bytes()).hexdigest())
        reports: list[dict[str, object]] = []
        for variant_index, variant in enumerate(M03R_V12_POSTHOC_AUDIT_VARIANTS):
            provisional = M03RV12PosthocAuditPanelReport(
                setting_index=setting_index,
                setting_id=M03R_V12_SETTING_IDS[setting_index],
                variant_id=variant.variant_id,
                fold_receipt_sha256=tuple(
                    hashlib.sha256(
                        f"{setting_index}:{variant_index}:{fold}".encode()
                    ).hexdigest()
                    for fold in range(6)
                ),
                mean_date_spearman_ic=0.01,
                positive_mean_ic_fold_count=4,
                mean_top_bottom_spread=0.001,
                positive_spread_fold_count=4,
                annualized_gross_active_return=0.01,
                annualized_net_active_return_by_cost=(0.01, 0.009, 0.008, 0.006),
                gross_active_lcb_by_block=(0.001, 0.001, 0.001),
                net_10bp_lcb_by_block=(0.0005, 0.0005, 0.0005),
                spread_lcb_by_block=(0.0001, 0.0001, 0.0001),
                aggregate_break_even_one_way_cost_basis_points=25.0,
                favorable_cost_dominance=False,
                mean_policy_one_way_turnover=0.01,
                mean_incremental_one_way_turnover=0.005,
                mean_active_mass=variant.maximum_active_one_way_mass,
                mean_no_action_fraction=0.25,
                median_signal_projection_retention=0.5,
                receipt_sha256="0" * 64,
            )
            report = replace(
                provisional,
                receipt_sha256=hashlib.sha256(
                    json.dumps(
                        provisional.unsigned_payload(),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            )
            report.validate()
            reports.append(asdict(report))
        unsigned = {
            "schema": "rl-quant.top2000-dev.m03r-v12-posthoc-audit-worker-terminal-v1",
            "protocol_sha256": M03R_V12_POSTHOC_AUDIT_PROTOCOL_SHA256,
            "audit_package_plan_sha256": package_plan_sha256,
            "audit_package_plan_file_sha256": package_plan_file_sha256,
            "setting_index": setting_index,
            "visible_device_count": 1,
            "device_name": "NVIDIA H100 80GB HBM3",
            "exact_one_h100_80gb": True,
            "fold_artifact_file_sha256": fold_artifact_hashes,
            "panel_reports": reports,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "economic_optimizer_updates": 0,
            "economic_generation_may_be_minted": False,
            "outer_2026_accessed": False,
            "posthoc_exploratory": True,
            "reportable": False,
            "promotion_eligible": False,
        }
        unsigned["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        _plain_json(root / "audit-terminal.json", unsigned)


def test_attach_lifecycle_validates_success_and_partial_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_lifecycle as lifecycle
    import rl_quant.training.top2000_m03r_v7_seadragon_lifecycle as common_lifecycle

    monkeypatch.setattr(lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        common_lifecycle, "SEADRAGON_QUANTTRADE_ROOT", str(tmp_path)
    )
    package = _package_plan()
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    rendered = render_m03r_v12_posthoc_audit_suspended_job(
        package=package,
        audit_plan_file_sha256="e" * 64,
        live=_live(now),
        template=M03RV11A15AuditTemplateConfig(
            job_name="qt-v12-posthoc-audit",
            run_id="qt-v12-posthoc-audit",
            service_account_name="default",
            pvc_claim_name="yding4-gpu-home",
        ),
        now_utc=now,
        mode="audit",
        capacity=_capacity(package),
    )
    first = _admitted_audit_job(rendered, resource_version="17")
    second = _admitted_audit_job(rendered, resource_version="18")
    binding = bind_m03r_v12_posthoc_audit_admitted_suspended_job(
        rendered=rendered,
        first_read=first,  # type: ignore[arg-type]
        second_read=second,  # type: ignore[arg-type]
        attached_owned_pod_uids=(),
    )
    activation = build_m03r_v7_exact_job_activation_request(
        binding, second  # type: ignore[arg-type]
    )
    rendered_path = tmp_path / "rendered.json"
    binding_path = tmp_path / "binding.json"
    activation_path = tmp_path / "activation.json"
    rendered_sha = _plain_json(rendered_path, asdict(rendered))
    binding_sha = _plain_json(binding_path, asdict(binding))
    activation_sha = _plain_json(activation_path, asdict(activation))
    output = tmp_path / "output"
    _write_lifecycle_outputs(
        output,
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256="e" * 64,
    )
    lifecycle_source_sha = hashlib.sha256(Path(lifecycle.__file__).read_bytes()).hexdigest()
    config = M03RV12PosthocAuditAttachConfig(
        job_name="qt-v12-posthoc-audit",
        run_id="qt-v12-posthoc-audit",
        job_uid=binding.job_uid,
        rendered_path=str(rendered_path),
        rendered_file_sha256=rendered_sha,
        binding_path=str(binding_path),
        binding_file_sha256=binding_sha,
        activation_request_path=str(activation_path),
        activation_request_file_sha256=activation_sha,
        output_root=str(output),
        evidence_root=str(tmp_path / "evidence"),
        phase_receipt_output_path=str(tmp_path / "phase-receipt.json"),
        audit_package_plan_sha256=package.package_plan_sha256,
        audit_package_plan_file_sha256="e" * 64,
        source_archive_sha256=package.artifacts.source_archive_sha256,
        capacity_receipt_sha256=_capacity(package).receipt_sha256,
        image_digest_sha256=package.artifacts.image_digest_sha256,
        lifecycle_source_sha256=lifecycle_source_sha,
        host_python_path="/usr/bin/python3",
        pythonpath=str(tmp_path),
    )
    config_path = tmp_path / "config.json"
    config_sha = _plain_json(config_path, asdict(config))
    active = copy.deepcopy(second)
    active["metadata"]["resourceVersion"] = "19"
    active["spec"]["suspend"] = False
    terminal = copy.deepcopy(active)
    terminal["metadata"]["resourceVersion"] = "20"
    terminal["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    pods = tuple(
        {
            "metadata": {
                "name": f"audit-pod-{index}",
                "uid": f"audit-pod-uid-{index}",
                "annotations": {
                    "batch.kubernetes.io/job-completion-index": str(index)
                },
                "ownerReferences": [{"uid": binding.job_uid, "controller": True}],
            },
            "status": {
                "phase": "Succeeded",
                "containerStatuses": [
                    {
                        "imageID": "containerd://registry/research@sha256:"
                        + package.artifacts.image_digest_sha256,
                        "state": {"terminated": {"exitCode": 0}},
                    }
                ],
            },
        }
        for index in range(3)
    )

    class _Transport:
        activated = False
        deleted = False

        def get_job(self, *, allow_absent: bool = False):
            del allow_absent
            if self.deleted:
                return None
            return terminal if self.activated else second

        def get_owned_pods(self):
            return () if not self.activated or self.deleted else pods

        def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
            assert pod_name.startswith("audit-pod-") and limit_bytes > 0
            return b"audit complete\n"

        def activate(self, request):
            assert request.job_uid == binding.job_uid
            self.activated = True
            return active

        def delete(self, request, options_path: Path) -> None:
            assert request.job_uid == binding.job_uid and options_path.is_file()
            self.deleted = True

    run_m03r_v12_posthoc_audit_attach_lifecycle(
        config_path,
        config_sha,
        transport=_Transport(),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    phase = json.loads((tmp_path / "phase-receipt.json").read_text())
    assert phase["passed"] is True
    assert phase["completion_count"] == 3
    assert len(phase["worker_outputs"]) == 3
    cleanup = json.loads((tmp_path / "evidence/cleanup-receipt.json").read_text())
    assert cleanup["first_job_absent"] is True
    assert cleanup["second_job_absent"] is True

    failed_config = replace(
        config,
        evidence_root=str(tmp_path / "failed-evidence"),
        phase_receipt_output_path=str(tmp_path / "failed-phase-receipt.json"),
    )
    failed_config_path = tmp_path / "failed-config.json"
    failed_config_sha = _plain_json(failed_config_path, asdict(failed_config))
    failed_terminal = copy.deepcopy(active)
    failed_terminal["metadata"]["resourceVersion"] = "21"
    failed_terminal["status"] = {
        "conditions": [{"type": "Failed", "status": "True"}]
    }
    retained_failed_pod = copy.deepcopy(pods[0])
    retained_failed_pod["status"]["phase"] = "Failed"
    retained_failed_pod["status"]["containerStatuses"][0]["state"] = {
        "terminated": {"exitCode": 1}
    }

    class _FailedTransport:
        activated = False
        deleted = False

        def get_job(self, *, allow_absent: bool = False):
            del allow_absent
            if self.deleted:
                return None
            return failed_terminal if self.activated else second

        def get_owned_pods(self):
            if not self.activated or self.deleted:
                return ()
            return (retained_failed_pod,)

        def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
            assert pod_name == "audit-pod-0" and limit_bytes > 0
            return b"worker failed before other completions started\n"

        def activate(self, request):
            assert request.job_uid == binding.job_uid
            self.activated = True
            return active

        def delete(self, request, options_path: Path) -> None:
            assert request.job_uid == binding.job_uid and options_path.is_file()
            self.deleted = True

    with pytest.raises(
        lifecycle.M03RV12PosthocAuditLifecycleError,
        match="evidence preserved and exact cleanup completed",
    ):
        run_m03r_v12_posthoc_audit_attach_lifecycle(
            failed_config_path,
            failed_config_sha,
            transport=_FailedTransport(),  # type: ignore[arg-type]
            sleep=lambda _: None,
        )
    failed_pods = json.loads(
        (tmp_path / "failed-evidence/terminal-pods.json").read_text()
    )
    assert len(failed_pods["items"]) == 1
    assert (
        tmp_path / "failed-evidence/terminal-log-index-0-audit-pod-0.txt"
    ).read_text() == "worker failed before other completions started\n"
    failed_cleanup = json.loads(
        (tmp_path / "failed-evidence/cleanup-receipt.json").read_text()
    )
    assert failed_cleanup["first_job_absent"] is True
    assert failed_cleanup["second_job_absent"] is True
    assert not (tmp_path / "failed-phase-receipt.json").exists()


def test_attach_lifecycle_import_does_not_require_worker_data_dependencies() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'pyarrow' or name.startswith('pyarrow.'):
        raise ModuleNotFoundError('blocked data dependency')
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import rl_quant.training.top2000_m03r_v12_posthoc_inference_audit_lifecycle
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_causal_action_mask_uses_only_origin_evidence() -> None:
    available = torch.tensor(
        [[True, True, True, True], [True, True, False, True]], dtype=torch.bool
    )
    weights = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    result = build_m03r_v12_posthoc_causal_action_mask(available, weights)
    assert torch.equal(
        result,
        torch.tensor(
            [[False, True, False, True], [False, True, False, True]],
            dtype=torch.bool,
        ),
    )
    assert result.device == available.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression boundary")
def test_causal_action_mask_accepts_cpu_weights_for_cuda_decisions() -> None:
    available = torch.ones((1, 4), dtype=torch.bool, device="cuda:0")
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    result = build_m03r_v12_posthoc_causal_action_mask(available, weights)
    assert result.device == available.device
    assert torch.equal(
        result,
        torch.tensor([[False, True, False, True]], device="cuda:0"),
    )


def test_label_validity_cannot_expand_the_causal_action_universe() -> None:
    inputs = _inputs()
    action = inputs.causal_action_mask.clone()
    action[0, 2] = False
    drifted = replace(inputs, causal_action_mask=action)
    with pytest.raises(M03RV12PosthocInferenceAuditError, match="inputs drifted"):
        drifted.validate()


def test_mean_and_rank_heads_are_scored_as_distinct_channels() -> None:
    inputs = _inputs()
    mean = build_m03r_v12_posthoc_audit_fold_evidence(inputs, _variant())
    rank = build_m03r_v12_posthoc_audit_fold_evidence(
        inputs, _variant("rank-score")
    )
    assert not torch.equal(mean.date_spearman_ic, rank.date_spearman_ic)
    assert not torch.equal(mean.target_weight_trace, rank.target_weight_trace)
    assert mean.input_receipt_sha256 == rank.input_receipt_sha256


def test_zero_and_sign_flipped_controls_are_target_blind() -> None:
    inputs = _inputs()
    original = build_m03r_v12_posthoc_audit_fold_evidence(inputs, _variant())
    zero = build_m03r_v12_posthoc_audit_fold_evidence(
        inputs, _variant(transform="zero")
    )
    flipped = build_m03r_v12_posthoc_audit_fold_evidence(
        inputs, _variant(transform="sign-flipped")
    )
    assert torch.equal(zero.date_spearman_ic, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(zero.date_top_bottom_spread, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(zero.active_mass, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(zero.policy_gross_returns, zero.benchmark_gross_returns)
    assert torch.allclose(flipped.date_spearman_ic, -original.date_spearman_ic)
    assert torch.allclose(
        flipped.date_top_bottom_spread, -original.date_top_bottom_spread
    )


def test_shuffled_control_is_deterministic_and_setting_bound() -> None:
    inputs = _inputs()
    variant = _variant(transform="shuffled")
    first = build_m03r_v12_posthoc_audit_fold_evidence(inputs, variant)
    second = build_m03r_v12_posthoc_audit_fold_evidence(inputs, variant)
    assert first.receipt_sha256 == second.receipt_sha256
    assert torch.equal(first.target_weight_trace, second.target_weight_trace)


def test_every_action_earns_one_post_fill_return_including_final_action() -> None:
    inputs = _inputs()
    result = build_m03r_v12_posthoc_audit_fold_evidence(inputs, _variant())
    assert result.policy_gross_returns.numel() == inputs.origin_indices.numel()
    assert result.target_weight_trace.shape[0] == inputs.origin_indices.numel()
    final_expected = (
        result.target_weight_trace[-1]
        * inputs.post_fill_asset_returns[-1]
    ).sum()
    assert result.policy_gross_returns[-1] == pytest.approx(float(final_expected))
    assert result.chronology_action_count_equals_return_count


def test_active_mass_ladder_is_monotone_and_never_exceeds_cap() -> None:
    inputs = _inputs()
    rows = tuple(
        build_m03r_v12_posthoc_audit_fold_evidence(inputs, _variant(cap=cap))
        for cap in (0.0025, 0.005, 0.01, 0.02)
    )
    means = [float(row.active_mass.mean()) for row in rows]
    assert means == sorted(means)
    for row, cap in zip(rows, (0.0025, 0.005, 0.01, 0.02), strict=True):
        assert bool((row.active_mass <= cap + 1.0e-12).all())


def test_allocator_preserves_a_float32_benchmark_row_sum() -> None:
    inputs = _inputs()
    benchmark = inputs.benchmark_target_weights.float()
    drifted = build_m03r_v12_posthoc_audit_inputs(
        setting_index=inputs.setting_index,
        fold_index=inputs.fold_index,
        checkpoint_file_sha256=inputs.checkpoint_file_sha256,
        checkpoint_model_state_sha256=inputs.checkpoint_model_state_sha256,
        source_array_sha256=inputs.source_array_sha256,
        asset_axis_sha256=inputs.asset_axis_sha256,
        action_mask_source_sha256=inputs.action_mask_source_sha256,
        post_fill_return_source_sha256=inputs.post_fill_return_source_sha256,
        origin_indices=inputs.origin_indices,
        raw_economic_mean=inputs.raw_economic_mean.float(),
        raw_rank_score=inputs.raw_rank_score.float(),
        economic_mean=inputs.economic_mean.float(),
        rank_score=inputs.rank_score.float(),
        selected_scale=inputs.selected_scale.float(),
        target_log_return=inputs.target_log_return.float(),
        label_valid=inputs.label_valid,
        causal_action_mask=inputs.causal_action_mask,
        fill_execution_mask=inputs.fill_execution_mask,
        post_fill_asset_returns=inputs.post_fill_asset_returns.float(),
        benchmark_target_weights=benchmark,
        asset_weight_caps=inputs.asset_weight_caps.float(),
    )
    result = build_m03r_v12_posthoc_audit_fold_evidence(drifted, _variant())
    assert torch.allclose(
        result.target_weight_trace.sum(1),
        benchmark.to(torch.float64).sum(1),
        atol=2.0e-10,
        rtol=0.0,
    )


def test_mutating_a_bound_input_array_fails_closed() -> None:
    inputs = _inputs()
    inputs.economic_mean[0, 1] += 1.0
    with pytest.raises(M03RV12PosthocInferenceAuditError, match="inputs drifted"):
        inputs.validate()


def test_panel_uses_one_concatenated_chronology_and_block_inference() -> None:
    folds = tuple(
        build_m03r_v12_posthoc_audit_fold_evidence(
            _inputs(fold_index=fold), _variant(cap=0.005)
        )
        for fold in range(6)
    )
    report = build_m03r_v12_posthoc_audit_panel_report(folds)
    assert len(report.fold_receipt_sha256) == 6
    assert len(report.gross_active_lcb_by_block) == 3
    assert len(report.net_10bp_lcb_by_block) == 3
    assert report.mean_active_mass == pytest.approx(0.005)
    assert not report.economic_generation_may_be_minted
    assert not report.outer_2026_accessed


def test_panel_rejects_an_incomplete_fold_family() -> None:
    row = build_m03r_v12_posthoc_audit_fold_evidence(_inputs(), _variant())
    with pytest.raises(M03RV12PosthocInferenceAuditError, match="six folds"):
        build_m03r_v12_posthoc_audit_panel_report((row,))
