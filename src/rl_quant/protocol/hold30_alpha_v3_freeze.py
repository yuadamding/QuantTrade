"""Fail-closed freeze schema for the Hold-30 alpha V3 experiment.

The renderer is pure and grants no launch authority.  V2 fold geometry may be
reused as implementation history, but V2 protocol generations and setting IDs
are rejected before a V3 manifest can be materialized.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.datasets.hold30_alpha import (
    HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA,
    HOLD30_ALPHA_FACTOR_USAGE,
    HOLD30_ALPHA_LABEL_RULE,
    HOLD30_ALPHA_LABEL_SCHEMA,
    HOLD30_ALPHA_MARKET_USAGE,
    HOLD30_ALPHA_RISK_FREE_USAGE,
    Hold30AlphaDataBindingReceipt,
    Hold30AlphaEvaluationPanel,
    Hold30ResidualAlphaLabels,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_C1_BENCHMARK_ID,
    HOLD30_ALPHA_C1_USAGE,
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_MECH8_SETTINGS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
    HOLD30_ALPHA_V3_DESIGN,
    HOLD30_ALPHA_V3_SUPERSEDED_GENERATION,
    hold30_alpha_v3_design_payload,
    validate_hold30_alpha_v3_artifact_identity,
)
from rl_quant.protocol.hold30_freeze import (
    HOLD30_FOLDS,
    HOLD30_GPU_PRODUCT,
    HOLD30_GPUS_PER_SETTING,
    HOLD30_SEEDS,
    Hold30FreezeError,
    render_hold30_folds,
    sha256_payload,
)
from rl_quant.training.hold30_alpha_plan import (
    Hold30AlphaTrainingPlan,
    Hold30AlphaTrainingPlanError,
)


class Hold30AlphaV3FreezeError(ValueError):
    """A launch-affecting V3 freeze invariant is missing or inconsistent."""


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30AlphaV3FreezeError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Hold30AlphaV3FreezeBindings:
    """Content-addressed inputs required before a V3 manifest may render."""

    repository_url: str
    git_commit: str
    git_tree: str
    clean_worktree: bool
    dirty_patch_sha256: str | None
    source_archive_sha256: str
    dependency_lock_sha256: str
    container_image_digest: str
    v3_rfc_sha256: str
    v3_adr_sha256: str
    v3_design_sha256: str
    v3_data_contract_sha256: str
    v3_checkpoint_contract_sha256: str
    superseded_v2_specification_sha256: str
    data_snapshot_sha256: str
    decision_axis_sha256: str
    split_arrays_sha256: str
    component_qualification_sha256: str
    software_qualification_sha256: str
    data_qualification_sha256: str
    capacity_qualification_sha256: str
    training_plan_sha256: str
    evaluation_plan_sha256: str
    artifact_inventory_sha256: str
    recovery_policy_sha256: str
    worker_template_sha256: str
    admitted_job_template_sha256: str
    namespace: str
    service_account: str
    executable_approval_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.repository_url:
            raise Hold30AlphaV3FreezeError("repository_url is required")
        for name in ("git_commit", "git_tree"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise Hold30AlphaV3FreezeError(
                    f"{name} must be a lowercase full 40-character Git SHA"
                )
        if self.clean_worktree and self.dirty_patch_sha256 is not None:
            raise Hold30AlphaV3FreezeError("a clean worktree cannot bind a dirty patch")
        if not self.clean_worktree and self.dirty_patch_sha256 is None:
            raise Hold30AlphaV3FreezeError("a dirty worktree must bind dirty_patch_sha256")
        digest_fields = (
            "source_archive_sha256",
            "dependency_lock_sha256",
            "v3_rfc_sha256",
            "v3_adr_sha256",
            "v3_design_sha256",
            "v3_data_contract_sha256",
            "v3_checkpoint_contract_sha256",
            "superseded_v2_specification_sha256",
            "data_snapshot_sha256",
            "decision_axis_sha256",
            "split_arrays_sha256",
            "component_qualification_sha256",
            "software_qualification_sha256",
            "data_qualification_sha256",
            "capacity_qualification_sha256",
            "training_plan_sha256",
            "evaluation_plan_sha256",
            "artifact_inventory_sha256",
            "recovery_policy_sha256",
            "worker_template_sha256",
            "admitted_job_template_sha256",
        )
        for name in digest_fields:
            _require_digest(name, getattr(self, name))
        if self.dirty_patch_sha256 is not None:
            _require_digest("dirty_patch_sha256", self.dirty_patch_sha256)
        if self.executable_approval_sha256 is not None:
            _require_digest("executable_approval_sha256", self.executable_approval_sha256)
        if not self.container_image_digest.startswith("sha256:"):
            raise Hold30AlphaV3FreezeError("container_image_digest must be digest-pinned")
        _require_digest("container_image_digest", self.container_image_digest[7:])
        if self.namespace != "yn-gpu-workload":
            raise Hold30AlphaV3FreezeError("V3 namespace must be yn-gpu-workload")
        if not self.service_account:
            raise Hold30AlphaV3FreezeError("service_account is required")


@dataclass(frozen=True, slots=True)
class Hold30AlphaV3DataContract:
    """Manifest-safe identities from the typed V3 data boundary."""

    protocol_generation: str
    source_axis_id: str
    training_benchmark_id: str
    c1_usage: tuple[str, ...]
    c1_trace_sha256: str
    panel_schema: str
    provenance_receipt_id: str
    panel_id: str
    binding_receipt_id: str
    label_schema: str
    label_rule: str
    labels_id: str
    horizons: tuple[int, ...]
    risk_free_usage: tuple[str, ...]
    market_usage: tuple[str, ...]
    factor_usage: tuple[str, ...]
    policy_feature_access: bool
    actor_access: bool
    auxiliary_only: bool

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaV3FreezeError("data contract must carry the V3 generation")
        if self.training_benchmark_id != HOLD30_ALPHA_C1_BENCHMARK_ID:
            raise Hold30AlphaV3FreezeError("data contract must bind the exact C1 benchmark")
        if self.c1_usage != HOLD30_ALPHA_C1_USAGE:
            raise Hold30AlphaV3FreezeError("C1 usage must remain action anchor and active benchmark")
        for name in (
            "source_axis_id",
            "c1_trace_sha256",
            "provenance_receipt_id",
            "panel_id",
            "binding_receipt_id",
            "labels_id",
        ):
            _require_digest(name, getattr(self, name))
        if self.panel_schema != HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA:
            raise Hold30AlphaV3FreezeError("evaluation panel schema drifted")
        if self.label_schema != HOLD30_ALPHA_LABEL_SCHEMA:
            raise Hold30AlphaV3FreezeError("residual-label schema drifted")
        if self.label_rule != HOLD30_ALPHA_LABEL_RULE:
            raise Hold30AlphaV3FreezeError("residual-label chronology drifted")
        if self.horizons != HOLD30_ALPHA_V3_DESIGN.alpha_horizons:
            raise Hold30AlphaV3FreezeError("residual-label horizons drifted")
        if self.risk_free_usage != HOLD30_ALPHA_RISK_FREE_USAGE:
            raise Hold30AlphaV3FreezeError("risk-free usage allowlist drifted")
        if self.market_usage != HOLD30_ALPHA_MARKET_USAGE:
            raise Hold30AlphaV3FreezeError("market usage allowlist drifted")
        if self.factor_usage != HOLD30_ALPHA_FACTOR_USAGE:
            raise Hold30AlphaV3FreezeError("factor usage allowlist drifted")
        if (
            self.policy_feature_access is not False
            or self.actor_access is not False
            or self.auxiliary_only is not True
        ):
            raise Hold30AlphaV3FreezeError(
                "residual labels and external return artifacts cannot enter the actor"
            )

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "protocol_generation": self.protocol_generation,
            "source_axis_id": self.source_axis_id,
            "training_benchmark": {
                "benchmark_id": self.training_benchmark_id,
                "usage": list(self.c1_usage),
                "c1_trace_sha256": self.c1_trace_sha256,
                "policy_feature_access": False,
            },
            "external_return_data": {
                "panel_schema": self.panel_schema,
                "provenance_receipt_id": self.provenance_receipt_id,
                "panel_id": self.panel_id,
                "binding_receipt_id": self.binding_receipt_id,
                "risk_free_usage": list(self.risk_free_usage),
                "market_usage": list(self.market_usage),
                "factor_usage": list(self.factor_usage),
                "policy_feature_access": False,
            },
            "residual_labels": {
                "label_schema": self.label_schema,
                "label_rule": self.label_rule,
                "labels_id": self.labels_id,
                "horizons": list(self.horizons),
                "actor_access": False,
                "auxiliary_only": True,
            },
        }


def bind_hold30_alpha_v3_data_contract(
    *,
    panel: Hold30AlphaEvaluationPanel,
    binding: Hold30AlphaDataBindingReceipt,
    labels: Hold30ResidualAlphaLabels,
) -> Hold30AlphaV3DataContract:
    """Bind only typed, already-materialized V3 data identities."""

    if not isinstance(panel, Hold30AlphaEvaluationPanel):
        raise Hold30AlphaV3FreezeError("typed Hold30AlphaEvaluationPanel is required")
    if not isinstance(binding, Hold30AlphaDataBindingReceipt):
        raise Hold30AlphaV3FreezeError("typed Hold30AlphaDataBindingReceipt is required")
    if not isinstance(labels, Hold30ResidualAlphaLabels):
        raise Hold30AlphaV3FreezeError("typed Hold30ResidualAlphaLabels is required")
    if (
        panel.source_axis_id != binding.source_axis_id
        or labels.source_axis_id != binding.source_axis_id
    ):
        raise Hold30AlphaV3FreezeError("panel, binding, and labels must share one axis")
    if panel.panel_id != binding.evaluation_panel_id:
        raise Hold30AlphaV3FreezeError("panel ID does not match the binding receipt")
    if panel.provenance.receipt_id != binding.evaluation_provenance_id:
        raise Hold30AlphaV3FreezeError("provenance ID does not match the binding receipt")
    return Hold30AlphaV3DataContract(
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        source_axis_id=binding.source_axis_id,
        training_benchmark_id=binding.c1_benchmark_id,
        c1_usage=HOLD30_ALPHA_C1_USAGE,
        c1_trace_sha256=binding.c1_trace_sha256,
        panel_schema=HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA,
        provenance_receipt_id=binding.evaluation_provenance_id,
        panel_id=binding.evaluation_panel_id,
        binding_receipt_id=binding.receipt_id,
        label_schema=HOLD30_ALPHA_LABEL_SCHEMA,
        label_rule=HOLD30_ALPHA_LABEL_RULE,
        labels_id=labels.labels_id,
        horizons=labels.horizons,
        risk_free_usage=panel.provenance.risk_free_usage,
        market_usage=panel.provenance.market_usage,
        factor_usage=panel.provenance.factor_usage,
        policy_feature_access=panel.provenance.policy_feature_access,
        actor_access=labels.actor_access,
        auxiliary_only=labels.auxiliary_only,
    )


def hold30_alpha_v3_trial_inventory() -> tuple[dict[str, Any], ...]:
    """Return the exact 8 settings x 6 folds x 5 paired seeds inventory."""

    return tuple(
        {
            "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
            "setting_index": setting.setting_index,
            "setting_id": setting.setting_id,
            "fold_index": fold_index,
            "seed": seed,
            "promotion_eligible": setting.promotion_eligible,
        }
        for setting in HOLD30_ALPHA_MECH8_SETTINGS
        for fold_index in range(HOLD30_FOLDS)
        for seed in HOLD30_SEEDS
    )


def render_hold30_alpha_v3_manifest(
    decision_axis: Sequence[str],
    bindings: Hold30AlphaV3FreezeBindings,
    data_contract: Hold30AlphaV3DataContract,
    training_plan: Hold30AlphaTrainingPlan,
    *,
    protocol_generation: str,
    setting_ids: Sequence[str],
    approval_state: str = "dry_run",
) -> dict[str, Any]:
    """Render a deterministic V3 manifest without discovering or launching work."""

    if protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
        if protocol_generation == HOLD30_ALPHA_V3_SUPERSEDED_GENERATION:
            raise Hold30AlphaV3FreezeError(
                "V2 was superseded before launch and cannot produce V3 artifacts"
            )
        raise Hold30AlphaV3FreezeError("manifest protocol generation is not V3")
    ids = tuple(setting_ids)
    if ids != HOLD30_ALPHA_MECH8_IDS:
        v2_ids = any(value.startswith(("hold30-m", "hold30-a")) for value in ids)
        if v2_ids:
            raise Hold30AlphaV3FreezeError("V2 setting IDs are invalid in a V3 manifest")
        raise Hold30AlphaV3FreezeError("V3 setting IDs must match the exact ordered inventory")
    for setting_id in ids:
        validate_hold30_alpha_v3_artifact_identity(
            protocol_generation=protocol_generation,
            setting_id=setting_id,
        )

    if approval_state not in {"dry_run", "software_qualified", "executable"}:
        raise Hold30AlphaV3FreezeError(
            "approval_state must be dry_run, software_qualified, or executable"
        )
    if approval_state == "executable" and bindings.executable_approval_sha256 is None:
        raise Hold30AlphaV3FreezeError(
            "executable manifests require executable_approval_sha256"
        )
    if not isinstance(training_plan, Hold30AlphaTrainingPlan):
        raise Hold30AlphaV3FreezeError(
            "manifest rendering requires a typed Hold30AlphaTrainingPlan"
        )
    if training_plan.receipt_id != bindings.training_plan_sha256:
        raise Hold30AlphaV3FreezeError("bound typed training-plan digest does not match")
    if approval_state == "executable":
        try:
            training_plan.require_resolved()
        except Hold30AlphaTrainingPlanError as exc:
            raise Hold30AlphaV3FreezeError(
                f"typed training plan is unresolved: {exc}"
            ) from exc
    if approval_state != "executable" and bindings.executable_approval_sha256 is not None:
        raise Hold30AlphaV3FreezeError(
            "executable approval cannot be attached to a non-executable manifest"
        )

    try:
        folds = render_hold30_folds(decision_axis)
    except Hold30FreezeError as exc:
        raise Hold30AlphaV3FreezeError(str(exc)) from exc
    axis_digest = sha256_payload(tuple(decision_axis))
    if axis_digest != bindings.decision_axis_sha256:
        raise Hold30AlphaV3FreezeError("bound decision-axis digest does not match")
    fold_payload = [asdict(fold) for fold in folds]
    if sha256_payload(fold_payload) != bindings.split_arrays_sha256:
        raise Hold30AlphaV3FreezeError("bound split-arrays digest does not match")

    checkpoint_payload = asdict(training_plan.checkpoint_contract)
    design_payload = hold30_alpha_v3_design_payload(
        checkpoint_contract=training_plan.checkpoint_contract
    )
    if sha256_payload(design_payload) != bindings.v3_design_sha256:
        raise Hold30AlphaV3FreezeError("bound V3 design digest does not match")
    if sha256_payload(checkpoint_payload) != bindings.v3_checkpoint_contract_sha256:
        raise Hold30AlphaV3FreezeError("bound checkpoint-contract digest does not match")
    training_plan_payload = training_plan.manifest_payload()
    if not (
        design_payload["design"]["checkpoint"]
        == training_plan_payload["checkpoint_contract"]
        == checkpoint_payload
    ):
        raise Hold30AlphaV3FreezeError(
            "design, training plan, and manifest checkpoint contracts differ"
        )

    if not isinstance(data_contract, Hold30AlphaV3DataContract):
        raise Hold30AlphaV3FreezeError(
            "manifest rendering requires a typed Hold30AlphaV3DataContract"
        )
    data_payload = data_contract.manifest_payload()
    if sha256_payload(data_payload) != bindings.v3_data_contract_sha256:
        raise Hold30AlphaV3FreezeError("bound V3 data-contract digest does not match")

    inventory = hold30_alpha_v3_trial_inventory()
    payload: dict[str, Any] = {
        "schema_version": 3,
        "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
        "supersedes": HOLD30_ALPHA_V3_SUPERSEDED_GENERATION,
        "superseded_before_launch": True,
        "v2_artifacts_reusable_as_implementation_history_only": True,
        "approval_state": approval_state,
        "render_grants_launch_authority": False,
        "lockbox_consumed": False,
        "design": design_payload,
        "checkpoint_contract": checkpoint_payload,
        "checkpoint_contract_source": "typed-training-plan",
        "data_contract": data_payload,
        "training_plan": training_plan_payload,
        "decision_axis": {
            "count": len(decision_axis),
            "first": decision_axis[0],
            "last": decision_axis[-1],
            "sha256": axis_digest,
        },
        "folds": fold_payload,
        "settings": [asdict(setting) for setting in HOLD30_ALPHA_MECH8_SETTINGS],
        "trial_inventory_count": len(inventory),
        "trial_inventory_sha256": sha256_payload(inventory),
        "compute": {
            "gpu_product": HOLD30_GPU_PRODUCT,
            "gpus_per_setting": HOLD30_GPUS_PER_SETTING,
            "world_size_per_trial": 2,
            "local_paths_per_rank": 1,
            "rank_sharding": "distinct-global-paths",
            "effective_paths_per_trial": 2,
            "concurrent_setting_workers": 8,
            "maximum_h100": 16,
            "namespace": bindings.namespace,
            "service_account": bindings.service_account,
            "worker_template_sha256": bindings.worker_template_sha256,
            "admitted_job_template_sha256": bindings.admitted_job_template_sha256,
            "scientific_fields_inferred_from_gpu_count": False,
        },
        "bindings": asdict(bindings),
    }
    if not (
        payload["design"]["design"]["checkpoint"]
        == payload["training_plan"]["checkpoint_contract"]
        == payload["checkpoint_contract"]
    ):
        raise Hold30AlphaV3FreezeError(
            "rendered manifest contains conflicting checkpoint contracts"
        )
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


__all__ = [
    "Hold30AlphaV3DataContract",
    "Hold30AlphaV3FreezeBindings",
    "Hold30AlphaV3FreezeError",
    "bind_hold30_alpha_v3_data_contract",
    "hold30_alpha_v3_trial_inventory",
    "render_hold30_alpha_v3_manifest",
]
