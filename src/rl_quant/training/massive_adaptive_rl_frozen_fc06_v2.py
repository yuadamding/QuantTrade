"""Create-only V5 frozen FC06 control paired with a Selection-V4 fold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    registered_massive_adaptive_rl_constant_actions_v1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_fit_runner_v1 import (
    MassiveAdaptiveRLFixedControlFitAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_selection_v1 import (
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v4 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV4,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)


MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_DATASET = "massive-adaptive-rl-frozen-fc06-v2"
MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA,
        "encoding": "canonical-json-exact-fc06-action-and-v5-lineage",
        "selection": "exact-policy-selection-v4-fold",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLFrozenFC06V2Error(ValueError):
    """The fit-selected FC06 control or Selection-V4 lineage differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFrozenFC06V2Error(f"{name} must be a lowercase SHA-256")
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLFrozenFC06V2Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLFrozenFC06V2Error(f"{name} is absent or invalid")
    return value


def _wall_clock_after(value: int) -> int:
    now = time.time_ns() // 1_000_000
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise MassiveAdaptiveRLFrozenFC06V2Error("frozen FC06 clock differs")
    return max(now, value + 1)


def frozen_fc06_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLFrozenFC06V2Error("frozen FC06 fold differs")
    return f"adaptive-rl/{manifest.experiment_id}/frozen-fc06-v2/fold-{fold_index}.json"


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFrozenFC06V2:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    validation_release_authority_receipt_sha256: str
    fold_validation_authority_receipt_sha256: str
    policy_selection_authority_receipt_sha256: str
    policy_selection_source_receipt_sha256: str
    policy_selection_commit_receipt_sha256: str
    policy_selection_committed_at_ms: int
    fold_index: int
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    fixed_control_selection_source_receipt_sha256: str
    fixed_control_selection_commit_receipt_sha256: str
    fixed_control_selection_committed_at_ms: int
    selected_control_id: str
    selected_action_receipt_sha256: str
    selected_action_values: tuple[float, ...]
    selected_candidate_validation_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_control_replayed: bool = False
    development_outer_control_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256
    implementation_source_sha256: str = MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA
    _runtime_selection: MassiveAdaptiveRLPolicySelectionAuthorityV4 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_fixed_selection: (
        MassiveAdaptiveRLFixedControlSelectionAuthorityV1 | None
    ) = field(default=None, compare=False, repr=False)
    _runtime_action: MassiveAdaptiveRLActionV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_control_replayed",
                "development_outer_control_authorized",
            }
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_control_replayed
            and self.development_outer_control_authorized
            and self.source_data_qualified
        )

    @property
    def runtime_action(self) -> MassiveAdaptiveRLActionV1:
        self.validate()
        if self._runtime_action is None or not self.development_stage_authorized:
            raise MassiveAdaptiveRLFrozenFC06V2Error(
                "frozen FC06 action has not been exactly replayed"
            )
        return self._runtime_action

    def validate(self) -> None:
        runtime_present = all(
            value is not None
            for value in (
                self._runtime_selection,
                self._runtime_fixed_selection,
                self._runtime_action,
            )
        )
        any_runtime = any(
            value is not None
            for value in (
                self._runtime_selection,
                self._runtime_fixed_selection,
                self._runtime_action,
            )
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or not self.selected_control_id
            or len(self.selected_action_values) != 10
            or any(
                type(value) is not float or not -1.0 <= value <= 1.0
                for value in self.selected_action_values
            )
            or not isinstance(self.selected_candidate_validation_eligible, bool)
            or isinstance(self.policy_selection_committed_at_ms, bool)
            or not isinstance(self.policy_selection_committed_at_ms, int)
            or self.policy_selection_committed_at_ms < 0
            or isinstance(self.fixed_control_selection_committed_at_ms, bool)
            or not isinstance(self.fixed_control_selection_committed_at_ms, int)
            or self.fixed_control_selection_committed_at_ms < 0
            or self.fixed_control_selection_committed_at_ms
            >= self.policy_selection_committed_at_ms
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime_present
            or self.runtime_control_replayed != runtime_present
            or self.development_outer_control_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFrozenFC06V2Error("frozen FC06 V2 differs")
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.policy_selection_authority_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.policy_selection_committed_at_ms
            ):
                raise MassiveAdaptiveRLFrozenFC06V2Error(
                    "frozen FC06 source transaction differs"
                )
        if runtime_present:
            assert self._runtime_selection is not None
            assert self._runtime_fixed_selection is not None
            assert self._runtime_action is not None
            self._runtime_selection.validate()
            self._runtime_fixed_selection.validate()
            self._runtime_action.validate()
            runtime_selection = self._runtime_fixed_selection.runtime_selection
            fold_validation = self._runtime_selection.fold_validation_authority
            fold_fit = (
                fold_validation.release_authority.four_fold_fit_authority.fold_fit(
                    self.fold_index
                )
            )
            fixed_fit = (
                fold_fit.training_workflow.runtime_workflow.fixed_control_fit_authority
            )
            fixed_fit.validate()
            action_values = (
                *self._runtime_action.bucket_controls,
                self._runtime_action.uncertainty_control,
                self._runtime_action.risk_control,
                self._runtime_action.trade_cost_control,
            )
            if (
                runtime_selection is None
                or not self._runtime_selection.development_stage_authorized
                or self._runtime_selection.semantic_receipt_sha256
                != self.policy_selection_authority_receipt_sha256
                or self._runtime_selection.source_receipt_sha256
                != self.policy_selection_source_receipt_sha256
                or self._runtime_selection.source_transaction_receipt_sha256
                != self.policy_selection_commit_receipt_sha256
                or self._runtime_selection.source_transaction_committed_at_ms
                != self.policy_selection_committed_at_ms
                or self._runtime_selection.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_selection.scientific_protocol_projection_sha256
                != self.scientific_protocol_projection_sha256
                or self._runtime_selection.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_selection.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_selection.validation_release_authority_receipt_sha256
                != self.validation_release_authority_receipt_sha256
                or self._runtime_selection.fold_validation_authority_receipt_sha256
                != self.fold_validation_authority_receipt_sha256
                or self._runtime_selection.fold_index != self.fold_index
                or self._runtime_selection.fixed_control_selection_authority_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or fold_validation.fixed_control_fit_authority_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or fixed_fit.semantic_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or self._runtime_selection.selected_candidate_validation_eligible
                != self.selected_candidate_validation_eligible
                or self._runtime_fixed_selection.loaded_source.receipt.receipt_sha256
                != self.fixed_control_selection_source_receipt_sha256
                or self._runtime_fixed_selection.loaded_source.commit.receipt_sha256
                != self.fixed_control_selection_commit_receipt_sha256
                or self._runtime_fixed_selection.loaded_source.commit.committed_at_ms
                != self.fixed_control_selection_committed_at_ms
                or runtime_selection.selected_control_id != self.selected_control_id
                or runtime_selection.selected_action_receipt_sha256
                != self.selected_action_receipt_sha256
                or self._runtime_action.semantic_receipt_sha256
                != self.selected_action_receipt_sha256
                or action_values != self.selected_action_values
                or self.source_data_qualified
                != bool(
                    self._runtime_selection.source_data_qualified
                    and fixed_fit.source_data_qualified
                    and self._runtime_fixed_selection.source_data_qualified
                )
            ):
                raise MassiveAdaptiveRLFrozenFC06V2Error(
                    "frozen FC06 runtime replay differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _expected(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    selection: MassiveAdaptiveRLPolicySelectionAuthorityV4,
) -> tuple[
    dict[str, object],
    MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    MassiveAdaptiveRLActionV1,
]:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(selection) is not MassiveAdaptiveRLPolicySelectionAuthorityV4
    ):
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 requires exact Manifest V5 and Selection V4"
        )
    manifest.validate()
    selection.validate()
    fold_validation = selection.fold_validation_authority
    fold_fit = fold_validation.release_authority.four_fold_fit_authority.fold_fit(
        selection.fold_index
    )
    workflow = fold_fit.training_workflow.runtime_workflow
    fixed_fit = workflow.fixed_control_fit_authority
    fixed_selection = workflow.fixed_control_selection_authority
    if (
        type(fixed_fit) is not MassiveAdaptiveRLFixedControlFitAuthorityV1
        or type(fixed_selection)
        is not MassiveAdaptiveRLFixedControlSelectionAuthorityV1
    ):
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 requires exact fit and selection authorities"
        )
    fixed_selection.validate()
    runtime_selection = fixed_selection.runtime_selection
    action_by_receipt = {
        action.semantic_receipt_sha256: action
        for _control_id, action in registered_massive_adaptive_rl_constant_actions_v1()
    }
    if (
        not selection.development_stage_authorized
        or runtime_selection is None
        or fixed_selection.semantic_receipt_sha256
        != selection.fixed_control_selection_authority_receipt_sha256
        or runtime_selection.selected_action_receipt_sha256 not in action_by_receipt
    ):
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "fit-selected FC06 differs from Selection V4"
        )
    action = action_by_receipt[runtime_selection.selected_action_receipt_sha256]
    values = (
        *action.bucket_controls,
        action.uncertainty_control,
        action.risk_control,
        action.trade_cost_control,
    )
    body: dict[str, object] = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": selection.execution_implementation_registration_receipt_sha256,
        "scientific_execution_fingerprint_sha256": selection.scientific_execution_fingerprint_sha256,
        "validation_release_authority_receipt_sha256": selection.validation_release_authority_receipt_sha256,
        "fold_validation_authority_receipt_sha256": selection.fold_validation_authority_receipt_sha256,
        "policy_selection_authority_receipt_sha256": selection.semantic_receipt_sha256,
        "policy_selection_source_receipt_sha256": _required_digest(
            "selection source", selection.source_receipt_sha256
        ),
        "policy_selection_commit_receipt_sha256": _required_digest(
            "selection commit", selection.source_transaction_receipt_sha256
        ),
        "policy_selection_committed_at_ms": _required_time(
            "selection time", selection.source_transaction_committed_at_ms
        ),
        "fold_index": selection.fold_index,
        "fixed_control_fit_authority_receipt_sha256": fixed_fit.semantic_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": fixed_selection.semantic_receipt_sha256,
        "fixed_control_selection_source_receipt_sha256": fixed_selection.loaded_source.receipt.receipt_sha256,
        "fixed_control_selection_commit_receipt_sha256": fixed_selection.loaded_source.commit.receipt_sha256,
        "fixed_control_selection_committed_at_ms": fixed_selection.loaded_source.commit.committed_at_ms,
        "selected_control_id": runtime_selection.selected_control_id,
        "selected_action_receipt_sha256": action.semantic_receipt_sha256,
        "selected_action_values": values,
        "selected_candidate_validation_eligible": selection.selected_candidate_validation_eligible,
        "source_data_qualified": bool(
            selection.source_data_qualified
            and fixed_fit.source_data_qualified
            and fixed_selection.source_data_qualified
        ),
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SCHEMA,
    }
    return body, fixed_selection, action


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFrozenFC06V2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    body["selected_action_values"] = tuple(
        cast(Sequence[float], body["selected_action_values"])
    )
    result = MassiveAdaptiveRLFrozenFC06V2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_frozen_fc06_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    verified_at_ms: int | None = None,
) -> MassiveAdaptiveRLFrozenFC06V2:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=frozen_fc06_relative_path_v2(
                manifest=manifest, fold_index=fold_index
            ),
            verified_at_ms=(
                time.time_ns() // 1_000_000
                if verified_at_ms is None
                else verified_at_ms
            ),
        ),
    )


def run_or_resume_massive_adaptive_rl_frozen_fc06_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    selection: MassiveAdaptiveRLPolicySelectionAuthorityV4,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFrozenFC06V2:
    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 materialization mode differs"
        )
    metadata, fixed_selection, action = _expected(
        manifest=manifest, selection=selection
    )
    relative = frozen_fc06_relative_path_v2(
        manifest=manifest, fold_index=selection.fold_index
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 transaction is incomplete"
        )
    if not complete:
        if not allow_materialize:
            raise MassiveAdaptiveRLFrozenFC06V2Error(
                "frozen FC06 is absent in read-only mode"
            )
        committed_at_ms = _wall_clock_after(
            _required_time(
                "selection time", selection.source_transaction_committed_at_ms
            )
        )
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(metadata)),
            root=root,
            relative_payload_path=relative,
            dataset_id=MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_DATASET,
            source_object_key=relative,
            requested_at_ms=committed_at_ms,
            downloaded_at_ms=committed_at_ms,
            schema_sha256=MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SCHEMA_SHA256,
            entitlement_receipt_sha256=selection.semantic_receipt_sha256,
            committed_at_ms=committed_at_ms,
            request_id=f"ADAPTIVE-RL-V5-FROZEN-FC06-{selection.fold_index}",
        )
    persisted = load_massive_adaptive_rl_frozen_fc06_v2(
        root=root, manifest=manifest, fold_index=selection.fold_index
    )
    if persisted.semantic_unsigned() != metadata:
        raise MassiveAdaptiveRLFrozenFC06V2Error(
            "frozen FC06 does not replay from Selection V4"
        )
    result = replace(
        persisted,
        runtime_control_replayed=True,
        development_outer_control_authorized=persisted.source_data_qualified,
        _runtime_selection=selection,
        _runtime_fixed_selection=fixed_selection,
        _runtime_action=action,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FROZEN_FC06_V2_SOURCE_SHA256",
    "MassiveAdaptiveRLFrozenFC06V2",
    "MassiveAdaptiveRLFrozenFC06V2Error",
    "frozen_fc06_relative_path_v2",
    "load_massive_adaptive_rl_frozen_fc06_v2",
    "run_or_resume_massive_adaptive_rl_frozen_fc06_v2",
]
