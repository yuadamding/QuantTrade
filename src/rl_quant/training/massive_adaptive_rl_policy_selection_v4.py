"""V5-native policy selection over an exact FoldValidation-V3 authority."""

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
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v3 import (
    MassiveAdaptiveRLFoldValidationAuthorityV3,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    MassiveAdaptiveRLCheckpointAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MassiveAdaptiveRLPolicySelectionV2,
    select_massive_adaptive_rl_policy_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA,
    MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)


MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_DATASET = (
    "massive-adaptive-rl-policy-selection-authority-v4"
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA,
            "encoding": "canonical-json-exact-fold-validation-v3-selection",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLPolicySelectionAuthorityV4Error(ValueError):
    """The V5 fold evidence, ranking replay, or selected checkpoint differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            f"{name} is absent or invalid"
        )
    return value


def _wall_clock_after(value: int) -> int:
    now = time.time_ns() // 1_000_000
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection clock differs"
        )
    return max(now, value + 1)


def policy_selection_authority_relative_path_v4(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection fold differs"
        )
    return (
        f"adaptive-rl/{manifest.experiment_id}/policy-selection-v4/"
        f"fold-{fold_index}.json"
    )


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
class MassiveAdaptiveRLPolicySelectionAuthorityV4:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    validation_release_authority_receipt_sha256: str
    fold_validation_authority_receipt_sha256: str
    fold_validation_source_receipt_sha256: str
    fold_validation_commit_receipt_sha256: str
    fold_validation_committed_at_ms: int
    fold_fit_authority_receipt_sha256: str
    fold_index: int
    selection_computation_receipt_sha256: str
    selected_checkpoint_authority_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    selected_update_index: int
    selected_candidate_receipt_sha256: str
    selected_candidate_validation_eligible: bool
    validation_eligibility_failures: tuple[str, ...]
    selection_pool_kind: str
    expected_candidate_checkpoint_authority_receipts: tuple[str, ...]
    candidate_receipts: tuple[str, ...]
    candidate_inventory_sha256: str
    ranked_candidate_receipts: tuple[str, ...]
    ranked_candidate_inventory_sha256: str
    training_forecast_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_selection_replayed: bool = False
    development_policy_selection_authorized: bool = False
    policy_freezing_authorized: bool = False
    outer_diagnostic_preparation_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_fold_validation: MassiveAdaptiveRLFoldValidationAuthorityV3 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_selection_computation: MassiveAdaptiveRLPolicySelectionV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_selected_checkpoint: MassiveAdaptiveRLCheckpointAuthorityV1 | None = field(
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
                "runtime_selection_replayed",
                "development_policy_selection_authorized",
                "policy_freezing_authorized",
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
            and self.runtime_selection_replayed
            and self.development_policy_selection_authorized
            and self.policy_freezing_authorized
            and self.source_data_qualified
        )

    @property
    def positive_profitability_authorization_eligible(self) -> bool:
        return bool(
            self.development_stage_authorized
            and self.selected_candidate_validation_eligible
        )

    @property
    def selected_checkpoint_authority(self) -> MassiveAdaptiveRLCheckpointAuthorityV1:
        self.validate()
        if (
            self._runtime_selected_checkpoint is None
            or not self.development_stage_authorized
        ):
            raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                "selected checkpoint has not been exactly replayed"
            )
        return self._runtime_selected_checkpoint

    @property
    def fold_validation_authority(self) -> MassiveAdaptiveRLFoldValidationAuthorityV3:
        self.validate()
        if (
            self._runtime_fold_validation is None
            or not self.development_stage_authorized
        ):
            raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                "fold-validation authority has not been exactly replayed"
            )
        return self._runtime_fold_validation

    def validate(self) -> None:
        runtime_present = all(
            value is not None
            for value in (
                self._runtime_manifest,
                self._runtime_fold_validation,
                self._runtime_selection_computation,
                self._runtime_selected_checkpoint,
            )
        )
        any_runtime = any(
            value is not None
            for value in (
                self._runtime_manifest,
                self._runtime_fold_validation,
                self._runtime_selection_computation,
                self._runtime_selected_checkpoint,
            )
        )
        candidate_count = self.fold_index + 1
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.fold_validation_committed_at_ms, bool)
            or not isinstance(self.fold_validation_committed_at_ms, int)
            or self.fold_validation_committed_at_ms < 0
            or isinstance(self.selected_update_index, bool)
            or not isinstance(self.selected_update_index, int)
            or self.selected_update_index < 0
            or len(self.expected_candidate_checkpoint_authority_receipts)
            != candidate_count
            or len(set(self.expected_candidate_checkpoint_authority_receipts))
            != candidate_count
            or len(self.candidate_receipts) != candidate_count
            or len(set(self.candidate_receipts)) != candidate_count
            or len(self.ranked_candidate_receipts) != candidate_count
            or set(self.ranked_candidate_receipts) != set(self.candidate_receipts)
            or self.selected_checkpoint_authority_receipt_sha256
            not in self.expected_candidate_checkpoint_authority_receipts
            or self.selected_candidate_receipt_sha256 not in self.candidate_receipts
            or self.candidate_inventory_sha256
            != semantic_sha256(self.candidate_receipts)
            or self.ranked_candidate_inventory_sha256
            != semantic_sha256(self.ranked_candidate_receipts)
            or any(
                not isinstance(value, str) or not value
                for value in self.validation_eligibility_failures
            )
            or self.validation_eligibility_failures
            != tuple(sorted(set(self.validation_eligibility_failures)))
            or not isinstance(self.selected_candidate_validation_eligible, bool)
            or self.selected_candidate_validation_eligible
            != (not self.validation_eligibility_failures)
            or self.selection_pool_kind not in {"eligible", "all-no-eligible"}
            or (self.selection_pool_kind == "eligible")
            != self.selected_candidate_validation_eligible
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime_present
            or self.runtime_selection_replayed != runtime_present
            or self.development_policy_selection_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.policy_freezing_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.outer_diagnostic_preparation_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                "policy-selection authority V4 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for inventory in (
            self.expected_candidate_checkpoint_authority_receipts,
            self.candidate_receipts,
            self.ranked_candidate_receipts,
        ):
            for value in inventory:
                _digest("policy-selection inventory", value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.fold_validation_authority_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.fold_validation_committed_at_ms
            ):
                raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                    "policy-selection source transaction differs"
                )
        if runtime_present:
            assert self._runtime_manifest is not None
            assert self._runtime_fold_validation is not None
            assert self._runtime_selection_computation is not None
            assert self._runtime_selected_checkpoint is not None
            for authority in (
                self._runtime_manifest,
                self._runtime_fold_validation,
                self._runtime_selection_computation,
                self._runtime_selected_checkpoint,
            ):
                authority.validate()
            witness = self._runtime_selection_computation
            if (
                not self._runtime_fold_validation.development_stage_authorized
                or self._runtime_manifest.semantic_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_fold_validation.semantic_receipt_sha256
                != self.fold_validation_authority_receipt_sha256
                or self._runtime_fold_validation.source_receipt_sha256
                != self.fold_validation_source_receipt_sha256
                or self._runtime_fold_validation.source_transaction_receipt_sha256
                != self.fold_validation_commit_receipt_sha256
                or self._runtime_fold_validation.source_transaction_committed_at_ms
                != self.fold_validation_committed_at_ms
                or self._runtime_fold_validation.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_fold_validation.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_fold_validation.release_authority_receipt_sha256
                != self.validation_release_authority_receipt_sha256
                or self._runtime_fold_validation.fold_fit_authority_receipt_sha256
                != self.fold_fit_authority_receipt_sha256
                or self._runtime_fold_validation.expected_candidate_checkpoint_authority_receipts
                != self.expected_candidate_checkpoint_authority_receipts
                or self._runtime_fold_validation.candidate_receipts
                != self.candidate_receipts
                or witness.semantic_receipt_sha256
                != self.selection_computation_receipt_sha256
                or witness.selected_checkpoint_authority_receipt_sha256
                != self.selected_checkpoint_authority_receipt_sha256
                or witness.selected_checkpoint_receipt_sha256
                != self.selected_checkpoint_receipt_sha256
                or witness.selected_model_state_receipt_sha256
                != self.selected_model_state_receipt_sha256
                or witness.selected_update_index != self.selected_update_index
                or witness.selected_candidate_receipt_sha256
                != self.selected_candidate_receipt_sha256
                or witness.selected_candidate_validation_eligible
                != self.selected_candidate_validation_eligible
                or witness.validation_eligibility_failures
                != self.validation_eligibility_failures
                or witness.ranked_candidate_receipts != self.ranked_candidate_receipts
                or witness.candidate_inventory_sha256 != self.candidate_inventory_sha256
                or witness.ranked_candidate_inventory_sha256
                != self.ranked_candidate_inventory_sha256
                or witness.training_forecast_authority_receipt_sha256
                != self.training_forecast_authority_receipt_sha256
                or witness.fixed_control_selection_authority_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or self._runtime_selected_checkpoint.semantic_receipt_sha256
                != self.selected_checkpoint_authority_receipt_sha256
                or self._runtime_selected_checkpoint.checkpoint_receipt_sha256
                != self.selected_checkpoint_receipt_sha256
                or self._runtime_selected_checkpoint.model_state_receipt_sha256
                != self.selected_model_state_receipt_sha256
                or self.source_data_qualified
                != bool(
                    self._runtime_fold_validation.source_data_qualified
                    and witness.source_data_qualified
                )
            ):
                raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                    "policy-selection runtime replay differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_policy_selection_authority_v4(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_validation: MassiveAdaptiveRLFoldValidationAuthorityV3,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(fold_validation) is not MassiveAdaptiveRLFoldValidationAuthorityV3
    ):
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy selection V4 requires exact Manifest V5 and FoldValidation V3"
        )
    manifest.validate()
    fold_validation.validate()
    if (
        not fold_validation.development_stage_authorized
        or fold_validation.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy selection V4 fold evidence is not authorized"
        )
    witness = select_massive_adaptive_rl_policy_v2(
        manifest=manifest.base_manifest,
        fold_fit_authority_receipt_sha256=fold_validation.fold_fit_authority_receipt_sha256,
        expected_candidate_checkpoint_authority_receipts=(
            fold_validation.expected_candidate_checkpoint_authority_receipts
        ),
        candidates=fold_validation.candidates,
    )
    fold_fit = fold_validation.release_authority.four_fold_fit_authority.fold_fit(
        fold_validation.fold_index
    )
    checkpoints = (
        fold_fit.training_workflow.runtime_workflow.policy_checkpoint_authorities
    )
    selected_by_receipt = {row.semantic_receipt_sha256: row for row in checkpoints}
    try:
        selected = selected_by_receipt[
            witness.selected_checkpoint_authority_receipt_sha256
        ]
    except KeyError as error:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "selected checkpoint is outside the released inventory"
        ) from error
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": fold_validation.execution_implementation_registration_receipt_sha256,
        "scientific_execution_fingerprint_sha256": fold_validation.scientific_execution_fingerprint_sha256,
        "validation_release_authority_receipt_sha256": fold_validation.release_authority_receipt_sha256,
        "fold_validation_authority_receipt_sha256": fold_validation.semantic_receipt_sha256,
        "fold_validation_source_receipt_sha256": _required_digest(
            "fold validation source", fold_validation.source_receipt_sha256
        ),
        "fold_validation_commit_receipt_sha256": _required_digest(
            "fold validation commit", fold_validation.source_transaction_receipt_sha256
        ),
        "fold_validation_committed_at_ms": _required_time(
            "fold validation time", fold_validation.source_transaction_committed_at_ms
        ),
        "fold_fit_authority_receipt_sha256": fold_validation.fold_fit_authority_receipt_sha256,
        "fold_index": fold_validation.fold_index,
        "selection_computation_receipt_sha256": witness.semantic_receipt_sha256,
        "selected_checkpoint_authority_receipt_sha256": witness.selected_checkpoint_authority_receipt_sha256,
        "selected_checkpoint_receipt_sha256": witness.selected_checkpoint_receipt_sha256,
        "selected_model_state_receipt_sha256": witness.selected_model_state_receipt_sha256,
        "selected_update_index": witness.selected_update_index,
        "selected_candidate_receipt_sha256": witness.selected_candidate_receipt_sha256,
        "selected_candidate_validation_eligible": witness.selected_candidate_validation_eligible,
        "validation_eligibility_failures": witness.validation_eligibility_failures,
        "selection_pool_kind": witness.selection_pool_kind,
        "expected_candidate_checkpoint_authority_receipts": witness.expected_candidate_checkpoint_authority_receipts,
        "candidate_receipts": witness.candidate_receipts,
        "candidate_inventory_sha256": witness.candidate_inventory_sha256,
        "ranked_candidate_receipts": witness.ranked_candidate_receipts,
        "ranked_candidate_inventory_sha256": witness.ranked_candidate_inventory_sha256,
        "training_forecast_authority_receipt_sha256": witness.training_forecast_authority_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": witness.fixed_control_selection_authority_receipt_sha256,
        "source_data_qualified": bool(
            fold_validation.source_data_qualified and witness.source_data_qualified
        ),
        "outer_diagnostic_preparation_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SCHEMA,
    }
    provisional = MassiveAdaptiveRLPolicySelectionAuthorityV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=bool(body["source_data_qualified"]),
        policy_freezing_authorized=bool(body["source_data_qualified"]),
        _runtime_manifest=manifest,
        _runtime_fold_validation=fold_validation,
        _runtime_selection_computation=witness,
        _runtime_selected_checkpoint=selected,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "validation_eligibility_failures",
        "expected_candidate_checkpoint_authority_receipts",
        "candidate_receipts",
        "ranked_candidate_receipts",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    result = MassiveAdaptiveRLPolicySelectionAuthorityV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_policy_selection_authority_v4(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    verified_at_ms: int | None = None,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=policy_selection_authority_relative_path_v4(
                manifest=manifest, fold_index=fold_index
            ),
            verified_at_ms=(
                time.time_ns() // 1_000_000
                if verified_at_ms is None
                else verified_at_ms
            ),
        ),
    )


def run_or_resume_massive_adaptive_rl_policy_selection_authority_v4(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_validation: MassiveAdaptiveRLFoldValidationAuthorityV3,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLPolicySelectionAuthorityV4:
    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection materialization mode differs"
        )
    expected = build_massive_adaptive_rl_policy_selection_authority_v4(
        manifest=manifest, fold_validation=fold_validation
    )
    relative = policy_selection_authority_relative_path_v4(
        manifest=manifest, fold_index=fold_validation.fold_index
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection transaction is incomplete"
        )
    if not complete:
        if not allow_materialize:
            raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
                "policy-selection authority is absent in read-only mode"
            )
        committed_at_ms = _wall_clock_after(expected.fold_validation_committed_at_ms)
        publish_massive_source_object(
            stream=BytesIO(canonical_json_file_bytes(expected.semantic_unsigned())),
            root=root,
            relative_payload_path=relative,
            dataset_id=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_DATASET,
            source_object_key=relative,
            requested_at_ms=committed_at_ms,
            downloaded_at_ms=committed_at_ms,
            schema_sha256=MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SCHEMA_SHA256,
            entitlement_receipt_sha256=fold_validation.semantic_receipt_sha256,
            committed_at_ms=committed_at_ms,
            request_id=f"ADAPTIVE-RL-V5-POLICY-SELECTION-{fold_validation.fold_index}",
        )
    persisted = load_massive_adaptive_rl_policy_selection_authority_v4(
        root=root,
        manifest=manifest,
        fold_index=fold_validation.fold_index,
    )
    if persisted.semantic_unsigned() != expected.semantic_unsigned():
        raise MassiveAdaptiveRLPolicySelectionAuthorityV4Error(
            "policy-selection authority does not replay"
        )
    result = replace(
        persisted,
        runtime_selection_replayed=True,
        development_policy_selection_authorized=expected.source_data_qualified,
        policy_freezing_authorized=expected.source_data_qualified,
        _runtime_manifest=manifest,
        _runtime_fold_validation=fold_validation,
        _runtime_selection_computation=expected._runtime_selection_computation,
        _runtime_selected_checkpoint=expected._runtime_selected_checkpoint,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_DATASET",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_POLICY_SELECTION_AUTHORITY_V4_SOURCE_SHA256",
    "MassiveAdaptiveRLPolicySelectionAuthorityV4",
    "MassiveAdaptiveRLPolicySelectionAuthorityV4Error",
    "build_massive_adaptive_rl_policy_selection_authority_v4",
    "load_massive_adaptive_rl_policy_selection_authority_v4",
    "policy_selection_authority_relative_path_v4",
    "run_or_resume_massive_adaptive_rl_policy_selection_authority_v4",
]
