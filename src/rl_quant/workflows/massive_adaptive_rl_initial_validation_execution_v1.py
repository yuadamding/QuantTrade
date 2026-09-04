"""Execute, select, and freeze the initially released V5 folds 0 and 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_rl_fold_validation_authority_v3 import (
    MassiveAdaptiveRLFoldValidationAuthorityV3,
    run_or_resume_massive_adaptive_rl_fold_validation_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MassiveAdaptiveFrozenRLPolicyV2,
    run_or_resume_massive_adaptive_frozen_rl_policy_v2,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
    run_or_resume_massive_adaptive_rl_frozen_fc06_v2,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v4 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV4,
    run_or_resume_massive_adaptive_rl_policy_selection_authority_v4,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_initial_validation_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "exact-initial-validation-release-v1",
        "folds": (0, 1),
        "outcomes": "all-released-checkpoints-plus-fc06",
        "selection": "exact-policy-selection-v4",
        "freeze": "ppo-v2-and-fc06-v2-before-outer-zero",
        "caller_economic_inputs": False,
        "resume": "create-only-or-exact-replay",
    }
)
MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1 = (
    "initial-policy-pair-validation-qualified"
)
MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1 = (
    "initial-policy-pair-diagnostic-only"
)


class MassiveAdaptiveRLInitialValidationExecutionV1Error(ValueError):
    """The initial release, selection, or paired frozen artifacts differ."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLInitialValidationExecutionV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    validation_release_authority_receipt_sha256: str
    execution_implementation_registration_receipt_sha256: str
    fold_indices: tuple[int, ...]
    fold_validation_authorities: tuple[MassiveAdaptiveRLFoldValidationAuthorityV3, ...]
    policy_selection_authorities: tuple[
        MassiveAdaptiveRLPolicySelectionAuthorityV4, ...
    ]
    frozen_ppo_policies: tuple[MassiveAdaptiveFrozenRLPolicyV2, ...]
    frozen_fc06_controls: tuple[MassiveAdaptiveRLFrozenFC06V2, ...]
    policy_schedule_disposition: str
    all_initial_policies_validation_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    initial_policy_freezing_complete: bool
    outer_zero_preparation_authorized: bool
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SPEC_SHA256
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "manifest_v5_receipt_sha256": self.manifest_v5_receipt_sha256,
            "validation_release_authority_receipt_sha256": (
                self.validation_release_authority_receipt_sha256
            ),
            "execution_implementation_registration_receipt_sha256": (
                self.execution_implementation_registration_receipt_sha256
            ),
            "fold_indices": self.fold_indices,
            "fold_validation_authority_receipts": tuple(
                row.semantic_receipt_sha256 for row in self.fold_validation_authorities
            ),
            "policy_selection_authority_receipts": tuple(
                row.semantic_receipt_sha256 for row in self.policy_selection_authorities
            ),
            "frozen_ppo_policy_receipts": tuple(
                row.semantic_receipt_sha256 for row in self.frozen_ppo_policies
            ),
            "frozen_fc06_control_receipts": tuple(
                row.semantic_receipt_sha256 for row in self.frozen_fc06_controls
            ),
            "policy_schedule_disposition": self.policy_schedule_disposition,
            "all_initial_policies_validation_eligible": (
                self.all_initial_policies_validation_eligible
            ),
            "source_data_qualified": self.source_data_qualified,
            "initial_policy_freezing_complete": self.initial_policy_freezing_complete,
            "outer_zero_preparation_authorized": (
                self.outer_zero_preparation_authorized
            ),
            "outer_access_authorized": False,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "live_trading_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        collections = (
            self.fold_validation_authorities,
            self.policy_selection_authorities,
            self.frozen_ppo_policies,
            self.frozen_fc06_controls,
        )
        expected_qualified = all(
            row.selected_candidate_validation_eligible
            for row in self.policy_selection_authorities
        )
        expected_source = all(
            row.source_data_qualified for rows in collections for row in rows
        )
        if (
            not self.experiment_id
            or self.fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
            or any(len(rows) != 2 for rows in collections)
            or any(
                tuple(row.fold_index for row in rows) != self.fold_indices
                for rows in collections
            )
            or any(
                not row.development_stage_authorized
                for rows in collections
                for row in rows
            )
            or any(
                row.experiment_id != self.experiment_id
                or row.manifest_v5_receipt_sha256 != self.manifest_v5_receipt_sha256
                or row.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                for rows in collections
                for row in rows
            )
            or any(
                row.release_authority_receipt_sha256
                != self.validation_release_authority_receipt_sha256
                for row in self.fold_validation_authorities
            )
            or any(
                row.validation_release_authority_receipt_sha256
                != self.validation_release_authority_receipt_sha256
                for rows in (
                    self.policy_selection_authorities,
                    self.frozen_ppo_policies,
                    self.frozen_fc06_controls,
                )
                for row in rows
            )
            or any(
                len({row.semantic_receipt_sha256 for row in rows}) != 2
                for rows in collections
            )
            or any(
                selection.fold_validation_authority_receipt_sha256
                != validation.semantic_receipt_sha256
                or ppo.policy_selection_authority_receipt_sha256
                != selection.semantic_receipt_sha256
                or fc06.policy_selection_authority_receipt_sha256
                != selection.semantic_receipt_sha256
                for validation, selection, ppo, fc06 in zip(*collections, strict=True)
            )
            or self.all_initial_policies_validation_eligible != expected_qualified
            or self.policy_schedule_disposition
            != (
                MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1
                if expected_qualified
                else MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1
            )
            or self.source_data_qualified != expected_source
            or self.initial_policy_freezing_complete != expected_source
            or self.outer_zero_preparation_authorized != expected_source
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLInitialValidationExecutionV1Error(
                "initial V5 validation execution differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def run_or_resume_massive_adaptive_rl_initial_validation_execution_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: (
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
    ),
    validation_release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLInitialValidationExecutionV1:
    """Run or cold-replay V0/V1 through paired PPO and FC06 freezes."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLInitialValidationExecutionV1Error(
            "initial V5 validation materialization mode differs"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        validation_release,
    ):
        authority.validate()
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(validation_release) is not MassiveAdaptiveRLValidationReleaseAuthorityV1
        or not execution_registration.development_execution_registered
        or not validation_release.development_stage_authorized
        or validation_release.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or validation_release.manifest_v5_registration_authority_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
        or validation_release.execution_implementation_registration_authority_receipt_sha256
        != execution_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLInitialValidationExecutionV1Error(
            "initial V5 validation execution roots differ"
        )
    capability = issue_massive_adaptive_rl_manifest_v5_initial_validation_execution_capability_v1(
        root=root, authority=manifest_registration
    )
    with (
        massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=root, experiment_id=manifest.experiment_id
        ),
        massive_adaptive_rl_manifest_v5_writer_scope_v1(
            root=root, capability=capability
        ),
    ):
        validations = tuple(
            run_or_resume_massive_adaptive_rl_fold_validation_v3(
                root=root,
                manifest=manifest,
                release=validation_release,
                fold_index=fold_index,
                allow_materialize=allow_materialize,
            )
            for fold_index in MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
        )
        selections = tuple(
            run_or_resume_massive_adaptive_rl_policy_selection_authority_v4(
                root=root,
                manifest=manifest,
                fold_validation=validation,
                allow_materialize=allow_materialize,
            )
            for validation in validations
        )
        frozen_ppo = tuple(
            run_or_resume_massive_adaptive_frozen_rl_policy_v2(
                root=root,
                manifest=manifest,
                selection=selection,
                allow_materialize=allow_materialize,
            )
            for selection in selections
        )
        frozen_fc06 = tuple(
            run_or_resume_massive_adaptive_rl_frozen_fc06_v2(
                root=root,
                manifest=manifest,
                selection=selection,
                allow_materialize=allow_materialize,
            )
            for selection in selections
        )
    eligible = all(row.selected_candidate_validation_eligible for row in selections)
    source_qualified = all(
        row.source_data_qualified
        for rows in (validations, selections, frozen_ppo, frozen_fc06)
        for row in rows
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "validation_release_authority_receipt_sha256": (
            validation_release.semantic_receipt_sha256
        ),
        "execution_implementation_registration_receipt_sha256": (
            execution_registration.semantic_receipt_sha256
        ),
        "fold_indices": MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
        "fold_validation_authorities": validations,
        "policy_selection_authorities": selections,
        "frozen_ppo_policies": frozen_ppo,
        "frozen_fc06_controls": frozen_fc06,
        "policy_schedule_disposition": (
            MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1
            if eligible
            else MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1
        ),
        "all_initial_policies_validation_eligible": eligible,
        "source_data_qualified": source_qualified,
        "initial_policy_freezing_complete": source_qualified,
        "outer_zero_preparation_authorized": source_qualified,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SPEC_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLInitialValidationExecutionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLInitialValidationExecutionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_EXECUTION_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_DIAGNOSTIC_V1",
    "MASSIVE_ADAPTIVE_RL_INITIAL_POLICY_PAIR_QUALIFIED_V1",
    "MassiveAdaptiveRLInitialValidationExecutionV1",
    "MassiveAdaptiveRLInitialValidationExecutionV1Error",
    "run_or_resume_massive_adaptive_rl_initial_validation_execution_v1",
]
