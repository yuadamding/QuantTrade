"""Aggregate one complete V5-native validation fold.

The public runner discovers the exact checkpoint population from the released
four-fold fit, evaluates every checkpoint plus FC06, derives the registered V4
candidate witnesses, and publishes one create-only fold authority.  It exposes
no caller surface for actions, targets, environments, metrics, or candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
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
from rl_quant.evaluation.massive_adaptive_rl_validation_outcome_authority_v3 import (
    MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3,
    MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3,
    MassiveAdaptiveRLValidationOutcomeAuthorityV3,
    run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3,
    run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    MassiveAdaptiveRLValidationReleaseAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v2 import (
    MassiveAdaptiveRLPolicyCandidateV2,
    build_massive_adaptive_rl_policy_candidate_v2,
)
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    massive_adaptive_rl_deterministic_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_process_state_v1 import (
    preserve_massive_adaptive_rl_process_rng_state_v1,
)


MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_DATASET = (
    "massive-adaptive-rl-fold-validation-authority-v3"
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA,
        "encoding": "canonical-json-exact-outcome-v3-inventory",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLFoldValidationAuthorityV3Error(ValueError):
    """The released fold, outcome inventory, or candidate computation differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            f"{name} is absent or invalid"
        )
    return value


def _wall_clock_after(value: int) -> int:
    now = time.time_ns() // 1_000_000
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold-validation clock differs"
        )
    return max(now, value + 1)


def fold_validation_authority_relative_path_v3(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold-validation index differs"
        )
    return (
        f"adaptive-rl/{manifest.experiment_id}/fold-validation-v3/"
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
class MassiveAdaptiveRLFoldValidationAuthorityV3:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    release_authority_receipt_sha256: str
    release_source_receipt_sha256: str
    release_commit_receipt_sha256: str
    release_committed_at_ms: int
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    four_fold_fit_authority_receipt_sha256: str
    fold_fit_authority_receipt_sha256: str
    fold_index: int
    expected_candidate_checkpoint_authority_receipts: tuple[str, ...]
    ppo_validation_outcome_receipts: tuple[str, ...]
    ppo_validation_outcome_source_receipts: tuple[str, ...]
    ppo_validation_outcome_commit_receipts: tuple[str, ...]
    ppo_validation_outcome_committed_at_ms: tuple[int, ...]
    fc06_validation_outcome_receipt_sha256: str
    fc06_validation_outcome_source_receipt_sha256: str
    fc06_validation_outcome_commit_receipt_sha256: str
    fc06_validation_outcome_committed_at_ms: int
    fixed_control_fit_authority_receipt_sha256: str
    fixed_control_selection_authority_receipt_sha256: str
    selected_fc06_action_receipt_sha256: str
    candidate_receipts: tuple[str, ...]
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_fold_economics_replayed: bool = False
    development_policy_selection_computation_authorized: bool = False
    policy_freezing_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_release: MassiveAdaptiveRLValidationReleaseAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_ppo_outcomes: tuple[MassiveAdaptiveRLValidationOutcomeAuthorityV3, ...] = (
        field(default=(), compare=False, repr=False)
    )
    _runtime_fc06_outcome: MassiveAdaptiveRLValidationOutcomeAuthorityV3 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_candidates: tuple[MassiveAdaptiveRLPolicyCandidateV2, ...] = field(
        default=(), compare=False, repr=False
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
                "runtime_fold_economics_replayed",
                "development_policy_selection_computation_authorized",
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
            and self.runtime_fold_economics_replayed
            and self.development_policy_selection_computation_authorized
            and self.source_data_qualified
        )

    @property
    def candidates(self) -> tuple[MassiveAdaptiveRLPolicyCandidateV2, ...]:
        self.validate()
        if not self.development_stage_authorized:
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "fold candidates have not been exactly replayed"
            )
        return self._runtime_candidates

    @property
    def release_authority(self) -> MassiveAdaptiveRLValidationReleaseAuthorityV1:
        self.validate()
        if self._runtime_release is None or not self.development_stage_authorized:
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "fold validation release has not been exactly replayed"
            )
        return self._runtime_release

    def validate(self) -> None:
        runtime_present = bool(
            self._runtime_manifest is not None
            and self._runtime_release is not None
            and len(self._runtime_ppo_outcomes)
            == len(self.expected_candidate_checkpoint_authority_receipts)
            and self._runtime_fc06_outcome is not None
            and len(self._runtime_candidates)
            == len(self.expected_candidate_checkpoint_authority_receipts)
        )
        any_runtime = bool(
            self._runtime_manifest is not None
            or self._runtime_release is not None
            or self._runtime_ppo_outcomes
            or self._runtime_fc06_outcome is not None
            or self._runtime_candidates
        )
        count = self.fold_index + 1
        lengths = tuple(
            len(value)
            for value in (
                self.expected_candidate_checkpoint_authority_receipts,
                self.ppo_validation_outcome_receipts,
                self.ppo_validation_outcome_source_receipts,
                self.ppo_validation_outcome_commit_receipts,
                self.ppo_validation_outcome_committed_at_ms,
                self.candidate_receipts,
            )
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or isinstance(self.release_committed_at_ms, bool)
            or not isinstance(self.release_committed_at_ms, int)
            or self.release_committed_at_ms < 0
            or set(lengths) != {count}
            or len(set(self.expected_candidate_checkpoint_authority_receipts)) != count
            or len(set(self.ppo_validation_outcome_receipts)) != count
            or len(set(self.ppo_validation_outcome_source_receipts)) != count
            or len(set(self.ppo_validation_outcome_commit_receipts)) != count
            or len(set(self.candidate_receipts)) != count
            or self.candidate_inventory_sha256
            != semantic_sha256(self.candidate_receipts)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= self.release_committed_at_ms
                for value in (
                    *self.ppo_validation_outcome_committed_at_ms,
                    self.fc06_validation_outcome_committed_at_ms,
                )
            )
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime_present
            or self.runtime_fold_economics_replayed != runtime_present
            or self.development_policy_selection_computation_authorized
            != bool(runtime_present and self.source_data_qualified)
            or self.policy_freezing_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "fold-validation authority V3 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for inventory in (
            self.expected_candidate_checkpoint_authority_receipts,
            self.ppo_validation_outcome_receipts,
            self.ppo_validation_outcome_source_receipts,
            self.ppo_validation_outcome_commit_receipts,
            self.candidate_receipts,
        ):
            for value in inventory:
                _digest("fold-validation inventory", value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.release_authority_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(
                    *self.ppo_validation_outcome_committed_at_ms,
                    self.fc06_validation_outcome_committed_at_ms,
                )
            ):
                raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                    "fold-validation source transaction differs"
                )
        if runtime_present:
            assert self._runtime_manifest is not None
            assert self._runtime_release is not None
            assert self._runtime_fc06_outcome is not None
            self._runtime_manifest.validate()
            self._runtime_release.validate()
            for outcome in (*self._runtime_ppo_outcomes, self._runtime_fc06_outcome):
                outcome.validate()
            for candidate in self._runtime_candidates:
                candidate.validate()
            if (
                self._runtime_manifest.semantic_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_release.semantic_receipt_sha256
                != self.release_authority_receipt_sha256
                or self._runtime_release.source_receipt_sha256
                != self.release_source_receipt_sha256
                or self._runtime_release.source_transaction_receipt_sha256
                != self.release_commit_receipt_sha256
                or self._runtime_release.source_transaction_committed_at_ms
                != self.release_committed_at_ms
                or not self._runtime_release.development_stage_authorized
                or self._runtime_release.execution_implementation_registration_authority_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_release.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_release.four_fold_fit_authority_receipt_sha256
                != self.four_fold_fit_authority_receipt_sha256
                or tuple(
                    row.checkpoint_authority_receipt_sha256
                    for row in self._runtime_ppo_outcomes
                )
                != self.expected_candidate_checkpoint_authority_receipts
                or tuple(
                    row.semantic_receipt_sha256 for row in self._runtime_ppo_outcomes
                )
                != self.ppo_validation_outcome_receipts
                or tuple(
                    row.source_receipt_sha256 for row in self._runtime_ppo_outcomes
                )
                != self.ppo_validation_outcome_source_receipts
                or tuple(
                    row.source_transaction_receipt_sha256
                    for row in self._runtime_ppo_outcomes
                )
                != self.ppo_validation_outcome_commit_receipts
                or tuple(
                    row.source_transaction_committed_at_ms
                    for row in self._runtime_ppo_outcomes
                )
                != self.ppo_validation_outcome_committed_at_ms
                or self._runtime_fc06_outcome.semantic_receipt_sha256
                != self.fc06_validation_outcome_receipt_sha256
                or self._runtime_fc06_outcome.source_receipt_sha256
                != self.fc06_validation_outcome_source_receipt_sha256
                or self._runtime_fc06_outcome.source_transaction_receipt_sha256
                != self.fc06_validation_outcome_commit_receipt_sha256
                or self._runtime_fc06_outcome.source_transaction_committed_at_ms
                != self.fc06_validation_outcome_committed_at_ms
                or self._runtime_fc06_outcome.fixed_control_fit_authority_receipt_sha256
                != self.fixed_control_fit_authority_receipt_sha256
                or self._runtime_fc06_outcome.fixed_control_selection_authority_receipt_sha256
                != self.fixed_control_selection_authority_receipt_sha256
                or self._runtime_fc06_outcome.selected_fc06_action_receipt_sha256
                != self.selected_fc06_action_receipt_sha256
                or tuple(
                    row.semantic_receipt_sha256 for row in self._runtime_candidates
                )
                != self.candidate_receipts
                or self.source_data_qualified
                != bool(
                    self._runtime_release.source_data_qualified
                    and all(
                        row.source_data_qualified for row in self._runtime_ppo_outcomes
                    )
                    and self._runtime_fc06_outcome.source_data_qualified
                    and all(
                        row.source_data_qualified for row in self._runtime_candidates
                    )
                )
                or any(
                    not row.development_stage_authorized
                    for row in self._runtime_ppo_outcomes
                )
                or not self._runtime_fc06_outcome.development_stage_authorized
            ):
                raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                    "fold-validation runtime evidence differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_fold_validation_authority_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    ppo_outcomes: Sequence[MassiveAdaptiveRLValidationOutcomeAuthorityV3],
    fc06_outcome: MassiveAdaptiveRLValidationOutcomeAuthorityV3,
) -> MassiveAdaptiveRLFoldValidationAuthorityV3:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(release) is not MassiveAdaptiveRLValidationReleaseAuthorityV1
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold validation requires exact V5 roots"
        )
    if type(fc06_outcome) is not MassiveAdaptiveRLValidationOutcomeAuthorityV3:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold validation requires exact FC06 outcome V3"
        )
    outcomes = tuple(ppo_outcomes)
    if any(
        type(row) is not MassiveAdaptiveRLValidationOutcomeAuthorityV3
        for row in outcomes
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold validation requires exact PPO outcome V3 inventory"
        )
    manifest.validate()
    release.validate()
    for outcome in (*outcomes, fc06_outcome):
        outcome.validate()
    expected = release.expected_candidate_checkpoint_authority_receipt_inventories[
        release.released_validation_fold_indices.index(fold_index)
    ]
    by_checkpoint = {
        cast(str, row.checkpoint_authority_receipt_sha256): row for row in outcomes
    }
    if (
        set(by_checkpoint) != set(expected)
        or len(by_checkpoint) != len(outcomes)
        or fc06_outcome.outcome_kind != MASSIVE_ADAPTIVE_RL_FC06_VALIDATION_OUTCOME_V3
        or fc06_outcome.fold_index != fold_index
        or fc06_outcome.release_authority_receipt_sha256
        != release.semantic_receipt_sha256
        or any(
            row.outcome_kind != MASSIVE_ADAPTIVE_RL_PPO_VALIDATION_OUTCOME_V3
            or row.fold_index != fold_index
            or row.release_authority_receipt_sha256 != release.semantic_receipt_sha256
            or not row.development_stage_authorized
            for row in outcomes
        )
        or not fc06_outcome.development_stage_authorized
        or fc06_outcome.fixed_control_selection_authority_receipt_sha256
        != release.four_fold_fit_authority.fold_fit(
            fold_index
        ).training_workflow.runtime_workflow.fixed_control_selection_authority.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold validation outcome coverage differs"
        )
    ordered = tuple(by_checkpoint[value] for value in expected)
    fold_fit = release.four_fold_fit_authority.fold_fit(fold_index)
    workflow = fold_fit.training_workflow.runtime_workflow
    fixed_selection = workflow.fixed_control_selection_authority
    fixed_fit = workflow.fixed_control_fit_authority
    fixed_trace = fc06_outcome.runtime_fixed_control_evaluation.policy_trace
    candidate_rows: list[MassiveAdaptiveRLPolicyCandidateV2] = []
    for outcome in ordered:
        checkpoint = outcome.runtime_checkpoint_authority.runtime_checkpoint
        if checkpoint is None:
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "released checkpoint runtime state is absent"
            )
        ladder = outcome.runtime_cost_ladder
        candidate_rows.append(
            build_massive_adaptive_rl_policy_candidate_v2(
                manifest=manifest.base_manifest,
                checkpoint_authority_receipt_sha256=cast(
                    str, outcome.checkpoint_authority_receipt_sha256
                ),
                checkpoint=checkpoint,
                primary_trace=ladder.primary.policy_trace,
                low_cost_trace=ladder.low_cost_trace,
                high_cost_trace=ladder.high_cost_trace,
                fixed_control_selection_authority=fixed_selection,
                fixed_control_validation_trace=fixed_trace,
            )
        )
    candidates = tuple(candidate_rows)
    source_receipts = tuple(
        _required_digest("PPO outcome source", row.source_receipt_sha256)
        for row in ordered
    )
    commit_receipts = tuple(
        _required_digest("PPO outcome commit", row.source_transaction_receipt_sha256)
        for row in ordered
    )
    committed_times = tuple(
        _required_time("PPO outcome time", row.source_transaction_committed_at_ms)
        for row in ordered
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "release_authority_receipt_sha256": release.semantic_receipt_sha256,
        "release_source_receipt_sha256": _required_digest(
            "release source", release.source_receipt_sha256
        ),
        "release_commit_receipt_sha256": _required_digest(
            "release commit", release.source_transaction_receipt_sha256
        ),
        "release_committed_at_ms": _required_time(
            "release time", release.source_transaction_committed_at_ms
        ),
        "execution_implementation_registration_receipt_sha256": release.execution_implementation_registration_authority_receipt_sha256,
        "scientific_execution_fingerprint_sha256": release.scientific_execution_fingerprint_sha256,
        "four_fold_fit_authority_receipt_sha256": release.four_fold_fit_authority_receipt_sha256,
        "fold_fit_authority_receipt_sha256": fold_fit.semantic_receipt_sha256,
        "fold_index": fold_index,
        "expected_candidate_checkpoint_authority_receipts": expected,
        "ppo_validation_outcome_receipts": tuple(
            row.semantic_receipt_sha256 for row in ordered
        ),
        "ppo_validation_outcome_source_receipts": source_receipts,
        "ppo_validation_outcome_commit_receipts": commit_receipts,
        "ppo_validation_outcome_committed_at_ms": committed_times,
        "fc06_validation_outcome_receipt_sha256": fc06_outcome.semantic_receipt_sha256,
        "fc06_validation_outcome_source_receipt_sha256": _required_digest(
            "FC06 outcome source", fc06_outcome.source_receipt_sha256
        ),
        "fc06_validation_outcome_commit_receipt_sha256": _required_digest(
            "FC06 outcome commit", fc06_outcome.source_transaction_receipt_sha256
        ),
        "fc06_validation_outcome_committed_at_ms": _required_time(
            "FC06 outcome time", fc06_outcome.source_transaction_committed_at_ms
        ),
        "fixed_control_fit_authority_receipt_sha256": fixed_fit.semantic_receipt_sha256,
        "fixed_control_selection_authority_receipt_sha256": fixed_selection.semantic_receipt_sha256,
        "selected_fc06_action_receipt_sha256": fc06_outcome.selected_fc06_action_receipt_sha256,
        "candidate_receipts": tuple(row.semantic_receipt_sha256 for row in candidates),
        "candidate_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in candidates)
        ),
        "source_data_qualified": bool(
            release.source_data_qualified
            and all(row.source_data_qualified for row in (*ordered, fc06_outcome))
            and all(row.source_data_qualified for row in candidates)
        ),
        "policy_freezing_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SCHEMA,
    }
    provisional = MassiveAdaptiveRLFoldValidationAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_fold_economics_replayed=True,
        development_policy_selection_computation_authorized=bool(
            body["source_data_qualified"]
        ),
        _runtime_manifest=manifest,
        _runtime_release=release,
        _runtime_ppo_outcomes=ordered,
        _runtime_fc06_outcome=fc06_outcome,
        _runtime_candidates=candidates,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFoldValidationAuthorityV3:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold-validation source is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "expected_candidate_checkpoint_authority_receipts",
        "ppo_validation_outcome_receipts",
        "ppo_validation_outcome_source_receipts",
        "ppo_validation_outcome_commit_receipts",
        "ppo_validation_outcome_committed_at_ms",
        "candidate_receipts",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    result = MassiveAdaptiveRLFoldValidationAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_fold_validation_authority_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    fold_index: int,
    verified_at_ms: int | None = None,
) -> MassiveAdaptiveRLFoldValidationAuthorityV3:
    return _parse(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=fold_validation_authority_relative_path_v3(
                manifest=manifest, fold_index=fold_index
            ),
            verified_at_ms=(
                time.time_ns() // 1_000_000
                if verified_at_ms is None
                else verified_at_ms
            ),
        ),
    )


def _persist(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    authority: MassiveAdaptiveRLFoldValidationAuthorityV3,
) -> None:
    authority.validate()
    relative = fold_validation_authority_relative_path_v3(
        manifest=manifest, fold_index=authority.fold_index
    )
    committed_at_ms = _wall_clock_after(
        max(
            *authority.ppo_validation_outcome_committed_at_ms,
            authority.fc06_validation_outcome_committed_at_ms,
        )
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=authority.release_authority_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-V5-FOLD-VALIDATION-{authority.fold_index}",
    )


def run_or_resume_massive_adaptive_rl_fold_validation_v3(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    release: MassiveAdaptiveRLValidationReleaseAuthorityV1,
    fold_index: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFoldValidationAuthorityV3:
    """Execute or exactly replay every released outcome for one fold."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold-validation materialization mode differs"
        )
    release.validate()
    if fold_index not in release.released_validation_fold_indices:
        raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
            "fold is not causally available in this validation release"
        )
    fold_fit = release.four_fold_fit_authority.fold_fit(fold_index)
    checkpoints = (
        fold_fit.training_workflow.runtime_workflow.policy_checkpoint_authorities
    )
    lock = (
        massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=root, experiment_id=manifest.experiment_id
        )
        if allow_materialize
        else nullcontext()
    )
    with (
        lock,
        preserve_massive_adaptive_rl_process_rng_state_v1(include_cuda=False),
        massive_adaptive_rl_deterministic_execution_v1(device="cpu"),
    ):
        ppo_outcomes = tuple(
            run_or_resume_massive_adaptive_rl_ppo_validation_outcome_v3(
                root=root,
                manifest=manifest,
                release=release,
                fold_index=fold_index,
                checkpoint_authority=checkpoint,
                allow_materialize=allow_materialize,
            )
            for checkpoint in checkpoints
        )
        fc06_outcome = run_or_resume_massive_adaptive_rl_fc06_validation_outcome_v3(
            root=root,
            manifest=manifest,
            release=release,
            fold_index=fold_index,
            allow_materialize=allow_materialize,
        )
        expected = build_massive_adaptive_rl_fold_validation_authority_v3(
            manifest=manifest,
            release=release,
            fold_index=fold_index,
            ppo_outcomes=ppo_outcomes,
            fc06_outcome=fc06_outcome,
        )
        relative = fold_validation_authority_relative_path_v3(
            manifest=manifest, fold_index=fold_index
        )
        complete, partial = _transaction_state(root=root, relative=relative)
        if partial:
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "fold-validation transaction is incomplete"
            )
        if not complete:
            if not allow_materialize:
                raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                    "fold-validation authority is absent in read-only mode"
                )
            _persist(root=root, manifest=manifest, authority=expected)
        persisted = load_massive_adaptive_rl_fold_validation_authority_v3(
            root=root, manifest=manifest, fold_index=fold_index
        )
        if persisted.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLFoldValidationAuthorityV3Error(
                "fold-validation authority does not replay"
            )
        result = replace(
            persisted,
            runtime_fold_economics_replayed=True,
            development_policy_selection_computation_authorized=expected.source_data_qualified,
            _runtime_manifest=manifest,
            _runtime_release=release,
            _runtime_ppo_outcomes=ppo_outcomes,
            _runtime_fc06_outcome=fc06_outcome,
            _runtime_candidates=expected._runtime_candidates,
        )
        result.validate()
        return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_DATASET",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_AUTHORITY_V3_SOURCE_SHA256",
    "MassiveAdaptiveRLFoldValidationAuthorityV3",
    "MassiveAdaptiveRLFoldValidationAuthorityV3Error",
    "build_massive_adaptive_rl_fold_validation_authority_v3",
    "fold_validation_authority_relative_path_v3",
    "load_massive_adaptive_rl_fold_validation_authority_v3",
    "run_or_resume_massive_adaptive_rl_fold_validation_v3",
]
