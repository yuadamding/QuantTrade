"""Manifest-V5 outer-access commitment bound to frozen policy pairs."""

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
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_frozen_rl_policy_v2 import (
    MassiveAdaptiveFrozenRLPolicyV2,
)
from rl_quant.training.massive_adaptive_rl_frozen_fc06_v2 import (
    MassiveAdaptiveRLFrozenFC06V2,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA,
    MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_DATASET = (
    "massive-adaptive-rl-outer-access-commitment-v2"
)
MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA,
        "encoding": "canonical-json-exact-v5-outer-access",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveOuterAccessCommitmentV2Error(ValueError):
    """Outer inputs, schedule, or frozen policy chronology differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: str | None) -> str:
    if value is None:
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(f"{name} is absent")
    return _digest(name, value)


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            f"{name} is absent or invalid"
        )
    return value


def _environment_identity_without_cost(
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> tuple[object, ...]:
    dates = tuple(row.decision_session_date for row in environment.inference_plan.rows)
    return (
        environment.forecast_archive.semantic_receipt_sha256,
        environment.calibration.semantic_receipt_sha256,
        environment.inference_plan.semantic_receipt_sha256,
        environment.fill_source.semantic_receipt_sha256,
        environment.daily_input_authority.semantic_receipt_sha256,
        environment.identity_authority.receipt_sha256,
        None
        if environment.economic_event_archive is None
        else environment.economic_event_archive.receipt_sha256,
        environment.compiler_config.receipt_sha256,
        environment.initial_capital,
        environment.maximum_fill_participation,
        tuple(environment.roots[date].semantic_receipt_sha256 for date in dates),
        tuple(environment.contexts[date].semantic_receipt_sha256 for date in dates),
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEnvironmentBundleV2:
    """Exact economic contexts opened only after one persisted commitment."""

    fold_index: int
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1 = field(repr=False)
    primary_environment: MassiveAdaptiveProfitabilityEnvV1 = field(repr=False)
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1 = field(repr=False)
    fixed_control_environment: MassiveAdaptiveProfitabilityEnvV1 = field(repr=False)
    decision_session_dates: tuple[str, ...]
    runtime_sources_v2_receipt_sha256: str
    split_plan_receipt_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_archive_receipt_sha256: str
    economic_compatibility_receipt_sha256: str
    environment_source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "decision_session_dates": self.decision_session_dates,
            "runtime_sources_v2_receipt_sha256": (
                self.runtime_sources_v2_receipt_sha256
            ),
            "split_plan_receipt_sha256": self.split_plan_receipt_sha256,
            "daily_input_authority_receipt_sha256": (
                self.daily_input_authority_receipt_sha256
            ),
            "fill_source_receipt_sha256": self.fill_source_receipt_sha256,
            "identity_authority_receipt_sha256": (
                self.identity_authority_receipt_sha256
            ),
            "economic_event_archive_receipt_sha256": (
                self.economic_event_archive_receipt_sha256
            ),
            "economic_compatibility_receipt_sha256": self.economic_compatibility_receipt_sha256,
            "environment_source_inventory_sha256": self.environment_source_inventory_sha256,
            "cost_basis_points": (10.0, 20.0, 40.0),
            "fixed_control_cost_basis_points": 20.0,
            "source_data_qualified": self.source_data_qualified,
        }

    @property
    def environments(self) -> tuple[MassiveAdaptiveProfitabilityEnvV1, ...]:
        return (
            self.low_cost_environment,
            self.primary_environment,
            self.high_cost_environment,
            self.fixed_control_environment,
        )

    def validate(self) -> None:
        identities = tuple(
            _environment_identity_without_cost(row) for row in self.environments
        )
        dates = tuple(
            tuple(item.decision_session_date for item in row.inference_plan.rows)
            for row in self.environments
        )
        if (
            isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.decision_session_dates) != 126
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or any(value != self.decision_session_dates for value in dates)
            or any(
                getattr(row.inference_plan, "fold_index", None) != self.fold_index
                for row in self.environments
            )
            or tuple(row.transaction_cost_basis_points for row in self.environments)
            != (10.0, 20.0, 40.0, 20.0)
            or len(set(identities)) != 1
            or self.economic_compatibility_receipt_sha256
            != self.primary_environment.economic_compatibility_receipt_sha256
            or self.environment_source_inventory_sha256
            != semantic_sha256(
                tuple(row.source_inventory_sha256 for row in self.environments)
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "outer environment bundle differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)


def _build_massive_adaptive_rl_outer_environment_bundle_v2(
    *,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    fold_index: int,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    fixed_control_environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLOuterEnvironmentBundleV2:
    outer_access.validate()
    if (
        not outer_access.outer_input_access_authorized
        or outer_access.source_transaction_receipt_sha256 is None
        or outer_access.fold_index != fold_index
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer environment requires a persisted replayed access commitment"
        )
    initial_inputs = outer_access.frozen_policy.policy_selection_authority.fold_validation_authority.release_authority.initial_validation_inputs
    economics = initial_inputs.manifest_v4.base_manifest.base_manifest
    runtime_sources_v2 = initial_inputs.runtime_sources_v2
    runtime_sources = runtime_sources_v2.base_runtime_sources_v1
    frozen_policy = outer_access.frozen_policy
    selection = frozen_policy.policy_selection_authority
    supervised_lineage = runtime_sources.fold(fold_index).supervised_lineage(fold_index)
    supervised_lineage.validate()
    access_time = _required_time(
        "outer access time", outer_access.source_transaction_committed_at_ms
    )
    environments = (
        low_cost_environment,
        primary_environment,
        high_cost_environment,
        fixed_control_environment,
    )
    if any(type(row) is not MassiveAdaptiveProfitabilityEnvV1 for row in environments):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer environment requires exact profitability environments"
        )
    for row in environments:
        row.forecast_archive.validate()
        row.calibration.validate()
        row.inference_plan.validate()
        row.fill_source.validate()
        row.daily_input_authority.validate()
        row.identity_authority.validate()
        if row.economic_event_archive is not None:
            row.economic_event_archive.validate()
    dates = tuple(
        row.decision_session_date for row in primary_environment.inference_plan.rows
    )
    body = {
        "fold_index": fold_index,
        "decision_session_dates": dates,
        "runtime_sources_v2_receipt_sha256": (
            runtime_sources_v2.semantic_receipt_sha256
        ),
        "split_plan_receipt_sha256": runtime_sources.split_plan.semantic_receipt_sha256,
        "daily_input_authority_receipt_sha256": (
            runtime_sources.daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_receipt_sha256": runtime_sources.fill_source.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": (
            runtime_sources.identity_authority.receipt_sha256
        ),
        "economic_event_archive_receipt_sha256": (
            runtime_sources.economic_event_archive.receipt_sha256
        ),
        "economic_compatibility_receipt_sha256": primary_environment.economic_compatibility_receipt_sha256,
        "environment_source_inventory_sha256": semantic_sha256(
            tuple(row.source_inventory_sha256 for row in environments)
        ),
        "source_data_qualified": bool(
            runtime_sources_v2.source_data_qualified
            and runtime_sources.source_data_qualified
            and economics.cost_ladder_basis_points == (10.0, 20.0, 40.0)
            and economics.primary_cost_basis_points == 20.0
            and all(
                row.daily_input_authority.semantic_receipt_sha256
                == runtime_sources.daily_input_authority.semantic_receipt_sha256
                and row.fill_source.semantic_receipt_sha256
                == runtime_sources.fill_source.semantic_receipt_sha256
                and row.identity_authority.receipt_sha256
                == runtime_sources.identity_authority.receipt_sha256
                and row.economic_event_archive is not None
                and row.economic_event_archive.receipt_sha256
                == runtime_sources.economic_event_archive.receipt_sha256
                and row.initial_capital == economics.primary_capital
                and row.maximum_fill_participation
                == economics.maximum_fill_participation
                for row in environments
            )
            and dates == outer_access.outer_decision_session_dates
            and all(
                bool(getattr(row.forecast_archive, "outer_forecast_authorized", False))
                and getattr(row.forecast_archive, "fold_index", None) == fold_index
                and getattr(
                    row.forecast_archive,
                    "selected_checkpoint_receipt_sha256",
                    None,
                )
                == supervised_lineage.selected_checkpoint.semantic_receipt_sha256
                and getattr(
                    row.forecast_archive,
                    "checkpoint_source_receipt_sha256",
                    None,
                )
                == supervised_lineage.selected_checkpoint.loaded_source.receipt.receipt_sha256
                and getattr(
                    row.forecast_archive,
                    "model_state_receipt_sha256",
                    None,
                )
                == supervised_lineage.selected_checkpoint.model_state_receipt_sha256
                and getattr(
                    row.forecast_archive,
                    "training_window_plan_receipt_sha256",
                    None,
                )
                == supervised_lineage.training_window.semantic_receipt_sha256
                and getattr(row.forecast_archive, "split_plan_receipt_sha256", None)
                == runtime_sources.split_plan.semantic_receipt_sha256
                and getattr(
                    row.forecast_archive,
                    "outer_inference_plan_receipt_sha256",
                    None,
                )
                == row.inference_plan.semantic_receipt_sha256
                and tuple(getattr(row.forecast_archive, "origin_session_dates", ()))
                == dates
                and row.calibration.semantic_receipt_sha256
                == supervised_lineage.calibration.semantic_receipt_sha256
                and getattr(
                    row.inference_plan,
                    "selected_checkpoint_receipt_sha256",
                    None,
                )
                == supervised_lineage.selected_checkpoint.semantic_receipt_sha256
                and getattr(row.inference_plan, "split_plan_receipt_sha256", None)
                == runtime_sources.split_plan.semantic_receipt_sha256
                and getattr(
                    getattr(row.forecast_archive, "loaded_source", None),
                    "commit",
                    None,
                )
                is not None
                and getattr(
                    getattr(
                        getattr(row.forecast_archive, "loaded_source", None),
                        "commit",
                        None,
                    ),
                    "committed_at_ms",
                    -1,
                )
                > access_time
                for row in environments
            )
            and selection.semantic_receipt_sha256
            == frozen_policy.policy_selection_authority_receipt_sha256
        ),
    }
    provisional = MassiveAdaptiveRLOuterEnvironmentBundleV2(
        low_cost_environment=low_cost_environment,
        primary_environment=primary_environment,
        high_cost_environment=high_cost_environment,
        fixed_control_environment=fixed_control_environment,
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    if not result.source_data_qualified:
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer environment is not commitment-bound and source-qualified"
        )
    return result


def outer_access_commitment_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if isinstance(fold_index, bool) or fold_index not in range(4):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error("outer-access fold differs")
    return f"adaptive-rl/{manifest.experiment_id}/outer-access-commitment-v2/fold-{fold_index}.json"


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterAccessCommitmentV2:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    manifest_v5_registration_receipt_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    fold_index: int
    policy_schedule_receipt_sha256: str
    policy_schedule_source_receipt_sha256: str
    policy_schedule_commit_receipt_sha256: str
    policy_schedule_committed_at_ms: int
    frozen_ppo_policy_receipt_sha256: str
    frozen_ppo_source_receipt_sha256: str
    frozen_ppo_commit_receipt_sha256: str
    frozen_ppo_committed_at_ms: int
    frozen_fc06_control_receipt_sha256: str
    frozen_fc06_source_receipt_sha256: str
    frozen_fc06_commit_receipt_sha256: str
    frozen_fc06_committed_at_ms: int
    outer_fold_receipt_sha256: str
    outer_decision_session_dates: tuple[str, ...]
    outer_decision_inventory_sha256: str
    policy_validation_eligible: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_commitment_replayed: bool = False
    outer_input_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA
    _runtime_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_frozen_policy: MassiveAdaptiveFrozenRLPolicyV2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_frozen_control: MassiveAdaptiveRLFrozenFC06V2 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_environment_bundle: MassiveAdaptiveRLOuterEnvironmentBundleV2 | None = (
        field(default=None, compare=False, repr=False)
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
                "runtime_commitment_replayed",
                "outer_input_access_authorized",
                "positive_profitability_authorization_eligible",
            }
        }

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
    def runtime_environment_bundle(self) -> MassiveAdaptiveRLOuterEnvironmentBundleV2:
        self.validate()
        if (
            self._runtime_environment_bundle is None
            or not self.outer_input_access_authorized
            or not self._runtime_environment_bundle.source_data_qualified
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "outer environment bundle has not been commitment-replayed"
            )
        return self._runtime_environment_bundle

    @property
    def frozen_policy(self) -> MassiveAdaptiveFrozenRLPolicyV2:
        self.validate()
        if (
            self._runtime_frozen_policy is None
            or not self.outer_input_access_authorized
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "frozen PPO policy is unavailable"
            )
        return self._runtime_frozen_policy

    @property
    def frozen_control(self) -> MassiveAdaptiveRLFrozenFC06V2:
        self.validate()
        if (
            self._runtime_frozen_control is None
            or not self.outer_input_access_authorized
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "frozen FC06 control is unavailable"
            )
        return self._runtime_frozen_control

    def validate(self) -> None:
        runtime_roots = (
            self._runtime_schedule,
            self._runtime_frozen_policy,
            self._runtime_frozen_control,
        )
        runtime = all(value is not None for value in runtime_roots)
        any_runtime = any(value is not None for value in runtime_roots)
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA
            or not self.experiment_id
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or len(self.outer_decision_session_dates) != 126
            or self.outer_decision_session_dates
            != tuple(sorted(set(self.outer_decision_session_dates)))
            or self.outer_decision_inventory_sha256
            != semantic_sha256(self.outer_decision_session_dates)
            or not isinstance(self.policy_validation_eligible, bool)
            or not isinstance(self.source_data_qualified, bool)
            or any_runtime != runtime
            or (self._runtime_environment_bundle is not None and not runtime)
            or self.runtime_commitment_replayed != runtime
            or self.outer_input_access_authorized
            != bool(runtime and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.positive_profitability_authorization_eligible
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "outer-access commitment V2 differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(
                    self.policy_schedule_committed_at_ms,
                    self.frozen_ppo_committed_at_ms,
                    self.frozen_fc06_committed_at_ms,
                )
            ):
                raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                    "outer-access source transaction differs"
                )
        if runtime:
            assert self._runtime_schedule is not None
            assert self._runtime_frozen_policy is not None
            assert self._runtime_frozen_control is not None
            for value in runtime_roots:
                getattr(value, "validate")()
            if (
                not self._runtime_schedule.authorizes_outer_fold(self.fold_index)
                or self._runtime_schedule.experiment_id != self.experiment_id
                or self._runtime_schedule.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_schedule.scientific_protocol_projection_sha256
                != self.scientific_protocol_projection_sha256
                or self._runtime_schedule.manifest_v5_registration_receipt_sha256
                != self.manifest_v5_registration_receipt_sha256
                or self._runtime_schedule.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_schedule.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_schedule.semantic_receipt_sha256
                != self.policy_schedule_receipt_sha256
                or self._runtime_schedule.source_receipt_sha256
                != self.policy_schedule_source_receipt_sha256
                or self._runtime_schedule.source_transaction_receipt_sha256
                != self.policy_schedule_commit_receipt_sha256
                or self._runtime_schedule.source_transaction_committed_at_ms
                != self.policy_schedule_committed_at_ms
                or self._runtime_frozen_policy.fold_index != self.fold_index
                or self._runtime_frozen_control.fold_index != self.fold_index
                or self._runtime_frozen_policy.experiment_id != self.experiment_id
                or self._runtime_frozen_control.experiment_id != self.experiment_id
                or self._runtime_frozen_policy.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_frozen_control.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_frozen_policy.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_frozen_control.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_frozen_policy.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_frozen_control.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_frozen_policy.policy_selection_authority_receipt_sha256
                != self._runtime_frozen_control.policy_selection_authority_receipt_sha256
                or self._runtime_frozen_policy.semantic_receipt_sha256
                != self.frozen_ppo_policy_receipt_sha256
                or self._runtime_frozen_policy.source_receipt_sha256
                != self.frozen_ppo_source_receipt_sha256
                or self._runtime_frozen_policy.source_transaction_receipt_sha256
                != self.frozen_ppo_commit_receipt_sha256
                or self._runtime_frozen_policy.source_transaction_committed_at_ms
                != self.frozen_ppo_committed_at_ms
                or self._runtime_frozen_control.semantic_receipt_sha256
                != self.frozen_fc06_control_receipt_sha256
                or self._runtime_frozen_control.source_receipt_sha256
                != self.frozen_fc06_source_receipt_sha256
                or self._runtime_frozen_control.source_transaction_receipt_sha256
                != self.frozen_fc06_commit_receipt_sha256
                or self._runtime_frozen_control.source_transaction_committed_at_ms
                != self.frozen_fc06_committed_at_ms
            ):
                raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                    "outer-access runtime lineage differs"
                )
            if self._runtime_environment_bundle is not None:
                self._runtime_environment_bundle.validate()
                if (
                    self._runtime_environment_bundle.decision_session_dates
                    != self.outer_decision_session_dates
                    or self._runtime_environment_bundle.fold_index != self.fold_index
                ):
                    raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                        "outer environment differs from its prior commitment"
                    )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _build(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV2,
    frozen_control: MassiveAdaptiveRLFrozenFC06V2,
) -> MassiveAdaptiveOuterAccessCommitmentV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(execution_registration)
        is not MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or type(frozen_policy) is not MassiveAdaptiveFrozenRLPolicyV2
        or type(frozen_control) is not MassiveAdaptiveRLFrozenFC06V2
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer access requires exact Manifest-V5 authority generations"
        )
    for authority in (
        manifest,
        manifest_registration,
        execution_registration,
        policy_schedule,
        frozen_policy,
        frozen_control,
    ):
        authority.validate()
    fold_index = frozen_policy.fold_index
    outer_fold_receipt = policy_schedule.outer_fold_receipts[fold_index]
    outer_session_dates = policy_schedule.outer_session_date_inventories[fold_index]
    if (
        frozen_control.fold_index != fold_index
        or not policy_schedule.authorizes_outer_fold(fold_index)
        or not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or manifest_registration.experiment_id != manifest.experiment_id
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.experiment_id != manifest.experiment_id
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or policy_schedule.experiment_id != manifest.experiment_id
        or policy_schedule.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or policy_schedule.scientific_protocol_projection_sha256
        != manifest.scientific_protocol_projection_sha256
        or policy_schedule.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
        or policy_schedule.execution_implementation_registration_receipt_sha256
        != execution_registration.semantic_receipt_sha256
        or policy_schedule.scientific_execution_fingerprint_sha256
        != execution_registration.scientific_execution_fingerprint_sha256
        or frozen_policy.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or frozen_control.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or frozen_policy.execution_implementation_registration_receipt_sha256
        != execution_registration.semantic_receipt_sha256
        or frozen_control.execution_implementation_registration_receipt_sha256
        != execution_registration.semantic_receipt_sha256
        or frozen_policy.scientific_execution_fingerprint_sha256
        != execution_registration.scientific_execution_fingerprint_sha256
        or frozen_control.scientific_execution_fingerprint_sha256
        != execution_registration.scientific_execution_fingerprint_sha256
        or frozen_policy.policy_selection_authority_receipt_sha256
        != frozen_control.policy_selection_authority_receipt_sha256
        or policy_schedule.frozen_policy(fold_index).semantic_receipt_sha256
        != frozen_policy.semantic_receipt_sha256
        or policy_schedule.frozen_control(fold_index).semantic_receipt_sha256
        != frozen_control.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error("outer-access roots differ")
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "manifest_v5_registration_receipt_sha256": manifest_registration.semantic_receipt_sha256,
        "execution_implementation_registration_receipt_sha256": execution_registration.semantic_receipt_sha256,
        "scientific_execution_fingerprint_sha256": execution_registration.scientific_execution_fingerprint_sha256,
        "fold_index": fold_index,
        "policy_schedule_receipt_sha256": policy_schedule.semantic_receipt_sha256,
        "policy_schedule_source_receipt_sha256": _required_digest(
            "policy schedule source", policy_schedule.source_receipt_sha256
        ),
        "policy_schedule_commit_receipt_sha256": _required_digest(
            "policy schedule commit", policy_schedule.source_transaction_receipt_sha256
        ),
        "policy_schedule_committed_at_ms": _required_time(
            "policy schedule time", policy_schedule.source_transaction_committed_at_ms
        ),
        "frozen_ppo_policy_receipt_sha256": frozen_policy.semantic_receipt_sha256,
        "frozen_ppo_source_receipt_sha256": _required_digest(
            "frozen PPO source", frozen_policy.source_receipt_sha256
        ),
        "frozen_ppo_commit_receipt_sha256": _required_digest(
            "frozen PPO commit", frozen_policy.source_transaction_receipt_sha256
        ),
        "frozen_ppo_committed_at_ms": _required_time(
            "frozen PPO time", frozen_policy.source_transaction_committed_at_ms
        ),
        "frozen_fc06_control_receipt_sha256": frozen_control.semantic_receipt_sha256,
        "frozen_fc06_source_receipt_sha256": _required_digest(
            "frozen FC06 source", frozen_control.source_receipt_sha256
        ),
        "frozen_fc06_commit_receipt_sha256": _required_digest(
            "frozen FC06 commit", frozen_control.source_transaction_receipt_sha256
        ),
        "frozen_fc06_committed_at_ms": _required_time(
            "frozen FC06 time", frozen_control.source_transaction_committed_at_ms
        ),
        "outer_fold_receipt_sha256": outer_fold_receipt,
        "outer_decision_session_dates": outer_session_dates,
        "outer_decision_inventory_sha256": semantic_sha256(outer_session_dates),
        "policy_validation_eligible": frozen_policy.selected_candidate_validation_eligible,
        "source_data_qualified": bool(
            execution_registration.source_data_qualified
            and policy_schedule.source_data_qualified
            and frozen_policy.source_data_qualified
            and frozen_control.source_data_qualified
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_OUTER_ACCESS_COMMITMENT_V2_SCHEMA,
    }
    provisional = MassiveAdaptiveOuterAccessCommitmentV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_commitment_replayed=True,
        outer_input_access_authorized=bool(body["source_data_qualified"]),
        _runtime_schedule=policy_schedule,
        _runtime_frozen_policy=frozen_policy,
        _runtime_frozen_control=frozen_control,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse(
    root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveOuterAccessCommitmentV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer-access payload is not canonical JSON"
        )
    body = dict(value)
    body["outer_decision_session_dates"] = tuple(
        str(item)
        for item in cast(Sequence[object], body["outer_decision_session_dates"])
    )
    result = MassiveAdaptiveOuterAccessCommitmentV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_outer_access_commitment_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    frozen_policy: MassiveAdaptiveFrozenRLPolicyV2,
    frozen_control: MassiveAdaptiveRLFrozenFC06V2,
    allow_materialize: bool = True,
) -> MassiveAdaptiveOuterAccessCommitmentV2:
    expected = _build(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        policy_schedule=policy_schedule,
        frozen_policy=frozen_policy,
        frozen_control=frozen_control,
    )
    relative = outer_access_commitment_relative_path_v2(
        manifest=manifest, fold_index=expected.fold_index
    )
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        payload = Path(root) / relative
        present = tuple(
            path.exists() or path.is_symlink()
            for path in (
                payload,
                payload.with_name(payload.name + ".receipt.json"),
                payload.with_name(payload.name + ".commit.json"),
            )
        )
        if any(present) and not all(present):
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "outer-access transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                    "outer-access commitment is absent"
                )
            committed_at_ms = (
                max(
                    time.time_ns() // 1_000_000,
                    expected.policy_schedule_committed_at_ms,
                    expected.frozen_ppo_committed_at_ms,
                    expected.frozen_fc06_committed_at_ms,
                )
                + 1
            )
            capability = issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1(
                root=root, authority=manifest_registration
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=root, capability=capability
            ):
                publish_massive_source_object(
                    stream=BytesIO(
                        canonical_json_file_bytes(expected.semantic_unsigned())
                    ),
                    root=root,
                    relative_payload_path=relative,
                    dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_DATASET,
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=MASSIVE_ADAPTIVE_RL_OUTER_ACCESS_COMMITMENT_V2_SOURCE_SCHEMA_SHA256,
                    entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=f"ADAPTIVE-RL-OUTER-ACCESS-V2-{manifest.experiment_id}-FOLD{expected.fold_index}",
                )
        parsed = _parse(
            root,
            load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveOuterAccessCommitmentV2Error(
                "outer-access commitment does not replay"
            )
        result = replace(
            parsed,
            runtime_commitment_replayed=True,
            outer_input_access_authorized=parsed.source_data_qualified,
            _runtime_schedule=policy_schedule,
            _runtime_frozen_policy=frozen_policy,
            _runtime_frozen_control=frozen_control,
        )
        result.validate()
        return result


def _authorize_massive_adaptive_outer_access_environment_v2(
    *,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    low_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    primary_environment: MassiveAdaptiveProfitabilityEnvV1,
    high_cost_environment: MassiveAdaptiveProfitabilityEnvV1,
    fixed_control_environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveOuterAccessCommitmentV2:
    """Attach package-owned outer inputs only after commitment publication.

    This helper is intentionally private.  The public commitment surface has no
    environment argument, and the public rollout surface accepts only the
    commitment returned here.
    """

    if type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2:
        raise MassiveAdaptiveOuterAccessCommitmentV2Error(
            "outer environment requires an exact V2 access commitment"
        )
    bundle = _build_massive_adaptive_rl_outer_environment_bundle_v2(
        outer_access=outer_access,
        fold_index=outer_access.fold_index,
        low_cost_environment=low_cost_environment,
        primary_environment=primary_environment,
        high_cost_environment=high_cost_environment,
        fixed_control_environment=fixed_control_environment,
    )
    result = replace(outer_access, _runtime_environment_bundle=bundle)
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveOuterAccessCommitmentV2",
    "MassiveAdaptiveOuterAccessCommitmentV2Error",
    "MassiveAdaptiveRLOuterEnvironmentBundleV2",
    "outer_access_commitment_relative_path_v2",
    "run_or_resume_massive_adaptive_outer_access_commitment_v2",
]
