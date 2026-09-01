"""Manifest-bound economic environments for causal adaptive-RL fitting.

The profitability environment is mutable, so it cannot itself be an immutable
experiment authority.  This module binds every environment input to one
reconstructed runtime-source graph, records the resulting immutable
configuration receipts, and creates fresh environments only from that retained
runtime witness.  Callers cannot substitute blocks, economic roots, capital,
costs, or participation limits on the authorizing path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLFitEnvironmentAuthorityV1,
    _ISSUER as _FIT_ENVIRONMENT_AUTHORITY_ISSUER,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLFitBlockRuntimeSourcesV1,
    MassiveAdaptiveRLRuntimeSourcesV1,
)


MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fit-environment-registry-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "inputs": "manifest-v3-and-witnessed-runtime-sources-v1-only",
        "coverage": "exactly-one-fresh-environment-per-causal-fit-archive",
        "economics": (
            "same-daily-fill-identity-event-capital-cost-participation-roots"
        ),
        "mutable_environment": "retained-runtime-witness-fresh-construction-only",
        "caller_fit_blocks": False,
        "caller_economic_roots": False,
        "caller_capital_or_costs": False,
        "profitability_reporting": False,
        "outer_evaluation": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFitEnvironmentRegistryV1Error(ValueError):
    """One fit environment is detached from its source or economic roots."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _authority_receipt(value: object) -> str:
    for name in ("semantic_receipt_sha256", "receipt_sha256"):
        observed = getattr(value, name, None)
        if isinstance(observed, str):
            return _digest("adaptive RL fit environment authority", observed)
    raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
        "adaptive RL fit environment authority receipt is absent"
    )


def _graph_runtime_authority(
    *, runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1, role: str
) -> object:
    return runtime_sources.runtime_source_graph_authority.runtime_authority(
        role=role,
        fold_index=None,
    )


def _validate_global_runtime_roots(
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> None:
    expected = {
        "session-authority": runtime_sources.session_authority,
        "condition-authority": runtime_sources.condition_authority,
        "identity-authority": runtime_sources.identity_authority,
        "economic-event-archive": runtime_sources.economic_event_archive,
        "daily-input-authority": runtime_sources.daily_input_authority,
        "fill-source-authority": runtime_sources.fill_source,
        "split-plan-authority": runtime_sources.split_plan,
    }
    for role, value in expected.items():
        witnessed = _graph_runtime_authority(runtime_sources=runtime_sources, role=role)
        if type(witnessed) is not type(value) or _authority_receipt(
            witnessed
        ) != _authority_receipt(value):
            raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                f"adaptive RL fit environment {role} differs from the runtime graph"
            )
    daily = runtime_sources.daily_input_authority
    fill = runtime_sources.fill_source
    identity = runtime_sources.identity_authority
    events = runtime_sources.economic_event_archive
    if (
        fill.daily_input_authority_semantic_receipt_sha256
        != daily.semantic_receipt_sha256
        or fill.session_authority_receipt_sha256
        != runtime_sources.session_authority.receipt_sha256
        or fill.condition_authority_receipt_sha256
        != runtime_sources.condition_authority.receipt_sha256
        or daily.session_authority_receipt_sha256
        != runtime_sources.session_authority.receipt_sha256
        or daily.condition_authority_receipt_sha256
        != runtime_sources.condition_authority.receipt_sha256
        or events.identity_authority_receipt_sha256 != identity.receipt_sha256
    ):
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            "adaptive RL fit environment global source edges differ"
        )


def _environment_from_block(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    block: MassiveAdaptiveRLFitBlockRuntimeSourcesV1,
) -> MassiveAdaptiveProfitabilityEnvV1:
    return MassiveAdaptiveProfitabilityEnvV1(
        forecast_archive=block.forecast_archive,
        calibration=block.calibration,
        inference_plan=block.inference_plan,
        decision_roots=block.decision_roots,
        context_origins=block.context_origins,
        fill_source=runtime_sources.fill_source,
        daily_input_authority=runtime_sources.daily_input_authority,
        identity_authority=runtime_sources.identity_authority,
        economic_event_archive=runtime_sources.economic_event_archive,
        initial_capital=manifest.base_manifest.primary_capital,
        transaction_cost_basis_points=(
            manifest.base_manifest.primary_cost_basis_points
        ),
        maximum_fill_participation=(manifest.base_manifest.maximum_fill_participation),
    )


def _environment_authority(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    block: MassiveAdaptiveRLFitBlockRuntimeSourcesV1,
    environment: MassiveAdaptiveProfitabilityEnvV1,
) -> MassiveAdaptiveRLFitEnvironmentAuthorityV1:
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if runtime_receipt is None:
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            "adaptive RL fit environment runtime graph witness is absent"
        )
    plan_dates = tuple(row.decision_session_date for row in block.inference_plan.rows)
    daily_dates = {
        row.source_session_date
        for row in runtime_sources.daily_input_authority.sessions
    }
    fill_dates = set(runtime_sources.fill_source.session_dates)
    required_fill_dates = {row.next_session_date for row in block.inference_plan.rows}
    security_ids = set(block.forecast_archive.security_ids)
    source_qualified = bool(
        runtime_sources.source_data_qualified
        and block.source_data_qualified
        and runtime_sources.daily_input_authority.source_transport_qualified
        and runtime_sources.daily_input_authority.daily_input_data_qualified
        and runtime_sources.fill_source.source_paths_replayed
        and runtime_sources.fill_source.source_data_qualified
        and set(plan_dates) <= daily_dates
        and required_fill_dates <= daily_dates
        and required_fill_dates <= fill_dates
        and security_ids
        <= set(runtime_sources.daily_input_authority.supported_security_ids)
        and security_ids <= set(runtime_sources.fill_source.supported_security_ids)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "outer_fold_index": block.outer_fold_index,
        "source_fold_index": block.source_fold_index,
        "block_index": block.block_index,
        "fit_block_receipt_sha256": block.semantic_receipt_sha256,
        "forecast_archive_receipt_sha256": (
            block.forecast_archive.semantic_receipt_sha256
        ),
        "inference_plan_receipt_sha256": (block.inference_plan.semantic_receipt_sha256),
        "calibration_receipt_sha256": block.calibration.semantic_receipt_sha256,
        "decision_root_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in block.decision_roots)
        ),
        "context_origin_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in block.context_origins)
        ),
        "daily_input_authority_receipt_sha256": (
            runtime_sources.daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_receipt_sha256": (
            runtime_sources.fill_source.semantic_receipt_sha256
        ),
        "identity_authority_receipt_sha256": (
            runtime_sources.identity_authority.receipt_sha256
        ),
        "economic_event_archive_receipt_sha256": (
            runtime_sources.economic_event_archive.receipt_sha256
        ),
        "compiler_config_receipt_sha256": (environment.compiler_config.receipt_sha256),
        "initial_capital": environment.initial_capital,
        "transaction_cost_basis_points": (environment.transaction_cost_basis_points),
        "maximum_fill_participation": environment.maximum_fill_participation,
        "environment_source_inventory_sha256": (environment.source_inventory_sha256),
        "economic_compatibility_receipt_sha256": (
            environment.economic_compatibility_receipt_sha256
        ),
        "source_data_qualified": source_qualified,
        "runtime_environment_replayed": True,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFitEnvironmentAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _runtime_environment=environment,
        _issuer=_FIT_ENVIRONMENT_AUTHORITY_ISSUER,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    result.validate_environment(environment)
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitEnvironmentRegistryV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    outer_fold_index: int
    environment_authorities: tuple[MassiveAdaptiveRLFitEnvironmentAuthorityV1, ...]
    forecast_archive_receipts: tuple[str, ...]
    environment_authority_inventory_sha256: str
    environment_registry_receipt_sha256: str
    source_data_qualified: bool
    runtime_environments_replayed: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SCHEMA
    _manifest: MassiveAdaptiveRLExperimentManifestV3 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "outer_fold_index": self.outer_fold_index,
            "environment_authority_receipts": tuple(
                row.semantic_receipt_sha256 for row in self.environment_authorities
            ),
            "forecast_archive_receipts": self.forecast_archive_receipts,
            "environment_authority_inventory_sha256": (
                self.environment_authority_inventory_sha256
            ),
            "environment_registry_receipt_sha256": (
                self.environment_registry_receipt_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "runtime_environments_replayed": self.runtime_environments_replayed,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        runtime_present = (
            self._manifest is not None and self._runtime_sources is not None
        )
        partial_runtime = (self._manifest is None) != (self._runtime_sources is None)
        for row in self.environment_authorities:
            row.validate()
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            self._manifest.validate()
            self._runtime_sources.validate()
        expected_qualified = bool(
            runtime_present
            and self.environment_authorities
            and all(row.source_data_qualified for row in self.environment_authorities)
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SCHEMA
            or not self.experiment_id
            or self.outer_fold_index not in range(4)
            or partial_runtime
            or not self.environment_authorities
            or tuple(row.block_index for row in self.environment_authorities)
            != tuple(range(len(self.environment_authorities)))
            or tuple(
                row.forecast_archive_receipt_sha256
                for row in self.environment_authorities
            )
            != self.forecast_archive_receipts
            or self.forecast_archive_receipts
            != tuple(dict.fromkeys(self.forecast_archive_receipts))
            or any(
                row.experiment_id != self.experiment_id
                or row.manifest_v3_receipt_sha256 != self.manifest_v3_receipt_sha256
                or row.runtime_sources_receipt_sha256
                != self.runtime_sources_receipt_sha256
                or row.runtime_graph_witness_receipt_sha256
                != self.runtime_graph_witness_receipt_sha256
                or row.outer_fold_index != self.outer_fold_index
                for row in self.environment_authorities
            )
            or self.environment_authority_inventory_sha256
            != semantic_sha256(
                tuple(
                    row.semantic_receipt_sha256 for row in self.environment_authorities
                )
            )
            or self.environment_registry_receipt_sha256
            != semantic_sha256(
                tuple(
                    (
                        row.forecast_archive_receipt_sha256,
                        row.semantic_receipt_sha256,
                    )
                    for row in self.environment_authorities
                )
            )
            or self.runtime_environments_replayed != runtime_present
            or self.source_data_qualified != expected_qualified
            or not self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                "adaptive RL fit environment registry differs"
            )
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            runtime_receipt = self._runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
            if (
                self._manifest.experiment_id != self.experiment_id
                or self._manifest.semantic_receipt_sha256
                != self.manifest_v3_receipt_sha256
                or self._runtime_sources.experiment_id != self.experiment_id
                or self._runtime_sources.manifest_v3_receipt_sha256
                != self.manifest_v3_receipt_sha256
                or self._runtime_sources.semantic_receipt_sha256
                != self.runtime_sources_receipt_sha256
                or runtime_receipt != self.runtime_graph_witness_receipt_sha256
            ):
                raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                    "adaptive RL fit environment registry runtime witness differs"
                )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            *self.forecast_archive_receipts,
            self.environment_authority_inventory_sha256,
            self.environment_registry_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fit environment registry", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def authority(
        self, forecast_archive_receipt_sha256: str
    ) -> MassiveAdaptiveRLFitEnvironmentAuthorityV1:
        self.validate()
        matches = tuple(
            row
            for row in self.environment_authorities
            if row.forecast_archive_receipt_sha256 == forecast_archive_receipt_sha256
        )
        if len(matches) != 1:
            raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                "adaptive RL fit environment authority is absent or duplicated"
            )
        return matches[0]

    def build_environments(self) -> Mapping[str, MassiveAdaptiveProfitabilityEnvV1]:
        """Create one fresh mutable environment per immutable fit authority."""

        self.validate()
        if self._manifest is None or self._runtime_sources is None:
            raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                "adaptive RL fit environment runtime witness is absent"
            )
        fold = self._runtime_sources.fold(self.outer_fold_index)
        environments: dict[str, MassiveAdaptiveProfitabilityEnvV1] = {}
        for row in self.environment_authorities:
            block = fold.fit_block(row.block_index)
            if block.semantic_receipt_sha256 != row.fit_block_receipt_sha256:
                raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                    "adaptive RL fit block differs from its environment authority"
                )
            environment = _environment_from_block(
                manifest=self._manifest,
                runtime_sources=self._runtime_sources,
                block=block,
            )
            row.validate_environment(environment)
            environments[row.forecast_archive_receipt_sha256] = environment
        if tuple(environments) != self.forecast_archive_receipts:
            raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
                "adaptive RL fit environment registry coverage differs"
            )
        return environments


def build_massive_adaptive_rl_fit_environment_registry_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    outer_fold_index: int,
) -> MassiveAdaptiveRLFitEnvironmentRegistryV1:
    """Build all causal fit environments from one witnessed source graph."""

    manifest.validate()
    runtime_sources.validate()
    if (
        type(runtime_sources) is not MassiveAdaptiveRLRuntimeSourcesV1
        or outer_fold_index not in manifest.base_manifest.fold_indices
        or manifest.experiment_id != runtime_sources.experiment_id
        or manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or not runtime_sources.source_data_qualified
    ):
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            "adaptive RL fit environment manifest or runtime sources differ"
        )
    _validate_global_runtime_roots(runtime_sources)
    runtime_receipt = (
        runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    )
    if runtime_receipt is None:
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            "adaptive RL fit environment runtime graph witness is absent"
        )
    fold = runtime_sources.fold(outer_fold_index)
    authorities: list[MassiveAdaptiveRLFitEnvironmentAuthorityV1] = []
    for block in fold.fit_blocks:
        environment = _environment_from_block(
            manifest=manifest,
            runtime_sources=runtime_sources,
            block=block,
        )
        authorities.append(
            _environment_authority(
                manifest=manifest,
                runtime_sources=runtime_sources,
                block=block,
                environment=environment,
            )
        )
    rows = tuple(authorities)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "outer_fold_index": outer_fold_index,
        "environment_authorities": rows,
        "forecast_archive_receipts": tuple(
            row.forecast_archive_receipt_sha256 for row in rows
        ),
        "environment_authority_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in rows)
        ),
        "environment_registry_receipt_sha256": semantic_sha256(
            tuple(
                (row.forecast_archive_receipt_sha256, row.semantic_receipt_sha256)
                for row in rows
            )
        ),
        "source_data_qualified": all(row.source_data_qualified for row in rows),
        "runtime_environments_replayed": True,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFitEnvironmentRegistryV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _manifest=manifest,
        _runtime_sources=runtime_sources,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    rebuilt = result.build_environments()
    if tuple(rebuilt) != result.forecast_archive_receipts:
        raise MassiveAdaptiveRLFitEnvironmentRegistryV1Error(
            "adaptive RL fit environment registry did not replay"
        )
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_REGISTRY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFitEnvironmentAuthorityV1",
    "MassiveAdaptiveRLFitEnvironmentRegistryV1",
    "MassiveAdaptiveRLFitEnvironmentRegistryV1Error",
    "build_massive_adaptive_rl_fit_environment_registry_v1",
]
