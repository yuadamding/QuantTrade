"""Immutable witness for one source-qualified adaptive-RL fit environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fit-environment-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "inputs": "manifest-v3-and-witnessed-runtime-sources-v1-only",
        "economics": (
            "same-daily-fill-identity-event-capital-cost-participation-roots"
        ),
        "runtime": "exact-fresh-environment-reconciliation",
        "caller_economic_roots": False,
        "caller_capital_or_costs": False,
        "profitability_reporting": False,
        "outer_evaluation": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFitEnvironmentAuthorityV1Error(ValueError):
    """One fit environment differs from its immutable authority."""


_ISSUER = object()


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFitEnvironmentAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitEnvironmentAuthorityV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    outer_fold_index: int
    source_fold_index: int
    block_index: int
    fit_block_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    decision_root_inventory_sha256: str
    context_origin_inventory_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_archive_receipt_sha256: str
    compiler_config_receipt_sha256: str
    initial_capital: float
    transaction_cost_basis_points: float
    maximum_fill_participation: float
    environment_source_inventory_sha256: str
    economic_compatibility_receipt_sha256: str
    source_data_qualified: bool
    runtime_environment_replayed: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA
    _runtime_environment: MassiveAdaptiveProfitabilityEnvV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _issuer: object = field(default=None, repr=False, compare=False)

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "semantic_receipt_sha256",
                "_runtime_environment",
                "_issuer",
            }
        }

    @staticmethod
    def _environment_source_data_qualified(
        environment: MassiveAdaptiveProfitabilityEnvV1,
    ) -> bool:
        plan_dates = tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        return bool(
            (
                getattr(
                    environment.forecast_archive,
                    "development_forecast_authorized",
                    False,
                )
                or getattr(
                    environment.forecast_archive,
                    "outer_forecast_authorized",
                    False,
                )
            )
            and environment.calibration.development_calibration_authorized
            and getattr(environment.inference_plan, "source_data_qualified", False)
            and all(environment.roots[date].source_data_qualified for date in plan_dates)
            and all(
                environment.contexts[date].source_data_qualified for date in plan_dates
            )
            and environment.daily_input_authority.source_transport_qualified
            and environment.daily_input_authority.daily_input_data_qualified
            and environment.fill_source.source_paths_replayed
            and environment.fill_source.source_data_qualified
            and environment.economic_event_archive is not None
        )

    def _validate_environment_config(
        self, environment: MassiveAdaptiveProfitabilityEnvV1
    ) -> None:
        plan_dates = tuple(
            row.decision_session_date for row in environment.inference_plan.rows
        )
        if (
            environment.forecast_archive.semantic_receipt_sha256
            != self.forecast_archive_receipt_sha256
            or environment.inference_plan.semantic_receipt_sha256
            != self.inference_plan_receipt_sha256
            or environment.calibration.semantic_receipt_sha256
            != self.calibration_receipt_sha256
            or semantic_sha256(
                tuple(
                    environment.roots[date].semantic_receipt_sha256
                    for date in plan_dates
                )
            )
            != self.decision_root_inventory_sha256
            or semantic_sha256(
                tuple(
                    environment.contexts[date].semantic_receipt_sha256
                    for date in plan_dates
                )
            )
            != self.context_origin_inventory_sha256
            or environment.daily_input_authority.semantic_receipt_sha256
            != self.daily_input_authority_receipt_sha256
            or environment.fill_source.semantic_receipt_sha256
            != self.fill_source_receipt_sha256
            or environment.identity_authority.receipt_sha256
            != self.identity_authority_receipt_sha256
            or environment.economic_event_archive is None
            or environment.economic_event_archive.receipt_sha256
            != self.economic_event_archive_receipt_sha256
            or environment.compiler_config.receipt_sha256
            != self.compiler_config_receipt_sha256
            or environment.initial_capital != self.initial_capital
            or environment.transaction_cost_basis_points
            != self.transaction_cost_basis_points
            or environment.maximum_fill_participation
            != self.maximum_fill_participation
            or environment.source_inventory_sha256
            != self.environment_source_inventory_sha256
            or environment.economic_compatibility_receipt_sha256
            != self.economic_compatibility_receipt_sha256
        ):
            raise MassiveAdaptiveRLFitEnvironmentAuthorityV1Error(
                "adaptive RL fit environment runtime differs from its authority"
            )

    def validate(self) -> None:
        runtime_present = self._runtime_environment is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA
            or self._issuer is not _ISSUER
            or not self.experiment_id
            or self.outer_fold_index not in range(4)
            or self.source_fold_index not in range(self.outer_fold_index + 1)
            or self.block_index < 0
            or self.initial_capital <= 0.0
            or self.transaction_cost_basis_points < 0.0
            or not 0.0 < self.maximum_fill_participation <= 1.0
            or self.runtime_environment_replayed != runtime_present
            or self.source_data_qualified
            != bool(
                runtime_present
                and self._runtime_environment is not None
                and self._environment_source_data_qualified(
                    self._runtime_environment
                )
            )
            or not self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFitEnvironmentAuthorityV1Error(
                "adaptive RL fit environment authority differs"
            )
        assert self._runtime_environment is not None
        self._validate_environment_config(self._runtime_environment)
        for name in (
            "manifest_v3_receipt_sha256",
            "runtime_sources_receipt_sha256",
            "runtime_graph_witness_receipt_sha256",
            "fit_block_receipt_sha256",
            "forecast_archive_receipt_sha256",
            "inference_plan_receipt_sha256",
            "calibration_receipt_sha256",
            "decision_root_inventory_sha256",
            "context_origin_inventory_sha256",
            "daily_input_authority_receipt_sha256",
            "fill_source_receipt_sha256",
            "identity_authority_receipt_sha256",
            "economic_event_archive_receipt_sha256",
            "compiler_config_receipt_sha256",
            "environment_source_inventory_sha256",
            "economic_compatibility_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())

    def validate_environment(
        self, environment: MassiveAdaptiveProfitabilityEnvV1
    ) -> None:
        """Reconcile one fresh mutable environment with this authority."""

        self.validate()
        self._validate_environment_config(environment)


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FIT_ENVIRONMENT_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFitEnvironmentAuthorityV1",
    "MassiveAdaptiveRLFitEnvironmentAuthorityV1Error",
]
