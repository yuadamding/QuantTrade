"""Manifest-V5 development profitability report from four outer-fold seals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_fold_seal_authority_v1 import (
    MassiveAdaptiveRLOuterFoldSealAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    _annualized_ratio,
    _nonwrapping_fold_cluster_lcb,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1,
    MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1,
    MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
)
from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_scope_v1 import (
    massive_adaptive_rl_vertical_qualification_experiment_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-profitability-report-authority-v2"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
            "payload": "four-seal-development-profitability-report",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(ValueError):
    """The four-fold economics, registered gates, or report replay differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            f"{name} must be finite"
        )
    return result


def _required_time(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            f"{name} is absent or invalid"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProfitabilityFoldReportV2:
    fold_index: int
    outer_fold_seal_receipt_sha256: str
    outer_rollout_authority_receipt_sha256: str
    decision_session_dates: tuple[str, ...]
    strategy_net_log_returns: tuple[float, ...]
    benchmark_net_log_returns: tuple[float, ...]
    neutral_net_log_returns: tuple[float, ...]
    fixed_control_net_log_returns: tuple[float, ...]
    active_log_returns: tuple[float, ...]
    incremental_rl_log_returns: tuple[float, ...]
    ppo_minus_fixed_control_log_returns: tuple[float, ...]
    cumulative_net_log_return: float
    terminal_liquidation_adjusted_return: float
    low_cost_terminal_liquidation_adjusted_return: float
    high_cost_terminal_liquidation_adjusted_return: float
    fixed_control_terminal_liquidation_adjusted_return: float
    fixed_control_low_cost_terminal_liquidation_adjusted_return: float
    fixed_control_high_cost_terminal_liquidation_adjusted_return: float
    ppo_cost_ladder_monotone: bool
    fixed_control_cost_ladder_monotone: bool
    annualized_net_return: float
    annualized_volatility: float
    net_sharpe_ratio: float
    maximum_drawdown: float
    source_data_qualified: bool
    semantic_receipt_sha256: str

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if descriptor.name != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        count = len(self.decision_session_dates)
        series = (
            self.strategy_net_log_returns,
            self.benchmark_net_log_returns,
            self.neutral_net_log_returns,
            self.fixed_control_net_log_returns,
            self.active_log_returns,
            self.incremental_rl_log_returns,
            self.ppo_minus_fixed_control_log_returns,
        )
        numbers = (
            *(value for row in series for value in row),
            self.cumulative_net_log_return,
            self.terminal_liquidation_adjusted_return,
            self.low_cost_terminal_liquidation_adjusted_return,
            self.high_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_terminal_liquidation_adjusted_return,
            self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
            self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            self.annualized_net_return,
            self.annualized_volatility,
            self.net_sharpe_ratio,
            self.maximum_drawdown,
        )
        if (
            isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or count != 126
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or any(len(row) != count for row in series)
            or any(not math.isfinite(value) for value in numbers)
            or min(
                self.terminal_liquidation_adjusted_return,
                self.low_cost_terminal_liquidation_adjusted_return,
                self.high_cost_terminal_liquidation_adjusted_return,
                self.fixed_control_terminal_liquidation_adjusted_return,
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return,
                self.fixed_control_high_cost_terminal_liquidation_adjusted_return,
            )
            <= -1.0
            or not math.isclose(
                self.cumulative_net_log_return,
                sum(self.strategy_net_log_returns),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                self.terminal_liquidation_adjusted_return,
                math.expm1(self.cumulative_net_log_return),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or self.annualized_volatility < 0.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not isinstance(self.ppo_cost_ladder_monotone, bool)
            or self.ppo_cost_ladder_monotone
            != (
                self.low_cost_terminal_liquidation_adjusted_return
                >= self.terminal_liquidation_adjusted_return
                >= self.high_cost_terminal_liquidation_adjusted_return
            )
            or not isinstance(self.fixed_control_cost_ladder_monotone, bool)
            or self.fixed_control_cost_ladder_monotone
            != (
                self.fixed_control_low_cost_terminal_liquidation_adjusted_return
                >= self.fixed_control_terminal_liquidation_adjusted_return
                >= self.fixed_control_high_cost_terminal_liquidation_adjusted_return
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                "V5 fold profitability report differs"
            )
        _digest("outer-fold seal", self.outer_fold_seal_receipt_sha256)
        _digest("outer rollout", self.outer_rollout_authority_receipt_sha256)


def _fold_report(
    seal: MassiveAdaptiveRLOuterFoldSealAuthorityV1,
) -> MassiveAdaptiveRLProfitabilityFoldReportV2:
    seal.validate()
    rollout = seal.rollout_authority.rollout
    rows = rollout.strategy_net_log_returns
    cumulative = sum(rows)
    body = {
        "fold_index": seal.fold_index,
        "outer_fold_seal_receipt_sha256": seal.semantic_receipt_sha256,
        "outer_rollout_authority_receipt_sha256": (
            seal.outer_rollout_authority_receipt_sha256
        ),
        "decision_session_dates": rollout.decision_session_dates,
        "strategy_net_log_returns": rows,
        "benchmark_net_log_returns": rollout.benchmark_net_log_returns,
        "neutral_net_log_returns": rollout.neutral_net_log_returns,
        "fixed_control_net_log_returns": rollout.fixed_control_net_log_returns,
        "active_log_returns": rollout.active_log_returns,
        "incremental_rl_log_returns": rollout.incremental_rl_log_returns,
        "ppo_minus_fixed_control_log_returns": (
            rollout.ppo_minus_fixed_control_log_returns
        ),
        "cumulative_net_log_return": cumulative,
        "terminal_liquidation_adjusted_return": (
            rollout.primary_terminal_liquidation_adjusted_return
        ),
        "low_cost_terminal_liquidation_adjusted_return": (
            rollout.low_cost_terminal_liquidation_adjusted_return
        ),
        "high_cost_terminal_liquidation_adjusted_return": (
            rollout.high_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_terminal_liquidation_adjusted_return
        ),
        "fixed_control_low_cost_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_low_cost_terminal_liquidation_adjusted_return
        ),
        "fixed_control_high_cost_terminal_liquidation_adjusted_return": (
            rollout.fixed_control_high_cost_terminal_liquidation_adjusted_return
        ),
        "ppo_cost_ladder_monotone": rollout.ppo_cost_ladder_monotone,
        "fixed_control_cost_ladder_monotone": (
            rollout.fixed_control_cost_ladder_monotone
        ),
        "annualized_net_return": math.expm1(252.0 * mean(rows)),
        "annualized_volatility": stdev(rows) * math.sqrt(252.0),
        "net_sharpe_ratio": _annualized_ratio(rows),
        "maximum_drawdown": rollout.maximum_drawdown,
        "source_data_qualified": bool(
            seal.source_data_qualified and rollout.source_data_qualified
        ),
    }
    provisional = MassiveAdaptiveRLProfitabilityFoldReportV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _cost_ladder_monotonicity_gate(
    fold_reports: Sequence[MassiveAdaptiveRLProfitabilityFoldReportV2],
) -> bool:
    """Treat observed nonmonotonicity as a failed gate, not invalid evidence."""

    return bool(
        len(fold_reports) == 4
        and all(
            row.ppo_cost_ladder_monotone and row.fixed_control_cost_ladder_monotone
            for row in fold_reports
        )
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProfitabilityReportAuthorityV2:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    scientific_protocol_projection_sha256: str
    execution_implementation_registration_receipt_sha256: str
    scientific_execution_fingerprint_sha256: str
    complete_policy_schedule_receipt_sha256: str
    policy_schedule_disposition: str
    policy_schedule_qualified: bool
    outer_fold_seal_receipts: tuple[str, ...]
    outer_fold_seal_source_receipts: tuple[str, ...]
    outer_fold_seal_commit_receipts: tuple[str, ...]
    outer_fold_seal_committed_at_ms: tuple[int, ...]
    fold_reports: tuple[MassiveAdaptiveRLProfitabilityFoldReportV2, ...]
    fold_report_inventory_sha256: str
    mean_primary_net_log_return: float
    primary_net_log_return_lcb95: float
    annualized_net_return: float
    annualized_volatility: float
    net_sharpe_ratio: float
    active_information_ratio: float
    incremental_information_ratio: float
    ppo_minus_fixed_information_ratio: float
    mean_terminal_liquidation_adjusted_return: float
    mean_high_cost_terminal_return: float
    maximum_fold_drawdown: float
    mean_high_cost_ppo_minus_fixed_log_return: float
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_report_replayed: bool = False
    development_profitability_reporting_authorized: bool = False
    profitability_gates_passed: bool = False
    positive_profitability_authorization_eligible: bool = False
    end_to_end_profitability_execution_complete: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA
    _runtime_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_seals: tuple[MassiveAdaptiveRLOuterFoldSealAuthorityV1, ...] = field(
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
                "runtime_report_replayed",
                "development_profitability_reporting_authorized",
                "profitability_gates_passed",
                "positive_profitability_authorization_eligible",
                "end_to_end_profitability_execution_complete",
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

    def validate(self) -> None:
        runtime = self._runtime_schedule is not None and len(self._runtime_seals) == 4
        expected_schedule_qualified = (
            self.policy_schedule_disposition
            == MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1
        )
        expected_gates = not self.failed_gate_names
        production_experiment = not (
            massive_adaptive_rl_vertical_qualification_experiment_v1(self.experiment_id)
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or len(self.outer_fold_seal_receipts) != 4
            or len(self.outer_fold_seal_source_receipts) != 4
            or len(self.outer_fold_seal_commit_receipts) != 4
            or len(self.outer_fold_seal_committed_at_ms) != 4
            or tuple(row.fold_index for row in self.fold_reports) != (0, 1, 2, 3)
            or self.fold_report_inventory_sha256
            != semantic_sha256(
                tuple(row.semantic_receipt_sha256 for row in self.fold_reports)
            )
            or self.policy_schedule_disposition
            not in {
                MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_QUALIFIED_V1,
                MASSIVE_ADAPTIVE_RL_POLICY_PREFIX_DIAGNOSTIC_V1,
            }
            or self.policy_schedule_qualified != expected_schedule_qualified
            or self.passed_gate_names != tuple(sorted(set(self.passed_gate_names)))
            or self.failed_gate_names != tuple(sorted(set(self.failed_gate_names)))
            or set(self.passed_gate_names) & set(self.failed_gate_names)
            or set(self.passed_gate_names) | set(self.failed_gate_names)
            != set(MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3)
            or any(
                not math.isfinite(value)
                for value in (
                    self.mean_primary_net_log_return,
                    self.primary_net_log_return_lcb95,
                    self.annualized_net_return,
                    self.annualized_volatility,
                    self.net_sharpe_ratio,
                    self.active_information_ratio,
                    self.incremental_information_ratio,
                    self.ppo_minus_fixed_information_ratio,
                    self.mean_terminal_liquidation_adjusted_return,
                    self.mean_high_cost_terminal_return,
                    self.maximum_fold_drawdown,
                    self.mean_high_cost_ppo_minus_fixed_log_return,
                )
            )
            or self.annualized_volatility < 0.0
            or not 0.0 <= self.maximum_fold_drawdown <= 1.0
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_report_replayed != runtime
            or self.development_profitability_reporting_authorized
            != bool(runtime and self.source_data_qualified)
            or self.profitability_gates_passed != bool(runtime and expected_gates)
            or self.positive_profitability_authorization_eligible
            != bool(
                runtime
                and self.source_data_qualified
                and production_experiment
                and expected_schedule_qualified
                and expected_gates
            )
            or self.end_to_end_profitability_execution_complete
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                "V5 profitability report authority differs"
            )
        for row in self.fold_reports:
            row.validate()
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        for inventory in (
            self.outer_fold_seal_receipts,
            self.outer_fold_seal_source_receipts,
            self.outer_fold_seal_commit_receipts,
        ):
            for value in inventory:
                _digest("profitability report inventory", value)
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= max(self.outer_fold_seal_committed_at_ms)
            ):
                raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                    "V5 profitability report source transaction differs"
                )
        if runtime:
            assert self._runtime_schedule is not None
            self._runtime_schedule.validate()
            for seal in self._runtime_seals:
                seal.validate()
            if (
                not self._runtime_schedule.complete_schedule
                or self._runtime_schedule.experiment_id != self.experiment_id
                or self._runtime_schedule.manifest_v5_receipt_sha256
                != self.manifest_v5_receipt_sha256
                or self._runtime_schedule.execution_implementation_registration_receipt_sha256
                != self.execution_implementation_registration_receipt_sha256
                or self._runtime_schedule.scientific_execution_fingerprint_sha256
                != self.scientific_execution_fingerprint_sha256
                or self._runtime_schedule.semantic_receipt_sha256
                != self.complete_policy_schedule_receipt_sha256
                or self._runtime_schedule.policy_schedule_disposition
                != self.policy_schedule_disposition
                or self._runtime_schedule.all_policies_validation_eligible
                != self.policy_schedule_qualified
                or tuple(seal.fold_index for seal in self._runtime_seals)
                != (0, 1, 2, 3)
                or tuple(seal.semantic_receipt_sha256 for seal in self._runtime_seals)
                != self.outer_fold_seal_receipts
                or tuple(seal.source_receipt_sha256 for seal in self._runtime_seals)
                != self.outer_fold_seal_source_receipts
                or tuple(
                    seal.source_transaction_receipt_sha256
                    for seal in self._runtime_seals
                )
                != self.outer_fold_seal_commit_receipts
                or tuple(
                    seal.source_transaction_committed_at_ms
                    for seal in self._runtime_seals
                )
                != self.outer_fold_seal_committed_at_ms
                or self._runtime_schedule.predecessor_outer_fold_seal_receipts
                != self.outer_fold_seal_receipts[:2]
                or any(
                    seal.experiment_id != self.experiment_id
                    or seal.manifest_v5_receipt_sha256
                    != self.manifest_v5_receipt_sha256
                    or seal.execution_implementation_registration_receipt_sha256
                    != self.execution_implementation_registration_receipt_sha256
                    or seal.scientific_execution_fingerprint_sha256
                    != self.scientific_execution_fingerprint_sha256
                    or seal.frozen_ppo_policy_receipt_sha256
                    != self._runtime_schedule.frozen_ppo_policy_receipts[
                        seal.fold_index
                    ]
                    or seal.frozen_fc06_control_receipt_sha256
                    != self._runtime_schedule.frozen_fc06_control_receipts[
                        seal.fold_index
                    ]
                    or seal.decision_session_dates
                    != self._runtime_schedule.outer_session_date_inventories[
                        seal.fold_index
                    ]
                    for seal in self._runtime_seals
                )
                or any(
                    seal.policy_schedule_receipt_sha256
                    != self.complete_policy_schedule_receipt_sha256
                    for seal in self._runtime_seals[2:]
                )
                or any(
                    not seal.development_outer_fold_sealed
                    for seal in self._runtime_seals
                )
            ):
                raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                    "V5 profitability report runtime lineage differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _build_report_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    seals: Sequence[MassiveAdaptiveRLOuterFoldSealAuthorityV1],
) -> dict[str, object]:
    ordered = tuple(seals)
    seal_commit_times = tuple(
        _required_time("outer seal time", row.source_transaction_committed_at_ms)
        for row in ordered
    )
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or any(
            type(row) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1
            for row in ordered
        )
        or tuple(row.fold_index for row in ordered) != (0, 1, 2, 3)
        or not schedule.complete_schedule
        or any(not row.development_outer_fold_sealed for row in ordered)
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "profitability report requires four ordered authenticated seals"
        )
    manifest.validate()
    schedule.validate()
    for row in ordered:
        row.validate()
    if (
        schedule.experiment_id != manifest.experiment_id
        or schedule.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or schedule.scientific_protocol_projection_sha256
        != manifest.scientific_protocol_projection_sha256
        or schedule.predecessor_outer_fold_seal_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered[:2])
        or any(
            left >= right
            for left, right in zip(seal_commit_times, seal_commit_times[1:])
        )
        or any(
            row.experiment_id != manifest.experiment_id
            or row.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
            or row.execution_implementation_registration_receipt_sha256
            != schedule.execution_implementation_registration_receipt_sha256
            or row.scientific_execution_fingerprint_sha256
            != schedule.scientific_execution_fingerprint_sha256
            or row.frozen_ppo_policy_receipt_sha256
            != schedule.frozen_ppo_policy_receipts[row.fold_index]
            or row.frozen_fc06_control_receipt_sha256
            != schedule.frozen_fc06_control_receipts[row.fold_index]
            or row.decision_session_dates
            != schedule.outer_session_date_inventories[row.fold_index]
            for row in ordered
        )
        or any(
            row.policy_schedule_receipt_sha256 != schedule.semantic_receipt_sha256
            for row in ordered[2:]
        )
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "profitability report lineage differs"
        )
    fold_reports = tuple(_fold_report(row) for row in ordered)
    net = tuple(row.strategy_net_log_returns for row in fold_reports)
    active = tuple(row.active_log_returns for row in fold_reports)
    incremental = tuple(row.incremental_rl_log_returns for row in fold_reports)
    ppo_fixed = tuple(row.ppo_minus_fixed_control_log_returns for row in fold_reports)
    flat_net = tuple(value for row in net for value in row)
    flat_active = tuple(value for row in active for value in row)
    flat_incremental = tuple(value for row in incremental for value in row)
    flat_ppo_fixed = tuple(value for row in ppo_fixed for value in row)
    net_lcb = _nonwrapping_fold_cluster_lcb(net)
    active_lcb = _nonwrapping_fold_cluster_lcb(active)
    incremental_lcb = _nonwrapping_fold_cluster_lcb(incremental)
    ppo_fixed_lcb = _nonwrapping_fold_cluster_lcb(ppo_fixed)
    high_cost = tuple(
        math.log1p(row.high_cost_terminal_liquidation_adjusted_return)
        for row in fold_reports
    )
    high_cost_ppo_fixed = tuple(
        left
        - math.log1p(row.fixed_control_high_cost_terminal_liquidation_adjusted_return)
        for left, row in zip(high_cost, fold_reports, strict=True)
    )
    threshold = manifest.base_manifest.base_manifest.base_manifest.maximum_fold_drawdown
    gate_results = {
        "cost-ladder-monotone": _cost_ladder_monotonicity_gate(fold_reports),
        "high-cost-mean-return-nonnegative": mean(high_cost) >= 0.0,
        "high-cost-ppo-minus-fixed-control-nonnegative": (
            mean(high_cost_ppo_fixed) >= 0.0
        ),
        "incremental-rl-lcb-positive": incremental_lcb > 0.0,
        "maximum-fold-drawdown": max(row.maximum_drawdown for row in fold_reports)
        <= threshold,
        "positive-incremental-folds-at-least-three": sum(
            sum(row) > 0.0 for row in incremental
        )
        >= 3,
        "positive-ppo-minus-fixed-folds-at-least-three": sum(
            sum(row) > 0.0 for row in ppo_fixed
        )
        >= 3,
        "positive-strategy-folds-at-least-three": sum(sum(row) > 0.0 for row in net)
        >= 3,
        "ppo-minus-fixed-control-lcb-positive": ppo_fixed_lcb > 0.0,
        "primary-net-log-return-lcb-positive": net_lcb > 0.0,
        "strategy-active-lcb-positive": active_lcb > 0.0,
    }
    if set(gate_results) != set(MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "registered profitability gate inventory differs"
        )
    source_receipts = tuple(
        _digest("outer seal source", row.source_receipt_sha256) for row in ordered
    )
    commit_receipts = tuple(
        _digest("outer seal commit", row.source_transaction_receipt_sha256)
        for row in ordered
    )
    commit_times = tuple(
        _required_time("outer seal time", row.source_transaction_committed_at_ms)
        for row in ordered
    )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "scientific_protocol_projection_sha256": manifest.scientific_protocol_projection_sha256,
        "execution_implementation_registration_receipt_sha256": (
            schedule.execution_implementation_registration_receipt_sha256
        ),
        "scientific_execution_fingerprint_sha256": (
            schedule.scientific_execution_fingerprint_sha256
        ),
        "complete_policy_schedule_receipt_sha256": schedule.semantic_receipt_sha256,
        "policy_schedule_disposition": schedule.policy_schedule_disposition,
        "policy_schedule_qualified": schedule.all_policies_validation_eligible,
        "outer_fold_seal_receipts": tuple(
            row.semantic_receipt_sha256 for row in ordered
        ),
        "outer_fold_seal_source_receipts": source_receipts,
        "outer_fold_seal_commit_receipts": commit_receipts,
        "outer_fold_seal_committed_at_ms": commit_times,
        "fold_reports": fold_reports,
        "fold_report_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in fold_reports)
        ),
        "mean_primary_net_log_return": mean(flat_net),
        "primary_net_log_return_lcb95": net_lcb,
        "annualized_net_return": math.expm1(252.0 * mean(flat_net)),
        "annualized_volatility": stdev(flat_net) * math.sqrt(252.0),
        "net_sharpe_ratio": _annualized_ratio(flat_net),
        "active_information_ratio": _annualized_ratio(flat_active),
        "incremental_information_ratio": _annualized_ratio(flat_incremental),
        "ppo_minus_fixed_information_ratio": _annualized_ratio(flat_ppo_fixed),
        "mean_terminal_liquidation_adjusted_return": mean(
            row.terminal_liquidation_adjusted_return for row in fold_reports
        ),
        "mean_high_cost_terminal_return": mean(
            row.high_cost_terminal_liquidation_adjusted_return for row in fold_reports
        ),
        "maximum_fold_drawdown": max(row.maximum_drawdown for row in fold_reports),
        "mean_high_cost_ppo_minus_fixed_log_return": mean(high_cost_ppo_fixed),
        "passed_gate_names": tuple(
            sorted(name for name, passed in gate_results.items() if passed)
        ),
        "failed_gate_names": tuple(
            sorted(name for name, passed in gate_results.items() if not passed)
        ),
        "source_data_qualified": bool(
            schedule.source_data_qualified
            and all(row.source_data_qualified for row in ordered)
        ),
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SCHEMA,
    }


def build_massive_adaptive_rl_profitability_report_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    outer_fold_seals: Sequence[MassiveAdaptiveRLOuterFoldSealAuthorityV1],
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or any(
            type(seal) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1
            for seal in outer_fold_seals
        )
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "profitability report requires exact Manifest-V5 authorities"
        )
    body = _build_report_body(
        manifest=manifest, schedule=policy_schedule, seals=outer_fold_seals
    )
    gates_pass = not body["failed_gate_names"]
    provisional = MassiveAdaptiveRLProfitabilityReportAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_report_replayed=True,
        development_profitability_reporting_authorized=bool(
            body["source_data_qualified"]
        ),
        profitability_gates_passed=gates_pass,
        positive_profitability_authorization_eligible=bool(
            body["source_data_qualified"]
            and not massive_adaptive_rl_vertical_qualification_experiment_v1(
                manifest.experiment_id
            )
            and body["policy_schedule_qualified"]
            and gates_pass
        ),
        _runtime_schedule=policy_schedule,
        _runtime_seals=tuple(outer_fold_seals),
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def profitability_report_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> str:
    manifest.validate()
    return f"adaptive-rl/{manifest.experiment_id}/profitability-report-authority-v2/report.json"


def _parse_fold(value: object) -> MassiveAdaptiveRLProfitabilityFoldReportV2:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "V5 fold report payload differs"
        )
    body = dict(value)
    for name in (
        "decision_session_dates",
        "strategy_net_log_returns",
        "benchmark_net_log_returns",
        "neutral_net_log_returns",
        "fixed_control_net_log_returns",
        "active_log_returns",
        "incremental_rl_log_returns",
        "ppo_minus_fixed_control_log_returns",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    result = MassiveAdaptiveRLProfitabilityFoldReportV2(**body)  # type: ignore[arg-type]
    result.validate()
    return result


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "V5 profitability report payload is not canonical JSON"
        )
    body = dict(value)
    body["fold_reports"] = tuple(
        _parse_fold(row) for row in cast(Sequence[object], body["fold_reports"])
    )
    for name in (
        "outer_fold_seal_receipts",
        "outer_fold_seal_source_receipts",
        "outer_fold_seal_commit_receipts",
        "outer_fold_seal_committed_at_ms",
        "passed_gate_names",
        "failed_gate_names",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    result = MassiveAdaptiveRLProfitabilityReportAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_profitability_report_authority_v2(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    policy_schedule: MassiveAdaptiveRLWalkForwardPolicyScheduleV1,
    outer_fold_seals: Sequence[MassiveAdaptiveRLOuterFoldSealAuthorityV1],
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(policy_schedule) is not MassiveAdaptiveRLWalkForwardPolicyScheduleV1
        or any(
            type(seal) is not MassiveAdaptiveRLOuterFoldSealAuthorityV1
            for seal in outer_fold_seals
        )
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "profitability report requires exact Manifest-V5 authorities"
        )
    manifest_registration.validate()
    if (
        not manifest_registration.development_protocol_registered
        or manifest_registration.experiment_id != manifest.experiment_id
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or policy_schedule.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
            "profitability report registration lineage differs"
        )
    expected = build_massive_adaptive_rl_profitability_report_authority_v2(
        manifest=manifest,
        policy_schedule=policy_schedule,
        outer_fold_seals=outer_fold_seals,
    )
    relative = profitability_report_authority_relative_path_v2(manifest=manifest)
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        payload = Path(root) / relative
        transaction_paths = (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
        present = tuple(
            path.exists() or path.is_symlink() for path in transaction_paths
        )
        if any(present) and not all(present):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                "V5 profitability report transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                    "V5 profitability report is absent"
                )
            committed_at_ms = (
                max(
                    time.time_ns() // 1_000_000,
                    *expected.outer_fold_seal_committed_at_ms,
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
                    dataset_id=MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_DATASET,
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V2_SOURCE_SCHEMA_SHA256,
                    entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=f"ADAPTIVE-RL-PROFITABILITY-REPORT-V2-{manifest.experiment_id}",
                )
        parsed = _parse(
            root=root,
            loaded=load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV2Error(
                "V5 profitability report does not replay"
            )
        gates_pass = not parsed.failed_gate_names
        result = replace(
            parsed,
            runtime_report_replayed=True,
            development_profitability_reporting_authorized=parsed.source_data_qualified,
            profitability_gates_passed=gates_pass,
            positive_profitability_authorization_eligible=bool(
                parsed.source_data_qualified
                and not massive_adaptive_rl_vertical_qualification_experiment_v1(
                    parsed.experiment_id
                )
                and parsed.policy_schedule_qualified
                and gates_pass
            ),
            _runtime_schedule=policy_schedule,
            _runtime_seals=tuple(outer_fold_seals),
        )
        result.validate()
        return result


__all__ = [
    "MassiveAdaptiveRLProfitabilityFoldReportV2",
    "MassiveAdaptiveRLProfitabilityReportAuthorityV2",
    "MassiveAdaptiveRLProfitabilityReportAuthorityV2Error",
    "build_massive_adaptive_rl_profitability_report_authority_v2",
    "profitability_report_authority_relative_path_v2",
    "run_or_resume_massive_adaptive_rl_profitability_report_authority_v2",
]
