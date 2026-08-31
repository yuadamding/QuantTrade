"""Create-only development profitability report authority for adaptive RL.

The report is downstream of replay-authorized V4 outer evidence.  It adds the
load-bearing absolute-profitability test that relative outer evidence cannot
provide: a one-sided 95% lower confidence bound over liquidation-adjusted
daily strategy net log returns.  Generic reloads remain diagnostic-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import mean, stdev

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_outer_evidence_v1 import (
    _nonwrapping_fold_cluster_lcb,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_authority_v4 import (
    MassiveAdaptiveRLOuterEvidenceAuthorityV4,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v4 import (
    MassiveAdaptiveAuthenticatedRLOuterFoldV4,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_v1 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
    MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
)


MASSIVE_ADAPTIVE_RL_PROFITABILITY_FOLD_REPORT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-profitability-fold-report-v1"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-profitability-report-v1"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-profitability-report-authority-v1"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-profitability-report-authority-v1"
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SCHEMA,
            "publication": "create-only-source-transaction",
            "generic_reload": "diagnostic-only",
            "replay": "v4-evidence-and-primary-ppo-rollouts",
        }
    )
)
MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "outer_evidence": "replay-authorized-v4",
        "absolute_economics": "primary-ppo-daily-net-log-return",
        "terminal_adjustment": "liquidation-adjusted-final-equity",
        "absolute_gate": "one-sided-95pct-lcb-positive",
        "bootstrap": "same-fold-cluster-nonwrapping-63-session-v1",
        "annualization_sessions": 252,
        "zero_volatility_ratio": "zero-v1",
        "development_reporting_only": True,
        "live_trading": False,
        "lockbox": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(ValueError):
    """The report did not replay from authenticated daily economics."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "adaptive RL profitability report artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(name: str, value: object) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            f"{name} must be finite"
        )
    return result


def _annualized_ratio(values: Sequence[float]) -> float:
    rows = tuple(float(value) for value in values)
    if len(rows) < 2:
        return 0.0
    dispersion = stdev(rows)
    if dispersion <= 1.0e-15:
        return 0.0
    return math.sqrt(252.0) * mean(rows) / dispersion


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProfitabilityFoldReportV1:
    fold_index: int
    outer_rollout_authority_receipt_sha256: str
    outer_rollout_receipt_sha256: str
    transition_inventory_sha256: str
    decision_session_dates: tuple[str, ...]
    liquidation_adjusted_strategy_net_log_returns: tuple[float, ...]
    cumulative_net_log_return: float
    terminal_liquidation_adjusted_return: float
    annualized_net_return: float
    annualized_volatility: float
    net_sharpe_ratio: float
    cumulative_active_log_return: float
    cumulative_incremental_log_return: float
    cumulative_ppo_minus_fixed_log_return: float
    maximum_drawdown: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_PROFITABILITY_FOLD_REPORT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        rows = self.liquidation_adjusted_strategy_net_log_returns
        numbers = (
            *rows,
            self.cumulative_net_log_return,
            self.terminal_liquidation_adjusted_return,
            self.annualized_net_return,
            self.annualized_volatility,
            self.net_sharpe_ratio,
            self.cumulative_active_log_return,
            self.cumulative_incremental_log_return,
            self.cumulative_ppo_minus_fixed_log_return,
            self.maximum_drawdown,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PROFITABILITY_FOLD_REPORT_V1_SCHEMA
            or not 0 <= self.fold_index < MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or len(self.decision_session_dates)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or len(rows) != len(self.decision_session_dates)
            or tuple(sorted(self.decision_session_dates))
            != self.decision_session_dates
            or len(set(self.decision_session_dates)) != len(self.decision_session_dates)
            or any(not math.isfinite(value) for value in numbers)
            or self.annualized_volatility < 0.0
            or not 0.0 <= self.maximum_drawdown <= 1.0
            or not math.isclose(
                self.cumulative_net_log_return,
                sum(rows),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                self.terminal_liquidation_adjusted_return,
                math.expm1(self.cumulative_net_log_return),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
                "adaptive RL fold profitability report differs"
            )
        for value in (
            self.outer_rollout_authority_receipt_sha256,
            self.outer_rollout_receipt_sha256,
            self.transition_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fold profitability report", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProfitabilityReportV1:
    outer_evidence_authority_v4_receipt_sha256: str
    outer_evidence_v4_receipt_sha256: str
    fold_reports: tuple[MassiveAdaptiveRLProfitabilityFoldReportV1, ...]
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
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for fold in self.fold_reports:
            fold.validate()
        absolute_gate = self.primary_net_log_return_lcb95 > 0.0
        expected_absolute_name = "primary-net-log-return-lcb-positive"
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V1_SCHEMA
            or tuple(fold.fold_index for fold in self.fold_reports)
            != tuple(range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1))
            or self.fold_report_inventory_sha256
            != semantic_sha256(
                tuple(fold.semantic_receipt_sha256 for fold in self.fold_reports)
            )
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
            or (expected_absolute_name in self.passed_gate_names) != absolute_gate
            or (expected_absolute_name in self.failed_gate_names) == absolute_gate
            or self.passed_gate_names != tuple(sorted(set(self.passed_gate_names)))
            or self.failed_gate_names != tuple(sorted(set(self.failed_gate_names)))
            or set(self.passed_gate_names) & set(self.failed_gate_names)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
                "adaptive RL profitability report differs"
            )
        for value in (
            self.outer_evidence_authority_v4_receipt_sha256,
            self.outer_evidence_v4_receipt_sha256,
            self.fold_report_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL profitability report", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProfitabilityReportAuthorityV1:
    report: MassiveAdaptiveRLProfitabilityReportV1
    report_source_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_report: MassiveAdaptiveRLProfitabilityReportV1 | None
    runtime_report_replayed: bool
    development_profitability_reporting_authorized: bool
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "report_receipt_sha256": self.report.semantic_receipt_sha256,
            "report_source_receipt_sha256": self.report_source_receipt_sha256,
            "source_data_qualified": self.source_data_qualified,
            "live_trading_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.report.validate()
        self.loaded_source.validate()
        runtime_present = self.runtime_report is not None
        expected_authorized = bool(
            runtime_present
            and self.source_data_qualified
            and not self.report.failed_gate_names
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SCHEMA
            or self.report_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.report.semantic_receipt_sha256
            or self.source_data_qualified != self.report.source_data_qualified
            or self.runtime_report_replayed != runtime_present
            or self.development_profitability_reporting_authorized
            != expected_authorized
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
                "adaptive RL profitability report authority differs"
            )
        if self.runtime_report is not None and (
            self.runtime_report.semantic_unsigned() != self.report.semantic_unsigned()
        ):
            raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
                "adaptive RL runtime profitability report differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _fold_report(
    *,
    fold: MassiveAdaptiveAuthenticatedRLOuterFoldV4,
    rollout_authority: MassiveAdaptiveRLOuterRolloutAuthorityV1,
) -> MassiveAdaptiveRLProfitabilityFoldReportV1:
    rollout_authority.validate()
    rollout = rollout_authority.runtime_rollout
    if (
        rollout is None
        or not rollout_authority.runtime_rollout_replayed
        or not rollout_authority.outer_evaluation_authorized
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "primary PPO outer rollout is not replay authorized"
        )
    authenticated_v2 = fold.authenticated_fold_v3.authenticated_fold_v2
    cost_fold = authenticated_v2.cost_fold
    if (
        rollout.fold_index != fold.fold_index
        or rollout_authority.semantic_receipt_sha256
        != authenticated_v2.outer_rollout_authority_receipt_sha256
        or rollout.semantic_receipt_sha256
        != authenticated_v2.outer_rollout_receipt_sha256
        or rollout.policy_trace.semantic_receipt_sha256
        != cost_fold.primary_trace_receipt_sha256
        or len(rollout.transitions) != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
        or len(rollout.policy_trace.decision_session_dates)
        != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "primary PPO rollout differs from authenticated V4 evidence"
        )
    for transition in rollout.transitions:
        transition.validate()
    if any(transition.truncated for transition in rollout.transitions):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "outer profitability report cannot contain rollout truncation"
        )
    if (
        any(transition.terminated for transition in rollout.transitions[:-1])
        or not rollout.transitions[-1].terminated
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "outer profitability report has an invalid economic endpoint"
        )
    rows = [
        float(transition.economic_step.strategy_net_log_return)
        for transition in rollout.transitions
    ]
    final_transition = rollout.transitions[-1]
    marked_equity = (
        final_transition.economic_step.strategy_posttrade_book.marked_equity
    )
    if marked_equity <= 0.0 or final_transition.strategy_liquidation_adjusted_equity <= 0.0:
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "outer strategy terminal equity is invalid"
        )
    rows[-1] += math.log(
        final_transition.strategy_liquidation_adjusted_equity / marked_equity
    )
    cumulative = sum(rows)
    terminal_return = rollout.policy_trace.terminal_liquidation_adjusted_return
    if not math.isclose(
        cumulative,
        math.log1p(terminal_return),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "outer daily net returns do not reconcile to terminal strategy wealth"
        )
    annualized_log = cumulative * 252.0 / len(rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_FOLD_REPORT_V1_SCHEMA,
        "fold_index": rollout.fold_index,
        "outer_rollout_authority_receipt_sha256": (
            rollout_authority.semantic_receipt_sha256
        ),
        "outer_rollout_receipt_sha256": rollout.semantic_receipt_sha256,
        "transition_inventory_sha256": rollout.transition_inventory_sha256,
        "decision_session_dates": rollout.policy_trace.decision_session_dates,
        "liquidation_adjusted_strategy_net_log_returns": tuple(rows),
        "cumulative_net_log_return": cumulative,
        "terminal_liquidation_adjusted_return": terminal_return,
        "annualized_net_return": math.expm1(annualized_log),
        "annualized_volatility": stdev(rows) * math.sqrt(252.0),
        "net_sharpe_ratio": _annualized_ratio(rows),
        "cumulative_active_log_return": sum(
            cost_fold.primary_strategy_active_log_returns
        ),
        "cumulative_incremental_log_return": sum(
            cost_fold.primary_incremental_rl_log_returns
        ),
        "cumulative_ppo_minus_fixed_log_return": sum(
            cost_fold.primary_ppo_minus_fixed_control_log_returns
        ),
        "maximum_drawdown": cost_fold.maximum_drawdown,
        "source_data_qualified": bool(
            rollout.source_data_qualified
            and rollout_authority.source_data_qualified
            and fold.source_data_qualified
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveRLProfitabilityFoldReportV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def build_massive_adaptive_rl_profitability_report_v1(
    *,
    outer_evidence_authority_v4: MassiveAdaptiveRLOuterEvidenceAuthorityV4,
    ppo_outer_rollout_authorities: Sequence[
        MassiveAdaptiveRLOuterRolloutAuthorityV1
    ],
) -> MassiveAdaptiveRLProfitabilityReportV1:
    """Reconcile authenticated daily economics and build a diagnostic report."""

    outer_evidence_authority_v4.validate()
    evidence = outer_evidence_authority_v4.runtime_evidence
    folds = outer_evidence_authority_v4.runtime_folds
    if (
        evidence is None
        or folds is None
        or not outer_evidence_authority_v4.runtime_evidence_replayed
        or len(folds) != MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
        or len(ppo_outer_rollout_authorities)
        != MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "V4 outer evidence is not replay authorized"
        )
    authority_by_fold = {
        authority.fold_index: authority for authority in ppo_outer_rollout_authorities
    }
    if tuple(sorted(authority_by_fold)) != tuple(
        range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1)
    ):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "profitability report requires one PPO rollout authority per fold"
        )
    ordered_folds = tuple(sorted(folds, key=lambda value: value.fold_index))
    fold_reports = tuple(
        _fold_report(fold=fold, rollout_authority=authority_by_fold[fold.fold_index])
        for fold in ordered_folds
    )
    net = tuple(
        fold.liquidation_adjusted_strategy_net_log_returns for fold in fold_reports
    )
    active = tuple(
        fold.authenticated_fold_v3.authenticated_fold_v2.cost_fold.primary_strategy_active_log_returns
        for fold in ordered_folds
    )
    incremental = tuple(
        fold.authenticated_fold_v3.authenticated_fold_v2.cost_fold.primary_incremental_rl_log_returns
        for fold in ordered_folds
    )
    ppo_minus_fixed = tuple(
        fold.authenticated_fold_v3.authenticated_fold_v2.cost_fold.primary_ppo_minus_fixed_control_log_returns
        for fold in ordered_folds
    )
    flat_net = tuple(value for rows in net for value in rows)
    flat_active = tuple(value for rows in active for value in rows)
    flat_incremental = tuple(value for rows in incremental for value in rows)
    flat_ppo_minus_fixed = tuple(
        value for rows in ppo_minus_fixed for value in rows
    )
    net_lcb = _nonwrapping_fold_cluster_lcb(net)
    inherited_passed = set(evidence.passed_gate_names)
    inherited_failed = set(evidence.failed_gate_names)
    absolute_gate = "primary-net-log-return-lcb-positive"
    (inherited_passed if net_lcb > 0.0 else inherited_failed).add(absolute_gate)
    mean_net = mean(flat_net)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_V1_SCHEMA,
        "outer_evidence_authority_v4_receipt_sha256": (
            outer_evidence_authority_v4.semantic_receipt_sha256
        ),
        "outer_evidence_v4_receipt_sha256": evidence.semantic_receipt_sha256,
        "fold_reports": fold_reports,
        "fold_report_inventory_sha256": semantic_sha256(
            tuple(fold.semantic_receipt_sha256 for fold in fold_reports)
        ),
        "mean_primary_net_log_return": mean_net,
        "primary_net_log_return_lcb95": net_lcb,
        "annualized_net_return": math.expm1(252.0 * mean_net),
        "annualized_volatility": stdev(flat_net) * math.sqrt(252.0),
        "net_sharpe_ratio": _annualized_ratio(flat_net),
        "active_information_ratio": _annualized_ratio(flat_active),
        "incremental_information_ratio": _annualized_ratio(flat_incremental),
        "ppo_minus_fixed_information_ratio": _annualized_ratio(
            flat_ppo_minus_fixed
        ),
        "mean_terminal_liquidation_adjusted_return": mean(
            fold.terminal_liquidation_adjusted_return for fold in fold_reports
        ),
        "mean_high_cost_terminal_return": (
            evidence.evidence_v3.evidence_v2.evidence_v1.mean_high_cost_terminal_return
        ),
        "maximum_fold_drawdown": max(
            fold.maximum_drawdown for fold in fold_reports
        ),
        "mean_high_cost_ppo_minus_fixed_log_return": (
            evidence.mean_high_cost_ppo_minus_fixed_control_log_return
        ),
        "passed_gate_names": tuple(sorted(inherited_passed)),
        "failed_gate_names": tuple(sorted(inherited_failed)),
        "source_data_qualified": bool(
            outer_evidence_authority_v4.source_data_qualified
            and evidence.source_data_qualified
            and all(fold.source_data_qualified for fold in fold_reports)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLProfitabilityReportV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _payload(report: MassiveAdaptiveRLProfitabilityReportV1) -> dict[str, object]:
    report.validate()
    return {
        **report.semantic_unsigned(),
        "semantic_receipt_sha256": report.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "adaptive RL profitability report is not canonical JSON"
        )
    return dict(value)


def _tuple_strings(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            f"{name} inventory differs"
        )
    return tuple(value)


def _tuple_floats(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            f"{name} inventory differs"
        )
    return tuple(_finite(name, item) for item in value)


def _parse_fold(value: object) -> MassiveAdaptiveRLProfitabilityFoldReportV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "adaptive RL fold report differs"
        )
    result = MassiveAdaptiveRLProfitabilityFoldReportV1(
        fold_index=int(value["fold_index"]),
        outer_rollout_authority_receipt_sha256=str(
            value["outer_rollout_authority_receipt_sha256"]
        ),
        outer_rollout_receipt_sha256=str(value["outer_rollout_receipt_sha256"]),
        transition_inventory_sha256=str(value["transition_inventory_sha256"]),
        decision_session_dates=_tuple_strings(
            "decision-session", value["decision_session_dates"]
        ),
        liquidation_adjusted_strategy_net_log_returns=_tuple_floats(
            "strategy-net-log-return",
            value["liquidation_adjusted_strategy_net_log_returns"],
        ),
        cumulative_net_log_return=_finite(
            "cumulative net log return", value["cumulative_net_log_return"]
        ),
        terminal_liquidation_adjusted_return=_finite(
            "terminal liquidation-adjusted return",
            value["terminal_liquidation_adjusted_return"],
        ),
        annualized_net_return=_finite(
            "annualized net return", value["annualized_net_return"]
        ),
        annualized_volatility=_finite(
            "annualized volatility", value["annualized_volatility"]
        ),
        net_sharpe_ratio=_finite("net Sharpe ratio", value["net_sharpe_ratio"]),
        cumulative_active_log_return=_finite(
            "cumulative active log return", value["cumulative_active_log_return"]
        ),
        cumulative_incremental_log_return=_finite(
            "cumulative incremental log return",
            value["cumulative_incremental_log_return"],
        ),
        cumulative_ppo_minus_fixed_log_return=_finite(
            "cumulative PPO-minus-fixed log return",
            value["cumulative_ppo_minus_fixed_log_return"],
        ),
        maximum_drawdown=_finite("maximum drawdown", value["maximum_drawdown"]),
        source_data_qualified=bool(value["source_data_qualified"]),
        semantic_receipt_sha256=str(value["semantic_receipt_sha256"]),
        protocol_receipt_sha256=str(value["protocol_receipt_sha256"]),
        schema=str(value["schema"]),
    )
    result.validate()
    return result


def parse_massive_adaptive_rl_profitability_report_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV1:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    fold_values = payload["fold_reports"]
    if not isinstance(fold_values, list):
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "adaptive RL fold report inventory differs"
        )
    report = MassiveAdaptiveRLProfitabilityReportV1(
        outer_evidence_authority_v4_receipt_sha256=str(
            payload["outer_evidence_authority_v4_receipt_sha256"]
        ),
        outer_evidence_v4_receipt_sha256=str(
            payload["outer_evidence_v4_receipt_sha256"]
        ),
        fold_reports=tuple(_parse_fold(value) for value in fold_values),
        fold_report_inventory_sha256=str(payload["fold_report_inventory_sha256"]),
        mean_primary_net_log_return=_finite(
            "mean primary net log return", payload["mean_primary_net_log_return"]
        ),
        primary_net_log_return_lcb95=_finite(
            "primary net log return LCB", payload["primary_net_log_return_lcb95"]
        ),
        annualized_net_return=_finite(
            "annualized net return", payload["annualized_net_return"]
        ),
        annualized_volatility=_finite(
            "annualized volatility", payload["annualized_volatility"]
        ),
        net_sharpe_ratio=_finite("net Sharpe ratio", payload["net_sharpe_ratio"]),
        active_information_ratio=_finite(
            "active information ratio", payload["active_information_ratio"]
        ),
        incremental_information_ratio=_finite(
            "incremental information ratio",
            payload["incremental_information_ratio"],
        ),
        ppo_minus_fixed_information_ratio=_finite(
            "PPO-minus-fixed information ratio",
            payload["ppo_minus_fixed_information_ratio"],
        ),
        mean_terminal_liquidation_adjusted_return=_finite(
            "mean terminal liquidation-adjusted return",
            payload["mean_terminal_liquidation_adjusted_return"],
        ),
        mean_high_cost_terminal_return=_finite(
            "mean high-cost terminal return",
            payload["mean_high_cost_terminal_return"],
        ),
        maximum_fold_drawdown=_finite(
            "maximum fold drawdown", payload["maximum_fold_drawdown"]
        ),
        mean_high_cost_ppo_minus_fixed_log_return=_finite(
            "mean high-cost PPO-minus-fixed log return",
            payload["mean_high_cost_ppo_minus_fixed_log_return"],
        ),
        passed_gate_names=_tuple_strings(
            "passed-gate", payload["passed_gate_names"]
        ),
        failed_gate_names=_tuple_strings(
            "failed-gate", payload["failed_gate_names"]
        ),
        source_data_qualified=bool(payload["source_data_qualified"]),
        semantic_receipt_sha256=str(payload["semantic_receipt_sha256"]),
        protocol_receipt_sha256=str(payload["protocol_receipt_sha256"]),
        specification_sha256=str(payload["specification_sha256"]),
        schema=str(payload["schema"]),
    )
    report.validate()
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SCHEMA,
        "report": report,
        "report_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_data_qualified": report.source_data_qualified,
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLProfitabilityReportAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_report=None,
        runtime_report_replayed=False,
        development_profitability_reporting_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_profitability_report_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLProfitabilityReportAuthorityV1,
    outer_evidence_authority_v4: MassiveAdaptiveRLOuterEvidenceAuthorityV4,
    ppo_outer_rollout_authorities: Sequence[
        MassiveAdaptiveRLOuterRolloutAuthorityV1
    ],
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV1:
    parsed = parse_massive_adaptive_rl_profitability_report_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    replayed = build_massive_adaptive_rl_profitability_report_v1(
        outer_evidence_authority_v4=outer_evidence_authority_v4,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
    )
    if replayed.semantic_unsigned() != parsed.report.semantic_unsigned():
        raise MassiveAdaptiveRLProfitabilityReportAuthorityV1Error(
            "adaptive RL profitability report did not replay"
        )
    result = replace(
        parsed,
        runtime_report=replayed,
        runtime_report_replayed=True,
        development_profitability_reporting_authorized=bool(
            replayed.source_data_qualified and not replayed.failed_gate_names
        ),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_profitability_report_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    outer_evidence_authority_v4: MassiveAdaptiveRLOuterEvidenceAuthorityV4,
    ppo_outer_rollout_authorities: Sequence[
        MassiveAdaptiveRLOuterRolloutAuthorityV1
    ],
    committed_at_ms: int,
) -> MassiveAdaptiveRLProfitabilityReportAuthorityV1:
    artifact = _artifact_id(artifact_id)
    report = build_massive_adaptive_rl_profitability_report_v1(
        outer_evidence_authority_v4=outer_evidence_authority_v4,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
    )
    relative = f"massive-adaptive/rl-profitability-report-authority-v1/{artifact}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(report))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=report.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-PROFITABILITY-REPORT-V1-{artifact}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return authorize_massive_adaptive_rl_profitability_report_authority_v1(
        root=root,
        authority=parse_massive_adaptive_rl_profitability_report_authority_v1(
            root=root, loaded_source=loaded
        ),
        outer_evidence_authority_v4=outer_evidence_authority_v4,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLProfitabilityFoldReportV1",
    "MassiveAdaptiveRLProfitabilityReportAuthorityV1",
    "MassiveAdaptiveRLProfitabilityReportAuthorityV1Error",
    "MassiveAdaptiveRLProfitabilityReportV1",
    "authorize_massive_adaptive_rl_profitability_report_authority_v1",
    "build_massive_adaptive_rl_profitability_report_v1",
    "materialize_massive_adaptive_rl_profitability_report_authority_v1",
    "parse_massive_adaptive_rl_profitability_report_authority_v1",
]
