"""Exact frozen source reconciliation for Massive P0 outer evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v2 import (
    MassiveProfitabilityEvaluationPlanV2,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MassiveProfitabilityFeatureAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MassiveProfitabilityFillSourceAuthorityV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
    build_massive_profitability_targets_v2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MassiveProfitabilityTerminalCoverageAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA = (
    "rl-quant.massive-profitability-evaluation-source-bundle-v3"
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_DATASET = (
    "massive-profitability-evaluation-source-bundle-v3"
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA,
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
        }
    )
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "scope": "embargoed-outer-folds-only",
        "roots": (
            "data-gate-v2",
            "phase-plan-v2",
            "evaluation-plan-v2",
            "accounting-freeze-v1",
            "origin-plan-v2",
            "daily-input-authority-v1",
            "fill-source-authority-v2",
            "terminal-coverage-authority-v1",
        ),
        "feature_binding": "exact-gate-subset-and-feature-accounting",
        "target_binding": "exact-gate-subset-rebuilt-from-frozen-target-accounting",
        "coverage_binding": "semantic-and-audit-receipts-exact-accounting-freeze-row",
        "lockbox": "excluded",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityEvaluationSourceBundleV3Error(ValueError):
    """Outer evaluation sources differ from the gate or accounting freeze."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 artifact ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationSourceDateV3:
    decision_session_date: str
    origin_receipt_sha256: str
    feature_semantic_receipt_sha256: str
    target_semantic_receipt_sha256: str
    feature_accounting_semantic_receipt_sha256: str
    target_accounting_semantic_receipt_sha256: str
    economic_coverage_semantic_receipt_sha256: str
    economic_coverage_audit_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.decision_session_date:
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source date is empty"
            )
        for value in (
            self.origin_receipt_sha256,
            self.feature_semantic_receipt_sha256,
            self.target_semantic_receipt_sha256,
            self.feature_accounting_semantic_receipt_sha256,
            self.target_accounting_semantic_receipt_sha256,
            self.economic_coverage_semantic_receipt_sha256,
            self.economic_coverage_audit_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("evaluation source date", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source date receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationSourceBundleV3:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    evaluation_plan_semantic_receipt_sha256: str
    archive_freeze_semantic_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    daily_input_authority_semantic_receipt_sha256: str
    fill_source_authority_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    terminal_accounting_mode: str
    outer_test_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityEvaluationSourceDateV3, ...]
    feature_inventory_sha256: str
    target_inventory_sha256: str
    feature_accounting_inventory_sha256: str
    target_accounting_inventory_sha256: str
    row_inventory_sha256: str
    source_bundle_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    component_audit_inventory_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "component_audit_inventory_sha256",
                "audit_receipt_sha256",
                "loaded_source",
            }
        }

    def validate(self) -> None:
        dates = tuple(row.decision_session_date for row in self.rows)
        if (
            self.schema != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SHA256
            or dates != self.outer_test_session_dates
            or dates != tuple(sorted(set(dates)))
            or self.terminal_accounting_mode
            not in {"exact-provider", "conservative-lower-bound"}
            or self.source_bundle_data_qualified is not True
            or not self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source bundle V3 identity or authorization differs"
            )
        for row in self.rows:
            row.validate()
        if (
            self.feature_inventory_sha256
            != semantic_sha256(
                tuple(row.feature_semantic_receipt_sha256 for row in self.rows)
            )
            or self.target_inventory_sha256
            != semantic_sha256(
                tuple(row.target_semantic_receipt_sha256 for row in self.rows)
            )
            or self.feature_accounting_inventory_sha256
            != semantic_sha256(
                tuple(
                    row.feature_accounting_semantic_receipt_sha256 for row in self.rows
                )
            )
            or self.target_accounting_inventory_sha256
            != semantic_sha256(
                tuple(
                    row.target_accounting_semantic_receipt_sha256 for row in self.rows
                )
            )
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source bundle V3 inventory differs"
            )
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.evaluation_plan_semantic_receipt_sha256,
            self.archive_freeze_semantic_receipt_sha256,
            self.accounting_freeze_semantic_receipt_sha256,
            self.origin_plan_semantic_receipt_sha256,
            self.daily_input_authority_semantic_receipt_sha256,
            self.fill_source_authority_semantic_receipt_sha256,
            self.terminal_authority_semantic_receipt_sha256,
            self.feature_inventory_sha256,
            self.target_inventory_sha256,
            self.feature_accounting_inventory_sha256,
            self.target_accounting_inventory_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.component_audit_inventory_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("evaluation source bundle V3", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source bundle V3 semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_plan_semantic_receipt_sha256
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "component_audit_inventory_sha256": (
                        self.component_audit_inventory_sha256
                    ),
                    "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
                }
            )
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "evaluation source bundle V3 committed source differs"
            )


_T = TypeVar("_T")


def _exact_date_map(name: str, rows: Sequence[_T]) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for row in rows:
        session_date = getattr(row, "decision_session_date", None)
        if (
            not isinstance(session_date, str)
            or not session_date
            or session_date in result
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                f"{name} dates are not exact and unique"
            )
        result[session_date] = row
    return result


def reconcile_massive_profitability_evaluation_source_date_v3(
    *,
    expected_feature_receipt_sha256: str,
    expected_target_receipt_sha256: str,
    feature: MassiveProfitabilityOriginFeaturesV3,
    target: MassiveProfitabilityTargetsV2,
    feature_accounting: MassiveProfitabilityFeatureAccountingAuthorityV2,
    target_accounting: MassiveProfitabilityTargetAccountingAuthorityV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_source_authority: MassiveProfitabilityFillSourceAuthorityV2,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
) -> MassiveProfitabilityEvaluationSourceDateV3:
    """Rebuild one target and bind one feature/target pair to exact frozen roots."""

    for root_authority in (
        feature,
        target,
        feature_accounting,
        target_accounting,
        accounting_freeze,
        origin_plan,
        daily_input_authority,
        fill_source_authority,
        terminal_authority,
    ):
        root_authority.validate()
    origin = next(
        (
            row
            for row in origin_plan.origin_plan_v1.origins
            if row.receipt_sha256 == feature.origin_receipt_sha256
        ),
        None,
    )
    frozen_coverage = next(
        (
            row
            for row in accounting_freeze.coverage_rows
            if row.origin_receipt_sha256 == feature.origin_receipt_sha256
        ),
        None,
    )
    if origin is None or frozen_coverage is None:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source date is absent from the frozen origin or accounting plan"
        )
    if (
        expected_feature_receipt_sha256 != feature.semantic_receipt_sha256
        or expected_target_receipt_sha256 != target.semantic_receipt_sha256
        or feature.decision_session_date != origin.decision_session_date
        or target.decision_session_date != origin.decision_session_date
        or target_accounting.decision_session_date != origin.decision_session_date
        or feature.origin_receipt_sha256 != origin.receipt_sha256
        or target.origin_receipt_sha256 != origin.receipt_sha256
        or target_accounting.origin_receipt_sha256 != origin.receipt_sha256
        or feature_accounting.origin_receipt_sha256 != origin.receipt_sha256
        or feature.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or target.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or target_accounting.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or feature_accounting.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or feature.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or feature_accounting.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or target_accounting.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or target_accounting.fill_source_authority_semantic_receipt_sha256
        != fill_source_authority.semantic_receipt_sha256
        or feature.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or feature_accounting.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or target.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or target_accounting.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or feature.accounting_freeze_semantic_receipt_sha256
        != accounting_freeze.semantic_receipt_sha256
        or target.accounting_freeze_semantic_receipt_sha256
        != accounting_freeze.semantic_receipt_sha256
        or feature.feature_accounting_authority_semantic_receipt_sha256
        != feature_accounting.semantic_receipt_sha256
        or target.target_accounting_authority_semantic_receipt_sha256
        != target_accounting.semantic_receipt_sha256
        or feature_accounting.economic_archive_audit_receipt_sha256
        != frozen_coverage.economic_coverage_audit_receipt_sha256
        or target_accounting.economic_archive_audit_receipt_sha256
        != frozen_coverage.economic_coverage_audit_receipt_sha256
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "outer feature or target is detached from its exact frozen sources"
        )
    regenerated = build_massive_profitability_targets_v2(
        accounting=target_accounting,
        origin_plan=origin_plan,
        accounting_freeze=accounting_freeze,
        terminal_authority=terminal_authority,
    )
    if regenerated.semantic_receipt_sha256 != target.semantic_receipt_sha256:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "outer target does not regenerate from its frozen accounting path"
        )
    body = {
        "decision_session_date": origin.decision_session_date,
        "origin_receipt_sha256": origin.receipt_sha256,
        "feature_semantic_receipt_sha256": feature.semantic_receipt_sha256,
        "target_semantic_receipt_sha256": target.semantic_receipt_sha256,
        "feature_accounting_semantic_receipt_sha256": (
            feature_accounting.semantic_receipt_sha256
        ),
        "target_accounting_semantic_receipt_sha256": (
            target_accounting.semantic_receipt_sha256
        ),
        "economic_coverage_semantic_receipt_sha256": (
            frozen_coverage.economic_coverage_semantic_receipt_sha256
        ),
        "economic_coverage_audit_receipt_sha256": (
            frozen_coverage.economic_coverage_audit_receipt_sha256
        ),
    }
    row = MassiveProfitabilityEvaluationSourceDateV3(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    row.validate()
    return row


def _reconcile_sources(
    *,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_source_authority: MassiveProfitabilityFillSourceAuthorityV2,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
) -> tuple[MassiveProfitabilityEvaluationSourceDateV3, ...]:
    for root_authority in (
        data_gate,
        phase_plan,
        evaluation_plan,
        accounting_freeze,
        origin_plan,
        daily_input_authority,
        fill_source_authority,
        terminal_authority,
    ):
        root_authority.validate()
    outer_dates = tuple(
        value
        for fold in phase_plan.outer_folds
        for value in fold.outer_test_session_dates
    )
    if (
        evaluation_plan.data_gate_semantic_receipt_sha256
        != data_gate.semantic_receipt_sha256
        or evaluation_plan.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or data_gate.accounting_freeze_semantic_receipt_sha256
        != accounting_freeze.semantic_receipt_sha256
        or data_gate.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or data_gate.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or data_gate.fill_source_authority_semantic_receipt_sha256
        != fill_source_authority.semantic_receipt_sha256
        or data_gate.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or accounting_freeze.archive_freeze_semantic_receipt_sha256
        != data_gate.archive_freeze_semantic_receipt_sha256
        or accounting_freeze.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
        or accounting_freeze.terminal_authority_semantic_receipt_sha256
        != terminal_authority.semantic_receipt_sha256
        or accounting_freeze.terminal_accounting_mode
        != terminal_authority.terminal_accounting_mode
        or daily_input_authority.archive_freeze_semantic_receipt_sha256
        != data_gate.archive_freeze_semantic_receipt_sha256
        or fill_source_authority.daily_input_authority_semantic_receipt_sha256
        != daily_input_authority.semantic_receipt_sha256
        or fill_source_authority.origin_plan_semantic_receipt_sha256
        != origin_plan.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation roots are detached from the DataGate or accounting freeze"
        )
    gate_feature_by_date = dict(
        zip(data_gate.gated_session_dates, data_gate.feature_receipts, strict=True)
    )
    gate_target_by_date = dict(
        zip(data_gate.gated_session_dates, data_gate.target_receipts, strict=True)
    )
    feature_by_date = _exact_date_map("outer feature", features)
    target_by_date = _exact_date_map("outer target", targets)
    target_accounting_by_date = _exact_date_map(
        "outer target accounting", target_accounting
    )
    feature_accounting_by_origin = {
        row.origin_receipt_sha256: row for row in feature_accounting
    }
    if len(feature_accounting_by_origin) != len(feature_accounting):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "feature accounting origins are not unique"
        )
    if (
        tuple(sorted(feature_by_date)) != outer_dates
        or tuple(sorted(target_by_date)) != outer_dates
        or tuple(sorted(target_accounting_by_date)) != outer_dates
        or any(session_date not in gate_feature_by_date for session_date in outer_dates)
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source dates differ from the embargoed outer folds"
        )
    rows: list[MassiveProfitabilityEvaluationSourceDateV3] = []
    for session_date in outer_dates:
        feature = feature_by_date[session_date]
        target = target_by_date[session_date]
        target_path = target_accounting_by_date[session_date]
        feature_path = feature_accounting_by_origin.get(feature.origin_receipt_sha256)
        if feature_path is None:
            raise MassiveProfitabilityEvaluationSourceBundleV3Error(
                "outer date lacks frozen feature accounting"
            )
        rows.append(
            reconcile_massive_profitability_evaluation_source_date_v3(
                expected_feature_receipt_sha256=gate_feature_by_date[session_date],
                expected_target_receipt_sha256=gate_target_by_date[session_date],
                feature=feature,
                target=target,
                feature_accounting=feature_path,
                target_accounting=target_path,
                accounting_freeze=accounting_freeze,
                origin_plan=origin_plan,
                daily_input_authority=daily_input_authority,
                fill_source_authority=fill_source_authority,
                terminal_authority=terminal_authority,
            )
        )
    return tuple(rows)


def materialize_massive_profitability_evaluation_source_bundle_v3(
    *,
    root: str | Path,
    artifact_id: str,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    evaluation_plan: MassiveProfitabilityEvaluationPlanV2,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    fill_source_authority: MassiveProfitabilityFillSourceAuthorityV2,
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    feature_accounting: Sequence[MassiveProfitabilityFeatureAccountingAuthorityV2],
    target_accounting: Sequence[MassiveProfitabilityTargetAccountingAuthorityV2],
    committed_at_ms: int,
) -> MassiveProfitabilityEvaluationSourceBundleV3:
    """Publish one exact gate/accounting-bound outer-evaluation source inventory."""

    rows = _reconcile_sources(
        data_gate=data_gate,
        phase_plan=phase_plan,
        evaluation_plan=evaluation_plan,
        accounting_freeze=accounting_freeze,
        origin_plan=origin_plan,
        daily_input_authority=daily_input_authority,
        fill_source_authority=fill_source_authority,
        terminal_authority=terminal_authority,
        features=features,
        targets=targets,
        feature_accounting=feature_accounting,
        target_accounting=target_accounting,
    )
    qualified = (
        data_gate.data_gate_passed
        and evaluation_plan.outer_evaluation_authorized
        and accounting_freeze.accounting_sources_frozen
        and daily_input_authority.daily_input_data_qualified
        and fill_source_authority.fill_source_data_qualified
        and terminal_authority.terminal_accounting_data_qualified
        and all(row.source_inputs_data_qualified for row in features)
        and all(row.economic_values_data_qualified for row in feature_accounting)
        and all(row.source_inputs_data_qualified for row in targets)
        and all(
            row.fill_sources_qualified
            and row.economic_values_data_qualified
            and row.terminal_accounting_complete
            for row in target_accounting
        )
    )
    if not qualified:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 roots are not data-qualified"
        )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "evaluation_plan_semantic_receipt_sha256": (
            evaluation_plan.semantic_receipt_sha256
        ),
        "archive_freeze_semantic_receipt_sha256": (
            data_gate.archive_freeze_semantic_receipt_sha256
        ),
        "accounting_freeze_semantic_receipt_sha256": (
            accounting_freeze.semantic_receipt_sha256
        ),
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "daily_input_authority_semantic_receipt_sha256": (
            daily_input_authority.semantic_receipt_sha256
        ),
        "fill_source_authority_semantic_receipt_sha256": (
            fill_source_authority.semantic_receipt_sha256
        ),
        "terminal_authority_semantic_receipt_sha256": (
            terminal_authority.semantic_receipt_sha256
        ),
        "terminal_accounting_mode": terminal_authority.terminal_accounting_mode,
        "outer_test_session_dates": tuple(row.decision_session_date for row in rows),
        "rows": tuple(asdict(row) for row in rows),
        "feature_inventory_sha256": semantic_sha256(
            tuple(row.feature_semantic_receipt_sha256 for row in rows)
        ),
        "target_inventory_sha256": semantic_sha256(
            tuple(row.target_semantic_receipt_sha256 for row in rows)
        ),
        "feature_accounting_inventory_sha256": semantic_sha256(
            tuple(row.feature_accounting_semantic_receipt_sha256 for row in rows)
        ),
        "target_accounting_inventory_sha256": semantic_sha256(
            tuple(row.target_accounting_semantic_receipt_sha256 for row in rows)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "source_bundle_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SHA256
        ),
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    component_audit = semantic_sha256(
        {
            "data_gate_audit_receipt_sha256": data_gate.audit_receipt_sha256,
            "accounting_freeze_audit_receipt_sha256": (
                accounting_freeze.audit_receipt_sha256
            ),
            "daily_input_audit_receipt_sha256": daily_input_authority.audit_receipt_sha256,
            "fill_source_audit_receipt_sha256": fill_source_authority.audit_receipt_sha256,
            "terminal_authority_audit_receipt_sha256": (
                terminal_authority.audit_receipt_sha256
            ),
            "feature_audit_receipts": tuple(
                row.audit_receipt_sha256 for row in features
            ),
            "target_audit_receipts": tuple(row.audit_receipt_sha256 for row in targets),
            "feature_accounting_audit_receipts": tuple(
                row.audit_receipt_sha256 for row in feature_accounting
            ),
            "target_accounting_audit_receipts": tuple(
                row.audit_receipt_sha256 for row in target_accounting
            ),
        }
    )
    payload = {
        **semantic,
        "component_audit_inventory_sha256": component_audit,
    }
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/evaluation-source-bundle-v3/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=evaluation_plan.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-EVALUATION-SOURCE-BUNDLE-V3-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    result = parse_massive_profitability_evaluation_source_bundle_v3(
        root=root, loaded_source=loaded
    )
    result.validate()
    return result


def parse_massive_profitability_evaluation_source_bundle_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityEvaluationSourceBundleV3:
    """Reload the exact source bundle summary and regenerate its canonical bytes."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 is not canonical JSON"
        )
    try:
        rows = tuple(
            MassiveProfitabilityEvaluationSourceDateV3(**row)
            for row in payload.pop("rows")
        )
        payload["outer_test_session_dates"] = tuple(payload["outer_test_session_dates"])
        component_audit = payload.pop("component_audit_inventory_sha256")
        result = MassiveProfitabilityEvaluationSourceBundleV3(
            **payload,
            rows=rows,
            component_audit_inventory_sha256=component_audit,
            audit_receipt_sha256=semantic_sha256(
                {
                    "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                    "component_audit_inventory_sha256": component_audit,
                    "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
                }
            ),
            loaded_source=loaded_source,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 values are malformed"
        ) from exc
    result.validate()
    expected = result.semantic_unsigned() | {
        "rows": tuple(asdict(row) for row in result.rows),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
        "component_audit_inventory_sha256": (result.component_audit_inventory_sha256),
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityEvaluationSourceBundleV3Error(
            "evaluation source bundle V3 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_DATASET",
    "MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SCHEMA",
    "MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V3_SPEC_SHA256",
    "MassiveProfitabilityEvaluationSourceBundleV3",
    "MassiveProfitabilityEvaluationSourceBundleV3Error",
    "MassiveProfitabilityEvaluationSourceDateV3",
    "materialize_massive_profitability_evaluation_source_bundle_v3",
    "parse_massive_profitability_evaluation_source_bundle_v3",
    "reconcile_massive_profitability_evaluation_source_date_v3",
]
