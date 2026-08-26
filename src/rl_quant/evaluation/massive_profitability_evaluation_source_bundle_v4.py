"""Promotable component-root reconciliation for embargoed outer evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v3 import (
    MassiveProfitabilityEvaluationPlanV3,
)
from rl_quant.evaluation.massive_profitability_evaluation_source_bundle_v3 import (
    _reconcile_sources,
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

MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA = (
    "rl-quant.massive-profitability-evaluation-source-bundle-v4"
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_DATASET = (
    "massive-profitability-evaluation-source-bundle-v4"
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "phase": "phase-plan-v2",
        "evaluation": "evaluation-plan-v3",
        "roots": "package-reconciled-no-independent-pnl-inputs",
        "coverage": "frozen-full-semantic-and-audit-plus-scoped-feature-and-target",
        "generic_reload": "nonauthorizing",
        "lockbox": "excluded",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityEvaluationSourceBundleV4Error(ValueError):
    """The V4 source bundle differs from its typed root graph."""


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationRuntimeSourcesV4:
    data_gate: MassiveProfitabilityDataGateV2
    phase_plan: MassiveProfitabilityPhasePlanV2
    evaluation_plan: MassiveProfitabilityEvaluationPlanV3
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1
    fill_source_authority: MassiveProfitabilityFillSourceAuthorityV2
    terminal_authority: MassiveProfitabilityTerminalCoverageAuthorityV1
    features: tuple[MassiveProfitabilityOriginFeaturesV3, ...]
    targets: tuple[MassiveProfitabilityTargetsV2, ...]
    feature_accounting: tuple[MassiveProfitabilityFeatureAccountingAuthorityV2, ...]
    target_accounting: tuple[MassiveProfitabilityTargetAccountingAuthorityV2, ...]


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationSourceDateV4:
    decision_session_date: str
    origin_receipt_sha256: str
    feature_semantic_receipt_sha256: str
    target_semantic_receipt_sha256: str
    feature_accounting_semantic_receipt_sha256: str
    target_accounting_semantic_receipt_sha256: str
    frozen_economic_coverage_semantic_receipt_sha256: str
    feature_scoped_economic_coverage_semantic_receipt_sha256: str
    target_scoped_economic_coverage_semantic_receipt_sha256: str
    economic_coverage_audit_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not self.decision_session_date
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV4Error(
                "source date V4 differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityEvaluationSourceBundleV4:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    evaluation_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    daily_input_authority_semantic_receipt_sha256: str
    fill_source_authority_semantic_receipt_sha256: str
    terminal_authority_semantic_receipt_sha256: str
    terminal_accounting_mode: str
    outer_test_session_dates: tuple[str, ...]
    rows: tuple[MassiveProfitabilityEvaluationSourceDateV4, ...]
    row_inventory_sha256: str
    committed_source_bundle_data_qualified: bool
    committed_outer_evaluation_authorized: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    source_bundle_data_qualified: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "loaded_source",
                "source_bundle_data_qualified",
                "outer_evaluation_authorized",
            }
        }

    def validate(self) -> None:
        dates = tuple(row.decision_session_date for row in self.rows)
        if (
            self.schema != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA
            or dates != self.outer_test_session_dates
            or dates != tuple(sorted(set(dates)))
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not self.committed_source_bundle_data_qualified
            or not self.committed_outer_evaluation_authorized
            or self.outer_evaluation_authorized and not self.source_bundle_data_qualified
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV4Error(
                "source bundle V4 identity or authorization differs"
            )
        for row in self.rows:
            row.validate()
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evaluation_plan_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityEvaluationSourceBundleV4Error(
                "source bundle V4 committed source differs"
            )


def _reconciled_rows(
    runtime: MassiveProfitabilityEvaluationRuntimeSourcesV4,
) -> tuple[MassiveProfitabilityEvaluationSourceDateV4, ...]:
    base = _reconcile_sources(
        data_gate=runtime.data_gate,
        phase_plan=runtime.phase_plan,
        evaluation_plan=runtime.evaluation_plan,  # type: ignore[arg-type]
        accounting_freeze=runtime.accounting_freeze,
        origin_plan=runtime.origin_plan,
        daily_input_authority=runtime.daily_input_authority,
        fill_source_authority=runtime.fill_source_authority,
        terminal_authority=runtime.terminal_authority,
        features=runtime.features,
        targets=runtime.targets,
        feature_accounting=runtime.feature_accounting,
        target_accounting=runtime.target_accounting,
    )
    feature_by_origin = {
        row.origin_receipt_sha256: row for row in runtime.feature_accounting
    }
    target_by_date = {
        row.decision_session_date: row for row in runtime.target_accounting
    }
    frozen_by_origin = {
        row.origin_receipt_sha256: row for row in runtime.accounting_freeze.coverage_rows
    }
    outer_origins = {row.origin_receipt_sha256 for row in base}
    if set(feature_by_origin) != outer_origins or set(target_by_date) != {
        row.decision_session_date for row in base
    }:
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 accounting inventory contains nonouter rows"
        )
    result = []
    for row in base:
        feature = feature_by_origin[row.origin_receipt_sha256]
        target = target_by_date[row.decision_session_date]
        frozen = frozen_by_origin[row.origin_receipt_sha256]
        body = {
            "decision_session_date": row.decision_session_date,
            "origin_receipt_sha256": row.origin_receipt_sha256,
            "feature_semantic_receipt_sha256": row.feature_semantic_receipt_sha256,
            "target_semantic_receipt_sha256": row.target_semantic_receipt_sha256,
            "feature_accounting_semantic_receipt_sha256": row.feature_accounting_semantic_receipt_sha256,
            "target_accounting_semantic_receipt_sha256": row.target_accounting_semantic_receipt_sha256,
            "frozen_economic_coverage_semantic_receipt_sha256": frozen.economic_coverage_semantic_receipt_sha256,
            "feature_scoped_economic_coverage_semantic_receipt_sha256": feature.economic_coverage_semantic_receipt_sha256,
            "target_scoped_economic_coverage_semantic_receipt_sha256": target.economic_coverage_semantic_receipt_sha256,
            "economic_coverage_audit_receipt_sha256": row.economic_coverage_audit_receipt_sha256,
        }
        item = MassiveProfitabilityEvaluationSourceDateV4(
            **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
        )
        item.validate()
        result.append(item)
    return tuple(result)


def materialize_massive_profitability_evaluation_source_bundle_v4(
    *,
    root: str | Path,
    artifact_id: str,
    runtime_sources: MassiveProfitabilityEvaluationRuntimeSourcesV4,
    committed_at_ms: int,
) -> MassiveProfitabilityEvaluationSourceBundleV4:
    rows = _reconciled_rows(runtime_sources)
    roots = runtime_sources
    qualified = (
        roots.data_gate.data_gate_passed
        and roots.evaluation_plan.outer_evaluation_authorized
        and roots.accounting_freeze.accounting_sources_frozen
        and roots.daily_input_authority.daily_input_data_qualified
        and roots.fill_source_authority.fill_source_data_qualified
        and roots.terminal_authority.terminal_accounting_data_qualified
    )
    if not qualified:
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 roots are not qualified"
        )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SCHEMA,
        "data_gate_semantic_receipt_sha256": roots.data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": roots.phase_plan.semantic_receipt_sha256,
        "evaluation_plan_semantic_receipt_sha256": roots.evaluation_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": roots.evaluation_plan.tournament_plan_receipt_sha256,
        "accounting_freeze_semantic_receipt_sha256": roots.accounting_freeze.semantic_receipt_sha256,
        "origin_plan_semantic_receipt_sha256": roots.origin_plan.semantic_receipt_sha256,
        "daily_input_authority_semantic_receipt_sha256": roots.daily_input_authority.semantic_receipt_sha256,
        "fill_source_authority_semantic_receipt_sha256": roots.fill_source_authority.semantic_receipt_sha256,
        "terminal_authority_semantic_receipt_sha256": roots.terminal_authority.semantic_receipt_sha256,
        "terminal_accounting_mode": roots.terminal_authority.terminal_accounting_mode,
        "outer_test_session_dates": tuple(row.decision_session_date for row in rows),
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        "committed_source_bundle_data_qualified": True,
        "committed_outer_evaluation_authorized": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(not (value.isalnum() or value in "-_") for value in artifact_id):
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 artifact ID is not path safe"
        )
    relative = f"massive-profitability/evaluation-source-bundle-v4/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_EVALUATION_SOURCE_BUNDLE_V4_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=roots.evaluation_plan.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-EVALUATION-SOURCE-V4-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_evaluation_source_bundle_v4(
        root=root, loaded_source=loaded
    )
    return authorize_massive_profitability_evaluation_source_bundle_v4(
        root=root, source_bundle=parsed, runtime_sources=runtime_sources
    )


def parse_massive_profitability_evaluation_source_bundle_v4(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityEvaluationSourceBundleV4:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 is not canonical JSON"
        )
    rows = tuple(
        MassiveProfitabilityEvaluationSourceDateV4(**row)
        for row in payload.pop("rows")
    )
    payload["outer_test_session_dates"] = tuple(payload["outer_test_session_dates"])
    result = MassiveProfitabilityEvaluationSourceBundleV4(
        **payload,
        rows=rows,
        loaded_source=loaded_source,
        source_bundle_data_qualified=False,
        outer_evaluation_authorized=False,
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "rows": tuple(asdict(row) for row in result.rows),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 canonical bytes differ"
        )
    return result


def authorize_massive_profitability_evaluation_source_bundle_v4(
    *,
    root: str | Path,
    source_bundle: MassiveProfitabilityEvaluationSourceBundleV4,
    runtime_sources: MassiveProfitabilityEvaluationRuntimeSourcesV4,
) -> MassiveProfitabilityEvaluationSourceBundleV4:
    parsed = parse_massive_profitability_evaluation_source_bundle_v4(
        root=root, loaded_source=source_bundle.loaded_source
    )
    rows = _reconciled_rows(runtime_sources)
    expected_roots = (
        runtime_sources.data_gate.semantic_receipt_sha256,
        runtime_sources.phase_plan.semantic_receipt_sha256,
        runtime_sources.evaluation_plan.semantic_receipt_sha256,
        runtime_sources.evaluation_plan.tournament_plan_receipt_sha256,
        runtime_sources.accounting_freeze.semantic_receipt_sha256,
        runtime_sources.origin_plan.semantic_receipt_sha256,
        runtime_sources.daily_input_authority.semantic_receipt_sha256,
        runtime_sources.fill_source_authority.semantic_receipt_sha256,
        runtime_sources.terminal_authority.semantic_receipt_sha256,
    )
    actual_roots = (
        parsed.data_gate_semantic_receipt_sha256,
        parsed.phase_plan_semantic_receipt_sha256,
        parsed.evaluation_plan_semantic_receipt_sha256,
        parsed.tournament_plan_receipt_sha256,
        parsed.accounting_freeze_semantic_receipt_sha256,
        parsed.origin_plan_semantic_receipt_sha256,
        parsed.daily_input_authority_semantic_receipt_sha256,
        parsed.fill_source_authority_semantic_receipt_sha256,
        parsed.terminal_authority_semantic_receipt_sha256,
    )
    if (
        parsed.semantic_receipt_sha256 != source_bundle.semantic_receipt_sha256
        or actual_roots != expected_roots
        or tuple(row.receipt_sha256 for row in parsed.rows)
        != tuple(row.receipt_sha256 for row in rows)
    ):
        raise MassiveProfitabilityEvaluationSourceBundleV4Error(
            "source bundle V4 does not reconcile to its runtime roots"
        )
    result = replace(
        parsed,
        source_bundle_data_qualified=True,
        outer_evaluation_authorized=True,
    )
    result.validate()
    return result


__all__ = [
    "MassiveProfitabilityEvaluationRuntimeSourcesV4",
    "MassiveProfitabilityEvaluationSourceBundleV4",
    "MassiveProfitabilityEvaluationSourceBundleV4Error",
    "MassiveProfitabilityEvaluationSourceDateV4",
    "authorize_massive_profitability_evaluation_source_bundle_v4",
    "materialize_massive_profitability_evaluation_source_bundle_v4",
    "parse_massive_profitability_evaluation_source_bundle_v4",
]
