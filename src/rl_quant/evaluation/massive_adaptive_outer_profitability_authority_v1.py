"""Replay authority for one frozen outer adaptive-profit trace."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    MassiveAdaptiveForecastCalibrationV1,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferencePlanV1,
)
from rl_quant.evaluation.massive_adaptive_profit_trace_v1 import (
    MassiveAdaptiveProfitTraceV1,
    build_massive_adaptive_profit_trace_v1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerConfigV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MassiveAdaptiveFillSourceV1,
)
from rl_quant.features.massive_economic_authority_v6 import (
    MassiveProviderEconomicArchiveAuthorityV6,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_authority_v2 import (
    MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
)

MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-profitability-authority-v1"
)
MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "replay": "complete-selection-gated-outer-economic-trace",
        "caller_returns": False,
        "caller_trades": False,
        "outer_conclusion": "downstream-statistical-evidence-only",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveOuterProfitabilityAuthorityV1Error(ValueError):
    """An outer trace cannot be reproduced from the frozen source package."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterProfitabilityAuthorityV1:
    fold_index: int
    trace_receipt_sha256: str
    trace_source_receipt_sha256: str
    source_inventory_sha256: str
    checkpoint_selection_authority_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    outer_forecast_archive_receipt_sha256: str
    outer_inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    compiler_config_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    deterministic_outer_trace_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "deterministic_outer_trace_replayed",
                "outer_evaluation_authorized",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SCHEMA
            or self.fold_index < 0
            or not isinstance(self.source_data_qualified, bool)
            or not self.deterministic_outer_trace_replayed
            or self.outer_evaluation_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterProfitabilityAuthorityV1Error(
                "outer profitability replay authority differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_outer_profitability_authority_v1(
    *,
    trace: MassiveAdaptiveProfitTraceV1,
    checkpoint_selection: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    outer_forecast_archive: MassiveAdaptiveOuterForecastArchiveV1,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    outer_inference_plan: MassiveAdaptiveOuterInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6,
    frozen_decision_trace: MassiveAdaptiveProfitTraceV1 | None = None,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveOuterProfitabilityAuthorityV1:
    """Reexecute one outer cost rung before granting statistical use."""

    trace.validate()
    checkpoint_selection.validate()
    outer_forecast_archive.validate()
    outer_inference_plan.validate()
    if (
        trace.evaluation_role != "outer_test"
        or trace.loaded_source is None
        or not trace.deterministic_profitability_replayed
        or trace.forecast_archive_receipt_sha256
        != outer_forecast_archive.semantic_receipt_sha256
        or trace.inference_plan_receipt_sha256
        != outer_inference_plan.semantic_receipt_sha256
        or outer_forecast_archive.checkpoint_selection_authority_receipt_sha256
        != checkpoint_selection.semantic_receipt_sha256
        or outer_forecast_archive.selected_checkpoint_receipt_sha256
        != checkpoint_selection.selection.selected_checkpoint_receipt_sha256
        or outer_inference_plan.checkpoint_selection_receipt_sha256
        != checkpoint_selection.semantic_receipt_sha256
        or outer_inference_plan.fold_index != outer_forecast_archive.fold_index
    ):
        raise MassiveAdaptiveOuterProfitabilityAuthorityV1Error(
            "outer trace is detached from its frozen selection or plan"
        )
    rebuilt = build_massive_adaptive_profit_trace_v1(
        forecast_archive=outer_forecast_archive,
        calibration=calibration,
        inference_plan=outer_inference_plan,
        decision_roots=decision_roots,
        context_origins=context_origins,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        economic_event_archive=economic_event_archive,
        frozen_decision_trace=frozen_decision_trace,
        initial_capital=trace.initial_capital,
        transaction_cost_basis_points=trace.transaction_cost_basis_points,
        maximum_fill_participation=trace.maximum_fill_participation,
        compiler_config=compiler_config,
    )
    if (
        rebuilt.semantic_unsigned() != trace.semantic_unsigned()
        or rebuilt.semantic_receipt_sha256 != trace.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveOuterProfitabilityAuthorityV1Error(
            "outer profitability trace does not replay from source roots"
        )
    source_qualified = bool(
        checkpoint_selection.development_checkpoint_selection_authorized
        and checkpoint_selection.runtime_selection_replayed
        and outer_forecast_archive.outer_forecast_authorized
        and outer_inference_plan.outer_inference_authorized
        and trace.source_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SCHEMA,
        "fold_index": outer_inference_plan.fold_index,
        "trace_receipt_sha256": trace.semantic_receipt_sha256,
        "trace_source_receipt_sha256": trace.loaded_source.receipt.receipt_sha256,
        "source_inventory_sha256": trace.source_inventory_sha256,
        "checkpoint_selection_authority_receipt_sha256": (
            checkpoint_selection.semantic_receipt_sha256
        ),
        "selected_checkpoint_receipt_sha256": (
            outer_forecast_archive.selected_checkpoint_receipt_sha256
        ),
        "outer_forecast_archive_receipt_sha256": (
            outer_forecast_archive.semantic_receipt_sha256
        ),
        "outer_inference_plan_receipt_sha256": (
            outer_inference_plan.semantic_receipt_sha256
        ),
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "compiler_config_receipt_sha256": trace.compiler_config_receipt_sha256,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_OUTER_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256
        ),
    }
    result = MassiveAdaptiveOuterProfitabilityAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        deterministic_outer_trace_replayed=True,
        outer_evaluation_authorized=source_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveOuterProfitabilityAuthorityV1",
    "MassiveAdaptiveOuterProfitabilityAuthorityV1Error",
    "build_massive_adaptive_outer_profitability_authority_v1",
]
