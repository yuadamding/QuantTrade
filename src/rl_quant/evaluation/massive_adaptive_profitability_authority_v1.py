"""Fail-closed replay authority for deterministic adaptive profit traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    MassiveAdaptiveForecastCalibrationV1,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferencePlanV1,
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
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-profitability-authority-v1"
)
MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "replay": "rebuild-complete-inner-validation-economic-trace",
        "caller_returns": False,
        "caller_trades": False,
        "development_only": True,
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveProfitabilityAuthorityV1Error(ValueError):
    """A deterministic trace cannot be reproduced from the exact roots."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitabilityAuthorityV1:
    trace_receipt_sha256: str
    source_inventory_sha256: str
    forecast_archive_receipt_sha256: str
    calibration_receipt_sha256: str
    inference_plan_receipt_sha256: str
    compiler_config_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    deterministic_trace_replayed: bool
    development_profitability_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "deterministic_trace_replayed",
                "development_profitability_authorized",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SCHEMA
            or not isinstance(self.source_data_qualified, bool)
            or not self.deterministic_trace_replayed
            or self.development_profitability_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitabilityAuthorityV1Error(
                "adaptive profitability authority differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_profitability_authority_v1(
    *,
    trace: MassiveAdaptiveProfitTraceV1,
    forecast_archive: MassiveAdaptiveForecastArchiveV2,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    inference_plan: MassiveAdaptiveInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    fill_source: MassiveAdaptiveFillSourceV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
    compiler_config: MassiveAdaptivePortfolioCompilerConfigV1 | None = None,
) -> MassiveAdaptiveProfitabilityAuthorityV1:
    """Reexecute the complete transition kernel before granting development use."""

    trace.validate()
    rebuilt = build_massive_adaptive_profit_trace_v1(
        forecast_archive=forecast_archive,
        calibration=calibration,
        inference_plan=inference_plan,
        decision_roots=decision_roots,
        context_origins=context_origins,
        fill_source=fill_source,
        daily_input_authority=daily_input_authority,
        identity_authority=identity_authority,
        initial_capital=trace.initial_capital,
        transaction_cost_basis_points=trace.transaction_cost_basis_points,
        maximum_fill_participation=trace.maximum_fill_participation,
        compiler_config=compiler_config,
    )
    if (
        rebuilt.semantic_unsigned() != trace.semantic_unsigned()
        or rebuilt.semantic_receipt_sha256 != trace.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveProfitabilityAuthorityV1Error(
            "adaptive profitability trace does not replay from its roots"
        )
    source_qualified = trace.source_data_qualified
    body = {
        "schema": MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SCHEMA,
        "trace_receipt_sha256": trace.semantic_receipt_sha256,
        "source_inventory_sha256": trace.source_inventory_sha256,
        "forecast_archive_receipt_sha256": trace.forecast_archive_receipt_sha256,
        "calibration_receipt_sha256": trace.calibration_receipt_sha256,
        "inference_plan_receipt_sha256": trace.inference_plan_receipt_sha256,
        "compiler_config_receipt_sha256": trace.compiler_config_receipt_sha256,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_PROFITABILITY_AUTHORITY_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveProfitabilityAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        deterministic_trace_replayed=True,
        development_profitability_authorized=source_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveProfitabilityAuthorityV1",
    "MassiveAdaptiveProfitabilityAuthorityV1Error",
    "build_massive_adaptive_profitability_authority_v1",
]
