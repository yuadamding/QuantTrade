"""Checkpoint/fold/benchmark-bound compiler inputs for adaptive profit V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_benchmark_authority_v1 import (
    MassiveAdaptiveBenchmarkAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_compiler_input_authority_v1 import (
    _causal_risk_and_liquidity,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
    MassiveAdaptiveForecastRowV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_runtime_protocol_v1 import (
    MassiveAdaptiveForecastRuntimeProtocol,
    MassiveAdaptiveInferenceRowRuntimeProtocol,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerInputsV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-compiler-input-authority-v2"
)
MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "calibration": "fold-checkpoint-model-state-bound-v2",
        "benchmark": "shared-buy-and-drift-authority-v1",
        "security_axis": "forecast-union-strategy-held-union-benchmark-held",
        "future_fill_data": False,
        "caller_arrays": False,
        "reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveCompilerInputAuthorityV2Error(ValueError):
    """Compiler inputs are detached from forecast/calibration/benchmark roots."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveCompilerInputAuthorityV2:
    decision_session_date: str
    fold_index: int
    checkpoint_receipt_sha256: str
    model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    forecast_row_receipt_sha256: str
    calibration_receipt_sha256: str
    benchmark_authority_receipt_sha256: str
    decision_root_receipt_sha256: str
    inference_row_receipt_sha256: str
    economic_book_receipt_sha256: str
    daily_input_receipt_sha256: str
    identity_authority_receipt_sha256: str
    compiler_input_receipt_sha256: str
    source_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_inputs: MassiveAdaptivePortfolioCompilerInputsV1 | None
    runtime_inputs_replayed: bool
    development_compiler_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        names = (
            "schema",
            "decision_session_date",
            "fold_index",
            "checkpoint_receipt_sha256",
            "model_state_receipt_sha256",
            "training_window_plan_receipt_sha256",
            "forecast_archive_receipt_sha256",
            "forecast_row_receipt_sha256",
            "calibration_receipt_sha256",
            "benchmark_authority_receipt_sha256",
            "decision_root_receipt_sha256",
            "inference_row_receipt_sha256",
            "economic_book_receipt_sha256",
            "daily_input_receipt_sha256",
            "identity_authority_receipt_sha256",
            "compiler_input_receipt_sha256",
            "source_inventory_sha256",
            "source_data_qualified",
            "profitability_reporting_authorized",
            "lockbox_access_authorized",
            "reinforcement_learning_authorized",
            "protocol_receipt_sha256",
            "specification_sha256",
        )
        return {name: getattr(self, name) for name in names}

    def validate(self) -> None:
        runtime_present = self.runtime_inputs is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SCHEMA
            or not self.decision_session_date
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_inputs_replayed != runtime_present
            or self.development_compiler_authorized
            != (runtime_present and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveCompilerInputAuthorityV2Error(
                "adaptive compiler-input authority v2 differs"
            )
        if self.runtime_inputs is not None:
            self.runtime_inputs.validate()
            if self.runtime_inputs.receipt_sha256 != self.compiler_input_receipt_sha256:
                raise MassiveAdaptiveCompilerInputAuthorityV2Error(
                    "runtime compiler inputs differ from the committed receipt"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _forecast_lineage(
    archive: MassiveAdaptiveForecastRuntimeProtocol,
) -> tuple[int, str, str, str, bool]:
    if isinstance(archive, MassiveAdaptiveForecastArchiveV2):
        return (
            archive.fold_index,
            archive.checkpoint_receipt_sha256,
            archive.model_state_receipt_sha256,
            archive.training_window_plan_receipt_sha256,
            archive.development_forecast_authorized,
        )
    if isinstance(archive, MassiveAdaptiveOuterForecastArchiveV1):
        return (
            archive.fold_index,
            archive.selected_checkpoint_receipt_sha256,
            archive.model_state_receipt_sha256,
            archive.training_window_plan_receipt_sha256,
            archive.outer_forecast_authorized,
        )
    checkpoint = getattr(archive, "checkpoint_receipt_sha256", None)
    if checkpoint is None:
        checkpoint = getattr(archive, "selected_checkpoint_receipt_sha256", None)
    if checkpoint is None:
        raise MassiveAdaptiveCompilerInputAuthorityV2Error(
            "compiler v2 requires checkpoint-bound forecast lineage"
        )
    lineage_fold = getattr(archive, "source_fold_index", archive.fold_index)
    return (
        int(lineage_fold),
        str(checkpoint),
        str(getattr(archive, "model_state_receipt_sha256")),
        str(getattr(archive, "training_window_plan_receipt_sha256")),
        bool(
            getattr(archive, "development_forecast_authorized", False)
            or getattr(archive, "outer_forecast_authorized", False)
        ),
    )


def build_massive_adaptive_compiler_input_authority_v2(
    *,
    forecast_archive: MassiveAdaptiveForecastRuntimeProtocol,
    forecast_row: MassiveAdaptiveForecastRowV2,
    calibration: MassiveAdaptiveForecastCalibrationV2,
    benchmark_authority: MassiveAdaptiveBenchmarkAuthorityV1,
    decision_root: MassiveAdaptiveDecisionRootV1,
    context_origin: MassiveAdaptiveContextOriginAuthorityV1,
    inference_row: MassiveAdaptiveInferenceRowRuntimeProtocol,
    book: MassiveAdaptiveEconomicBookV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveAdaptiveCompilerInputAuthorityV2:
    """Derive one compiler input with no free benchmark or calibration arrays."""

    for value in (
        forecast_archive,
        forecast_row,
        calibration,
        benchmark_authority,
        decision_root,
        context_origin,
        book,
        daily_input_authority,
        identity_authority,
    ):
        value.validate()
    inference_row.validate(
        maximum_context_sessions=len(inference_row.context_session_dates)
    )
    fold, checkpoint, model_state, training_window, archive_qualified = (
        _forecast_lineage(forecast_archive)
    )
    if (
        forecast_archive.runtime_rows is None
        or not forecast_archive.runtime_forecasts_replayed
        or forecast_row.receipt_sha256 not in forecast_archive.row_receipts
        or calibration.fold_index != fold
        or calibration.checkpoint_receipt_sha256 != checkpoint
        or calibration.model_state_receipt_sha256 != model_state
        or calibration.training_window_plan_receipt_sha256 != training_window
        or calibration.calibration_fit_stop_session_date
        >= forecast_row.decision_session_date
        or forecast_row.decision_session_date != decision_root.decision_session_date
        or forecast_row.decision_session_date != inference_row.decision_session_date
        or forecast_row.decision_session_date != book.decision_session_date
        or benchmark_authority.decision_session_date
        != forecast_row.decision_session_date
        or benchmark_authority.benchmark_book_receipt_sha256
        == book.semantic_receipt_sha256
        or forecast_row.decision_root_receipt_sha256
        != decision_root.semantic_receipt_sha256
        or forecast_row.inference_row_receipt_sha256 != inference_row.receipt_sha256
        or decision_root.context_origin_receipt_sha256
        != context_origin.semantic_receipt_sha256
        or context_origin.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
        or decision_root.context_security_ids != forecast_row.security_ids
    ):
        raise MassiveAdaptiveCompilerInputAuthorityV2Error(
            "forecast, calibration, fold, benchmark, or decision roots differ"
        )
    axis = benchmark_authority.security_ids
    expected_axis = tuple(
        sorted(set(forecast_row.security_ids) | set(book.shares_by_security()))
    )
    if not set(expected_axis) <= set(axis):
        raise MassiveAdaptiveCompilerInputAuthorityV2Error(
            "benchmark axis omits forecast or strategy-held securities"
        )
    masters = {row.security_id: row for row in identity_authority.security_master}
    if not set(axis) <= set(masters):
        raise MassiveAdaptiveCompilerInputAuthorityV2Error(
            "compiler v2 axis contains an unknown permanent security"
        )
    bucket_count = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    bias = np.asarray(calibration.mean_bias, dtype=np.float64)
    multiplier = np.asarray(calibration.scale_multiplier, dtype=np.float64)
    correlation = np.asarray(calibration.horizon_error_correlation, dtype=np.float64)
    expected: NDArray[np.float64] = np.zeros(
        (len(axis), bucket_count), dtype=np.float64
    )
    scales = np.ones_like(expected)
    valid: NDArray[np.bool_] = np.zeros(len(axis), dtype=np.bool_)
    index_by_security = {security_id: index for index, security_id in enumerate(axis)}
    means = forecast_row.residual_mean.numpy().astype(np.float64) + bias
    calibrated_scales = (
        forecast_row.residual_scale.numpy().astype(np.float64) * multiplier
    )
    for forecast_index, security_id in enumerate(forecast_row.security_ids):
        index = index_by_security[security_id]
        expected[index] = means[forecast_index]
        scales[index] = calibrated_scales[forecast_index]
        valid[index] = bool(forecast_row.valid[forecast_index])
    bucket_covariance = np.asarray(
        [np.diag(row) @ correlation @ np.diag(row) for row in scales],
        dtype=np.float64,
    )
    risk, betas, adv, risk_population = _causal_risk_and_liquidity(
        security_ids=axis,
        decision_session_date=forecast_row.decision_session_date,
        daily=daily_input_authority,
        fallback_variances=np.maximum(np.square(scales[:, 0]), 1.0e-10),
    )
    pretrade = book.weights(axis)
    forced_exit = np.asarray(
        [weight > 0.0 and not valid[index] for index, weight in enumerate(pretrade)],
        dtype=bool,
    )
    cost: NDArray[np.float64] = np.full(len(axis), 20.0, dtype=np.float64)
    risk_receipt = semantic_sha256(
        {
            "daily": daily_input_authority.semantic_receipt_sha256,
            "population": risk_population,
            "method": "causal-63-session-shrunk-close-return-covariance-v1",
        }
    )
    cost_receipt = semantic_sha256(
        {"one_way_basis_points": 20.0, "future_curve": "flat-v1"}
    )
    eligibility_receipt = semantic_sha256(
        {
            "decision_root": decision_root.semantic_receipt_sha256,
            "forecast_valid": forecast_row.array_receipts[-1],
            "future_fill_observed": False,
        }
    )
    inputs = MassiveAdaptivePortfolioCompilerInputsV1(
        decision_id=forecast_row.decision_session_date,
        security_ids=axis,
        issuer_ids=tuple(masters[security_id].issuer_id for security_id in axis),
        bucket_expected_residual_returns=tuple(
            tuple(float(value) for value in row) for row in expected
        ),
        bucket_covariances=tuple(
            tuple(tuple(float(value) for value in inner) for inner in row)
            for row in bucket_covariance
        ),
        pretrade_weights=pretrade,
        benchmark_weights=benchmark_authority.compiler_benchmark_weights,
        risk_covariance=tuple(tuple(float(value) for value in row) for row in risk),
        active_betas=tuple(float(value) for value in betas),
        trailing_adv_notional=tuple(float(value) for value in adv),
        entry_cost_basis_points=tuple(float(value) for value in cost),
        current_exit_cost_basis_points=tuple(float(value) for value in cost),
        expected_future_exit_cost_basis_points=tuple(
            (20.0,) * bucket_count for _ in axis
        ),
        buy_eligible=tuple(bool(value) for value in valid),
        forced_exit=tuple(bool(value) for value in forced_exit),
        capital=book.marked_equity,
        forecast_receipt_sha256=forecast_row.receipt_sha256,
        risk_receipt_sha256=risk_receipt,
        cost_receipt_sha256=cost_receipt,
        portfolio_state_receipt_sha256=book.semantic_receipt_sha256,
        eligibility_receipt_sha256=eligibility_receipt,
    )
    inputs.validate()
    source_inventory = semantic_sha256(
        (
            forecast_archive.semantic_receipt_sha256,
            forecast_row.receipt_sha256,
            calibration.semantic_receipt_sha256,
            benchmark_authority.semantic_receipt_sha256,
            decision_root.semantic_receipt_sha256,
            context_origin.semantic_receipt_sha256,
            inference_row.receipt_sha256,
            book.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
        )
    )
    source_qualified = bool(
        archive_qualified
        and calibration.development_calibration_authorized
        and benchmark_authority.source_data_qualified
        and decision_root.source_data_qualified
        and daily_input_authority.daily_input_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SCHEMA,
        "decision_session_date": forecast_row.decision_session_date,
        "fold_index": fold,
        "checkpoint_receipt_sha256": checkpoint,
        "model_state_receipt_sha256": model_state,
        "training_window_plan_receipt_sha256": training_window,
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "forecast_row_receipt_sha256": forecast_row.receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
        "benchmark_authority_receipt_sha256": benchmark_authority.semantic_receipt_sha256,
        "decision_root_receipt_sha256": decision_root.semantic_receipt_sha256,
        "inference_row_receipt_sha256": inference_row.receipt_sha256,
        "economic_book_receipt_sha256": book.semantic_receipt_sha256,
        "daily_input_receipt_sha256": daily_input_authority.semantic_receipt_sha256,
        "identity_authority_receipt_sha256": identity_authority.receipt_sha256,
        "compiler_input_receipt_sha256": inputs.receipt_sha256,
        "source_inventory_sha256": source_inventory,
        "source_data_qualified": source_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V2_SPEC_SHA256,
    }
    result = MassiveAdaptiveCompilerInputAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_inputs=inputs,
        runtime_inputs_replayed=True,
        development_compiler_authorized=source_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveCompilerInputAuthorityV2",
    "MassiveAdaptiveCompilerInputAuthorityV2Error",
    "build_massive_adaptive_compiler_input_authority_v2",
]
