"""Package-derived bridge from calibrated forecasts to the adaptive compiler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
    MassiveAdaptiveForecastRowV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v1 import (
    MassiveAdaptiveForecastCalibrationV1,
)
from rl_quant.evaluation.massive_adaptive_inference_plan_v1 import (
    MassiveAdaptiveInferenceRowV1,
)
from rl_quant.evaluation.massive_adaptive_outer_forecast_archive_v1 import (
    MassiveAdaptiveOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_outer_inference_plan_v1 import (
    MassiveAdaptiveOuterInferenceRowV1,
)
from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.execution.massive_adaptive_portfolio_compiler_v1 import (
    MassiveAdaptivePortfolioCompilerInputsV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-compiler-input-authority-v1"
)
MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "forecast": (
            "replayed-inner-validation-v2-or-selection-gated-outer-v1"
            "-plus-training-calibration-v1"
        ),
        "book": "continuous-cash-share-book-v1",
        "risk": "causal-63-session-shrunk-close-return-covariance",
        "liquidity": "causal-63-session-mean-dollar-volume",
        "benchmark": "equal-weight-current-action-support",
        "cost": "frozen-20-basis-point-one-way",
        "future_fill_data": False,
        "caller_arrays": False,
        "reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveCompilerInputAuthorityV1Error(ValueError):
    """Compiler arrays differ from their forecast, book, and source roots."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveCompilerInputAuthorityV1:
    decision_session_date: str
    forecast_archive_receipt_sha256: str
    forecast_row_receipt_sha256: str
    calibration_receipt_sha256: str
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
    specification_sha256: str = MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        names = (
            "schema",
            "decision_session_date",
            "forecast_archive_receipt_sha256",
            "forecast_row_receipt_sha256",
            "calibration_receipt_sha256",
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
            self.schema != MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SCHEMA
            or not self.decision_session_date
            or not isinstance(self.source_data_qualified, bool)
            or self.runtime_inputs_replayed != runtime_present
            or self.development_compiler_authorized
            != (runtime_present and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveCompilerInputAuthorityV1Error(
                "adaptive compiler-input authority differs"
            )
        for name in (
            "forecast_archive_receipt_sha256",
            "forecast_row_receipt_sha256",
            "calibration_receipt_sha256",
            "decision_root_receipt_sha256",
            "inference_row_receipt_sha256",
            "economic_book_receipt_sha256",
            "daily_input_receipt_sha256",
            "identity_authority_receipt_sha256",
            "compiler_input_receipt_sha256",
            "source_inventory_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.runtime_inputs is not None:
            self.runtime_inputs.validate()
            if self.runtime_inputs.receipt_sha256 != self.compiler_input_receipt_sha256:
                raise MassiveAdaptiveCompilerInputAuthorityV1Error(
                    "runtime compiler inputs differ from committed receipt"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _causal_risk_and_liquidity(
    *,
    security_ids: tuple[str, ...],
    decision_session_date: str,
    daily: MassiveProfitabilityDailyInputAuthorityV1,
    fallback_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    dates = tuple(
        row.source_session_date
        for row in daily.sessions
        if row.source_session_date <= decision_session_date
    )[-63:]
    if not dates or dates[-1] != decision_session_date:
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            "daily input chronology does not reach the decision close"
        )
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    closes = np.full((len(dates), len(security_ids)), np.nan, dtype=np.float64)
    dollars = np.full_like(closes, np.nan)
    receipts: list[str] = []
    for time_index, session_date in enumerate(dates):
        for asset_index, security_id in enumerate(security_ids):
            row = daily.row(session_date=session_date, security_id=security_id)
            receipts.append(row.receipt_sha256)
            if row.bars_valid[close_index] and row.bars_values[close_index] > 0.0:
                closes[time_index, asset_index] = row.bars_values[close_index]
            if row.bars_valid[dollar_index] and row.bars_values[dollar_index] > 0.0:
                dollars[time_index, asset_index] = row.bars_values[dollar_index]
    returns = closes[1:] / closes[:-1] - 1.0
    valid_return = np.isfinite(returns)
    filled = np.where(valid_return, returns, 0.0)
    support = valid_return.astype(np.float64)
    counts = support.T @ support
    demeaned = filled.copy()
    for asset in range(len(security_ids)):
        observed = returns[:, asset][valid_return[:, asset]]
        mean = float(observed.mean()) if observed.size else 0.0
        demeaned[:, asset] = np.where(
            valid_return[:, asset], returns[:, asset] - mean, 0.0
        )
    covariance = demeaned.T @ demeaned / np.maximum(counts - 1, 1)
    diagonal = np.diag(covariance).copy()
    diagonal = np.where(diagonal > 1.0e-10, diagonal, fallback_variances)
    covariance = 0.25 * covariance + 0.75 * np.diag(diagonal)
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(values, 1.0e-10)) @ vectors.T
    market = np.full(len(security_ids), 1.0 / len(security_ids))
    market_variance = float(market @ covariance @ market)
    betas = covariance @ market / max(market_variance, 1.0e-10)
    adv = np.nanmean(dollars, axis=0)
    if np.isnan(adv).any() or (adv <= 0.0).any():
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            "every compiler security requires causal positive dollar volume"
        )
    return covariance, betas, adv, semantic_sha256(tuple(receipts))


def build_massive_adaptive_compiler_input_authority_v1(
    *,
    forecast_archive: (
        MassiveAdaptiveForecastArchiveV2 | MassiveAdaptiveOuterForecastArchiveV1
    ),
    forecast_row: MassiveAdaptiveForecastRowV2,
    calibration: MassiveAdaptiveForecastCalibrationV1,
    decision_root: MassiveAdaptiveDecisionRootV1,
    context_origin: MassiveAdaptiveContextOriginAuthorityV1,
    inference_row: MassiveAdaptiveInferenceRowV1 | MassiveAdaptiveOuterInferenceRowV1,
    book: MassiveAdaptiveEconomicBookV1,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveAdaptiveCompilerInputAuthorityV1:
    """Materialize every compiler array from package-owned causal roots."""

    forecast_archive.validate()
    forecast_row.validate()
    calibration.validate()
    decision_root.validate()
    context_origin.validate()
    book.validate()
    daily_input_authority.validate()
    identity_authority.validate()
    inference_row.validate(
        maximum_context_sessions=len(inference_row.context_session_dates)
    )
    if (
        forecast_archive.runtime_rows is None
        or not forecast_archive.runtime_forecasts_replayed
        or forecast_row.receipt_sha256 not in forecast_archive.row_receipts
        or forecast_row.decision_session_date != decision_root.decision_session_date
        or forecast_row.decision_session_date != inference_row.decision_session_date
        or forecast_row.decision_session_date != book.decision_session_date
        or forecast_row.decision_root_receipt_sha256
        != decision_root.semantic_receipt_sha256
        or forecast_row.inference_row_receipt_sha256 != inference_row.receipt_sha256
        or decision_root.context_origin_receipt_sha256
        != context_origin.semantic_receipt_sha256
        or context_origin.identity_authority_receipt_sha256
        != identity_authority.receipt_sha256
    ):
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            "forecast, decision, inference, or book roots differ"
        )
    if decision_root.context_security_ids != forecast_row.security_ids:
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            "forecast security axis differs from the source-owned context"
        )
    security_ids = tuple(
        sorted(set(forecast_row.security_ids) | set(book.shares_by_security()))
    )
    master_by_security = {
        row.security_id: row for row in identity_authority.security_master
    }
    if not set(security_ids) <= set(master_by_security):
        raise MassiveAdaptiveCompilerInputAuthorityV1Error(
            "forecast contains an unknown permanent security"
        )
    bias = np.asarray(calibration.mean_bias, dtype=np.float64)
    multiplier = np.asarray(calibration.scale_multiplier, dtype=np.float64)
    correlation = np.asarray(calibration.horizon_error_correlation, dtype=np.float64)
    bucket_count = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
    expected = np.zeros((len(security_ids), bucket_count), dtype=np.float64)
    scales = np.ones_like(expected)
    valid = np.zeros(len(security_ids), dtype=bool)
    union_index = {security_id: index for index, security_id in enumerate(security_ids)}
    forecast_expected = forecast_row.residual_mean.numpy().astype(np.float64) + bias
    forecast_scales = (
        forecast_row.residual_scale.numpy().astype(np.float64) * multiplier
    )
    for forecast_index, security_id in enumerate(forecast_row.security_ids):
        index = union_index[security_id]
        expected[index] = forecast_expected[forecast_index]
        scales[index] = forecast_scales[forecast_index]
        valid[index] = bool(forecast_row.valid[forecast_index])
    bucket_covariance = np.asarray(
        [np.diag(row) @ correlation @ np.diag(row) for row in scales],
        dtype=np.float64,
    )
    fallback_variance = np.maximum(np.square(scales[:, 0]), 1.0e-10)
    risk, betas, adv, risk_population = _causal_risk_and_liquidity(
        security_ids=security_ids,
        decision_session_date=forecast_row.decision_session_date,
        daily=daily_input_authority,
        fallback_variances=fallback_variance,
    )
    benchmark = np.zeros(len(valid), dtype=np.float64)
    if bool(valid.any()):
        benchmark[valid] = 1.0 / float(valid.sum())
    forced_exit = np.asarray(
        [
            weight > 0.0 and not valid[index]
            for index, weight in enumerate(book.weights(security_ids))
        ],
        dtype=bool,
    )
    cost = np.full(len(valid), 20.0, dtype=np.float64)
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
        security_ids=security_ids,
        issuer_ids=tuple(
            master_by_security[security_id].issuer_id for security_id in security_ids
        ),
        bucket_expected_residual_returns=tuple(
            tuple(float(value) for value in row) for row in expected
        ),
        bucket_covariances=tuple(
            tuple(tuple(float(value) for value in inner) for inner in row)
            for row in bucket_covariance
        ),
        pretrade_weights=book.weights(security_ids),
        benchmark_weights=tuple(float(value) for value in benchmark),
        risk_covariance=tuple(tuple(float(value) for value in row) for row in risk),
        active_betas=tuple(float(value) for value in betas),
        trailing_adv_notional=tuple(float(value) for value in adv),
        entry_cost_basis_points=tuple(float(value) for value in cost),
        current_exit_cost_basis_points=tuple(float(value) for value in cost),
        expected_future_exit_cost_basis_points=tuple(
            (20.0,) * len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS) for _ in valid
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
            decision_root.semantic_receipt_sha256,
            context_origin.semantic_receipt_sha256,
            inference_row.receipt_sha256,
            book.semantic_receipt_sha256,
            daily_input_authority.semantic_receipt_sha256,
            identity_authority.receipt_sha256,
        )
    )
    archive_qualified = bool(
        isinstance(forecast_archive, MassiveAdaptiveForecastArchiveV2)
        and forecast_archive.development_forecast_authorized
        or isinstance(forecast_archive, MassiveAdaptiveOuterForecastArchiveV1)
        and forecast_archive.outer_forecast_authorized
    )
    source_qualified = bool(
        archive_qualified
        and isinstance(calibration, MassiveAdaptiveForecastCalibrationV1)
        and isinstance(decision_root, MassiveAdaptiveDecisionRootV1)
        and isinstance(context_origin, MassiveAdaptiveContextOriginAuthorityV1)
        and isinstance(
            inference_row,
            (MassiveAdaptiveInferenceRowV1, MassiveAdaptiveOuterInferenceRowV1),
        )
        and isinstance(book, MassiveAdaptiveEconomicBookV1)
        and isinstance(
            daily_input_authority, MassiveProfitabilityDailyInputAuthorityV1
        )
        and isinstance(identity_authority, PITSecurityUniverseAuthority)
        and calibration.development_calibration_authorized
        and decision_root.source_data_qualified
        and daily_input_authority.daily_input_data_qualified
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SCHEMA,
        "decision_session_date": forecast_row.decision_session_date,
        "forecast_archive_receipt_sha256": forecast_archive.semantic_receipt_sha256,
        "forecast_row_receipt_sha256": forecast_row.receipt_sha256,
        "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
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
        "specification_sha256": MASSIVE_ADAPTIVE_COMPILER_INPUT_AUTHORITY_V1_SPEC_SHA256,
    }
    result = MassiveAdaptiveCompilerInputAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_inputs=inputs,
        runtime_inputs_replayed=True,
        development_compiler_authorized=source_qualified,
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveCompilerInputAuthorityV1",
    "MassiveAdaptiveCompilerInputAuthorityV1Error",
    "build_massive_adaptive_compiler_input_authority_v1",
]
