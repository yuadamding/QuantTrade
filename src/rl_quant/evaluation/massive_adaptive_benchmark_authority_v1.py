"""One benchmark identity shared by compiler constraints and realized P&L."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-benchmark-authority-v1"
)
MASSIVE_ADAPTIVE_BENCHMARK_ENTRY_TURNOVER_V1 = 0.10
MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "identity": "pit-equal-weight-staged-entry-then-buy-and-drift",
        "initial_state": "cash",
        "entry": "same-next-morning-fill-and-cost-rules-as-strategy",
        "entry_one_way_turnover": MASSIVE_ADAPTIVE_BENCHMARK_ENTRY_TURNOVER_V1,
        "compiler_reference": "actual-prefill-economic-benchmark-book",
        "execution_target": "pit-equal-weight-on-first-fill-then-current-book",
        "rebalance_after_entry": False,
        "rl": False,
    }
)


class MassiveAdaptiveBenchmarkAuthorityV1Error(ValueError):
    """Benchmark target and compiler reference do not share one book."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveBenchmarkAuthorityV1:
    decision_session_date: str
    security_ids: tuple[str, ...]
    target_weights: tuple[float, ...]
    compiler_benchmark_weights: tuple[float, ...]
    benchmark_book_receipt_sha256: str
    forecast_row_receipt_sha256: str
    first_entry: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SCHEMA
            or not self.decision_session_date
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.target_weights) != len(self.security_ids)
            or len(self.compiler_benchmark_weights) != len(self.security_ids)
            or any(not math.isfinite(value) or value < 0.0 for value in self.target_weights)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.compiler_benchmark_weights
            )
            or sum(self.target_weights) > 1.0 + 1.0e-10
            or sum(self.compiler_benchmark_weights) > 1.0 + 1.0e-10
            or not isinstance(self.first_entry, bool)
            or not isinstance(self.source_data_qualified, bool)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveBenchmarkAuthorityV1Error(
                "adaptive benchmark authority differs"
            )
        if self.first_entry and not (
            sum(self.compiler_benchmark_weights)
            < sum(self.target_weights)
            <= min(
                1.0,
                sum(self.compiler_benchmark_weights)
                + MASSIVE_ADAPTIVE_BENCHMARK_ENTRY_TURNOVER_V1,
            )
            + 1.0e-10
        ):
            raise MassiveAdaptiveBenchmarkAuthorityV1Error(
                "benchmark funding step exceeds its turnover envelope"
            )
        if not self.first_entry and self.compiler_benchmark_weights != self.target_weights:
            raise MassiveAdaptiveBenchmarkAuthorityV1Error(
                "drifted benchmark target and compiler reference differ"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_benchmark_authority_v1(
    *,
    decision_session_date: str,
    security_ids: Sequence[str],
    forecast_security_ids: Sequence[str],
    forecast_valid: Sequence[bool],
    forecast_row_receipt_sha256: str,
    benchmark_book: MassiveAdaptiveEconomicBookV1,
    source_data_qualified: bool,
) -> MassiveAdaptiveBenchmarkAuthorityV1:
    """Enter PIT equal weight once, then expose the drifted book everywhere."""

    benchmark_book.validate()
    axis = tuple(security_ids)
    if axis != tuple(sorted(set(axis))) or not axis:
        raise MassiveAdaptiveBenchmarkAuthorityV1Error(
            "benchmark security axis is not canonical"
        )
    if benchmark_book.decision_session_date != decision_session_date:
        raise MassiveAdaptiveBenchmarkAuthorityV1Error(
            "benchmark book and decision date differ"
        )
    compiler_weights = benchmark_book.weights(axis)
    risky_weight = sum(compiler_weights)
    first_entry = risky_weight < 1.0 - 1.0e-10
    if first_entry:
        eligible = tuple(
            security_id
            for security_id, valid in zip(
                forecast_security_ids, forecast_valid, strict=True
            )
            if bool(valid)
        )
        if not eligible:
            raise MassiveAdaptiveBenchmarkAuthorityV1Error(
                "initial benchmark has no valid action support"
            )
        eligible_set = set(eligible)
        entry_notional = min(
            MASSIVE_ADAPTIVE_BENCHMARK_ENTRY_TURNOVER_V1,
            1.0 - risky_weight,
        )
        weights = tuple(
            current
            + (entry_notional / len(eligible) if security_id in eligible_set else 0.0)
            for security_id, current in zip(axis, compiler_weights, strict=True)
        )
    else:
        weights = benchmark_book.weights(axis)
    body = {
        "schema": MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "security_ids": axis,
        "target_weights": weights,
        "compiler_benchmark_weights": compiler_weights,
        "benchmark_book_receipt_sha256": benchmark_book.semantic_receipt_sha256,
        "forecast_row_receipt_sha256": forecast_row_receipt_sha256,
        "first_entry": first_entry,
        "source_data_qualified": bool(source_data_qualified),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_BENCHMARK_AUTHORITY_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveBenchmarkAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveBenchmarkAuthorityV1",
    "MassiveAdaptiveBenchmarkAuthorityV1Error",
    "build_massive_adaptive_benchmark_authority_v1",
]
