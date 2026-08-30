"""Cash-funded initialization authority for adaptive profitability traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from rl_quant.execution.massive_adaptive_economic_book_v1 import (
    MassiveAdaptiveEconomicBookV1,
    initial_massive_adaptive_economic_book_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-initial-book-authority-v1"
)
MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "strategy_initialization": "all-cash-at-first-decision-close",
        "neutral_initialization": "same-all-cash-book-as-strategy",
        "benchmark_initialization": "all-cash-at-first-decision-close",
        "first_positions": "next-session-source-fill-only",
        "free_initial_position": False,
        "rl": False,
    }
)


class MassiveAdaptiveInitialBookAuthorityV1Error(ValueError):
    """Initial books are not the declared all-cash economic state."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveInitialBookAuthorityV1:
    decision_session_date: str
    initial_capital: float
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    strategy_book: MassiveAdaptiveEconomicBookV1
    neutral_book: MassiveAdaptiveEconomicBookV1
    benchmark_book: MassiveAdaptiveEconomicBookV1
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        self.strategy_book.validate()
        self.neutral_book.validate()
        self.benchmark_book.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SCHEMA
            or not self.decision_session_date
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0.0
            or self.strategy_book.decision_session_date != self.decision_session_date
            or self.neutral_book.decision_session_date != self.decision_session_date
            or self.benchmark_book.decision_session_date != self.decision_session_date
            or self.strategy_book.holdings
            or self.neutral_book.holdings
            or self.benchmark_book.holdings
            or self.strategy_book.cash != self.initial_capital
            or self.neutral_book.cash != self.initial_capital
            or self.benchmark_book.cash != self.initial_capital
            or self.strategy_book.marked_equity != self.initial_capital
            or self.neutral_book.marked_equity != self.initial_capital
            or self.strategy_book != self.neutral_book
            or self.benchmark_book.marked_equity != self.initial_capital
            or not isinstance(self.source_data_qualified, bool)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveInitialBookAuthorityV1Error(
                "adaptive initial-book authority differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_initial_book_authority_v1(
    *,
    decision_session_date: str,
    initial_capital: float,
    forecast_archive_receipt_sha256: str,
    inference_plan_receipt_sha256: str,
    source_data_qualified: bool,
) -> MassiveAdaptiveInitialBookAuthorityV1:
    """Create two independently receipted all-cash books at the first close."""

    root = semantic_sha256(
        {
            "decision_session_date": decision_session_date,
            "initial_capital": initial_capital,
            "forecast_archive": forecast_archive_receipt_sha256,
            "inference_plan": inference_plan_receipt_sha256,
            "policy": "all-cash-v1",
        }
    )
    strategy = initial_massive_adaptive_economic_book_v1(
        decision_session_date=decision_session_date,
        capital=initial_capital,
        initialization_receipt_sha256=semantic_sha256((root, "strategy")),
    )
    benchmark = initial_massive_adaptive_economic_book_v1(
        decision_session_date=decision_session_date,
        capital=initial_capital,
        initialization_receipt_sha256=semantic_sha256((root, "benchmark")),
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SCHEMA,
        "decision_session_date": decision_session_date,
        "initial_capital": initial_capital,
        "forecast_archive_receipt_sha256": forecast_archive_receipt_sha256,
        "inference_plan_receipt_sha256": inference_plan_receipt_sha256,
        "strategy_book": strategy,
        "neutral_book": strategy,
        "benchmark_book": benchmark,
        "source_data_qualified": bool(source_data_qualified),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_INITIAL_BOOK_AUTHORITY_V1_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveInitialBookAuthorityV1(
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
    "MassiveAdaptiveInitialBookAuthorityV1",
    "MassiveAdaptiveInitialBookAuthorityV1Error",
    "build_massive_adaptive_initial_book_authority_v1",
]
