"""Exact signal, repair, benchmark-mechanics, and residual P&L attribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.protocol.canonical_artifact import semantic_sha256


SIGNAL_ATTRIBUTION_LEDGER_SCHEMA = "rl-quant.signal-attribution-ledger-v1"
SIGNAL_PROMOTION_EVIDENCE_SCHEMA = "rl-quant.signal-promotion-evidence-v1"
_TOLERANCE = 2e-12


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PITAlphaDataError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PITAlphaDataError(f"{name} is outside its finite domain")
    return result


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ActiveReturnAttribution:
    """One-period exact active-return decomposition.

    ``benchmark_cost_advantage`` is the benchmark cost added back when active
    return is policy-minus-benchmark.  It is deliberately not part of signal
    P&L or signal break-even.
    """

    session_index: int
    signal_gross_return: float
    signal_cost: float
    repair_gross_return: float
    repair_cost: float
    benchmark_cost_advantage: float
    other_active_return: float
    signal_created_one_way_turnover: float
    repair_one_way_turnover: float
    benchmark_one_way_turnover: float
    policy_one_way_turnover: float
    reported_active_net_return: float

    def validate(self) -> None:
        if (
            isinstance(self.session_index, bool)
            or not isinstance(self.session_index, int)
            or self.session_index < 0
        ):
            raise PITAlphaDataError("attribution session index must be nonnegative")
        for name in (
            "signal_gross_return",
            "repair_gross_return",
            "other_active_return",
            "reported_active_net_return",
        ):
            _finite(name, getattr(self, name))
        for name in (
            "signal_cost",
            "repair_cost",
            "benchmark_cost_advantage",
            "signal_created_one_way_turnover",
            "repair_one_way_turnover",
            "benchmark_one_way_turnover",
            "policy_one_way_turnover",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if self.signal_cost > 0.0 and self.signal_created_one_way_turnover <= 0.0:
            raise PITAlphaDataError("signal cost exists without signal-created turnover")
        if self.repair_cost > 0.0 and self.repair_one_way_turnover <= 0.0:
            raise PITAlphaDataError("repair cost exists without repair turnover")
        if self.benchmark_cost_advantage > 0.0 and self.benchmark_one_way_turnover <= 0.0:
            raise PITAlphaDataError("benchmark cost advantage exists without benchmark turnover")
        if not math.isclose(
            self.reported_active_net_return,
            self.reconciled_active_net_return,
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ):
            raise PITAlphaDataError("active return does not reconcile to attributed P&L")

    @property
    def signal_net_return(self) -> float:
        return self.signal_gross_return - self.signal_cost

    @property
    def repair_net_return(self) -> float:
        return self.repair_gross_return - self.repair_cost

    @property
    def reconciled_active_net_return(self) -> float:
        return (
            self.signal_net_return
            + self.repair_net_return
            + self.benchmark_cost_advantage
            + self.other_active_return
        )

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "session_index": self.session_index,
            "signal_gross_return": self.signal_gross_return,
            "signal_cost": self.signal_cost,
            "signal_net_return": self.signal_net_return,
            "repair_gross_return": self.repair_gross_return,
            "repair_cost": self.repair_cost,
            "repair_net_return": self.repair_net_return,
            "benchmark_cost_advantage": self.benchmark_cost_advantage,
            "other_active_return": self.other_active_return,
            "signal_created_one_way_turnover": self.signal_created_one_way_turnover,
            "repair_one_way_turnover": self.repair_one_way_turnover,
            "benchmark_one_way_turnover": self.benchmark_one_way_turnover,
            "policy_one_way_turnover": self.policy_one_way_turnover,
            "reported_active_net_return": self.reported_active_net_return,
        }


@dataclass(frozen=True, slots=True)
class SignalAttributionLedger:
    dataset_receipt_sha256: str
    experiment_spec_sha256: str
    periods: tuple[ActiveReturnAttribution, ...]
    receipt_sha256: str
    schema: str = SIGNAL_ATTRIBUTION_LEDGER_SCHEMA

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "experiment_spec_sha256": self.experiment_spec_sha256,
            "periods": tuple(period.payload() for period in self.periods),
        }

    def validate(self) -> None:
        _digest("attribution dataset receipt", self.dataset_receipt_sha256)
        _digest("attribution experiment receipt", self.experiment_spec_sha256)
        _digest("attribution receipt", self.receipt_sha256)
        if self.schema != SIGNAL_ATTRIBUTION_LEDGER_SCHEMA or not self.periods:
            raise PITAlphaDataError("signal attribution ledger schema or inventory drifted")
        sessions: list[int] = []
        for period in self.periods:
            period.validate()
            sessions.append(period.session_index)
        if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
            raise PITAlphaDataError("attribution sessions must be sorted and unique")
        if self.receipt_sha256 != semantic_sha256(self._payload()):
            raise PITAlphaDataError("signal attribution ledger receipt drifted")

    @property
    def signal_gross_return(self) -> float:
        return math.fsum(period.signal_gross_return for period in self.periods)

    @property
    def signal_cost(self) -> float:
        return math.fsum(period.signal_cost for period in self.periods)

    @property
    def signal_net_return(self) -> float:
        return math.fsum(period.signal_net_return for period in self.periods)

    @property
    def repair_net_return(self) -> float:
        return math.fsum(period.repair_net_return for period in self.periods)

    @property
    def benchmark_cost_advantage(self) -> float:
        return math.fsum(period.benchmark_cost_advantage for period in self.periods)

    @property
    def other_active_return(self) -> float:
        return math.fsum(period.other_active_return for period in self.periods)

    @property
    def total_active_net_return(self) -> float:
        return math.fsum(period.reported_active_net_return for period in self.periods)

    @property
    def signal_created_one_way_turnover(self) -> float:
        return math.fsum(period.signal_created_one_way_turnover for period in self.periods)

    @property
    def signal_break_even_one_way_cost_basis_points(self) -> float | None:
        turnover = self.signal_created_one_way_turnover
        if turnover <= 0.0 or self.signal_gross_return <= 0.0:
            return None
        return 10_000.0 * self.signal_gross_return / turnover


def build_signal_attribution_ledger(
    *,
    dataset_receipt_sha256: str,
    experiment_spec_sha256: str,
    periods: Sequence[ActiveReturnAttribution],
) -> SignalAttributionLedger:
    payload = {
        "schema": SIGNAL_ATTRIBUTION_LEDGER_SCHEMA,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "experiment_spec_sha256": experiment_spec_sha256,
        "periods": tuple(period.payload() for period in periods),
    }
    result = SignalAttributionLedger(
        dataset_receipt_sha256=dataset_receipt_sha256,
        experiment_spec_sha256=experiment_spec_sha256,
        periods=tuple(periods),
        receipt_sha256=semantic_sha256(payload),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class SignalPromotionEvidence:
    """Promotion gate whose primary evidence cannot be supplied by repairs."""

    attribution_receipt_sha256: str
    signal_net_return_lcb95: float
    factor_adjusted_signal_alpha_lcb95: float
    signal_break_even_one_way_cost_basis_points: float | None
    estimated_median_one_way_cost_basis_points: float
    minimum_absolute_break_even_basis_points: float = 10.0
    decision: Literal["pass", "fail"] = "fail"
    schema: str = SIGNAL_PROMOTION_EVIDENCE_SCHEMA

    @property
    def required_break_even_basis_points(self) -> float:
        return max(
            self.minimum_absolute_break_even_basis_points,
            2.0 * self.estimated_median_one_way_cost_basis_points,
        )

    @property
    def expected_decision(self) -> Literal["pass", "fail"]:
        break_even = self.signal_break_even_one_way_cost_basis_points
        passed = (
            self.signal_net_return_lcb95 > 0.0
            and self.factor_adjusted_signal_alpha_lcb95 > 0.0
            and break_even is not None
            and break_even >= self.required_break_even_basis_points
        )
        return "pass" if passed else "fail"

    def validate(self) -> None:
        _digest("promotion attribution receipt", self.attribution_receipt_sha256)
        for name in (
            "signal_net_return_lcb95",
            "factor_adjusted_signal_alpha_lcb95",
        ):
            _finite(name, getattr(self, name))
        for name in (
            "estimated_median_one_way_cost_basis_points",
            "minimum_absolute_break_even_basis_points",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if self.minimum_absolute_break_even_basis_points < 10.0:
            raise PITAlphaDataError("absolute signal break-even cannot be lowered below 10 bp")
        if self.signal_break_even_one_way_cost_basis_points is not None:
            _finite(
                "signal break-even",
                self.signal_break_even_one_way_cost_basis_points,
                minimum=0.0,
            )
        if self.schema != SIGNAL_PROMOTION_EVIDENCE_SCHEMA:
            raise PITAlphaDataError("signal promotion schema drifted")
        if self.decision != self.expected_decision:
            raise PITAlphaDataError("signal promotion decision does not follow its frozen gates")


def evaluate_signal_promotion(
    ledger: SignalAttributionLedger,
    *,
    signal_net_return_lcb95: float,
    factor_adjusted_signal_alpha_lcb95: float,
    estimated_median_one_way_cost_basis_points: float,
    minimum_absolute_break_even_basis_points: float = 10.0,
) -> SignalPromotionEvidence:
    ledger.validate()
    provisional = SignalPromotionEvidence(
        attribution_receipt_sha256=ledger.receipt_sha256,
        signal_net_return_lcb95=signal_net_return_lcb95,
        factor_adjusted_signal_alpha_lcb95=factor_adjusted_signal_alpha_lcb95,
        signal_break_even_one_way_cost_basis_points=(
            ledger.signal_break_even_one_way_cost_basis_points
        ),
        estimated_median_one_way_cost_basis_points=(
            estimated_median_one_way_cost_basis_points
        ),
        minimum_absolute_break_even_basis_points=(
            minimum_absolute_break_even_basis_points
        ),
    )
    result = SignalPromotionEvidence(
        attribution_receipt_sha256=provisional.attribution_receipt_sha256,
        signal_net_return_lcb95=provisional.signal_net_return_lcb95,
        factor_adjusted_signal_alpha_lcb95=(
            provisional.factor_adjusted_signal_alpha_lcb95
        ),
        signal_break_even_one_way_cost_basis_points=(
            provisional.signal_break_even_one_way_cost_basis_points
        ),
        estimated_median_one_way_cost_basis_points=(
            provisional.estimated_median_one_way_cost_basis_points
        ),
        minimum_absolute_break_even_basis_points=(
            provisional.minimum_absolute_break_even_basis_points
        ),
        decision=provisional.expected_decision,
    )
    result.validate()
    return result


__all__ = [
    "SIGNAL_ATTRIBUTION_LEDGER_SCHEMA",
    "SIGNAL_PROMOTION_EVIDENCE_SCHEMA",
    "ActiveReturnAttribution",
    "SignalAttributionLedger",
    "SignalPromotionEvidence",
    "build_signal_attribution_ledger",
    "evaluate_signal_promotion",
]
