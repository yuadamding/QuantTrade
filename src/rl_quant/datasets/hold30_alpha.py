"""Point-in-time alpha data boundary for the Hold-30 V3 screen.

The actor is trained relative to the frozen C1 investable benchmark already
bound by :class:`~rl_quant.datasets.hold30.Hold30DatasetSequence`.  External
return series have narrower, explicit roles: PIT CASH/risk-free is available
to accounting, the two total-Sharpe ablations, checkpoint ranking, and
evaluation; PIT cap-weight market is available to beta constraints; and
declared factors are evaluator-only.  None of them is a policy feature.

Auxiliary residual-alpha labels are built only for role-exact score origins.
A decision at position ``t`` fills at ``t + 1``; consequently its first
position return is outbound row ``t + 1``.  A forced exit earns the inbound
stock return and then follows the explicit CASH total-return series for the
rest of the horizon.  Labels never cross a declared split boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rl_quant.datasets.hold30 import Hold30DatasetSequence
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_C1_BENCHMARK_ID,
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
)

HOLD30_ALPHA_LABEL_RULE = (
    "decision-t-fills-t-plus-1;returns-t-plus-1-through-t-plus-H;"
    "forced-exit-then-explicit-cash;subtract-identical-window-C1-net-log;"
    "right-censor-at-split"
)
HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA = (
    "rl-quant.hold30-alpha-evaluator-data-v3"
)
HOLD30_ALPHA_LABEL_SCHEMA = "rl-quant.hold30-alpha-residual-labels-v3"
HOLD30_ALPHA_RISK_FREE_USAGE = (
    "portfolio-accounting",
    "a06-a07-total-sharpe-objective",
    "checkpoint-ranking",
    "evaluation",
)
HOLD30_ALPHA_MARKET_USAGE = (
    "beta-objective",
    "checkpoint-eligibility",
    "evaluation",
)
HOLD30_ALPHA_FACTOR_USAGE = ("evaluation-only",)

_DIGEST_CHARS = frozenset("0123456789abcdef")
_FACTOR_CONVENTIONS = frozenset(
    {"zero-investment", "excess-over-risk-free", "total-return"}
)


class Hold30AlphaDataError(ValueError):
    """A V3 data role, chronology, or content binding is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise Hold30AlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_id(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Hold30AlphaDataError(f"{name} must be a non-empty stable identifier")
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_cpu_float64(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise Hold30AlphaDataError(f"{name} must be a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise Hold30AlphaDataError(
            f"{name} must have shape {shape}; got {tuple(value.shape)}"
        )
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise Hold30AlphaDataError(f"{name} must be a CPU float64 tensor")
    if value.requires_grad or not bool(torch.isfinite(value).all()):
        raise Hold30AlphaDataError(f"{name} must be finite and detached")


def _require_cpu_bool(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.device.type != "cpu"
        or value.dtype != torch.bool
    ):
        raise Hold30AlphaDataError(f"{name} must be a CPU bool tensor of shape {shape}")


@dataclass(frozen=True, slots=True)
class Hold30AlphaEvaluationProvenance:
    """Content receipts and exact permitted roles for external return series."""

    risk_free_id: str
    market_benchmark_id: str
    factor_model_id: str
    factor_names: tuple[str, ...]
    factor_return_conventions: tuple[str, ...]
    risk_free_artifact_sha256: str
    market_artifact_sha256: str
    factor_artifact_sha256: str
    factor_plan_sha256: str
    risk_free_usage: tuple[str, ...] = HOLD30_ALPHA_RISK_FREE_USAGE
    market_usage: tuple[str, ...] = HOLD30_ALPHA_MARKET_USAGE
    factor_usage: tuple[str, ...] = HOLD30_ALPHA_FACTOR_USAGE
    policy_feature_access: bool = False

    def __post_init__(self) -> None:
        for name in ("risk_free_id", "market_benchmark_id", "factor_model_id"):
            _require_id(name, getattr(self, name))
        if (
            not isinstance(self.factor_names, tuple)
            or not self.factor_names
            or len(set(self.factor_names)) != len(self.factor_names)
        ):
            raise Hold30AlphaDataError("factor_names must be a non-empty unique tuple")
        for factor_name in self.factor_names:
            _require_id("factor_name", factor_name)
        if (
            not isinstance(self.factor_return_conventions, tuple)
            or len(self.factor_return_conventions) != len(self.factor_names)
            or any(
                convention not in _FACTOR_CONVENTIONS
                for convention in self.factor_return_conventions
            )
        ):
            raise Hold30AlphaDataError(
                "each factor needs an explicit supported return convention"
            )
        for name in (
            "risk_free_artifact_sha256",
            "market_artifact_sha256",
            "factor_artifact_sha256",
            "factor_plan_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.risk_free_usage != HOLD30_ALPHA_RISK_FREE_USAGE:
            raise Hold30AlphaDataError(
                "risk-free usage must remain accounting/Sharpe/"
                "checkpoint-ranking/evaluation only"
            )
        if self.market_usage != HOLD30_ALPHA_MARKET_USAGE:
            raise Hold30AlphaDataError(
                "market usage must remain beta/checkpoint/evaluation only"
            )
        if self.factor_usage != HOLD30_ALPHA_FACTOR_USAGE:
            raise Hold30AlphaDataError("declared factors must remain evaluator-only")
        if self.policy_feature_access is not False:
            raise Hold30AlphaDataError(
                "external return artifacts cannot become policy features"
            )

    @property
    def receipt_id(self) -> str:
        return _canonical_digest(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class Hold30AlphaEvaluationPanel:
    """Aligned role-separated risk-free, market, and factor return arrays."""

    source_axis_id: str
    risk_free_returns: torch.Tensor
    risk_free_valid: torch.Tensor
    market_total_returns: torch.Tensor
    market_valid: torch.Tensor
    factor_returns: torch.Tensor
    factor_valid: torch.Tensor
    provenance: Hold30AlphaEvaluationProvenance
    protocol_generation: str = HOLD30_ALPHA_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
        _require_digest("source_axis_id", self.source_axis_id)
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaDataError("evaluation panel rejects another generation")
        if not isinstance(self.provenance, Hold30AlphaEvaluationProvenance):
            raise Hold30AlphaDataError("typed evaluation provenance is required")
        _require_cpu_float64("risk_free_returns", self.risk_free_returns)
        if self.risk_free_returns.ndim != 2:
            raise Hold30AlphaDataError(
                "risk_free_returns must have shape [outbound_row, batch]"
            )
        rows, batch = self.risk_free_returns.shape
        base_shape = (rows, batch)
        factor_shape = (rows, batch, len(self.provenance.factor_names))
        _require_cpu_bool("risk_free_valid", self.risk_free_valid, base_shape)
        _require_cpu_float64(
            "market_total_returns", self.market_total_returns, base_shape
        )
        _require_cpu_bool("market_valid", self.market_valid, base_shape)
        _require_cpu_float64("factor_returns", self.factor_returns, factor_shape)
        _require_cpu_bool("factor_valid", self.factor_valid, factor_shape)
        for name, values, valid in (
            ("risk_free", self.risk_free_returns, self.risk_free_valid),
            ("market", self.market_total_returns, self.market_valid),
            ("factor", self.factor_returns, self.factor_valid),
        ):
            if bool((values.masked_select(~valid) != 0).any()):
                raise Hold30AlphaDataError(
                    f"invalid {name} return cells must be exact zero"
                )
        if bool((self.risk_free_returns[self.risk_free_valid] <= -1).any()):
            raise Hold30AlphaDataError("valid risk-free returns must exceed -100%")
        if bool((self.market_total_returns[self.market_valid] <= -1).any()):
            raise Hold30AlphaDataError("valid market returns must exceed -100%")

    @property
    def panel_id(self) -> str:
        return _canonical_digest(
            {
                "schema": HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA,
                "protocol_generation": self.protocol_generation,
                "source_axis_id": self.source_axis_id,
                "provenance_receipt_id": self.provenance.receipt_id,
                "risk_free_returns_sha256": _tensor_digest(
                    self.risk_free_returns
                ),
                "risk_free_valid_sha256": _tensor_digest(self.risk_free_valid),
                "market_total_returns_sha256": _tensor_digest(
                    self.market_total_returns
                ),
                "market_valid_sha256": _tensor_digest(self.market_valid),
                "factor_returns_sha256": _tensor_digest(self.factor_returns),
                "factor_valid_sha256": _tensor_digest(self.factor_valid),
            }
        )


@dataclass(frozen=True, slots=True)
class Hold30AlphaDataBindingReceipt:
    """Receipt proving that V3 auxiliary/evaluation data share the C1 axis."""

    protocol_generation: str
    source_axis_id: str
    c1_benchmark_id: str
    c1_trace_sha256: str
    cash_returns_sha256: str
    evaluation_panel_id: str
    evaluation_provenance_id: str
    global_path_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaDataError("data binding rejects another generation")
        if self.c1_benchmark_id != HOLD30_ALPHA_C1_BENCHMARK_ID:
            raise Hold30AlphaDataError("V3 training benchmark must remain frozen C1")
        for name in (
            "source_axis_id",
            "c1_trace_sha256",
            "cash_returns_sha256",
            "evaluation_panel_id",
            "evaluation_provenance_id",
        ):
            _require_digest(name, getattr(self, name))
        if (
            not isinstance(self.global_path_ids, tuple)
            or not self.global_path_ids
            or any(
                isinstance(path_id, bool)
                or not isinstance(path_id, int)
                or path_id < 0
                for path_id in self.global_path_ids
            )
            or tuple(sorted(self.global_path_ids)) != self.global_path_ids
            or len(set(self.global_path_ids)) != len(self.global_path_ids)
        ):
            raise Hold30AlphaDataError(
                "global_path_ids must be sorted unique nonnegative integers"
            )

    @property
    def receipt_id(self) -> str:
        return _canonical_digest(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


def bind_hold30_alpha_evaluation_panel(
    sequence: Hold30DatasetSequence,
    panel: Hold30AlphaEvaluationPanel,
) -> Hold30AlphaDataBindingReceipt:
    """Bind external return data to one exact C1 sequence without actor access."""

    if not isinstance(sequence, Hold30DatasetSequence):
        raise Hold30AlphaDataError("sequence must be a Hold30DatasetSequence")
    if not isinstance(panel, Hold30AlphaEvaluationPanel):
        raise Hold30AlphaDataError("panel must be a Hold30AlphaEvaluationPanel")
    if panel.source_axis_id != sequence.axis_id:
        raise Hold30AlphaDataError("evaluation panel axis does not match C1 sequence")
    expected = (sequence.n_positions - 1, sequence.batch_size)
    if tuple(panel.risk_free_returns.shape) != expected:
        raise Hold30AlphaDataError(
            f"evaluation panel must have outbound shape {expected}"
        )
    if (
        sequence.asset_returns.device.type != "cpu"
        or sequence.asset_returns.dtype != torch.float64
    ):
        raise Hold30AlphaDataError(
            "V3 data qualification requires CPU float64 sequence returns"
        )
    if not bool(panel.risk_free_valid.all()):
        raise Hold30AlphaDataError("the explicit PIT risk-free series must be complete")
    if not bool(panel.market_valid.all()):
        raise Hold30AlphaDataError("the explicit PIT market series must be complete")
    if not bool(panel.factor_valid.all()):
        raise Hold30AlphaDataError("the declared PIT factor panel must be complete")
    cash = sequence.asset_returns[..., sequence.cash_index]
    if not torch.equal(panel.risk_free_returns, cash):
        raise Hold30AlphaDataError(
            "PIT risk-free returns must equal the sequence CASH series bitwise"
        )
    return Hold30AlphaDataBindingReceipt(
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        source_axis_id=sequence.axis_id,
        c1_benchmark_id=HOLD30_ALPHA_C1_BENCHMARK_ID,
        c1_trace_sha256=sequence.provenance.c1_benchmark_trace_sha256,
        cash_returns_sha256=_tensor_digest(cash),
        evaluation_panel_id=panel.panel_id,
        evaluation_provenance_id=panel.provenance.receipt_id,
        global_path_ids=tuple(range(sequence.batch_size)),
    )


@dataclass(frozen=True, slots=True)
class Hold30AlphaLabelDomain:
    """One split-local outbound-return interval ``[start, stop)``."""

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_id("label domain name", self.name)
        for field_name in ("start", "stop"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Hold30AlphaDataError(
                    f"label domain {field_name} must be a nonnegative integer"
                )
        if self.stop <= self.start:
            raise Hold30AlphaDataError("label domains must be non-empty")


def _validate_label_domains(
    domains: Sequence[Hold30AlphaLabelDomain],
    rows: int,
) -> tuple[Hold30AlphaLabelDomain, ...]:
    if not isinstance(domains, (tuple, list)) or not domains:
        raise Hold30AlphaDataError("at least one label domain is required")
    result = tuple(domains)
    if any(not isinstance(domain, Hold30AlphaLabelDomain) for domain in result):
        raise Hold30AlphaDataError("label domains must be typed")
    if len({domain.name for domain in result}) != len(result):
        raise Hold30AlphaDataError("label domain names must be unique")
    cursor = 0
    for domain in result:
        if domain.start != cursor:
            raise Hold30AlphaDataError(
                "label domains must be ordered, disjoint, and contiguous"
            )
        cursor = domain.stop
    if cursor != rows:
        raise Hold30AlphaDataError(
            "label domains must partition every outbound-return row"
        )
    return result


@dataclass(frozen=True, slots=True)
class Hold30ResidualAlphaLabels:
    """Right-censored auxiliary residual-alpha targets for score origins."""

    source_axis_id: str
    horizons: tuple[int, ...]
    origin_rows: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    censored: torch.Tensor
    domains: tuple[Hold30AlphaLabelDomain, ...]
    cash_index: int
    protocol_generation: str = HOLD30_ALPHA_PROTOCOL_GENERATION
    auxiliary_only: bool = True
    actor_access: bool = False
    label_rule: str = HOLD30_ALPHA_LABEL_RULE

    def __post_init__(self) -> None:
        _require_digest("source_axis_id", self.source_axis_id)
        if self.protocol_generation != HOLD30_ALPHA_PROTOCOL_GENERATION:
            raise Hold30AlphaDataError("labels reject another protocol generation")
        if self.horizons != HOLD30_ALPHA_HORIZONS:
            raise Hold30AlphaDataError("alpha horizons must be exactly (5, 21, 30, 63)")
        if self.auxiliary_only is not True or self.actor_access is not False:
            raise Hold30AlphaDataError("future alpha labels must remain auxiliary-only")
        if self.label_rule != HOLD30_ALPHA_LABEL_RULE:
            raise Hold30AlphaDataError("residual-alpha label chronology drifted")
        if (
            not isinstance(self.origin_rows, torch.Tensor)
            or self.origin_rows.device.type != "cpu"
            or self.origin_rows.dtype != torch.int64
            or self.origin_rows.ndim != 1
            or self.origin_rows.numel() == 0
        ):
            raise Hold30AlphaDataError("origin_rows must be a non-empty CPU int64 vector")
        if self.origin_rows.numel() > 1 and bool(
            (self.origin_rows[1:] <= self.origin_rows[:-1]).any()
        ):
            raise Hold30AlphaDataError("origin_rows must be strictly increasing")
        _require_cpu_float64("values", self.values)
        if self.values.ndim != 4 or self.values.shape[:2] != (
            len(self.horizons),
            self.origin_rows.numel(),
        ):
            raise Hold30AlphaDataError(
                "values must have shape [horizon, score_origin, batch, asset]"
            )
        shape = tuple(self.values.shape)
        _require_cpu_bool("valid", self.valid, shape)
        _require_cpu_bool("censored", self.censored, shape)
        if bool((self.valid & self.censored).any()):
            raise Hold30AlphaDataError("a label cannot be both valid and censored")
        if bool((self.values.masked_select(~self.valid) != 0).any()):
            raise Hold30AlphaDataError("invalid labels must be exact zero")
        if not 0 <= self.cash_index < shape[-1]:
            raise Hold30AlphaDataError("cash_index lies outside the label asset axis")
        if bool(self.valid[..., self.cash_index].any()) or bool(
            self.censored[..., self.cash_index].any()
        ):
            raise Hold30AlphaDataError("CASH cannot have a residual stock-alpha label")

    @property
    def labels_id(self) -> str:
        return _canonical_digest(
            {
                "schema": HOLD30_ALPHA_LABEL_SCHEMA,
                "protocol_generation": self.protocol_generation,
                "source_axis_id": self.source_axis_id,
                "horizons": self.horizons,
                "origin_rows_sha256": _tensor_digest(self.origin_rows),
                "values_sha256": _tensor_digest(self.values),
                "valid_sha256": _tensor_digest(self.valid),
                "censored_sha256": _tensor_digest(self.censored),
                "domains": [
                    {"name": domain.name, "start": domain.start, "stop": domain.stop}
                    for domain in self.domains
                ],
                "cash_index": self.cash_index,
                "auxiliary_only": self.auxiliary_only,
                "actor_access": self.actor_access,
                "label_rule": self.label_rule,
                "training_benchmark_id": HOLD30_ALPHA_C1_BENCHMARK_ID,
            }
        )


def build_hold30_residual_alpha_labels(
    sequence: Hold30DatasetSequence,
    *,
    domains: Sequence[Hold30AlphaLabelDomain],
) -> Hold30ResidualAlphaLabels:
    """Build exact 5/21/30/63-session C1-residual labels.

    Eligible risky names are frozen at ``a_trade[t]``.  If a name later
    becomes unavailable, its hypothetical notional moves to CASH and remains
    there even if the name subsequently re-enters the universe.
    """

    if not isinstance(sequence, Hold30DatasetSequence):
        raise Hold30AlphaDataError("sequence must be a Hold30DatasetSequence")
    if (
        sequence.asset_returns.device.type != "cpu"
        or sequence.asset_returns.dtype != torch.float64
    ):
        raise Hold30AlphaDataError(
            "V3 residual labels require CPU float64 sequence returns"
        )
    rows = sequence.n_positions - 1
    split_domains = _validate_label_domains(domains, rows)
    origins = sequence.roles.score_indices.clone()
    batch, assets = sequence.batch_size, sequence.num_assets
    shape = (len(HOLD30_ALPHA_HORIZONS), origins.numel(), batch, assets)
    values = torch.zeros(shape, dtype=torch.float64)
    valid = torch.zeros(shape, dtype=torch.bool)
    censored = torch.zeros(shape, dtype=torch.bool)
    risky = torch.ones((batch, assets), dtype=torch.bool)
    risky[..., sequence.cash_index] = False

    domain_by_row: list[Hold30AlphaLabelDomain] = [split_domains[0]] * rows
    for domain in split_domains:
        domain_by_row[domain.start : domain.stop] = [domain] * (
            domain.stop - domain.start
        )

    for origin_index, origin_value in enumerate(origins.tolist()):
        origin = int(origin_value)
        if not 0 <= origin < rows:
            raise Hold30AlphaDataError("a score origin has no outbound transition")
        eligible = sequence.a_trade[origin].detach().to(device="cpu") & risky
        domain = domain_by_row[origin]
        for horizon_index, horizon in enumerate(HOLD30_ALPHA_HORIZONS):
            final_return_row = origin + horizon
            if final_return_row >= domain.stop:
                censored[horizon_index, origin_index] = eligible
                continue

            still_held = eligible.clone()
            stock_log_wealth = torch.zeros((batch, assets), dtype=torch.float64)
            for return_row in range(origin + 1, final_return_row + 1):
                # The book at return_row was established at that position's
                # fill. A failed fill mask therefore moves the notional to
                # CASH before this row; once forced out it never re-enters.
                still_held &= sequence.fill_trade[return_row].to(device="cpu")
                realized = torch.where(
                    still_held,
                    sequence.asset_returns[return_row],
                    sequence.asset_returns[
                        return_row, :, sequence.cash_index
                    ].unsqueeze(-1),
                )
                stock_log_wealth += torch.log1p(realized)

            c1_log_wealth = torch.log1p(
                sequence.c1_benchmark_net_returns[
                    origin + 1 : final_return_row + 1
                ]
            ).sum(dim=0)
            residual = stock_log_wealth - c1_log_wealth.unsqueeze(-1)
            valid[horizon_index, origin_index] = eligible
            values[horizon_index, origin_index] = torch.where(
                eligible, residual, torch.zeros_like(residual)
            )

    return Hold30ResidualAlphaLabels(
        source_axis_id=sequence.axis_id,
        horizons=HOLD30_ALPHA_HORIZONS,
        origin_rows=origins,
        values=values,
        valid=valid,
        censored=censored,
        domains=split_domains,
        cash_index=sequence.cash_index,
    )


def verify_hold30_residual_alpha_labels(
    sequence: Hold30DatasetSequence,
    labels: Hold30ResidualAlphaLabels,
) -> None:
    """Rebuild labels from bound source tensors and require exact equality."""

    if labels.source_axis_id != sequence.axis_id:
        raise Hold30AlphaDataError("label axis does not match the source sequence")
    rebuilt = build_hold30_residual_alpha_labels(sequence, domains=labels.domains)
    if rebuilt.labels_id != labels.labels_id:
        raise Hold30AlphaDataError("residual-alpha label receipt does not recompute")


__all__ = [
    "HOLD30_ALPHA_EVALUATION_PANEL_SCHEMA",
    "HOLD30_ALPHA_FACTOR_USAGE",
    "HOLD30_ALPHA_LABEL_RULE",
    "HOLD30_ALPHA_LABEL_SCHEMA",
    "HOLD30_ALPHA_MARKET_USAGE",
    "HOLD30_ALPHA_RISK_FREE_USAGE",
    "Hold30AlphaDataBindingReceipt",
    "Hold30AlphaDataError",
    "Hold30AlphaEvaluationPanel",
    "Hold30AlphaEvaluationProvenance",
    "Hold30AlphaLabelDomain",
    "Hold30ResidualAlphaLabels",
    "bind_hold30_alpha_evaluation_panel",
    "build_hold30_residual_alpha_labels",
    "verify_hold30_residual_alpha_labels",
]
