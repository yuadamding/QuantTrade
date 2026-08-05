"""Fail-closed point-in-time data contract for the Hold-30 mechanism screen.

The Hold-30 runtime deliberately delays a decision at position ``t`` until
the next fill at position ``t + 1``.  A dataset must therefore preserve two
different information sets instead of publishing one convenient availability
mask:

* decision membership/tradability, which the actor may observe at ``t``; and
* fill membership/tradability, which is applied only at ``t + 1``.

``a_trade[t]`` is their exact intersection.  A constituent added at the fill
cannot be bought by a decision that did not know it, while a constituent
deleted at the fill is repaired out of the book before discretionary trading.

This module does not discover a universe, infer missing event history, or fall
back to a full-sample TOP2000 list.  Callers must supply an event-sourced
point-in-time receipt, explicit cash and C1 benchmark rules, and all tensors.
It is a lower-layer dataset contract, so :meth:`Hold30DatasetSequence.runtime_kwargs`
returns a structurally compatible mapping without importing the training or
environment packages.

The two registered signal-destruction transforms randomize only ordinary risky
*outcomes*.  Policy-visible state, CASH, mandatory corporate-action/delisting
outcomes, membership, availability, and costs remain at their legal dates.
The transformed return view is deliberately not accepted by
:meth:`Hold30DatasetSequence.runtime_kwargs`: C1, labels, drift, and endpoints
must be rebuilt explicitly before a transformed dataset can enter a runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal

import torch


HOLD30_PRELOCKBOX_CUTOFF_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z
HOLD30_WARMUP_POSITIONS = 63
HOLD30_SUPPORT_POSITIONS = 30
HOLD30_CREDIT_RETURNS = 30
HOLD30_CASH_ASSET_ID = "CASH"
HOLD30_BENCHMARK_ID = "C1"
HOLD30_UNIVERSE_MODE = "point_in_time_events"
HOLD30_CASH_RETURN_RULE = "explicit_one_step_total_return"


class Hold30DatasetError(ValueError):
    """A scientific or chronology invariant is absent or inconsistent."""


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30DatasetError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_tensor_shape(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise Hold30DatasetError(
            f"{name} must have shape {shape}; got {getattr(value, 'shape', None)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise Hold30DatasetError(f"{name} must have dtype {dtype}; got {value.dtype}")


def _require_floating(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    _require_tensor_shape(name, value, shape)
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise Hold30DatasetError(f"{name} must be a finite floating-point tensor")


def _require_integer(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    _require_tensor_shape(name, value, shape)
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise Hold30DatasetError(f"{name} must use an integer dtype")


def _as_cpu_int64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    _require_integer(name, value, shape)
    return value.detach().to(device="cpu", dtype=torch.int64)


@dataclass(frozen=True, slots=True)
class Hold30PointInTimeProvenance:
    """Content receipts required before a sequence may enter the screen."""

    data_snapshot_sha256: str
    raw_market_data_sha256: str
    universe_events_sha256: str
    tradability_events_sha256: str
    corporate_actions_sha256: str
    identifier_events_sha256: str
    c1_benchmark_trace_sha256: str
    risk_limits_sha256: str
    universe_mode: str
    universe_rule_id: str
    stable_asset_id_namespace: str
    benchmark_id: str
    cash_asset_id: str
    cash_return_rule: str

    def __post_init__(self) -> None:
        for name in (
            "data_snapshot_sha256",
            "raw_market_data_sha256",
            "universe_events_sha256",
            "tradability_events_sha256",
            "corporate_actions_sha256",
            "identifier_events_sha256",
            "c1_benchmark_trace_sha256",
            "risk_limits_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if self.universe_mode != HOLD30_UNIVERSE_MODE:
            raise Hold30DatasetError(
                "universe_mode must be point_in_time_events; static or future-selected "
                "TOP2000 fallback is forbidden"
            )
        for name in ("universe_rule_id", "stable_asset_id_namespace"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise Hold30DatasetError(f"{name} must be a non-empty stable identifier")
        if self.benchmark_id != HOLD30_BENCHMARK_ID:
            raise Hold30DatasetError("the benchmark receipt must identify C1")
        if self.cash_asset_id != HOLD30_CASH_ASSET_ID:
            raise Hold30DatasetError("the synthetic cash asset must be identified as CASH")
        if self.cash_return_rule != HOLD30_CASH_RETURN_RULE:
            raise Hold30DatasetError(
                "cash_return_rule must bind an explicit one-step total-return series"
            )

    @property
    def receipt_id(self) -> str:
        return _canonical_digest(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class Hold30AsOfEvidence:
    """As-of receipts for masks, corporate actions, and identifier history.

    Known-at timestamps are epoch milliseconds.  Mask snapshots require an
    explicit receipt for every cell; ``-1`` is reserved only for corporate or
    identifier histories whose cumulative event version is still zero.
    Corporate-action factors are cumulative total-return adjustment factors as
    known at each decision position.  Identifier versions refer to the
    event-sourced mapping into the stable canonical asset axis.
    """

    decision_membership_known_at_ms: torch.Tensor
    decision_tradability_known_at_ms: torch.Tensor
    fill_membership_known_at_ms: torch.Tensor
    fill_tradability_known_at_ms: torch.Tensor
    corporate_action_factor: torch.Tensor
    corporate_action_version: torch.Tensor
    corporate_action_known_at_ms: torch.Tensor
    identifier_version: torch.Tensor
    identifier_known_at_ms: torch.Tensor

    def validate(
        self,
        *,
        shape: tuple[int, int, int],
        decision_timestamps_ms: torch.Tensor,
        fill_timestamps_ms: torch.Tensor,
        cash_index: int,
    ) -> None:
        positions, _, _ = shape
        decision_ts = _as_cpu_int64(
            "decision_timestamps_ms", decision_timestamps_ms, (positions,)
        ).view(positions, 1, 1)
        fill_ts = _as_cpu_int64(
            "fill_timestamps_ms", fill_timestamps_ms, (positions,)
        ).view(positions, 1, 1)

        for name, ceiling in (
            ("decision_membership_known_at_ms", decision_ts),
            ("decision_tradability_known_at_ms", decision_ts),
            ("fill_membership_known_at_ms", fill_ts),
            ("fill_tradability_known_at_ms", fill_ts),
        ):
            values = _as_cpu_int64(name, getattr(self, name), shape)
            if bool((values < 0).any()) or bool((values > ceiling).any()):
                raise Hold30DatasetError(
                    f"{name} must contain nonnegative receipts no later than its information time"
                )

        _require_floating("corporate_action_factor", self.corporate_action_factor, shape)
        if bool((self.corporate_action_factor <= 0).any()):
            raise Hold30DatasetError("corporate_action_factor must be strictly positive")
        for prefix in ("corporate_action", "identifier"):
            versions = _as_cpu_int64(f"{prefix}_version", getattr(self, f"{prefix}_version"), shape)
            known = _as_cpu_int64(
                f"{prefix}_known_at_ms", getattr(self, f"{prefix}_known_at_ms"), shape
            )
            if bool((versions < 0).any()):
                raise Hold30DatasetError(f"{prefix}_version cannot be negative")
            if not torch.equal(known.eq(-1), versions.eq(0)):
                raise Hold30DatasetError(
                    f"{prefix}_known_at_ms must be -1 exactly while event version is zero"
                )
            populated = versions > 0
            if bool((known[populated] > decision_ts.expand(shape)[populated]).any()):
                raise Hold30DatasetError(f"{prefix} evidence contains a future-available event")
            if positions > 1 and bool((versions[1:] < versions[:-1]).any()):
                raise Hold30DatasetError(f"{prefix}_version must be cumulative and nondecreasing")
            if positions > 1 and bool((known[1:] < known[:-1]).any()):
                raise Hold30DatasetError(f"{prefix}_known_at_ms must be nondecreasing")

        corporate_versions = self.corporate_action_version.detach().to(device="cpu", dtype=torch.int64)
        factors = self.corporate_action_factor.detach().to(device="cpu")
        unchanged = corporate_versions[1:] == corporate_versions[:-1]
        changed_factor = factors[1:] != factors[:-1]
        if bool((unchanged & changed_factor).any()):
            raise Hold30DatasetError(
                "corporate_action_factor changed without a new corporate-action event version"
            )

        cash_factor = self.corporate_action_factor[..., cash_index]
        if not bool(torch.equal(cash_factor, torch.ones_like(cash_factor))):
            raise Hold30DatasetError("CASH corporate_action_factor must be exactly one")
        for name in ("corporate_action_version", "identifier_version"):
            if bool((getattr(self, name)[..., cash_index] != 0).any()):
                raise Hold30DatasetError(f"CASH {name} must remain zero")


@dataclass(frozen=True, slots=True)
class Hold30DatasetRoles:
    """Materialized role masks for one canonical Hold-30 chronology."""

    warmup: torch.Tensor
    score: torch.Tensor
    support: torch.Tensor
    terminal: torch.Tensor
    score_indices: torch.Tensor
    utility_rows: torch.Tensor
    replay_terminal_rows: torch.Tensor

    @classmethod
    def materialize(cls, n_positions: int) -> "Hold30DatasetRoles":
        minimum = HOLD30_WARMUP_POSITIONS + HOLD30_SUPPORT_POSITIONS + 2
        if isinstance(n_positions, bool) or not isinstance(n_positions, int) or n_positions < minimum:
            raise Hold30DatasetError(
                f"Hold-30 chronology needs at least {minimum} positions; got {n_positions!r}"
            )
        terminal_index = n_positions - 1
        support_start = terminal_index - HOLD30_SUPPORT_POSITIONS
        score_indices = torch.arange(HOLD30_WARMUP_POSITIONS, support_start, dtype=torch.int64)
        if score_indices.numel() == 0:
            raise Hold30DatasetError("Hold-30 chronology contains no score-bearing origin")

        warmup = torch.zeros(n_positions, dtype=torch.bool)
        score = torch.zeros(n_positions, dtype=torch.bool)
        support = torch.zeros(n_positions, dtype=torch.bool)
        terminal = torch.zeros(n_positions, dtype=torch.bool)
        warmup[:HOLD30_WARMUP_POSITIONS] = True
        score[score_indices] = True
        support[support_start:terminal_index] = True
        terminal[terminal_index] = True
        offsets = torch.arange(HOLD30_CREDIT_RETURNS + 1, dtype=torch.int64)
        utility_rows = score_indices.unsqueeze(1) + offsets.unsqueeze(0)
        replay_terminal_rows = score_indices + HOLD30_CREDIT_RETURNS + 1
        if not bool((warmup | score | support | terminal).all()):
            raise AssertionError("Hold-30 roles do not partition every position")
        if bool(((warmup.to(torch.int8) + score + support + terminal) != 1).any()):
            raise AssertionError("Hold-30 roles overlap")
        return cls(
            warmup=warmup,
            score=score,
            support=support,
            terminal=terminal,
            score_indices=score_indices,
            utility_rows=utility_rows,
            replay_terminal_rows=replay_terminal_rows,
        )


@dataclass(frozen=True, slots=True)
class Hold30NullReceipt:
    """Content-addressable receipt for one ordinary-outcome null transform."""

    kind: Literal["N_time", "N_xs"]
    seed: int
    source_axis_id: str
    randomization_axis_id: str
    domains: tuple[tuple[str, int, int], ...]
    domains_sha256: str
    input_outcomes_sha256: str
    ordinary_valid_sha256: str
    mandatory_mask_sha256: str
    active_mask_sha256: str
    mapping_sha256: str
    output_outcomes_sha256: str
    transform_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"N_time", "N_xs"}:
            raise Hold30DatasetError("null kind must be N_time or N_xs")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise Hold30DatasetError("null seed must be an integer in [0, 2**63)")
        if self.domains != tuple(
            (domain.name, domain.start, domain.stop)
            for domain in _validate_null_domains(
                tuple(Hold30NullDomain(*value) for value in self.domains),
                self.domains[-1][2] if self.domains else 0,
            )
        ):
            raise Hold30DatasetError("null receipt domains are not canonical")
        for name in (
            "source_axis_id",
            "randomization_axis_id",
            "domains_sha256",
            "input_outcomes_sha256",
            "ordinary_valid_sha256",
            "mandatory_mask_sha256",
            "active_mask_sha256",
            "mapping_sha256",
            "output_outcomes_sha256",
            "transform_id",
        ):
            _require_digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Hold30NullDomain:
    """One outbound-return domain that a null mapping may never cross."""

    name: Literal["train", "validation", "outer"]
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.name not in {"train", "validation", "outer"}:
            raise Hold30DatasetError("null domain name must be train, validation, or outer")
        for field_name in ("start", "stop"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Hold30DatasetError(f"null domain {field_name} must be a nonnegative integer")
        if self.stop <= self.start:
            raise Hold30DatasetError("null domains must contain at least one outbound row")


@dataclass(frozen=True, slots=True)
class Hold30NullView:
    """Transformed ordinary outcomes plus the exact source-index mapping."""

    asset_returns: torch.Tensor
    source_index: torch.Tensor
    receipt: Hold30NullReceipt


def _validate_null_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise Hold30DatasetError("null seed must be an integer in [0, 2**63)")
    return seed


def _null_hash(seed: int, *indices: int) -> bytes:
    """Hash unsigned big-endian ``seed || index...`` without platform RNG state."""

    _validate_null_seed(seed)
    material = seed.to_bytes(8, "big", signed=False)
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise Hold30DatasetError("null-transform hash indices must be nonnegative integers")
        material += index.to_bytes(8, "big", signed=False)
    return hashlib.sha256(material).digest()


def _validate_null_domains(
    domains: tuple[Hold30NullDomain, ...],
    n_rows: int,
) -> tuple[Hold30NullDomain, ...]:
    if not isinstance(domains, tuple) or len(domains) != 3:
        raise Hold30DatasetError(
            "null transforms require exactly train, validation, and outer domains"
        )
    expected_names = ("train", "validation", "outer")
    if tuple(domain.name for domain in domains) != expected_names:
        raise Hold30DatasetError(
            "null domains must be ordered train, validation, outer"
        )
    cursor = 0
    for domain in domains:
        if not isinstance(domain, Hold30NullDomain):
            raise Hold30DatasetError("null domains must contain Hold30NullDomain values")
        if domain.start != cursor:
            raise Hold30DatasetError(
                "null domains must be disjoint, contiguous, and start at outbound row zero"
            )
        cursor = domain.stop
    if cursor != n_rows:
        raise Hold30DatasetError("null domains must partition every outbound-return row")
    return domains


def _validate_outcome_transform_inputs(
    asset_returns: torch.Tensor,
    ordinary_return_valid: torch.Tensor,
    mandatory_return_mask: torch.Tensor,
    destination_active: torch.Tensor,
    *,
    domains: tuple[Hold30NullDomain, ...],
    cash_index: int,
) -> tuple[int, int, int, tuple[Hold30NullDomain, ...]]:
    if not isinstance(asset_returns, torch.Tensor) or asset_returns.ndim != 3:
        raise Hold30DatasetError("asset_returns must have shape [outbound_row, batch, asset]")
    rows, batch, assets = asset_returns.shape
    _require_floating("asset_returns", asset_returns, (rows, batch, assets))
    for name, value in (
        ("ordinary_return_valid", ordinary_return_valid),
        ("mandatory_return_mask", mandatory_return_mask),
        ("destination_active", destination_active),
    ):
        _require_tensor_shape(name, value, (rows, batch, assets), dtype=torch.bool)
        if value.device != asset_returns.device:
            raise Hold30DatasetError(f"{name} must share asset_returns device")
    if not 0 <= cash_index < assets:
        raise Hold30DatasetError("cash_index is outside the asset axis")
    if bool(ordinary_return_valid[..., cash_index].any()) or bool(
        mandatory_return_mask[..., cash_index].any()
    ):
        raise Hold30DatasetError("CASH cannot be ordinary-transformable or mandatory")
    if bool((ordinary_return_valid & mandatory_return_mask).any()):
        raise Hold30DatasetError("ordinary and mandatory outcome masks must be disjoint")
    risky = torch.ones_like(destination_active, dtype=torch.bool)
    risky[..., cash_index] = False
    required = destination_active & risky & ~mandatory_return_mask
    if not torch.equal(ordinary_return_valid, required):
        raise Hold30DatasetError(
            "ordinary_return_valid must equal active, risky, nonmandatory outcomes"
        )
    return rows, batch, assets, _validate_null_domains(domains, rows)


def _deterministic_hopcroft_karp(
    destinations: tuple[int, ...],
    adjacency: dict[int, tuple[int, ...]],
) -> dict[int, int]:
    """Return a deterministic perfect matching or fail closed."""

    unmatched = -1
    pair_destination = {destination: unmatched for destination in destinations}
    pair_source = {source: unmatched for source in destinations}
    distance: dict[int, int] = {}

    def bfs() -> bool:
        queue: list[int] = []
        head = 0
        for destination in destinations:
            if pair_destination[destination] == unmatched:
                distance[destination] = 0
                queue.append(destination)
            else:
                distance[destination] = -1
        found = False
        while head < len(queue):
            destination = queue[head]
            head += 1
            for source in adjacency[destination]:
                paired = pair_source[source]
                if paired == unmatched:
                    found = True
                elif distance[paired] < 0:
                    distance[paired] = distance[destination] + 1
                    queue.append(paired)
        return found

    def dfs(destination: int) -> bool:
        for source in adjacency[destination]:
            paired = pair_source[source]
            if paired == unmatched or (
                distance.get(paired, -1) == distance[destination] + 1 and dfs(paired)
            ):
                pair_destination[destination] = source
                pair_source[source] = destination
                return True
        distance[destination] = -1
        return False

    matched = 0
    while bfs():
        for destination in destinations:
            if pair_destination[destination] == unmatched and dfs(destination):
                matched += 1
    if matched != len(destinations):
        raise Hold30DatasetError(
            "N_time has no perfect within-domain date matching under the 31-position "
            "separation and ordinary-return-validity rules"
        )
    return pair_destination


def _receipt(
    *,
    kind: Literal["N_time", "N_xs"],
    seed: int,
    source_axis_id: str,
    randomization_axis_id: str,
    domains: tuple[Hold30NullDomain, ...],
    input_outcomes: torch.Tensor,
    output_outcomes: torch.Tensor,
    ordinary_return_valid: torch.Tensor,
    mandatory_return_mask: torch.Tensor,
    destination_active: torch.Tensor,
    source_index: torch.Tensor,
) -> Hold30NullReceipt:
    domain_values = tuple((domain.name, domain.start, domain.stop) for domain in domains)
    domains_sha256 = _canonical_digest(domain_values)
    input_outcomes_sha256 = _tensor_digest(input_outcomes)
    ordinary_valid_sha256 = _tensor_digest(ordinary_return_valid)
    mandatory_mask_sha256 = _tensor_digest(mandatory_return_mask)
    active_mask_sha256 = _tensor_digest(destination_active)
    mapping_sha256 = _tensor_digest(source_index)
    output_outcomes_sha256 = _tensor_digest(output_outcomes)
    transform_id = _canonical_digest(
        {
            "contract": "hold30-ordinary-outcome-null-v1",
            "kind": kind,
            "seed": seed,
            "source_axis_id": source_axis_id,
            "randomization_axis_id": randomization_axis_id,
            "domains": domain_values,
            "domains_sha256": domains_sha256,
            "input_outcomes_sha256": input_outcomes_sha256,
            "ordinary_valid_sha256": ordinary_valid_sha256,
            "mandatory_mask_sha256": mandatory_mask_sha256,
            "active_mask_sha256": active_mask_sha256,
            "mapping_sha256": mapping_sha256,
            "output_outcomes_sha256": output_outcomes_sha256,
        }
    )
    return Hold30NullReceipt(
        kind=kind,
        seed=seed,
        source_axis_id=source_axis_id,
        randomization_axis_id=randomization_axis_id,
        domains=domain_values,
        domains_sha256=domains_sha256,
        input_outcomes_sha256=input_outcomes_sha256,
        ordinary_valid_sha256=ordinary_valid_sha256,
        mandatory_mask_sha256=mandatory_mask_sha256,
        active_mask_sha256=active_mask_sha256,
        mapping_sha256=mapping_sha256,
        output_outcomes_sha256=output_outcomes_sha256,
        transform_id=transform_id,
    )


def n_time_transform(
    asset_returns: torch.Tensor,
    ordinary_return_valid: torch.Tensor,
    mandatory_return_mask: torch.Tensor,
    destination_active: torch.Tensor,
    *,
    domains: tuple[Hold30NullDomain, ...],
    seed: int,
    source_axis_id: str,
    randomization_axis_id: str,
    cash_index: int = 0,
) -> Hold30NullView:
    """Match complete ordinary risky return vectors to distant dates by role."""

    rows, _batch, _assets, domains = _validate_outcome_transform_inputs(
        asset_returns,
        ordinary_return_valid,
        mandatory_return_mask,
        destination_active,
        domains=domains,
        cash_index=cash_index,
    )
    _validate_null_seed(seed)
    transformed = asset_returns.clone()
    source_index = torch.arange(rows, dtype=torch.int64)
    ordinary_cpu = ordinary_return_valid.detach().to(device="cpu")
    required_cpu = (
        destination_active & ~mandatory_return_mask
    ).detach().to(device="cpu")
    required_cpu[..., cash_index] = False

    for domain in domains:
        destinations = tuple(range(domain.start, domain.stop))
        valid_source_cache: dict[bytes, torch.Tensor] = {}
        adjacency: dict[int, tuple[int, ...]] = {}
        domain_sources = torch.arange(domain.start, domain.stop, dtype=torch.int64)
        for destination in destinations:
            required_here = required_cpu[destination]
            cache_key = required_here.contiguous().numpy().tobytes(order="C")
            valid_sources = valid_source_cache.get(cache_key)
            if valid_sources is None:
                required_flat = required_here.flatten()
                source_rows = ordinary_cpu[domain.start : domain.stop].reshape(
                    domain.stop - domain.start, -1
                )
                valid_local = source_rows[:, required_flat].all(dim=1)
                valid_sources = domain_sources[valid_local]
                valid_source_cache[cache_key] = valid_sources
            legal = [
                int(source)
                for source in valid_sources.tolist()
                if abs(int(source) - destination) >= HOLD30_CREDIT_RETURNS + 1
            ]
            legal.sort(key=lambda source: (_null_hash(seed, destination, source), source))
            adjacency[destination] = tuple(legal)
        matching = _deterministic_hopcroft_karp(destinations, adjacency)
        for destination in destinations:
            source = matching[destination]
            source_index[destination] = source
            assign = required_cpu[destination].to(device=asset_returns.device)
            transformed[destination][assign] = asset_returns[source][assign]

    # CASH, mandatory outcomes, and inactive cells are bitwise unchanged.
    fixed = ~ordinary_return_valid
    if not torch.equal(transformed.masked_select(fixed), asset_returns.masked_select(fixed)):
        raise AssertionError("N_time modified a fixed outcome")
    receipt = _receipt(
        kind="N_time",
        seed=seed,
        source_axis_id=source_axis_id,
        randomization_axis_id=randomization_axis_id,
        domains=domains,
        input_outcomes=asset_returns,
        output_outcomes=transformed,
        ordinary_return_valid=ordinary_return_valid,
        mandatory_return_mask=mandatory_return_mask,
        destination_active=destination_active,
        source_index=source_index,
    )
    return Hold30NullView(transformed, source_index, receipt)


def n_xs_transform(
    asset_returns: torch.Tensor,
    ordinary_return_valid: torch.Tensor,
    mandatory_return_mask: torch.Tensor,
    destination_active: torch.Tensor,
    *,
    domains: tuple[Hold30NullDomain, ...],
    seed: int,
    source_axis_id: str,
    randomization_axis_id: str,
    cash_index: int = 0,
) -> Hold30NullView:
    """Apply a deterministic nonidentity cyclic ordinary-return permutation."""

    rows, batch, assets, domains = _validate_outcome_transform_inputs(
        asset_returns,
        ordinary_return_valid,
        mandatory_return_mask,
        destination_active,
        domains=domains,
        cash_index=cash_index,
    )
    _validate_null_seed(seed)
    transformed = asset_returns.clone()
    source_index = torch.arange(assets, dtype=torch.int64).view(1, 1, assets).expand(
        rows, batch, assets
    ).clone()
    eligible_cpu = ordinary_return_valid.detach().to(device="cpu")
    for position in range(rows):
        for batch_index in range(batch):
            assets_here = torch.nonzero(
                eligible_cpu[position, batch_index], as_tuple=False
            ).flatten()
            assets_here = assets_here[assets_here != cash_index]
            if assets_here.numel() < 2:
                raise Hold30DatasetError(
                    "N_xs requires at least two ordinary eligible risky outcomes "
                    f"at outbound row {position}, batch {batch_index}"
                )
            shift = 1 + int.from_bytes(
                _null_hash(seed, position, batch_index), "big"
            ) % (assets_here.numel() - 1)
            sources = torch.roll(assets_here, shifts=-shift)
            device_assets = assets_here.to(device=asset_returns.device)
            device_sources = sources.to(device=asset_returns.device)
            transformed[position, batch_index, device_assets] = asset_returns[
                position, batch_index, device_sources
            ]
            source_index[position, batch_index, assets_here] = sources
    fixed = ~ordinary_return_valid
    if not torch.equal(transformed.masked_select(fixed), asset_returns.masked_select(fixed)):
        raise AssertionError("N_xs modified a fixed outcome")
    receipt = _receipt(
        kind="N_xs",
        seed=seed,
        source_axis_id=source_axis_id,
        randomization_axis_id=randomization_axis_id,
        domains=domains,
        input_outcomes=asset_returns,
        output_outcomes=transformed,
        ordinary_return_valid=ordinary_return_valid,
        mandatory_return_mask=mandatory_return_mask,
        destination_active=destination_active,
        source_index=source_index,
    )
    return Hold30NullView(transformed, source_index, receipt)


def _validate_event_sourced_mask(
    name: str,
    mask: torch.Tensor,
    known_at_ms: torch.Tensor,
) -> None:
    if mask.shape[0] < 2:
        return
    changes = mask[1:].detach().to(device="cpu") != mask[:-1].detach().to(device="cpu")
    known = known_at_ms.detach().to(device="cpu", dtype=torch.int64)
    if bool((changes & (known[1:] <= known[:-1])).any()):
        raise Hold30DatasetError(
            f"{name} changed without a strictly newer point-in-time event receipt"
        )


@dataclass(frozen=True, slots=True)
class Hold30DatasetSequence:
    """One fully materialized pre-lockbox Hold-30 chronology.

    ``n_positions`` includes a final state-only terminal observation.  One-step
    returns and C1 net returns therefore have ``n_positions - 1`` rows.  The
    C1 weights are the realizable books at each position and may not allocate
    to a name that was unknown at the preceding decision or illegal at its
    fill.
    """

    decision_timestamps_ms: torch.Tensor
    fill_timestamps_ms: torch.Tensor
    asset_ids: tuple[str, ...]
    decision_state: torch.Tensor
    decision_membership: torch.Tensor
    decision_tradability: torch.Tensor
    fill_membership: torch.Tensor
    fill_tradability: torch.Tensor
    asset_returns: torch.Tensor
    ordinary_return_valid: torch.Tensor
    mandatory_return_mask: torch.Tensor
    c1_benchmark_weights: torch.Tensor
    c1_benchmark_net_returns: torch.Tensor
    risk_asset_caps: torch.Tensor
    risk_gross_max: torch.Tensor
    cost_rate: torch.Tensor
    asof_evidence: Hold30AsOfEvidence
    provenance: Hold30PointInTimeProvenance
    cash_index: int = 0
    roles: Hold30DatasetRoles = field(init=False, repr=False, compare=False)
    decision_trade: torch.Tensor = field(init=False, repr=False, compare=False)
    fill_trade: torch.Tensor = field(init=False, repr=False, compare=False)
    a_trade: torch.Tensor = field(init=False, repr=False, compare=False)
    randomization_axis_id: str = field(init=False, repr=False, compare=False)
    axis_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_state, torch.Tensor) or self.decision_state.ndim < 4:
            raise Hold30DatasetError(
                "decision_state must have shape [position, batch, asset, ...]"
            )
        positions, batch, assets = self.decision_state.shape[:3]
        object.__setattr__(self, "roles", Hold30DatasetRoles.materialize(positions))
        if self.cash_index != 0:
            raise Hold30DatasetError("Hold-30 fixes synthetic CASH at asset index 0")
        if (
            not isinstance(self.asset_ids, tuple)
            or len(self.asset_ids) != assets
            or any(
                not isinstance(asset_id, str)
                or not asset_id
                or asset_id != asset_id.strip()
                for asset_id in self.asset_ids
            )
            or len(set(self.asset_ids)) != assets
        ):
            raise Hold30DatasetError(
                "asset_ids must be an ordered tuple of unique, non-empty stable IDs"
            )
        if self.asset_ids[self.cash_index] != HOLD30_CASH_ASSET_ID:
            raise Hold30DatasetError("asset_ids[0] must be CASH")
        if not isinstance(self.provenance, Hold30PointInTimeProvenance):
            raise Hold30DatasetError("complete Hold30PointInTimeProvenance is required")
        if not isinstance(self.asof_evidence, Hold30AsOfEvidence):
            raise Hold30DatasetError("complete Hold30AsOfEvidence is required")

        decision_ts = _as_cpu_int64(
            "decision_timestamps_ms", self.decision_timestamps_ms, (positions,)
        )
        fill_ts = _as_cpu_int64("fill_timestamps_ms", self.fill_timestamps_ms, (positions,))
        if bool((decision_ts < 0).any()) or bool((fill_ts < 0).any()):
            raise Hold30DatasetError("decision and fill timestamps must be nonnegative")
        if positions > 1 and bool((decision_ts[1:] <= decision_ts[:-1]).any()):
            raise Hold30DatasetError("decision timestamps must be strictly increasing and unique")
        if int(decision_ts[-1]) >= HOLD30_PRELOCKBOX_CUTOFF_MS:
            cutoff = datetime.fromtimestamp(
                HOLD30_PRELOCKBOX_CUTOFF_MS / 1000, tz=timezone.utc
            ).isoformat()
            raise Hold30DatasetError(f"every decision position must precede {cutoff}")
        if int(fill_ts[0]) > int(decision_ts[0]):
            raise Hold30DatasetError("initial fill timestamp cannot follow the first decision state")
        if positions > 1 and (
            bool((fill_ts[1:] <= decision_ts[:-1]).any())
            or bool((fill_ts[1:] > decision_ts[1:]).any())
        ):
            raise Hold30DatasetError(
                "fill[t+1] must occur strictly after decision[t] and no later than decision[t+1]"
            )

        position_shape = (positions, batch, assets)
        return_shape = (positions - 1, batch, assets)
        for name in (
            "decision_membership",
            "decision_tradability",
            "fill_membership",
            "fill_tradability",
        ):
            value = getattr(self, name)
            _require_tensor_shape(name, value, position_shape, dtype=torch.bool)
            if value.device != self.decision_state.device:
                raise Hold30DatasetError(f"{name} must share decision_state device")
            if not bool(value[..., self.cash_index].all()):
                raise Hold30DatasetError(f"CASH must always be true in {name}")
        decision_trade = self.decision_membership & self.decision_tradability
        fill_trade = self.fill_membership & self.fill_tradability
        object.__setattr__(self, "decision_trade", decision_trade)
        object.__setattr__(self, "fill_trade", fill_trade)
        object.__setattr__(self, "a_trade", decision_trade[:-1] & fill_trade[1:])
        if not self.decision_state.is_floating_point() or not bool(
            torch.isfinite(self.decision_state).all()
        ):
            raise Hold30DatasetError("decision_state must be finite and floating point")

        _require_floating("asset_returns", self.asset_returns, return_shape)
        for name in ("ordinary_return_valid", "mandatory_return_mask"):
            value = getattr(self, name)
            _require_tensor_shape(name, value, return_shape, dtype=torch.bool)
            if value.device != self.asset_returns.device:
                raise Hold30DatasetError(f"{name} must share asset_returns device")
        if bool(self.ordinary_return_valid[..., self.cash_index].any()) or bool(
            self.mandatory_return_mask[..., self.cash_index].any()
        ):
            raise Hold30DatasetError("CASH cannot be ordinary-transformable or mandatory")
        if bool((self.ordinary_return_valid & self.mandatory_return_mask).any()):
            raise Hold30DatasetError("ordinary and mandatory outcome masks must be disjoint")
        risky_returns = torch.ones_like(self.ordinary_return_valid, dtype=torch.bool)
        risky_returns[..., self.cash_index] = False
        expected_ordinary = (
            self.fill_membership[:-1] & risky_returns & ~self.mandatory_return_mask
        )
        if not torch.equal(self.ordinary_return_valid, expected_ordinary):
            raise Hold30DatasetError(
                "ordinary_return_valid must equal fill-active, risky, nonmandatory outcomes"
            )
        _require_floating(
            "c1_benchmark_weights", self.c1_benchmark_weights, position_shape
        )
        _require_floating(
            "c1_benchmark_net_returns", self.c1_benchmark_net_returns, (positions - 1, batch)
        )
        _require_floating("risk_asset_caps", self.risk_asset_caps, position_shape)
        _require_floating("risk_gross_max", self.risk_gross_max, (positions, batch))
        _require_floating("cost_rate", self.cost_rate, (positions - 1, batch))
        reference = self.asset_returns
        for name in (
            "c1_benchmark_weights",
            "c1_benchmark_net_returns",
            "risk_asset_caps",
            "risk_gross_max",
            "cost_rate",
        ):
            value = getattr(self, name)
            if value.device != reference.device or value.dtype != reference.dtype:
                raise Hold30DatasetError(
                    f"{name} must share asset_returns dtype and device"
                )
        if self.decision_state.device != reference.device:
            raise Hold30DatasetError("decision_state and asset_returns must share a device")
        if bool((self.asset_returns <= -1).any()) or bool(
            (self.c1_benchmark_net_returns <= -1).any()
        ):
            raise Hold30DatasetError("policy and C1 simple returns must be greater than -1")
        if bool((self.cost_rate < 0).any()):
            raise Hold30DatasetError("cost_rate cannot be negative")
        if bool((self.risk_asset_caps < 0).any()) or bool(
            (self.risk_asset_caps > 1).any()
        ):
            raise Hold30DatasetError("risk_asset_caps must lie in [0, 1]")
        if bool((self.risk_gross_max < 0).any()) or bool((self.risk_gross_max > 1).any()):
            raise Hold30DatasetError("risk_gross_max must lie in [0, 1]")

        self.asof_evidence.validate(
            shape=position_shape,
            decision_timestamps_ms=decision_ts,
            fill_timestamps_ms=fill_ts,
            cash_index=self.cash_index,
        )
        _validate_event_sourced_mask(
            "decision_membership",
            self.decision_membership,
            self.asof_evidence.decision_membership_known_at_ms,
        )
        _validate_event_sourced_mask(
            "decision_tradability",
            self.decision_tradability,
            self.asof_evidence.decision_tradability_known_at_ms,
        )
        _validate_event_sourced_mask(
            "fill_membership",
            self.fill_membership,
            self.asof_evidence.fill_membership_known_at_ms,
        )
        _validate_event_sourced_mask(
            "fill_tradability",
            self.fill_tradability,
            self.asof_evidence.fill_tradability_known_at_ms,
        )

        totals = self.c1_benchmark_weights.sum(-1)
        if bool((self.c1_benchmark_weights < 0).any()) or not bool(
            torch.allclose(totals, torch.ones_like(totals), atol=1e-7, rtol=1e-7)
        ):
            raise Hold30DatasetError("C1 benchmark weights must be nonnegative simplexes")
        allowed_c1 = self._c1_allowed_mask()
        if bool((self.c1_benchmark_weights.masked_select(~allowed_c1).abs() > 1e-10).any()):
            raise Hold30DatasetError(
                "C1 allocated to an asset unavailable to the preceding decision/fill contract"
            )
        risky = torch.ones_like(self.c1_benchmark_weights, dtype=torch.bool)
        risky[..., self.cash_index] = False
        if bool(
            (
                self.c1_benchmark_weights.masked_fill(~risky, 0.0)
                > self.risk_asset_caps + 1e-8
            ).any()
        ):
            raise Hold30DatasetError("C1 benchmark exceeds a fill-time name cap")
        c1_gross = self.c1_benchmark_weights.masked_fill(~risky, 0.0).sum(-1)
        if bool((c1_gross > self.risk_gross_max + 1e-8).any()):
            raise Hold30DatasetError("C1 benchmark exceeds the fill-time gross ceiling")
        nontrade_risky = (~self.fill_trade) & risky
        if bool((self.risk_asset_caps.masked_select(nontrade_risky) != 0).any()):
            raise Hold30DatasetError(
                "fill-time risk name caps must be zero for non-tradeable risky assets"
            )

        randomization_axis_id = _canonical_digest(
            {
                "decision_timestamps_sha256": _tensor_digest(self.decision_timestamps_ms),
                "asset_ids": self.asset_ids,
                "decision_membership_sha256": _tensor_digest(self.decision_membership),
                "decision_tradability_sha256": _tensor_digest(self.decision_tradability),
                "cash_index": self.cash_index,
                "role_geometry": {
                    "warmup": HOLD30_WARMUP_POSITIONS,
                    "support": HOLD30_SUPPORT_POSITIONS,
                    "credit_returns": HOLD30_CREDIT_RETURNS,
                },
            }
        )
        object.__setattr__(self, "randomization_axis_id", randomization_axis_id)
        object.__setattr__(
            self,
            "axis_id",
            _canonical_digest(
                {
                    "contract": "hold30-point-in-time-sequence-v1",
                    "provenance_receipt_id": self.provenance.receipt_id,
                    "randomization_axis_id": randomization_axis_id,
                    "fill_timestamps_sha256": _tensor_digest(self.fill_timestamps_ms),
                    "fill_membership_sha256": _tensor_digest(self.fill_membership),
                    "fill_tradability_sha256": _tensor_digest(self.fill_tradability),
                }
            ),
        )

    @property
    def n_positions(self) -> int:
        return int(self.decision_state.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.decision_state.shape[1])

    @property
    def num_assets(self) -> int:
        return int(self.decision_state.shape[2])

    def _c1_allowed_mask(self) -> torch.Tensor:
        allowed = torch.empty_like(self.fill_trade)
        allowed[0] = self.decision_trade[0] & self.fill_trade[0]
        allowed[1:] = self.a_trade
        allowed[..., self.cash_index] = True
        return allowed

    def n_time(
        self,
        seed: int,
        *,
        domains: tuple[Hold30NullDomain, ...],
    ) -> Hold30NullView:
        return n_time_transform(
            self.asset_returns,
            self.ordinary_return_valid,
            self.mandatory_return_mask,
            self.fill_membership[:-1],
            domains=domains,
            seed=seed,
            source_axis_id=self.axis_id,
            randomization_axis_id=self.randomization_axis_id,
            cash_index=self.cash_index,
        )

    def n_xs(
        self,
        seed: int,
        *,
        domains: tuple[Hold30NullDomain, ...],
    ) -> Hold30NullView:
        return n_xs_transform(
            self.asset_returns,
            self.ordinary_return_valid,
            self.mandatory_return_mask,
            self.fill_membership[:-1],
            domains=domains,
            seed=seed,
            source_axis_id=self.axis_id,
            randomization_axis_id=self.randomization_axis_id,
            cash_index=self.cash_index,
        )

    def runtime_kwargs(
        self,
        *,
        initial_ledger: Any,
        initial_equity: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return exact keyword names consumed by ``Hold30Sequence``.

        The caller owns construction of the cohort ledger so this lower-layer
        module never imports :mod:`rl_quant.envs`.  Outcome-null views are not
        accepted here: callers must rebuild C1, labels, drift, and provenance,
        then validate a complete transformed sequence before asking for runtime
        arguments.
        """
        return {
            "decision_state": self.decision_state,
            "asset_returns": self.asset_returns,
            "decision_available": self.decision_trade,
            "fill_membership": self.fill_membership,
            "fill_availability": self.fill_tradability,
            "benchmark_weights": self.c1_benchmark_weights,
            "risk_asset_caps": self.risk_asset_caps,
            "risk_gross_max": self.risk_gross_max,
            "benchmark_net_returns": self.c1_benchmark_net_returns,
            "initial_ledger": initial_ledger,
            "cost_rate": self.cost_rate,
            "initial_equity": initial_equity,
            # Retention/survival telemetry is defined only for loss-bearing
            # score origins. Warm-up establishes state and support completes
            # credit, but neither population may create tracked entry units.
            "track_entry_units": self.roles.score[:-1].clone(),
            "axis_id": self.axis_id,
        }


__all__ = [
    "HOLD30_BENCHMARK_ID",
    "HOLD30_CASH_ASSET_ID",
    "HOLD30_CASH_RETURN_RULE",
    "HOLD30_PRELOCKBOX_CUTOFF_MS",
    "HOLD30_UNIVERSE_MODE",
    "Hold30AsOfEvidence",
    "Hold30DatasetError",
    "Hold30DatasetRoles",
    "Hold30DatasetSequence",
    "Hold30NullDomain",
    "Hold30NullReceipt",
    "Hold30NullView",
    "Hold30PointInTimeProvenance",
    "n_time_transform",
    "n_xs_transform",
]
