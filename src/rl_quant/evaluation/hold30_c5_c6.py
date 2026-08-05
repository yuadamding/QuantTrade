"""Sealed C5 supervised-ranker and C6 empirical-intent diagnostics.

This module implements only the second bounded control tranche.  It does not
construct C7/C8 banks or calculate statistical endpoints.  V2 closes the two
choices intentionally left open by v1: C5 schedule keys derive from one
manifest-bound 32-byte control root and C6 moves intents only among the exact
63 outer-score rows.  Every derived schedule is deterministic, outcome-blind,
self-describing, and content addressed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
from torch import nn

from rl_quant.datasets.hold30 import Hold30DatasetSequence
from rl_quant.datasets.hold30_folds import Hold30DevelopmentFold
from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.evaluation.hold30_controls import (
    HOLD30_PRIMARY_COST_BPS,
    Hold30ControlGrossTrace,
    assemble_hold30_control_trace,
)
from rl_quant.evaluation.hold30_ensemble_runtime import (
    EnsemblePolicy,
    EnsembleStateProvider,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.protocol.hold30 import (
    HOLD30_MECH8_SETTINGS,
    HOLD30_PROTOCOL_GENERATION,
    resolve_hold30_setting,
)
from rl_quant.protocol.hold30_freeze import (
    HOLD30_FOLDS,
    HOLD30_SCORE_DAYS,
    HOLD30_SEEDS,
)
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30DecisionStateProvider,
    Hold30Sequence,
    Hold30Transition,
)

HOLD30_C5_HORIZON = 30
HOLD30_C5_PAIRS_PER_DATE = 1_024
HOLD30_C5_DATES_PER_UPDATE = 16
HOLD30_C5_MICROBATCH_DATES = 4
HOLD30_C5_MAX_UPDATES = 128
HOLD30_C5_VALIDATION_CADENCE = 8
HOLD30_C5_VALIDATION_PATIENCE = 4
HOLD30_C5_MIN_SELECTION_UPDATE = 32
HOLD30_C5_SELECTION_TOLERANCE = 1e-4
HOLD30_C5_LEARNING_RATE = 3e-4
HOLD30_C5_WEIGHT_DECAY = 1e-4
HOLD30_C5_ADAM_EPS = 1e-5
HOLD30_C5_GRAD_CLIP = 0.5
HOLD30_C6_REPLICATES = 64
HOLD30_SCHEDULE_ENCODING = "sha256-key-plus-domain-plus-u64be-v1"
HOLD30_C5_CONTROL_KEY_ENCODING = (
    "sha256(root32-plus-len16be-utf8-protocol-plus-u16be-fold-plus-"
    "len16be-utf8-purpose)-v1"
)
HOLD30_C5_PAIR_PURPOSE = "c5-pair-schedule"
HOLD30_C5_DATE_PURPOSE = "c5-date-schedule"
HOLD30_C6_OUTER_SCORE_DOMAIN = "outer-score-63"

_INTENT_FIELDS = (
    "entry_scores",
    "target_logits",
    "gate",
    "hazard_residual",
    "exposure_residual",
)
_CAUSES = tuple(TurnoverCause)


class Hold30C5C6Error(ValueError):
    """A C5/C6 artifact violates the frozen scientific contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30C5C6Error(f"{name} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def c5_model_state_sha256(module: nn.Module) -> str:
    """Content-address a C5 model state without pickle byte instability."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    return _payload_sha256(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(module.state_dict().items())
        }
    )


def _hash_u64(
    key_sha256: str,
    domain: str,
    *values: int,
) -> bytes:
    _require_digest("schedule key", key_sha256)
    if not isinstance(domain, str) or not domain or domain != domain.strip():
        raise Hold30C5C6Error("hash domain must be a non-empty stripped string")
    encoded_domain = domain.encode("utf-8")
    if len(encoded_domain) > 65_535:
        raise Hold30C5C6Error("hash domain is too long")
    material = bytearray(bytes.fromhex(key_sha256))
    material.extend(len(encoded_domain).to_bytes(2, "big"))
    material.extend(encoded_domain)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Hold30C5C6Error("schedule hash values must be unsigned integers")
        material.extend(value.to_bytes(8, "big", signed=False))
    return hashlib.sha256(material).digest()


def _derive_control_key(
    root_seed_hex: str,
    protocol_generation: str,
    fold_index: int,
    purpose: str,
) -> str:
    """Derive one schedule key with an unambiguous byte encoding."""

    _require_digest("control_root_seed_hex", root_seed_hex)
    if protocol_generation != HOLD30_PROTOCOL_GENERATION:
        raise Hold30C5C6Error("C5 control root belongs to another protocol")
    if isinstance(fold_index, bool) or fold_index not in range(HOLD30_FOLDS):
        raise Hold30C5C6Error("C5 schedule fold_index must be in [0,5]")
    if purpose not in {HOLD30_C5_PAIR_PURPOSE, HOLD30_C5_DATE_PURPOSE}:
        raise Hold30C5C6Error("unknown C5 control schedule purpose")
    protocol = protocol_generation.encode("utf-8")
    encoded_purpose = purpose.encode("utf-8")
    material = bytearray(bytes.fromhex(root_seed_hex))
    material.extend(len(protocol).to_bytes(2, "big"))
    material.extend(protocol)
    material.extend(fold_index.to_bytes(2, "big", signed=False))
    material.extend(len(encoded_purpose).to_bytes(2, "big"))
    material.extend(encoded_purpose)
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class Hold30C5ScheduleKeyBinding:
    """Manifest-bound root and its two purpose-separated C5 schedule keys."""

    control_root_seed_hex: str
    executable_manifest_sha256: str
    fold_index: int
    pair_key_sha256: str
    date_key_sha256: str
    protocol_generation: str = HOLD30_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
        _require_digest("control_root_seed_hex", self.control_root_seed_hex)
        _require_digest("executable_manifest_sha256", self.executable_manifest_sha256)
        _require_digest("pair_key_sha256", self.pair_key_sha256)
        _require_digest("date_key_sha256", self.date_key_sha256)
        expected_pair = _derive_control_key(
            self.control_root_seed_hex,
            self.protocol_generation,
            self.fold_index,
            HOLD30_C5_PAIR_PURPOSE,
        )
        expected_date = _derive_control_key(
            self.control_root_seed_hex,
            self.protocol_generation,
            self.fold_index,
            HOLD30_C5_DATE_PURPOSE,
        )
        if (
            self.pair_key_sha256 != expected_pair
            or self.date_key_sha256 != expected_date
        ):
            raise Hold30C5C6Error("C5 schedule keys do not derive from the frozen root")

    @property
    def pair_domain(self) -> str:
        return (
            f"{self.protocol_generation}/fold-{self.fold_index}/"
            f"{HOLD30_C5_PAIR_PURPOSE}"
        )

    @property
    def date_domain(self) -> str:
        return (
            f"{self.protocol_generation}/fold-{self.fold_index}/"
            f"{HOLD30_C5_DATE_PURPOSE}"
        )

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c5-schedule-key-binding",
            "schema_version": 1,
            "protocol_generation": self.protocol_generation,
            "fold_index": self.fold_index,
            "control_root_seed_hex": self.control_root_seed_hex,
            "executable_manifest_sha256": self.executable_manifest_sha256,
            "derivation_encoding": HOLD30_C5_CONTROL_KEY_ENCODING,
            "pair_purpose": HOLD30_C5_PAIR_PURPOSE,
            "pair_key_sha256": self.pair_key_sha256,
            "pair_domain": self.pair_domain,
            "date_purpose": HOLD30_C5_DATE_PURPOSE,
            "date_key_sha256": self.date_key_sha256,
            "date_domain": self.date_domain,
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.receipt_payload)


def derive_c5_schedule_key_binding(
    *,
    control_root_seed_hex: str,
    executable_manifest_sha256: str,
    fold_index: int,
) -> Hold30C5ScheduleKeyBinding:
    """Derive both C5 keys from one explicit 32-byte manifest-bound root."""

    return Hold30C5ScheduleKeyBinding(
        control_root_seed_hex=control_root_seed_hex,
        executable_manifest_sha256=executable_manifest_sha256,
        fold_index=fold_index,
        pair_key_sha256=_derive_control_key(
            control_root_seed_hex,
            HOLD30_PROTOCOL_GENERATION,
            fold_index,
            HOLD30_C5_PAIR_PURPOSE,
        ),
        date_key_sha256=_derive_control_key(
            control_root_seed_hex,
            HOLD30_PROTOCOL_GENERATION,
            fold_index,
            HOLD30_C5_DATE_PURPOSE,
        ),
    )


def _first_score_row(sequence: Hold30DatasetSequence) -> int:
    rows = torch.where(sequence.roles.score[:-1].to(device="cpu"))[0]
    if rows.numel() == 0:
        raise Hold30C5C6Error("sequence has no score-bearing decision row")
    return int(rows[0])


@dataclass(frozen=True, slots=True)
class Hold30C5LabelSet:
    """Exact 30-session post-fill excess-log-return labels."""

    source_axis_id: str
    asset_ids: tuple[str, ...]
    values: torch.Tensor
    valid: torch.Tensor
    censored: torch.Tensor
    score_rows: torch.Tensor
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_digest("source_axis_id", self.source_axis_id)
        _require_digest("receipt_sha256", self.receipt_sha256)
        if (
            not isinstance(self.asset_ids, tuple)
            or len(self.asset_ids) != self.values.shape[-1]
            or len(set(self.asset_ids)) != len(self.asset_ids)
            or self.asset_ids[0] != "CASH"
        ):
            raise Hold30C5C6Error("C5 labels require the stable CASH-first asset axis")
        if self.values.ndim != 3 or self.values.dtype != torch.float64:
            raise Hold30C5C6Error(
                "C5 label values must be float64 [decision,batch,asset]"
            )
        if not bool(torch.isfinite(self.values).all()):
            raise Hold30C5C6Error("C5 labels must be finite")
        if (
            self.valid.shape != self.values.shape
            or self.valid.dtype != torch.bool
            or self.censored.shape != self.values.shape
            or self.censored.dtype != torch.bool
        ):
            raise Hold30C5C6Error("C5 valid/censored masks must match the label tensor")
        if self.score_rows.dtype != torch.bool or tuple(self.score_rows.shape) != (
            self.values.shape[0],
        ):
            raise Hold30C5C6Error("score_rows must be boolean [decision]")
        if bool((self.valid & self.censored).any()):
            raise Hold30C5C6Error("C5 valid and censored masks overlap")
        if bool(self.valid[~self.score_rows].any()) or bool(
            self.censored[~self.score_rows].any()
        ):
            raise Hold30C5C6Error("C5 labels escaped the permitted score role")
        if bool((self.values.masked_select(~self.valid) != 0).any()):
            raise Hold30C5C6Error("invalid C5 labels must be stored as exact zero")
        if _payload_sha256(self.receipt_payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C5 label receipt self-hash mismatch")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c5-labels",
            "schema_version": 1,
            "source_axis_id": self.source_axis_id,
            "asset_ids": list(self.asset_ids),
            "horizon_returns": HOLD30_C5_HORIZON,
            "first_return_offset_after_decision": 1,
            "forced_exit_destination": "frozen_cash_return_series",
            "score_rows_sha256": _tensor_sha256(self.score_rows),
            "values_sha256": _tensor_sha256(self.values),
            "valid_sha256": _tensor_sha256(self.valid),
            "censored_sha256": _tensor_sha256(self.censored),
            "outer_access": False,
        }

    @property
    def labels_sha256(self) -> str:
        return _payload_sha256(
            {
                "source_axis_id": self.source_axis_id,
                "asset_ids": list(self.asset_ids),
                "values_sha256": _tensor_sha256(self.values),
                "valid_sha256": _tensor_sha256(self.valid),
                "censored_sha256": _tensor_sha256(self.censored),
                "score_rows_sha256": _tensor_sha256(self.score_rows),
            }
        )


def build_c5_labels(sequence: Hold30DatasetSequence) -> Hold30C5LabelSet:
    """Build labels only for this role-exact development sequence.

    A scored decision ``t`` fills at ``t+1``.  Its stock path therefore uses
    return rows ``t+1..t+30``.  A forced exit earns its inbound stock outcome
    before moving to the frozen CASH series for every later row.
    """

    if not isinstance(sequence, Hold30DatasetSequence):
        raise TypeError("sequence must be Hold30DatasetSequence")
    if sequence.asset_returns.dtype != torch.float64:
        raise Hold30C5C6Error("sealed C5 labels require float64 returns")
    if not torch.equal(sequence.cost_rate, torch.full_like(sequence.cost_rate, 0.002)):
        raise Hold30C5C6Error("C5 labels require the exact primary 20-bp C1 trace")
    rows, batch, assets = sequence.asset_returns.shape
    values = sequence.asset_returns.new_zeros((rows, batch, assets))
    valid = torch.zeros((rows, batch, assets), dtype=torch.bool, device=values.device)
    censored = torch.zeros_like(valid)
    score_rows = sequence.roles.score[:-1].to(device=values.device).clone()
    risky = torch.ones((batch, assets), dtype=torch.bool, device=values.device)
    risky[:, sequence.cash_index] = False

    for origin in torch.where(score_rows.to(device="cpu"))[0].tolist():
        eligible = sequence.a_trade[origin] & risky
        support_terminal = origin + HOLD30_C5_HORIZON + 1
        if support_terminal >= sequence.n_positions:
            censored[origin] = eligible
            continue
        alive = eligible.clone()
        stock_log = values.new_zeros((batch, assets))
        for return_row in range(origin + 1, origin + HOLD30_C5_HORIZON + 1):
            cash = sequence.asset_returns[return_row, :, sequence.cash_index].unsqueeze(
                -1
            )
            realized = torch.where(alive, sequence.asset_returns[return_row], cash)
            stock_log = stock_log + torch.log1p(realized)
            next_fill = return_row + 1
            alive = (
                alive
                & sequence.fill_membership[next_fill]
                & sequence.fill_tradability[next_fill]
            )
        benchmark_log = torch.log1p(
            sequence.c1_benchmark_net_returns[
                origin + 1 : origin + HOLD30_C5_HORIZON + 1
            ]
        ).sum(dim=0)
        values[origin] = torch.where(
            eligible,
            stock_log - benchmark_log.unsqueeze(-1),
            torch.zeros_like(stock_log),
        )
        valid[origin] = eligible

    if bool(censored.any()):
        raise Hold30C5C6Error(
            "a C5 score row lacks complete 31-position support inside its permitted split"
        )
    payload = {
        "schema": "rl-quant.hold30.c5-labels",
        "schema_version": 1,
        "source_axis_id": sequence.axis_id,
        "asset_ids": list(sequence.asset_ids),
        "horizon_returns": HOLD30_C5_HORIZON,
        "first_return_offset_after_decision": 1,
        "forced_exit_destination": "frozen_cash_return_series",
        "score_rows_sha256": _tensor_sha256(score_rows),
        "values_sha256": _tensor_sha256(values),
        "valid_sha256": _tensor_sha256(valid),
        "censored_sha256": _tensor_sha256(censored),
        "outer_access": False,
    }
    return Hold30C5LabelSet(
        source_axis_id=sequence.axis_id,
        asset_ids=sequence.asset_ids,
        values=values,
        valid=valid,
        censored=censored,
        score_rows=score_rows,
        receipt_sha256=_payload_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class Hold30C5PairSchedule:
    """One outcome-blind set of 1,024 unordered pairs per date/batch."""

    source_axis_id: str
    asset_ids: tuple[str, ...]
    date_rows: tuple[int, ...]
    pairs: torch.Tensor
    key_binding: Hold30C5ScheduleKeyBinding
    hash_key_sha256: str
    hash_domain: str
    decision_trade_sha256: str
    fill_trade_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_digest("source_axis_id", self.source_axis_id)
        _require_digest("hash_key_sha256", self.hash_key_sha256)
        _require_digest("decision_trade_sha256", self.decision_trade_sha256)
        _require_digest("fill_trade_sha256", self.fill_trade_sha256)
        _require_digest("receipt_sha256", self.receipt_sha256)
        if not isinstance(self.key_binding, Hold30C5ScheduleKeyBinding):
            raise TypeError("key_binding must be Hold30C5ScheduleKeyBinding")
        if (
            self.hash_key_sha256 != self.key_binding.pair_key_sha256
            or self.hash_domain != self.key_binding.pair_domain
        ):
            raise Hold30C5C6Error(
                "pair schedule key/domain escaped its control-root binding"
            )
        if not isinstance(self.hash_domain, str) or not self.hash_domain:
            raise Hold30C5C6Error("pair schedule hash_domain is required")
        if tuple(sorted(set(self.date_rows))) != self.date_rows:
            raise Hold30C5C6Error(
                "pair schedule date rows must be unique and increasing"
            )
        if (
            self.pairs.dtype != torch.int64
            or self.pairs.ndim != 4
            or self.pairs.shape[0] != len(self.date_rows)
            or self.pairs.shape[2:] != (HOLD30_C5_PAIRS_PER_DATE, 2)
        ):
            raise Hold30C5C6Error("pair schedule must be int64 [date,batch,1024,2]")
        if bool((self.pairs < 0).any()) or bool(
            (self.pairs >= len(self.asset_ids)).any()
        ):
            raise Hold30C5C6Error("pair schedule contains an out-of-axis asset")
        if bool((self.pairs[..., 0] == self.pairs[..., 1]).any()):
            raise Hold30C5C6Error("C5 pairs must contain two distinct assets")
        for date in range(self.pairs.shape[0]):
            for batch in range(self.pairs.shape[1]):
                rows = self.pairs[date, batch].detach().to(device="cpu")
                pair_values = tuple(
                    tuple(int(value) for value in row) for row in rows.tolist()
                )
                if len(set(pair_values)) != HOLD30_C5_PAIRS_PER_DATE:
                    raise Hold30C5C6Error(
                        "C5 pair schedule contains duplicate unordered pairs"
                    )
                if any(
                    self.asset_ids[left] >= self.asset_ids[right]
                    for left, right in pair_values
                ):
                    raise Hold30C5C6Error("C5 pairs must use stable-asset-ID order")
        if _payload_sha256(self.receipt_payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C5 pair-schedule receipt self-hash mismatch")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c5-pair-schedule",
            "schema_version": 1,
            "source_axis_id": self.source_axis_id,
            "asset_ids": list(self.asset_ids),
            "date_rows": list(self.date_rows),
            "pairs_per_date_batch": HOLD30_C5_PAIRS_PER_DATE,
            "eligibility_rule": "decision_and_legal_tplus1_fill_trade_without_outcomes",
            "hash_encoding": HOLD30_SCHEDULE_ENCODING,
            "key_binding_sha256": self.key_binding.receipt_sha256,
            "hash_key_sha256": self.hash_key_sha256,
            "hash_domain": self.hash_domain,
            "decision_trade_sha256": self.decision_trade_sha256,
            "fill_trade_sha256": self.fill_trade_sha256,
            "pairs_sha256": _tensor_sha256(self.pairs),
            "outcomes_read": False,
            "outer_access": False,
        }

    @property
    def pairs_sha256(self) -> str:
        return _tensor_sha256(self.pairs)

    def pairs_for(self, rows: Sequence[int]) -> torch.Tensor:
        lookup = {row: index for index, row in enumerate(self.date_rows)}
        try:
            indexes = [lookup[int(row)] for row in rows]
        except KeyError as exc:
            raise Hold30C5C6Error(
                "date schedule references a row without frozen pairs"
            ) from exc
        return self.pairs[indexes]


def _pair_rank_to_assets(
    eligible: Sequence[int],
    rank: int,
) -> tuple[int, int]:
    remaining = rank
    count = len(eligible)
    for left_index in range(count - 1):
        width = count - left_index - 1
        if remaining < width:
            return eligible[left_index], eligible[left_index + 1 + remaining]
        remaining -= width
    raise AssertionError("pair rank escaped the combination space")


def materialize_c5_pair_schedule(
    sequence: Hold30DatasetSequence,
    *,
    date_rows: Iterable[int],
    key_binding: Hold30C5ScheduleKeyBinding,
) -> Hold30C5PairSchedule:
    """Derive fixed pairs from legal decision/fill masks without reading returns."""

    if not isinstance(key_binding, Hold30C5ScheduleKeyBinding):
        raise TypeError("key_binding must be Hold30C5ScheduleKeyBinding")
    hash_key_sha256 = key_binding.pair_key_sha256
    hash_domain = key_binding.pair_domain
    rows = tuple(date_rows)
    if tuple(sorted(set(rows))) != rows:
        raise Hold30C5C6Error("date_rows must be unique and increasing")
    score = sequence.roles.score[:-1].to(device="cpu")
    if any(
        row < 0 or row >= sequence.n_positions - 1 or not bool(score[row])
        for row in rows
    ):
        raise Hold30C5C6Error("pair schedules are restricted to permitted score rows")
    pairs = torch.empty(
        (len(rows), sequence.batch_size, HOLD30_C5_PAIRS_PER_DATE, 2),
        dtype=torch.int64,
    )
    for date_index, row in enumerate(rows):
        timestamp = int(sequence.decision_timestamps_ms[row])
        for batch in range(sequence.batch_size):
            eligible = [
                asset
                for asset in range(sequence.num_assets)
                if asset != sequence.cash_index
                and bool(sequence.a_trade[row, batch, asset])
            ]
            eligible.sort(key=lambda asset: sequence.asset_ids[asset])
            pair_count = len(eligible) * (len(eligible) - 1) // 2
            if pair_count < HOLD30_C5_PAIRS_PER_DATE:
                raise Hold30C5C6Error(
                    "a C5 date/batch has fewer than 1,024 eligible unordered pairs"
                )
            selected: list[int] = []
            selected_set: set[int] = set()
            counter = 0
            while len(selected) < HOLD30_C5_PAIRS_PER_DATE:
                candidate = (
                    int.from_bytes(
                        _hash_u64(
                            hash_key_sha256,
                            hash_domain,
                            row,
                            timestamp,
                            batch,
                            counter,
                        ),
                        "big",
                    )
                    % pair_count
                )
                counter += 1
                if candidate in selected_set:
                    continue
                selected_set.add(candidate)
                selected.append(candidate)
            pairs[date_index, batch] = torch.tensor(
                [_pair_rank_to_assets(eligible, rank) for rank in selected],
                dtype=torch.int64,
            )
    payload = {
        "schema": "rl-quant.hold30.c5-pair-schedule",
        "schema_version": 1,
        "source_axis_id": sequence.axis_id,
        "asset_ids": list(sequence.asset_ids),
        "date_rows": list(rows),
        "pairs_per_date_batch": HOLD30_C5_PAIRS_PER_DATE,
        "eligibility_rule": "decision_and_legal_tplus1_fill_trade_without_outcomes",
        "hash_encoding": HOLD30_SCHEDULE_ENCODING,
        "key_binding_sha256": key_binding.receipt_sha256,
        "hash_key_sha256": hash_key_sha256,
        "hash_domain": hash_domain,
        "decision_trade_sha256": _tensor_sha256(sequence.decision_trade),
        "fill_trade_sha256": _tensor_sha256(sequence.fill_trade),
        "pairs_sha256": _tensor_sha256(pairs),
        "outcomes_read": False,
        "outer_access": False,
    }
    return Hold30C5PairSchedule(
        source_axis_id=sequence.axis_id,
        asset_ids=sequence.asset_ids,
        date_rows=rows,
        pairs=pairs,
        key_binding=key_binding,
        hash_key_sha256=hash_key_sha256,
        hash_domain=hash_domain,
        decision_trade_sha256=payload["decision_trade_sha256"],
        fill_trade_sha256=payload["fill_trade_sha256"],
        receipt_sha256=_payload_sha256(payload),
    )


def verify_c5_pair_schedule(
    sequence: Hold30DatasetSequence,
    schedule: Hold30C5PairSchedule,
) -> None:
    """Re-derive a pair schedule and reject altered tensors or receipts."""

    expected = materialize_c5_pair_schedule(
        sequence,
        date_rows=schedule.date_rows,
        key_binding=schedule.key_binding,
    )
    if (
        schedule.source_axis_id != expected.source_axis_id
        or schedule.asset_ids != expected.asset_ids
        or schedule.receipt_sha256 != expected.receipt_sha256
        or not torch.equal(schedule.pairs, expected.pairs)
    ):
        raise Hold30C5C6Error("C5 pair schedule failed deterministic reconstruction")


@dataclass(frozen=True, slots=True)
class Hold30C5DateSchedule:
    """Exactly 128 hash-derived, distinct-16-date optimizer batches."""

    source_axis_id: str
    permitted_rows: tuple[int, ...]
    update_rows: torch.Tensor
    cycle_ids: torch.Tensor
    completed_cycles: torch.Tensor
    key_binding: Hold30C5ScheduleKeyBinding
    hash_key_sha256: str
    hash_domain: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_digest("source_axis_id", self.source_axis_id)
        _require_digest("hash_key_sha256", self.hash_key_sha256)
        _require_digest("receipt_sha256", self.receipt_sha256)
        if not isinstance(self.key_binding, Hold30C5ScheduleKeyBinding):
            raise TypeError("key_binding must be Hold30C5ScheduleKeyBinding")
        if (
            self.hash_key_sha256 != self.key_binding.date_key_sha256
            or self.hash_domain != self.key_binding.date_domain
        ):
            raise Hold30C5C6Error(
                "date schedule key/domain escaped its control-root binding"
            )
        expected = (HOLD30_C5_MAX_UPDATES, HOLD30_C5_DATES_PER_UPDATE)
        if (
            self.update_rows.dtype != torch.int64
            or tuple(self.update_rows.shape) != expected
        ):
            raise Hold30C5C6Error("C5 update rows must be int64 [128,16]")
        if (
            self.cycle_ids.dtype != torch.int64
            or tuple(self.cycle_ids.shape) != expected
        ):
            raise Hold30C5C6Error("C5 cycle IDs must be int64 [128,16]")
        if self.completed_cycles.dtype != torch.int64 or tuple(
            self.completed_cycles.shape
        ) != (HOLD30_C5_MAX_UPDATES,):
            raise Hold30C5C6Error("completed_cycles must be int64 [128]")
        permitted = set(self.permitted_rows)
        if tuple(sorted(permitted)) != self.permitted_rows or len(permitted) < 16:
            raise Hold30C5C6Error(
                "C5 needs at least 16 unique permitted training dates"
            )
        if any(
            int(value) not in permitted for value in self.update_rows.flatten().tolist()
        ):
            raise Hold30C5C6Error("date schedule escaped the permitted training rows")
        if any(len(set(row.tolist())) != 16 for row in self.update_rows):
            raise Hold30C5C6Error("every C5 update must use exactly 16 distinct dates")
        if bool((self.completed_cycles[1:] < self.completed_cycles[:-1]).any()):
            raise Hold30C5C6Error("completed cycle count cannot decrease")
        if _payload_sha256(self.receipt_payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C5 date-schedule receipt self-hash mismatch")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c5-date-schedule",
            "schema_version": 1,
            "source_axis_id": self.source_axis_id,
            "permitted_rows": list(self.permitted_rows),
            "updates": HOLD30_C5_MAX_UPDATES,
            "dates_per_update": HOLD30_C5_DATES_PER_UPDATE,
            "microbatches": 4,
            "dates_per_microbatch": HOLD30_C5_MICROBATCH_DATES,
            "cycle_rule": "hash_ordered_permutation_no_repeat_until_exhausted",
            "cross_cycle_update_rule": "stable_skip_current_update_duplicates",
            "hash_encoding": HOLD30_SCHEDULE_ENCODING,
            "key_binding_sha256": self.key_binding.receipt_sha256,
            "hash_key_sha256": self.hash_key_sha256,
            "hash_domain": self.hash_domain,
            "update_rows_sha256": _tensor_sha256(self.update_rows),
            "cycle_ids_sha256": _tensor_sha256(self.cycle_ids),
            "completed_cycles_sha256": _tensor_sha256(self.completed_cycles),
            "outcomes_read": False,
            "outer_access": False,
        }

    def rows_for_update(self, update: int) -> tuple[int, ...]:
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or not 1 <= update <= 128
        ):
            raise Hold30C5C6Error("C5 update must be in [1,128]")
        return tuple(int(value) for value in self.update_rows[update - 1].tolist())


def materialize_c5_date_schedule(
    sequence: Hold30DatasetSequence,
    *,
    permitted_rows: Iterable[int],
    key_binding: Hold30C5ScheduleKeyBinding,
) -> Hold30C5DateSchedule:
    """Materialize cyclic date permutations without reading outcomes."""

    if not isinstance(key_binding, Hold30C5ScheduleKeyBinding):
        raise TypeError("key_binding must be Hold30C5ScheduleKeyBinding")
    hash_key_sha256 = key_binding.date_key_sha256
    hash_domain = key_binding.date_domain
    permitted = tuple(permitted_rows)
    score = sequence.roles.score[:-1].to(device="cpu")
    if (
        tuple(sorted(set(permitted))) != permitted
        or len(permitted) < HOLD30_C5_DATES_PER_UPDATE
        or any(
            row < 0 or row >= sequence.n_positions - 1 or not bool(score[row])
            for row in permitted
        )
    ):
        raise Hold30C5C6Error(
            "permitted_rows must be at least 16 unique increasing training score rows"
        )
    update_rows = torch.empty(
        (HOLD30_C5_MAX_UPDATES, HOLD30_C5_DATES_PER_UPDATE),
        dtype=torch.int64,
    )
    cycle_ids = torch.empty_like(update_rows)
    completed = torch.empty((HOLD30_C5_MAX_UPDATES,), dtype=torch.int64)
    remaining: list[int] = []
    cycle = -1
    completed_count = 0
    for update in range(HOLD30_C5_MAX_UPDATES):
        selected: list[int] = []
        selected_cycles: list[int] = []
        while len(selected) < HOLD30_C5_DATES_PER_UPDATE:
            if not remaining:
                cycle += 1
                remaining = sorted(
                    permitted,
                    key=lambda row: (
                        _hash_u64(
                            hash_key_sha256,
                            hash_domain,
                            cycle,
                            row,
                            int(sequence.decision_timestamps_ms[row]),
                        ),
                        row,
                    ),
                )
            index = next(
                (
                    position
                    for position, row in enumerate(remaining)
                    if row not in selected
                ),
                None,
            )
            if index is None:
                raise AssertionError(
                    "at least sixteen permitted dates must avoid an update duplicate"
                )
            selected.append(remaining.pop(index))
            selected_cycles.append(cycle)
            if not remaining:
                completed_count += 1
        update_rows[update] = torch.tensor(selected, dtype=torch.int64)
        cycle_ids[update] = torch.tensor(selected_cycles, dtype=torch.int64)
        completed[update] = completed_count
    payload = {
        "schema": "rl-quant.hold30.c5-date-schedule",
        "schema_version": 1,
        "source_axis_id": sequence.axis_id,
        "permitted_rows": list(permitted),
        "updates": HOLD30_C5_MAX_UPDATES,
        "dates_per_update": HOLD30_C5_DATES_PER_UPDATE,
        "microbatches": 4,
        "dates_per_microbatch": HOLD30_C5_MICROBATCH_DATES,
        "cycle_rule": "hash_ordered_permutation_no_repeat_until_exhausted",
        "cross_cycle_update_rule": "stable_skip_current_update_duplicates",
        "hash_encoding": HOLD30_SCHEDULE_ENCODING,
        "key_binding_sha256": key_binding.receipt_sha256,
        "hash_key_sha256": hash_key_sha256,
        "hash_domain": hash_domain,
        "update_rows_sha256": _tensor_sha256(update_rows),
        "cycle_ids_sha256": _tensor_sha256(cycle_ids),
        "completed_cycles_sha256": _tensor_sha256(completed),
        "outcomes_read": False,
        "outer_access": False,
    }
    return Hold30C5DateSchedule(
        source_axis_id=sequence.axis_id,
        permitted_rows=permitted,
        update_rows=update_rows,
        cycle_ids=cycle_ids,
        completed_cycles=completed,
        key_binding=key_binding,
        hash_key_sha256=hash_key_sha256,
        hash_domain=hash_domain,
        receipt_sha256=_payload_sha256(payload),
    )


def verify_c5_date_schedule(
    sequence: Hold30DatasetSequence,
    schedule: Hold30C5DateSchedule,
) -> None:
    """Re-derive a date schedule and reject altered cycles or receipts."""

    expected = materialize_c5_date_schedule(
        sequence,
        permitted_rows=schedule.permitted_rows,
        key_binding=schedule.key_binding,
    )
    if (
        schedule.source_axis_id != expected.source_axis_id
        or schedule.receipt_sha256 != expected.receipt_sha256
        or not torch.equal(schedule.update_rows, expected.update_rows)
        or not torch.equal(schedule.cycle_ids, expected.cycle_ids)
        or not torch.equal(schedule.completed_cycles, expected.completed_cycles)
    ):
        raise Hold30C5C6Error("C5 date schedule failed deterministic reconstruction")


class Hold30C5TrainableStateProvider(Protocol):
    trains_upstream_encoder: bool

    def replay_origin_states(
        self,
        policy: nn.Module,
        sequence: Hold30Sequence,
        origins: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class Hold30C5FitBinding:
    """Outer-free proof that an optimizer sees only one development training axis."""

    fold_index: int
    development_receipt_sha256: str
    fold_sha256: str
    training_axis_id: str
    inner_validation_axis_id: str
    training_absolute_range: tuple[int, int]
    validation_absolute_range: tuple[int, int]
    outer_absolute_range: tuple[int, int]
    receipt_sha256: str
    role: str = "expanding_training"
    outer_access: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or self.fold_index not in range(6):
            raise Hold30C5C6Error("C5 fit binding fold_index must be in [0,5]")
        for name in (
            "development_receipt_sha256",
            "fold_sha256",
            "training_axis_id",
            "inner_validation_axis_id",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        ranges = (
            self.training_absolute_range,
            self.validation_absolute_range,
            self.outer_absolute_range,
        )
        if any(
            len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            )
            or value[0] < 0
            or value[0] >= value[1]
            for value in ranges
        ):
            raise Hold30C5C6Error("C5 fit binding contains a malformed absolute range")
        if self.role != "expanding_training" or self.outer_access is not False:
            raise Hold30C5C6Error(
                "C5 fitting is restricted to outer-free expanding training"
            )
        payload = {
            "schema": "rl-quant.hold30.c5-fit-binding",
            "schema_version": 1,
            "fold_index": self.fold_index,
            "development_receipt_sha256": self.development_receipt_sha256,
            "fold_sha256": self.fold_sha256,
            "training_axis_id": self.training_axis_id,
            "inner_validation_axis_id": self.inner_validation_axis_id,
            "training_absolute_range": list(self.training_absolute_range),
            "validation_absolute_range": list(self.validation_absolute_range),
            "outer_absolute_range": list(self.outer_absolute_range),
            "role": self.role,
            "outer_access": self.outer_access,
        }
        if _payload_sha256(payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C5 fit binding self-hash mismatch")


def bind_c5_development_fold(development: Hold30DevelopmentFold) -> Hold30C5FitBinding:
    """Create a training-only binding without materializing an outer tensor."""

    if not isinstance(development, Hold30DevelopmentFold):
        raise TypeError("development must be Hold30DevelopmentFold")
    payload = {
        "schema": "rl-quant.hold30.c5-fit-binding",
        "schema_version": 1,
        "fold_index": development.fold.fold_index,
        "development_receipt_sha256": development.receipt_sha256,
        "fold_sha256": development.fold_sha256,
        "training_axis_id": development.training.axis_id,
        "inner_validation_axis_id": development.inner_validation.axis_id,
        "training_absolute_range": list(development.training_absolute_range),
        "validation_absolute_range": list(development.validation_absolute_range),
        "outer_absolute_range": list(development.outer_absolute_range),
        "role": "expanding_training",
        "outer_access": False,
    }
    return Hold30C5FitBinding(
        fold_index=development.fold.fold_index,
        development_receipt_sha256=development.receipt_sha256,
        fold_sha256=development.fold_sha256,
        training_axis_id=development.training.axis_id,
        inner_validation_axis_id=development.inner_validation.axis_id,
        training_absolute_range=development.training_absolute_range,
        validation_absolute_range=development.validation_absolute_range,
        outer_absolute_range=development.outer_absolute_range,
        receipt_sha256=_payload_sha256(payload),
    )


def _c5_entry_scores(
    policy: nn.Module,
    state: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only the market-state entry head used by C5.

    Production ``DailyCrossSectionPolicy`` objects expose this path through
    ``hold30_intent``; its registered H2 implementation zeros the holdings
    feature before the entry head.  Small qualification doubles may expose a
    narrower ``c5_entry_scores(state, available)`` API.
    """

    if (
        state.ndim != 4
        or available.dtype != torch.bool
        or available.shape != state.shape[:3]
    ):
        raise Hold30C5C6Error(
            "C5 state/availability must be [date,batch,asset,feature] and boolean prefix"
        )
    dates, batch, assets, feature = state.shape
    flat_state = state.reshape(dates * batch, assets, feature)
    flat_available = available.reshape(dates * batch, assets)
    direct = getattr(policy, "c5_entry_scores", None)
    if callable(direct):
        value = direct(flat_state, flat_available)
    else:
        switches = getattr(policy, "hold30_switches", None)
        if getattr(switches, "mechanism", None) != "H2":
            raise Hold30C5C6Error(
                "production C5 requires a registered H2 market-only entry head"
            )
        if getattr(getattr(policy, "config", None), "hold30_setting", None) != (
            "hold30-m02-age-hazard"
        ):
            raise Hold30C5C6Error(
                "production C5 must use the canonical H2 architecture"
            )
        method = getattr(policy, "hold30_intent", None)
        if not callable(method):
            raise Hold30C5C6Error("C5 policy lacks a market-only entry-score API")
        age_dim = int(getattr(getattr(policy, "config", None), "age_summary_dim", 5))
        intent = method(
            flat_state,
            torch.zeros_like(flat_available, dtype=flat_state.dtype),
            flat_available,
            flat_state.new_zeros((dates * batch, assets, age_dim)),
        )
        value = intent.entry_scores
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (dates * batch, assets)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise Hold30C5C6Error("C5 entry scores must be finite [date*batch,asset]")
    return value.reshape(dates, batch, assets)


def _pairwise_loss_sum(
    scores: torch.Tensor,
    labels: Hold30C5LabelSet,
    rows: Sequence[int],
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if scores.ndim != 3:
        raise Hold30C5C6Error("C5 scores must have shape [date,batch,asset]")
    if tuple(pairs.shape) != (
        len(rows),
        scores.shape[1],
        HOLD30_C5_PAIRS_PER_DATE,
        2,
    ):
        raise Hold30C5C6Error("C5 pair tensor does not match the score minibatch")
    row_index = torch.as_tensor(rows, dtype=torch.int64, device=labels.values.device)
    label_values = labels.values.index_select(0, row_index)
    label_valid = labels.valid.index_select(0, row_index)
    pair_device = pairs.to(device=scores.device)
    left = pair_device[..., 0]
    right = pair_device[..., 1]
    left_score = scores.gather(-1, left)
    right_score = scores.gather(-1, right)
    label_values = label_values.to(device=scores.device)
    label_valid = label_valid.to(device=scores.device)
    left_label = label_values.gather(-1, left)
    right_label = label_values.gather(-1, right)
    valid = label_valid.gather(-1, left) & label_valid.gather(-1, right)
    difference = left_label - right_label
    # An exact label tie has no sign and is therefore an invalid pair; it is
    # never replaced after outcomes are known.
    valid &= difference != 0
    signed_margin = torch.sign(difference).to(dtype=scores.dtype) * (
        left_score - right_score
    )
    terms = torch.nn.functional.softplus(-signed_margin)
    count = int(valid.sum().detach().to(device="cpu"))
    return terms.masked_select(valid).sum(), count


def build_c5_optimizer(policy: nn.Module) -> torch.optim.AdamW:
    """Construct the exact frozen C5 AdamW optimizer."""

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch module")
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise Hold30C5C6Error("C5 policy has no trainable parameter")
    return torch.optim.AdamW(
        parameters,
        lr=HOLD30_C5_LEARNING_RATE,
        weight_decay=HOLD30_C5_WEIGHT_DECAY,
        eps=HOLD30_C5_ADAM_EPS,
    )


def _validate_c5_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise Hold30C5C6Error("C5 requires torch.optim.AdamW")
    for group in optimizer.param_groups:
        expected = {
            "lr": HOLD30_C5_LEARNING_RATE,
            "weight_decay": HOLD30_C5_WEIGHT_DECAY,
            "eps": HOLD30_C5_ADAM_EPS,
        }
        if any(
            float(group.get(name, float("nan"))) != value
            for name, value in expected.items()
        ):
            raise Hold30C5C6Error(
                "C5 AdamW hyperparameters differ from the frozen contract"
            )


def _provider_binding_sha256(provider: object) -> str:
    binding = getattr(provider, "binding_config", None)
    if callable(binding):
        binding = binding()
    if not isinstance(binding, Mapping):
        raise Hold30C5C6Error(
            "C5 state provider requires a receipt-bound binding_config"
        )
    material = dict(binding)
    claimed = material.pop("binding_sha256", None)
    _require_digest("state-provider binding_sha256", claimed)
    if _payload_sha256(material) != claimed:
        raise Hold30C5C6Error("C5 state-provider binding self-hash mismatch")
    return claimed


@dataclass(frozen=True, slots=True)
class Hold30C5UpdateResult:
    update: int
    mean_pair_loss: float
    valid_pair_count: int
    gradient_norm_before_clip: float
    date_rows: tuple[int, ...]
    completed_cycles: int
    model_state_sha256: str
    receipt_sha256: str


def train_c5_update(
    policy: nn.Module,
    state_provider: Hold30C5TrainableStateProvider,
    sequence: Hold30Sequence,
    labels: Hold30C5LabelSet,
    pair_schedule: Hold30C5PairSchedule,
    date_schedule: Hold30C5DateSchedule,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    fit_binding: Hold30C5FitBinding,
) -> Hold30C5UpdateResult:
    """Run one 16-date/four-microbatch C5 optimizer update."""

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch module")
    if not isinstance(fit_binding, Hold30C5FitBinding):
        raise TypeError("fit_binding must be Hold30C5FitBinding")
    if sequence.axis_id != fit_binding.training_axis_id:
        raise Hold30C5C6Error("C5 optimizer received validation or outer data")
    if sequence.n_positions != (
        fit_binding.training_absolute_range[1] - fit_binding.training_absolute_range[0]
    ):
        raise Hold30C5C6Error(
            "C5 training sequence length differs from its development binding"
        )
    if getattr(state_provider, "trains_upstream_encoder", None) is not True:
        raise Hold30C5C6Error(
            "C5 training requires a differentiable provider that trains the common encoder"
        )
    if not callable(getattr(state_provider, "replay_origin_states", None)):
        raise Hold30C5C6Error("C5 state provider lacks replay_origin_states")
    if sequence.axis_id != labels.source_axis_id:
        raise Hold30C5C6Error("C5 labels belong to a different training sequence")
    if (
        pair_schedule.source_axis_id != sequence.axis_id
        or date_schedule.source_axis_id != sequence.axis_id
    ):
        raise Hold30C5C6Error("C5 schedules belong to a different training sequence")
    if (
        pair_schedule.key_binding.receipt_sha256
        != date_schedule.key_binding.receipt_sha256
    ):
        raise Hold30C5C6Error("C5 pair/date schedules do not share one control root")
    _validate_c5_optimizer(optimizer)
    rows = date_schedule.rows_for_update(update)
    pair_rows = pair_schedule.pairs_for(rows)

    # Determine the global denominator before building any autograd graph so
    # every microbatch contributes total-valid-pair loss / total valid pairs.
    zero_scores = labels.values.new_zeros(
        (len(rows), sequence.batch_size, sequence.num_assets)
    )
    _, total_valid = _pairwise_loss_sum(zero_scores, labels, rows, pair_rows)
    if total_valid <= 0:
        raise Hold30C5C6Error("C5 update has no valid scheduled pair")

    policy.train(True)
    optimizer.zero_grad(set_to_none=True)
    detached_loss_sum = 0.0
    observed_valid = 0
    for start in range(0, HOLD30_C5_DATES_PER_UPDATE, HOLD30_C5_MICROBATCH_DATES):
        micro_rows = rows[start : start + HOLD30_C5_MICROBATCH_DATES]
        origin_tensor = torch.tensor(micro_rows, dtype=torch.int64)
        states = state_provider.replay_origin_states(policy, sequence, origin_tensor)
        expected_prefix = (
            HOLD30_C5_MICROBATCH_DATES,
            sequence.batch_size,
            sequence.num_assets,
        )
        if (
            not isinstance(states, torch.Tensor)
            or tuple(states.shape[:3]) != expected_prefix
        ):
            raise Hold30C5C6Error(
                "C5 provider must return [four_dates,batch,asset,feature]"
            )
        available = sequence.decision_available[
            torch.as_tensor(
                micro_rows, dtype=torch.int64, device=sequence.decision_available.device
            )
        ]
        scores = _c5_entry_scores(policy, states, available)
        loss_sum, valid_count = _pairwise_loss_sum(
            scores,
            labels,
            micro_rows,
            pair_rows[start : start + HOLD30_C5_MICROBATCH_DATES],
        )
        (loss_sum / total_valid).backward()
        detached_loss_sum += float(loss_sum.detach().to(device="cpu"))
        observed_valid += valid_count
    if observed_valid != total_valid:
        raise AssertionError("C5 microbatch valid-pair accounting changed")
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, HOLD30_C5_GRAD_CLIP)
    if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
        raise Hold30C5C6Error("C5 gradient norm is non-finite")
    optimizer.step()
    model_sha = c5_model_state_sha256(policy)
    payload = {
        "schema": "rl-quant.hold30.c5-training-update",
        "schema_version": 1,
        "source_axis_id": sequence.axis_id,
        "update": update,
        "date_rows": list(rows),
        "unique_dates": len(set(rows)),
        "completed_cycles": int(date_schedule.completed_cycles[update - 1]),
        "valid_pair_count": total_valid,
        "mean_pair_loss": detached_loss_sum / total_valid,
        "gradient_norm_before_clip": float(
            torch.as_tensor(gradient_norm).detach().cpu()
        ),
        "gradient_clip": HOLD30_C5_GRAD_CLIP,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": HOLD30_C5_LEARNING_RATE,
            "weight_decay": HOLD30_C5_WEIGHT_DECAY,
            "epsilon": HOLD30_C5_ADAM_EPS,
            "steps": 1,
        },
        "microbatches": 4,
        "dates_per_microbatch": HOLD30_C5_MICROBATCH_DATES,
        "labels_receipt_sha256": labels.receipt_sha256,
        "pair_schedule_receipt_sha256": pair_schedule.receipt_sha256,
        "date_schedule_receipt_sha256": date_schedule.receipt_sha256,
        "state_provider_binding_sha256": _provider_binding_sha256(state_provider),
        "fit_binding_sha256": fit_binding.receipt_sha256,
        "model_state_sha256": model_sha,
        "outer_access": False,
    }
    return Hold30C5UpdateResult(
        update=update,
        mean_pair_loss=detached_loss_sum / total_valid,
        valid_pair_count=total_valid,
        gradient_norm_before_clip=float(torch.as_tensor(gradient_norm).detach().cpu()),
        date_rows=rows,
        completed_cycles=int(date_schedule.completed_cycles[update - 1]),
        model_state_sha256=model_sha,
        receipt_sha256=_payload_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class Hold30C5CohortIdentity:
    fold_index: int
    executable_manifest_sha256: str
    development_receipt_sha256: str
    fold_sha256: str
    training_axis_id: str
    inner_validation_axis_id: str
    training_labels_sha256: str
    validation_labels_sha256: str
    control_schedule_binding_sha256: str
    stage1_normalization_receipt_sha256: str
    pair_schedule_sha256: str
    date_schedule_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.fold_index, bool) or self.fold_index not in range(6):
            raise Hold30C5C6Error("C5 fold_index must be in [0,5]")
        for name in self.__dataclass_fields__:
            if name != "fold_index":
                _require_digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Hold30C5CheckpointReference:
    seed: int
    update: int
    checkpoint_id: str
    model_state_sha256: str
    checkpoint_receipt_sha256: str
    stage1_normalization_receipt_sha256: str
    training_labels_sha256: str
    control_schedule_binding_sha256: str
    pair_schedule_sha256: str
    date_schedule_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed not in HOLD30_SEEDS:
            raise Hold30C5C6Error(f"C5 seed must be one of {HOLD30_SEEDS}")
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or not 0 <= self.update <= HOLD30_C5_MAX_UPDATES
            or (self.update and self.update % HOLD30_C5_VALIDATION_CADENCE)
        ):
            raise Hold30C5C6Error(
                "C5 checkpoint update violates the validation cadence"
            )
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise Hold30C5C6Error("C5 checkpoint_id must be non-empty")
        for name in (
            "model_state_sha256",
            "checkpoint_receipt_sha256",
            "stage1_normalization_receipt_sha256",
            "training_labels_sha256",
            "control_schedule_binding_sha256",
            "pair_schedule_sha256",
            "date_schedule_sha256",
        ):
            _require_digest(name, getattr(self, name))


def _ordered_c5_refs(
    references: Sequence[Hold30C5CheckpointReference],
    *,
    update: int,
    identity: Hold30C5CohortIdentity,
) -> tuple[Hold30C5CheckpointReference, ...]:
    refs = tuple(references)
    if len(refs) != 5 or tuple(reference.seed for reference in refs) != HOLD30_SEEDS:
        raise Hold30C5C6Error(
            "C5 requires exactly five references in frozen seed order"
        )
    if any(reference.update != update for reference in refs):
        raise Hold30C5C6Error("all C5 seed checkpoints must share one update")
    if len({reference.checkpoint_id for reference in refs}) != 5:
        raise Hold30C5C6Error("C5 checkpoint IDs must be unique")
    for reference in refs:
        if (
            reference.training_labels_sha256 != identity.training_labels_sha256
            or reference.stage1_normalization_receipt_sha256
            != identity.stage1_normalization_receipt_sha256
            or reference.control_schedule_binding_sha256
            != identity.control_schedule_binding_sha256
            or reference.pair_schedule_sha256 != identity.pair_schedule_sha256
            or reference.date_schedule_sha256 != identity.date_schedule_sha256
        ):
            raise Hold30C5C6Error(
                "C5 checkpoint does not bind the shared training evidence"
            )
    if len({reference.stage1_normalization_receipt_sha256 for reference in refs}) != 1:
        raise Hold30C5C6Error(
            "all C5 seeds must bind the common fold Stage-1 normalization receipt"
        )
    return refs


@dataclass(frozen=True, slots=True)
class Hold30C5ValidationScore:
    update: int
    active_log_wealth: float
    discretionary_turnover: float
    trace_sha256: str
    inner_validation_axis_id: str
    validation_labels_sha256: str
    role: str = "inner_validation"
    cost_bps: int = HOLD30_PRIMARY_COST_BPS
    continuing_wealth: bool = True
    outer_access: bool = False
    ensemble_member_count: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
            or self.update > HOLD30_C5_MAX_UPDATES
            or self.update % HOLD30_C5_VALIDATION_CADENCE
        ):
            raise Hold30C5C6Error("C5 validation update violates the frozen cadence")
        for name in ("active_log_wealth", "discretionary_turnover"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise Hold30C5C6Error(f"{name} must be finite")
        if self.discretionary_turnover < 0:
            raise Hold30C5C6Error("C5 validation turnover cannot be negative")
        for name in (
            "trace_sha256",
            "inner_validation_axis_id",
            "validation_labels_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.role != "inner_validation"
            or self.cost_bps != HOLD30_PRIMARY_COST_BPS
            or self.continuing_wealth is not True
            or self.outer_access is not False
            or self.ensemble_member_count != 5
        ):
            raise Hold30C5C6Error(
                "C5 selection accepts only five-member continuing 20-bp inner validation"
            )


@dataclass(frozen=True, slots=True)
class Hold30C5ValidationRecord:
    score: Hold30C5ValidationScore
    checkpoints: tuple[Hold30C5CheckpointReference, ...]

    @property
    def update(self) -> int:
        return self.score.update


def _validate_c5_records(
    identity: Hold30C5CohortIdentity,
    records: Sequence[Hold30C5ValidationRecord],
) -> tuple[Hold30C5ValidationRecord, ...]:
    rows = tuple(records)
    if not rows:
        raise Hold30C5C6Error("C5 has no validation evidence")
    expected = tuple(
        range(
            HOLD30_C5_VALIDATION_CADENCE,
            rows[-1].update + 1,
            HOLD30_C5_VALIDATION_CADENCE,
        )
    )
    if tuple(row.update for row in rows) != expected:
        raise Hold30C5C6Error("C5 validations must form an exact cadence prefix")
    seen_traces: set[str] = set()
    for row in rows:
        _ordered_c5_refs(row.checkpoints, update=row.update, identity=identity)
        if (
            row.score.inner_validation_axis_id != identity.inner_validation_axis_id
            or row.score.validation_labels_sha256 != identity.validation_labels_sha256
        ):
            raise Hold30C5C6Error("C5 validation used unbound or outer evidence")
        if row.score.trace_sha256 in seen_traces:
            raise Hold30C5C6Error("C5 validation trace digests must be update-specific")
        seen_traces.add(row.score.trace_sha256)
    return rows


def select_c5_shared_checkpoint(
    identity: Hold30C5CohortIdentity,
    records: Sequence[Hold30C5ValidationRecord],
) -> Hold30C5ValidationRecord:
    """Select one update for all seeds; never search checkpoint combinations."""

    rows = _validate_c5_records(identity, records)
    eligible = tuple(
        row for row in rows if row.update >= HOLD30_C5_MIN_SELECTION_UPDATE
    )
    if not eligible:
        raise Hold30C5C6Error("C5 selection requires the common 32-update minimum")
    maximum = max(float(row.score.active_log_wealth) for row in eligible)
    within = tuple(
        row
        for row in eligible
        if float(row.score.active_log_wealth) >= maximum - HOLD30_C5_SELECTION_TOLERANCE
    )
    return min(
        within,
        key=lambda row: (
            row.update,
            float(row.score.discretionary_turnover),
            tuple(reference.checkpoint_id for reference in row.checkpoints),
        ),
    )


@dataclass(frozen=True, slots=True)
class Hold30C5SelectionOutcome:
    identity: Hold30C5CohortIdentity
    initial_checkpoints: tuple[Hold30C5CheckpointReference, ...]
    validations: tuple[Hold30C5ValidationRecord, ...]
    selected: Hold30C5ValidationRecord
    stopped_update: int
    stop_reason: str

    def __post_init__(self) -> None:
        _ordered_c5_refs(self.initial_checkpoints, update=0, identity=self.identity)
        rows = _validate_c5_records(self.identity, self.validations)
        if self.selected not in rows or self.selected != select_c5_shared_checkpoint(
            self.identity, rows
        ):
            raise Hold30C5C6Error("C5 selected checkpoint violates the shared rule")
        if self.stopped_update != rows[-1].update:
            raise Hold30C5C6Error("C5 stopped update differs from final validation")
        expected = _c5_stop_reason(rows, terminal=True)
        if self.stop_reason != expected:
            raise Hold30C5C6Error("C5 stop reason violates patience/ceiling")

    def receipt(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "rl-quant.hold30.c5-seed-cohort-selection",
            "schema_version": 1,
            "identity": asdict(self.identity),
            "initial_checkpoints": [
                asdict(value) for value in self.initial_checkpoints
            ],
            "validations": [
                {
                    "score": asdict(row.score),
                    "checkpoints": [asdict(value) for value in row.checkpoints],
                }
                for row in self.validations
            ],
            "selected_update": self.selected.update,
            "selected_checkpoints": [
                asdict(value) for value in self.selected.checkpoints
            ],
            "stopped_update": self.stopped_update,
            "stop_reason": self.stop_reason,
            "selection_rule": {
                "validation_cadence": HOLD30_C5_VALIDATION_CADENCE,
                "minimum_update": HOLD30_C5_MIN_SELECTION_UPDATE,
                "maximum_updates": HOLD30_C5_MAX_UPDATES,
                "patience_validations": HOLD30_C5_VALIDATION_PATIENCE,
                "active_log_wealth_tolerance": HOLD30_C5_SELECTION_TOLERANCE,
                "priority": ["earliest_update", "lower_turnover", "checkpoint_ids"],
            },
            "outer_access": False,
            "checkpoint_selection_complete": True,
        }
        payload["receipt_sha256"] = _payload_sha256(payload)
        return payload


def _c5_stop_reason(
    records: Sequence[Hold30C5ValidationRecord],
    *,
    terminal: bool,
) -> str | None:
    best = -math.inf
    stale = 0
    rows = tuple(records)
    for index, row in enumerate(rows):
        value = float(row.score.active_log_wealth)
        if value > best:
            best = value
            stale = 0
        else:
            stale += 1
        if (
            row.update >= HOLD30_C5_MIN_SELECTION_UPDATE
            and stale >= HOLD30_C5_VALIDATION_PATIENCE
        ):
            if terminal and index != len(rows) - 1:
                raise Hold30C5C6Error(
                    "C5 validation continued after patience was exhausted"
                )
            return "validation_patience_exhausted" if index == len(rows) - 1 else None
    if rows and rows[-1].update == HOLD30_C5_MAX_UPDATES:
        return "maximum_updates"
    if terminal:
        raise Hold30C5C6Error(
            "C5 validation prefix is an interruption, not a terminal outcome"
        )
    return None


AdvanceC5 = Callable[[int], Sequence[Hold30C5CheckpointReference]]
ValidateC5 = Callable[
    [int, tuple[Hold30C5CheckpointReference, ...]], Hold30C5ValidationScore
]


def coordinate_c5_seed_cohort(
    identity: Hold30C5CohortIdentity,
    initial_checkpoints: Sequence[Hold30C5CheckpointReference],
    *,
    advance_cohort: AdvanceC5,
    validate_ensemble: ValidateC5,
) -> Hold30C5SelectionOutcome:
    """Advance and stop all five C5 seeds synchronously."""

    initial = _ordered_c5_refs(initial_checkpoints, update=0, identity=identity)
    records: list[Hold30C5ValidationRecord] = []
    for update in range(
        HOLD30_C5_VALIDATION_CADENCE,
        HOLD30_C5_MAX_UPDATES + 1,
        HOLD30_C5_VALIDATION_CADENCE,
    ):
        refs = _ordered_c5_refs(
            advance_cohort(update), update=update, identity=identity
        )
        score = validate_ensemble(update, refs)
        if not isinstance(score, Hold30C5ValidationScore) or score.update != update:
            raise Hold30C5C6Error(
                "C5 validation callback changed the synchronized update"
            )
        record = Hold30C5ValidationRecord(score, refs)
        records.append(record)
        reason = _c5_stop_reason(records, terminal=False)
        if reason is not None:
            return Hold30C5SelectionOutcome(
                identity=identity,
                initial_checkpoints=initial,
                validations=tuple(records),
                selected=select_c5_shared_checkpoint(identity, records),
                stopped_update=update,
                stop_reason=reason,
            )
    raise AssertionError("C5 coordinator must stop at the frozen maximum")


def verify_c5_selection_receipt(
    receipt: Mapping[str, Any],
) -> Hold30C5SelectionOutcome:
    """Reject altered, extended, or internally inconsistent C5 receipts."""

    if not isinstance(receipt, Mapping):
        raise Hold30C5C6Error("C5 selection receipt must be a mapping")
    expected_fields = {
        "schema",
        "schema_version",
        "identity",
        "initial_checkpoints",
        "validations",
        "selected_update",
        "selected_checkpoints",
        "stopped_update",
        "stop_reason",
        "selection_rule",
        "outer_access",
        "checkpoint_selection_complete",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise Hold30C5C6Error("C5 selection receipt has missing or unknown fields")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    _require_digest("receipt_sha256", claimed)
    if _payload_sha256(unsigned) != claimed:
        raise Hold30C5C6Error("C5 selection receipt self-hash mismatch")
    if (
        receipt["schema"] != "rl-quant.hold30.c5-seed-cohort-selection"
        or receipt["schema_version"] != 1
        or receipt["outer_access"] is not False
        or receipt["checkpoint_selection_complete"] is not True
    ):
        raise Hold30C5C6Error("C5 selection receipt metadata is invalid")
    try:
        identity = Hold30C5CohortIdentity(**receipt["identity"])
        initial = tuple(
            Hold30C5CheckpointReference(**value)
            for value in receipt["initial_checkpoints"]
        )
        validations = tuple(
            Hold30C5ValidationRecord(
                Hold30C5ValidationScore(**row["score"]),
                tuple(
                    Hold30C5CheckpointReference(**value) for value in row["checkpoints"]
                ),
            )
            for row in receipt["validations"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Hold30C5C6Error("C5 selection receipt payload is malformed") from exc
    selected = next(
        (row for row in validations if row.update == receipt["selected_update"]),
        None,
    )
    if (
        selected is None
        or [asdict(value) for value in selected.checkpoints]
        != receipt["selected_checkpoints"]
    ):
        raise Hold30C5C6Error("C5 selected checkpoint payload is inconsistent")
    outcome = Hold30C5SelectionOutcome(
        identity=identity,
        initial_checkpoints=initial,
        validations=validations,
        selected=selected,
        stopped_update=receipt["stopped_update"],
        stop_reason=receipt["stop_reason"],
    )
    if outcome.receipt() != dict(receipt):
        raise Hold30C5C6Error("C5 selection receipt rule metadata is inconsistent")
    return outcome


def _runtime_sequence(
    sequence: Hold30DatasetSequence,
    *,
    decision_state: torch.Tensor | None = None,
) -> Hold30Sequence:
    if not bool(
        torch.equal(sequence.cost_rate, torch.full_like(sequence.cost_rate, 0.002))
    ):
        raise Hold30C5C6Error("C5/C6 canonical chronology requires exact 20-bp costs")
    initial = CohortLedger.from_staggered_endowment(
        sequence.c1_benchmark_weights[0],
        cash_index=sequence.cash_index,
        youngest_age=0,
        oldest_age=29,
        track_initial_units=False,
    )
    state = sequence.decision_state if decision_state is None else decision_state
    return Hold30Sequence(
        decision_state=state,
        asset_returns=sequence.asset_returns,
        decision_available=sequence.decision_trade,
        fill_membership=sequence.fill_membership,
        fill_availability=sequence.fill_tradability,
        benchmark_weights=sequence.c1_benchmark_weights,
        risk_asset_caps=sequence.risk_asset_caps,
        risk_gross_max=sequence.risk_gross_max,
        benchmark_net_returns=sequence.c1_benchmark_net_returns,
        initial_ledger=initial,
        cost_rate=HOLD30_PRIMARY_COST_BPS / 10_000.0,
        track_entry_units=sequence.roles.score[:-1],
        axis_id=sequence.axis_id,
    )


def _trace_from_transitions(
    control_id: str,
    sequence: Hold30DatasetSequence,
    runtime_sequence: Hold30Sequence,
    transitions: Sequence[Hold30Transition],
    *,
    outer_start: int,
    fitting_rows: Iterable[int],
    source_receipt_sha256: str,
    strategy_inputs_sha256: str,
) -> Hold30ControlGrossTrace:
    rows = sequence.n_positions - 1
    values = tuple(transitions)
    if len(values) != rows or tuple(item.decision_index for item in values) != tuple(
        range(rows)
    ):
        raise Hold30C5C6Error(
            "control runtime did not produce the complete ordered chronology"
        )
    initial = runtime_sequence.initial_ledger.weights
    weights = torch.stack((initial, *(item.post_cost_weights for item in values)))
    pretrade = torch.stack(tuple(item.execution_pretrade_weights for item in values))
    gross = torch.stack(tuple(item.holding_return for item in values))
    zero = torch.zeros_like(pretrade)
    deltas = {
        TurnoverCause.STARTUP: zero,
        TurnoverCause.MEMBERSHIP_FORCED: torch.stack(
            tuple(
                item.membership_repaired_weights - item.execution_pretrade_weights
                for item in values
            )
        ),
        TurnoverCause.AVAILABILITY_FORCED: torch.stack(
            tuple(
                item.availability_repaired_weights - item.membership_repaired_weights
                for item in values
            )
        ),
        TurnoverCause.RISK_FORCED: torch.stack(
            tuple(
                item.risk_repaired_weights - item.availability_repaired_weights
                for item in values
            )
        ),
        TurnoverCause.DISCRETIONARY: torch.stack(
            tuple(item.pre_cost_weights - item.risk_repaired_weights for item in values)
        ),
        TurnoverCause.TERMINAL: zero,
    }
    return assemble_hold30_control_trace(
        control_id,
        sequence,
        weights=weights.to(dtype=torch.float64),
        pretrade_weights=pretrade.to(dtype=torch.float64),
        gross_returns=gross.to(dtype=torch.float64),
        deltas={
            cause: value.to(dtype=torch.float64) for cause, value in deltas.items()
        },
        score_mask=sequence.roles.score[:-1].to(device=sequence.asset_returns.device),
        outer_start=outer_start,
        fitting_rows=fitting_rows,
        source_receipt_sha256=source_receipt_sha256,
        strategy_inputs_sha256=strategy_inputs_sha256,
    )


class _FrozenC5Member(nn.Module):
    """Expose a trained C5 entry head as an H2 intent with zero residuals."""

    def __init__(self, member: nn.Module) -> None:
        super().__init__()
        self.member = member

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del prev_weights, age_summaries
        entry = _c5_entry_scores(
            self.member,
            state_t.unsqueeze(0),
            available.unsqueeze(0),
        )[0]
        return Hold30Intent(
            entry_scores=entry,
            hazard_residual=torch.zeros_like(entry),
            exposure_residual=entry.new_zeros((entry.shape[0],)),
        )


class _C5MemberStateProvider:
    """Bind an original trainable model to its evaluation state provider."""

    trains_upstream_encoder = False

    def __init__(
        self, provider: Hold30DecisionStateProvider, member: nn.Module
    ) -> None:
        self.provider = provider
        self.member = member

    @property
    def binding_config(self) -> Mapping[str, Any]:
        value = getattr(self.provider, "binding_config", None)
        value = value() if callable(value) else value
        if not isinstance(value, Mapping):
            raise Hold30C5C6Error("C5 member provider lacks binding_config")
        return value

    @property
    def decision_available(self) -> torch.Tensor:
        value = getattr(self.provider, "decision_available", None)
        if value is None:
            value = getattr(getattr(self.provider, "inputs", None), "available", None)
        if not isinstance(value, torch.Tensor):
            raise Hold30C5C6Error("C5 member provider lacks decision availability")
        return value

    def canonical_states(
        self,
        _wrapped_policy: nn.Module,
        sequence: Hold30Sequence,
    ) -> torch.Tensor | Sequence[torch.Tensor]:
        return self.provider.canonical_states(self.member, sequence)

    def replay_origin_states(self, *_args: object) -> torch.Tensor:
        raise RuntimeError(
            "C5 frozen evaluation providers cannot enter a training replay"
        )


def construct_c5_control(
    sequence: Hold30DatasetSequence,
    member_policies: Sequence[nn.Module],
    member_state_providers: Sequence[Hold30DecisionStateProvider],
    identity: Hold30C5CohortIdentity,
    checkpoints: Sequence[Hold30C5CheckpointReference],
    *,
    checkpoint_bundle_receipt_sha256: str,
    outer_start: int | None = None,
    fitting_rows: Iterable[int] = (),
) -> Hold30ControlGrossTrace:
    """Execute one exact five-seed C5 score ensemble through H2 once."""

    _require_digest(
        "checkpoint_bundle_receipt_sha256", checkpoint_bundle_receipt_sha256
    )
    fit_rows = tuple(fitting_rows)
    if fit_rows:
        raise Hold30C5C6Error("C5 control execution cannot fit on evaluation rows")
    members = tuple(member_policies)
    providers = tuple(member_state_providers)
    if len(members) != 5 or len(providers) != 5:
        raise Hold30C5C6Error("C5 control requires exactly five policies/providers")
    refs = tuple(checkpoints)
    if not refs:
        raise Hold30C5C6Error("C5 control requires a same-update checkpoint bundle")
    refs = _ordered_c5_refs(refs, update=refs[0].update, identity=identity)
    for member, reference in zip(members, refs, strict=True):
        if c5_model_state_sha256(member) != reference.model_state_sha256:
            raise Hold30C5C6Error(
                "a live C5 model differs from its selected checkpoint"
            )
    adapters = tuple(_FrozenC5Member(member) for member in members)
    ensemble = EnsemblePolicy("H2", adapters, cash_index=sequence.cash_index)
    frozen_providers = tuple(
        _C5MemberStateProvider(provider, member)
        for provider, member in zip(providers, members, strict=True)
    )
    provider = EnsembleStateProvider(frozen_providers)
    runtime_sequence = _runtime_sequence(sequence)
    runtime = Hold30ChronologicalRuntime("H2", state_provider=provider)
    with torch.no_grad():
        _terminal, transitions = runtime.run_to_terminal(ensemble, runtime_sequence)
    provider_bindings = [
        dict(
            value()
            if callable(value := getattr(item, "binding_config", None))
            else value
        )
        for item in providers
    ]
    strategy_sha = _payload_sha256(
        {
            "rule": "five_seed_supervised_30_session_excess_ranker",
            "checkpoint_bundle_receipt_sha256": checkpoint_bundle_receipt_sha256,
            "checkpoint_model_sha256s": [value.model_state_sha256 for value in refs],
            "member_provider_binding_sha256s": [
                _payload_sha256(value) for value in provider_bindings
            ],
            "ensemble": "center_clip_minus2_plus2_then_mean",
            "builder": "canonical_H2",
            "hazard_residual": 0,
            "exposure_residual": 0,
        }
    )
    return _trace_from_transitions(
        "C5",
        sequence,
        runtime_sequence,
        transitions,
        outer_start=_first_score_row(sequence) if outer_start is None else outer_start,
        fitting_rows=fit_rows,
        source_receipt_sha256=checkpoint_bundle_receipt_sha256,
        strategy_inputs_sha256=strategy_sha,
    )


def construct_selected_c5_control(
    sequence: Hold30DatasetSequence,
    member_policies: Sequence[nn.Module],
    member_state_providers: Sequence[Hold30DecisionStateProvider],
    selection_receipt: Mapping[str, Any],
    *,
    outer_start: int | None = None,
) -> Hold30ControlGrossTrace:
    """Execute only the shared update certified by a terminal C5 receipt."""

    outcome = verify_c5_selection_receipt(selection_receipt)
    receipt_sha256 = selection_receipt["receipt_sha256"]
    _require_digest("C5 selection receipt", receipt_sha256)
    return construct_c5_control(
        sequence,
        member_policies,
        member_state_providers,
        outcome.identity,
        outcome.selected.checkpoints,
        checkpoint_bundle_receipt_sha256=receipt_sha256,
        outer_start=outer_start,
        fitting_rows=(),
    )


@dataclass(frozen=True, slots=True)
class Hold30EmpiricalIntentTrace:
    """Raw, already-realized intent sequence captured before C6 permutation."""

    mechanism: str
    setting_id: str
    fold_index: int
    ensemble_member_count: int
    source_id: str
    source_axis_id: str
    source_trace_sha256: str
    outer_score_rows: tuple[int, ...]
    decision_available: torch.Tensor
    entry_scores: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    hazard_residual: torch.Tensor | None = None
    exposure_residual: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mechanism not in {"H0", "H1", "H2", "H3"}:
            raise Hold30C5C6Error(
                "empirical intent mechanism must be H0, H1, H2, or H3"
            )
        try:
            setting = resolve_hold30_setting(self.setting_id)
        except ValueError as exc:
            raise Hold30C5C6Error("C6 source has an unknown stable setting ID") from exc
        if setting.mechanism != self.mechanism:
            raise Hold30C5C6Error("C6 source mechanism differs from its stable setting")
        if isinstance(self.fold_index, bool) or self.fold_index not in range(
            HOLD30_FOLDS
        ):
            raise Hold30C5C6Error("C6 source fold_index must be in [0,5]")
        if self.ensemble_member_count != len(HOLD30_SEEDS):
            raise Hold30C5C6Error("C6 source must be the deployed five-seed ensemble")
        expected_source_id = f"{self.setting_id}/fold-{self.fold_index}"
        if self.source_id != expected_source_id:
            raise Hold30C5C6Error(
                "C6 source_id must be the stable setting/fold deployment ID"
            )
        _require_digest("source_axis_id", self.source_axis_id)
        _require_digest("source_trace_sha256", self.source_trace_sha256)
        if (
            self.decision_available.ndim != 3
            or self.decision_available.dtype != torch.bool
        ):
            raise Hold30C5C6Error(
                "decision_available must be boolean [decision,batch,asset]"
            )
        decisions, batch, assets = self.decision_available.shape
        if (
            tuple(sorted(set(self.outer_score_rows))) != self.outer_score_rows
            or len(self.outer_score_rows) != HOLD30_SCORE_DAYS
            or any(row < 0 or row >= decisions for row in self.outer_score_rows)
        ):
            raise Hold30C5C6Error("C6 source requires exactly 63 outer-score rows")
        matrix = (decisions, batch, assets)
        vector = (decisions, batch)
        requirements: dict[str, tuple[int, ...]]
        prohibited: set[str]
        if self.mechanism in {"H0", "H1"}:
            requirements = {"target_logits": matrix, "gate": vector}
            prohibited = {"entry_scores", "hazard_residual", "exposure_residual"}
        elif self.mechanism == "H2":
            requirements = {
                "entry_scores": matrix,
                "hazard_residual": matrix,
                "exposure_residual": vector,
            }
            prohibited = {"target_logits", "gate"}
        else:
            requirements = {"entry_scores": matrix}
            prohibited = {
                "target_logits",
                "gate",
                "hazard_residual",
                "exposure_residual",
            }
        for name, shape in requirements.items():
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise Hold30C5C6Error(
                    f"empirical {name} must be finite with shape {shape}"
                )
        if any(getattr(self, name) is not None for name in prohibited):
            raise Hold30C5C6Error(
                "empirical intent populated a mechanism-incompatible field"
            )

    @property
    def n_decisions(self) -> int:
        return int(self.decision_available.shape[0])

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.empirical-intent-trace",
            "schema_version": 1,
            "protocol_generation": HOLD30_PROTOCOL_GENERATION,
            "mechanism": self.mechanism,
            "setting_id": self.setting_id,
            "fold_index": self.fold_index,
            "ensemble_member_count": self.ensemble_member_count,
            "deployment_rule": "canonical_five_seed_deployed_raw_intents",
            "source_id": self.source_id,
            "source_axis_id": self.source_axis_id,
            "source_trace_sha256": self.source_trace_sha256,
            "decision_available_sha256": _tensor_sha256(self.decision_available),
            "outer_score_rows": list(self.outer_score_rows),
            "intent_sha256s": {
                name: _tensor_sha256(value)
                for name in _INTENT_FIELDS
                if (value := getattr(self, name)) is not None
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256(self.receipt_payload)

    def intent_at(self, decision: int) -> Hold30Intent:
        if isinstance(decision, bool) or not 0 <= decision < self.n_decisions:
            raise Hold30C5C6Error(
                "empirical intent index lies outside the source trace"
            )
        return Hold30Intent(
            **{
                name: None
                if (value := getattr(self, name)) is None
                else value[decision]
                for name in _INTENT_FIELDS
            }
        )


def capture_empirical_intents(
    canonical: Hold30CanonicalTrace,
    *,
    mechanism: str,
    setting_id: str,
    fold_index: int,
    source_id: str,
    source_axis_id: str,
    source_trace_sha256: str,
    outer_score_rows: Iterable[int],
) -> Hold30EmpiricalIntentTrace:
    """Capture raw pending intents, never realized weights or returns."""

    if not isinstance(canonical, Hold30CanonicalTrace) or not canonical.pending_intents:
        raise Hold30C5C6Error("C6 requires a complete canonical source trace")
    pendings = canonical.pending_intents
    if tuple(value.decision_index for value in pendings) != tuple(range(len(pendings))):
        raise Hold30C5C6Error(
            "canonical pending intents are not a complete ordered sequence"
        )
    if any(value.axis_id != source_axis_id for value in pendings):
        raise Hold30C5C6Error("canonical pending intents belong to another axis")
    fields: dict[str, torch.Tensor | None] = {}
    for name in _INTENT_FIELDS:
        values = tuple(getattr(item.intent, name) for item in pendings)
        fields[name] = (
            None
            if all(value is None for value in values)
            else torch.stack(tuple(value for value in values if value is not None))
        )
        if fields[name] is not None and any(value is None for value in values):
            raise Hold30C5C6Error(
                "canonical intent field presence changed across decisions"
            )
    return Hold30EmpiricalIntentTrace(
        mechanism=mechanism,
        setting_id=setting_id,
        fold_index=fold_index,
        ensemble_member_count=len(HOLD30_SEEDS),
        source_id=source_id,
        source_axis_id=source_axis_id,
        source_trace_sha256=source_trace_sha256,
        outer_score_rows=tuple(outer_score_rows),
        decision_available=torch.stack(
            tuple(value.decision_available for value in pendings)
        ),
        **fields,
    )


@dataclass(frozen=True, slots=True)
class Hold30C6SourceInventory:
    """Exhaustive 8-setting by 6-fold C6 empirical-intent source bank."""

    sources: tuple[Hold30EmpiricalIntentTrace, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        expected = tuple(
            (setting.setting_id, fold_index)
            for setting in HOLD30_MECH8_SETTINGS
            for fold_index in range(HOLD30_FOLDS)
        )
        observed = tuple(
            (source.setting_id, source.fold_index) for source in self.sources
        )
        if observed != expected:
            raise Hold30C5C6Error(
                "C6 source inventory must contain every stable setting/fold in order"
            )
        if len({source.receipt_sha256 for source in self.sources}) != len(expected):
            raise Hold30C5C6Error(
                "C6 source inventory contains duplicate source traces"
            )
        _require_digest("receipt_sha256", self.receipt_sha256)
        if _payload_sha256(self.receipt_payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C6 source-inventory receipt self-hash mismatch")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c6-source-inventory",
            "schema_version": 1,
            "protocol_generation": HOLD30_PROTOCOL_GENERATION,
            "source_rule": "canonical_five_seed_deployed_raw_intents",
            "settings": len(HOLD30_MECH8_SETTINGS),
            "folds": HOLD30_FOLDS,
            "entries": [
                {
                    "setting_id": source.setting_id,
                    "fold_index": source.fold_index,
                    "mechanism": source.mechanism,
                    "source_intent_receipt_sha256": source.receipt_sha256,
                }
                for source in self.sources
            ],
        }

    def require(self, source: Hold30EmpiricalIntentTrace) -> None:
        match = next(
            (
                item
                for item in self.sources
                if (item.setting_id, item.fold_index)
                == (source.setting_id, source.fold_index)
            ),
            None,
        )
        if match is None or match.receipt_sha256 != source.receipt_sha256:
            raise Hold30C5C6Error("C6 source is absent from the exhaustive inventory")


def bind_c6_source_inventory(
    sources: Sequence[Hold30EmpiricalIntentTrace],
) -> Hold30C6SourceInventory:
    """Seal exactly one empirical source for each stable setting and fold."""

    values = tuple(sources)
    payload = {
        "schema": "rl-quant.hold30.c6-source-inventory",
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "source_rule": "canonical_five_seed_deployed_raw_intents",
        "settings": len(HOLD30_MECH8_SETTINGS),
        "folds": HOLD30_FOLDS,
        "entries": [
            {
                "setting_id": source.setting_id,
                "fold_index": source.fold_index,
                "mechanism": source.mechanism,
                "source_intent_receipt_sha256": source.receipt_sha256,
            }
            for source in values
        ],
    }
    return Hold30C6SourceInventory(values, _payload_sha256(payload))


@dataclass(frozen=True, slots=True)
class Hold30C6PermutationDomain:
    """Explicit timestamp rows within which empirical intents may move."""

    name: str
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise Hold30C5C6Error("C6 domain name must be explicit")
        if tuple(sorted(set(self.rows))) != self.rows or len(self.rows) < 2:
            raise Hold30C5C6Error("each C6 permutation domain needs >=2 unique rows")


@dataclass(frozen=True, slots=True)
class Hold30C6PermutationSchedule:
    source_axis_id: str
    source_intent_receipt_sha256: str
    domains: tuple[Hold30C6PermutationDomain, ...]
    mappings: torch.Tensor
    hash_key_sha256: str
    hash_domain: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_axis_id",
            "source_intent_receipt_sha256",
            "hash_key_sha256",
            "receipt_sha256",
        ):
            _require_digest(name, getattr(self, name))
        if (
            self.mappings.dtype != torch.int64
            or self.mappings.ndim != 2
            or self.mappings.shape[0] != HOLD30_C6_REPLICATES
        ):
            raise Hold30C5C6Error("C6 mappings must be int64 [64,decision]")
        decisions = self.mappings.shape[1]
        if (
            len(self.domains) != 1
            or self.domains[0].name != HOLD30_C6_OUTER_SCORE_DOMAIN
            or len(self.domains[0].rows) != HOLD30_SCORE_DAYS
        ):
            raise Hold30C5C6Error(
                "C6 must permute exactly one 63-row outer-score domain"
            )
        domain_rows: set[int] = set()
        for domain in self.domains:
            if any(row < 0 or row >= decisions for row in domain.rows):
                raise Hold30C5C6Error("C6 domain row lies outside the decision axis")
            if domain_rows.intersection(domain.rows):
                raise Hold30C5C6Error("C6 permutation domains overlap")
            domain_rows.update(domain.rows)
        fixed = set(range(decisions)) - domain_rows
        if torch.unique(self.mappings, dim=0).shape[0] != HOLD30_C6_REPLICATES:
            raise Hold30C5C6Error("C6 requires 64 distinct permutation mappings")
        for mapping in self.mappings.detach().to(device="cpu"):
            if any(int(mapping[row]) != row for row in fixed):
                raise Hold30C5C6Error("C6 changed a row outside its frozen domains")
            for domain in self.domains:
                observed = tuple(int(mapping[row]) for row in domain.rows)
                if set(observed) != set(domain.rows):
                    raise Hold30C5C6Error(
                        "C6 mapping is not a within-domain permutation"
                    )
            if torch.equal(mapping, torch.arange(decisions)):
                raise Hold30C5C6Error("C6 replicate cannot be the identity mapping")
        if _payload_sha256(self.receipt_payload) != self.receipt_sha256:
            raise Hold30C5C6Error("C6 permutation-schedule receipt self-hash mismatch")

    @property
    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": "rl-quant.hold30.c6-time-permutation-schedule",
            "schema_version": 1,
            "source_axis_id": self.source_axis_id,
            "source_intent_receipt_sha256": self.source_intent_receipt_sha256,
            "replicates": HOLD30_C6_REPLICATES,
            "domains": [asdict(value) for value in self.domains],
            "hash_encoding": HOLD30_SCHEDULE_ENCODING,
            "hash_key_sha256": self.hash_key_sha256,
            "hash_domain": self.hash_domain,
            "collision_rule": "increment_attempt_until_distinct_nonidentity",
            "mappings_sha256": _tensor_sha256(self.mappings),
            "outcomes_read": False,
            "noncausal": True,
            "deployable": False,
        }


def materialize_c6_permutation_schedule(
    source: Hold30EmpiricalIntentTrace,
    *,
    domains: Sequence[Hold30C6PermutationDomain],
    hash_key_sha256: str,
    hash_domain: str,
) -> Hold30C6PermutationSchedule:
    """Build 64 outcome-blind permutations of only the 63 outer-score rows."""

    _require_digest("hash_key_sha256", hash_key_sha256)
    frozen_domains = tuple(domains)
    # Invoke validation independently of the schedule object before deriving.
    occupied: set[int] = set()
    for domain in frozen_domains:
        if not isinstance(domain, Hold30C6PermutationDomain):
            raise TypeError("domains must contain Hold30C6PermutationDomain")
        if any(
            row >= source.n_decisions for row in domain.rows
        ) or occupied.intersection(domain.rows):
            raise Hold30C5C6Error("C6 domains overlap or escape the source intent axis")
        occupied.update(domain.rows)
    required_domain = Hold30C6PermutationDomain(
        HOLD30_C6_OUTER_SCORE_DOMAIN,
        source.outer_score_rows,
    )
    if frozen_domains != (required_domain,):
        raise Hold30C5C6Error(
            "C6 domain must equal the source's exact 63 outer-score rows"
        )
    permutation_capacity = math.prod(
        math.factorial(len(domain.rows)) for domain in frozen_domains
    )
    if permutation_capacity - 1 < HOLD30_C6_REPLICATES:
        raise Hold30C5C6Error(
            "the explicit C6 domains cannot produce 64 distinct nonidentity mappings"
        )
    mappings = torch.arange(source.n_decisions, dtype=torch.int64).repeat(
        HOLD30_C6_REPLICATES, 1
    )
    accepted: set[tuple[int, ...]] = set()
    for replicate in range(HOLD30_C6_REPLICATES):
        for attempt in range(65_536):
            candidate = torch.arange(source.n_decisions, dtype=torch.int64)
            for domain_index, domain in enumerate(frozen_domains):
                permuted = sorted(
                    domain.rows,
                    key=lambda row: (
                        _hash_u64(
                            hash_key_sha256,
                            hash_domain,
                            replicate,
                            attempt,
                            domain_index,
                            row,
                        ),
                        row,
                    ),
                )
                candidate[torch.tensor(domain.rows)] = torch.tensor(
                    permuted, dtype=torch.int64
                )
            material = tuple(int(value) for value in candidate.tolist())
            if material == tuple(range(source.n_decisions)) or material in accepted:
                continue
            mappings[replicate] = candidate
            accepted.add(material)
            break
        else:  # pragma: no cover - capacity check and SHA-256 make this unreachable.
            raise Hold30C5C6Error("could not derive 64 distinct C6 permutations")
    payload = {
        "schema": "rl-quant.hold30.c6-time-permutation-schedule",
        "schema_version": 1,
        "source_axis_id": source.source_axis_id,
        "source_intent_receipt_sha256": source.receipt_sha256,
        "replicates": HOLD30_C6_REPLICATES,
        "domains": [asdict(value) for value in frozen_domains],
        "hash_encoding": HOLD30_SCHEDULE_ENCODING,
        "hash_key_sha256": hash_key_sha256,
        "hash_domain": hash_domain,
        "collision_rule": "increment_attempt_until_distinct_nonidentity",
        "mappings_sha256": _tensor_sha256(mappings),
        "outcomes_read": False,
        "noncausal": True,
        "deployable": False,
    }
    return Hold30C6PermutationSchedule(
        source_axis_id=source.source_axis_id,
        source_intent_receipt_sha256=source.receipt_sha256,
        domains=frozen_domains,
        mappings=mappings,
        hash_key_sha256=hash_key_sha256,
        hash_domain=hash_domain,
        receipt_sha256=_payload_sha256(payload),
    )


def verify_c6_permutation_schedule(
    source: Hold30EmpiricalIntentTrace,
    schedule: Hold30C6PermutationSchedule,
) -> None:
    """Re-derive all 64 C6 mappings and reject any tamper."""

    expected = materialize_c6_permutation_schedule(
        source,
        domains=schedule.domains,
        hash_key_sha256=schedule.hash_key_sha256,
        hash_domain=schedule.hash_domain,
    )
    if (
        schedule.source_axis_id != expected.source_axis_id
        or schedule.source_intent_receipt_sha256
        != expected.source_intent_receipt_sha256
        or schedule.receipt_sha256 != expected.receipt_sha256
        or not torch.equal(schedule.mappings, expected.mappings)
    ):
        raise Hold30C5C6Error(
            "C6 permutation schedule failed deterministic reconstruction"
        )


class _C6PermutedIntentPolicy(nn.Module):
    def __init__(
        self,
        source: Hold30EmpiricalIntentTrace,
        mapping: torch.Tensor,
    ) -> None:
        super().__init__()
        self.source = source
        self.mapping = mapping.detach().to(device="cpu", dtype=torch.int64).clone()

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        del prev_weights, available, age_summaries
        indexes = state_t[..., 0]
        first = int(indexes[0, 0].detach().to(device="cpu"))
        if not bool((indexes == first).all()):
            raise Hold30C5C6Error("C6 index state is inconsistent within a decision")
        return self.source.intent_at(int(self.mapping[first]))


def construct_c6_controls(
    sequence: Hold30DatasetSequence,
    source: Hold30EmpiricalIntentTrace,
    schedule: Hold30C6PermutationSchedule,
    source_inventory: Hold30C6SourceInventory,
    *,
    outer_start: int | None = None,
    fitting_rows: Iterable[int] = (),
) -> tuple[Hold30ControlGrossTrace, ...]:
    """Execute all 64 noncausal C6 intent permutations at canonical 20 bp."""

    if (
        source.source_axis_id != sequence.axis_id
        or schedule.source_axis_id != sequence.axis_id
    ):
        raise Hold30C5C6Error("C6 source/schedule belongs to another economic axis")
    if not isinstance(source_inventory, Hold30C6SourceInventory):
        raise TypeError("source_inventory must be Hold30C6SourceInventory")
    source_inventory.require(source)
    fit_rows = tuple(fitting_rows)
    if fit_rows:
        raise Hold30C5C6Error("C6 diagnostics cannot fit on evaluation rows")
    if schedule.source_intent_receipt_sha256 != source.receipt_sha256:
        raise Hold30C5C6Error("C6 schedule does not bind the empirical intent trace")
    if (
        source.n_decisions != sequence.n_positions - 1
        or schedule.mappings.shape[1] != source.n_decisions
    ):
        raise Hold30C5C6Error(
            "C6 source/schedule decision count differs from the sequence"
        )
    sequence_outer_rows = tuple(
        int(value) for value in sequence.roles.score_indices.tolist()
    )
    if (
        len(sequence_outer_rows) != HOLD30_SCORE_DAYS
        or source.outer_score_rows != sequence_outer_rows
    ):
        raise Hold30C5C6Error(
            "C6 source must bind this sequence's exact 63 outer-score rows"
        )
    verify_c6_permutation_schedule(source, schedule)
    if source.decision_available.device != sequence.asset_returns.device or any(
        value is not None and value.device != sequence.asset_returns.device
        for value in (getattr(source, name) for name in _INTENT_FIELDS)
    ):
        raise Hold30C5C6Error(
            "C6 source intents must share the economic sequence device"
        )
    if not torch.equal(
        source.decision_available.to(device=sequence.decision_trade.device),
        sequence.decision_trade[:-1],
    ):
        raise Hold30C5C6Error(
            "C6 source empirical intents bind a different decision mask"
        )
    positions = sequence.n_positions
    index_state = sequence.asset_returns.new_empty(
        (positions, sequence.batch_size, sequence.num_assets, 1)
    )
    for position in range(positions):
        index_state[position].fill_(position)
    runtime_sequence = _runtime_sequence(sequence, decision_state=index_state)
    traces: list[Hold30ControlGrossTrace] = []
    first_score = _first_score_row(sequence) if outer_start is None else outer_start
    for replicate in range(HOLD30_C6_REPLICATES):
        mapping = schedule.mappings[replicate]
        policy = _C6PermutedIntentPolicy(source, mapping)
        runtime = Hold30ChronologicalRuntime(source.mechanism)
        with torch.no_grad():
            _terminal, transitions = runtime.run_to_terminal(policy, runtime_sequence)
        strategy_sha = _payload_sha256(
            {
                "rule": "time_permuted_empirical_raw_intents",
                "source_id": source.source_id,
                "source_inventory_receipt_sha256": source_inventory.receipt_sha256,
                "source_intent_receipt_sha256": source.receipt_sha256,
                "schedule_receipt_sha256": schedule.receipt_sha256,
                "replicate": replicate,
                "mapping_sha256": _tensor_sha256(mapping),
                "noncausal": True,
                "deployable": False,
            }
        )
        traces.append(
            _trace_from_transitions(
                "C6",
                sequence,
                runtime_sequence,
                transitions,
                outer_start=first_score,
                fitting_rows=fit_rows,
                source_receipt_sha256=source.source_trace_sha256,
                strategy_inputs_sha256=strategy_sha,
            )
        )
    return tuple(traces)


__all__ = [
    "HOLD30_C5_ADAM_EPS",
    "HOLD30_C5_CONTROL_KEY_ENCODING",
    "HOLD30_C5_DATES_PER_UPDATE",
    "HOLD30_C5_DATE_PURPOSE",
    "HOLD30_C5_GRAD_CLIP",
    "HOLD30_C5_HORIZON",
    "HOLD30_C5_LEARNING_RATE",
    "HOLD30_C5_MAX_UPDATES",
    "HOLD30_C5_MICROBATCH_DATES",
    "HOLD30_C5_MIN_SELECTION_UPDATE",
    "HOLD30_C5_PAIRS_PER_DATE",
    "HOLD30_C5_PAIR_PURPOSE",
    "HOLD30_C5_SELECTION_TOLERANCE",
    "HOLD30_C5_VALIDATION_CADENCE",
    "HOLD30_C5_VALIDATION_PATIENCE",
    "HOLD30_C5_WEIGHT_DECAY",
    "HOLD30_C6_OUTER_SCORE_DOMAIN",
    "HOLD30_C6_REPLICATES",
    "HOLD30_SCHEDULE_ENCODING",
    "Hold30C5C6Error",
    "Hold30C5CheckpointReference",
    "Hold30C5CohortIdentity",
    "Hold30C5DateSchedule",
    "Hold30C5FitBinding",
    "Hold30C5LabelSet",
    "Hold30C5PairSchedule",
    "Hold30C5ScheduleKeyBinding",
    "Hold30C5SelectionOutcome",
    "Hold30C5UpdateResult",
    "Hold30C5ValidationRecord",
    "Hold30C5ValidationScore",
    "Hold30C6PermutationDomain",
    "Hold30C6PermutationSchedule",
    "Hold30C6SourceInventory",
    "Hold30EmpiricalIntentTrace",
    "bind_c5_development_fold",
    "bind_c6_source_inventory",
    "build_c5_labels",
    "build_c5_optimizer",
    "c5_model_state_sha256",
    "capture_empirical_intents",
    "construct_c5_control",
    "construct_c6_controls",
    "construct_selected_c5_control",
    "coordinate_c5_seed_cohort",
    "derive_c5_schedule_key_binding",
    "materialize_c5_date_schedule",
    "materialize_c5_pair_schedule",
    "materialize_c6_permutation_schedule",
    "select_c5_shared_checkpoint",
    "train_c5_update",
    "verify_c5_date_schedule",
    "verify_c5_pair_schedule",
    "verify_c5_selection_receipt",
    "verify_c6_permutation_schedule",
]
