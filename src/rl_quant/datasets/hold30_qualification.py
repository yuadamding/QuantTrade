"""Fail-closed in-memory data qualification for Hold-30 mechanism v2.

This gate validates an already materialized :class:`Hold30DatasetSequence`.
It never discovers data, falls back to a local TOP2000 tree, opens a lockbox,
or grants launch/scientific authority.  Passing means only that the supplied
pre-2026 PIT data, benchmark chronology, monthly schedule, fold geometry, and
external source hashes are internally receipt-complete.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

import torch

from rl_quant.datasets.hold30 import (
    HOLD30_PRELOCKBOX_CUTOFF_MS,
    HOLD30_UNIVERSE_MODE,
    Hold30DatasetError,
    Hold30DatasetSequence,
)
from rl_quant.datasets.hold30_null_rebuild import (
    HOLD30_C1_ACTIVE_COUNT,
    _rebuild_c1,
    _validate_monthly_schedule,
)
from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import (
    HOLD30_FOLDS,
    HOLD30_MIN_AXIS_POSITIONS,
    Hold30FreezeError,
    render_hold30_folds,
    sha256_payload,
)


HOLD30_DATA_QUALIFICATION_SCHEMA = 1
HOLD30_PRIMARY_COST_RATE = 0.002
HOLD30_REQUIRED_EXTERNAL_ARTIFACTS: dict[str, str] = {
    "data/data-snapshot.manifest.json": "data_snapshot_sha256",
    "data/raw-market-data.manifest.json": "raw_market_data_sha256",
    "data/universe-events.parquet": "universe_events_sha256",
    "data/tradability-events.parquet": "tradability_events_sha256",
    "data/corporate-actions.parquet": "corporate_actions_sha256",
    "data/identifier-events.parquet": "identifier_events_sha256",
    "data/benchmark-trace.parquet": "c1_benchmark_trace_sha256",
    "data/risk-limits.json": "risk_limits_sha256",
}

_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "protocol_generation",
    "passed",
    "launch_authorized",
    "scientific_qualification",
    "counts",
    "axis",
    "provenance",
    "economic_contract",
    "external_artifacts",
    "checks",
    "digests",
    "receipt_sha256",
}
_COUNT_FIELDS = {"positions", "batches", "assets", "active_risky", "folds"}
_AXIS_FIELDS = {"first", "last", "cutoff", "sha256"}
_PROVENANCE_FIELDS = {
    "receipt_id",
    "universe_mode",
    "universe_rule_id",
    "stable_asset_id_namespace",
}
_ECONOMIC_CONTRACT_FIELDS = {
    "primary_cost_rate",
    "primary_cost_basis_points",
    "turnover_basis",
}
_DIGEST_FIELDS = {
    "decision_timestamps_sha256",
    "fill_timestamps_sha256",
    "asset_ids_sha256",
    "decision_state_sha256",
    "decision_membership_sha256",
    "decision_tradability_sha256",
    "fill_membership_sha256",
    "fill_tradability_sha256",
    "asset_returns_sha256",
    "ordinary_return_valid_sha256",
    "mandatory_return_mask_sha256",
    "c1_weights_sha256",
    "c1_net_returns_sha256",
    "risk_asset_caps_sha256",
    "risk_gross_max_sha256",
    "cost_rate_sha256",
    "asof_evidence_sha256",
    "monthly_schedule_sha256",
    "folds_sha256",
    "tensor_bundle_sha256",
    "external_artifacts_sha256",
}
_CHECKS = (
    "validated_sequence_reconstruction",
    "minimum_prelockbox_axis",
    "causal_decision_fill_chronology",
    "pit_active300_membership",
    "point_in_time_universe_provenance",
    "monthly_event_chronology",
    "fresh_c1_economic_trace",
    "exact_primary_20bp_cost",
    "six_fold_geometry",
    "asof_and_corporate_action_evidence",
    "external_source_hash_closure",
)


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


def _tensor_digest(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30DatasetError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _decision_axis(sequence: Hold30DatasetSequence) -> tuple[str, ...]:
    timestamps = sequence.decision_timestamps_ms.detach().to(device="cpu", dtype=torch.int64)
    return tuple(
        datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date().isoformat()
        for value in timestamps.tolist()
    )


def _validate_universe_provenance(sequence: Hold30DatasetSequence) -> None:
    provenance = sequence.provenance
    if provenance.universe_mode != HOLD30_UNIVERSE_MODE:
        raise Hold30DatasetError("qualification requires point_in_time_events universe mode")
    identity = " ".join(
        (
            provenance.universe_mode,
            provenance.universe_rule_id,
            provenance.stable_asset_id_namespace,
        )
    ).casefold()
    forbidden = ("top2000", "future", "static")
    if any(token in identity for token in forbidden):
        raise Hold30DatasetError(
            "future-selected/static TOP2000 provenance is forbidden for Hold-30"
        )


def _validate_active300(sequence: Hold30DatasetSequence) -> None:
    for name in ("decision_membership", "fill_membership"):
        membership = getattr(sequence, name).detach().to(device="cpu").clone()
        membership[..., sequence.cash_index] = False
        count = membership.sum(dim=-1)
        if bool((count != HOLD30_C1_ACTIVE_COUNT).any()):
            raise Hold30DatasetError(
                f"{name} must contain exactly {HOLD30_C1_ACTIVE_COUNT} PIT risky members"
            )


def _month_ordinal(timestamp_ms: int) -> int:
    value = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return value.year * 12 + value.month - 1


def _validate_calendar_months(
    sequence: Hold30DatasetSequence,
    monthly_rebalance: torch.Tensor,
) -> torch.Tensor:
    schedule = _validate_monthly_schedule(sequence, monthly_rebalance)
    timestamps = sequence.decision_timestamps_ms.detach().to(device="cpu", dtype=torch.int64)
    months = tuple(_month_ordinal(int(value)) for value in timestamps.tolist())
    observed = tuple(dict.fromkeys(months))
    if any(right != left + 1 for left, right in zip(observed, observed[1:])):
        raise Hold30DatasetError("decision axis skips an entire calendar month")
    for month in observed:
        positions = [index for index, value in enumerate(months) if value == month]
        count = int(schedule[positions].sum())
        if count != 1:
            raise Hold30DatasetError(
                "monthly_rebalance must contain exactly one frozen event per observed month"
            )
    return schedule


def _validate_external_artifacts(
    sequence: Hold30DatasetSequence,
    artifacts: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(artifacts, Mapping):
        raise Hold30DatasetError("external_artifacts must be a path-to-digest mapping")
    supplied = set(artifacts)
    expected = set(HOLD30_REQUIRED_EXTERNAL_ARTIFACTS)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise Hold30DatasetError(f"external artifact digests are missing: {missing}")
    if unknown:
        raise Hold30DatasetError(f"unknown external artifact digest paths: {unknown}")
    canonical: dict[str, str] = {}
    for path in sorted(expected):
        value = _require_digest(path, artifacts[path])
        provenance_field = HOLD30_REQUIRED_EXTERNAL_ARTIFACTS[path]
        expected_digest = getattr(sequence.provenance, provenance_field)
        if value != expected_digest:
            raise Hold30DatasetError(
                f"external artifact digest for {path} does not match provenance"
            )
        canonical[path] = value
    return canonical


def _asof_digest(sequence: Hold30DatasetSequence) -> str:
    return _canonical_digest(
        {
            name: _tensor_digest(getattr(sequence.asof_evidence, name))
            for name in sequence.asof_evidence.__dataclass_fields__
        }
    )


def _tensor_digests(sequence: Hold30DatasetSequence) -> dict[str, str]:
    return {
        "decision_timestamps_sha256": _tensor_digest(sequence.decision_timestamps_ms),
        "fill_timestamps_sha256": _tensor_digest(sequence.fill_timestamps_ms),
        "asset_ids_sha256": _canonical_digest(sequence.asset_ids),
        "decision_state_sha256": _tensor_digest(sequence.decision_state),
        "decision_membership_sha256": _tensor_digest(sequence.decision_membership),
        "decision_tradability_sha256": _tensor_digest(sequence.decision_tradability),
        "fill_membership_sha256": _tensor_digest(sequence.fill_membership),
        "fill_tradability_sha256": _tensor_digest(sequence.fill_tradability),
        "asset_returns_sha256": _tensor_digest(sequence.asset_returns),
        "ordinary_return_valid_sha256": _tensor_digest(sequence.ordinary_return_valid),
        "mandatory_return_mask_sha256": _tensor_digest(sequence.mandatory_return_mask),
        "c1_weights_sha256": _tensor_digest(sequence.c1_benchmark_weights),
        "c1_net_returns_sha256": _tensor_digest(sequence.c1_benchmark_net_returns),
        "risk_asset_caps_sha256": _tensor_digest(sequence.risk_asset_caps),
        "risk_gross_max_sha256": _tensor_digest(sequence.risk_gross_max),
        "cost_rate_sha256": _tensor_digest(sequence.cost_rate),
        "asof_evidence_sha256": _asof_digest(sequence),
    }


def qualify_hold30_dataset(
    sequence: Hold30DatasetSequence,
    *,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and content-address one complete pre-lockbox data generation."""

    if not isinstance(sequence, Hold30DatasetSequence):
        raise Hold30DatasetError("sequence must be a Hold30DatasetSequence")
    # Tensors inside a frozen dataclass remain mutable. Reconstructing reruns
    # every dataset/as-of/corporate-action invariant and exposes cached-axis
    # tampering rather than trusting construction-time validation.
    validated = replace(sequence)
    if (
        validated.axis_id != sequence.axis_id
        or validated.randomization_axis_id != sequence.randomization_axis_id
    ):
        raise Hold30DatasetError("sequence tensors changed after their axis receipt was created")
    if validated.n_positions < HOLD30_MIN_AXIS_POSITIONS:
        raise Hold30DatasetError(
            f"Hold-30 data qualification requires N >= {HOLD30_MIN_AXIS_POSITIONS}; "
            f"got {validated.n_positions}"
        )
    if int(validated.decision_timestamps_ms[-1]) >= HOLD30_PRELOCKBOX_CUTOFF_MS or int(
        validated.fill_timestamps_ms.max()
    ) >= HOLD30_PRELOCKBOX_CUTOFF_MS:
        raise Hold30DatasetError("every decision and fill must remain strictly pre-2026")
    expected_cost = validated.cost_rate.new_full(
        validated.cost_rate.shape,
        HOLD30_PRIMARY_COST_RATE,
    )
    if not torch.equal(validated.cost_rate, expected_cost):
        raise Hold30DatasetError(
            "Hold-30 data qualification requires cost_rate exactly 0.002 (20 bp) "
            "on every row and batch"
        )

    _validate_active300(validated)
    _validate_universe_provenance(validated)
    schedule = _validate_calendar_months(validated, monthly_rebalance)
    rebuilt_c1 = _rebuild_c1(
        validated,
        validated.asset_returns,
        schedule.to(device=validated.asset_returns.device),
    )
    if not torch.equal(rebuilt_c1.weights, validated.c1_benchmark_weights) or not torch.equal(
        rebuilt_c1.net_returns, validated.c1_benchmark_net_returns
    ):
        raise Hold30DatasetError(
            "C1 benchmark trace is stale or violates the frozen monthly drift/repair/cost chronology"
        )
    if rebuilt_c1.trace_sha256 != validated.provenance.c1_benchmark_trace_sha256:
        raise Hold30DatasetError("C1 trace digest does not match point-in-time provenance")
    artifacts = _validate_external_artifacts(validated, external_artifacts)

    axis = _decision_axis(validated)
    try:
        folds = render_hold30_folds(axis)
    except Hold30FreezeError as exc:
        raise Hold30DatasetError(f"frozen fold rendering failed: {exc}") from exc
    if len(folds) != HOLD30_FOLDS or tuple(fold.fold_index for fold in folds) != tuple(
        range(HOLD30_FOLDS)
    ):
        raise Hold30DatasetError("fold renderer did not return the exact six ordered folds")
    fold_payload = [asdict(fold) for fold in folds]

    digests = _tensor_digests(validated)
    digests["monthly_schedule_sha256"] = _tensor_digest(schedule)
    digests["folds_sha256"] = sha256_payload(fold_payload)
    digests["external_artifacts_sha256"] = _canonical_digest(artifacts)
    digests["tensor_bundle_sha256"] = _canonical_digest(digests)
    receipt: dict[str, Any] = {
        "schema_version": HOLD30_DATA_QUALIFICATION_SCHEMA,
        "receipt_type": "prelockbox-hold30-data-qualification",
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "passed": True,
        "launch_authorized": False,
        "scientific_qualification": False,
        "counts": {
            "positions": validated.n_positions,
            "batches": validated.batch_size,
            "assets": validated.num_assets,
            "active_risky": HOLD30_C1_ACTIVE_COUNT,
            "folds": len(folds),
        },
        "axis": {
            "first": axis[0],
            "last": axis[-1],
            "cutoff": "2026-01-01",
            "sha256": sha256_payload(axis),
        },
        "provenance": {
            "receipt_id": validated.provenance.receipt_id,
            "universe_mode": validated.provenance.universe_mode,
            "universe_rule_id": validated.provenance.universe_rule_id,
            "stable_asset_id_namespace": validated.provenance.stable_asset_id_namespace,
        },
        "economic_contract": {
            "primary_cost_rate": HOLD30_PRIMARY_COST_RATE,
            "primary_cost_basis_points": 20,
            "turnover_basis": "executed_one_way",
        },
        "external_artifacts": artifacts,
        "checks": list(_CHECKS),
        "digests": digests,
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    verify_hold30_data_qualification_receipt(receipt)
    return receipt


def verify_hold30_data_qualification_receipt(receipt: Mapping[str, Any]) -> None:
    """Verify canonical receipt shape and self-hash; reject partial extensions."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise Hold30DatasetError("data qualification receipt has partial or unknown fields")
    for name, expected in (
        ("counts", _COUNT_FIELDS),
        ("axis", _AXIS_FIELDS),
        ("provenance", _PROVENANCE_FIELDS),
        ("economic_contract", _ECONOMIC_CONTRACT_FIELDS),
        ("digests", _DIGEST_FIELDS),
    ):
        value = receipt[name]
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Hold30DatasetError(f"data qualification receipt {name} fields are not exact")
    if not isinstance(receipt["external_artifacts"], Mapping) or set(
        receipt["external_artifacts"]
    ) != set(HOLD30_REQUIRED_EXTERNAL_ARTIFACTS):
        raise Hold30DatasetError("data qualification receipt artifact fields are not exact")
    if tuple(receipt["checks"]) != _CHECKS:
        raise Hold30DatasetError("data qualification receipt checks are not exact")
    if (
        receipt["schema_version"] != HOLD30_DATA_QUALIFICATION_SCHEMA
        or receipt["receipt_type"] != "prelockbox-hold30-data-qualification"
        or receipt["protocol_generation"] != HOLD30_PROTOCOL_GENERATION
        or receipt["passed"] is not True
        or receipt["launch_authorized"] is not False
        or receipt["scientific_qualification"] is not False
    ):
        raise Hold30DatasetError("data qualification receipt authority/status fields are invalid")
    if receipt["economic_contract"] != {
        "primary_cost_rate": HOLD30_PRIMARY_COST_RATE,
        "primary_cost_basis_points": 20,
        "turnover_basis": "executed_one_way",
    }:
        raise Hold30DatasetError("data qualification receipt economic contract is invalid")
    counts = receipt["counts"]
    if (
        isinstance(counts["positions"], bool)
        or not isinstance(counts["positions"], int)
        or counts["positions"] < HOLD30_MIN_AXIS_POSITIONS
        or isinstance(counts["batches"], bool)
        or not isinstance(counts["batches"], int)
        or counts["batches"] <= 0
        or isinstance(counts["assets"], bool)
        or not isinstance(counts["assets"], int)
        or counts["assets"] < HOLD30_C1_ACTIVE_COUNT + 1
        or counts["active_risky"] != HOLD30_C1_ACTIVE_COUNT
        or counts["folds"] != HOLD30_FOLDS
    ):
        raise Hold30DatasetError("data qualification receipt counts are invalid")
    axis = receipt["axis"]
    if axis["cutoff"] != "2026-01-01" or not (
        isinstance(axis["first"], str)
        and isinstance(axis["last"], str)
        and axis["first"] < axis["last"] < axis["cutoff"]
    ):
        raise Hold30DatasetError("data qualification receipt axis bounds are invalid")
    _require_digest("axis.sha256", axis["sha256"])
    provenance = receipt["provenance"]
    if (
        provenance["universe_mode"] != HOLD30_UNIVERSE_MODE
        or not isinstance(provenance["universe_rule_id"], str)
        or not isinstance(provenance["stable_asset_id_namespace"], str)
    ):
        raise Hold30DatasetError("data qualification receipt universe mode is invalid")
    identity = " ".join(
        (
            provenance["universe_mode"],
            provenance["universe_rule_id"],
            provenance["stable_asset_id_namespace"],
        )
    ).casefold()
    if any(token in identity for token in ("top2000", "future", "static")):
        raise Hold30DatasetError("data qualification receipt contains forbidden universe provenance")
    for name, value in receipt["digests"].items():
        _require_digest(name, value)
    for name, value in receipt["external_artifacts"].items():
        _require_digest(name, value)
    if _canonical_digest(receipt["external_artifacts"]) != receipt["digests"][
        "external_artifacts_sha256"
    ]:
        raise Hold30DatasetError("data qualification receipt artifact digest does not match")
    bundle_inputs = dict(receipt["digests"])
    claimed_bundle = bundle_inputs.pop("tensor_bundle_sha256")
    if _canonical_digest(bundle_inputs) != claimed_bundle:
        raise Hold30DatasetError("data qualification receipt tensor bundle digest does not match")
    _require_digest("provenance.receipt_id", receipt["provenance"]["receipt_id"])
    claimed = _require_digest("receipt_sha256", receipt["receipt_sha256"])
    payload = dict(receipt)
    del payload["receipt_sha256"]
    if _canonical_digest(payload) != claimed:
        raise Hold30DatasetError("data qualification receipt self-hash does not match")


def verify_hold30_dataset_against_qualification(
    sequence: Hold30DatasetSequence,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    receipt: Mapping[str, Any],
) -> None:
    """Recompute live data evidence and require the exact qualified receipt.

    Receipt-shape verification alone cannot detect mutation of a tensor-backed
    sequence after qualification. This verifier first validates the supplied
    receipt, then reruns every data gate against the live sequence, schedule,
    and external bindings and requires canonical byte equality.
    """

    verify_hold30_data_qualification_receipt(receipt)
    recomputed = qualify_hold30_dataset(
        sequence,
        monthly_rebalance=monthly_rebalance,
        external_artifacts=external_artifacts,
    )
    if _canonical_json(recomputed) != _canonical_json(receipt):
        raise Hold30DatasetError(
            "live Hold-30 data no longer matches the qualified receipt"
        )


__all__ = [
    "HOLD30_DATA_QUALIFICATION_SCHEMA",
    "HOLD30_PRIMARY_COST_RATE",
    "HOLD30_REQUIRED_EXTERNAL_ARTIFACTS",
    "qualify_hold30_dataset",
    "verify_hold30_dataset_against_qualification",
    "verify_hold30_data_qualification_receipt",
]
