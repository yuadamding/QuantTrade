"""Typed daily trade-tape artifacts for finalized Massive validation V0."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import median

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
    load_massive_persisted_security_rows_v2,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_DAILY_TAPE_V0_FIELDS = (
    "log_trade_count",
    "median_trade_size",
    "p90_trade_size",
    "large_trade_dollar_fraction",
    "quote_free_signed_dollar_flow_proxy",
    "absolute_signed_flow_imbalance",
    "price_response_per_signed_dollar",
    "trf_off_exchange_dollar_fraction",
    "venue_entropy",
    "largest_venue_share",
    "tape_a_fraction",
    "tape_b_fraction",
    "tape_c_fraction",
    "special_condition_fraction",
    "correction_replacement_fraction",
)
MASSIVE_DAILY_TAPE_V0_SCHEMA = "rl-quant.massive-daily-tape-v0"
MASSIVE_DAILY_TAPE_V0_DATASET = "massive-finalized-daily-tape-v0"
MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_DAILY_TAPE_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "input": "terminal-active-regular-plus-complete-correction-timeline",
        "fields": MASSIVE_DAILY_TAPE_V0_FIELDS,
        "signed_flow": "quote-free-tick-rule;zero-change-carries-prior;initial-zero",
        "large_trade": "price-times-size-at-least-100000-usd",
        "quantile": "nearest-rank",
        "trf_proxy": "trf-id-present",
        "venue_weight": "dollar-volume",
        "condition_proxy": "nonempty-condition-inventory",
        "correction_fraction": "replacement-or-cancellation-or-late-report/events",
        "aggressor_side_claimed": False,
        "investor_identity_claimed": False,
    }
)
MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_DAILY_TAPE_V0_SCHEMA,
        "fields": MASSIVE_DAILY_TAPE_V0_FIELDS,
        "value_type": "finite-float64",
        "mask_type": "boolean",
    }
)


class MassiveDailyTapeV0Error(ValueError):
    """Daily tape bytes or semantics differ from the V0 contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveDailyTapeV0Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveDailyTapeRowV0:
    security_id: str
    values: tuple[float, ...]
    valid: tuple[bool, ...]
    source_active_inventory_sha256: str
    source_correction_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id or self.security_id != self.security_id.strip():
            raise MassiveDailyTapeV0Error("tape security identity is invalid")
        if (
            len(self.values) != len(MASSIVE_DAILY_TAPE_V0_FIELDS)
            or len(self.valid) != len(self.values)
            or any(not isinstance(value, bool) for value in self.valid)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in self.values
            )
        ):
            raise MassiveDailyTapeV0Error("tape values or masks are malformed")
        for name in (
            "source_active_inventory_sha256",
            "source_correction_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyTapeV0Error("tape row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveDailyTapeArtifactV0:
    source_session_date: str
    persisted_partition_manifest_receipt_sha256: str
    condition_authority_receipt_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    rows: tuple[MassiveDailyTapeRowV0, ...]
    row_inventory_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_DAILY_TAPE_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_DAILY_TAPE_V0_SCHEMA:
            raise MassiveDailyTapeV0Error("daily tape schema drifted")
        for name in (
            "persisted_partition_manifest_receipt_sha256",
            "condition_authority_receipt_sha256",
            "feature_spec_receipt_sha256",
            "feature_source_sha256",
            "row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.feature_spec_receipt_sha256 != MASSIVE_DAILY_TAPE_V0_SPEC_SHA256
            or self.feature_source_sha256 != MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256
        ):
            raise MassiveDailyTapeV0Error("daily tape implementation drifted")
        security_ids = tuple(row.security_id for row in self.rows)
        if not security_ids or security_ids != tuple(sorted(set(security_ids))):
            raise MassiveDailyTapeV0Error("daily tape rows are not canonical")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveDailyTapeV0Error("daily tape inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_DAILY_TAPE_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveDailyTapeV0Error("daily tape source contract differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveDailyTapeV0Error("daily tape artifact receipt differs")


def _nearest_rank(values: tuple[Decimal, ...], quantile: float) -> Decimal:
    ordered = tuple(sorted(values))
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _row(
    security_id: str,
    active_rows: tuple[object, ...],
    corrections: tuple[dict[str, object], ...],
    condition_authority: MassiveConditionAuthority,
) -> MassiveDailyTapeRowV0:
    ordered = tuple(
        sorted(
            active_rows,
            key=lambda row: (
                row.canonical_record.participant_timestamp_ns,
                row.canonical_record.sip_timestamp_ns,
                row.canonical_record.sequence_number,
                row.source_row_number,
            ),
        )
    )
    if not ordered:
        raise MassiveDailyTapeV0Error("daily tape requires active regular trades")
    for row in ordered:
        condition_authority.resolve(row.canonical_record.conditions)
    prices = tuple(Decimal(row.canonical_record.price_decimal) for row in ordered)
    sizes = tuple(Decimal(row.canonical_record.size_decimal) for row in ordered)
    dollars = tuple(price * size for price, size in zip(prices, sizes, strict=True))
    total_dollars = sum(dollars, Decimal(0))
    if total_dollars <= 0:
        raise MassiveDailyTapeV0Error("daily tape dollar volume is nonpositive")
    signs: list[int] = []
    prior_sign = 0
    for index, price in enumerate(prices):
        sign = (
            0
            if index == 0
            else 1
            if price > prices[index - 1]
            else -1
            if price < prices[index - 1]
            else prior_sign
        )
        signs.append(sign)
        if sign:
            prior_sign = sign
    signed_flow = sum(
        (Decimal(sign) * value for sign, value in zip(signs, dollars, strict=True)),
        Decimal(0),
    )
    venue_dollars: defaultdict[int, Decimal] = defaultdict(Decimal)
    for row, value in zip(ordered, dollars, strict=True):
        venue_dollars[row.canonical_record.exchange_id] += value
    venue_shares = tuple(
        float(value / total_dollars) for value in venue_dollars.values()
    )
    entropy = -sum(value * math.log(value) for value in venue_shares if value > 0)
    tapes = {
        tape: sum(
            value
            for row, value in zip(ordered, dollars, strict=True)
            if row.canonical_record.tape_id == tape
        )
        / total_dollars
        for tape in (1, 2, 3)
    }
    price_move = prices[-1] - prices[0]
    response = Decimal(0) if signed_flow == 0 else price_move / abs(signed_flow)
    values = (
        math.log1p(len(ordered)),
        float(median(sizes)),
        float(_nearest_rank(sizes, 0.90)),
        float(
            sum(value for value in dollars if value >= Decimal("100000"))
            / total_dollars
        ),
        float(signed_flow),
        float(abs(signed_flow) / total_dollars),
        float(response),
        float(
            sum(
                value
                for row, value in zip(ordered, dollars, strict=True)
                if row.canonical_record.trf_id is not None
            )
            / total_dollars
        ),
        entropy,
        max(venue_shares),
        float(tapes[1]),
        float(tapes[2]),
        float(tapes[3]),
        sum(bool(row.canonical_record.conditions) for row in ordered) / len(ordered),
        len(corrections) / len(ordered),
    )
    body = {
        "security_id": security_id,
        "values": values,
        "valid": (True,) * len(values),
        "source_active_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "source_correction_inventory_sha256": semantic_sha256(corrections),
    }
    result = MassiveDailyTapeRowV0(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _payload(
    *,
    source_session_date: str,
    persisted_receipt: str,
    condition_authority_receipt: str,
    rows: tuple[MassiveDailyTapeRowV0, ...],
) -> dict[str, object]:
    return {
        "schema": MASSIVE_DAILY_TAPE_V0_SCHEMA,
        "source_session_date": source_session_date,
        "persisted_partition_manifest_receipt_sha256": persisted_receipt,
        "condition_authority_receipt_sha256": condition_authority_receipt,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256,
        "feature_names": MASSIVE_DAILY_TAPE_V0_FIELDS,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
    }


def materialize_massive_daily_tape_v0(
    *,
    persisted_root: str | Path,
    output_root: str | Path,
    manifest: MassivePersistedPartitionManifestV1,
    condition_authority: MassiveConditionAuthority,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveDailyTapeArtifactV0:
    manifest.validate()
    condition_authority.validate()
    rows = []
    for partition in manifest.partitions:
        _, active, corrections = load_massive_persisted_security_rows_v2(
            root=persisted_root, partition=partition
        )
        if active:
            rows.append(
                _row(
                    partition.security_id,
                    active,
                    corrections,
                    condition_authority,
                )
            )
    ordered = tuple(sorted(rows, key=lambda row: row.security_id))
    if not ordered:
        raise MassiveDailyTapeV0Error("daily tape contains no valid securities")
    payload = _payload(
        source_session_date=manifest.source_session_date,
        persisted_receipt=manifest.receipt_sha256,
        condition_authority_receipt=condition_authority.receipt_sha256,
        rows=ordered,
    )
    relative = (
        f"massive-finalized-v0/session={manifest.source_session_date}/daily-tape.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_DAILY_TAPE_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root, relative_payload_path=relative, verified_at_ms=published_at_ms
    )
    body = {
        "source_session_date": manifest.source_session_date,
        "persisted_partition_manifest_receipt_sha256": manifest.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "feature_spec_receipt_sha256": MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
        "feature_source_sha256": MASSIVE_DAILY_TAPE_V0_SOURCE_SHA256,
        "rows": ordered,
        "row_inventory_sha256": payload["row_inventory_sha256"],
        "loaded_source": loaded,
        "schema": MASSIVE_DAILY_TAPE_V0_SCHEMA,
    }
    provisional = MassiveDailyTapeArtifactV0(
        **body,
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = MassiveDailyTapeArtifactV0(
        **body,
        receipt_sha256=semantic_sha256(provisional.unsigned()),  # type: ignore[arg-type]
    )
    validate_massive_daily_tape_v0(root=output_root, artifact=result)
    return result


def validate_massive_daily_tape_v0(
    *, root: str | Path, artifact: MassiveDailyTapeArtifactV0
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveDailyTapeV0Error("daily tape source is not JSON") from exc
    expected = _payload(
        source_session_date=artifact.source_session_date,
        persisted_receipt=artifact.persisted_partition_manifest_receipt_sha256,
        condition_authority_receipt=artifact.condition_authority_receipt_sha256,
        rows=artifact.rows,
    )
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        expected
    ):
        raise MassiveDailyTapeV0Error("daily tape bytes differ")


__all__ = [
    "MASSIVE_DAILY_TAPE_V0_DATASET",
    "MASSIVE_DAILY_TAPE_V0_FIELDS",
    "MASSIVE_DAILY_TAPE_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_DAILY_TAPE_V0_SPEC_SHA256",
    "MassiveDailyTapeArtifactV0",
    "MassiveDailyTapeRowV0",
    "MassiveDailyTapeV0Error",
    "materialize_massive_daily_tape_v0",
    "validate_massive_daily_tape_v0",
]
