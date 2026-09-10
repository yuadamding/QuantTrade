"""Lossless QT200 historical-message interpretation before native trade replay.

The native executable-trade contract deliberately requires positive quantity and
a provider trade ID. A zero-volume corrected-close *candidate* is a different
message, not a malformed executable trade. This adapter identifies that narrow
shape without inventing an ID, correction target, historical condition mapping,
or strategy availability. Its outputs are diagnostic and cannot enter bars,
features, or fills. In particular, today's condition response does not qualify
the same mapping for 2017.

Use ``historical_message_category_v1`` in selected-row preflight counters and
``interpret_selected_historical_message_v1`` for source-bound retained examples.
Both consume the losslessly preserved fields; neither relaxes canonicalization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    MassiveSelectedOriginalTradeRowV1,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_COLUMNS, _parse_conditions,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


SCHEMA = "rl-quant.qt200-historical-message-interpretation-v1"
SOURCE_IDENTITY_FIELDS = frozenset((
    "source_object_key", "compressed_sha256", "source_receipt_sha256",
    "receipt_file_sha256", "commit_file_sha256",
))
_PRICE_ONLY_CATEGORY = "zero-volume-corrected-close-candidate"
CORRECTION_REFERENCE_URL = "https://massive.com/glossary/conditions-indicators"
_CORRECTION_CANDIDATES = {
    # The provider's NYSE glossary describes snapshot/revision orientation, not
    # a simple append-only replace-current-payload stream. Applicability and
    # exact predecessor linkage for the acquired historical object are still
    # unqualified. In particular 12 is NOT corrected replacement data.
    1: "late-corrected-snapshot-candidate",
    7: "error-original-data-candidate",
    8: "cancellation-original-data-candidate",
    10: "cancellation-message-candidate",
    11: "error-message-candidate",
    12: "correction-original-data-candidate",
}


class Qt200HistoricalMessageError(ValueError):
    """Lossless provenance or bounded diagnostic input is inconsistent."""


def _values(values: Sequence[str]) -> tuple[str, ...]:
    if (isinstance(values, (str, bytes))
            or len(values) != len(MASSIVE_FLAT_TRADE_COLUMNS)
            or any(not isinstance(value, str) for value in values)):
        raise Qt200HistoricalMessageError("exact original trade columns required")
    return tuple(values)


def _number(raw: str) -> Decimal | None:
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _uint64(raw: str) -> int | None:
    # No whitespace/sign/float coercion, and no arbitrary-size integer parsing.
    normalized = raw.lstrip("0") or "0"
    if (not raw or not raw.isascii() or not raw.isdecimal()
            or len(normalized) > 20):
        return None
    value = int(normalized)
    return value if value < 2**64 else None


def historical_message_category_v1(original_values: Sequence[str]) -> str:
    """Classify retained shape only; this function never authorizes economics.

    The price-only profile is intentionally the observed Tape 3 / exchange 12
    / TRF 0 / correction 0 / conditions 38,41 profile. Other zero-volume,
    negative, malformed, or administrative records are not silently accepted.
    A nonzero correction code has at most a glossary candidate interpretation,
    never a resolved lifecycle, even with a provider ID.
    """
    values = _values(original_values)
    price, size = _number(values[6]), _number(values[9])
    try:
        conditions = _parse_conditions(values[1])
    except ValueError:
        return "unsupported-condition-encoding"
    if len(conditions) != len(set(conditions)):
        return "unsupported-condition-encoding"
    conditions = tuple(sorted(conditions))
    if price is None or price <= 0:
        return "unsupported-price"
    if size is None:
        return "unsupported-quantity"
    if size < 0:
        return "negative-quantity-unresolved"
    correction = _uint64(values[2])
    if size == 0:
        if (conditions == (38, 41) and correction == 0
                and _uint64(values[10]) == 3 and _uint64(values[3]) == 12
                and _uint64(values[11]) == 0 and bool(values[4].strip())
                and _uint64(values[5]) not in (None, 0)
                and _uint64(values[8]) not in (None, 0)):
            return _PRICE_ONLY_CATEGORY
        return "zero-quantity-message-unresolved"
    if correction != 0:
        return _CORRECTION_CANDIDATES.get(correction,
            "historical-correction-lifecycle-unresolved")
    if not values[4].strip():
        return "positive-volume-provider-id-unresolved"
    return "positive-volume-trade-candidate"


def interpret_selected_historical_message_v1(
    row: MassiveSelectedOriginalTradeRowV1, *,
    source_identity: Mapping[str, str],
    condition_authority: MassiveConditionAuthority | None = None,
) -> dict[str, Any]:
    """Build nonauthorizing evidence from a real lossless selected source row.

    Source identity is the exact completed scanner transaction, not a free
    trade identity. The caller must publish only after complete source scanning
    succeeds. A row may have a source-record identity and no provider ID.
    Reference-condition results are explicitly *candidate* price-statistic
    rules, never historical authorization. There is no qualification override.
    """
    row.validate()
    if set(source_identity) != SOURCE_IDENTITY_FIELDS:
        raise Qt200HistoricalMessageError("exact source transaction identity required")
    identity = dict(source_identity)
    key = identity["source_object_key"]
    if not isinstance(key, str):
        raise Qt200HistoricalMessageError("source object key must be canonical")
    path = PurePosixPath(key)
    if (not key or not path.parts or path.is_absolute() or str(path) != key
            or any(part in {".", ".."} for part in path.parts)):
        raise Qt200HistoricalMessageError("source object key must be canonical")
    for name, digest in identity.items():
        if name != "source_object_key" and (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise Qt200HistoricalMessageError("source identity requires lowercase SHA-256")
    values = row.original_values
    category = historical_message_category_v1(values)
    participant, sip, trf = (_uint64(values[index]) for index in (5, 8, 12))
    # Not a qualified strategy-availability time: entitlement, receive, and
    # revision-delay rules are additional prerequisites. Never clamp to close.
    clock_floor = (max(participant, sip)
                   if participant is not None and sip is not None
                   and participant > 0 and sip > 0 else None)
    candidate_rules = None
    rule_receipt = None
    if condition_authority is not None:
        condition_authority.validate()
        rule_receipt = condition_authority.receipt_sha256
        if category == _PRICE_ONLY_CATEGORY:
            try:
                open_close, high_low, volume = condition_authority.resolve((38, 41))
            except ValueError:
                category = "zero-volume-condition-reference-unresolved"
            else:
                candidate_rules = dict(updates_open_close=open_close,
                    updates_high_low=high_low, updates_volume=volume)
                if volume:
                    category = "zero-volume-condition-reference-conflict"
    body = dict(
        schema=SCHEMA, source_identity=identity,
        record_id=semantic_sha256(dict(source_identity=identity,
            source_row_number=row.source_row_number, raw_line_sha256=row.raw_line_sha256)),
        selected_row_receipt_sha256=row.receipt_sha256,
        source_row_number=row.source_row_number, raw_line_sha256=row.raw_line_sha256,
        original_values=values, provider_trade_id=values[4] if values[4].strip() else None,
        provider_trade_id_raw=values[4], provider_sequence_number=_uint64(values[7]),
        message_category=category, correction_target_reference=None,
        correction_reference_url=CORRECTION_REFERENCE_URL,
        historical_correction_mapping_qualified=False,
        correction_resolution_status="historical-contract-unresolved",
        participant_timestamp_ns=participant, sip_timestamp_ns=sip, trf_timestamp_ns=trf,
        vendor_clock_lower_bound_ns=clock_floor, strategy_available_at_ns=None,
        candidate_consolidated_price_rules=candidate_rules,
        condition_reference_receipt_sha256=rule_receipt,
        candidate_executable_share_volume="0" if _number(values[9]) == 0 else None,
        canonicalization_error=row.canonicalization_error,
        source_completion_required=True, diagnostic_only=True,
        historical_condition_mapping_qualified=False, identity_qualified=False,
        correction_replay_qualified=False, strategy_availability_qualified=False,
        bar_price_update_authorized=False, volume_update_authorized=False,
        fill_eligible=False, training_ready=False,
    )
    return dict(body, receipt_sha256=semantic_sha256(body))
