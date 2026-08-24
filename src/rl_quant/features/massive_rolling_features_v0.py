"""Causal rolling BARS_V0 and TAPE_V0 feature artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Sequence

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_FIELDS,
    MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
    MassiveDailyBarsArtifactV0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_daily_tape_v0 import (
    MASSIVE_DAILY_TAPE_V0_FIELDS,
    MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
    MassiveDailyTapeArtifactV0,
    validate_massive_daily_tape_v0,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)


MASSIVE_ROLLING_BARS_V0_FIELDS = (
    "return_1",
    "return_5",
    "return_21",
    "return_63",
    "reversal_1",
    "reversal_5",
    "trend_21_minus_5",
    "trend_63_minus_21",
    "realized_volatility_5",
    "realized_volatility_21",
    "downside_volatility_21",
    "high_low_range",
    "close_location",
    "log_dollar_volume",
    "dollar_volume_surprise_21",
    "amihud_21",
    "listing_age_days",
    "valid_history_fraction_63",
)
MASSIVE_ROLLING_TAPE_V0_FIELDS = MASSIVE_DAILY_TAPE_V0_FIELDS
MASSIVE_ROLLING_FEATURES_V0_SCHEMA = "rl-quant.massive-rolling-features-v0"
MASSIVE_ROLLING_FEATURES_V0_DATASET = "massive-finalized-rolling-features-v0"
MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "daily_bars_spec": MASSIVE_DAILY_BARS_V0_SPEC_SHA256,
        "daily_tape_spec": MASSIVE_DAILY_TAPE_V0_SPEC_SHA256,
        "bars_fields": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "returns": "simple-close-to-close",
        "volatility": "population-standard-deviation-of-daily-simple-returns",
        "downside": "population-standard-deviation-of-negative-daily-returns;zero-if-none",
        "volume_surprise": "current/mean-prior-21-minus-one",
        "amihud": "mean-absolute-return-over-dollar-volume-prior-21",
        "minimum_observations": "complete-window-per-feature",
        "missing": "zero-value-plus-false-mask",
        "horizon_preference": None,
    }
)
MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ROLLING_FEATURES_V0_SCHEMA,
        "bars_fields": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_fields": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "value_type": "finite-float64",
        "mask_type": "boolean",
    }
)


class MassiveRollingFeaturesV0Error(ValueError):
    """Rolling feature bytes or causal support differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveRollingFeaturesV0Error(f"{name} must be a lowercase SHA-256")
    return value


def _feature(value: float | None) -> tuple[float, bool]:
    if value is None or not math.isfinite(value):
        return 0.0, False
    return float(value), True


@dataclass(frozen=True, slots=True)
class MassiveRollingFeatureRowV0:
    security_id: str
    bars_values: tuple[float, ...]
    bars_valid: tuple[bool, ...]
    tape_values: tuple[float, ...]
    tape_valid: tuple[bool, ...]
    latest_daily_bars_row_receipt_sha256: str
    latest_daily_tape_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if not self.security_id:
            raise MassiveRollingFeaturesV0Error("rolling security ID is absent")
        pairs = (
            (self.bars_values, self.bars_valid, MASSIVE_ROLLING_BARS_V0_FIELDS),
            (self.tape_values, self.tape_valid, MASSIVE_ROLLING_TAPE_V0_FIELDS),
        )
        for values, valid, fields in pairs:
            if (
                len(values) != len(fields)
                or len(valid) != len(fields)
                or any(not isinstance(flag, bool) for flag in valid)
                or any(not math.isfinite(float(value)) for value in values)
            ):
                raise MassiveRollingFeaturesV0Error(
                    "rolling feature values or masks are malformed"
                )
        for name in (
            "latest_daily_bars_row_receipt_sha256",
            "latest_daily_tape_row_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRollingFeaturesV0Error("rolling row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveRollingFeatureArtifactV0:
    source_session_date: str
    daily_bars_artifact_receipts: tuple[str, ...]
    daily_tape_artifact_receipts: tuple[str, ...]
    identity_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    rows: tuple[MassiveRollingFeatureRowV0, ...]
    row_inventory_sha256: str
    maximum_input_session_date: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ROLLING_FEATURES_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ROLLING_FEATURES_V0_SCHEMA:
            raise MassiveRollingFeaturesV0Error("rolling feature schema drifted")
        if self.maximum_input_session_date != self.source_session_date:
            raise MassiveRollingFeaturesV0Error(
                "rolling features include a future session"
            )
        for inventory in (
            self.daily_bars_artifact_receipts,
            self.daily_tape_artifact_receipts,
        ):
            if not inventory or inventory != tuple(sorted(set(inventory))):
                raise MassiveRollingFeaturesV0Error(
                    "rolling daily input inventory is not canonical"
                )
            for value in inventory:
                _digest("rolling daily input", value)
        for name in (
            "identity_authority_receipt_sha256",
            "condition_authority_receipt_sha256",
            "feature_spec_receipt_sha256",
            "feature_source_sha256",
            "row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.feature_spec_receipt_sha256 != MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256
            or self.feature_source_sha256 != MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256
        ):
            raise MassiveRollingFeaturesV0Error("rolling implementation drifted")
        securities = tuple(row.security_id for row in self.rows)
        if not securities or securities != tuple(sorted(set(securities))):
            raise MassiveRollingFeaturesV0Error("rolling rows are not canonical")
        for row in self.rows:
            row.validate()
        if self.row_inventory_sha256 != semantic_sha256(
            tuple(row.receipt_sha256 for row in self.rows)
        ):
            raise MassiveRollingFeaturesV0Error("rolling row inventory differs")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_ROLLING_FEATURES_V0_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveRollingFeaturesV0Error("rolling source contract differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRollingFeaturesV0Error("rolling artifact receipt differs")


def _return(closes: Sequence[float], sessions: int) -> float | None:
    if len(closes) <= sessions or closes[-sessions - 1] == 0:
        return None
    return closes[-1] / closes[-sessions - 1] - 1.0


def _daily_returns(closes: Sequence[float], sessions: int) -> tuple[float, ...] | None:
    if len(closes) <= sessions:
        return None
    window = closes[-sessions - 1 :]
    if any(value == 0 for value in window[:-1]):
        return None
    return tuple(
        window[index] / window[index - 1] - 1.0 for index in range(1, len(window))
    )


def _bars_features(
    history: Sequence[object], *, listing_at_ms: int, source_session_date: str
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    range_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("high_low_range")
    location_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close_location")
    closes = tuple(float(row.values[close_index]) for row in history)
    dollars = tuple(float(row.values[dollar_index]) for row in history)
    returns = {window: _return(closes, window) for window in (1, 5, 21, 63)}
    daily5 = _daily_returns(closes, 5)
    daily21 = _daily_returns(closes, 21)
    prior21 = dollars[-22:-1] if len(dollars) >= 22 else ()
    volume_surprise = (
        None
        if len(prior21) != 21 or sum(prior21) <= 0
        else dollars[-1] / (sum(prior21) / 21.0) - 1.0
    )
    amihud = None
    if (
        daily21 is not None
        and len(dollars) >= 21
        and all(value > 0 for value in dollars[-21:])
    ):
        amihud = (
            sum(
                abs(ret) / volume
                for ret, volume in zip(daily21, dollars[-21:], strict=True)
            )
            / 21.0
        )
    source_ms = int(
        datetime.combine(
            date.fromisoformat(source_session_date),
            datetime.min.time(),
            tzinfo=timezone.utc,
        ).timestamp()
        * 1000
    )
    listing_age_days = max(0.0, (source_ms - listing_at_ms) / 86_400_000.0)
    raw = (
        returns[1],
        returns[5],
        returns[21],
        returns[63],
        None if returns[1] is None else -returns[1],
        None if returns[5] is None else -returns[5],
        None if returns[21] is None or returns[5] is None else returns[21] - returns[5],
        None
        if returns[63] is None or returns[21] is None
        else returns[63] - returns[21],
        None if daily5 is None else pstdev(daily5),
        None if daily21 is None else pstdev(daily21),
        None
        if daily21 is None
        else pstdev(tuple(value for value in daily21 if value < 0))
        if sum(value < 0 for value in daily21) >= 2
        else 0.0,
        float(history[-1].values[range_index]),
        float(history[-1].values[location_index]),
        math.log1p(max(0.0, dollars[-1])),
        volume_surprise,
        amihud,
        listing_age_days,
        min(1.0, len(history) / 64.0),
    )
    encoded = tuple(_feature(value) for value in raw)
    return tuple(value for value, _ in encoded), tuple(valid for _, valid in encoded)


def _payload(artifact: MassiveRollingFeatureArtifactV0) -> dict[str, object]:
    return {
        "schema": artifact.schema,
        "source_session_date": artifact.source_session_date,
        "daily_bars_artifact_receipts": artifact.daily_bars_artifact_receipts,
        "daily_tape_artifact_receipts": artifact.daily_tape_artifact_receipts,
        "identity_authority_receipt_sha256": artifact.identity_authority_receipt_sha256,
        "condition_authority_receipt_sha256": artifact.condition_authority_receipt_sha256,
        "feature_spec_receipt_sha256": artifact.feature_spec_receipt_sha256,
        "feature_source_sha256": artifact.feature_source_sha256,
        "bars_feature_names": MASSIVE_ROLLING_BARS_V0_FIELDS,
        "tape_feature_names": MASSIVE_ROLLING_TAPE_V0_FIELDS,
        "rows": tuple(asdict(row) for row in artifact.rows),
        "row_inventory_sha256": artifact.row_inventory_sha256,
        "maximum_input_session_date": artifact.maximum_input_session_date,
    }


def materialize_massive_rolling_features_v0(
    *,
    daily_feature_root: str | Path,
    output_root: str | Path,
    bars_artifacts: Sequence[MassiveDailyBarsArtifactV0],
    tape_artifacts: Sequence[MassiveDailyTapeArtifactV0],
    identity_authority: PITSecurityUniverseAuthority,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveRollingFeatureArtifactV0:
    identity_authority.validate()
    bars = tuple(sorted(bars_artifacts, key=lambda row: row.source_session_date))
    tape = tuple(sorted(tape_artifacts, key=lambda row: row.source_session_date))
    if (
        not bars
        or tuple(row.source_session_date for row in bars)
        != tuple(row.source_session_date for row in tape)
        or len({row.source_session_date for row in bars}) != len(bars)
    ):
        raise MassiveRollingFeaturesV0Error("rolling daily support differs")
    for artifact in bars:
        validate_massive_daily_bars_v0(root=daily_feature_root, artifact=artifact)
    for artifact in tape:
        validate_massive_daily_tape_v0(root=daily_feature_root, artifact=artifact)
    condition_receipts = {
        artifact.condition_authority_receipt_sha256 for artifact in (*bars, *tape)
    }
    if len(condition_receipts) != 1:
        raise MassiveRollingFeaturesV0Error(
            "rolling condition authority changes inside one feature history"
        )
    current_date = bars[-1].source_session_date
    master = {row.security_id: row for row in identity_authority.security_master}
    bars_by_security: dict[str, list[object]] = {}
    for artifact in bars:
        for row in artifact.rows:
            bars_by_security.setdefault(row.security_id, []).append(row)
    current_tape = {row.security_id: row for row in tape[-1].rows}
    current_bars = {row.security_id: row for row in bars[-1].rows}
    rows = []
    for security_id in sorted(set(current_bars) & set(current_tape)):
        if security_id not in master:
            raise MassiveRollingFeaturesV0Error(
                "rolling security is absent from PIT identity"
            )
        bars_values, bars_valid = _bars_features(
            bars_by_security[security_id],
            listing_at_ms=master[security_id].listing_at_ms,
            source_session_date=current_date,
        )
        tape_row = current_tape[security_id]
        body = {
            "security_id": security_id,
            "bars_values": bars_values,
            "bars_valid": bars_valid,
            "tape_values": tuple(float(value) for value in tape_row.values),
            "tape_valid": tape_row.valid,
            "latest_daily_bars_row_receipt_sha256": current_bars[
                security_id
            ].receipt_sha256,
            "latest_daily_tape_row_receipt_sha256": tape_row.receipt_sha256,
        }
        row = MassiveRollingFeatureRowV0(
            **body,
            receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
        )
        row.validate()
        rows.append(row)
    ordered = tuple(rows)
    if not ordered:
        raise MassiveRollingFeaturesV0Error("rolling features have no common support")
    relative = f"massive-finalized-v0/session={current_date}/rolling-features.json"
    placeholder = MassiveRollingFeatureArtifactV0(
        source_session_date=current_date,
        daily_bars_artifact_receipts=tuple(sorted(row.receipt_sha256 for row in bars)),
        daily_tape_artifact_receipts=tuple(sorted(row.receipt_sha256 for row in tape)),
        identity_authority_receipt_sha256=identity_authority.receipt_sha256,
        condition_authority_receipt_sha256=next(iter(condition_receipts)),
        feature_spec_receipt_sha256=MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256,
        feature_source_sha256=MASSIVE_ROLLING_FEATURES_V0_SOURCE_SHA256,
        rows=ordered,
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        maximum_input_session_date=current_date,
        loaded_source=bars[-1].loaded_source,
        receipt_sha256="0" * 64,
    )
    payload = _payload(placeholder)
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ROLLING_FEATURES_V0_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root, relative_payload_path=relative, verified_at_ms=published_at_ms
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    validate_massive_rolling_features_v0(root=output_root, artifact=result)
    return result


def validate_massive_rolling_features_v0(
    *, root: str | Path, artifact: MassiveRollingFeatureArtifactV0
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveRollingFeaturesV0Error("rolling source is not JSON") from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveRollingFeaturesV0Error("rolling feature bytes differ")


__all__ = [
    "MASSIVE_ROLLING_BARS_V0_FIELDS",
    "MASSIVE_ROLLING_FEATURES_V0_DATASET",
    "MASSIVE_ROLLING_FEATURES_V0_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ROLLING_FEATURES_V0_SPEC_SHA256",
    "MASSIVE_ROLLING_TAPE_V0_FIELDS",
    "MassiveRollingFeatureArtifactV0",
    "MassiveRollingFeatureRowV0",
    "MassiveRollingFeaturesV0Error",
    "materialize_massive_rolling_features_v0",
    "validate_massive_rolling_features_v0",
]
