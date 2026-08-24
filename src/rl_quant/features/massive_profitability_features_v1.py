"""Minimal session-aligned bars and tape features for profitability P0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_economic_return_index_v1 import (
    MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256,
    MassiveEconomicReturnIndexArtifactV1,
    MassiveEconomicReturnRowV1,
    validate_massive_economic_return_index_v1,
)
from rl_quant.features.massive_session_panel_v1 import (
    MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
    MassiveSessionPanelArtifactV1,
    MassiveSessionPanelRowV1,
    validate_massive_session_panel_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)


BARS_MIN_V1_FIELDS = (
    "economic_total_return_1",
    "economic_total_return_5",
    "economic_total_return_21",
    "economic_total_return_63",
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
    "listing_age_sessions",
    "valid_history_fraction_63",
)
TAPE_MIN_V1_FIELDS = (
    "log_trade_count",
    "median_trade_size",
    "p90_trade_size",
    "large_trade_dollar_fraction",
    "signed_dollar_flow_fraction",
    "absolute_signed_flow_imbalance",
    "trf_off_exchange_dollar_fraction",
    "venue_entropy",
    "largest_venue_share",
    "tape_a_fraction",
    "tape_b_fraction",
    "tape_c_fraction",
    "replacement_event_fraction",
    "cancellation_event_fraction",
    "late_report_event_fraction",
)
MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA = "rl-quant.massive-profitability-features-v1"
MASSIVE_PROFITABILITY_FEATURES_V1_DATASET = (
    "massive-finalized-profitability-features-v1"
)
MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_FEATURES_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "session_panel_spec": MASSIVE_SESSION_PANEL_V1_SPEC_SHA256,
        "economic_index_spec": MASSIVE_ECONOMIC_RETURN_INDEX_V1_SPEC_SHA256,
        "bars_fields": BARS_MIN_V1_FIELDS,
        "tape_fields": TAPE_MIN_V1_FIELDS,
        "offsets": "exact-XNYS-session-index-never-observed-row-index",
        "return": "economic-value[t]/economic-value[t-h]-1",
        "reversal": "negative-economic-total-return",
        "trend": "economic-return-over-[t-21,t-5]-and-[t-63,t-21]",
        "volatility_minimums": {"5": 4, "21": 16},
        "volume_surprise": "log1p(current)-log1p(mean(prior-21))",
        "amihud": "mean(abs(exact-one-session-economic-return)/dollar-volume)",
        "history_fraction": "valid-economic-value-count-over-exact-63-sessions",
        "correction_denominator": "complete-event-timeline",
        "signed_flow": "quote-free-proxy-divided-by-total-dollar-volume",
        "excluded_tape_fields": (
            "special_condition_fraction",
            "price_response_per_signed_dollar",
        ),
        "missing": "zero-value-plus-independent-false-mask",
        "aggressor_side_claimed": False,
        "institutional_flow_claimed": False,
    }
)
MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA,
        "row_key": ("source_session_index", "security_id"),
        "bars_fields": BARS_MIN_V1_FIELDS,
        "tape_fields": TAPE_MIN_V1_FIELDS,
    }
)


class MassiveProfitabilityFeaturesV1Error(ValueError):
    """The session-aligned profitability feature contract differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityFeaturesV1Error(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFeatureRowV1:
    source_session_index: int
    source_session_date: str
    security_id: str
    pit_member: bool
    listed: bool
    tradable: bool
    observed_regular_trade: bool
    halt_or_no_print: bool
    bars_values: tuple[float, ...]
    bars_valid: tuple[bool, ...]
    tape_values: tuple[float, ...]
    tape_valid: tuple[bool, ...]
    session_panel_row_receipt_sha256: str
    economic_return_row_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            isinstance(self.source_session_index, bool)
            or not isinstance(self.source_session_index, int)
            or self.source_session_index < 0
            or not self.source_session_date
            or not self.security_id
        ):
            raise MassiveProfitabilityFeaturesV1Error("feature row identity is invalid")
        for value in (
            self.pit_member,
            self.listed,
            self.tradable,
            self.observed_regular_trade,
            self.halt_or_no_print,
        ):
            if not isinstance(value, bool):
                raise MassiveProfitabilityFeaturesV1Error(
                    "feature row state must be Boolean"
                )
        for values, masks, fields in (
            (self.bars_values, self.bars_valid, BARS_MIN_V1_FIELDS),
            (self.tape_values, self.tape_valid, TAPE_MIN_V1_FIELDS),
        ):
            if (
                len(values) != len(fields)
                or len(masks) != len(fields)
                or any(not isinstance(flag, bool) for flag in masks)
                or any(not math.isfinite(float(value)) for value in values)
                or any(value != 0.0 for value, valid in zip(values, masks) if not valid)
            ):
                raise MassiveProfitabilityFeaturesV1Error(
                    "feature values and masks differ"
                )
        for receipt in (
            self.session_panel_row_receipt_sha256,
            self.economic_return_row_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("feature row digest", receipt)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityFeaturesV1Error("feature row receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityFeatureArtifactV1:
    session_panel_receipt_sha256: str
    economic_return_index_receipt_sha256: str
    rows: tuple[MassiveProfitabilityFeatureRowV1, ...]
    row_count: int
    member_row_count: int
    row_inventory_sha256: str
    feature_spec_receipt_sha256: str
    feature_source_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_FEATURES_V1_SCHEMA:
            raise MassiveProfitabilityFeaturesV1Error("feature artifact schema drifted")
        for receipt in (
            self.session_panel_receipt_sha256,
            self.economic_return_index_receipt_sha256,
            self.row_inventory_sha256,
            self.feature_spec_receipt_sha256,
            self.feature_source_sha256,
            self.receipt_sha256,
        ):
            _digest("feature artifact digest", receipt)
        if (
            self.feature_spec_receipt_sha256
            != MASSIVE_PROFITABILITY_FEATURES_V1_SPEC_SHA256
            or self.feature_source_sha256
            != MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityFeaturesV1Error("feature implementation drifted")
        keys = tuple((row.source_session_index, row.security_id) for row in self.rows)
        if not keys or keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityFeaturesV1Error(
                "feature artifact rows are not canonical"
            )
        for row in self.rows:
            row.validate()
        if (
            self.row_count != len(self.rows)
            or self.member_row_count != sum(row.pit_member for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
        ):
            raise MassiveProfitabilityFeaturesV1Error("feature artifact counts differ")
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_FEATURES_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityFeaturesV1Error(
                "feature artifact source contract differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityFeaturesV1Error(
                "feature artifact receipt differs"
            )


def _daily_value(
    row: MassiveSessionPanelRowV1,
    fields: tuple[str, ...],
    name: str,
    *,
    tape: bool = False,
) -> tuple[float, bool]:
    index = fields.index(name)
    values = row.tape_values if tape else row.bars_values
    masks = row.tape_valid if tape else row.bars_valid
    return float(values[index]), masks[index]


def _economic_return(
    rows: dict[int, MassiveEconomicReturnRowV1],
    end_index: int,
    horizon: int,
) -> tuple[float, bool]:
    start = rows.get(end_index - horizon)
    end = rows.get(end_index)
    if (
        start is None
        or end is None
        or not start.economic_value_valid
        or not end.economic_value_valid
        or start.economic_value <= 0.0
    ):
        return 0.0, False
    return end.economic_value / start.economic_value - 1.0, True


def _exact_daily_returns(
    rows: dict[int, MassiveEconomicReturnRowV1],
    end_index: int,
    count: int,
) -> list[tuple[int, float]]:
    output: list[tuple[int, float]] = []
    for index in range(end_index - count + 1, end_index + 1):
        value, valid = _economic_return(rows, index, 1)
        if valid:
            output.append((index, value))
    return output


def _bars_features(
    *,
    row: MassiveSessionPanelRowV1,
    panel_by_index: dict[int, MassiveSessionPanelRowV1],
    economic_by_index: dict[int, MassiveEconomicReturnRowV1],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values: dict[str, float] = {name: 0.0 for name in BARS_MIN_V1_FIELDS}
    valid: dict[str, bool] = {name: False for name in BARS_MIN_V1_FIELDS}
    returns: dict[int, float] = {}
    for horizon in (1, 5, 21, 63):
        value, is_valid = _economic_return(
            economic_by_index, row.source_session_index, horizon
        )
        name = f"economic_total_return_{horizon}"
        values[name] = value if is_valid else 0.0
        valid[name] = is_valid
        if is_valid:
            returns[horizon] = value
    for horizon in (1, 5):
        name = f"reversal_{horizon}"
        if horizon in returns:
            values[name] = -returns[horizon]
            valid[name] = True
    for name, end_offset, interval in (
        ("trend_21_minus_5", 5, 16),
        ("trend_63_minus_21", 21, 42),
    ):
        trend, trend_valid = _economic_return(
            economic_by_index,
            row.source_session_index - end_offset,
            interval,
        )
        if trend_valid:
            values[name] = trend
            valid[name] = True
    for window, minimum in ((5, 4), (21, 16)):
        daily = _exact_daily_returns(
            economic_by_index, row.source_session_index, window
        )
        name = f"realized_volatility_{window}"
        if len(daily) >= minimum:
            values[name] = pstdev(value for _, value in daily)
            valid[name] = True
        if window == 21 and len(daily) >= minimum:
            values["downside_volatility_21"] = math.sqrt(
                fmean(min(value, 0.0) ** 2 for _, value in daily)
            )
            valid["downside_volatility_21"] = True
    for name in ("high_low_range", "close_location"):
        value, is_valid = _daily_value(row, MASSIVE_DAILY_BARS_V0_FIELDS, name)
        values[name] = value if is_valid else 0.0
        valid[name] = is_valid
    dollar, dollar_valid = _daily_value(
        row, MASSIVE_DAILY_BARS_V0_FIELDS, "dollar_volume"
    )
    if dollar_valid and dollar >= 0.0:
        values["log_dollar_volume"] = math.log1p(dollar)
        valid["log_dollar_volume"] = True
    prior_dollars: list[float] = []
    complete_prior_window = row.source_session_index >= 21
    if complete_prior_window:
        for index in range(row.source_session_index - 21, row.source_session_index):
            candidate = panel_by_index[index]
            candidate_value, candidate_valid = _daily_value(
                candidate, MASSIVE_DAILY_BARS_V0_FIELDS, "dollar_volume"
            )
            if candidate_valid and candidate_value >= 0.0:
                prior_dollars.append(candidate_value)
    if dollar_valid and dollar >= 0.0 and len(prior_dollars) >= 16:
        values["dollar_volume_surprise_21"] = math.log1p(dollar) - math.log1p(
            fmean(prior_dollars)
        )
        valid["dollar_volume_surprise_21"] = True
    daily_21 = _exact_daily_returns(economic_by_index, row.source_session_index, 21)
    amihud: list[float] = []
    for index, daily_return in daily_21:
        candidate = panel_by_index[index]
        candidate_dollar, candidate_valid = _daily_value(
            candidate, MASSIVE_DAILY_BARS_V0_FIELDS, "dollar_volume"
        )
        if candidate_valid and candidate_dollar > 0.0:
            amihud.append(abs(daily_return) / candidate_dollar)
    if len(amihud) >= 16:
        values["amihud_21"] = fmean(amihud)
        valid["amihud_21"] = True
    listed_indices = tuple(
        index for index, candidate in panel_by_index.items() if candidate.listed
    )
    if row.listed and listed_indices:
        values["listing_age_sessions"] = float(
            row.source_session_index - min(listed_indices) + 1
        )
        valid["listing_age_sessions"] = True
    if row.source_session_index >= 62:
        history = tuple(
            economic_by_index[index]
            for index in range(
                row.source_session_index - 62, row.source_session_index + 1
            )
        )
        values["valid_history_fraction_63"] = (
            sum(candidate.economic_value_valid for candidate in history) / 63.0
        )
        valid["valid_history_fraction_63"] = True
    return (
        tuple(values[name] for name in BARS_MIN_V1_FIELDS),
        tuple(valid[name] for name in BARS_MIN_V1_FIELDS),
    )


def _tape_features(
    row: MassiveSessionPanelRowV1,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    values: dict[str, float] = {name: 0.0 for name in TAPE_MIN_V1_FIELDS}
    valid: dict[str, bool] = {name: False for name in TAPE_MIN_V1_FIELDS}
    direct = (
        "log_trade_count",
        "median_trade_size",
        "p90_trade_size",
        "large_trade_dollar_fraction",
        "absolute_signed_flow_imbalance",
        "trf_off_exchange_dollar_fraction",
        "venue_entropy",
        "largest_venue_share",
        "tape_a_fraction",
        "tape_b_fraction",
        "tape_c_fraction",
    )
    for name in direct:
        value, is_valid = _daily_value(
            row, MASSIVE_DAILY_TAPE_V0_FIELDS, name, tape=True
        )
        values[name] = value if is_valid else 0.0
        valid[name] = is_valid
    signed_flow, signed_valid = _daily_value(
        row,
        MASSIVE_DAILY_TAPE_V0_FIELDS,
        "quote_free_signed_dollar_flow_proxy",
        tape=True,
    )
    dollar, dollar_valid = _daily_value(
        row, MASSIVE_DAILY_BARS_V0_FIELDS, "dollar_volume"
    )
    if signed_valid and dollar_valid and dollar > 0.0:
        values["signed_dollar_flow_fraction"] = signed_flow / dollar
        valid["signed_dollar_flow_fraction"] = True
    if row.event_timeline_count > 0:
        denominator = float(row.event_timeline_count)
        for name, count in (
            ("replacement_event_fraction", row.replacement_event_count),
            ("cancellation_event_fraction", row.cancellation_event_count),
            ("late_report_event_fraction", row.late_report_event_count),
        ):
            values[name] = count / denominator
            valid[name] = True
    return (
        tuple(values[name] for name in TAPE_MIN_V1_FIELDS),
        tuple(valid[name] for name in TAPE_MIN_V1_FIELDS),
    )


def build_massive_profitability_feature_rows_v1(
    *,
    panel_rows: Sequence[MassiveSessionPanelRowV1],
    economic_rows: Sequence[MassiveEconomicReturnRowV1],
) -> tuple[MassiveProfitabilityFeatureRowV1, ...]:
    """Derive every feature on exact session coordinates and explicit masks."""

    panels = tuple(panel_rows)
    economics = tuple(economic_rows)
    panel_keys = tuple((row.source_session_index, row.security_id) for row in panels)
    economic_keys = tuple(
        (row.source_session_index, row.security_id) for row in economics
    )
    if (
        not panel_keys
        or panel_keys != tuple(sorted(set(panel_keys)))
        or panel_keys != economic_keys
    ):
        raise MassiveProfitabilityFeaturesV1Error(
            "panel and economic row support differs"
        )
    for panel_row in panels:
        panel_row.validate()
    for economic_row in economics:
        economic_row.validate()
    panel_by_security: dict[str, dict[int, MassiveSessionPanelRowV1]] = {}
    economic_by_security: dict[str, dict[int, MassiveEconomicReturnRowV1]] = {}
    for panel_row in panels:
        panel_by_security.setdefault(panel_row.security_id, {})[
            panel_row.source_session_index
        ] = panel_row
    for economic_row in economics:
        economic_by_security.setdefault(economic_row.security_id, {})[
            economic_row.source_session_index
        ] = economic_row
    output: list[MassiveProfitabilityFeatureRowV1] = []
    for panel, economic in zip(panels, economics, strict=True):
        if (
            economic.source_session_date != panel.source_session_date
            or economic.listed != panel.listed
            or economic.session_panel_row_receipt_sha256 != panel.receipt_sha256
        ):
            raise MassiveProfitabilityFeaturesV1Error(
                "economic row does not derive from its session-panel row"
            )
        bars_values, bars_valid = _bars_features(
            row=panel,
            panel_by_index=panel_by_security[panel.security_id],
            economic_by_index=economic_by_security[panel.security_id],
        )
        tape_values, tape_valid = _tape_features(panel)
        provisional = MassiveProfitabilityFeatureRowV1(
            source_session_index=panel.source_session_index,
            source_session_date=panel.source_session_date,
            security_id=panel.security_id,
            pit_member=panel.pit_member,
            listed=panel.listed,
            tradable=panel.tradable,
            observed_regular_trade=panel.observed_regular_trade,
            halt_or_no_print=panel.halt_or_no_print,
            bars_values=bars_values,
            bars_valid=bars_valid,
            tape_values=tape_values,
            tape_valid=tape_valid,
            session_panel_row_receipt_sha256=panel.receipt_sha256,
            economic_return_row_receipt_sha256=economic.receipt_sha256,
            receipt_sha256="0" * 64,
        )
        feature_row = replace(
            provisional,
            receipt_sha256=semantic_sha256(provisional.unsigned()),
        )
        feature_row.validate()
        output.append(feature_row)
    return tuple(output)


def _payload(artifact: MassiveProfitabilityFeatureArtifactV1) -> dict[str, object]:
    return {
        "schema": artifact.schema,
        "session_panel_receipt_sha256": artifact.session_panel_receipt_sha256,
        "economic_return_index_receipt_sha256": (
            artifact.economic_return_index_receipt_sha256
        ),
        "bars_feature_names": BARS_MIN_V1_FIELDS,
        "tape_feature_names": TAPE_MIN_V1_FIELDS,
        "rows": tuple(asdict(row) for row in artifact.rows),
        "row_count": artifact.row_count,
        "member_row_count": artifact.member_row_count,
        "row_inventory_sha256": artifact.row_inventory_sha256,
        "feature_spec_receipt_sha256": artifact.feature_spec_receipt_sha256,
        "feature_source_sha256": artifact.feature_source_sha256,
    }


def materialize_massive_profitability_features_v1(
    *,
    session_panel_root: str | Path,
    economic_index_root: str | Path,
    output_root: str | Path,
    session_panel: MassiveSessionPanelArtifactV1,
    economic_index: MassiveEconomicReturnIndexArtifactV1,
    entitlement_receipt_sha256: str,
    published_at_ms: int,
) -> MassiveProfitabilityFeatureArtifactV1:
    validate_massive_session_panel_v1(root=session_panel_root, artifact=session_panel)
    validate_massive_economic_return_index_v1(
        root=economic_index_root, artifact=economic_index
    )
    if economic_index.session_panel_receipt_sha256 != session_panel.receipt_sha256:
        raise MassiveProfitabilityFeaturesV1Error(
            "economic index does not derive from the session panel"
        )
    rows = build_massive_profitability_feature_rows_v1(
        panel_rows=session_panel.rows,
        economic_rows=economic_index.rows,
    )
    relative = (
        "massive-profitability-p0/features-v1/"
        f"{session_panel.start_session_date}-{session_panel.end_session_date}.json"
    )
    placeholder = MassiveProfitabilityFeatureArtifactV1(
        session_panel_receipt_sha256=session_panel.receipt_sha256,
        economic_return_index_receipt_sha256=economic_index.receipt_sha256,
        rows=rows,
        row_count=len(rows),
        member_row_count=sum(row.pit_member for row in rows),
        row_inventory_sha256=semantic_sha256(tuple(row.receipt_sha256 for row in rows)),
        feature_spec_receipt_sha256=MASSIVE_PROFITABILITY_FEATURES_V1_SPEC_SHA256,
        feature_source_sha256=MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SHA256,
        loaded_source=session_panel.loaded_source,
        receipt_sha256="0" * 64,
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(placeholder))),
        root=output_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_FEATURES_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=published_at_ms,
        downloaded_at_ms=published_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=published_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=output_root,
        relative_payload_path=relative,
        verified_at_ms=published_at_ms,
    )
    provisional = replace(placeholder, loaded_source=loaded)
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    validate_massive_profitability_features_v1(root=output_root, artifact=result)
    return result


def validate_massive_profitability_features_v1(
    *,
    root: str | Path,
    artifact: MassiveProfitabilityFeatureArtifactV1,
) -> None:
    artifact.validate()
    raw = read_loaded_massive_source_bytes(
        root=root, loaded_source=artifact.loaded_source
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityFeaturesV1Error(
            "profitability feature source is not JSON"
        ) from exc
    if raw != canonical_json_file_bytes(payload) or raw != canonical_json_file_bytes(
        _payload(artifact)
    ):
        raise MassiveProfitabilityFeaturesV1Error("profitability feature bytes differ")


__all__ = [
    "BARS_MIN_V1_FIELDS",
    "MASSIVE_PROFITABILITY_FEATURES_V1_DATASET",
    "MASSIVE_PROFITABILITY_FEATURES_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_FEATURES_V1_SPEC_SHA256",
    "TAPE_MIN_V1_FIELDS",
    "MassiveProfitabilityFeatureArtifactV1",
    "MassiveProfitabilityFeatureRowV1",
    "MassiveProfitabilityFeaturesV1Error",
    "build_massive_profitability_feature_rows_v1",
    "materialize_massive_profitability_features_v1",
    "validate_massive_profitability_features_v1",
]
