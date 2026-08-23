"""File-backed extraction and canonical feature artifacts for replay parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256,
    MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
    MassiveTradeExtractionEvidence,
)
from rl_quant.data_sources.massive.trade_replay import MassiveTradeReplayResult
from rl_quant.data_sources.massive.websocket_capture import (
    MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
    MASSIVE_WEBSOCKET_CAPTURE_PARSER_SOURCE_SHA256,
    MassiveDelayedWebSocketCaptureAuthority,
    MassiveDelayedWebSocketEvent,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_payload,
    file_sha256,
    semantic_sha256,
)


MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA = (
    "rl-quant.massive-trade-extraction-manifest-v2"
)
MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_WEBSOCKET_CAPTURE_FILE_SCHEMA,
        "parser": "canonical-jsonl-server-batch-v1",
        "selection": "exact-pit-ticker-and-session-date",
        "availability": "actual-local-receive-time",
    }
)
MASSIVE_REPLAY_FEATURE_SPEC_SCHEMA = "rl-quant.massive-replay-feature-spec-v1"
MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA = (
    "rl-quant.massive-replay-feature-artifact-v3"
)
MASSIVE_REPLAY_FEATURE_IDS = (
    "active_state_inventory_sha256",
    "active_event_count",
    "cancelled_event_count",
    "open_close_event_count",
    "high_low_event_count",
    "volume_event_count",
    "share_volume_decimal",
    "dollar_volume_decimal",
)
MASSIVE_REPLAY_FEATURE_SCHEMA_SHA256 = semantic_sha256(MASSIVE_REPLAY_FEATURE_IDS)
MASSIVE_REPLAY_FEATURE_ALGORITHM_SPEC_SHA256 = semantic_sha256(
    {
        "algorithm": "massive-replay-parity-features-v2",
        "feature_ids": MASSIVE_REPLAY_FEATURE_IDS,
        "volume_source": "active-events-with-updates-volume",
        "dollar_volume": "decimal-size-times-normalized-price",
    }
)
# Bind the implementation bytes as well as the descriptive algorithm contract.
MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256 = file_sha256(Path(__file__))


class MassiveReplayArtifactError(ValueError):
    """A replay extraction or feature artifact is not evidence-derived."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveReplayArtifactError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveTradeExtractionManifest:
    source_receipt_sha256: str
    source_commit_receipt_sha256: str
    loaded_source_receipt_sha256: str
    parser_spec_sha256: str
    parser_source_sha256: str | None
    parser_evidence_receipt_sha256: str | None
    security_id: str
    session_date: str
    selected_row_count: int
    selected_source_row_inventory_sha256: str
    selected_row_provenance_inventory_sha256: str | None
    complete_for_security_session: bool
    canonical_parser_qualified: bool
    receipt_sha256: str
    schema: str = MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA:
            raise MassiveReplayArtifactError("extraction manifest schema drifted")
        for name in (
            "source_receipt_sha256",
            "source_commit_receipt_sha256",
            "loaded_source_receipt_sha256",
            "parser_spec_sha256",
            "selected_source_row_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if not self.security_id or not self.session_date:
            raise MassiveReplayArtifactError("extraction identity is absent")
        if (
            isinstance(self.selected_row_count, bool)
            or not isinstance(self.selected_row_count, int)
            or self.selected_row_count <= 0
        ):
            raise MassiveReplayArtifactError("extraction must select at least one row")
        if self.complete_for_security_session is not True:
            raise MassiveReplayArtifactError("security-session extraction is incomplete")
        if not isinstance(self.canonical_parser_qualified, bool):
            raise MassiveReplayArtifactError("parser qualification must be Boolean")
        if self.canonical_parser_qualified:
            for name in (
                "parser_source_sha256",
                "parser_evidence_receipt_sha256",
                "selected_row_provenance_inventory_sha256",
            ):
                _digest(name, getattr(self, name))
            if self.parser_spec_sha256 not in {
                MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256,
                MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
            }:
                raise MassiveReplayArtifactError("canonical parser spec is unrecognized")
            expected_source = (
                MASSIVE_WEBSOCKET_CAPTURE_PARSER_SOURCE_SHA256
                if self.parser_spec_sha256
                == MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256
                else MASSIVE_FLAT_TRADE_PARSER_SOURCE_SHA256
            )
            if self.parser_source_sha256 != expected_source:
                raise MassiveReplayArtifactError("canonical parser source differs")
        elif any(
            value is not None
            for value in (
                self.parser_source_sha256,
                self.parser_evidence_receipt_sha256,
                self.selected_row_provenance_inventory_sha256,
            )
        ):
            raise MassiveReplayArtifactError(
                "unqualified extraction claims parser evidence"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayArtifactError("extraction manifest receipt differs")

    @classmethod
    def from_loaded_replay(
        cls,
        *,
        loaded_source: LoadedMassiveSourceObject,
        replay: MassiveTradeReplayResult,
        parser_spec_sha256: str,
    ) -> MassiveTradeExtractionManifest:
        """Bind a replay inventory to a source transaction reopened from disk."""

        loaded_source.validate()
        replay.validate()
        if replay.source_object_receipt_sha256 != loaded_source.receipt.receipt_sha256:
            raise MassiveReplayArtifactError("replay used another source object")
        parser = _digest("parser spec", parser_spec_sha256)
        body = {
            "schema": MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA,
            "source_receipt_sha256": loaded_source.receipt.receipt_sha256,
            "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
            "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            "parser_spec_sha256": parser,
            "parser_source_sha256": None,
            "parser_evidence_receipt_sha256": None,
            "security_id": replay.security_id,
            "session_date": replay.session_date,
            "selected_row_count": replay.input_event_count,
            "selected_source_row_inventory_sha256": replay.input_source_record_inventory_sha256,
            "selected_row_provenance_inventory_sha256": None,
            "complete_for_security_session": True,
            "canonical_parser_qualified": False,
        }
        value = cls(
            source_receipt_sha256=loaded_source.receipt.receipt_sha256,
            source_commit_receipt_sha256=loaded_source.commit.receipt_sha256,
            loaded_source_receipt_sha256=loaded_source.receipt_sha256,
            parser_spec_sha256=parser,
            parser_source_sha256=None,
            parser_evidence_receipt_sha256=None,
            security_id=replay.security_id,
            session_date=replay.session_date,
            selected_row_count=replay.input_event_count,
            selected_source_row_inventory_sha256=replay.input_source_record_inventory_sha256,
            selected_row_provenance_inventory_sha256=None,
            complete_for_security_session=True,
            canonical_parser_qualified=False,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value

    @classmethod
    def from_flat_file_evidence(
        cls,
        *,
        evidence: MassiveTradeExtractionEvidence,
        replay: MassiveTradeReplayResult,
    ) -> MassiveTradeExtractionManifest:
        """Accept only the parser that consumed every committed flat-file row."""

        evidence.validate()
        replay.validate()
        if evidence.security_id != replay.security_id or evidence.session_date != replay.session_date:
            raise MassiveReplayArtifactError("flat extraction and replay identities differ")
        if evidence.source_receipt_sha256 != replay.source_object_receipt_sha256:
            raise MassiveReplayArtifactError("flat extraction and replay sources differ")
        if evidence.selected_row_count != replay.input_event_count:
            raise MassiveReplayArtifactError("flat extraction and replay counts differ")
        if (
            evidence.selected_raw_source_record_inventory_sha256
            != replay.input_source_record_inventory_sha256
        ):
            raise MassiveReplayArtifactError("flat extraction row inventory differs")
        body = {
            "schema": MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA,
            "source_receipt_sha256": evidence.source_receipt_sha256,
            "source_commit_receipt_sha256": evidence.source_commit_receipt_sha256,
            "loaded_source_receipt_sha256": evidence.loaded_source_receipt_sha256,
            "parser_spec_sha256": evidence.parser_spec_sha256,
            "parser_source_sha256": evidence.parser_source_sha256,
            "parser_evidence_receipt_sha256": evidence.receipt_sha256,
            "security_id": replay.security_id,
            "session_date": replay.session_date,
            "selected_row_count": replay.input_event_count,
            "selected_source_row_inventory_sha256": replay.input_source_record_inventory_sha256,
            "selected_row_provenance_inventory_sha256": evidence.selected_row_provenance_inventory_sha256,
            "complete_for_security_session": evidence.complete_for_security_session,
            "canonical_parser_qualified": True,
        }
        value = cls(
            source_receipt_sha256=evidence.source_receipt_sha256,
            source_commit_receipt_sha256=evidence.source_commit_receipt_sha256,
            loaded_source_receipt_sha256=evidence.loaded_source_receipt_sha256,
            parser_spec_sha256=evidence.parser_spec_sha256,
            parser_source_sha256=evidence.parser_source_sha256,
            parser_evidence_receipt_sha256=evidence.receipt_sha256,
            security_id=replay.security_id,
            session_date=replay.session_date,
            selected_row_count=replay.input_event_count,
            selected_source_row_inventory_sha256=replay.input_source_record_inventory_sha256,
            selected_row_provenance_inventory_sha256=evidence.selected_row_provenance_inventory_sha256,
            complete_for_security_session=evidence.complete_for_security_session,
            canonical_parser_qualified=True,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value

    @classmethod
    def from_delayed_capture(
        cls,
        *,
        loaded_source: LoadedMassiveSourceObject,
        capture: MassiveDelayedWebSocketCaptureAuthority,
        capture_events: tuple[MassiveDelayedWebSocketEvent, ...],
        replay: MassiveTradeReplayResult,
        source_ticker: str,
    ) -> MassiveTradeExtractionManifest:
        """Bind a replay to every matching row parsed from committed capture bytes."""

        loaded_source.validate()
        capture.validate()
        replay.validate()
        if not source_ticker or source_ticker != source_ticker.strip():
            raise MassiveReplayArtifactError("delayed source ticker is not canonical")
        if not capture.capture_file_parser_qualified:
            raise MassiveReplayArtifactError(
                "delayed capture was not derived by the canonical file parser"
            )
        if capture.loaded_source_receipt_sha256 != loaded_source.receipt_sha256:
            raise MassiveReplayArtifactError("delayed capture used another loaded source")
        parser_evidence = capture.parser_evidence
        if parser_evidence is None:  # defensive against property/field divergence
            raise MassiveReplayArtifactError(
                "delayed capture parser evidence is absent"
            )
        if (
            capture.raw_capture_source_receipt_sha256
            != loaded_source.receipt.receipt_sha256
            or replay.source_object_receipt_sha256
            != loaded_source.receipt.receipt_sha256
        ):
            raise MassiveReplayArtifactError("delayed extraction source differs")
        for event in capture_events:
            event.validate()
        if len(capture_events) != capture.event_count:
            raise MassiveReplayArtifactError("capture event count differs from parsed rows")
        event_inventory = semantic_sha256(
            tuple(
                sorted(
                    (
                        event.received_at_ns,
                        event.ticker,
                        event.sequence_number,
                        event.payload_sha256,
                    )
                    for event in capture_events
                )
            )
        )
        payload_inventory = semantic_sha256(
            tuple(
                sorted(
                    (event.ticker, event.sequence_number, event.payload_sha256)
                    for event in capture_events
                )
            )
        )
        if (
            event_inventory != capture.event_inventory_sha256
            or payload_inventory != capture.payload_inventory_sha256
        ):
            raise MassiveReplayArtifactError("capture inventory differs from parsed rows")
        selected = tuple(
            sorted(
                (
                    event.ticker,
                    event.sequence_number,
                    event.payload_sha256,
                )
                for event in capture_events
                if event.ticker == source_ticker
                and event.session_date == replay.session_date
            )
        )
        if not selected or len(selected) != replay.input_event_count:
            raise MassiveReplayArtifactError("delayed capture selection is incomplete")
        selected_inventory = semantic_sha256(selected)
        if selected_inventory != replay.input_source_record_inventory_sha256:
            raise MassiveReplayArtifactError("delayed capture row inventory differs")
        body = {
            "schema": MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA,
            "source_receipt_sha256": loaded_source.receipt.receipt_sha256,
            "source_commit_receipt_sha256": loaded_source.commit.receipt_sha256,
            "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            "parser_spec_sha256": MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256,
            "parser_source_sha256": parser_evidence.parser_source_sha256,
            "parser_evidence_receipt_sha256": parser_evidence.receipt_sha256,
            "security_id": replay.security_id,
            "session_date": replay.session_date,
            "selected_row_count": len(selected),
            "selected_source_row_inventory_sha256": selected_inventory,
            "selected_row_provenance_inventory_sha256": selected_inventory,
            "complete_for_security_session": True,
            "canonical_parser_qualified": True,
        }
        value = cls(
            source_receipt_sha256=loaded_source.receipt.receipt_sha256,
            source_commit_receipt_sha256=loaded_source.commit.receipt_sha256,
            loaded_source_receipt_sha256=loaded_source.receipt_sha256,
            parser_spec_sha256=MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256,
            parser_source_sha256=parser_evidence.parser_source_sha256,
            parser_evidence_receipt_sha256=parser_evidence.receipt_sha256,
            security_id=replay.security_id,
            session_date=replay.session_date,
            selected_row_count=len(selected),
            selected_source_row_inventory_sha256=selected_inventory,
            selected_row_provenance_inventory_sha256=selected_inventory,
            complete_for_security_session=True,
            canonical_parser_qualified=True,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveReplayFeatureSpec:
    feature_ids: tuple[str, ...]
    materializer_source_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_REPLAY_FEATURE_SPEC_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_REPLAY_FEATURE_SPEC_SCHEMA:
            raise MassiveReplayArtifactError("feature spec schema drifted")
        if self.feature_ids != MASSIVE_REPLAY_FEATURE_IDS:
            raise MassiveReplayArtifactError("feature spec inventory drifted")
        if self.materializer_source_sha256 != (
            MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256
        ):
            raise MassiveReplayArtifactError("feature materializer identity drifted")
        _digest("feature spec receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayArtifactError("feature spec receipt differs")

    @classmethod
    def canonical(cls) -> MassiveReplayFeatureSpec:
        body = {
            "schema": MASSIVE_REPLAY_FEATURE_SPEC_SCHEMA,
            "feature_ids": MASSIVE_REPLAY_FEATURE_IDS,
            "materializer_source_sha256": MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256,
        }
        value = cls(
            feature_ids=MASSIVE_REPLAY_FEATURE_IDS,
            materializer_source_sha256=MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveReplayFeatureArtifact:
    security_id: str
    session_date: str
    input_replay_receipt_sha256: str
    feature_spec_receipt_sha256: str
    feature_materializer_source_sha256: str
    feature_schema_sha256: str
    canonical_feature_payload_json: str
    output_feature_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA

    @property
    def feature_payload_sha256(self) -> str:
        return self.output_feature_receipt_sha256

    @property
    def source_replay_receipt_sha256(self) -> str:
        return self.input_replay_receipt_sha256

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA:
            raise MassiveReplayArtifactError("feature artifact schema drifted")
        if not self.security_id or not self.session_date:
            raise MassiveReplayArtifactError("feature identity is absent")
        for name in (
            "input_replay_receipt_sha256",
            "feature_spec_receipt_sha256",
            "feature_materializer_source_sha256",
            "feature_schema_sha256",
            "output_feature_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.feature_materializer_source_sha256 != (
            MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256
        ):
            raise MassiveReplayArtifactError("feature materializer source drifted")
        canonical_spec = MassiveReplayFeatureSpec.canonical()
        if self.feature_spec_receipt_sha256 != canonical_spec.receipt_sha256:
            raise MassiveReplayArtifactError("feature artifact used a noncanonical spec")
        if self.feature_schema_sha256 != MASSIVE_REPLAY_FEATURE_SCHEMA_SHA256:
            raise MassiveReplayArtifactError("feature artifact schema receipt drifted")
        payload = self.canonical_feature_payload_json.encode("ascii")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict) or tuple(sorted(parsed)) != tuple(
            sorted(MASSIVE_REPLAY_FEATURE_IDS)
        ):
            raise MassiveReplayArtifactError("feature payload inventory drifted")
        for name in MASSIVE_REPLAY_FEATURE_IDS[1:6]:
            value = parsed[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveReplayArtifactError(
                    f"feature payload {name} must be a nonnegative integer"
                )
        _digest(
            "active-state feature",
            parsed["active_state_inventory_sha256"],
        )
        for name in ("share_volume_decimal", "dollar_volume_decimal"):
            value = parsed[name]
            if not isinstance(value, str):
                raise MassiveReplayArtifactError(
                    f"feature payload {name} must be decimal text"
                )
            try:
                number = Decimal(value)
            except (InvalidOperation, ValueError) as exc:
                raise MassiveReplayArtifactError(
                    f"feature payload {name} is not decimal"
                ) from exc
            if not number.is_finite() or number < 0:
                raise MassiveReplayArtifactError(
                    f"feature payload {name} must be finite and nonnegative"
                )
            if format(number.normalize(), "f") != value:
                raise MassiveReplayArtifactError(
                    f"feature payload {name} is not canonical decimal text"
                )
        if canonical_json_payload(parsed) != payload:
            raise MassiveReplayArtifactError("feature payload is not canonical JSON")
        if self.output_feature_receipt_sha256 != semantic_sha256(parsed):
            raise MassiveReplayArtifactError("feature output receipt differs")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayArtifactError("feature artifact receipt differs")


def materialize_massive_replay_features(
    replay: MassiveTradeReplayResult,
    *,
    specification: MassiveReplayFeatureSpec,
) -> MassiveReplayFeatureArtifact:
    """Produce the only accepted feature-parity payload from replay state."""

    replay.validate()
    specification.validate()
    share_volume = sum(
        (
            Decimal(event.decimal_size)
            for event in replay.active_events
            if event.updates_volume
        ),
        start=Decimal(0),
    )
    dollar_volume = sum(
        (
            Decimal(event.decimal_size) * Decimal(str(event.price))
            for event in replay.active_events
            if event.updates_volume
        ),
        start=Decimal(0),
    )
    payload = {
        "active_state_inventory_sha256": replay.active_state_inventory_sha256,
        "active_event_count": len(replay.active_events),
        "cancelled_event_count": len(replay.cancelled_event_keys),
        "open_close_event_count": sum(
            event.updates_open_close for event in replay.active_events
        ),
        "high_low_event_count": sum(
            event.updates_high_low for event in replay.active_events
        ),
        "volume_event_count": sum(
            event.updates_volume for event in replay.active_events
        ),
        "share_volume_decimal": format(share_volume.normalize(), "f"),
        "dollar_volume_decimal": format(dollar_volume.normalize(), "f"),
    }
    canonical = canonical_json_payload(payload).decode("ascii")
    body = {
        "schema": MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA,
        "security_id": replay.security_id,
        "session_date": replay.session_date,
        "input_replay_receipt_sha256": replay.receipt_sha256,
        "feature_spec_receipt_sha256": specification.receipt_sha256,
        "feature_materializer_source_sha256": specification.materializer_source_sha256,
        "feature_schema_sha256": MASSIVE_REPLAY_FEATURE_SCHEMA_SHA256,
        "canonical_feature_payload_json": canonical,
        "output_feature_receipt_sha256": semantic_sha256(payload),
    }
    value = MassiveReplayFeatureArtifact(
        **body,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


__all__ = [
    "MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256",
    "MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA",
    "MASSIVE_REPLAY_FEATURE_ALGORITHM_SPEC_SHA256",
    "MASSIVE_REPLAY_FEATURE_IDS",
    "MASSIVE_REPLAY_FEATURE_MATERIALIZER_SOURCE_SHA256",
    "MASSIVE_REPLAY_FEATURE_SCHEMA_SHA256",
    "MASSIVE_REPLAY_FEATURE_SPEC_SCHEMA",
    "MASSIVE_TRADE_EXTRACTION_MANIFEST_SCHEMA",
    "MassiveReplayArtifactError",
    "MassiveReplayFeatureArtifact",
    "MassiveReplayFeatureSpec",
    "MassiveTradeExtractionManifest",
    "materialize_massive_replay_features",
]
