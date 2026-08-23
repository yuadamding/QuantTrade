"""Committed-byte delayed-stream versus finalized-file replay qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from rl_quant.alpha.pit_universe import SourcedTickerHistoryRecord
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.entitlement import MassiveEntitlementAuthority
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import LoadedMassiveSourceObject
from rl_quant.data_sources.massive.trade_extraction import (
    MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256,
)
from rl_quant.data_sources.massive.trade_replay import MassiveTradeReplayResult
from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketCaptureAuthority,
)
from rl_quant.evaluation.massive_replay_artifacts import (
    MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256,
    MassiveReplayFeatureArtifact,
    MassiveTradeExtractionManifest,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA = "rl-quant.massive-delayed-replay-v3"
MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA = "rl-quant.massive-replay-parity-evidence-v2"
MASSIVE_TICKER_CHANGE_CANARY_SCHEMA = "rl-quant.massive-ticker-change-canary-v2"

MassiveReplayCanaryKind = Literal[
    "correction-activity",
    "early-close-session",
    "normal-session",
    "special-condition",
    "ticker-change-identity",
    "trf-trades",
]
REQUIRED_MASSIVE_REPLAY_CANARIES: tuple[MassiveReplayCanaryKind, ...] = (
    "correction-activity",
    "early-close-session",
    "normal-session",
    "special-condition",
    "ticker-change-identity",
    "trf-trades",
)


class MassiveReplayParityError(ValueError):
    """Delayed capture and finalized replay do not establish parity."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveReplayParityError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MassiveTickerChangeCanaryEvidence:
    prior_record: SourcedTickerHistoryRecord
    current_record: SourcedTickerHistoryRecord
    ticker_history_authority_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_TICKER_CHANGE_CANARY_SCHEMA

    @property
    def security_id(self) -> str:
        return self.current_record.security_id

    @property
    def current_ticker(self) -> str:
        return self.current_record.ticker

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prior_record": asdict(self.prior_record),
            "current_record": asdict(self.current_record),
            "ticker_history_authority_receipt_sha256": self.ticker_history_authority_receipt_sha256,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TICKER_CHANGE_CANARY_SCHEMA:
            raise MassiveReplayParityError("ticker-change canary schema drifted")
        self.prior_record.validate()
        self.current_record.validate()
        if self.prior_record.security_id != self.current_record.security_id:
            raise MassiveReplayParityError("ticker transition changed security identity")
        if self.prior_record.ticker == self.current_record.ticker:
            raise MassiveReplayParityError("ticker transition did not change ticker")
        if self.prior_record.valid_to_ms != self.current_record.valid_from_ms:
            raise MassiveReplayParityError("ticker records are not adjacent")
        if self.prior_record.primary_exchange != self.current_record.primary_exchange:
            raise MassiveReplayParityError("ticker transition changed primary exchange")
        _digest("ticker history authority", self.ticker_history_authority_receipt_sha256)
        _digest("ticker transition receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("ticker transition receipt differs")

    @classmethod
    def build(
        cls,
        *,
        prior_record: SourcedTickerHistoryRecord,
        current_record: SourcedTickerHistoryRecord,
        ticker_history_authority_receipt_sha256: str,
    ) -> MassiveTickerChangeCanaryEvidence:
        body = {
            "schema": MASSIVE_TICKER_CHANGE_CANARY_SCHEMA,
            "prior_record": asdict(prior_record),
            "current_record": asdict(current_record),
            "ticker_history_authority_receipt_sha256": ticker_history_authority_receipt_sha256,
        }
        value = cls(
            prior_record=prior_record,
            current_record=current_record,
            ticker_history_authority_receipt_sha256=ticker_history_authority_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveReplayParityInput:
    canary_kind: MassiveReplayCanaryKind
    capture: MassiveDelayedWebSocketCaptureAuthority
    delayed_source: LoadedMassiveSourceObject
    finalized_source: LoadedMassiveSourceObject
    delayed_extraction: MassiveTradeExtractionManifest
    finalized_extraction: MassiveTradeExtractionManifest
    decision_clock: MassiveDecisionClockAuthority
    session: MassiveExchangeSession
    delayed_replay: MassiveTradeReplayResult
    finalized_replay: MassiveTradeReplayResult
    delayed_features: MassiveReplayFeatureArtifact
    finalized_features: MassiveReplayFeatureArtifact
    ticker_change: MassiveTickerChangeCanaryEvidence | None = None


@dataclass(frozen=True, slots=True)
class MassiveReplayParityEvidence:
    canary_kind: MassiveReplayCanaryKind
    security_id: str
    session_date: str
    decision_clock_receipt_sha256: str
    websocket_capture_receipt_sha256: str
    delayed_loaded_source_receipt_sha256: str
    finalized_loaded_source_receipt_sha256: str
    delayed_extraction_receipt_sha256: str
    finalized_extraction_receipt_sha256: str
    delayed_replay_receipt_sha256: str
    finalized_replay_receipt_sha256: str
    delayed_active_state_sha256: str
    finalized_active_state_sha256: str
    delayed_feature_artifact_receipt_sha256: str
    finalized_feature_artifact_receipt_sha256: str
    feature_spec_receipt_sha256: str
    delayed_feature_sha256: str
    finalized_feature_sha256: str
    event_exact: bool
    feature_exact: bool
    canary_observed: bool
    capture_complete: bool
    committed_sources_complete: bool
    failure_reason: str | None
    receipt_sha256: str
    schema: str = MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate(self) -> None:
        if self.schema != MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA:
            raise MassiveReplayParityError("parity evidence schema drifted")
        if self.canary_kind not in REQUIRED_MASSIVE_REPLAY_CANARIES:
            raise MassiveReplayParityError("parity canary kind is unsupported")
        if not self.security_id or not self.session_date:
            raise MassiveReplayParityError("parity identity is absent")
        digest_fields = (
            "decision_clock_receipt_sha256",
            "websocket_capture_receipt_sha256",
            "delayed_loaded_source_receipt_sha256",
            "finalized_loaded_source_receipt_sha256",
            "delayed_extraction_receipt_sha256",
            "finalized_extraction_receipt_sha256",
            "delayed_replay_receipt_sha256",
            "finalized_replay_receipt_sha256",
            "delayed_active_state_sha256",
            "finalized_active_state_sha256",
            "delayed_feature_artifact_receipt_sha256",
            "finalized_feature_artifact_receipt_sha256",
            "feature_spec_receipt_sha256",
            "delayed_feature_sha256",
            "finalized_feature_sha256",
            "receipt_sha256",
        )
        for name in digest_fields:
            _digest(name, getattr(self, name))
        if self.event_exact is not (
            self.delayed_active_state_sha256 == self.finalized_active_state_sha256
        ):
            raise MassiveReplayParityError("event parity flag differs from evidence")
        if self.feature_exact is not (
            self.delayed_feature_sha256 == self.finalized_feature_sha256
        ):
            raise MassiveReplayParityError("feature parity flag differs from evidence")
        exact = all(
            (
                self.event_exact,
                self.feature_exact,
                self.canary_observed,
                self.capture_complete,
                self.committed_sources_complete,
            )
        )
        if exact and self.failure_reason is not None:
            raise MassiveReplayParityError("exact parity cannot have a failure reason")
        if not exact and not self.failure_reason:
            raise MassiveReplayParityError("failed parity needs a reason")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("parity evidence receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveDelayedReplayAuthority:
    entitlement_receipt_sha256: str
    runtime_entitlement_qualified: bool
    canonical_source_parsers_qualified: bool
    session_authority_receipt_sha256: str
    correction_semantics_receipt_sha256: str
    condition_authority_receipt_sha256: str
    parity_rows: tuple[MassiveReplayParityEvidence, ...]
    canary_kinds_present: tuple[str, ...]
    compared_session_count: int
    compared_symbol_day_count: int
    failed_event_symbol_days: tuple[str, ...]
    failed_feature_symbol_days: tuple[str, ...]
    development_asof_replay_authorized: bool
    historical_asof_replay_authorized: bool
    predictive_training_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("receipt_sha256")
        return payload

    def validate(self) -> None:
        if self.schema != MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA:
            raise MassiveReplayParityError("delayed replay schema drifted")
        for name in (
            "entitlement_receipt_sha256",
            "session_authority_receipt_sha256",
            "correction_semantics_receipt_sha256",
            "condition_authority_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if not self.parity_rows:
            raise MassiveReplayParityError("delayed replay has no parity evidence")
        keys = tuple(
            (row.security_id, row.session_date, row.canary_kind)
            for row in self.parity_rows
        )
        if keys != tuple(sorted(set(keys))):
            raise MassiveReplayParityError("parity evidence is not sorted and unique")
        for row in self.parity_rows:
            row.validate()
        sessions = {row.session_date for row in self.parity_rows}
        symbol_days = {(row.security_id, row.session_date) for row in self.parity_rows}
        if self.compared_session_count != len(sessions):
            raise MassiveReplayParityError("compared session count drifted")
        if self.compared_symbol_day_count != len(symbol_days):
            raise MassiveReplayParityError("compared symbol-day count drifted")
        failed_events, failed_features = _failed_rows(self.parity_rows)
        if self.failed_event_symbol_days != failed_events:
            raise MassiveReplayParityError("failed event inventory drifted")
        if self.failed_feature_symbol_days != failed_features:
            raise MassiveReplayParityError("failed feature inventory drifted")
        canaries = tuple(
            sorted({row.canary_kind for row in self.parity_rows if row.canary_observed})
        )
        if self.canary_kinds_present != canaries:
            raise MassiveReplayParityError("observed canary inventory drifted")
        coverage = (
            len(sessions) >= 2
            and len(symbol_days) >= len(REQUIRED_MASSIVE_REPLAY_CANARIES)
            and set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(canaries)
        )
        exact = (
            not failed_events
            and not failed_features
            and coverage
            and self.runtime_entitlement_qualified
            and self.canonical_source_parsers_qualified
        )
        if not self.development_asof_replay_authorized:
            raise MassiveReplayParityError("development replay must remain available")
        if self.historical_asof_replay_authorized is not exact:
            raise MassiveReplayParityError("historical replay authority differs")
        if self.predictive_training_authorized:
            raise MassiveReplayParityError("parity cannot authorize training")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("delayed replay receipt differs")


def _failed_rows(
    rows: Sequence[MassiveReplayParityEvidence],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    failed_events = tuple(
        f"{row.security_id}:{row.session_date}:{row.canary_kind}"
        for row in rows
        if not all(
            (
                row.event_exact,
                row.canary_observed,
                row.capture_complete,
                row.committed_sources_complete,
            )
        )
    )
    failed_features = tuple(
        f"{row.security_id}:{row.session_date}:{row.canary_kind}"
        for row in rows
        if not all(
            (
                row.feature_exact,
                row.canary_observed,
                row.capture_complete,
                row.committed_sources_complete,
            )
        )
    )
    return failed_events, failed_features


def _canary_observed(
    row: MassiveReplayParityInput,
    *,
    condition_authority: MassiveConditionAuthority,
) -> bool:
    events = (*row.delayed_replay.active_events, *row.finalized_replay.active_events)
    if row.canary_kind == "normal-session":
        return row.session.special_session_reason is None and (
            row.session.scheduled_five_minute_intervals == 78
        )
    if row.canary_kind == "early-close-session":
        return row.session.special_session_reason is not None and (
            row.session.scheduled_five_minute_intervals < 78
        )
    if row.canary_kind == "correction-activity":
        return bool(
            row.delayed_replay.cancelled_event_keys
            or row.finalized_replay.cancelled_event_keys
            or any(event.correction_kind != "new-trade" for event in events)
        )
    if row.canary_kind == "special-condition":
        special = {
            rule.condition_id
            for rule in condition_authority.rules
            if not all(
                (
                    rule.updates_open_close,
                    rule.updates_high_low,
                    rule.updates_volume,
                )
            )
            or rule.name.casefold() != "regular sale"
        }
        return any(special.intersection(event.conditions) for event in events)
    if row.canary_kind == "trf-trades":
        return any(event.trf_id is not None for event in events)
    if row.canary_kind == "ticker-change-identity":
        if row.ticker_change is None or not events:
            return False
        row.ticker_change.validate()
        return (
            row.ticker_change.security_id == row.delayed_replay.security_id
            and row.ticker_change.current_ticker == events[0].source_ticker
            and row.ticker_change.current_record.valid_from_ms * 1_000_000
            <= row.decision_clock.decision_at_ns
            and row.ticker_change.ticker_history_authority_receipt_sha256
            == row.delayed_replay.ticker_history_receipt_sha256
        )
    return False


def _derive_parity_evidence(
    row: MassiveReplayParityInput,
    *,
    entitlement_authority: MassiveEntitlementAuthority,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
) -> MassiveReplayParityEvidence:
    for artifact in (
        row.capture,
        row.delayed_source,
        row.finalized_source,
        row.delayed_extraction,
        row.finalized_extraction,
        row.decision_clock,
        row.session,
        row.delayed_replay,
        row.finalized_replay,
        row.delayed_features,
        row.finalized_features,
    ):
        artifact.validate()
    if row.capture.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
        raise MassiveReplayParityError("capture entitlement differs")
    for source in (row.delayed_source, row.finalized_source):
        if source.receipt.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
            raise MassiveReplayParityError("source entitlement differs")
    if row.capture.raw_capture_source_receipt_sha256 != row.delayed_source.receipt.receipt_sha256:
        raise MassiveReplayParityError("capture source is not the committed delayed source")
    if row.capture.payload_inventory_sha256 != row.delayed_replay.input_source_record_inventory_sha256:
        raise MassiveReplayParityError("capture payload inventory differs from replay")
    for source, extraction, replay in (
        (row.delayed_source, row.delayed_extraction, row.delayed_replay),
        (row.finalized_source, row.finalized_extraction, row.finalized_replay),
    ):
        if replay.source_object_receipt_sha256 != source.receipt.receipt_sha256:
            raise MassiveReplayParityError("replay source differs from committed bytes")
        if extraction.loaded_source_receipt_sha256 != source.receipt_sha256:
            raise MassiveReplayParityError("extraction used another loaded source")
        if extraction.source_commit_receipt_sha256 != source.commit.receipt_sha256:
            raise MassiveReplayParityError("extraction used another source commit")
        if extraction.selected_source_row_inventory_sha256 != replay.input_source_record_inventory_sha256:
            raise MassiveReplayParityError("extraction row inventory differs")
        if extraction.selected_row_count != replay.input_event_count:
            raise MassiveReplayParityError("extraction count differs")
        if replay.decision_clock_receipt_sha256 != row.decision_clock.receipt_sha256:
            raise MassiveReplayParityError("replay used another decision clock")
    if row.capture.lifecycle.decision_clock_receipt_sha256 != row.decision_clock.receipt_sha256:
        raise MassiveReplayParityError("capture used another decision clock")
    if row.capture.lifecycle.required_capture_end_ns != row.decision_clock.decision_at_ns:
        raise MassiveReplayParityError("capture cutoff differs from decision clock")
    for replay in (row.delayed_replay, row.finalized_replay):
        if replay.condition_authority_receipt_sha256 != condition_authority.receipt_sha256:
            raise MassiveReplayParityError("replay condition authority differs")
        if replay.correction_authority_receipt_sha256 != correction_authority.receipt_sha256:
            raise MassiveReplayParityError("replay correction authority differs")
        if replay.session_authority_receipt_sha256 != session_authority.receipt_sha256:
            raise MassiveReplayParityError("replay session authority differs")
    if session_authority.resolve(
        exchange=row.session.exchange, session_date=row.session.session_date
    ) != row.session:
        raise MassiveReplayParityError("parity session was not authority-resolved")
    identities = {
        (row.delayed_replay.security_id, row.delayed_replay.session_date),
        (row.finalized_replay.security_id, row.finalized_replay.session_date),
        (row.delayed_features.security_id, row.delayed_features.session_date),
        (row.finalized_features.security_id, row.finalized_features.session_date),
    }
    if len(identities) != 1 or next(iter(identities))[1] != row.capture.session_date:
        raise MassiveReplayParityError("parity artifacts mix security-session identities")
    if row.delayed_features.input_replay_receipt_sha256 != row.delayed_replay.receipt_sha256:
        raise MassiveReplayParityError("delayed features used another replay")
    if row.finalized_features.input_replay_receipt_sha256 != row.finalized_replay.receipt_sha256:
        raise MassiveReplayParityError("final features used another replay")
    if row.delayed_features.feature_spec_receipt_sha256 != row.finalized_features.feature_spec_receipt_sha256:
        raise MassiveReplayParityError("feature specifications differ")
    delayed_state = row.delayed_replay.active_state_inventory_sha256
    finalized_state = row.finalized_replay.active_state_inventory_sha256
    event_exact = delayed_state == finalized_state
    delayed_feature = row.delayed_features.output_feature_receipt_sha256
    finalized_feature = row.finalized_features.output_feature_receipt_sha256
    feature_exact = delayed_feature == finalized_feature
    canary = _canary_observed(row, condition_authority=condition_authority)
    committed = (
        row.capture.capture_file_parser_qualified
        and row.capture.loaded_source_receipt_sha256
        == row.delayed_source.receipt_sha256
        and row.delayed_extraction.complete_for_security_session
        and row.finalized_extraction.complete_for_security_session
        and row.delayed_extraction.canonical_parser_qualified
        and row.finalized_extraction.canonical_parser_qualified
        and row.delayed_extraction.parser_spec_sha256
        == MASSIVE_DELAYED_CAPTURE_PARSER_SPEC_SHA256
        and row.finalized_extraction.parser_spec_sha256
        == MASSIVE_FLAT_TRADE_PARSER_SPEC_SHA256
    )
    failures = []
    if not row.capture.capture_complete:
        failures.append("capture-incomplete")
    if not committed:
        failures.append("source-extraction-incomplete")
    if not event_exact:
        failures.append("event-mismatch")
    if not feature_exact:
        failures.append("feature-mismatch")
    if not canary:
        failures.append("canary-not-observed")
    security_id, session_date = next(iter(identities))
    body = {
        "schema": MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA,
        "canary_kind": row.canary_kind,
        "security_id": security_id,
        "session_date": session_date,
        "decision_clock_receipt_sha256": row.decision_clock.receipt_sha256,
        "websocket_capture_receipt_sha256": row.capture.receipt_sha256,
        "delayed_loaded_source_receipt_sha256": row.delayed_source.receipt_sha256,
        "finalized_loaded_source_receipt_sha256": row.finalized_source.receipt_sha256,
        "delayed_extraction_receipt_sha256": row.delayed_extraction.receipt_sha256,
        "finalized_extraction_receipt_sha256": row.finalized_extraction.receipt_sha256,
        "delayed_replay_receipt_sha256": row.delayed_replay.receipt_sha256,
        "finalized_replay_receipt_sha256": row.finalized_replay.receipt_sha256,
        "delayed_active_state_sha256": delayed_state,
        "finalized_active_state_sha256": finalized_state,
        "delayed_feature_artifact_receipt_sha256": row.delayed_features.receipt_sha256,
        "finalized_feature_artifact_receipt_sha256": row.finalized_features.receipt_sha256,
        "feature_spec_receipt_sha256": row.delayed_features.feature_spec_receipt_sha256,
        "delayed_feature_sha256": delayed_feature,
        "finalized_feature_sha256": finalized_feature,
        "event_exact": event_exact,
        "feature_exact": feature_exact,
        "canary_observed": canary,
        "capture_complete": row.capture.capture_complete,
        "committed_sources_complete": committed,
        "failure_reason": "+".join(failures) or None,
    }
    value = MassiveReplayParityEvidence(
        canary_kind=row.canary_kind,
        security_id=security_id,
        session_date=session_date,
        decision_clock_receipt_sha256=row.decision_clock.receipt_sha256,
        websocket_capture_receipt_sha256=row.capture.receipt_sha256,
        delayed_loaded_source_receipt_sha256=row.delayed_source.receipt_sha256,
        finalized_loaded_source_receipt_sha256=row.finalized_source.receipt_sha256,
        delayed_extraction_receipt_sha256=row.delayed_extraction.receipt_sha256,
        finalized_extraction_receipt_sha256=row.finalized_extraction.receipt_sha256,
        delayed_replay_receipt_sha256=row.delayed_replay.receipt_sha256,
        finalized_replay_receipt_sha256=row.finalized_replay.receipt_sha256,
        delayed_active_state_sha256=delayed_state,
        finalized_active_state_sha256=finalized_state,
        delayed_feature_artifact_receipt_sha256=row.delayed_features.receipt_sha256,
        finalized_feature_artifact_receipt_sha256=row.finalized_features.receipt_sha256,
        feature_spec_receipt_sha256=row.delayed_features.feature_spec_receipt_sha256,
        delayed_feature_sha256=delayed_feature,
        finalized_feature_sha256=finalized_feature,
        event_exact=event_exact,
        feature_exact=feature_exact,
        canary_observed=canary,
        capture_complete=row.capture.capture_complete,
        committed_sources_complete=committed,
        failure_reason="+".join(failures) or None,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def build_massive_delayed_replay_authority(
    inputs: Sequence[MassiveReplayParityInput],
    *,
    entitlement_authority: MassiveEntitlementAuthority,
    session_authority: MassiveSessionAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
) -> MassiveDelayedReplayAuthority:
    """Derive parity only from committed sources and canonical features."""

    entitlement_authority.validate()
    session_authority.validate()
    condition_authority.validate()
    correction_authority.validate()
    if not inputs:
        raise MassiveReplayParityError("delayed replay has no evidence inputs")
    rows = tuple(
        sorted(
            (
                _derive_parity_evidence(
                    row,
                    entitlement_authority=entitlement_authority,
                    session_authority=session_authority,
                    condition_authority=condition_authority,
                    correction_authority=correction_authority,
                )
                for row in inputs
            ),
            key=lambda row: (row.security_id, row.session_date, row.canary_kind),
        )
    )
    keys = tuple((row.security_id, row.session_date, row.canary_kind) for row in rows)
    if keys != tuple(sorted(set(keys))):
        raise MassiveReplayParityError("parity inputs contain duplicate evidence")
    sessions = {row.session_date for row in rows}
    symbol_days = {(row.security_id, row.session_date) for row in rows}
    failed_events, failed_features = _failed_rows(rows)
    canaries = tuple(sorted({row.canary_kind for row in rows if row.canary_observed}))
    coverage = (
        len(sessions) >= 2
        and len(symbol_days) >= len(REQUIRED_MASSIVE_REPLAY_CANARIES)
        and set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(canaries)
    )
    runtime_qualified = bool(
        getattr(entitlement_authority, "runtime_entitlement_qualified", False)
    )
    canonical_source_parsers_qualified = all(
        row.committed_sources_complete for row in rows
    )
    historical = (
        not failed_events
        and not failed_features
        and coverage
        and runtime_qualified
        and canonical_source_parsers_qualified
    )
    receipt_body = {
        "schema": MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA,
        "entitlement_receipt_sha256": entitlement_authority.receipt_sha256,
        "runtime_entitlement_qualified": runtime_qualified,
        "canonical_source_parsers_qualified": canonical_source_parsers_qualified,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "correction_semantics_receipt_sha256": correction_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "parity_rows": [asdict(row) for row in rows],
        "canary_kinds_present": canaries,
        "compared_session_count": len(sessions),
        "compared_symbol_day_count": len(symbol_days),
        "failed_event_symbol_days": failed_events,
        "failed_feature_symbol_days": failed_features,
        "development_asof_replay_authorized": True,
        "historical_asof_replay_authorized": historical,
        "predictive_training_authorized": False,
    }
    authority = MassiveDelayedReplayAuthority(
        entitlement_receipt_sha256=entitlement_authority.receipt_sha256,
        runtime_entitlement_qualified=runtime_qualified,
        canonical_source_parsers_qualified=canonical_source_parsers_qualified,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        correction_semantics_receipt_sha256=correction_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        parity_rows=rows,
        canary_kinds_present=canaries,
        compared_session_count=len(sessions),
        compared_symbol_day_count=len(symbol_days),
        failed_event_symbol_days=failed_events,
        failed_feature_symbol_days=failed_features,
        development_asof_replay_authorized=True,
        historical_asof_replay_authorized=historical,
        predictive_training_authorized=False,
        receipt_sha256=semantic_sha256(receipt_body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA",
    "MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA",
    "MASSIVE_TICKER_CHANGE_CANARY_SCHEMA",
    "REQUIRED_MASSIVE_REPLAY_CANARIES",
    "MassiveDelayedReplayAuthority",
    "MassiveReplayCanaryKind",
    "MassiveReplayFeatureArtifact",
    "MassiveReplayParityError",
    "MassiveReplayParityEvidence",
    "MassiveReplayParityInput",
    "MassiveTickerChangeCanaryEvidence",
    "build_massive_delayed_replay_authority",
]
