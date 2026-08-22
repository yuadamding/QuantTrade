"""Evidence-derived delayed-stream versus finalized-file replay qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal, Mapping, Sequence

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.entitlement import MassiveEntitlementAuthority
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import MassiveSourceObjectReceipt
from rl_quant.data_sources.massive.trade_replay import MassiveTradeReplayResult
from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketCaptureAuthority,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_payload,
    semantic_sha256,
)


MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA = "rl-quant.massive-delayed-replay-v2"
MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA = (
    "rl-quant.massive-replay-feature-artifact-v1"
)
MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA = "rl-quant.massive-replay-parity-evidence-v1"
MASSIVE_TICKER_CHANGE_CANARY_SCHEMA = "rl-quant.massive-ticker-change-canary-v1"

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


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveReplayParityError(f"{name} must be canonical text")
    return value


@dataclass(frozen=True, slots=True)
class MassiveReplayFeatureArtifact:
    security_id: str
    session_date: str
    source_replay_receipt_sha256: str
    canonical_feature_payload_json: str
    feature_payload_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA:
            raise MassiveReplayParityError("feature artifact schema drifted")
        _text("security ID", self.security_id)
        _text("session date", self.session_date)
        _digest("source replay receipt", self.source_replay_receipt_sha256)
        _text("canonical feature payload", self.canonical_feature_payload_json)
        payload = json.loads(self.canonical_feature_payload_json)
        if canonical_json_payload(payload).decode("ascii") != self.canonical_feature_payload_json:
            raise MassiveReplayParityError("feature payload is not canonical JSON")
        _digest("feature payload SHA", self.feature_payload_sha256)
        if self.feature_payload_sha256 != semantic_sha256(payload):
            raise MassiveReplayParityError("feature payload SHA differs")
        _digest("feature artifact receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("feature artifact receipt differs")

    @classmethod
    def build(
        cls,
        *,
        security_id: str,
        session_date: str,
        source_replay_receipt_sha256: str,
        feature_payload: Mapping[str, object],
    ) -> MassiveReplayFeatureArtifact:
        canonical = canonical_json_payload(feature_payload).decode("ascii")
        body = {
            "schema": MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA,
            "security_id": security_id,
            "session_date": session_date,
            "source_replay_receipt_sha256": source_replay_receipt_sha256,
            "canonical_feature_payload_json": canonical,
            "feature_payload_sha256": semantic_sha256(feature_payload),
        }
        value = cls(receipt_sha256=semantic_sha256(body), **body)
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveTickerChangeCanaryEvidence:
    security_id: str
    session_date: str
    prior_ticker: str
    current_ticker: str
    identity_authority_receipt_sha256: str
    ticker_history_receipt_sha256: str
    transition_source_receipt_sha256: str
    receipt_sha256: str
    schema: str = MASSIVE_TICKER_CHANGE_CANARY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_TICKER_CHANGE_CANARY_SCHEMA:
            raise MassiveReplayParityError("ticker-change canary schema drifted")
        for name in (
            "security_id",
            "session_date",
            "prior_ticker",
            "current_ticker",
        ):
            _text(name, getattr(self, name))
        if self.prior_ticker == self.current_ticker:
            raise MassiveReplayParityError("ticker-change canary did not change ticker")
        for name in (
            "identity_authority_receipt_sha256",
            "ticker_history_receipt_sha256",
            "transition_source_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("ticker-change canary receipt differs")

    @classmethod
    def build(
        cls,
        *,
        security_id: str,
        session_date: str,
        prior_ticker: str,
        current_ticker: str,
        identity_authority_receipt_sha256: str,
        ticker_history_receipt_sha256: str,
        transition_source_receipt_sha256: str,
    ) -> MassiveTickerChangeCanaryEvidence:
        body = {
            "schema": MASSIVE_TICKER_CHANGE_CANARY_SCHEMA,
            "security_id": security_id,
            "session_date": session_date,
            "prior_ticker": prior_ticker,
            "current_ticker": current_ticker,
            "identity_authority_receipt_sha256": identity_authority_receipt_sha256,
            "ticker_history_receipt_sha256": ticker_history_receipt_sha256,
            "transition_source_receipt_sha256": transition_source_receipt_sha256,
        }
        value = cls(
            security_id=security_id,
            session_date=session_date,
            prior_ticker=prior_ticker,
            current_ticker=current_ticker,
            identity_authority_receipt_sha256=identity_authority_receipt_sha256,
            ticker_history_receipt_sha256=ticker_history_receipt_sha256,
            transition_source_receipt_sha256=transition_source_receipt_sha256,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class MassiveReplayParityInput:
    canary_kind: MassiveReplayCanaryKind
    capture: MassiveDelayedWebSocketCaptureAuthority
    finalized_source: MassiveSourceObjectReceipt
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
    websocket_capture_receipt_sha256: str
    finalized_source_receipt_sha256: str
    delayed_replay_receipt_sha256: str
    finalized_replay_receipt_sha256: str
    delayed_active_state_sha256: str
    finalized_active_state_sha256: str
    delayed_feature_artifact_receipt_sha256: str
    finalized_feature_artifact_receipt_sha256: str
    delayed_feature_sha256: str
    finalized_feature_sha256: str
    event_exact: bool
    feature_exact: bool
    canary_observed: bool
    capture_complete: bool
    failure_reason: str | None
    receipt_sha256: str
    schema: str = MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA:
            raise MassiveReplayParityError("parity evidence schema drifted")
        if self.canary_kind not in REQUIRED_MASSIVE_REPLAY_CANARIES:
            raise MassiveReplayParityError("parity canary kind is unsupported")
        _text("security ID", self.security_id)
        _text("session date", self.session_date)
        for name in (
            "websocket_capture_receipt_sha256",
            "finalized_source_receipt_sha256",
            "delayed_replay_receipt_sha256",
            "finalized_replay_receipt_sha256",
            "delayed_active_state_sha256",
            "finalized_active_state_sha256",
            "delayed_feature_artifact_receipt_sha256",
            "finalized_feature_artifact_receipt_sha256",
            "delayed_feature_sha256",
            "finalized_feature_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.event_exact is not (
            self.delayed_active_state_sha256 == self.finalized_active_state_sha256
        ):
            raise MassiveReplayParityError("event parity flag differs from evidence")
        if self.feature_exact is not (
            self.delayed_feature_sha256 == self.finalized_feature_sha256
        ):
            raise MassiveReplayParityError("feature parity flag differs from evidence")
        if not isinstance(self.capture_complete, bool):
            raise MassiveReplayParityError("capture-complete flag must be Boolean")
        exact = (
            self.event_exact
            and self.feature_exact
            and self.canary_observed
            and self.capture_complete
        )
        if exact and self.failure_reason is not None:
            raise MassiveReplayParityError("exact parity cannot have a failure reason")
        if not exact and (
            not self.failure_reason or self.failure_reason != self.failure_reason.strip()
        ):
            raise MassiveReplayParityError("failed parity needs a canonical reason")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("parity evidence receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveDelayedReplayAuthority:
    entitlement_receipt_sha256: str
    session_authority_receipt_sha256: str
    correction_semantics_receipt_sha256: str
    condition_authority_receipt_sha256: str
    parity_rows: tuple[MassiveReplayParityEvidence, ...]
    canary_kinds_present: tuple[str, ...]
    compared_session_count: int
    compared_symbol_day_count: int
    exact_event_symbol_day_count: int
    failed_event_symbol_days: tuple[str, ...]
    exact_feature_symbol_day_count: int
    failed_feature_symbol_days: tuple[str, ...]
    development_asof_replay_authorized: bool
    historical_asof_replay_authorized: bool
    predictive_training_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "entitlement_receipt_sha256": self.entitlement_receipt_sha256,
            "session_authority_receipt_sha256": self.session_authority_receipt_sha256,
            "correction_semantics_receipt_sha256": self.correction_semantics_receipt_sha256,
            "condition_authority_receipt_sha256": self.condition_authority_receipt_sha256,
            "parity_rows": [asdict(row) for row in self.parity_rows],
            "canary_kinds_present": self.canary_kinds_present,
            "compared_session_count": self.compared_session_count,
            "compared_symbol_day_count": self.compared_symbol_day_count,
            "exact_event_symbol_day_count": self.exact_event_symbol_day_count,
            "failed_event_symbol_days": self.failed_event_symbol_days,
            "exact_feature_symbol_day_count": self.exact_feature_symbol_day_count,
            "failed_feature_symbol_days": self.failed_feature_symbol_days,
            "development_asof_replay_authorized": self.development_asof_replay_authorized,
            "historical_asof_replay_authorized": self.historical_asof_replay_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
        }

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
        failed_events = tuple(
            f"{row.security_id}:{row.session_date}:{row.canary_kind}"
            for row in self.parity_rows
            if not row.event_exact or not row.canary_observed or not row.capture_complete
        )
        failed_features = tuple(
            f"{row.security_id}:{row.session_date}:{row.canary_kind}"
            for row in self.parity_rows
            if not row.feature_exact or not row.canary_observed or not row.capture_complete
        )
        if self.exact_event_symbol_day_count != sum(
            row.event_exact and row.canary_observed and row.capture_complete
            for row in self.parity_rows
        ):
            raise MassiveReplayParityError("exact event count drifted")
        if self.exact_feature_symbol_day_count != sum(
            row.feature_exact and row.canary_observed and row.capture_complete
            for row in self.parity_rows
        ):
            raise MassiveReplayParityError("exact feature count drifted")
        if self.failed_event_symbol_days != failed_events:
            raise MassiveReplayParityError("failed event inventory drifted")
        if self.failed_feature_symbol_days != failed_features:
            raise MassiveReplayParityError("failed feature inventory drifted")
        expected_canaries = tuple(
            sorted({row.canary_kind for row in self.parity_rows if row.canary_observed})
        )
        if self.canary_kinds_present != expected_canaries:
            raise MassiveReplayParityError("observed canary inventory drifted")
        coverage = (
            len(sessions) >= 2
            and len(symbol_days) >= len(REQUIRED_MASSIVE_REPLAY_CANARIES)
            and set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(expected_canaries)
        )
        exact = not failed_events and not failed_features and coverage
        if self.development_asof_replay_authorized is not True:
            raise MassiveReplayParityError("validated evidence must allow development replay")
        if self.historical_asof_replay_authorized is not exact:
            raise MassiveReplayParityError("historical replay authority differs from evidence")
        if self.predictive_training_authorized:
            raise MassiveReplayParityError(
                "replay parity alone cannot authorize predictive training"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveReplayParityError("delayed replay receipt differs")


def _canary_observed(
    row: MassiveReplayParityInput,
    *,
    condition_authority: MassiveConditionAuthority,
) -> bool:
    events = (*row.delayed_replay.active_events, *row.finalized_replay.active_events)
    if row.canary_kind == "normal-session":
        return (
            row.session.special_session_reason is None
            and row.session.scheduled_five_minute_intervals == 78
        )
    if row.canary_kind == "early-close-session":
        return (
            row.session.special_session_reason is not None
            and row.session.scheduled_five_minute_intervals < 78
        )
    if row.canary_kind == "correction-activity":
        return bool(
            row.delayed_replay.cancelled_event_keys
            or row.finalized_replay.cancelled_event_keys
            or any(event.correction_kind != "new-trade" for event in events)
        )
    if row.canary_kind == "special-condition":
        special_ids = {
            rule.condition_id
            for rule in condition_authority.rules
            if not (
                rule.updates_open_close
                and rule.updates_high_low
                and rule.updates_volume
            )
            or rule.name.casefold() != "regular sale"
        }
        return any(special_ids.intersection(event.conditions) for event in events)
    if row.canary_kind == "trf-trades":
        return any(event.trf_id is not None for event in events)
    if row.canary_kind == "ticker-change-identity":
        if row.ticker_change is None:
            return False
        row.ticker_change.validate()
        tickers = {event.source_ticker for event in events}
        return (
            row.ticker_change.security_id == row.delayed_replay.security_id
            and row.ticker_change.session_date == row.delayed_replay.session_date
            and row.ticker_change.current_ticker in tickers
            and row.ticker_change.identity_authority_receipt_sha256
            == row.delayed_replay.identity_authority_receipt_sha256
            and row.ticker_change.ticker_history_receipt_sha256
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
    row.capture.validate()
    row.finalized_source.validate()
    row.session.validate()
    row.delayed_replay.validate()
    row.finalized_replay.validate()
    row.delayed_features.validate()
    row.finalized_features.validate()
    if row.capture.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
        raise MassiveReplayParityError("capture entitlement differs from authority")
    if row.finalized_source.entitlement_receipt_sha256 != entitlement_authority.receipt_sha256:
        raise MassiveReplayParityError("final source entitlement differs from authority")
    if row.capture.raw_capture_source_receipt_sha256 != row.delayed_replay.source_object_receipt_sha256:
        raise MassiveReplayParityError("capture source differs from delayed replay")
    if row.capture.payload_inventory_sha256 != row.delayed_replay.input_source_record_inventory_sha256:
        raise MassiveReplayParityError("capture payload inventory differs from delayed replay")
    if row.finalized_source.receipt_sha256 != row.finalized_replay.source_object_receipt_sha256:
        raise MassiveReplayParityError("final source differs from finalized replay")
    if (
        row.delayed_replay.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
        or row.finalized_replay.condition_authority_receipt_sha256
        != condition_authority.receipt_sha256
    ):
        raise MassiveReplayParityError("replay condition authority differs")
    if (
        row.delayed_replay.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
        or row.finalized_replay.correction_authority_receipt_sha256
        != correction_authority.receipt_sha256
    ):
        raise MassiveReplayParityError("replay correction authority differs")
    if (
        row.delayed_replay.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or row.finalized_replay.session_authority_receipt_sha256
        != session_authority.receipt_sha256
    ):
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
    if (
        row.delayed_features.source_replay_receipt_sha256
        != row.delayed_replay.receipt_sha256
    ):
        raise MassiveReplayParityError("delayed features used another replay")
    if (
        row.finalized_features.source_replay_receipt_sha256
        != row.finalized_replay.receipt_sha256
    ):
        raise MassiveReplayParityError("final features used another replay")
    delayed_state = row.delayed_replay.active_state_inventory_sha256
    finalized_state = row.finalized_replay.active_state_inventory_sha256
    canary_observed = _canary_observed(
        row, condition_authority=condition_authority
    )
    event_exact = delayed_state == finalized_state
    feature_exact = (
        row.delayed_features.feature_payload_sha256
        == row.finalized_features.feature_payload_sha256
    )
    failures = []
    if not row.capture.capture_complete:
        failures.append("capture-incomplete")
    if not event_exact:
        failures.append("event-mismatch")
    if not feature_exact:
        failures.append("feature-mismatch")
    if not canary_observed:
        failures.append("canary-not-observed")
    failure_reason = "+".join(failures) or None
    security_id, session_date = next(iter(identities))
    body = {
        "schema": MASSIVE_REPLAY_PARITY_EVIDENCE_SCHEMA,
        "canary_kind": row.canary_kind,
        "security_id": security_id,
        "session_date": session_date,
        "websocket_capture_receipt_sha256": row.capture.receipt_sha256,
        "finalized_source_receipt_sha256": row.finalized_source.receipt_sha256,
        "delayed_replay_receipt_sha256": row.delayed_replay.receipt_sha256,
        "finalized_replay_receipt_sha256": row.finalized_replay.receipt_sha256,
        "delayed_active_state_sha256": delayed_state,
        "finalized_active_state_sha256": finalized_state,
        "delayed_feature_artifact_receipt_sha256": row.delayed_features.receipt_sha256,
        "finalized_feature_artifact_receipt_sha256": row.finalized_features.receipt_sha256,
        "delayed_feature_sha256": row.delayed_features.feature_payload_sha256,
        "finalized_feature_sha256": row.finalized_features.feature_payload_sha256,
        "event_exact": event_exact,
        "feature_exact": feature_exact,
        "canary_observed": canary_observed,
        "capture_complete": row.capture.capture_complete,
        "failure_reason": failure_reason,
    }
    value = MassiveReplayParityEvidence(
        canary_kind=row.canary_kind,
        security_id=security_id,
        session_date=session_date,
        websocket_capture_receipt_sha256=row.capture.receipt_sha256,
        finalized_source_receipt_sha256=row.finalized_source.receipt_sha256,
        delayed_replay_receipt_sha256=row.delayed_replay.receipt_sha256,
        finalized_replay_receipt_sha256=row.finalized_replay.receipt_sha256,
        delayed_active_state_sha256=delayed_state,
        finalized_active_state_sha256=finalized_state,
        delayed_feature_artifact_receipt_sha256=row.delayed_features.receipt_sha256,
        finalized_feature_artifact_receipt_sha256=row.finalized_features.receipt_sha256,
        delayed_feature_sha256=row.delayed_features.feature_payload_sha256,
        finalized_feature_sha256=row.finalized_features.feature_payload_sha256,
        event_exact=event_exact,
        feature_exact=feature_exact,
        canary_observed=canary_observed,
        capture_complete=row.capture.capture_complete,
        failure_reason=failure_reason,
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
    """Derive replay parity from typed capture, source, replay, and feature evidence."""

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
    failed_events = tuple(
        f"{row.security_id}:{row.session_date}:{row.canary_kind}"
        for row in rows
        if not row.event_exact or not row.canary_observed or not row.capture_complete
    )
    failed_features = tuple(
        f"{row.security_id}:{row.session_date}:{row.canary_kind}"
        for row in rows
        if not row.feature_exact or not row.canary_observed or not row.capture_complete
    )
    canaries = tuple(sorted({row.canary_kind for row in rows if row.canary_observed}))
    coverage = (
        len(sessions) >= 2
        and len(symbol_days) >= len(REQUIRED_MASSIVE_REPLAY_CANARIES)
        and set(REQUIRED_MASSIVE_REPLAY_CANARIES).issubset(canaries)
    )
    historical = not failed_events and not failed_features and coverage
    body = {
        "schema": MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA,
        "entitlement_receipt_sha256": entitlement_authority.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "correction_semantics_receipt_sha256": correction_authority.receipt_sha256,
        "condition_authority_receipt_sha256": condition_authority.receipt_sha256,
        "parity_rows": [asdict(row) for row in rows],
        "canary_kinds_present": canaries,
        "compared_session_count": len(sessions),
        "compared_symbol_day_count": len(symbol_days),
        "exact_event_symbol_day_count": sum(
            row.event_exact and row.canary_observed and row.capture_complete
            for row in rows
        ),
        "failed_event_symbol_days": failed_events,
        "exact_feature_symbol_day_count": sum(
            row.feature_exact and row.canary_observed and row.capture_complete
            for row in rows
        ),
        "failed_feature_symbol_days": failed_features,
        "development_asof_replay_authorized": True,
        "historical_asof_replay_authorized": historical,
        "predictive_training_authorized": False,
    }
    authority = MassiveDelayedReplayAuthority(
        entitlement_receipt_sha256=entitlement_authority.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        correction_semantics_receipt_sha256=correction_authority.receipt_sha256,
        condition_authority_receipt_sha256=condition_authority.receipt_sha256,
        parity_rows=rows,
        canary_kinds_present=canaries,
        compared_session_count=len(sessions),
        compared_symbol_day_count=len(symbol_days),
        exact_event_symbol_day_count=sum(
            row.event_exact and row.canary_observed and row.capture_complete
            for row in rows
        ),
        failed_event_symbol_days=failed_events,
        exact_feature_symbol_day_count=sum(
            row.feature_exact and row.canary_observed and row.capture_complete
            for row in rows
        ),
        failed_feature_symbol_days=failed_features,
        development_asof_replay_authorized=True,
        historical_asof_replay_authorized=historical,
        predictive_training_authorized=False,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_DELAYED_REPLAY_AUTHORITY_SCHEMA",
    "MASSIVE_REPLAY_FEATURE_ARTIFACT_SCHEMA",
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
