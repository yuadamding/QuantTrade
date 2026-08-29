"""Source-bound corporate, terminal, and cash transitions for adaptive books."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from rl_quant.alpha.accounting import (
    EconomicPosition,
    apply_cash_return,
    apply_corporate_action,
)
from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_economic_authority_v6 import (
    MassiveEconomicAuthorityV6Error,
    MassiveOrderedEconomicEventV6,
    MassiveProviderEconomicArchiveAuthorityV6,
    MassiveResolvedEconomicAuthorityAtOriginV6,
    resolve_massive_economic_authority_at_origin_v6,
)
from rl_quant.features.massive_economic_event_source_v5 import (
    MassiveSourcedCashReturnV5,
    MassiveSourcedCorporateActionV5,
    MassiveSourcedTerminalEventV5,
    sourced_effective_at_ms_v5,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)

MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-economic-event-transition-v1"
)


class MassiveAdaptiveEconomicEventTransitionV1Error(ValueError):
    """An economic event interval is detached from source time or identity."""


def _identity_scope(
    *,
    authority: MassiveResolvedEconomicAuthorityAtOriginV6,
    identity_authority: PITSecurityUniverseAuthority,
) -> str:
    masters = {row.security_id: row for row in identity_authority.security_master}
    security_ids: set[str] = set()
    for ordered in authority.selected_events:
        source = ordered.source_event
        if isinstance(
            source, (MassiveSourcedCorporateActionV5, MassiveSourcedTerminalEventV5)
        ):
            security_ids.add(source.event.security_id)
            if source.event.successor_security_id is not None:
                security_ids.add(source.event.successor_security_id)
    try:
        return semantic_sha256(
            tuple(
                (
                    security_id,
                    masters[security_id].corporate_action_chain_id,
                    masters[security_id].identity_source_receipt_sha256,
                )
                for security_id in sorted(security_ids)
            )
        )
    except KeyError as exc:
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic event references an identity outside the frozen authority"
        ) from exc


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicEventTransitionV1:
    prior_session_date: str
    fill_session_date: str
    prior_close_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    fill_close_at_ms: int
    prefill_events: tuple[MassiveOrderedEconomicEventV6, ...]
    postfill_events: tuple[MassiveOrderedEconomicEventV6, ...]
    provider_archive_receipt_sha256: str
    resolved_authority_receipt_sha256: str
    applied_event_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        events = (*self.prefill_events, *self.postfill_events)
        if (
            self.schema != MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V1_SCHEMA
            or not self.prior_session_date
            or self.fill_session_date <= self.prior_session_date
            or not self.prior_close_at_ms < self.fill_start_at_ms
            or not self.fill_start_at_ms < self.fill_end_at_ms <= self.fill_close_at_ms
            or not isinstance(self.source_data_qualified, bool)
            or not self.source_data_qualified
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.applied_event_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in events))
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveEconomicEventTransitionV1Error(
                "adaptive economic event transition differs"
            )
        for row in events:
            row.validate()
        if any(
            not self.prior_close_at_ms
            < sourced_effective_at_ms_v5(row.source_event)
            <= self.fill_start_at_ms
            for row in self.prefill_events
        ) or any(
            not self.fill_start_at_ms
            < sourced_effective_at_ms_v5(row.source_event)
            <= self.fill_close_at_ms
            for row in self.postfill_events
        ):
            raise MassiveAdaptiveEconomicEventTransitionV1Error(
                "economic event was assigned to the wrong execution phase"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_economic_event_transition_v1(
    *,
    prior_session_date: str,
    fill_session_date: str,
    provider_archive: MassiveProviderEconomicArchiveAuthorityV6,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveAdaptiveEconomicEventTransitionV1:
    """Bind one close-to-close interval to the origin-resolved V6 event source."""

    if not isinstance(provider_archive, MassiveProviderEconomicArchiveAuthorityV6):
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic transition requires the package-owned V6 provider archive"
        )
    try:
        provider_archive.validate()
    except MassiveEconomicAuthorityV6Error as exc:
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic event archive failed source replay validation"
        ) from exc
    daily_input_authority.validate()
    identity_authority.validate()
    sessions = {
        row.source_session_date: row for row in daily_input_authority.sessions
    }
    try:
        prior_close = sessions[prior_session_date].regular_close_at_ms
        fill_close = sessions[fill_session_date].regular_close_at_ms
    except (KeyError, AttributeError) as exc:
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic transition sessions are outside the qualified daily authority"
        ) from exc
    fill_start, fill_end = adaptive_fill_clock_v1(fill_session_date)
    if provider_archive.identity_authority_receipt_sha256 != identity_authority.receipt_sha256:
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic event archive and identity authority differ"
        )
    resolved_authority = resolve_massive_economic_authority_at_origin_v6(
        archive=provider_archive,
        identity_authority=identity_authority,
        decision_at_ms=fill_close,
    )
    if (
        resolved_authority.decision_at_ms != fill_close
        or resolved_authority.identity_scope_receipt_sha256
        != _identity_scope(
            authority=resolved_authority, identity_authority=identity_authority
        )
    ):
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "economic event authority has the wrong close or identity scope"
        )
    interval = tuple(
        row
        for row in resolved_authority.selected_events
        if prior_close
        < sourced_effective_at_ms_v5(row.source_event)
        <= fill_close
    )
    prefill = tuple(
        row
        for row in interval
        if sourced_effective_at_ms_v5(row.source_event) <= fill_start
    )
    postfill = tuple(row for row in interval if row not in prefill)
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V1_SCHEMA,
        "prior_session_date": prior_session_date,
        "fill_session_date": fill_session_date,
        "prior_close_at_ms": prior_close,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
        "fill_close_at_ms": fill_close,
        "prefill_events": prefill,
        "postfill_events": postfill,
        "provider_archive_receipt_sha256": provider_archive.receipt_sha256,
        "resolved_authority_receipt_sha256": resolved_authority.receipt_sha256,
        "applied_event_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in interval)
        ),
        "source_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveEconomicEventTransitionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _apply_events(
    *,
    shares: Mapping[str, float],
    cash: float,
    events: Sequence[MassiveOrderedEconomicEventV6],
    accrue_cash: bool,
) -> tuple[dict[str, float], float]:
    position = EconomicPosition.from_mapping(shares, cash=cash)
    for ordered in events:
        source = ordered.source_event
        if isinstance(source, MassiveSourcedCashReturnV5):
            if accrue_cash:
                position = apply_cash_return(
                    position, source.cash_return.one_step_return
                )
            continue
        if source.event.security_id in position.as_mapping():
            position = apply_corporate_action(position, source.event)
    return position.as_mapping(), position.cash


def apply_massive_adaptive_prefill_events_v1(
    *,
    transition: MassiveAdaptiveEconomicEventTransitionV1,
    existing_shares: Mapping[str, float],
    cash: float,
    requested_shares: Mapping[str, float],
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Repair holdings and the pending target before the morning fill."""

    transition.validate()
    target = {
        security_id: existing_shares.get(security_id, 0.0)
        + requested_shares.get(security_id, 0.0)
        for security_id in set(existing_shares) | set(requested_shares)
    }
    if any(value < -1.0e-10 for value in target.values()):
        raise MassiveAdaptiveEconomicEventTransitionV1Error(
            "pending target is short before event repair"
        )
    target = {key: max(0.0, value) for key, value in target.items() if value > 1e-12}
    repaired_existing, repaired_cash = _apply_events(
        shares=existing_shares,
        cash=cash,
        events=transition.prefill_events,
        accrue_cash=True,
    )
    repaired_target, _ = _apply_events(
        shares=target,
        cash=0.0,
        events=transition.prefill_events,
        accrue_cash=False,
    )
    requested = {
        security_id: repaired_target.get(security_id, 0.0)
        - repaired_existing.get(security_id, 0.0)
        for security_id in set(repaired_existing) | set(repaired_target)
    }
    return (
        repaired_existing,
        repaired_cash,
        {key: value for key, value in requested.items() if abs(value) > 1e-12},
    )


def apply_massive_adaptive_postfill_events_v1(
    *,
    transition: MassiveAdaptiveEconomicEventTransitionV1,
    shares: Mapping[str, float],
    cash: float,
) -> tuple[dict[str, float], float]:
    """Apply source-ordered events after the fill and through the close mark."""

    transition.validate()
    return _apply_events(
        shares=shares,
        cash=cash,
        events=transition.postfill_events,
        accrue_cash=True,
    )


__all__ = [
    "MassiveAdaptiveEconomicEventTransitionV1",
    "MassiveAdaptiveEconomicEventTransitionV1Error",
    "apply_massive_adaptive_postfill_events_v1",
    "apply_massive_adaptive_prefill_events_v1",
    "build_massive_adaptive_economic_event_transition_v1",
]
