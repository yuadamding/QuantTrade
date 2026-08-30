"""Dual-cutoff economic-event transition for adaptive morning execution.

The pre-fill snapshot is resolved at the start of the 09:35 fill window.  A
separate close snapshot may add later-known events only to post-fill
accounting; it can never retroactively repair the morning order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.evaluation.massive_adaptive_economic_event_transition_v1 import (
    apply_massive_adaptive_postfill_events_v1,
    apply_massive_adaptive_prefill_events_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_economic_authority_v6 import (
    MassiveEconomicAuthorityV6Error,
    MassiveOrderedEconomicEventV6,
    MassiveProviderEconomicArchiveAuthorityV6,
    MassiveResolvedEconomicAuthorityAtOriginV6,
    resolve_massive_economic_authority_at_origin_v6,
)
from rl_quant.features.massive_economic_event_source_v5 import (
    MassiveSourcedCorporateActionV5,
    MassiveSourcedTerminalEventV5,
    sourced_available_at_ms_v5,
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

MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V2_SCHEMA = (
    "rl-quant.massive-adaptive-economic-event-transition-v2"
)


class MassiveAdaptiveEconomicEventTransitionV2Error(ValueError):
    """The fill-start or close event snapshot is not source-causal."""


def _identity_scope(
    authority: MassiveResolvedEconomicAuthorityAtOriginV6,
    identity: PITSecurityUniverseAuthority,
) -> str:
    masters = {row.security_id: row for row in identity.security_master}
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
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
            "economic event references identity outside the frozen authority"
        ) from exc


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicEventTransitionV2:
    prior_session_date: str
    fill_session_date: str
    prior_close_at_ms: int
    fill_start_at_ms: int
    fill_end_at_ms: int
    fill_close_at_ms: int
    prefill_events: tuple[MassiveOrderedEconomicEventV6, ...]
    postfill_events: tuple[MassiveOrderedEconomicEventV6, ...]
    provider_archive_receipt_sha256: str
    prefill_resolved_authority_receipt_sha256: str
    postfill_resolved_authority_receipt_sha256: str
    applied_event_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        events = (*self.prefill_events, *self.postfill_events)
        prefill_receipts = {row.receipt_sha256 for row in self.prefill_events}
        if (
            self.schema != MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V2_SCHEMA
            or not self.prior_session_date
            or self.fill_session_date <= self.prior_session_date
            or not self.prior_close_at_ms < self.fill_start_at_ms
            or not self.fill_start_at_ms < self.fill_end_at_ms <= self.fill_close_at_ms
            or not self.source_data_qualified
            or len(prefill_receipts) != len(self.prefill_events)
            or any(row.receipt_sha256 in prefill_receipts for row in self.postfill_events)
            or self.applied_event_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in events))
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveEconomicEventTransitionV2Error(
                "adaptive economic event transition v2 differs"
            )
        for row in events:
            row.validate()
        if any(
            not self.prior_close_at_ms
            < sourced_effective_at_ms_v5(row.source_event)
            <= self.fill_start_at_ms
            or sourced_available_at_ms_v5(row.source_event) > self.fill_start_at_ms
            for row in self.prefill_events
        ):
            raise MassiveAdaptiveEconomicEventTransitionV2Error(
                "prefill event was not known by the fill-start cutoff"
            )
        if any(
            not self.prior_close_at_ms
            < sourced_effective_at_ms_v5(row.source_event)
            <= self.fill_close_at_ms
            or sourced_available_at_ms_v5(row.source_event) > self.fill_close_at_ms
            for row in self.postfill_events
        ):
            raise MassiveAdaptiveEconomicEventTransitionV2Error(
                "postfill event was not known by the close cutoff"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_economic_event_transition_v2(
    *,
    prior_session_date: str,
    fill_session_date: str,
    provider_archive: MassiveProviderEconomicArchiveAuthorityV6,
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1,
    identity_authority: PITSecurityUniverseAuthority,
) -> MassiveAdaptiveEconomicEventTransitionV2:
    """Resolve independent fill-start and close as-of snapshots."""

    if not isinstance(provider_archive, MassiveProviderEconomicArchiveAuthorityV6):
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
            "economic transition v2 requires the package-owned V6 archive"
        )
    try:
        provider_archive.validate()
    except MassiveEconomicAuthorityV6Error as exc:
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
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
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
            "economic transition sessions are outside the daily authority"
        ) from exc
    fill_start, fill_end = adaptive_fill_clock_v1(fill_session_date)
    if provider_archive.identity_authority_receipt_sha256 != identity_authority.receipt_sha256:
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
            "economic event archive and identity authority differ"
        )
    prefill_authority = resolve_massive_economic_authority_at_origin_v6(
        archive=provider_archive,
        identity_authority=identity_authority,
        decision_at_ms=fill_start,
    )
    postfill_authority = resolve_massive_economic_authority_at_origin_v6(
        archive=provider_archive,
        identity_authority=identity_authority,
        decision_at_ms=fill_close,
    )
    if (
        prefill_authority.decision_at_ms != fill_start
        or postfill_authority.decision_at_ms != fill_close
        or prefill_authority.identity_scope_receipt_sha256
        != _identity_scope(prefill_authority, identity_authority)
        or postfill_authority.identity_scope_receipt_sha256
        != _identity_scope(postfill_authority, identity_authority)
    ):
        raise MassiveAdaptiveEconomicEventTransitionV2Error(
            "economic event authority has the wrong cutoff or identity scope"
        )
    prefill = tuple(
        row
        for row in prefill_authority.selected_events
        if prior_close
        < sourced_effective_at_ms_v5(row.source_event)
        <= fill_start
    )
    prefill_receipts = {row.receipt_sha256 for row in prefill}
    postfill = tuple(
        row
        for row in postfill_authority.selected_events
        if prior_close
        < sourced_effective_at_ms_v5(row.source_event)
        <= fill_close
        and row.receipt_sha256 not in prefill_receipts
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_EVENT_TRANSITION_V2_SCHEMA,
        "prior_session_date": prior_session_date,
        "fill_session_date": fill_session_date,
        "prior_close_at_ms": prior_close,
        "fill_start_at_ms": fill_start,
        "fill_end_at_ms": fill_end,
        "fill_close_at_ms": fill_close,
        "prefill_events": prefill,
        "postfill_events": postfill,
        "provider_archive_receipt_sha256": provider_archive.receipt_sha256,
        "prefill_resolved_authority_receipt_sha256": prefill_authority.receipt_sha256,
        "postfill_resolved_authority_receipt_sha256": postfill_authority.receipt_sha256,
        "applied_event_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in (*prefill, *postfill))
        ),
        "source_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveEconomicEventTransitionV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveEconomicEventTransitionV2",
    "MassiveAdaptiveEconomicEventTransitionV2Error",
    "apply_massive_adaptive_postfill_events_v1",
    "apply_massive_adaptive_prefill_events_v1",
    "build_massive_adaptive_economic_event_transition_v2",
]
