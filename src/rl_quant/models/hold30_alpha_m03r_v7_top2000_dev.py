"""Fail-closed model capability routes for TOP2000 development compatibility.

The outer identity is always the disjoint TOP2000 development protocol.  Where
the model-facing causal semantics exactly match an existing M03R v6 policy, this
module identifies that internal construction capability.  It never relabels a
v6 checkpoint or makes a development artifact promotable/reportable.

Loss, execution, ensemble, cache, and evaluation semantics remain outside this
model-only adapter.  A setting without a truthful existing model capability is
rejected instead of being approximated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rl_quant.models.daily_policy import (
    Hold30ModelSwitches,
    resolve_hold30_m03r_v6_model_switches,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_PROTOCOL_GENERATION as M03R_V6_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    resolve_m03r_v6_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_CACHE_REQUIREMENT,
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_SETTINGS,
    M03RTop2000DevSetting,
    resolve_m03r_top2000_dev_setting,
)

M03R_TOP2000_DEV_MODEL_ROUTE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-v6-policy-capability-route-v1"
)
M03R_TOP2000_DEV_POLICY_MECHANISM_GENERATION = "m03r-v3"


class M03RTop2000DevModelRouteError(ValueError):
    """A development model route is unsupported or inconsistent."""


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_V6_MODEL_SETTING_BY_REVIEWED_V7_ID: dict[str, str | None] = {
    "M03R-soft-persistence-active-alpha-hold30-v7": (
        "M03R-soft-persistence-active-alpha-hold30"
    ),
    "P00-no-soft-persistence-v7": "M03R-soft-persistence-active-alpha-hold30",
    "P10-soft-persistence-10bp-v7": ("M03R-soft-persistence-active-alpha-hold30"),
    "A08-fixed-exit-hazard-v7": "A08-fixed-exit-hazard-v6",
    "A11-no-exact-hold-atom-v7": "A11-no-exact-hold-atom",
    "A09-no-long-context-v7": "A09-no-long-context-v6",
    "M02-active-risk-no-alpha-heads-v7": "M02-active-risk-no-alpha-heads-v6",
    "A04-no-downside-score-adjustment-v7": ("A04-no-downside-score-adjustment-v6"),
    # Existing m03r-v3 always chooses either a confidence head or a learned
    # active-risk head. It has no fixed 2% model output, so approximation is
    # forbidden until a distinct capability is implemented.
    "A12-fixed-2pct-active-risk-budget-v7": None,
    # Factor projection is post-model execution semantics; the alpha policy is
    # otherwise canonical and can be constructed truthfully.
    "A10-no-factor-neutral-projection-v7": (
        "M03R-soft-persistence-active-alpha-hold30"
    ),
    "A06-sharpe-overlay-v7": "A06-sharpe-overlay-v6",
    "A07-direct-sharpe-v7": "A07-direct-sharpe-v6",
}


def _required_non_model_bindings(setting: M03RTop2000DevSetting) -> tuple[str, ...]:
    result = [
        "top2000-dev-cache-contract",
        "v7-nav-session-proportional-persistence-objective",
        "cause-typed-chronological-ledger",
        "development-only-receipt-writer",
    ]
    if setting.factor_sector_neutral_projection:
        result.append("factor-sector-neutral-projection-execution")
    else:
        result.append("no-factor-sector-projection-execution")
    if setting.sharpe_mode == "separate-total-risk-overlay":
        result.append("separate-overlay-optimizer-and-objective")
    elif setting.sharpe_mode == "direct-full-batch-two-pass-gradient":
        result.append("full-effective-batch-two-pass-sharpe-objective")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class M03RTop2000DevModelRoute:
    """One outer development identity and its internal v6 policy capability."""

    setting_index: int
    setting_id: str
    reviewed_v7_setting_id: str
    source_v6_model_setting_id: str | None
    model_capability_supported: bool
    unsupported_model_semantics: tuple[str, ...]
    required_non_model_bindings: tuple[str, ...]
    source_policy_protocol_generation: str = M03R_V6_PROTOCOL_GENERATION
    source_policy_mechanism_generation: str = (
        M03R_TOP2000_DEV_POLICY_MECHANISM_GENERATION
    )
    cache_contract_bound: bool = False
    development_only: bool = True
    training_authorized: bool = False
    promotion_eligible: bool = False
    scientific_reporting_eligible: bool = False
    protocol_generation: str = M03R_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_TOP2000_DEV_DESIGN_ID
    schema: str = M03R_TOP2000_DEV_MODEL_ROUTE_SCHEMA

    def __post_init__(self) -> None:
        setting = resolve_m03r_top2000_dev_setting(self.setting_id)
        if (
            self.setting_index != setting.setting_index
            or self.reviewed_v7_setting_id != setting.reviewed_v7_setting_id
        ):
            raise M03RTop2000DevModelRouteError(
                "TOP2000 development model route identity drifted"
            )
        expected_source = _V6_MODEL_SETTING_BY_REVIEWED_V7_ID[
            setting.reviewed_v7_setting_id
        ]
        if self.source_v6_model_setting_id != expected_source:
            raise M03RTop2000DevModelRouteError("v6 policy capability mapping drifted")
        expected_supported = expected_source is not None
        if self.model_capability_supported is not expected_supported:
            raise M03RTop2000DevModelRouteError("model support status drifted")
        if expected_supported:
            if self.unsupported_model_semantics:
                raise M03RTop2000DevModelRouteError(
                    "supported model route cannot claim unsupported semantics"
                )
            _validate_model_capability(setting, self._source_switches())
            assert self.source_v6_model_setting_id is not None
            source_setting = resolve_m03r_v6_setting(self.source_v6_model_setting_id)
            if (
                source_setting.slow_context_trading_sessions
                != setting.learned_temporal_context_trading_sessions
            ):
                raise M03RTop2000DevModelRouteError(
                    "existing v6 context capability cannot preserve the reviewed row"
                )
        elif self.unsupported_model_semantics != (
            "fixed-2pct-active-risk-budget-has-no-existing-m03r-v3-model-path",
        ):
            raise M03RTop2000DevModelRouteError(
                "unsupported fixed-budget route must name the exact missing capability"
            )
        if self.required_non_model_bindings != _required_non_model_bindings(setting):
            raise M03RTop2000DevModelRouteError(
                "non-model compatibility bindings drifted"
            )
        if (
            self.source_policy_protocol_generation != M03R_V6_PROTOCOL_GENERATION
            or self.source_policy_mechanism_generation
            != M03R_TOP2000_DEV_POLICY_MECHANISM_GENERATION
            or self.schema != M03R_TOP2000_DEV_MODEL_ROUTE_SCHEMA
        ):
            raise M03RTop2000DevModelRouteError("model source identity drifted")
        if (
            self.cache_contract_bound
            or not self.development_only
            or self.training_authorized
            or self.promotion_eligible
            or self.scientific_reporting_eligible
        ):
            raise M03RTop2000DevModelRouteError(
                "model routes remain unbound, development-only, and non-authorizing"
            )

    def _source_switches(self) -> Hold30ModelSwitches:
        if self.source_v6_model_setting_id is None:
            raise M03RTop2000DevModelRouteError(
                "TOP2000 development setting has unsupported existing model semantics: "
                + ", ".join(self.unsupported_model_semantics)
            )
        return resolve_hold30_m03r_v6_model_switches(self.source_v6_model_setting_id)

    @property
    def route_sha256(self) -> str:
        return _sha256(asdict(self))


def _validate_model_capability(
    setting: M03RTop2000DevSetting,
    switches: Hold30ModelSwitches,
) -> None:
    """Validate fields actually consumed by the current m03r-v3 policy."""

    expected = {
        "mechanism": "H2",
        "use_age_input": True,
        "use_exposure_timing": False,
        "use_alpha_head": setting.residual_alpha_head_mode != "none",
        "use_uncertainty": setting.residual_alpha_head_mode == "mean-and-downside",
        "use_total_risk_overlay": (
            setting.sharpe_mode == "separate-total-risk-overlay"
        ),
        "use_direct_sharpe": (
            setting.sharpe_mode == "direct-full-batch-two-pass-gradient"
        ),
        "use_confidence_scaled_active_risk": (
            setting.active_risk_budget_mode
            == M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE
        ),
        "use_three_way_exit_action": setting.exit_hazard_mode == "learned-age-aware",
        "allow_exact_hold_atom": (
            setting.exit_hazard_mode == "learned-age-aware"
            and setting.exact_hold_action_supported
        ),
    }
    observed = {name: getattr(switches, name) for name in expected}
    if observed != expected:
        raise M03RTop2000DevModelRouteError(
            f"existing v6 model capability cannot preserve {setting.setting_id}: "
            f"expected {expected}, observed {observed}"
        )


def _build_route(setting: M03RTop2000DevSetting) -> M03RTop2000DevModelRoute:
    source = _V6_MODEL_SETTING_BY_REVIEWED_V7_ID[setting.reviewed_v7_setting_id]
    unsupported = (
        ()
        if source is not None
        else ("fixed-2pct-active-risk-budget-has-no-existing-m03r-v3-model-path",)
    )
    return M03RTop2000DevModelRoute(
        setting_index=setting.setting_index,
        setting_id=setting.setting_id,
        reviewed_v7_setting_id=setting.reviewed_v7_setting_id,
        source_v6_model_setting_id=source,
        model_capability_supported=source is not None,
        unsupported_model_semantics=unsupported,
        required_non_model_bindings=_required_non_model_bindings(setting),
    )


M03R_TOP2000_DEV_MODEL_ROUTES = tuple(
    _build_route(setting) for setting in M03R_TOP2000_DEV_SETTINGS
)
M03R_TOP2000_DEV_MODEL_ROUTES_BY_ID = {
    route.setting_id: route for route in M03R_TOP2000_DEV_MODEL_ROUTES
}


def resolve_m03r_top2000_dev_model_route(
    setting_id: str,
) -> M03RTop2000DevModelRoute:
    """Resolve the typed route, including an explicit unsupported route."""

    resolve_m03r_top2000_dev_setting(setting_id)
    return M03R_TOP2000_DEV_MODEL_ROUTES_BY_ID[setting_id]


def resolve_m03r_top2000_dev_model_switches(
    setting_id: str,
) -> Hold30ModelSwitches:
    """Return a truthful internal v6 model capability or fail closed.

    The returned switch object retains its original v6 setting identity.  It is
    an internal construction capability, never the identity of the outer
    TOP2000 development artifact.  Loss-only switch fields remain subordinate
    to the separately bound TOP2000 v7 objective contract.
    """

    route = resolve_m03r_top2000_dev_model_route(setting_id)
    return route._source_switches()


def require_m03r_top2000_dev_model_route_training_ready(setting_id: str) -> None:
    """Fail closed until cache and all non-model development adapters are bound."""

    route = resolve_m03r_top2000_dev_model_route(setting_id)
    if not route.model_capability_supported:
        route._source_switches()
    blockers = list(route.required_non_model_bindings)
    if not M03R_TOP2000_DEV_CACHE_REQUIREMENT.cache_contract_bound:
        blockers.append("top2000-dev-cache-contract-is-unbound")
    raise M03RTop2000DevModelRouteError(
        "TOP2000 development model route is not training ready: " + ", ".join(blockers)
    )


__all__ = [
    "M03R_TOP2000_DEV_MODEL_ROUTES",
    "M03R_TOP2000_DEV_MODEL_ROUTES_BY_ID",
    "M03R_TOP2000_DEV_MODEL_ROUTE_SCHEMA",
    "M03R_TOP2000_DEV_POLICY_MECHANISM_GENERATION",
    "M03RTop2000DevModelRoute",
    "M03RTop2000DevModelRouteError",
    "require_m03r_top2000_dev_model_route_training_ready",
    "resolve_m03r_top2000_dev_model_route",
    "resolve_m03r_top2000_dev_model_switches",
]
