"""Frozen scientific design for ``prelockbox-hold30-alpha-mech8-v3``.

V2 remains available in :mod:`rl_quant.protocol.hold30` as an audit record.
This module intentionally uses a disjoint protocol generation and disjoint
setting IDs so an artifact-producing V3 path cannot silently consume V2
checkpoints, manifests, or result rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

HOLD30_ALPHA_V3_PROTOCOL_GENERATION = "prelockbox-hold30-alpha-mech8-v3"
HOLD30_ALPHA_V3_SUPERSEDED_GENERATION = "prelockbox-hold30-mech8-v2"
HOLD30_ALPHA_V3_BASE_DESIGN = "daily_raw_pit300_hold30_alpha_v3"
HOLD30_ALPHA_V3_CANONICAL_ID = "hold30a-m03-alpha-core"

HOLD30_ALPHA_HORIZONS = (5, 21, 30, 63)
HOLD30_ALPHA_PRIMARY_HORIZON = 30
HOLD30_ALPHA_TRAIN_COST_BPS = 20
HOLD30_ALPHA_VALIDATION_COSTS_BPS = (10, 20, 40)
HOLD30_ALPHA_TE_MIN_ANNUAL = 0.02
HOLD30_ALPHA_TE_TARGET_ANNUAL = 0.04
HOLD30_ALPHA_TE_MAX_ANNUAL = 0.06
HOLD30_ALPHA_BETA_TARGET = 1.0
HOLD30_ALPHA_BETA_TOLERANCE = 0.1
HOLD30_ALPHA_C1_BENCHMARK_ID = (
    "C1-monthly-pit-active300-equal-weight-buy-and-drift"
)
HOLD30_ALPHA_C1_USAGE = ("action-anchor", "active-objective-and-label-benchmark")


class Hold30AlphaV3ProtocolError(ValueError):
    """A V3 protocol identity or invariant is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class Hold30AlphaCheckpointContract:
    """Eligibility-first, lexicographic V3 checkpoint contract.

    Thresholds not supplied by the scientific decision remain ``None``.  That
    incompleteness is intentional and blocks an executable manifest rather
    than allowing an implementation to invent a result-moving value.
    """

    maximum_updates: int
    checkpoint_every_updates: int
    validation_every_updates: int
    minimum_updates: int
    validation_patience: int
    required_active_costs_bps: tuple[int, ...]
    eligibility_order: tuple[str, ...]
    ranking_order: tuple[str, ...]
    projection_distance_max: float | None
    forced_turnover_fraction_max: float | None
    retain: tuple[str, ...]
    deterministic_deployed_action: bool
    shared_six_fold_five_seed_update: bool

    def __post_init__(self) -> None:
        if self.maximum_updates != 128:
            raise Hold30AlphaV3ProtocolError("V3 maximum_updates must be 128")
        if self.checkpoint_every_updates != 8 or self.validation_every_updates != 8:
            raise Hold30AlphaV3ProtocolError(
                "V3 checkpoints and validation must occur every 8 updates"
            )
        if self.minimum_updates != 32 or self.validation_patience != 4:
            raise Hold30AlphaV3ProtocolError(
                "V3 requires 32 minimum updates and four validations of patience"
            )
        if self.required_active_costs_bps != (20, 40):
            raise Hold30AlphaV3ProtocolError(
                "V3 checkpoint eligibility requires active results at 20 and 40 bp"
            )
        if self.eligibility_order != (
            "complete-coverage",
            "active-return-available-at-20bp-and-40bp",
            "annual-tracking-error-in-[0.02,0.06]",
            "market-beta-in-[0.9,1.1]",
            "median-discretionary-sold-age-in-[20,40]",
            "projection-distance-at-or-below-frozen-maximum",
            "forced-turnover-fraction-at-or-below-frozen-maximum",
        ):
            raise Hold30AlphaV3ProtocolError("V3 checkpoint eligibility order drifted")
        if self.ranking_order != (
            "median-20bp-active-return-across-folds-and-seeds-desc",
            "20bp-active-information-ratio-across-folds-and-seeds-desc",
            "20bp-total-sharpe-across-folds-and-seeds-desc",
            "20bp-maximum-drawdown-across-folds-and-seeds-asc",
            "20bp-turnover-and-cost-across-folds-and-seeds-asc",
            "earlier-update-asc",
            "lexical-checkpoint-id-asc",
        ):
            raise Hold30AlphaV3ProtocolError("V3 checkpoint ranking order drifted")
        for name in ("projection_distance_max", "forced_turnover_fraction_max"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise Hold30AlphaV3ProtocolError(f"{name} must be nonnegative or None")
        if self.retain != ("initial", "selected", "final"):
            raise Hold30AlphaV3ProtocolError("V3 must retain initial, selected, and final")
        if (
            not self.deterministic_deployed_action
            or not self.shared_six_fold_five_seed_update
        ):
            raise Hold30AlphaV3ProtocolError(
                "V3 selection requires deterministic actions and one update shared "
                "across all six folds and five seeds"
            )

    @property
    def result_moving_thresholds_complete(self) -> bool:
        return (
            self.projection_distance_max is not None
            and self.forced_turnover_fraction_max is not None
        )


HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT = Hold30AlphaCheckpointContract(
    maximum_updates=128,
    checkpoint_every_updates=8,
    validation_every_updates=8,
    minimum_updates=32,
    validation_patience=4,
    required_active_costs_bps=(20, 40),
    eligibility_order=(
        "complete-coverage",
        "active-return-available-at-20bp-and-40bp",
        "annual-tracking-error-in-[0.02,0.06]",
        "market-beta-in-[0.9,1.1]",
        "median-discretionary-sold-age-in-[20,40]",
        "projection-distance-at-or-below-frozen-maximum",
        "forced-turnover-fraction-at-or-below-frozen-maximum",
    ),
    ranking_order=(
        "median-20bp-active-return-across-folds-and-seeds-desc",
        "20bp-active-information-ratio-across-folds-and-seeds-desc",
        "20bp-total-sharpe-across-folds-and-seeds-desc",
        "20bp-maximum-drawdown-across-folds-and-seeds-asc",
        "20bp-turnover-and-cost-across-folds-and-seeds-asc",
        "earlier-update-asc",
        "lexical-checkpoint-id-asc",
    ),
    projection_distance_max=None,
    forced_turnover_fraction_max=None,
    retain=("initial", "selected", "final"),
    deterministic_deployed_action=True,
    shared_six_fold_five_seed_update=True,
)


@dataclass(frozen=True, slots=True)
class Hold30AlphaDesign:
    """Common design fields that every V3 setting must inherit exactly."""

    design_id: str
    decisions_per_trading_session: int
    target_holding_sessions: int
    scored_tail_sessions: int
    bptt_sessions: int
    alpha_horizons: tuple[int, ...]
    primary_alpha_horizon: int
    training_benchmark_id: str
    training_cost_bps: int
    validation_costs_bps: tuple[int, ...]
    te_min_annual: float
    te_target_annual: float
    te_max_annual: float
    beta_target: float
    beta_tolerance: float
    checkpoint: Hold30AlphaCheckpointContract

    def __post_init__(self) -> None:
        if self.design_id != HOLD30_ALPHA_V3_BASE_DESIGN:
            raise Hold30AlphaV3ProtocolError("unexpected V3 base design ID")
        if self.decisions_per_trading_session != 1:
            raise Hold30AlphaV3ProtocolError("V3 makes exactly one decision per session")
        if self.target_holding_sessions != 30:
            raise Hold30AlphaV3ProtocolError("V3 target holding duration must be 30 sessions")
        if self.scored_tail_sessions != 63 or self.bptt_sessions != 63:
            raise Hold30AlphaV3ProtocolError("V3 scored tail and BPTT must both be 63")
        if self.alpha_horizons != HOLD30_ALPHA_HORIZONS:
            raise Hold30AlphaV3ProtocolError("V3 alpha horizons must be (5, 21, 30, 63)")
        if self.primary_alpha_horizon != HOLD30_ALPHA_PRIMARY_HORIZON:
            raise Hold30AlphaV3ProtocolError("V3 primary alpha horizon must be 30")
        if self.training_benchmark_id != HOLD30_ALPHA_C1_BENCHMARK_ID:
            raise Hold30AlphaV3ProtocolError("V3 training benchmark must be the frozen C1")
        if self.training_cost_bps != HOLD30_ALPHA_TRAIN_COST_BPS:
            raise Hold30AlphaV3ProtocolError("V3 training cost must be 20 bp")
        if self.validation_costs_bps != HOLD30_ALPHA_VALIDATION_COSTS_BPS:
            raise Hold30AlphaV3ProtocolError("V3 validation costs must be (10, 20, 40) bp")
        if (
            self.te_min_annual,
            self.te_target_annual,
            self.te_max_annual,
        ) != (
            HOLD30_ALPHA_TE_MIN_ANNUAL,
            HOLD30_ALPHA_TE_TARGET_ANNUAL,
            HOLD30_ALPHA_TE_MAX_ANNUAL,
        ):
            raise Hold30AlphaV3ProtocolError("V3 annual TE band must be 2%/4%/6%")
        if (self.beta_target, self.beta_tolerance) != (
            HOLD30_ALPHA_BETA_TARGET,
            HOLD30_ALPHA_BETA_TOLERANCE,
        ):
            raise Hold30AlphaV3ProtocolError("V3 beta target must be 1.0 +/- 0.1")


HOLD30_ALPHA_V3_DESIGN = Hold30AlphaDesign(
    design_id=HOLD30_ALPHA_V3_BASE_DESIGN,
    decisions_per_trading_session=1,
    target_holding_sessions=30,
    scored_tail_sessions=63,
    bptt_sessions=63,
    alpha_horizons=HOLD30_ALPHA_HORIZONS,
    primary_alpha_horizon=HOLD30_ALPHA_PRIMARY_HORIZON,
    training_benchmark_id=HOLD30_ALPHA_C1_BENCHMARK_ID,
    training_cost_bps=HOLD30_ALPHA_TRAIN_COST_BPS,
    validation_costs_bps=HOLD30_ALPHA_VALIDATION_COSTS_BPS,
    te_min_annual=HOLD30_ALPHA_TE_MIN_ANNUAL,
    te_target_annual=HOLD30_ALPHA_TE_TARGET_ANNUAL,
    te_max_annual=HOLD30_ALPHA_TE_MAX_ANNUAL,
    beta_target=HOLD30_ALPHA_BETA_TARGET,
    beta_tolerance=HOLD30_ALPHA_BETA_TOLERANCE,
    checkpoint=HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
)


@dataclass(frozen=True, slots=True)
class Hold30AlphaSetting:
    """One immutable row in the V3 eight-setting scientific inventory."""

    setting_index: int
    setting_id: str
    mechanism: str
    objective_mode: str
    description: str
    promotion_eligible: bool
    ablation_of: str | None
    age_aware: bool
    supervised_residual_alpha_heads: bool
    uncertainty_downside_heads: bool
    te_band_mode: str
    beta_targeting: bool
    sharpe_mode: str

    def __post_init__(self) -> None:
        parts = self.setting_id.split("-", 2)
        index_token = parts[1] if len(parts) == 3 else ""
        if (
            parts[0] != "hold30a"
            or len(index_token) != 3
            or index_token[0] not in {"m", "a"}
            or not index_token[1:].isdigit()
            or int(index_token[1:]) != self.setting_index
        ):
            raise Hold30AlphaV3ProtocolError(
                f"V3 setting {self.setting_id!r} must carry index {self.setting_index:02d}"
            )
        if self.objective_mode not in {
            "absolute-net-log-return",
            "c1-active-mean",
            "c1-active-alpha-mean-downside",
            "c1-active-alpha-mean-only",
        }:
            raise Hold30AlphaV3ProtocolError("unknown V3 objective_mode")
        if self.te_band_mode not in {"none", "min-target-max", "target-max-no-floor"}:
            raise Hold30AlphaV3ProtocolError("unknown V3 te_band_mode")
        if self.sharpe_mode not in {
            "none",
            "separate-total-risk-overlay",
            "direct-two-pass-gradient",
        }:
            raise Hold30AlphaV3ProtocolError("unknown V3 sharpe_mode")
        if self.promotion_eligible and self.setting_id != HOLD30_ALPHA_V3_CANONICAL_ID:
            raise Hold30AlphaV3ProtocolError("only canonical m03 may be promotion eligible")
        if self.sharpe_mode != "none" and self.promotion_eligible:
            raise Hold30AlphaV3ProtocolError("Sharpe diagnostics are never promotion eligible")
        if self.uncertainty_downside_heads and not self.supervised_residual_alpha_heads:
            raise Hold30AlphaV3ProtocolError(
                "downside/uncertainty heads require supervised residual-alpha heads"
            )

    @property
    def te_floor_annual(self) -> float | None:
        return (
            HOLD30_ALPHA_TE_MIN_ANNUAL
            if self.te_band_mode == "min-target-max"
            else None
        )

    @property
    def te_target_annual(self) -> float | None:
        return (
            HOLD30_ALPHA_TE_TARGET_ANNUAL
            if self.te_band_mode != "none"
            else None
        )

    @property
    def te_ceiling_annual(self) -> float | None:
        return (
            HOLD30_ALPHA_TE_MAX_ANNUAL
            if self.te_band_mode != "none"
            else None
        )

    @property
    def beta_band(self) -> tuple[float, float] | None:
        if not self.beta_targeting:
            return None
        return (
            HOLD30_ALPHA_BETA_TARGET - HOLD30_ALPHA_BETA_TOLERANCE,
            HOLD30_ALPHA_BETA_TARGET + HOLD30_ALPHA_BETA_TOLERANCE,
        )


HOLD30_ALPHA_V3_SETTINGS: tuple[Hold30AlphaSetting, ...] = (
    Hold30AlphaSetting(
        0,
        "hold30a-m00-legacy-absolute",
        "legacy-scalar-gate",
        "absolute-net-log-return",
        "Legacy scalar-gate absolute-return control under the common V3 ledger.",
        False,
        None,
        False,
        False,
        False,
        "none",
        False,
        "none",
    ),
    Hold30AlphaSetting(
        1,
        "hold30a-m01-persistent-absolute",
        "age-aware-hazard",
        "absolute-net-log-return",
        "Age-aware Hold-30 mechanism trained on absolute net log return.",
        False,
        None,
        True,
        False,
        False,
        "none",
        False,
        "none",
    ),
    Hold30AlphaSetting(
        2,
        "hold30a-m02-active-te",
        "age-aware-hazard",
        "c1-active-mean",
        "Benchmark-relative mean objective with the annual 2%/4%/6% TE band.",
        False,
        None,
        True,
        False,
        False,
        "min-target-max",
        False,
        "none",
    ),
    Hold30AlphaSetting(
        3,
        HOLD30_ALPHA_V3_CANONICAL_ID,
        "age-aware-alpha-hazard",
        "c1-active-alpha-mean-downside",
        "Canonical 30-day alpha mean/downside-uncertainty policy with TE and beta bands.",
        True,
        None,
        True,
        True,
        True,
        "min-target-max",
        True,
        "none",
    ),
    Hold30AlphaSetting(
        4,
        "hold30a-a04-no-uncertainty",
        "age-aware-alpha-hazard",
        "c1-active-alpha-mean-only",
        "Canonical m03 with only the downside/uncertainty heads removed.",
        False,
        HOLD30_ALPHA_V3_CANONICAL_ID,
        True,
        True,
        False,
        "min-target-max",
        True,
        "none",
    ),
    Hold30AlphaSetting(
        5,
        "hold30a-a05-no-te-floor",
        "age-aware-alpha-hazard",
        "c1-active-alpha-mean-downside",
        "Canonical m03 with only the 2% annual TE floor removed.",
        False,
        HOLD30_ALPHA_V3_CANONICAL_ID,
        True,
        True,
        True,
        "target-max-no-floor",
        True,
        "none",
    ),
    Hold30AlphaSetting(
        6,
        "hold30a-a06-sharpe-overlay",
        "age-aware-alpha-hazard",
        "c1-active-alpha-mean-downside",
        "Canonical m03 plus a separately optimized total-risk/Sharpe overlay.",
        False,
        HOLD30_ALPHA_V3_CANONICAL_ID,
        True,
        True,
        True,
        "min-target-max",
        True,
        "separate-total-risk-overlay",
    ),
    Hold30AlphaSetting(
        7,
        "hold30a-a07-direct-sharpe",
        "age-aware-alpha-hazard",
        "c1-active-alpha-mean-downside",
        "Canonical m03 plus a direct two-pass Sharpe-gradient term.",
        False,
        HOLD30_ALPHA_V3_CANONICAL_ID,
        True,
        True,
        True,
        "min-target-max",
        True,
        "direct-two-pass-gradient",
    ),
)

HOLD30_ALPHA_V3_IDS = tuple(setting.setting_id for setting in HOLD30_ALPHA_V3_SETTINGS)
HOLD30_ALPHA_V3_BY_ID = {setting.setting_id: setting for setting in HOLD30_ALPHA_V3_SETTINGS}

# Public single-source names used by V3 model, evaluator, and workflow code.
# The explicit ``_V3_`` spellings above remain available in receipts and tests,
# while these concise aliases keep downstream imports generation-agnostic.
HOLD30_ALPHA_PROTOCOL_GENERATION = HOLD30_ALPHA_V3_PROTOCOL_GENERATION
HOLD30_ALPHA_BASE_DESIGN = HOLD30_ALPHA_V3_BASE_DESIGN
HOLD30_ALPHA_MECH8_SETTINGS = HOLD30_ALPHA_V3_SETTINGS
HOLD30_ALPHA_MECH8_IDS = HOLD30_ALPHA_V3_IDS
HOLD30_ALPHA_MECH8_BY_ID = HOLD30_ALPHA_V3_BY_ID

if tuple(setting.setting_index for setting in HOLD30_ALPHA_V3_SETTINGS) != tuple(range(8)):
    raise RuntimeError("V3 setting indexes must be contiguous and ordered from 0 through 7")
if len(HOLD30_ALPHA_V3_BY_ID) != 8:
    raise RuntimeError("V3 setting IDs must be unique")
if [row.setting_id for row in HOLD30_ALPHA_V3_SETTINGS if row.promotion_eligible] != [
    HOLD30_ALPHA_V3_CANONICAL_ID
]:
    raise RuntimeError("V3 must have exactly one promotion candidate: canonical m03")
if tuple(row.supervised_residual_alpha_heads for row in HOLD30_ALPHA_V3_SETTINGS) != (
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
):
    raise RuntimeError("V3 residual-alpha heads must be absent in m00/m01/m02 only")


def resolve_hold30_alpha_v3_setting(setting_id: str) -> Hold30AlphaSetting:
    """Resolve only a stable V3 ID; V2 IDs and aliases fail closed."""

    try:
        return HOLD30_ALPHA_V3_BY_ID[setting_id]
    except KeyError as exc:
        if setting_id.startswith(("hold30-m", "hold30-a")):
            raise Hold30AlphaV3ProtocolError(
                f"V2 setting ID {setting_id!r} is incompatible with "
                f"{HOLD30_ALPHA_V3_PROTOCOL_GENERATION}"
            ) from exc
        valid = ", ".join(HOLD30_ALPHA_V3_IDS)
        raise Hold30AlphaV3ProtocolError(
            f"unknown Hold-30 alpha V3 setting {setting_id!r}; expected one of: {valid}"
        ) from exc


def resolve_hold30_alpha_setting(setting_id: str) -> Hold30AlphaSetting:
    """Resolve a V3 setting through the shared downstream-facing API."""

    return resolve_hold30_alpha_v3_setting(setting_id)


def validate_hold30_alpha_v3_artifact_identity(
    *, protocol_generation: str, setting_id: str
) -> Hold30AlphaSetting:
    """Reject any non-V3 generation before an artifact may be produced."""

    if protocol_generation != HOLD30_ALPHA_V3_PROTOCOL_GENERATION:
        if protocol_generation == HOLD30_ALPHA_V3_SUPERSEDED_GENERATION:
            raise Hold30AlphaV3ProtocolError(
                "prelockbox-hold30-mech8-v2 was superseded before launch and "
                "cannot identify a V3 artifact"
            )
        raise Hold30AlphaV3ProtocolError(
            f"protocol_generation must be {HOLD30_ALPHA_V3_PROTOCOL_GENERATION!r}"
        )
    return resolve_hold30_alpha_v3_setting(setting_id)


def hold30_alpha_v3_design_payload(
    *,
    checkpoint_contract: Hold30AlphaCheckpointContract = (
        HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT
    ),
) -> dict[str, Any]:
    """Return the JSON-ready design with one explicit checkpoint contract.

    The protocol default remains unresolved and therefore launch-incapable.
    Manifest rendering passes the typed training plan's contract here so a
    future resolved manifest cannot retain stale ``None`` thresholds inside
    its design payload.
    """

    if not isinstance(checkpoint_contract, Hold30AlphaCheckpointContract):
        raise Hold30AlphaV3ProtocolError(
            "design payload requires a typed V3 checkpoint contract"
        )
    design = asdict(HOLD30_ALPHA_V3_DESIGN)
    design["checkpoint"] = asdict(checkpoint_contract)

    return {
        "schema_version": 3,
        "protocol_generation": HOLD30_ALPHA_V3_PROTOCOL_GENERATION,
        "supersedes": HOLD30_ALPHA_V3_SUPERSEDED_GENERATION,
        "v2_reusable_as_implementation_history_only": True,
        "design": design,
        "settings": [asdict(setting) for setting in HOLD30_ALPHA_V3_SETTINGS],
    }


__all__ = [
    "HOLD30_ALPHA_BASE_DESIGN",
    "HOLD30_ALPHA_BETA_TARGET",
    "HOLD30_ALPHA_BETA_TOLERANCE",
    "HOLD30_ALPHA_C1_BENCHMARK_ID",
    "HOLD30_ALPHA_C1_USAGE",
    "HOLD30_ALPHA_HORIZONS",
    "HOLD30_ALPHA_MECH8_BY_ID",
    "HOLD30_ALPHA_MECH8_IDS",
    "HOLD30_ALPHA_MECH8_SETTINGS",
    "HOLD30_ALPHA_PRIMARY_HORIZON",
    "HOLD30_ALPHA_PROTOCOL_GENERATION",
    "HOLD30_ALPHA_TE_MAX_ANNUAL",
    "HOLD30_ALPHA_TE_MIN_ANNUAL",
    "HOLD30_ALPHA_TE_TARGET_ANNUAL",
    "HOLD30_ALPHA_TRAIN_COST_BPS",
    "HOLD30_ALPHA_V3_BASE_DESIGN",
    "HOLD30_ALPHA_V3_BY_ID",
    "HOLD30_ALPHA_V3_CANONICAL_ID",
    "HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT",
    "HOLD30_ALPHA_V3_DESIGN",
    "HOLD30_ALPHA_V3_IDS",
    "HOLD30_ALPHA_V3_PROTOCOL_GENERATION",
    "HOLD30_ALPHA_V3_SETTINGS",
    "HOLD30_ALPHA_V3_SUPERSEDED_GENERATION",
    "HOLD30_ALPHA_VALIDATION_COSTS_BPS",
    "Hold30AlphaCheckpointContract",
    "Hold30AlphaDesign",
    "Hold30AlphaSetting",
    "Hold30AlphaV3ProtocolError",
    "hold30_alpha_v3_design_payload",
    "resolve_hold30_alpha_setting",
    "resolve_hold30_alpha_v3_setting",
    "validate_hold30_alpha_v3_artifact_identity",
]
