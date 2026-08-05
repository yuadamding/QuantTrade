"""Immutable identities for the superseded Hold-30 V2 mechanism screen.

V2 was superseded before launch by ``prelockbox-hold30-alpha-mech8-v3``.  This
module remains the V2 source of truth for audit and implementation-history
tests only.  It does not authorize new work, and V3 artifact paths deliberately
reject every stable ID below.
"""
from __future__ import annotations

from dataclasses import dataclass


HOLD30_PROTOCOL_GENERATION = "prelockbox-hold30-mech8-v2"
HOLD30_BASE_DESIGN = "daily_raw_pit300_hold30"
HOLD30_CANONICAL_ID = "hold30-m02-age-hazard"


@dataclass(frozen=True, slots=True)
class Hold30Setting:
    """One immutable row in the eight-setting Hold-30 inventory."""

    setting_index: int
    setting_id: str
    mechanism: str
    model_family: str
    description: str
    promotion_eligible: bool = False
    ablation_of: str | None = None
    use_position_age: bool = False
    use_early_exit_penalty: bool = False
    use_turnover_penalty: bool = False
    use_exposure_timing: bool = False

    def __post_init__(self) -> None:
        identity_parts = self.setting_id.split("-", 2)
        index_token = identity_parts[1] if len(identity_parts) == 3 else ""
        if (
            identity_parts[0] != "hold30"
            or len(index_token) != 3
            or index_token[0] not in {"m", "a"}
            or not index_token[1:].isdigit()
            or int(index_token[1:]) != self.setting_index
        ):
            raise ValueError(
                f"setting {self.setting_id!r} must carry its stable two-digit "
                f"index {self.setting_index:02d}"
            )
        if self.mechanism not in {"H0", "H1", "H2", "H3"}:
            raise ValueError(
                f"setting {self.setting_id!r} has unknown mechanism {self.mechanism!r}"
            )
        if self.promotion_eligible and (
            self.mechanism != "H2" or self.ablation_of is not None
        ):
            raise ValueError(
                "only the canonical, non-ablated H2 mechanism may be promotion eligible"
            )


HOLD30_MECH8_SETTINGS: tuple[Hold30Setting, ...] = (
    Hold30Setting(
        0,
        "hold30-m00-legacy-gate",
        "H0",
        "scalar_gate",
        "Ported scalar-gate mechanism control under the common Hold-30 economic ledger.",
    ),
    Hold30Setting(
        1,
        "hold30-m01-slow-gate",
        "H1",
        "scalar_gate",
        "Slow-gate transitional control with actual discretionary-turnover regularization.",
        use_turnover_penalty=True,
    ),
    Hold30Setting(
        2,
        HOLD30_CANONICAL_ID,
        "H2",
        "age_hazard",
        "Canonical per-stock age-aware hazard, entry-score, and risky-exposure policy.",
        promotion_eligible=True,
        use_position_age=True,
        use_early_exit_penalty=True,
        use_turnover_penalty=True,
        use_exposure_timing=True,
    ),
    Hold30Setting(
        3,
        "hold30-m03-sleeve30",
        "H3",
        "sleeve30",
        "Structural staggered 30-sleeve holding-duration control.",
    ),
    Hold30Setting(
        4,
        "hold30-a04-no-age-input",
        "H2",
        "age_hazard",
        "H2 ablation with position-age summaries hidden from the actor.",
        ablation_of=HOLD30_CANONICAL_ID,
        use_early_exit_penalty=True,
        use_turnover_penalty=True,
        use_exposure_timing=True,
    ),
    Hold30Setting(
        5,
        "hold30-a05-no-early-penalty",
        "H2",
        "age_hazard",
        "H2 ablation without the discretionary early-exit penalty.",
        ablation_of=HOLD30_CANONICAL_ID,
        use_position_age=True,
        use_turnover_penalty=True,
        use_exposure_timing=True,
    ),
    Hold30Setting(
        6,
        "hold30-a06-no-turn-penalty",
        "H2",
        "age_hazard",
        "H2 ablation without excess-discretionary-turnover regularization.",
        ablation_of=HOLD30_CANONICAL_ID,
        use_position_age=True,
        use_early_exit_penalty=True,
        use_exposure_timing=True,
    ),
    Hold30Setting(
        7,
        "hold30-a07-no-exp-timing",
        "H2",
        "age_hazard",
        "H2 ablation with the risky-exposure timing residual fixed to zero.",
        ablation_of=HOLD30_CANONICAL_ID,
        use_position_age=True,
        use_early_exit_penalty=True,
        use_turnover_penalty=True,
    ),
)

HOLD30_MECH8_IDS: tuple[str, ...] = tuple(
    setting.setting_id for setting in HOLD30_MECH8_SETTINGS
)
HOLD30_MECH8_BY_ID: dict[str, Hold30Setting] = {
    setting.setting_id: setting for setting in HOLD30_MECH8_SETTINGS
}

if tuple(setting.setting_index for setting in HOLD30_MECH8_SETTINGS) != tuple(range(8)):
    raise RuntimeError("Hold-30 setting indexes must be contiguous and ordered from 0 through 7")
if len(HOLD30_MECH8_BY_ID) != len(HOLD30_MECH8_SETTINGS):
    raise RuntimeError("Hold-30 setting IDs must be unique")
if sum(setting.promotion_eligible for setting in HOLD30_MECH8_SETTINGS) != 1:
    raise RuntimeError("Hold-30 must have exactly one promotion-eligible setting")


def resolve_hold30_setting(setting_id: str) -> Hold30Setting:
    """Resolve a stable ID and reject aliases in artifact-producing paths."""

    try:
        return HOLD30_MECH8_BY_ID[setting_id]
    except KeyError as exc:
        valid = ", ".join(HOLD30_MECH8_IDS)
        raise ValueError(
            f"unknown Hold-30 setting {setting_id!r}; expected one of: {valid}"
        ) from exc


__all__ = [
    "HOLD30_BASE_DESIGN",
    "HOLD30_CANONICAL_ID",
    "HOLD30_MECH8_BY_ID",
    "HOLD30_MECH8_IDS",
    "HOLD30_MECH8_SETTINGS",
    "HOLD30_PROTOCOL_GENERATION",
    "Hold30Setting",
    "resolve_hold30_setting",
]
