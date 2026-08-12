"""Frozen post-v9 representation study for TOP2000 M03R predictive alpha.

V10 is a development-only response to the completed negative v9 predictive
gate.  It keeps factor-residual targets and changes only cross-sectional rank
geometry.  It does not authorize economic policy optimization or 2026 access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

M03R_V10_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v10-rank-geometry-v1"
)
M03R_V10_DESIGN_ID = "daily-ohlcv-top2000-dev-m03r-v10-factor-residual-rank-v1"
M03R_V10_HORIZONS = (5, 21, 30, 63)
M03R_V10_ELIGIBLE_EXECUTION_HORIZONS = (21, 30)
M03R_V10_SETTING_IDS = (
    "V10-P0-factor-residual-standardized-listwise-control",
    "V10-P1-factor-residual-rank-gaussian-correlation",
    "V10-P2-factor-residual-rank-gaussian-21-30-only",
)
M03R_V10_PREDECESSOR_RUN_ID = "qt-m03r-v9-predictive-s17-20260811-a04"
M03R_V10_PREDECESSOR_COMPLETION_COVERAGE_SHA256 = (
    "2acebe99ceec46315b45ac3c6b62e4bdac003ea9bc72ed9db17eef62f34858ee"
)
M03R_V10_PREDECESSOR_TERMINAL_EVIDENCE_SHA256 = (
    "bbb337415ee905be0e63965364b87851c64daf9c1ae2e9fed4e5e31a7151ba2b"
)
M03R_V10_PREDECESSOR_CLEANUP_RECEIPT_FILE_SHA256 = (
    "68913de7aa8ee4290471363bbd113db88682911d40b6d9909ddacd2619a9898d"
)


class M03RV10ProtocolError(ValueError):
    """The frozen v10 research contract drifted."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV10PredictiveSetting:
    setting_index: int
    setting_id: str
    ranking_objective: Literal[
        "standardized-return-listwise",
        "rank-gaussian-correlation",
    ]
    horizon_loss_weights: tuple[float, float, float, float]
    component_weights: tuple[float, float, float] = (0.50, 0.30, 0.20)
    target_mode: Literal["factor-residual"] = "factor-residual"

    def __post_init__(self) -> None:
        expected = (
            (
                0,
                M03R_V10_SETTING_IDS[0],
                "standardized-return-listwise",
                (0.10, 0.35, 0.40, 0.15),
            ),
            (
                1,
                M03R_V10_SETTING_IDS[1],
                "rank-gaussian-correlation",
                (0.10, 0.35, 0.40, 0.15),
            ),
            (
                2,
                M03R_V10_SETTING_IDS[2],
                "rank-gaussian-correlation",
                (0.0, 7.0 / 15.0, 8.0 / 15.0, 0.0),
            ),
        )
        observed = (
            self.setting_index,
            self.setting_id,
            self.ranking_objective,
            self.horizon_loss_weights,
        )
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or not 0 <= self.setting_index < len(expected)
            or observed != expected[self.setting_index]
            or self.component_weights != (0.50, 0.30, 0.20)
            or self.target_mode != "factor-residual"
            or abs(sum(self.horizon_loss_weights) - 1.0) > 1.0e-12
            or abs(sum(self.component_weights) - 1.0) > 1.0e-12
        ):
            raise M03RV10ProtocolError("v10 predictive setting drifted")

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


M03R_V10_SETTINGS = (
    M03RV10PredictiveSetting(
        0,
        M03R_V10_SETTING_IDS[0],
        "standardized-return-listwise",
        (0.10, 0.35, 0.40, 0.15),
    ),
    M03RV10PredictiveSetting(
        1,
        M03R_V10_SETTING_IDS[1],
        "rank-gaussian-correlation",
        (0.10, 0.35, 0.40, 0.15),
    ),
    M03RV10PredictiveSetting(
        2,
        M03R_V10_SETTING_IDS[2],
        "rank-gaussian-correlation",
        (0.0, 7.0 / 15.0, 8.0 / 15.0, 0.0),
    ),
)


@dataclass(frozen=True, slots=True)
class M03RV10PredictiveSpec:
    optimizer_updates: int = 64
    early_stopping_enabled: bool = False
    qualification_updates: tuple[int, ...] = (64,)
    minimum_mean_spearman_rank_ic: float = 0.020
    minimum_positive_rank_ic_fold_count: int = 4
    minimum_mean_top_bottom_spread: float = 0.0
    minimum_positive_spread_fold_count: int = 4
    minimum_simple_sleeve_gross_active_return: float = 0.0
    minimum_simple_sleeve_net_active_return_10bp: float = 0.0
    minimum_gross_positive_fold_count: int = 4
    minimum_break_even_one_way_cost_basis_points: float = 10.0
    seed: int = 17
    chronological_fold_count: int = 6
    h100s_per_worker: int = 2
    maximum_workers: int = 3
    maximum_h100_requests: int = 6
    economic_optimizer_updates: int = 0
    predecessor_gate_passed: bool = False
    v9_state_reuse_authorized: bool = False
    outer_2026_access_authorized: bool = False
    future_selected_universe: bool = True
    development_only: bool = True
    reportable: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if (
            self.optimizer_updates != 64
            or self.early_stopping_enabled
            or self.qualification_updates != (64,)
            or self.minimum_mean_spearman_rank_ic != 0.020
            or self.minimum_positive_rank_ic_fold_count != 4
            or self.minimum_positive_spread_fold_count != 4
            or self.minimum_gross_positive_fold_count != 4
            or self.minimum_break_even_one_way_cost_basis_points != 10.0
            or self.seed != 17
            or self.chronological_fold_count != 6
            or self.h100s_per_worker != 2
            or self.maximum_workers != 3
            or self.maximum_h100_requests != 6
            or self.economic_optimizer_updates != 0
            or self.predecessor_gate_passed
            or self.v9_state_reuse_authorized
            or self.outer_2026_access_authorized
            or not self.future_selected_universe
            or not self.development_only
            or self.reportable
            or self.promotable
        ):
            raise M03RV10ProtocolError("v10 predictive specification drifted")


M03R_V10_PREDICTIVE_SPEC = M03RV10PredictiveSpec()
M03R_V10_PROTOCOL_SHA256 = _sha256(
    {
        "generation": M03R_V10_PROTOCOL_GENERATION,
        "design_id": M03R_V10_DESIGN_ID,
        "horizons": M03R_V10_HORIZONS,
        "eligible_execution_horizons": M03R_V10_ELIGIBLE_EXECUTION_HORIZONS,
        "settings": tuple(asdict(row) for row in M03R_V10_SETTINGS),
        "spec": asdict(M03R_V10_PREDICTIVE_SPEC),
        "predecessor": {
            "run_id": M03R_V10_PREDECESSOR_RUN_ID,
            "completion_coverage_sha256": (
                M03R_V10_PREDECESSOR_COMPLETION_COVERAGE_SHA256
            ),
            "terminal_evidence_sha256": (M03R_V10_PREDECESSOR_TERMINAL_EVIDENCE_SHA256),
            "cleanup_receipt_file_sha256": (
                M03R_V10_PREDECESSOR_CLEANUP_RECEIPT_FILE_SHA256
            ),
            "gate_passed": False,
            "selected_horizon": None,
        },
    }
)


def resolve_m03r_v10_setting(value: int | str) -> M03RV10PredictiveSetting:
    if isinstance(value, bool):
        raise M03RV10ProtocolError("boolean is not a v10 setting identity")
    for row in M03R_V10_SETTINGS:
        if value in {row.setting_index, row.setting_id}:
            return row
    raise M03RV10ProtocolError(f"unknown v10 setting: {value!r}")


__all__ = [
    "M03R_V10_DESIGN_ID",
    "M03R_V10_ELIGIBLE_EXECUTION_HORIZONS",
    "M03R_V10_HORIZONS",
    "M03R_V10_PREDICTIVE_SPEC",
    "M03R_V10_PROTOCOL_GENERATION",
    "M03R_V10_PROTOCOL_SHA256",
    "M03R_V10_SETTINGS",
    "M03R_V10_SETTING_IDS",
    "M03RV10PredictiveSetting",
    "M03RV10ProtocolError",
    "resolve_m03r_v10_setting",
]
