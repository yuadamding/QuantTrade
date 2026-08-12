"""Fail-closed placeholder for the not-yet-authorized M03R-v9 economic panel.

The predictive protocol contains zero economic optimizer updates.  A passing
predictive qualification is necessary but not sufficient to mutate policy
parameters: the selected predictor/horizon/calibration and v2 action must be
frozen under a new economic-generation protocol first.  This module prevents
an operator from mistaking the predictive package for that future trainer.
"""

from __future__ import annotations

from rl_quant.training.top2000_m03r_v9_selection import (
    M03RV9PredictiveQualification,
)


class M03RV9EconomicWorkerNotAuthorized(RuntimeError):
    """No v9 economic generation has been frozen or authorized."""


def require_m03r_v9_economic_generation(
    qualification: M03RV9PredictiveQualification,
) -> None:
    qualification.__post_init__()
    raise M03RV9EconomicWorkerNotAuthorized(
        "predictive qualification does not itself authorize training; freeze a "
        "new source-homogeneous economic protocol after a setting-horizon passes"
    )


__all__ = [
    "M03RV9EconomicWorkerNotAuthorized",
    "require_m03r_v9_economic_generation",
]
