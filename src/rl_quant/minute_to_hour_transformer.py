"""Backward-compatibility shim. The minute->hour transformer workflow split into datasets/ (rl_quant.datasets.hour_from_subhour), envs/ (rl_quant.envs.minute_to_hour) and training/ (rl_quant.training.minute_to_hour) in the protocol-first reorganization (see architecture_migration_plan.md)."""

from rl_quant.datasets.hour_from_subhour import *  # noqa: F401,F403
from rl_quant.envs.minute_to_hour import *  # noqa: F401,F403
from rl_quant.training.minute_to_hour import *  # noqa: F401,F403

# Private helpers that existing tests reach through this legacy module path (via `from ... import _x` or
# __import__(..., fromlist=["_x"])). Star-imports drop underscore names, so re-export them explicitly to keep
# the old path a faithful drop-in during the transition.
from rl_quant.datasets.hour_from_subhour import (  # noqa: F401
    _action_feature_mean_std,
    _build_split,
    _canonicalize_subhour_payload,
    _file_sha256,
    _load_payload,
    _masked_mean_std,
    _timestamp_to_epoch_ms,
)
from rl_quant.training.minute_to_hour import _assert_checkpoint_schema  # noqa: F401
