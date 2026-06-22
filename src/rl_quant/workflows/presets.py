"""Named workflow presets.

These replace the ``DEFAULT_ARGS`` lists that used to live inside thin wrapper scripts (e.g.
``train_hourly_from_second_context_rl.py``). Storing them in one registry keeps paths and defaults
in a single place and lets the ``qt`` CLI expand ``--preset NAME`` into the underlying workflow's
CLI arguments. Each preset reproduces the exact roots its original wrapper used (the second/minute
TRAIN wrappers used the shared workspace data root; the minute BUILD wrapper used repo-local paths),
so behavior is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rl_quant.paths import default_data_root


@dataclass(frozen=True)
class Preset:
    workflow: str  # "<group>.<workflow>" the preset targets, e.g. "train.second"
    description: str
    build_args: Callable[[], list[str]]  # () -> underlying-script CLI args (roots resolved here)


def _train_second_context() -> list[str]:
    data_root = default_data_root()
    return [
        "--dataset",
        str(data_root / "rl_hour_from_second" / "top500_1s_recent" / "hour_from_second_dataset.pt"),
        "--output-dir", str(data_root / "rl_hour_from_second_runs"),
        "--run-name", "second_to_hour_causal_transformer",
        "--d-model", "192", "--n-heads", "6", "--second-layers", "2", "--hour-layers", "3",
        "--max-second-tokens", "512", "--episode-length", "32",
        "--max-switches-per-day", "2", "--max-switches-per-episode", "3", "--max-order-legs-per-episode", "6",
    ]


def _build_second_context() -> list[str]:
    data_root = default_data_root()
    second_aggs = data_root / "polygon" / "second_aggs" / "top500_common_stocks_2025_to_2026-06-15"
    universe = data_root / "polygon" / "universes" / "top_500_s3_volume_common_stocks_2026-06-12.csv"
    return [
        "--source-bar-interval", "1s",
        "--stock-bar-dir", str(second_aggs), "--action-bar-dir", str(second_aggs),
        "--stock-universe", str(universe), "--action-universe", str(universe),
        "--output-dir", str(data_root / "rl_hour_from_second" / "top500_1s_recent"),
        "--dataset-file-name", "hour_from_second_dataset.pt",
        "--start", "2026-06-12T00:00:00+00:00", "--end-exclusive", "2026-06-13T00:00:00+00:00",
        "--stock-limit", "500", "--action-count", "500", "--context-bars-per-hour", "3600",
        "--min-active-stock-fraction", "0.01", "--min-context-valid-fraction", "0.005",
        "--max-action-staleness-seconds", "300", "--dense-hourly-grid", "--allow-missing-action-context",
    ]


PRESETS: dict[str, Preset] = {
    "train.second": Preset(
        "train.second", "Hourly decisions from Polygon 1-second top-500 context.", _train_second_context
    ),
    "build.second": Preset(
        "build.second", "Build the hourly-from-1-second top-500 context dataset.", _build_second_context
    ),
}


def resolve_preset(name: str) -> list[str]:
    """Expand a preset name into underlying-workflow CLI arguments."""
    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise SystemExit(f"qt: unknown preset {name!r}; run `qt preset list` to see available presets.") from exc
    return list(preset.build_args())
