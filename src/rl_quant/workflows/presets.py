"""Named workflow presets.

These replace the ``DEFAULT_ARGS`` lists that used to live inside thin wrapper scripts (e.g.
``train_hourly_from_second_context_rl.py``). Storing them in one registry keeps paths and defaults
in a single place and lets the ``qt`` CLI expand ``--preset NAME`` into the underlying workflow's
CLI arguments. Each preset reproduces the exact roots its original wrapper used (the subhour/minute
TRAIN wrappers used the shared workspace data root; the minute BUILD wrapper used repo-local paths),
so behavior is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rl_quant.paths import default_data_root, project_root


@dataclass(frozen=True)
class Preset:
    workflow: str  # "<group>.<workflow>" the preset targets, e.g. "train.subhour"
    description: str
    build_args: Callable[[], list[str]]  # () -> underlying-script CLI args (roots resolved here)


def _train_subhour_second_context() -> list[str]:
    data_root = default_data_root()
    return [
        "--dataset",
        str(data_root / "rl_hour_from_second" / "top500_1s_recent" / "hour_from_second_dataset.pt"),
        "--output-dir", str(data_root / "rl_hour_from_second_runs"),
        "--run-name", "second_to_hour_causal_transformer",
        "--d-model", "192", "--n-heads", "6", "--minute-layers", "2", "--hour-layers", "3",
        "--max-subhour-tokens", "512", "--episode-length", "32",
        "--max-switches-per-day", "2", "--max-switches-per-episode", "3", "--max-order-legs-per-episode", "6",
    ]


def _build_subhour_second_context() -> list[str]:
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


def _train_direct_bar_minute() -> list[str]:
    data_root = default_data_root()
    return [
        "--dataset", str(data_root / "rl_minute" / "top_volume_1m_recent" / "minute_transformer_dataset.pt"),
        "--output-dir", str(data_root / "rl_minute_runs"),
        "--lookback", "128",
        "--train-end", "2026-06-05T23:59:59+00:00", "--val-end", "2026-06-10T23:59:59+00:00",
        "--test-start", "2026-06-11T00:00:00+00:00",
        "--episode-length", "128", "--switch-cost-bps", "2", "--min-hold-bars", "15", "--cooldown-bars", "5",
        "--max-switches-per-day", "4", "--max-switches-per-episode", "8",
        "--max-order-legs-per-day", "8", "--max-order-legs-per-episode", "16",
        "--q-switch-margin-bps", "5", "--extra-switch-penalty-bps", "1",
    ]


def _build_direct_bar_minute() -> list[str]:
    # The original minute-build wrapper used REPO-LOCAL data/derived dirs (not the shared workspace
    # data root), so reproduce that exactly.
    root = project_root()
    return [
        "--bar-interval", "1m",
        "--stock-bar-dir",
        str(root / "data" / "minute_ohlcv" / "top_us_volume_stocks_nasdaq_1000_2026-06-14_1m_2026-05-25_2026-06-15"),
        "--etf-bar-dir",
        str(root / "data" / "minute_ohlcv" / "top_us_volume_etfs_500_2026-06-14_1m_2026-05-25_2026-06-15"),
        "--stock-universe", str(root / "derived" / "universes" / "top_us_volume_stocks_nasdaq_1000_2026-06-14.csv"),
        "--etf-universe", str(root / "derived" / "universes" / "top_us_volume_etfs_500_2026-06-14.csv"),
        "--output-dir", str(root / "data" / "rl_minute" / "top_volume_1m_recent"),
        "--dataset-file-name", "minute_transformer_dataset.pt",
        "--start", "2026-05-25T00:00:00+00:00", "--end-exclusive", "2026-06-15T00:00:00+00:00",
        "--drop-session-gaps", "--require-same-session-lookback",
    ]


PRESETS: dict[str, Preset] = {
    "train.subhour.second-context": Preset(
        "train.subhour", "Hourly decisions from Polygon 1-second top-500 context.", _train_subhour_second_context
    ),
    "build.subhour.second-context": Preset(
        "build.subhour", "Build the hourly-from-1-second top-500 context dataset.", _build_subhour_second_context
    ),
    "train.direct-bar.minute": Preset(
        "train.direct-bar", "Minute direct-bar causal transformer DQN (recent top-volume).", _train_direct_bar_minute
    ),
    "build.direct-bar.minute": Preset(
        "build.direct-bar", "Build the recent top-volume 1-minute direct-bar dataset.", _build_direct_bar_minute
    ),
}


def resolve_preset(name: str) -> list[str]:
    """Expand a preset name into underlying-workflow CLI arguments."""
    try:
        preset = PRESETS[name]
    except KeyError as exc:
        raise SystemExit(f"qt: unknown preset {name!r}; run `qt preset list` to see available presets.") from exc
    return list(preset.build_args())
