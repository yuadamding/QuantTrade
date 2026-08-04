"""Bars-only recurrent PPO workflow with a physically isolated 2026 lockbox.

This module is intentionally development-only.  The checked TOP2000 universe
was selected after the beginning of its historical sample, so no result from
this workflow is reportable or suitable for a capital decision.  The workflow
still enforces a useful research boundary:

* nested purged folds use only pre-2026 dates for validation-only screening;
* confirmation reveals only held-out pre-2026 fold tests, then four final
  members refit on the full pre-2026 cache;
* no 2026 bars file is stat'ed/opened and no 2026 market row/path enters a
  search cache identity, loader call, or trial artifact;
* only :mod:`rl_quant.workflows.top2000_pipeline` can seal four confirmed
  checkpoints and consume the single-use 2026 lockbox.

Inputs are raw regular-session OHLCV bars, point-in-time membership needed for
the action mask, and portfolio state owned by :class:`VectorPortfolioEnv`.
News and stock/market covariates are neither loaded nor represented.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import nn
from torch.nn import functional as F

from rl_quant.datasets.provenance import declared_universe_actions, source_symbol_to_action_index
from rl_quant.datasets.raw_window import (
    BAR_FIELDS,
    RawWindowConfig,
    build_window,
)
from rl_quant.datasets.walk_forward import WalkForwardConfig, WalkForwardFold, generate_walk_forward_folds
from rl_quant.envs import HistoricalMarketData, PortfolioConstraints, VectorPortfolioEnv
from rl_quant.execution import FixedTurnoverTargetWeightExecution
from rl_quant.rl import (
    MaskedDirichlet,
    ObservationBatch,
    OnPolicyRolloutCoordinator,
    PPOActorCritic,
    PPOConfig,
    PPOModelOutput,
    RecurrentPPO,
)


PROTOCOL_VERSION = 1
FEATURE_CACHE_VERSION = 1
TRAIN_END = dt.date(2024, 12, 31)
VALIDATION_START = dt.date(2025, 1, 1)
VALIDATION_END = dt.date(2025, 12, 31)
TEST_START = dt.date(2026, 1, 1)
DEVELOPMENT_LABEL = "development-only"
DEVELOPMENT_ACK = "I acknowledge TOP2000 results are development-only"
_PARTITION = re.compile(r"^(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$")
DEFAULT_WALK_FORWARD = {
    "initial_train_size": 378,
    "validation_size": 63,
    "test_size": 63,
    "label_horizon": 22,
    "purge_size": 22,
    "embargo_size": 5,
    "max_train_size": None,
    "fold_count": 3,
}


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bytes_or_empty(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(_canonical_json(dict(payload)))


def _development_reasons(root: Path, first_train_date: dt.date) -> tuple[str, ...]:
    reasons = [
        "This runner is not a reportability path; it produces development diagnostics only.",
    ]
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        reasons.append(f"Dataset manifest is unavailable or invalid: {type(exc).__name__}.")
        return tuple(reasons)
    if not isinstance(manifest, dict):
        reasons.append("Dataset manifest is not a JSON object.")
        return tuple(reasons)
    selection = manifest.get("universe_selection_date")
    try:
        selected_on = dt.date.fromisoformat(str(selection)[:10])
    except (TypeError, ValueError):
        reasons.append("Universe selection date is missing or invalid.")
    else:
        if selected_on > first_train_date:
            reasons.append(
                f"Universe was selected on {selected_on.isoformat()}, after sample start "
                f"{first_train_date.isoformat()}."
            )
    return tuple(reasons)


@dataclass(frozen=True)
class PartitionRef:
    name: str
    start: str
    end: str
    source_signature: str


@dataclass(frozen=True)
class SearchPlan:
    protocol_version: int
    label: str
    development_only: bool
    development_reasons: tuple[str, ...]
    base_dataset_identity: str
    search_identity: str
    lockbox_partition_names_hash: str
    train: tuple[PartitionRef, ...]
    validation: tuple[PartitionRef, ...]
    bar_seconds: int

    def public_dict(self) -> dict[str, Any]:
        """Return the frozen search payload; it deliberately contains no test path/date."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationPlan:
    protocol_version: int
    label: str
    development_only: bool
    base_dataset_identity: str
    search_identity: str
    lockbox_partition_names_hash: str
    test_identity: str
    test: tuple[PartitionRef, ...]
    bar_seconds: int


def _partition_dates(name: str) -> tuple[dt.date, dt.date]:
    match = _PARTITION.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized bars partition name {name!r}.")
    start, end = (dt.date.fromisoformat(value) for value in match.groups())
    if end < start:
        raise ValueError(f"Partition {name!r} ends before it starts.")
    return start, end


def _bars_only_config(bar_seconds: int) -> RawWindowConfig:
    if isinstance(bar_seconds, bool) or not isinstance(bar_seconds, int) or bar_seconds <= 0:
        raise ValueError("bar_seconds must be a positive integer.")
    if 300 % bar_seconds:
        raise ValueError("bar_seconds must divide the 300-second decision block.")
    return RawWindowConfig(
        block_seconds=300,
        bar_seconds=bar_seconds,
        use_news=False,
        cov_fields=(),
        max_news=1,
    )


def _base_dataset_identity(root: Path) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": _sha256_bytes(_read_bytes_or_empty(root / "manifest.json")),
        "universe_sha256": _sha256_bytes(_read_bytes_or_empty(root / "universe.json")),
        "membership_sha256": _sha256_bytes(
            _read_bytes_or_empty(root / "universe_membership.parquet")
        ),
    }
    return _sha256_bytes(_canonical_json(payload))


def _classified_partition_names(root: Path) -> tuple[list[str], list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    partitions = root / "partitions"
    if not partitions.is_dir():
        raise ValueError(f"Dataset partitions directory is missing: {partitions}.")
    # Enumerate names only.  Search classification must not call exists/stat/open
    # on a lockbox bars file; source validation happens later for the selected
    # pre-2026 refs only.
    names = sorted(path.name for path in partitions.iterdir() if _PARTITION.fullmatch(path.name))
    for name in names:
        start, end = _partition_dates(name)
        # Any file containing a 2026 exchange date belongs wholly to the
        # lockbox.  In particular, search never opens/stats the known
        # 2025-12-30_to_2026-01-03 crossing file; its two 2025 rows are
        # deliberately sacrificed to preserve physical isolation.
        if end >= TEST_START:
            test.append(name)
        elif end <= TRAIN_END:
            train.append(name)
        elif end <= VALIDATION_END:
            # Includes a 2024->2025 crossing partition.  The derived cache
            # retains it once and final refit splits its rows by exchange date.
            validation.append(name)
        else:
            raise ValueError(f"Partition {name!r} cannot be assigned to the calendar protocol.")
    if not train:
        raise ValueError("No through-2024 training partitions were found.")
    if not validation:
        raise ValueError("No 2025 validation partitions were found.")
    if not test:
        raise ValueError("No 2026-or-later lockbox partitions were found.")
    return train, validation, test


def _partition_refs(
    root: Path,
    names: Sequence[str],
    cfg: RawWindowConfig,
    *,
    full_content: bool = False,
) -> tuple[PartitionRef, ...]:
    refs: list[PartitionRef] = []
    for name in names:
        start, end = _partition_dates(name)
        refs.append(
            PartitionRef(
                name=name,
                start=start.isoformat(),
                end=end.isoformat(),
                source_signature=_portable_bars_signature(
                    root, name, cfg, full_content=full_content
                ),
            )
        )
    return tuple(refs)


def _parquet_footer_digest(path: Path) -> tuple[int, int, int, str, str]:
    """Portable, metadata-only Parquet evidence (no inode/timestamp binding)."""

    size = path.stat().st_size
    if size < 12:
        raise ValueError(f"{path} is too small to be a Parquet file.")
    with path.open("rb") as source:
        source.seek(-8, os.SEEK_END)
        trailer = source.read(8)
        if trailer[4:] != b"PAR1":
            raise ValueError(f"{path} does not have a Parquet footer.")
        footer_size = int.from_bytes(trailer[:4], "little")
        if footer_size <= 0 or footer_size + 8 > size:
            raise ValueError(f"{path} has an invalid Parquet footer length.")
        source.seek(-(footer_size + 8), os.SEEK_END)
        footer = source.read(footer_size + 8)
    parquet = pq.ParquetFile(path)
    schema_hash = _sha256_bytes(str(parquet.schema_arrow).encode())
    return (
        size,
        int(parquet.metadata.num_rows),
        int(parquet.metadata.num_row_groups),
        schema_hash,
        _sha256_bytes(footer),
    )


def _portable_bars_signature(
    root: Path,
    name: str,
    cfg: RawWindowConfig,
    *,
    full_content: bool = False,
) -> str:
    path = root / "partitions" / name / "bars.parquet"
    size, rows, row_groups, schema_hash, footer_hash = _parquet_footer_digest(path)
    payload = {
        "relative_path": path.relative_to(root).as_posix(),
        "size": size,
        "rows": rows,
        "row_groups": row_groups,
        "schema_sha256": schema_hash,
        "footer_sha256": footer_hash,
        "content_sha256": _sha256_file(path) if full_content else None,
        "bar_seconds": cfg.bar_seconds,
        "bar_fields": list(cfg.bar_fields),
    }
    return _sha256_bytes(_canonical_json(payload))


def build_search_plan(root: str | Path, *, bar_seconds: int = 300) -> SearchPlan:
    """Plan search without stat'ing or hashing any 2026 bars file."""

    root = Path(root)
    cfg = _bars_only_config(bar_seconds)
    train_names, validation_names, test_names = _classified_partition_names(root)
    train = _partition_refs(root, train_names, cfg)
    validation = _partition_refs(root, validation_names, cfg)
    base_identity = _base_dataset_identity(root)
    lockbox_names_hash = _sha256_bytes(_canonical_json(sorted(test_names)))
    identity_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "validation_start": VALIDATION_START.isoformat(),
            "validation_end": VALIDATION_END.isoformat(),
        },
        "base_dataset_identity": base_identity,
        "bar_seconds": bar_seconds,
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "train": [asdict(value) for value in train],
        "validation": [asdict(value) for value in validation],
    }
    first_train = min(dt.date.fromisoformat(value.start) for value in train)
    return SearchPlan(
        protocol_version=PROTOCOL_VERSION,
        label=DEVELOPMENT_LABEL,
        development_only=True,
        development_reasons=_development_reasons(root, first_train),
        base_dataset_identity=base_identity,
        search_identity=_sha256_bytes(_canonical_json(identity_payload)),
        lockbox_partition_names_hash=lockbox_names_hash,
        train=train,
        validation=validation,
        bar_seconds=bar_seconds,
    )


def build_evaluation_plan(root: str | Path, *, bar_seconds: int = 300) -> EvaluationPlan:
    """Open the lockbox namespace only for the separate evaluation path."""

    root = Path(root)
    cfg = _bars_only_config(bar_seconds)
    search = build_search_plan(root, bar_seconds=bar_seconds)
    _train_names, _validation_names, test_names = _classified_partition_names(root)
    # This internal gate is invoked only after the single-use access marker.
    # Unlike the large pre-2026 planning inventory, the final lockbox identity
    # hashes every Parquet byte so data-page mutations cannot evade the receipt.
    test = _partition_refs(root, test_names, cfg, full_content=True)
    test_identity = _sha256_bytes(
        _canonical_json(
            {
                "protocol_version": PROTOCOL_VERSION,
                "base_dataset_identity": search.base_dataset_identity,
                "lockbox_partition_names_hash": search.lockbox_partition_names_hash,
                "bar_seconds": bar_seconds,
                "test": [asdict(value) for value in test],
            }
        )
    )
    return EvaluationPlan(
        protocol_version=PROTOCOL_VERSION,
        label=DEVELOPMENT_LABEL,
        development_only=True,
        base_dataset_identity=search.base_dataset_identity,
        search_identity=search.search_identity,
        lockbox_partition_names_hash=search.lockbox_partition_names_hash,
        test_identity=test_identity,
        test=test,
        bar_seconds=bar_seconds,
    )


class BarsOnlyObservationAdapter:
    """Create O(A) shared per-asset inputs from current daily OHLCV and weights."""

    observation_key = "asset_features"
    bars_key = "daily_ohlcv"
    asset_feature_dim = len(BAR_FIELDS) + 3  # normalized OHLCV, weight, availability, cash indicator

    def __init__(self, *, cash_index: int = 0, epsilon: float = 1e-6) -> None:
        self.cash_index = int(cash_index)
        self.epsilon = float(epsilon)

    def build(
        self,
        data: HistoricalMarketData,
        *,
        time_index: int,
        weights: torch.Tensor,
        equity: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> ObservationBatch:
        del equity
        if set(data.features) != {self.bars_key}:
            raise ValueError(
                "Bars-only PPO accepts exactly the daily_ohlcv feature; news/covariate fields are forbidden."
            )
        bars = data.features[self.bars_key][:, time_index]
        expected = (data.batch_size, data.num_assets, len(BAR_FIELDS))
        if bars.shape != expected or not bars.is_floating_point():
            raise ValueError(f"daily_ohlcv must have floating shape {expected} at each state.")
        available = data.availability[:, time_index]
        risky = available.clone()
        risky[:, self.cash_index] = False
        work = bars.float() if bars.dtype in (torch.float16, torch.bfloat16) else bars
        transformed = torch.empty_like(work)
        transformed[..., :4] = work[..., :4].clamp_min(self.epsilon).log()
        transformed[..., 4] = work[..., 4].clamp_min(0.0).log1p()
        mask = risky.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = torch.where(mask, transformed, 0.0).sum(dim=1, keepdim=True) / count
        centered = transformed - mean
        variance = torch.where(mask, centered.square(), 0.0).sum(dim=1, keepdim=True) / count
        normalized = centered / variance.add(self.epsilon).sqrt()
        normalized = torch.where(mask, normalized, 0.0).to(dtype=bars.dtype)
        cash = torch.zeros_like(weights)
        cash[:, self.cash_index] = 1.0
        features = torch.cat(
            (
                normalized,
                weights.unsqueeze(-1),
                available.to(dtype=bars.dtype).unsqueeze(-1),
                cash.unsqueeze(-1),
            ),
            dim=-1,
        )
        return ObservationBatch(
            tensors={self.observation_key: features},
            action_mask=available,
            episode_start=episode_start,
        )


class SharedAssetRecurrentActorCritic(PPOActorCritic):
    """Permutation-equivariant O(A) actor with one recurrent global market state.

    Parameters do not grow with the action count.  A shared asset encoder and
    shared score head produce one Dirichlet concentration per action, while a
    masked mean feeds a GRU state and value head.
    """

    def __init__(
        self,
        *,
        observation_key: str,
        asset_feature_dim: int,
        hidden_dim: int,
        action_dim: int,
        shared_mlp_layers: int = 1,
        min_concentration: float = 1e-3,
        max_concentration: float = 1e3,
    ) -> None:
        super().__init__()
        if not observation_key or min(asset_feature_dim, hidden_dim, action_dim, shared_mlp_layers) <= 0:
            raise ValueError("Observation key and all model dimensions must be positive/non-empty.")
        self.observation_key = observation_key
        self.asset_feature_dim = int(asset_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.shared_mlp_layers = int(shared_mlp_layers)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        encoder_layers: list[nn.Module] = []
        input_dim = asset_feature_dim
        for layer_index in range(shared_mlp_layers):
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            if layer_index + 1 < shared_mlp_layers:
                encoder_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.asset_encoder = nn.Sequential(*encoder_layers)
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.asset_actor = nn.Linear(hidden_dim, hidden_dim)
        self.global_actor = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        modules = [module for module in self.asset_encoder.modules() if isinstance(module, nn.Linear)]
        for module in (
            *modules,
            self.recurrent,
            self.asset_actor,
            self.global_actor,
            self.score_head,
            self.value_head,
        ):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.orthogonal_(parameter, gain=math.sqrt(2.0))
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.score_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def initial_recurrent_state(self, observation: ObservationBatch) -> Mapping[str, torch.Tensor]:
        inputs = observation.tensors.get(self.observation_key)
        if inputs is None or inputs.ndim != 3:
            raise ValueError(f"Missing single-step per-asset observation {self.observation_key!r}.")
        return {
            "hidden": torch.zeros(
                (inputs.shape[0], self.hidden_dim), dtype=inputs.dtype, device=inputs.device
            )
        }

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        action_mask: torch.Tensor | None = None,
        recurrent_state: Mapping[str, torch.Tensor] | None = None,
        episode_start: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        burn_in: int = 0,
    ) -> PPOModelOutput:
        if self.observation_key not in observations:
            raise ValueError(f"Missing observation field {self.observation_key!r}.")
        inputs = observations[self.observation_key]
        if not inputs.is_floating_point() or inputs.shape[-2:] != (
            self.action_dim,
            self.asset_feature_dim,
        ):
            raise ValueError(
                f"{self.observation_key} needs [..., {self.action_dim}, {self.asset_feature_dim}]."
            )
        if inputs.ndim not in (3, 4):
            raise ValueError("Shared asset actor expects [batch, asset, feature] or [batch, time, asset, feature].")
        single_step = inputs.ndim == 3
        sequence = inputs.unsqueeze(1) if single_step else inputs
        batch_size, sequence_length = sequence.shape[:2]
        if burn_in < 0 or burn_in >= sequence_length or (single_step and burn_in):
            raise ValueError("burn_in is invalid for this input sequence.")
        device, dtype = sequence.device, sequence.dtype
        if recurrent_state:
            if set(recurrent_state) != {"hidden"}:
                raise ValueError("Recurrent state must contain exactly 'hidden'.")
            hidden = recurrent_state["hidden"]
            if hidden.shape != (batch_size, self.hidden_dim) or hidden.device != device or hidden.dtype != dtype:
                raise ValueError("hidden state has the wrong shape, device, or dtype.")
        else:
            hidden = torch.zeros((batch_size, self.hidden_dim), dtype=dtype, device=device)
        starts = (
            torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
            if episode_start is None
            else (episode_start.unsqueeze(1) if single_step else episode_start)
        )
        valid = (
            torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
            if valid_mask is None
            else (valid_mask.unsqueeze(1) if single_step else valid_mask)
        )
        if starts.shape != valid.shape or starts.dtype != torch.bool or valid.dtype != torch.bool:
            raise ValueError("episode_start and valid_mask must be bool [batch, time].")
        if action_mask is None:
            mask = torch.ones(
                (batch_size, sequence_length, self.action_dim), dtype=torch.bool, device=device
            )
        else:
            mask = action_mask.unsqueeze(1) if single_step else action_mask
            if mask.shape != (batch_size, sequence_length, self.action_dim) or mask.dtype != torch.bool:
                raise ValueError("action_mask must be bool [batch, time, action].")
            mask = torch.where(valid.unsqueeze(-1), mask, torch.ones_like(mask))

        concentrations: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for step in range(sequence_length):
            if step == burn_in and burn_in:
                hidden = hidden.detach()
            hidden = torch.where(starts[:, step].unsqueeze(-1), torch.zeros_like(hidden), hidden)
            asset_state = torch.tanh(self.asset_encoder(sequence[:, step]))
            active = mask[:, step].unsqueeze(-1)
            pooled = torch.where(active, asset_state, 0.0).sum(dim=1) / active.sum(dim=1).clamp_min(1)
            candidate = self.recurrent(pooled, hidden)
            hidden = torch.where(valid[:, step].unsqueeze(-1), candidate, hidden)
            logits = self.score_head(
                torch.tanh(self.asset_actor(asset_state) + self.global_actor(hidden).unsqueeze(1))
            ).squeeze(-1)
            concentration = (F.softplus(logits.float()) + self.min_concentration).clamp_max(
                self.max_concentration
            )
            concentrations.append(concentration)
            values.append(self.value_head(hidden).squeeze(-1))
        concentration_tensor = torch.stack(concentrations, dim=1)
        value_tensor = torch.stack(values, dim=1)
        if single_step:
            distribution = MaskedDirichlet(concentration_tensor[:, 0], mask[:, 0])
            value_tensor = value_tensor[:, 0]
        else:
            distribution = MaskedDirichlet(concentration_tensor, mask)
        return PPOModelOutput(
            distribution=distribution,
            value=value_tensor,
            recurrent_state={"hidden": hidden},
        )


def market_data_from_daily_ohlcv(
    daily_ohlcv: torch.Tensor,
    availability: torch.Tensor,
    exchange_dates: Sequence[str],
) -> HistoricalMarketData:
    """Convert chronological, causal daily bars into the historical MDP tensor."""

    if daily_ohlcv.ndim != 3 or daily_ohlcv.shape[-1] != len(BAR_FIELDS):
        raise ValueError("daily_ohlcv must have shape [date, action, 5].")
    dates, actions, _features = daily_ohlcv.shape
    if dates < 2 or actions < 2:
        raise ValueError("Need at least two exchange dates and CASH plus one risky action.")
    if availability.shape != (dates, actions) or availability.dtype != torch.bool:
        raise ValueError("availability must be bool [date, action].")
    parsed = tuple(dt.date.fromisoformat(value) for value in exchange_dates)
    if len(parsed) != dates or any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("exchange_dates must be unique and strictly increasing.")
    if not bool(availability[:, 0].all().item()):
        raise ValueError("Synthetic CASH must be available on every date.")
    if not daily_ohlcv.is_floating_point() or not bool(torch.isfinite(daily_ohlcv).all().item()):
        raise ValueError("daily_ohlcv must be finite floating point; mask unavailable rows to zero.")
    risky_available = availability[:, 1:]
    risky_bars = daily_ohlcv[:, 1:]
    if bool((risky_bars[..., :4][risky_available] <= 0).any().item()):
        raise ValueError("Available risky OHLC prices must be positive.")
    if bool((risky_bars[..., 4][risky_available] < 0).any().item()):
        raise ValueError("Available risky volume must be nonnegative.")
    close = daily_ohlcv[..., 3]
    pair_valid = availability[:-1] & availability[1:]
    safe_previous = torch.where(pair_valid, close[:-1], torch.ones_like(close[:-1]))
    simple_returns = torch.where(pair_valid, close[1:] / safe_previous - 1.0, 0.0)
    simple_returns[:, 0] = 0.0
    if not bool(torch.isfinite(simple_returns).all().item()) or bool((simple_returns <= -1.0).any().item()):
        raise ValueError("Daily simple returns must be finite and greater than -1.")
    ids = torch.tensor(
        [[int(value.strftime("%Y%m%d")) for value in parsed[:-1]]],
        dtype=torch.long,
        device=daily_ohlcv.device,
    )
    return HistoricalMarketData(
        features={"daily_ohlcv": daily_ohlcv.unsqueeze(0)},
        asset_returns=simple_returns.unsqueeze(0),
        availability=availability.unsqueeze(0),
        decision_ids=ids,
    )


def _daily_from_window(window: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    bars = window["bars"]
    mask = window["bar_mask"]
    if bars.ndim != 4 or bars.shape[-1] != len(BAR_FIELDS) or mask.shape != bars.shape[:-1]:
        raise ValueError("Raw bars window has an unexpected tensor schema.")
    count = mask.sum(dim=-1)
    valid = count > 0
    first = mask.to(dtype=torch.int64).argmax(dim=-1)
    last = mask.shape[-1] - 1 - mask.flip(-1).to(dtype=torch.int64).argmax(dim=-1)
    open_price = bars[..., 0].gather(-1, first.unsqueeze(-1)).squeeze(-1)
    close_price = bars[..., 3].gather(-1, last.unsqueeze(-1)).squeeze(-1)
    high = bars[..., 1].masked_fill(~mask, -torch.inf).amax(dim=-1)
    low = bars[..., 2].masked_fill(~mask, torch.inf).amin(dim=-1)
    volume = bars[..., 4].masked_fill(~mask, 0.0).sum(dim=-1)
    daily = torch.stack((open_price, high, low, close_price, volume), dim=-1)
    daily = torch.where(valid.unsqueeze(-1), daily, 0.0)
    close_block = window["session_close_block"].to(dtype=torch.long)
    day_index = torch.arange(daily.shape[0])
    available = window["avail"][day_index, close_block] & valid
    daily[:, 0] = 0.0
    available[:, 0] = True
    return daily.contiguous(), available.contiguous(), list(window["dates"])


def load_market_data(
    root: str | Path,
    partitions: Sequence[PartitionRef],
    *,
    bar_seconds: int,
    device: str | torch.device,
    date_start: dt.date | None = None,
    date_end: dt.date | None = None,
) -> tuple[HistoricalMarketData, tuple[str, ...]]:
    """Load only the explicitly supplied split partitions and aggregate daily bars."""

    root = Path(root)
    cfg = _bars_only_config(bar_seconds)
    actions = declared_universe_actions(root)
    stock_to_index = source_symbol_to_action_index(root)
    by_date: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for partition in partitions:
        built = build_window(root, partition.name, stock_to_index, len(actions), cfg)
        if built is None:
            raise ValueError(f"Bars partition {partition.name!r} contains no usable exchange date.")
        bars, availability, dates = _daily_from_window(built)
        for index, date_value in enumerate(dates):
            incoming = (bars[index], availability[index])
            previous = by_date.get(date_value)
            if previous is not None and (
                not torch.equal(previous[0], incoming[0]) or not torch.equal(previous[1], incoming[1])
            ):
                raise ValueError(f"Overlapping partitions disagree on exchange date {date_value}.")
            by_date[date_value] = incoming
    ordered_dates = tuple(
        value
        for value in sorted(by_date)
        if (date_start is None or dt.date.fromisoformat(value) >= date_start)
        and (date_end is None or dt.date.fromisoformat(value) <= date_end)
    )
    if len(ordered_dates) < 2:
        raise ValueError("Filtered exchange-date range contains fewer than two states.")
    daily = torch.stack([by_date[value][0] for value in ordered_dates])
    availability = torch.stack([by_date[value][1] for value in ordered_dates])
    data = market_data_from_daily_ohlcv(daily, availability, ordered_dates).to(device)
    return data, ordered_dates


def _build_daily_partition_task(
    arguments: tuple[str, str, dict[str, int], int, RawWindowConfig],
) -> tuple[str, torch.Tensor, torch.Tensor, list[str]]:
    root_value, name, stock_to_index, action_count, cfg = arguments
    built = build_window(Path(root_value), name, stock_to_index, action_count, cfg)
    if built is None:
        raise ValueError(f"Bars partition {name!r} contains no usable exchange date.")
    bars, availability, dates = _daily_from_window(built)
    return name, bars, availability, dates


def _merge_daily_partitions(
    results: Sequence[tuple[str, torch.Tensor, torch.Tensor, list[str]]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    by_date: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    for name, bars, availability, dates in results:
        for index, date_value in enumerate(dates):
            incoming = (bars[index], availability[index], name)
            previous = by_date.get(date_value)
            if previous is not None and (
                not torch.equal(previous[0], incoming[0])
                or not torch.equal(previous[1], incoming[1])
            ):
                raise ValueError(
                    f"Overlapping partitions {previous[2]!r} and {name!r} disagree on {date_value}."
                )
            by_date[date_value] = incoming
    ordered_dates = tuple(sorted(by_date))
    if len(ordered_dates) < 2:
        raise ValueError("Derived daily-bars cache needs at least two exchange dates.")
    daily = torch.stack([by_date[value][0] for value in ordered_dates]).contiguous()
    availability = torch.stack([by_date[value][1] for value in ordered_dates]).contiguous()
    return daily, availability, ordered_dates


def build_daily_cache(
    root: str | Path,
    output_path: str | Path,
    *,
    bar_seconds: int,
    workers: int,
    acknowledgement: str,
) -> dict[str, Any]:
    """CPU prebuild of one portable, immutable, pre-2026 daily-bars cache."""

    if acknowledgement != DEVELOPMENT_ACK:
        raise ValueError(f"Cache build requires --development-ack {DEVELOPMENT_ACK!r}.")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer.")
    root, output_path = Path(root), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable cache {output_path}.")
    plan = build_search_plan(root, bar_seconds=bar_seconds)
    cfg = _bars_only_config(bar_seconds)
    actions = declared_universe_actions(root)
    stock_to_index = source_symbol_to_action_index(root)
    refs = (*plan.train, *plan.validation)
    arguments = [
        (str(root), ref.name, stock_to_index, len(actions), cfg)
        for ref in refs
    ]
    if workers == 1:
        results = [_build_daily_partition_task(value) for value in arguments]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
            results = list(pool.map(_build_daily_partition_task, arguments))
    daily, availability, dates = _merge_daily_partitions(results)
    if any(dt.date.fromisoformat(value) >= TEST_START for value in dates):
        raise RuntimeError("Pre-2026 cache builder encountered a lockbox exchange date.")
    # Detect a source replacement during the potentially long build.
    post_build_plan = build_search_plan(root, bar_seconds=bar_seconds)
    if post_build_plan.search_identity != plan.search_identity:
        raise RuntimeError("Source inventory changed during cache construction; discard and retry.")
    action_hash = _sha256_bytes(_canonical_json(actions))
    date_hash = _sha256_bytes(_canonical_json(list(dates)))
    cache_identity = _sha256_bytes(
        _canonical_json(
            {
                "feature_cache_version": FEATURE_CACHE_VERSION,
                "search_identity": plan.search_identity,
                "action_hash": action_hash,
                "date_hash": date_hash,
                "bar_seconds": bar_seconds,
            }
        )
    )
    payload = {
        "schema_version": 1,
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "label": DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "search_identity": plan.search_identity,
        "base_dataset_identity": plan.base_dataset_identity,
        "lockbox_partition_names_hash": plan.lockbox_partition_names_hash,
        "cache_identity": cache_identity,
        "bar_seconds": bar_seconds,
        "actions": tuple(actions),
        "action_hash": action_hash,
        "exchange_dates": dates,
        "date_hash": date_hash,
        "daily_ohlcv": daily,
        "availability": availability,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return {
        "cache_path": str(output_path),
        "cache_sha256": _sha256_file(output_path),
        "cache_identity": cache_identity,
        "search_identity": plan.search_identity,
        "exchange_date_range": [dates[0], dates[-1]],
        "exchange_dates": len(dates),
        "actions": len(actions),
    }


def load_daily_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required immutable daily-bars cache is absent: {path}.")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_sha256:
        raise ValueError("Daily-bars cache SHA-256 mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "feature_cache_version",
        "development_only",
        "bars_only",
        "search_identity",
        "base_dataset_identity",
        "lockbox_partition_names_hash",
        "cache_identity",
        "actions",
        "action_hash",
        "exchange_dates",
        "date_hash",
        "daily_ohlcv",
        "availability",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Daily-bars cache schema is incomplete.")
    if (
        payload["schema_version"] != 1
        or payload["feature_cache_version"] != FEATURE_CACHE_VERSION
        or payload["development_only"] is not True
        or payload["bars_only"] is not True
    ):
        raise ValueError("Daily-bars cache schema/labels are incompatible.")
    dates = tuple(payload["exchange_dates"])
    actions = tuple(payload["actions"])
    if any(dt.date.fromisoformat(value) >= TEST_START for value in dates):
        raise ValueError("Search cache illegally contains a 2026-or-later exchange date.")
    if _sha256_bytes(_canonical_json(list(dates))) != payload["date_hash"]:
        raise ValueError("Daily-bars cache date identity mismatch.")
    if _sha256_bytes(_canonical_json(list(actions))) != payload["action_hash"]:
        raise ValueError("Daily-bars cache action identity mismatch.")
    daily = payload["daily_ohlcv"]
    availability = payload["availability"]
    if daily.shape != (len(dates), len(actions), len(BAR_FIELDS)):
        raise ValueError("Daily-bars cache tensor shape does not match date/action identities.")
    if availability.shape != daily.shape[:2] or availability.dtype != torch.bool:
        raise ValueError("Daily-bars cache availability schema is invalid.")
    payload["daily_ohlcv"] = daily.to(device)
    payload["availability"] = availability.to(device)
    payload["exchange_dates"] = dates
    payload["actions"] = actions
    payload["cache_sha256"] = actual_hash
    return payload


def _market_from_cache_range(
    cache: Mapping[str, Any],
    start: int,
    stop: int,
) -> HistoricalMarketData:
    """Build one chronology from half-open decision positions [start, stop)."""

    dates = cache["exchange_dates"]
    if not 0 <= start < stop < len(dates):
        raise ValueError(f"Decision range [{start}, {stop}) is outside {len(dates) - 1} decisions.")
    return market_data_from_daily_ohlcv(
        cache["daily_ohlcv"][start : stop + 1],
        cache["availability"][start : stop + 1],
        dates[start : stop + 1],
    )


def sampled_parallel_market_data(
    cache: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    num_envs: int,
    max_episode_steps: int,
    seed: int,
) -> tuple[HistoricalMarketData, tuple[tuple[int, int], ...]]:
    """Sample distinct-start, equal-length chronological episodes.

    Historical ranges may overlap because the first 378-decision fold cannot
    contain eight disjoint 63-step episodes.  Starts are sampled without
    replacement and change deterministically per update; each environment is a
    real chronology and rollout collection ends exactly at its 63-step terminal
    rather than resetting/replaying an episode prefix.
    """

    decision_count = stop - start
    if num_envs <= 0 or max_episode_steps <= 0:
        raise ValueError("num_envs and max_episode_steps must be positive.")
    horizon = max_episode_steps
    start_count = decision_count - horizon + 1
    if start_count < num_envs:
        raise ValueError("Training block has too few distinct full-horizon episode starts.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled_offsets = torch.randperm(start_count, generator=generator)[:num_envs].sort().values.tolist()
    ranges: list[tuple[int, int]] = []
    bars: list[torch.Tensor] = []
    availability: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for env, offset in enumerate(sampled_offsets):
        episode_start = start + int(offset)
        episode_stop = episode_start + horizon
        one = _market_from_cache_range(cache, episode_start, episode_stop)
        ranges.append((episode_start, episode_stop))
        bars.append(one.features["daily_ohlcv"][0])
        availability.append(one.availability[0])
        returns.append(one.asset_returns[0])
    data = HistoricalMarketData(
        features={"daily_ohlcv": torch.stack(bars)},
        asset_returns=torch.stack(returns),
        availability=torch.stack(availability),
        # Overlapping sampled episodes can contain the same real decision.
        # Fabricating a unique ID would violate HistoricalMarketData's exact
        # identity contract; chronological ranges remain in the receipt.
        decision_ids=None,
    )
    return data, tuple(ranges)


@dataclass(frozen=True)
class TrialConfig:
    seed: int = 17
    hidden_dim: int = 128
    shared_mlp_layers: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    entropy_coefficient: float = 0.0
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_sequences: int = 8
    target_kl: float = 0.02
    rollout_horizon: int = 63
    num_envs: int = 8
    sequence_length: int = 21
    burn_in: int = 21
    updates: int = 256
    max_asset_weight: float = 0.01
    max_turnover: float = 0.20
    max_drawdown: float = 0.20
    cost_bps: float = 10.0
    discount: float = 0.99
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        integers = {
            "seed": self.seed,
            "hidden_dim": self.hidden_dim,
            "shared_mlp_layers": self.shared_mlp_layers,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_sequences": self.minibatch_sequences,
            "rollout_horizon": self.rollout_horizon,
            "num_envs": self.num_envs,
            "sequence_length": self.sequence_length,
            "updates": self.updates,
        }
        for name, value in integers.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if isinstance(self.burn_in, bool) or not isinstance(self.burn_in, int) or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer.")
        if self.sequence_length > self.rollout_horizon:
            raise ValueError("sequence_length cannot exceed rollout_horizon.")
        sequence_width = self.burn_in + self.sequence_length
        if self.burn_in >= sequence_width:
            raise ValueError("burn_in must be smaller than recurrent sequence width.")

        positive = (
            "learning_rate",
            "clip_range",
            "value_clip_range",
            "max_grad_norm",
            "target_kl",
        )
        nonnegative = (
            "weight_decay",
            "value_coefficient",
            "entropy_coefficient",
            "cost_bps",
        )
        for name in (*positive, *nonnegative):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, not bool.")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite.")
            if name in positive and numeric <= 0:
                raise ValueError(f"{name} must be positive.")
            if name in nonnegative and numeric < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in ("learning_rate", "clip_range", "value_clip_range", "target_kl"):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} cannot exceed 1.")
        bounded = {
            "discount": (0.0, 1.0, True),
            "gae_lambda": (0.0, 1.0, True),
            "max_turnover": (0.0, 1.0, True),
            "max_asset_weight": (0.0, 1.0, False),
            "max_drawdown": (0.0, 1.0, False),
        }
        for name, (lower, upper, include_lower) in bounded.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, not bool.")
            numeric = float(value)
            lower_ok = numeric >= lower if include_lower else numeric > lower
            if not math.isfinite(numeric) or not lower_ok or numeric > upper:
                bracket = "[" if include_lower else "("
                raise ValueError(f"{name} must lie in {bracket}{lower}, {upper}].")


@dataclass
class PPOStack:
    environment: VectorPortfolioEnv
    model: SharedAssetRecurrentActorCritic
    algorithm: RecurrentPPO
    coordinator: OnPolicyRolloutCoordinator


def _build_environment(data: HistoricalMarketData, trial: TrialConfig) -> VectorPortfolioEnv:
    adapter = BarsOnlyObservationAdapter(cash_index=0)
    return VectorPortfolioEnv(
        data,
        cash_index=0,
        constraints=PortfolioConstraints(
            max_asset_weight=trial.max_asset_weight,
            max_leverage=1.0,
            max_turnover=trial.max_turnover,
            max_drawdown=trial.max_drawdown,
        ),
        execution_model=FixedTurnoverTargetWeightExecution(cost_bps=trial.cost_bps),
        discount=trial.discount,
        observation_adapter=adapter,
    )


def build_ppo_stack(data: HistoricalMarketData, trial: TrialConfig) -> PPOStack:
    adapter = BarsOnlyObservationAdapter(cash_index=0)
    environment = _build_environment(data, trial)
    model = SharedAssetRecurrentActorCritic(
        observation_key=adapter.observation_key,
        asset_feature_dim=adapter.asset_feature_dim,
        hidden_dim=trial.hidden_dim,
        action_dim=data.num_assets,
        shared_mlp_layers=trial.shared_mlp_layers,
    ).to(data.device)
    algorithm = RecurrentPPO(
        model,
        PPOConfig(
            learning_rate=trial.learning_rate,
            weight_decay=trial.weight_decay,
            clip_range=trial.clip_range,
            value_clip_range=trial.value_clip_range,
            value_coefficient=trial.value_coefficient,
            entropy_coefficient=trial.entropy_coefficient,
            max_grad_norm=trial.max_grad_norm,
            epochs=trial.ppo_epochs,
            minibatch_sequences=trial.minibatch_sequences,
            target_kl=trial.target_kl,
            seed=trial.seed,
        ),
    )
    coordinator = OnPolicyRolloutCoordinator(
        environment,
        algorithm,
        horizon=trial.rollout_horizon,
        gae_lambda=trial.gae_lambda,
    )
    return PPOStack(environment, model, algorithm, coordinator)


def _coordinator_for(
    data: HistoricalMarketData,
    algorithm: RecurrentPPO,
    trial: TrialConfig,
) -> tuple[VectorPortfolioEnv, OnPolicyRolloutCoordinator]:
    environment = _build_environment(data, trial)
    coordinator = OnPolicyRolloutCoordinator(
        environment,
        algorithm,
        horizon=trial.rollout_horizon,
        gae_lambda=trial.gae_lambda,
    )
    return environment, coordinator


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_stack(stack: PPOStack, trial: TrialConfig) -> list[dict[str, float | int | bool]]:
    continuation = None
    history: list[dict[str, float | int | bool]] = []
    for update in range(trial.updates):
        result = stack.coordinator.collect(continuation)
        continuation = result.continuation
        sequence = result.buffer.recurrent_sequences(
            sequence_length=trial.sequence_length,
            burn_in=trial.burn_in,
        )
        metrics = stack.algorithm.update(sequence)
        history.append(
            {
                "update": update + 1,
                "reward_mean": result.metrics.reward_mean,
                **{
                    name: bool(value) if isinstance(value, bool) else float(value)
                    for name, value in metrics.items()
                },
            }
        )
    return history


def train_cache_block(
    cache: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    trial: TrialConfig,
) -> tuple[PPOStack, list[dict[str, float | int | bool]], dict[str, Any]]:
    """Train with a deterministic, newly sampled distinct-start 8-env batch per update."""

    stack: PPOStack | None = None
    history: list[dict[str, float | int | bool]] = []
    all_ranges: list[list[list[int]]] = []
    for update in range(trial.updates):
        data, ranges = sampled_parallel_market_data(
            cache,
            start=start,
            stop=stop,
            num_envs=trial.num_envs,
            max_episode_steps=trial.rollout_horizon,
            seed=trial.seed + update,
        )
        all_ranges.append([[left, right] for left, right in ranges])
        if stack is None:
            stack = build_ppo_stack(data, trial)
        else:
            environment, coordinator = _coordinator_for(data, stack.algorithm, trial)
            stack.environment = environment
            stack.coordinator = coordinator
        result = stack.coordinator.collect()
        sequence = result.buffer.recurrent_sequences(
            sequence_length=trial.sequence_length,
            burn_in=trial.burn_in,
        )
        metrics = stack.algorithm.update(sequence)
        history.append(
            {
                "update": update + 1,
                "reward_mean": result.metrics.reward_mean,
                **{
                    name: bool(value) if isinstance(value, bool) else float(value)
                    for name, value in metrics.items()
                },
            }
        )
    if stack is None:
        raise RuntimeError("Training completed no updates.")
    sampling = {
        "schema_version": 1,
        "updates": trial.updates,
        "num_envs": trial.num_envs,
        "ranges_sha256": _sha256_bytes(_canonical_json(all_ranges)),
        "first_update_ranges": all_ranges[0],
        "last_update_ranges": all_ranges[-1],
        "overlap_allowed": True,
        "first_update_unique_decision_fraction": len(
            {
                position
                for left, right in all_ranges[0]
                for position in range(left, right)
            }
        )
        / (trial.num_envs * trial.rollout_horizon),
    }
    return stack, history, sampling


@torch.no_grad()
def evaluate_model(
    model: SharedAssetRecurrentActorCritic,
    data: HistoricalMarketData,
    trial: TrialConfig,
) -> dict[str, Any]:
    if data.batch_size != 1:
        raise ValueError("Deterministic evaluation requires exactly one chronological environment.")
    adapter = BarsOnlyObservationAdapter(cash_index=0)
    environment = VectorPortfolioEnv(
        data,
        cash_index=0,
        constraints=PortfolioConstraints(
            max_asset_weight=trial.max_asset_weight,
            max_leverage=1.0,
            max_turnover=trial.max_turnover,
            max_drawdown=trial.max_drawdown,
        ),
        execution_model=FixedTurnoverTargetWeightExecution(cost_bps=trial.cost_bps),
        discount=trial.discount,
        observation_adapter=adapter,
    )
    evaluator = RecurrentPPO(model, PPOConfig(seed=trial.seed))
    observation, _info = environment.reset()
    state = evaluator.initial_recurrent_state(observation)
    returns: list[torch.Tensor] = []
    turnovers: list[torch.Tensor] = []
    while True:
        action = evaluator.act(observation, deterministic=True, recurrent_state=state)
        transition = environment.step(action)
        returns.append(transition.reward)
        turnovers.append(transition.info["recent_turnover"])
        state = action.recurrent_state
        observation = transition.next_observation
        if bool(transition.done.all().item()):
            break
    daily = torch.cat(returns).double()
    mean = daily.mean()
    std = daily.std(unbiased=False)
    sharpe = torch.where(std > 0, mean / std * math.sqrt(252.0), torch.zeros_like(mean))
    equity_curve = torch.cat(
        (
            torch.ones(1, dtype=daily.dtype, device=daily.device),
            torch.cumprod(1.0 + daily, dim=0),
        )
    )
    peaks = torch.cummax(equity_curve, dim=0).values
    max_drawdown = (1.0 - equity_curve / peaks.clamp_min(1e-12)).max()
    turnover = torch.cat(turnovers).double()
    risky_available = data.availability[:, :-1, 1:].any(dim=-1).reshape(-1)
    risky_coverage = risky_available.double().mean()
    return {
        "observations": int(daily.numel()),
        "decision_coverage": float(risky_coverage.item()),
        "net_total_return": float(equity_curve[-1].item() - 1.0),
        "net_annualized_sharpe": float(sharpe.item()),
        "max_drawdown": float(max_drawdown.item()),
        "mean_total_one_way_turnover": float(turnover.mean().item()),
        "cost_bps": float(trial.cost_bps),
        "daily_net_returns": [float(value) for value in daily.cpu().tolist()],
        "daily_total_one_way_turnover": [float(value) for value in turnover.cpu().tolist()],
        "daily_risky_available": [bool(value) for value in risky_available.cpu().tolist()],
    }


def evaluate_cost_ladder(
    model: SharedAssetRecurrentActorCritic,
    data: HistoricalMarketData,
    trial: TrialConfig,
) -> dict[str, dict[str, Any]]:
    """Deterministic 0/10/20/40-bp accounting sensitivities."""

    costs = (0.0, 10.0, 20.0, 40.0)
    results = {
        f"{cost:g}bp": evaluate_model(model, data, replace(trial, cost_bps=cost))
        for cost in costs
    }
    return {
        "gross_0bp": results["0bp"],
        "base": results["10bp"],
        "stress_20bp": results["20bp"],
        "stress_40bp": results["40bp"],
    }


def run_trial(
    root: str | Path,
    output_dir: str | Path,
    trial: TrialConfig,
    *,
    bar_seconds: int,
    device: str,
    acknowledgement: str,
) -> dict[str, Any]:
    if acknowledgement != DEVELOPMENT_ACK:
        raise ValueError(f"Training requires --development-ack {DEVELOPMENT_ACK!r}.")
    del root, output_dir, trial, bar_seconds, device
    raise RuntimeError(
        "The direct fixed-calendar trial shortcut is retired. Build the immutable daily cache, "
        "run the eight-setting serial-fold screen, and use the confirmation/refit ensemble workflow."
    )


def _trial_from_mapping(value: Mapping[str, Any]) -> TrialConfig:
    allowed = set(TrialConfig.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"Unknown trial config fields: {unknown}.")
    if missing:
        raise ValueError(f"Frozen trial config is missing fields: {missing}.")
    return TrialConfig(**dict(value))


def walk_forward_folds(cache: Mapping[str, Any]) -> tuple[WalkForwardFold, ...]:
    config = WalkForwardConfig(
        decision_axis_id=f"daily-cache-sha256:{cache['cache_identity']}",
        initial_train_size=378,
        validation_size=63,
        test_size=63,
        label_horizon=22,
        purge_size=22,
        embargo_size=5,
        max_train_size=None,
        fold_count=3,
    )
    # The final state has no next-day transition, so it is label support rather
    # than a decision position.
    return generate_walk_forward_folds(cache["exchange_dates"][:-1], config)


def fold_descriptor(fold: WalkForwardFold) -> dict[str, Any]:
    identity = fold.identity
    return {
        "fold_index": identity.ordinal,
        "fold_id": fold.fold_id,
        "train_start": identity.train_start_position,
        "train_stop": identity.train_stop_position,
        "validation_start": identity.validation_start_position,
        "validation_stop": identity.validation_stop_position,
        "test_start": identity.test_start_position,
        "test_stop": identity.test_stop_position,
        "train_first_date": identity.train_first_date.isoformat(),
        "train_last_date": identity.train_last_date.isoformat(),
        "validation_first_date": identity.validation_first_date.isoformat(),
        "validation_last_date": identity.validation_last_date.isoformat(),
        "test_first_date": identity.test_first_date.isoformat(),
        "test_last_date": identity.test_last_date.isoformat(),
        "label_horizon": identity.label_horizon,
        "purge_size": identity.purge_size,
        "embargo_size": identity.embargo_size,
    }


def _mean_metric(folds: Sequence[Mapping[str, Any]], section: str, metric: str) -> float:
    values = [float(value[section][metric]) for value in folds]
    return sum(values) / len(values)


def run_screen_worker(
    cache: Mapping[str, Any],
    output_dir: str | Path,
    *,
    setting_index: int,
    setting_id: str,
    trial: TrialConfig,
    selected_fold_indexes: Sequence[int],
    frozen_folds: Sequence[Mapping[str, Any]],
    screen_plan_sha256: str,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Run all three purged pre-2026 folds serially for one setting."""

    folds = walk_forward_folds(cache)
    expected_descriptors = [fold_descriptor(value) for value in folds]
    if list(frozen_folds) != expected_descriptors:
        raise ValueError("Frozen plan fold boundaries do not match the immutable daily cache.")
    if tuple(selected_fold_indexes) != tuple(range(len(folds))):
        raise ValueError("Each screen setting must execute all three folds serially.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    fold_receipts: list[dict[str, Any]] = []
    for fold in folds:
        _seed_everything(trial.seed)
        stack, history, sampling = train_cache_block(
            cache,
            start=fold.train.start_position,
            stop=fold.train.stop_position,
            trial=trial,
        )
        validation_data = _market_from_cache_range(
            cache, fold.validation.start_position, fold.validation.stop_position
        )
        validation_cost_ladder = evaluate_cost_ladder(stack.model, validation_data, trial)
        checkpoint_path = output_dir / f"fold-{fold.identity.ordinal:02d}.pt"
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "schema_version": 1,
                "label": DEVELOPMENT_LABEL,
                "development_only": True,
                "cache_identity": cache["cache_identity"],
                "screen_plan_sha256": screen_plan_sha256,
                "runtime": dict(runtime),
                "setting_index": setting_index,
                "setting_id": setting_id,
                "fold": fold_descriptor(fold),
                "trial_config": asdict(trial),
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in stack.model.state_dict().items()
                },
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        fold_receipts.append(
            {
                "fold": fold_descriptor(fold),
                "sampling": sampling,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "validation_metrics": validation_cost_ladder["base"],
                "validation_cost_ladder": validation_cost_ladder,
                "fold_test_status": "sealed-for-post-selection-confirmation",
                "last_training_metrics": history[-1],
            }
        )
    receipt = {
        "schema_version": 1,
        "artifact_kind": "screen-validation-only",
        "label": DEVELOPMENT_LABEL,
        "development_only": True,
        "bars_only": True,
        "search_identity": cache["search_identity"],
        "base_dataset_identity": cache["base_dataset_identity"],
        "lockbox_partition_names_hash": cache["lockbox_partition_names_hash"],
        "cache_identity": cache["cache_identity"],
        "cache_sha256": cache["cache_sha256"],
        "screen_plan_sha256": screen_plan_sha256,
        "runtime": dict(runtime),
        "setting_index": setting_index,
        "setting_id": setting_id,
        "trial_config": asdict(trial),
        "folds": fold_receipts,
        "aggregate": {
            "mean_validation_sharpe": _mean_metric(
                fold_receipts, "validation_metrics", "net_annualized_sharpe"
            ),
            "mean_validation_sharpe_20bp": sum(
                float(value["validation_cost_ladder"]["stress_20bp"]["net_annualized_sharpe"])
                for value in fold_receipts
            )
            / len(fold_receipts),
            "mean_validation_return_20bp": sum(
                float(value["validation_cost_ladder"]["stress_20bp"]["net_total_return"])
                for value in fold_receipts
            )
            / len(fold_receipts),
        },
    }
    _write_exclusive_json(output_dir / "screen-receipt.json", receipt)
    return receipt


def run_indexed_worker(
    cache_path: str | Path,
    cache_sha256: str,
    plan_path: str | Path,
    expected_sha256: str,
    output_root: str | Path,
    *,
    index: int,
    device: str,
    acknowledgement: str,
) -> dict[str, Any]:
    if acknowledgement != DEVELOPMENT_ACK:
        raise ValueError(f"Worker requires --development-ack {DEVELOPMENT_ACK!r}.")
    plan_path = Path(plan_path)
    raw = plan_path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError("Frozen trial plan SHA-256 mismatch.")
    payload = json.loads(raw)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("trials"), list):
        raise ValueError("Frozen trial plan needs schema_version=1 and a trials array.")
    cache = load_daily_cache(cache_path, expected_sha256=cache_sha256, device=device)
    if payload.get("cache_sha256") != cache_sha256:
        raise ValueError("Frozen trial plan cache SHA-256 does not match the worker argument.")
    if payload.get("cache_identity") != cache["cache_identity"]:
        raise ValueError("Frozen trial plan cache identity mismatch.")
    if payload.get("search_identity") != cache["search_identity"]:
        raise ValueError("Frozen trial plan search identity mismatch.")
    trials = payload["trials"]
    if len(trials) != 8:
        raise ValueError("Screen plan must contain exactly eight setting rows.")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(trials):
        raise ValueError(f"Worker index {index} is outside [0, {len(trials)}).")
    row = trials[index]
    if set(row) != {"global_index", "setting_id", "fold_indexes", "config"}:
        raise ValueError("Each screen plan row needs global_index, setting_id, fold_indexes, and config.")
    if row["global_index"] != index or not isinstance(row["setting_id"], str) or not row["setting_id"]:
        raise ValueError("Screen setting row identity/index is invalid.")
    trial = _trial_from_mapping(row["config"])
    runtime_fields = {
        "image_ref": payload.get("image_ref"),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "orchestration_manifest_sha256": payload.get("orchestration_manifest_sha256"),
    }
    if any(not isinstance(value, str) or not value for value in runtime_fields.values()):
        raise ValueError("Frozen screen plan lacks immutable image/source/orchestration bindings.")
    return run_screen_worker(
        cache,
        Path(output_root) / f"trial-{index:04d}",
        setting_index=index,
        setting_id=row["setting_id"],
        trial=trial,
        selected_fold_indexes=row["fold_indexes"],
        frozen_folds=payload.get("folds", []),
        screen_plan_sha256=expected_sha256,
        runtime=runtime_fields,
    )


def synthetic_market(*, actions: int = 8, dates: int = 40, device: str = "cpu") -> HistoricalMarketData:
    if actions < 2 or dates < 3:
        raise ValueError("Synthetic smoke data needs >=2 actions and >=3 dates.")
    generator = torch.Generator(device="cpu").manual_seed(1234)
    returns = torch.randn((dates - 1, actions - 1), generator=generator) * 0.005
    close = torch.cat((torch.full((1, actions - 1), 100.0), 100.0 * (1.0 + returns).cumprod(0)))
    open_price = close * (1.0 + torch.randn(close.shape, generator=generator) * 0.001)
    high = torch.maximum(open_price, close) * 1.002
    low = torch.minimum(open_price, close) * 0.998
    volume = torch.full_like(close, 1_000_000.0)
    risky = torch.stack((open_price, high, low, close, volume), dim=-1)
    bars = torch.zeros((dates, actions, 5))
    bars[:, 1:] = risky
    available = torch.ones((dates, actions), dtype=torch.bool)
    exchange_dates = [
        (dt.date(2024, 1, 2) + dt.timedelta(days=index)).isoformat() for index in range(dates)
    ]
    return market_data_from_daily_ohlcv(bars, available, exchange_dates).to(device)


def smoke(*, device: str = "cpu") -> dict[str, Any]:
    trial = TrialConfig(
        seed=7,
        hidden_dim=16,
        ppo_epochs=1,
        rollout_horizon=8,
        sequence_length=4,
        burn_in=1,
        updates=1,
        max_asset_weight=0.5,
    )
    data = synthetic_market(device=device)
    _seed_everything(trial.seed)
    stack = build_ppo_stack(data, trial)
    history = train_stack(stack, trial)
    return {"status": "ok", "actions": data.num_assets, "last_training_metrics": history[-1]}


def _trial_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = TrialConfig()
    for name in (
        "seed",
        "hidden_dim",
        "shared_mlp_layers",
        "ppo_epochs",
        "minibatch_sequences",
        "rollout_horizon",
        "num_envs",
        "sequence_length",
        "burn_in",
        "updates",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in (
        "learning_rate",
        "weight_decay",
        "entropy_coefficient",
        "clip_range",
        "value_clip_range",
        "value_coefficient",
        "max_grad_norm",
        "target_kl",
        "max_asset_weight",
        "max_turnover",
        "max_drawdown",
        "cost_bps",
        "discount",
        "gae_lambda",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))


def _trial_from_args(args: argparse.Namespace) -> TrialConfig:
    return TrialConfig(**{name: getattr(args, name) for name in TrialConfig.__dataclass_fields__})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-search", help="Emit a search plan that contains no 2026 partition.")
    plan.add_argument("data_root")
    plan.add_argument("--bar-seconds", type=int, default=300)
    plan.add_argument("--output")
    cache = commands.add_parser("build-cache", help="Build one immutable pre-2026 daily-bars cache.")
    cache.add_argument("data_root")
    cache.add_argument("output_path")
    cache.add_argument("--bar-seconds", type=int, default=300)
    cache.add_argument("--workers", type=int, default=1)
    cache.add_argument("--development-ack", required=True)
    worker = commands.add_parser("worker", help="Run one SHA-pinned immutable trial-plan row.")
    worker.add_argument("--cache", required=True)
    worker.add_argument("--cache-sha256", required=True)
    worker.add_argument("--plan", required=True)
    worker.add_argument("--plan-sha256", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--index", type=int)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--development-ack", required=True)
    smoke_parser = commands.add_parser("smoke", help="Exercise PPO wiring on tiny synthetic bars.")
    smoke_parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan-search":
        payload = build_search_plan(args.data_root, bar_seconds=args.bar_seconds).public_dict()
        if args.output:
            _write_exclusive_json(Path(args.output), payload)
        else:
            print(_canonical_json(payload).decode(), end="")
    elif args.command == "build-cache":
        print(
            json.dumps(
                build_daily_cache(
                    args.data_root,
                    args.output_path,
                    bar_seconds=args.bar_seconds,
                    workers=args.workers,
                    acknowledgement=args.development_ack,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "worker":
        raw_index = args.index if args.index is not None else os.environ.get("JOB_COMPLETION_INDEX")
        if raw_index is None:
            raise ValueError("worker needs --index or JOB_COMPLETION_INDEX.")
        print(
            json.dumps(
                run_indexed_worker(
                    args.cache,
                    args.cache_sha256,
                    args.plan,
                    args.plan_sha256,
                    args.output_root,
                    index=int(raw_index),
                    device=args.device,
                    acknowledgement=args.development_ack,
                ),
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(smoke(device=args.device), sort_keys=True))
    return 0


__all__ = [
    "BarsOnlyObservationAdapter",
    "DEVELOPMENT_ACK",
    "EvaluationPlan",
    "PPOStack",
    "SearchPlan",
    "SharedAssetRecurrentActorCritic",
    "TrialConfig",
    "build_daily_cache",
    "build_evaluation_plan",
    "build_ppo_stack",
    "build_search_plan",
    "evaluate_model",
    "fold_descriptor",
    "load_daily_cache",
    "load_market_data",
    "main",
    "market_data_from_daily_ohlcv",
    "run_indexed_worker",
    "run_screen_worker",
    "smoke",
    "sampled_parallel_market_data",
    "synthetic_market",
    "train_stack",
]


if __name__ == "__main__":
    raise SystemExit(main())
