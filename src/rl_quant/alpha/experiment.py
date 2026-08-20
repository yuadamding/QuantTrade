"""Immutable alpha experiment specifications and project-wide trial ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from rl_quant.alpha.contracts import PITAlphaDataError
from rl_quant.protocol.canonical_artifact import semantic_sha256


ALPHA_EXPERIMENT_SPEC_SCHEMA = "rl-quant.alpha-experiment-spec-v1"
ALPHA_TRIAL_REGISTRY_SCHEMA = "rl-quant.alpha-trial-registry-v1"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PITAlphaDataError(f"{name} must be a non-empty canonical string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PITAlphaDataError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class AlphaModelConfig:
    intraday_token_dimension: int
    intraday_layers: int
    intraday_heads: int
    cross_day_dimension: int
    cross_day_layers: int
    cross_day_heads: int
    market_latent_count: int
    dropout: float

    def validate(self) -> None:
        for name in (
            "intraday_token_dimension",
            "intraday_layers",
            "intraday_heads",
            "cross_day_dimension",
            "cross_day_layers",
            "cross_day_heads",
            "market_latent_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PITAlphaDataError(f"{name} must be a positive integer")
        if self.intraday_token_dimension % self.intraday_heads != 0:
            raise PITAlphaDataError("intraday dimension must divide across attention heads")
        if self.cross_day_dimension % self.cross_day_heads != 0:
            raise PITAlphaDataError("cross-day dimension must divide across attention heads")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= float(self.dropout) < 1.0:
            raise PITAlphaDataError("model dropout is outside [0, 1)")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AlphaOptimizerConfig:
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    warmup_fraction: float
    terminal_learning_rate_fraction: float
    gradient_clip_norm: float
    terminal_epoch: int
    precision: Literal["bf16", "fp32"] = "bf16"

    def validate(self) -> None:
        for name in (
            "encoder_learning_rate",
            "head_learning_rate",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise PITAlphaDataError(f"{name} must be positive")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0.0:
            raise PITAlphaDataError("weight decay cannot be negative")
        for name in ("warmup_fraction", "terminal_learning_rate_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
                raise PITAlphaDataError(f"{name} must be in (0, 1]")
        if (
            isinstance(self.terminal_epoch, bool)
            or not isinstance(self.terminal_epoch, int)
            or self.terminal_epoch <= 0
        ):
            raise PITAlphaDataError("terminal epoch must be a positive fixed integer")
        if self.precision not in {"bf16", "fp32"}:
            raise PITAlphaDataError("optimizer precision is unsupported")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AlphaFoldConfig:
    outer_fold_count: int
    minimum_outer_sessions: int
    purge_sessions: int
    embargo_sessions: int
    outer_economic_support_disjoint: bool
    terminal_checkpoint_fixed: bool

    def validate(self) -> None:
        for name in (
            "outer_fold_count",
            "minimum_outer_sessions",
            "purge_sessions",
            "embargo_sessions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PITAlphaDataError(f"{name} must be a nonnegative integer")
        if self.outer_fold_count < 2 or self.minimum_outer_sessions <= 0:
            raise PITAlphaDataError("chronological validation needs multiple nonempty folds")
        if not self.outer_economic_support_disjoint or not self.terminal_checkpoint_fixed:
            raise PITAlphaDataError("outer support and checkpoint selection must be frozen")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AlphaExperimentSpec:
    dataset_receipt_sha256: str
    universe_rule_sha256: str
    target_spec_sha256: str
    decision_time_rule: str
    fill_time_rule: str
    input_modalities: tuple[str, ...]
    intraday_resolution_seconds: int
    context_sessions: int
    primary_horizon: int
    auxiliary_horizons: tuple[int, ...]
    objective_kind: str
    model_config: AlphaModelConfig
    optimizer_config: AlphaOptimizerConfig
    fold_config: AlphaFoldConfig
    seed: int
    trial_registry_parent_sha256: str
    economic_optimization_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    prospective_access_authorized: bool = False
    schema: str = ALPHA_EXPERIMENT_SPEC_SCHEMA

    def validate(self) -> None:
        for name in (
            "dataset_receipt_sha256",
            "universe_rule_sha256",
            "target_spec_sha256",
            "trial_registry_parent_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in ("decision_time_rule", "fill_time_rule", "objective_kind"):
            _text(name, getattr(self, name))
        if (
            not self.input_modalities
            or tuple(sorted(set(self.input_modalities))) != self.input_modalities
        ):
            raise PITAlphaDataError("input modalities must be sorted and unique")
        for name in ("intraday_resolution_seconds", "context_sessions", "primary_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PITAlphaDataError(f"{name} must be a positive integer")
        if tuple(sorted(set(self.auxiliary_horizons))) != self.auxiliary_horizons:
            raise PITAlphaDataError("auxiliary horizons must be sorted and unique")
        if self.primary_horizon in self.auxiliary_horizons:
            raise PITAlphaDataError("primary horizon cannot be duplicated as auxiliary")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PITAlphaDataError("experiment seed must be nonnegative")
        self.model_config.validate()
        self.optimizer_config.validate()
        self.fold_config.validate()
        if any(
            (
                self.economic_optimization_authorized,
                self.reinforcement_learning_authorized,
                self.prospective_access_authorized,
            )
        ):
            raise PITAlphaDataError("a predictive experiment spec cannot authorize downstream stages")
        if self.schema != ALPHA_EXPERIMENT_SPEC_SCHEMA:
            raise PITAlphaDataError("alpha experiment schema drifted")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "universe_rule_sha256": self.universe_rule_sha256,
            "target_spec_sha256": self.target_spec_sha256,
            "decision_time_rule": self.decision_time_rule,
            "fill_time_rule": self.fill_time_rule,
            "input_modalities": self.input_modalities,
            "intraday_resolution_seconds": self.intraday_resolution_seconds,
            "context_sessions": self.context_sessions,
            "primary_horizon": self.primary_horizon,
            "auxiliary_horizons": self.auxiliary_horizons,
            "objective_kind": self.objective_kind,
            "model_config": self.model_config.payload(),
            "optimizer_config": self.optimizer_config.payload(),
            "fold_config": self.fold_config.payload(),
            "seed": self.seed,
            "trial_registry_parent_sha256": self.trial_registry_parent_sha256,
            "economic_optimization_authorized": self.economic_optimization_authorized,
            "reinforcement_learning_authorized": self.reinforcement_learning_authorized,
            "prospective_access_authorized": self.prospective_access_authorized,
        }

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())


TrialChangeKind = Literal[
    "architecture",
    "data-modality",
    "target",
    "loss",
    "seed",
    "risk-budget",
    "portfolio-rule",
    "cost-model",
    "threshold",
    "historical-generation",
]


@dataclass(frozen=True, slots=True)
class AlphaDiscoverySetting:
    setting_id: str
    round_id: Literal["A", "B"]
    input_modalities: tuple[str, ...]
    context_sessions: int
    market_latent_count: int
    multi_horizon: bool
    rank_loss: bool
    distributional: bool
    comparison_setting_id: str | None

    def validate(self) -> None:
        _text("discovery setting ID", self.setting_id)
        if self.round_id not in {"A", "B"} or not self.setting_id.startswith(
            self.round_id
        ):
            raise PITAlphaDataError("discovery setting round identity drifted")
        if (
            not self.input_modalities
            or tuple(sorted(set(self.input_modalities))) != self.input_modalities
            or self.context_sessions not in {252, 504}
            or self.market_latent_count not in {0, 32}
        ):
            raise PITAlphaDataError("discovery representation inventory drifted")
        if self.round_id == "A" and self.comparison_setting_id not in {None, "A00", "A02"}:
            raise PITAlphaDataError("Round A comparison identity drifted")
        if self.round_id == "B" and self.comparison_setting_id not in {"A07", "B02"}:
            raise PITAlphaDataError("Round B must compare with its frozen bars-only control")


ALPHA_DISCOVERY_SETTINGS: tuple[AlphaDiscoverySetting, ...] = (
    AlphaDiscoverySetting("A00", "A", ("bars-daily",), 252, 0, False, False, False, None),
    AlphaDiscoverySetting("A01", "A", ("bars-5m",), 252, 0, False, False, False, "A00"),
    AlphaDiscoverySetting("A02", "A", ("bars-5m",), 252, 32, False, False, False, "A00"),
    AlphaDiscoverySetting("A03", "A", ("bars-5m",), 504, 32, False, False, False, "A02"),
    AlphaDiscoverySetting("A04", "A", ("bars-5m",), 252, 32, True, False, False, "A02"),
    AlphaDiscoverySetting("A05", "A", ("bars-5m",), 252, 32, False, True, False, "A02"),
    AlphaDiscoverySetting("A06", "A", ("bars-5m",), 252, 32, False, False, True, "A02"),
    AlphaDiscoverySetting("A07", "A", ("bars-5m",), 252, 32, False, True, True, "A02"),
    AlphaDiscoverySetting("B00", "B", ("bars-5m", "quotes"), 252, 32, False, True, True, "A07"),
    AlphaDiscoverySetting("B01", "B", ("bars-5m", "trades"), 252, 32, False, True, True, "A07"),
    AlphaDiscoverySetting(
        "B02", "B", ("bars-5m", "quotes", "trades"), 252, 32, False, True, True, "A07"
    ),
    AlphaDiscoverySetting(
        "B03", "B", ("bars-5m", "quotes", "trades"), 252, 32, True, True, True, "B02"
    ),
)


def validate_alpha_discovery_settings(
    settings: Sequence[AlphaDiscoverySetting] = ALPHA_DISCOVERY_SETTINGS,
) -> None:
    expected_ids = tuple(f"A{index:02d}" for index in range(8)) + tuple(
        f"B{index:02d}" for index in range(4)
    )
    for setting in settings:
        setting.validate()
    if tuple(setting.setting_id for setting in settings) != expected_ids:
        raise PITAlphaDataError("alpha discovery panel must contain exact A00-A07 and B00-B03")
    if sum(setting.input_modalities == ("bars-daily",) for setting in settings) != 1:
        raise PITAlphaDataError("alpha discovery panel needs exactly one corrected daily control")


@dataclass(frozen=True, slots=True)
class AlphaTrialRecord:
    trial_index: int
    trial_id: str
    experiment_spec_sha256: str
    change_kind: TrialChangeKind
    declared_at_ms: int
    outer_outcomes_opened_at_declaration: bool
    result_receipt_sha256: str | None = None

    def validate(self) -> None:
        if (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 0
        ):
            raise PITAlphaDataError("trial index must be nonnegative")
        _text("trial ID", self.trial_id)
        _digest("trial experiment receipt", self.experiment_spec_sha256)
        if self.change_kind not in {
            "architecture",
            "data-modality",
            "target",
            "loss",
            "seed",
            "risk-budget",
            "portfolio-rule",
            "cost-model",
            "threshold",
            "historical-generation",
        }:
            raise PITAlphaDataError("trial change kind is unsupported")
        if (
            isinstance(self.declared_at_ms, bool)
            or not isinstance(self.declared_at_ms, int)
            or self.declared_at_ms < 0
        ):
            raise PITAlphaDataError("trial declaration timestamp is invalid")
        if self.outer_outcomes_opened_at_declaration:
            raise PITAlphaDataError("a result-moving trial was declared after outer access")
        if self.result_receipt_sha256 is not None:
            _digest("trial result receipt", self.result_receipt_sha256)

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "trial_index": self.trial_index,
            "trial_id": self.trial_id,
            "experiment_spec_sha256": self.experiment_spec_sha256,
            "change_kind": self.change_kind,
            "declared_at_ms": self.declared_at_ms,
            "outer_outcomes_opened_at_declaration": (
                self.outer_outcomes_opened_at_declaration
            ),
            "result_receipt_sha256": self.result_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class AlphaTrialRegistry:
    project_id: str
    records: tuple[AlphaTrialRecord, ...]
    parent_registry_sha256: str | None
    receipt_sha256: str
    schema: str = ALPHA_TRIAL_REGISTRY_SCHEMA

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "parent_registry_sha256": self.parent_registry_sha256,
            "records": tuple(record.payload() for record in self.records),
        }

    def validate(self) -> None:
        _text("trial registry project ID", self.project_id)
        if self.parent_registry_sha256 is not None:
            _digest("parent trial registry", self.parent_registry_sha256)
        _digest("trial registry receipt", self.receipt_sha256)
        if self.schema != ALPHA_TRIAL_REGISTRY_SCHEMA:
            raise PITAlphaDataError("alpha trial registry schema drifted")
        for record in self.records:
            record.validate()
        if tuple(record.trial_index for record in self.records) != tuple(
            range(len(self.records))
        ):
            raise PITAlphaDataError("trial registry indices are not contiguous")
        trial_ids = tuple(record.trial_id for record in self.records)
        if len(set(trial_ids)) != len(trial_ids):
            raise PITAlphaDataError("trial registry IDs are not unique")
        if self.receipt_sha256 != semantic_sha256(self._payload()):
            raise PITAlphaDataError("alpha trial registry receipt drifted")


def build_alpha_trial_registry(
    *,
    project_id: str,
    records: Sequence[AlphaTrialRecord],
    parent_registry_sha256: str | None = None,
) -> AlphaTrialRegistry:
    payload = {
        "schema": ALPHA_TRIAL_REGISTRY_SCHEMA,
        "project_id": project_id,
        "parent_registry_sha256": parent_registry_sha256,
        "records": tuple(record.payload() for record in records),
    }
    result = AlphaTrialRegistry(
        project_id=project_id,
        records=tuple(records),
        parent_registry_sha256=parent_registry_sha256,
        receipt_sha256=semantic_sha256(payload),
    )
    result.validate()
    return result


__all__ = [
    "ALPHA_DISCOVERY_SETTINGS",
    "ALPHA_EXPERIMENT_SPEC_SCHEMA",
    "ALPHA_TRIAL_REGISTRY_SCHEMA",
    "AlphaExperimentSpec",
    "AlphaDiscoverySetting",
    "AlphaFoldConfig",
    "AlphaModelConfig",
    "AlphaOptimizerConfig",
    "AlphaTrialRecord",
    "AlphaTrialRegistry",
    "TrialChangeKind",
    "build_alpha_trial_registry",
    "validate_alpha_discovery_settings",
]
