"""Embargo-aware adapter for the minimal Massive P0 model tournament."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    read_loaded_massive_source_bytes,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
    MassiveProfitabilityDateTensorV1,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
    train_massive_profitability_fold_v1,
)

MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA = (
    "rl-quant.massive-profitability-tournament-v2"
)
MASSIVE_PROFITABILITY_TOURNAMENT_V2_DATASET = "massive-profitability-tournament-v2"
MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TOURNAMENT_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "gate": "massive-profitability-data-gate-v2",
        "phase": "massive-profitability-phase-plan-v2",
        "settings": MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
        "development_seeds": MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
        "confirmation_seeds": MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        "embargo": "maturation-only-no-entry-training-validation-or-retuning",
        "lockbox_targets": "excluded",
        "publication": "create-only-canonical-json",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityTournamentV2Error(ValueError):
    """The tournament differs from the embargoed P0 experiment."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTournamentV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingFoldV2:
    fold_index: int
    fit_session_dates: tuple[str, ...]
    inner_purge_session_dates: tuple[str, ...]
    inner_validation_session_dates: tuple[str, ...]
    outer_purge_session_dates: tuple[str, ...]
    outer_test_session_dates: tuple[str, ...]
    fit_inventory_sha256: str
    inner_validation_inventory_sha256: str
    outer_test_inventory_sha256: str
    receipt_sha256: str

    def validate(self) -> None:
        # The phase row owns the frozen geometry and receipt. Reconstructing it
        # here prevents the lower trainer from silently selecting another fold.
        body = asdict(self)
        body.pop("receipt_sha256")
        inventories = (
            self.fit_session_dates,
            self.inner_purge_session_dates,
            self.inner_validation_session_dates,
            self.outer_purge_session_dates,
            self.outer_test_session_dates,
        )
        if (
            not 0 <= self.fold_index < 4
            or any(not values or values != tuple(sorted(set(values))) for values in inventories)
            or len(self.outer_test_session_dates) != 126
            or len(self.inner_purge_session_dates) != 63
            or len(self.inner_validation_session_dates) != 126
            or len(self.outer_purge_session_dates) != 63
            or len(self.fit_session_dates) < 756
            or self.fit_inventory_sha256 != semantic_sha256(self.fit_session_dates)
            or self.inner_validation_inventory_sha256
            != semantic_sha256(self.inner_validation_session_dates)
            or self.outer_test_inventory_sha256
            != semantic_sha256(self.outer_test_session_dates)
            or self.receipt_sha256 != semantic_sha256(body)
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "training fold V2 geometry differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTournamentDatasetV2:
    dates: tuple[MassiveProfitabilityDateTensorV1, ...]
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    entry_session_dates: tuple[str, ...]
    maturation_only_session_dates: tuple[str, ...]
    dataset_receipt_sha256: str

    def validate(self) -> None:
        dates = tuple(row.decision_session_date for row in self.dates)
        for row in self.dates:
            row.validate()
        if (
            not dates
            or dates != tuple(sorted(set(dates)))
            or self.entry_session_dates != tuple(
                value for value in dates if value not in set(self.maturation_only_session_dates)
            )
            or self.maturation_only_session_dates
            != tuple(sorted(set(self.maturation_only_session_dates)))
            or set(self.entry_session_dates) & set(self.maturation_only_session_dates)
            or self.dataset_receipt_sha256
            != semantic_sha256(
                {
                    "dates": tuple(row.source_array_sha256 for row in self.dates),
                    "data_gate": self.data_gate_semantic_receipt_sha256,
                    "phase_plan": self.phase_plan_semantic_receipt_sha256,
                    "entry_session_dates": self.entry_session_dates,
                    "maturation_only_session_dates": self.maturation_only_session_dates,
                }
            )
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "tournament dataset V2 differs"
            )
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.dataset_receipt_sha256,
        ):
            _digest("tournament dataset V2", value)

    def by_date(self) -> dict[str, MassiveProfitabilityDateTensorV1]:
        self.validate()
        return {row.decision_session_date: row for row in self.dates}


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTournamentPlanV2:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    fold_receipts: tuple[str, ...]
    embargo_session_dates: tuple[str, ...]
    embargo_inventory_sha256: str
    settings: tuple[str, ...]
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    specification_sha256: str
    implementation_source_sha256: str
    receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    development_training_authorized: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"receipt_sha256", "loaded_source"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA
            or len(self.fold_receipts) != 4
            or len(self.embargo_session_dates) != 63
            or self.embargo_session_dates != tuple(sorted(set(self.embargo_session_dates)))
            or self.embargo_inventory_sha256 != semantic_sha256(self.embargo_session_dates)
            or self.settings != MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1
            or self.development_seeds != MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1
            or self.confirmation_seeds != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or self.specification_sha256 != MASSIVE_PROFITABILITY_TOURNAMENT_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256
            or not self.development_training_authorized
            or not self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "tournament plan V2 identity or authorization differs"
            )
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            *self.fold_receipts,
            self.embargo_inventory_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.receipt_sha256,
        ):
            _digest("tournament plan V2", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id != MASSIVE_PROFITABILITY_TOURNAMENT_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.data_gate_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "tournament plan V2 committed source differs"
            )


def parse_massive_profitability_tournament_plan_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityTournamentPlanV2:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityTournamentV2Error(
            "tournament plan V2 source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityTournamentV2Error(
            "tournament plan V2 source is not canonical JSON"
        )
    for name in (
        "fold_receipts",
        "embargo_session_dates",
        "settings",
        "development_seeds",
        "confirmation_seeds",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityTournamentPlanV2(**payload, loaded_source=loaded_source)
    result.validate()
    expected = result.unsigned() | {"receipt_sha256": result.receipt_sha256}
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityTournamentV2Error(
            "tournament plan V2 canonical bytes differ"
        )
    return result


def train_massive_profitability_fold_v2(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    seed: int,
    config: MassiveProfitabilityTrainingConfigV1 | None = None,
    device: str | torch.device = "cpu",
) -> MassiveProfitabilityTrainedRunV1:
    """Reuse the frozen model math under V2 plan, fold, and embargo identities."""

    if set(fold.outer_test_session_dates) & set(dataset.maturation_only_session_dates):
        raise MassiveProfitabilityTournamentV2Error(
            "embargo sessions cannot become tournament entries"
        )
    return train_massive_profitability_fold_v1(
        dataset=dataset,  # type: ignore[arg-type]
        tournament_plan=tournament_plan,  # type: ignore[arg-type]
        fold=fold,  # type: ignore[arg-type]
        setting_id=setting_id,
        seed=seed,
        config=config,
        device=device,
    )


__all__ = [
    "MASSIVE_PROFITABILITY_TOURNAMENT_V2_DATASET",
    "MASSIVE_PROFITABILITY_TOURNAMENT_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_TOURNAMENT_V2_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_TOURNAMENT_V2_SPEC_SHA256",
    "MassiveProfitabilityTournamentDatasetV2",
    "MassiveProfitabilityTournamentPlanV2",
    "MassiveProfitabilityTournamentV2Error",
    "MassiveProfitabilityTrainingFoldV2",
    "parse_massive_profitability_tournament_plan_v2",
    "train_massive_profitability_fold_v2",
]
