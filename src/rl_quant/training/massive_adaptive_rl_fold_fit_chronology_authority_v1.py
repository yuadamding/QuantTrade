"""Fit-only chronology commitment derived from Manifest V3 and source replay.

Policy-selection inference is intentionally absent.  This authority commits
the registered fit, inner-validation, and outer date partitions without
opening validation tensors or economic outcomes, and authorizes only fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)


MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fold-fit-chronology-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source": "manifest-v3-runtime-sources-v1-and-training-forecast-v2",
        "fit": "exact-126-times-fold-index-plus-one-tail",
        "validation": "split-plan-date-commitment-only",
        "outer": "split-plan-date-commitment-only",
        "policy_selection_access": False,
        "outer_access": False,
        "caller_dates": False,
        "profitability_reporting": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error(ValueError):
    """The package-derived fold-fit chronology or its witnesses differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldFitChronologyAuthorityV1:
    experiment_id: str
    fold_index: int
    manifest_v3_receipt_sha256: str
    runtime_sources_receipt_sha256: str
    runtime_graph_witness_receipt_sha256: str
    training_forecast_authority_receipt_sha256: str
    split_plan_receipt_sha256: str
    rl_fit_origin_dates: tuple[str, ...]
    rl_validation_origin_dates: tuple[str, ...]
    outer_origin_dates: tuple[str, ...]
    rl_fit_origin_inventory_sha256: str
    rl_validation_origin_inventory_sha256: str
    outer_origin_inventory_sha256: str
    source_data_qualified: bool
    runtime_chronology_replayed: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    development_policy_selection_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SCHEMA
    _manifest: MassiveAdaptiveRLExperimentManifestV3 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _training_forecast_authority: (
        MassiveAdaptiveRLTrainingForecastAuthorityV2 | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "fold_index": self.fold_index,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "runtime_sources_receipt_sha256": self.runtime_sources_receipt_sha256,
            "runtime_graph_witness_receipt_sha256": (
                self.runtime_graph_witness_receipt_sha256
            ),
            "training_forecast_authority_receipt_sha256": (
                self.training_forecast_authority_receipt_sha256
            ),
            "split_plan_receipt_sha256": self.split_plan_receipt_sha256,
            "rl_fit_origin_dates": self.rl_fit_origin_dates,
            "rl_validation_origin_dates": self.rl_validation_origin_dates,
            "outer_origin_dates": self.outer_origin_dates,
            "rl_fit_origin_inventory_sha256": (
                self.rl_fit_origin_inventory_sha256
            ),
            "rl_validation_origin_inventory_sha256": (
                self.rl_validation_origin_inventory_sha256
            ),
            "outer_origin_inventory_sha256": self.outer_origin_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "runtime_chronology_replayed": self.runtime_chronology_replayed,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        witnesses = (
            self._manifest,
            self._runtime_sources,
            self._training_forecast_authority,
        )
        runtime_present = all(value is not None for value in witnesses)
        partial_runtime = any(value is not None for value in witnesses) and not (
            runtime_present
        )
        if runtime_present:
            assert self._manifest is not None
            assert self._runtime_sources is not None
            assert self._training_forecast_authority is not None
            self._manifest.validate()
            self._runtime_sources.validate()
            self._training_forecast_authority.validate()
            manifest = self._manifest
            runtime_sources = self._runtime_sources
            training = self._training_forecast_authority
            schedule = manifest.base_manifest.schedule(self.fold_index)
            split_fold = runtime_sources.split_plan.outer_folds[self.fold_index]
            expected_fit = split_fold.fit_session_dates[
                -schedule.rl_fit_session_count :
            ]
            expected_validation = split_fold.inner_validation_session_dates
            expected_outer = split_fold.outer_test_session_dates
            runtime_receipt = runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
        else:
            manifest = None
            runtime_sources = None
            training = None
            expected_fit = ()
            expected_validation = ()
            expected_outer = ()
            runtime_receipt = None
        fit = set(self.rl_fit_origin_dates)
        validation = set(self.rl_validation_origin_dates)
        outer = set(self.outer_origin_dates)
        expected_qualified = bool(
            runtime_present
            and manifest is not None
            and runtime_sources is not None
            and training is not None
            and runtime_sources.source_data_qualified
            and training.source_data_qualified
            and training.reinforcement_learning_authorized
            and runtime_sources.split_plan.candidate_source_data_qualified
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SCHEMA
            or partial_runtime
            or not runtime_present
            or not self.experiment_id
            or self.fold_index not in range(4)
            or not fit
            or not validation
            or not outer
            or fit & validation
            or fit & outer
            or validation & outer
            or self.rl_fit_origin_dates != tuple(sorted(fit))
            or self.rl_validation_origin_dates != tuple(sorted(validation))
            or self.outer_origin_dates != tuple(sorted(outer))
            or self.rl_fit_origin_dates[-1] >= self.rl_validation_origin_dates[0]
            or self.rl_validation_origin_dates[-1] >= self.outer_origin_dates[0]
            or self.rl_fit_origin_dates != expected_fit
            or self.rl_validation_origin_dates != expected_validation
            or self.outer_origin_dates != expected_outer
            or self.rl_fit_origin_inventory_sha256
            != semantic_sha256(self.rl_fit_origin_dates)
            or self.rl_validation_origin_inventory_sha256
            != semantic_sha256(self.rl_validation_origin_dates)
            or self.outer_origin_inventory_sha256
            != semantic_sha256(self.outer_origin_dates)
            or manifest is None
            or runtime_sources is None
            or training is None
            or self.experiment_id != manifest.experiment_id
            or self.experiment_id != runtime_sources.experiment_id
            or self.manifest_v3_receipt_sha256
            != manifest.semantic_receipt_sha256
            or self.manifest_v3_receipt_sha256
            != runtime_sources.manifest_v3_receipt_sha256
            or self.runtime_sources_receipt_sha256
            != runtime_sources.semantic_receipt_sha256
            or self.runtime_graph_witness_receipt_sha256 != runtime_receipt
            or self.training_forecast_authority_receipt_sha256
            != training.semantic_receipt_sha256
            or training.outer_fold_index != self.fold_index
            or self.split_plan_receipt_sha256
            != runtime_sources.split_plan.semantic_receipt_sha256
            or training.split_plan_receipt_sha256
            != self.split_plan_receipt_sha256
            or self.runtime_chronology_replayed != runtime_present
            or self.source_data_qualified != expected_qualified
            or not self.source_data_qualified
            or self.development_rl_training_authorized != expected_qualified
            or self.development_policy_selection_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error(
                "adaptive RL fold-fit chronology authority differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.runtime_sources_receipt_sha256,
            self.runtime_graph_witness_receipt_sha256,
            self.training_forecast_authority_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.rl_fit_origin_inventory_sha256,
            self.rl_validation_origin_inventory_sha256,
            self.outer_origin_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fold-fit chronology authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_fold_fit_chronology_authority_v1(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
    training_forecast_authority: MassiveAdaptiveRLTrainingForecastAuthorityV2,
) -> MassiveAdaptiveRLFoldFitChronologyAuthorityV1:
    """Commit all date roles while authorizing only the causal fit prefix."""

    manifest.validate()
    runtime_sources.validate()
    training_forecast_authority.validate()
    fold_index = training_forecast_authority.outer_fold_index
    if (
        manifest.experiment_id != runtime_sources.experiment_id
        or manifest.semantic_receipt_sha256
        != runtime_sources.manifest_v3_receipt_sha256
        or fold_index not in manifest.base_manifest.fold_indices
        or training_forecast_authority.split_plan_receipt_sha256
        != runtime_sources.split_plan.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error(
            "adaptive RL fold-fit chronology inputs differ"
        )
    runtime_receipt = runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
    if runtime_receipt is None:
        raise MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error(
            "adaptive RL fold-fit runtime graph witness is absent"
        )
    split_fold = runtime_sources.split_plan.outer_folds[fold_index]
    schedule = manifest.base_manifest.schedule(fold_index)
    fit_dates = split_fold.fit_session_dates[-schedule.rl_fit_session_count :]
    validation_dates = split_fold.inner_validation_session_dates
    outer_dates = split_fold.outer_test_session_dates
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "fold_index": fold_index,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "runtime_sources_receipt_sha256": runtime_sources.semantic_receipt_sha256,
        "runtime_graph_witness_receipt_sha256": runtime_receipt,
        "training_forecast_authority_receipt_sha256": (
            training_forecast_authority.semantic_receipt_sha256
        ),
        "split_plan_receipt_sha256": runtime_sources.split_plan.semantic_receipt_sha256,
        "rl_fit_origin_dates": fit_dates,
        "rl_validation_origin_dates": validation_dates,
        "outer_origin_dates": outer_dates,
        "rl_fit_origin_inventory_sha256": semantic_sha256(fit_dates),
        "rl_validation_origin_inventory_sha256": semantic_sha256(validation_dates),
        "outer_origin_inventory_sha256": semantic_sha256(outer_dates),
        "source_data_qualified": True,
        "runtime_chronology_replayed": True,
        "development_policy_selection_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLFoldFitChronologyAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_rl_training_authorized=True,
        _manifest=manifest,
        _runtime_sources=runtime_sources,
        _training_forecast_authority=training_forecast_authority,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FOLD_FIT_CHRONOLOGY_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldFitChronologyAuthorityV1",
    "MassiveAdaptiveRLFoldFitChronologyAuthorityV1Error",
    "build_massive_adaptive_rl_fold_fit_chronology_authority_v1",
]
