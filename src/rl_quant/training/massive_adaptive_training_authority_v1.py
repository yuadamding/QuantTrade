"""Root-bound promotion authority for adaptive supervised training.

Decision tensors, decision roots, economic target replays, split geometry, and
window rows remain nonauthorizing on their own.  This authority reconciles the
exact inventories for one fold/role.  The trainer rebuilds it from the roots
before every run instead of trusting a caller-set authorization flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.features.massive_adaptive_target_archive_v1 import (
    MassiveAdaptiveTargetArchiveV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)


MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-training-authority-v1"
)
MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "model_inputs": "runtime-replayed-decision-tensor-v1",
        "decision_roots": (
            "separate-full-chronology-and-target-bearing-origin-inventories"
        ),
        "targets": (
            "promoted-target-archive-v1-over-eligible-window-origins-only"
        ),
        "split": "frozen-126-session-split-plan-v1",
        "windows": "package-derived-window-plan-v1",
        "promotion": "trainer-rebuilds-authority-from-live-roots",
        "outer_or_lockbox_access": False,
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveTrainingAuthorityV1Error(ValueError):
    """Adaptive training roots or authorization inventory differ."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveTrainingAuthorityV1:
    fold_index: int
    split_role: str
    origin_session_dates: tuple[str, ...]
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    target_archive_receipt_sha256: str
    target_root_inventory_sha256: str
    source_target_inventory_sha256: str
    target_experiment_inventory_sha256: str
    split_plan_receipt_sha256: str
    window_plan_receipt_sha256: str
    source_inventory_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    runtime_roots_replayed: bool
    source_data_qualified: bool
    development_training_authorized: bool
    outer_test_accessed: bool
    lockbox_accessed: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SCHEMA
            or self.split_role not in {"training", "inner_validation"}
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SOURCE_SHA256
            or not self.runtime_roots_replayed
            or not isinstance(self.source_data_qualified, bool)
            or self.development_training_authorized
            != (self.source_data_qualified and self.split_role == "training")
            or self.outer_test_accessed
            or self.lockbox_accessed
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveTrainingAuthorityV1Error(
                "adaptive training authority identity or qualification differs"
            )
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_training_authority_v1(
    *,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    target_archive: MassiveAdaptiveTargetArchiveV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveTrainingAuthorityV1:
    """Reconcile every root required for one fold/role training inventory."""

    decision_tensor.validate()
    target_archive.validate()
    split_plan.validate()
    window_plan.validate()
    if decision_tensor.runtime_tensor is None or not decision_tensor.runtime_source_replayed:
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive decision tensor has not been package replayed"
        )
    ordered_roots = tuple(
        sorted(decision_roots, key=lambda row: row.decision_session_date)
    )
    if any(
        not isinstance(row, MassiveAdaptiveDecisionRootV1)
        for row in ordered_roots
    ):
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive training requires decision roots and a target archive"
        )
    for decision_root in ordered_roots:
        decision_root.validate()
    if (
        not target_archive.runtime_roots_replayed
        or target_archive.runtime_target_roots is None
        or target_archive.runtime_source_targets is None
    ):
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive target archive has not been package replayed"
        )
    ordered_target_roots = target_archive.runtime_target_roots
    ordered_targets = target_archive.runtime_source_targets
    root_by_date = {row.decision_session_date: row for row in ordered_roots}
    target_by_date = {row.decision_session_date: row for row in ordered_targets}
    expected_dates = tuple(row.origin_session_date for row in window_plan.rows)
    if not set(expected_dates) <= set(root_by_date):
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive window origin is absent from the full decision roots"
        )
    origin_roots = tuple(root_by_date[date] for date in expected_dates)
    origin_root_receipts = tuple(
        row.semantic_receipt_sha256 for row in origin_roots
    )
    if (
        len(root_by_date) != len(ordered_roots)
        or len(target_by_date) != len(ordered_targets)
        or tuple(root_by_date) != decision_tensor.decision_session_dates
        or tuple(row.feature_semantic_receipt_sha256 for row in ordered_roots)
        != decision_tensor.feature_semantic_receipts
        or tuple(row.action_origin_receipt_sha256 for row in ordered_roots)
        != decision_tensor.action_origin_receipts
        or target_archive.origin_decision_root_receipts != origin_root_receipts
        or target_archive.decision_session_dates != expected_dates
        or target_archive.source_target_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_targets)
        or target_archive.target_root_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_target_roots)
        or tuple(target_by_date) != expected_dates
        or tuple(
            row.decision_session_date for row in ordered_target_roots
        )
        != expected_dates
        or window_plan.decision_tensor_receipt_sha256
        != decision_tensor.semantic_receipt_sha256
        or window_plan.full_decision_root_inventory_sha256
        != semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered_roots)
        )
        or window_plan.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_root_receipts)
        or window_plan.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
        or any(
            target.security_ids != root_by_date[date].action_security_ids
            or target.origin_authority_receipt_sha256
            != root_by_date[date].action_origin_receipt_sha256
            or target.decision_clock_receipt_sha256
            != root_by_date[date].decision_clock_receipt_sha256
            or target.session_authority_receipt_sha256
            != root_by_date[date].session_authority_receipt_sha256
            or target_by_date[date].semantic_receipt_sha256
            != next(
                row.source_target_receipt_sha256
                for row in ordered_target_roots
                if row.decision_session_date == date
            )
            or not target.source_paths_replayed
            for date, target in target_by_date.items()
        )
    ):
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive training root, target, split, or window inventories differ"
        )
    full_root_inventory = semantic_sha256(
        tuple(row.semantic_receipt_sha256 for row in ordered_roots)
    )
    origin_root_inventory = semantic_sha256(origin_root_receipts)
    target_inventory = semantic_sha256(
        tuple(row.semantic_receipt_sha256 for row in ordered_targets)
    )
    source_inventory = semantic_sha256(
        {
            "decision_tensor": decision_tensor.semantic_receipt_sha256,
            "full_decision_roots": full_root_inventory,
            "origin_decision_roots": origin_root_inventory,
            "target_archive": target_archive.semantic_receipt_sha256,
            "target_roots": target_archive.target_root_inventory_sha256,
            "source_targets": target_inventory,
            "target_experiments": target_archive.experiment_inventory_sha256,
            "split_plan": split_plan.semantic_receipt_sha256,
            "window_plan": window_plan.semantic_receipt_sha256,
        }
    )
    qualified = (
        decision_tensor.committed_source_data_qualified
        and split_plan.candidate_source_data_qualified
        and target_archive.committed_source_data_qualified
        and target_archive.development_training_authorized
        and all(row.source_data_qualified for row in ordered_roots)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SCHEMA,
        "fold_index": window_plan.fold_index,
        "split_role": window_plan.split_role,
        "origin_session_dates": expected_dates,
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": full_root_inventory,
        "origin_decision_root_inventory_sha256": origin_root_inventory,
        "target_archive_receipt_sha256": target_archive.semantic_receipt_sha256,
        "target_root_inventory_sha256": target_archive.target_root_inventory_sha256,
        "source_target_inventory_sha256": target_inventory,
        "target_experiment_inventory_sha256": (
            target_archive.experiment_inventory_sha256
        ),
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "window_plan_receipt_sha256": window_plan.semantic_receipt_sha256,
        "source_inventory_sha256": source_inventory,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SOURCE_SHA256,
        "runtime_roots_replayed": True,
        "source_data_qualified": qualified,
        "development_training_authorized": (
            qualified and window_plan.split_role == "training"
        ),
        "outer_test_accessed": False,
        "lockbox_accessed": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveTrainingAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_TRAINING_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveTrainingAuthorityV1",
    "MassiveAdaptiveTrainingAuthorityV1Error",
    "build_massive_adaptive_training_authority_v1",
]
