from __future__ import annotations

import pytest

from rl_quant.evaluation.massive_adaptive_outer_evidence_v1 import (
    MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA,
    MassiveAdaptiveOuterCostFoldV1,
    MassiveAdaptiveOuterEvidenceV1Error,
    build_massive_adaptive_outer_evidence_v1,
)
from rl_quant.evaluation.massive_adaptive_outer_evidence_authority_v1 import (
    authorize_massive_adaptive_outer_evidence_authority_v1,
    materialize_massive_adaptive_outer_evidence_authority_v1,
    parse_massive_adaptive_outer_evidence_authority_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)


def _fold(index: int) -> MassiveAdaptiveOuterCostFoldV1:
    body = {
        "schema": MASSIVE_ADAPTIVE_OUTER_COST_FOLD_V1_SCHEMA,
        "fold_index": index,
        "selected_checkpoint_receipt_sha256": semantic_sha256(
            ("checkpoint", index)
        ),
        "checkpoint_selection_authority_receipt_sha256": semantic_sha256(
            ("selection", index)
        ),
        "low_cost_trace_receipt_sha256": semantic_sha256(("low", index)),
        "primary_trace_receipt_sha256": semantic_sha256(("primary", index)),
        "high_cost_trace_receipt_sha256": semantic_sha256(("high", index)),
        "low_cost_authority_receipt_sha256": semantic_sha256(
            ("low-authority", index)
        ),
        "primary_authority_receipt_sha256": semantic_sha256(
            ("primary-authority", index)
        ),
        "high_cost_authority_receipt_sha256": semantic_sha256(
            ("high-authority", index)
        ),
        "decision_target_inventory_sha256": semantic_sha256(("targets", index)),
        "primary_net_returns": (0.001,) * 126,
        "primary_active_log_returns": (0.0005,) * 126,
        "low_cost_terminal_return": 0.20,
        "primary_terminal_return": 0.10,
        "high_cost_terminal_return": 0.01,
        "source_data_qualified": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveOuterCostFoldV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def test_outer_evidence_derives_deterministic_fold_cluster_bounds(tmp_path) -> None:
    folds = tuple(_fold(index) for index in range(4))
    first = build_massive_adaptive_outer_evidence_v1(folds)
    second = build_massive_adaptive_outer_evidence_v1(tuple(reversed(folds)))

    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.primary_net_return_lcb95 == pytest.approx(0.001)
    assert first.primary_active_log_return_lcb95 == pytest.approx(0.0005)
    assert not first.failed_gate_names
    assert first.positive_primary_fold_count == 4
    assert first.cost_ladder_monotone
    assert not first.source_data_qualified
    assert not first.outer_development_conclusion_authorized
    assert not first.profitability_reporting_authorized
    assert not first.lockbox_access_authorized
    assert not first.reinforcement_learning_authorized
    authority = materialize_massive_adaptive_outer_evidence_authority_v1(
        root=tmp_path,
        artifact_id="synthetic-outer-evidence",
        folds=folds,
        committed_at_ms=1,
    )
    assert authority.runtime_evidence_replayed
    assert authority.runtime_folds == folds
    assert not authority.outer_development_conclusion_authorized
    generic = parse_massive_adaptive_outer_evidence_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert not generic.runtime_evidence_replayed
    assert generic.runtime_folds is None
    replayed = authorize_massive_adaptive_outer_evidence_authority_v1(
        root=tmp_path,
        authority=generic,
        folds=tuple(reversed(folds)),
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256


def test_outer_evidence_requires_exact_four_fold_inventory() -> None:
    with pytest.raises(
        MassiveAdaptiveOuterEvidenceV1Error,
        match="exactly folds zero through three",
    ):
        build_massive_adaptive_outer_evidence_v1(tuple(_fold(index) for index in range(3)))
