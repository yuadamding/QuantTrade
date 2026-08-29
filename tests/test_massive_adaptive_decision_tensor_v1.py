from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from rl_quant.alpha.targets import OriginExposurePanel
from rl_quant.features.massive_adaptive_alpha_targets_v1 import (
    MassiveAdaptiveEconomicPathV1,
    build_massive_adaptive_alpha_targets_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1Error,
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1,
    MassiveAdaptiveOriginAuthorityV1,
    MassiveAdaptiveOriginExposureRowV1,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    MassiveProfitabilityOriginFeatureRowV2,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    MassiveAdaptiveAlphaTermStructureModelV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.training.adaptive_alpha_supervised_v1 import (
    massive_adaptive_alpha_supervised_loss_v1,
)
from rl_quant.workflows.adaptive_alpha_training_inputs_v2 import (
    build_massive_adaptive_alpha_training_batch_v2,
)
from tests.test_massive_profitability_v6_vertical_slice import (
    _feature_and_target,
)


_SECURITIES = tuple(f"SEC-{index:02d}" for index in range(8))


def _feature(day_index: int):
    date = f"2024-09-{day_index + 2:02d}"
    history = tuple(f"2024-06-{index + 1:02d}" for index in range(30)) + tuple(
        f"2024-07-{index + 1:02d}" for index in range(31)
    ) + tuple(f"2024-08-{index + 1:02d}" for index in range(3))
    feature, _target = _feature_and_target(
        decision_session_date=date,
        source_session_date=history[-1],
        input_session_dates=history,
        date_index=day_index,
    )
    rows = list(feature.rows)
    template = feature.rows[0]
    for asset_index, security_id in enumerate(_SECURITIES[5:], start=5):
        bars = list(template.bars_values)
        bars[0] += float(asset_index)
        body = template.unsigned() | {
            "security_id": security_id,
            "decision_membership_rank": asset_index + 1,
            "bars_values": tuple(bars),
            "source_panel_row_receipt_sha256": semantic_sha256(
                (date, security_id, "panel")
            ),
            "feature_accounting_security_inventory_sha256": semantic_sha256(
                (date, security_id, "feature-accounting")
            ),
            "tape_population_row_receipt_sha256": semantic_sha256(
                (date, security_id, "tape-population")
            ),
        }
        rows.append(
            MassiveProfitabilityOriginFeatureRowV2(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    changed = replace(
        feature,
        rows=tuple(rows),
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256=semantic_sha256((date, "expanded-feature-audit")),
    )
    result = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    result.validate()
    return result


def _origin(feature, *, action_ids: tuple[str, ...] = _SECURITIES[:7]):
    rows = []
    for rank, security_id in enumerate(action_ids, start=1):
        body = {
            "security_id": security_id,
            "universe_rank": rank,
            "exposures": (
                1.0,
                *(1.0 if rank == column else 0.0 for column in range(1, 6)),
            ),
            "regression_weight": 1.0,
            "qualified": True,
            "membership_row_receipt_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "membership")
            ),
            "identity_row_receipt_sha256": semantic_sha256(
                (security_id, "identity")
            ),
            "daily_row_inventory_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "daily")
            ),
            "economic_history_receipt_sha256": semantic_sha256(
                (feature.decision_session_date, security_id, "economic")
            ),
        }
        rows.append(
            MassiveAdaptiveOriginExposureRowV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    decision_at_ms = int(feature.decision_session_date[-2:]) * 1_000_000
    panel = OriginExposurePanel(
        origin_at_ms=decision_at_ms,
        available_at_ms=decision_at_ms,
        asset_ids=action_ids,
        exposure_names=MASSIVE_ADAPTIVE_ORIGIN_EXPOSURES_V1,
        exposures=tuple(row.exposures for row in rows),
        regression_weights=tuple(row.regression_weight for row in rows),
        qualified_asset_mask=tuple(row.qualified for row in rows),
        source_receipt_sha256=semantic_sha256(
            (feature.decision_session_date, "panel")
        ),
    )
    provisional = MassiveAdaptiveOriginAuthorityV1(
        decision_session_date=feature.decision_session_date,
        decision_at_ms=decision_at_ms,
        membership_effective_at_ms=decision_at_ms - 2,
        membership_available_at_ms=decision_at_ms - 1,
        history_session_dates=feature.input_session_dates,
        security_ids=action_ids,
        universe_ranks=tuple(range(1, len(action_ids) + 1)),
        rows=tuple(rows),
        exposure_panel=panel,
        decision_clock_receipt_sha256=semantic_sha256(
            (feature.decision_session_date, "clock")
        ),
        session_authority_receipt_sha256=semantic_sha256("sessions"),
        action_universe_rule_receipt_sha256=(
            MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.action_universe_rule.receipt_sha256
        ),
        membership_group_inventory_sha256=semantic_sha256(
            (feature.decision_session_date, "membership-group")
        ),
        selected_identity_inventory_sha256=semantic_sha256(
            (feature.decision_session_date, "identity-inventory")
        ),
        selected_daily_session_inventory_sha256=semantic_sha256(
            (feature.decision_session_date, "sessions")
        ),
        selected_daily_row_inventory_sha256=semantic_sha256(
            (feature.decision_session_date, "daily-rows")
        ),
        scoped_economic_event_inventory_sha256=semantic_sha256(
            (feature.decision_session_date, "events")
        ),
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        protocol_receipt_sha256=MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        specification_sha256=MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_ADAPTIVE_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
        semantic_receipt_sha256="0" * 64,
        source_paths_replayed=True,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _targets(origin):
    paths = []
    for asset_index, security_id in enumerate(origin.security_ids):
        alpha = 0.0005 * (asset_index + 1)
        values = tuple(
            100.0 * (1.0 + alpha * offset) for offset in range(127)
        )
        economic_at = tuple(
            origin.decision_at_ms + 1_000 + offset for offset in range(127)
        )
        body = {
            "schema": "rl-quant.massive-adaptive-economic-path-v1",
            "security_id": security_id,
            "decision_at_ms": origin.decision_at_ms,
            "fill_at_ms": economic_at[0],
            "economic_at_ms": economic_at,
            "available_at_ms": tuple(value + 1 for value in economic_at),
            "values": values,
            "valid": (True,) * 127,
            "terminal": (False,) * 127,
            "mark_kinds": ("market",) * 127,
            "mark_receipts": tuple(
                semantic_sha256((origin.decision_session_date, security_id, offset))
                for offset in range(127)
            ),
            "unresolved_terminal_fallback_session_offset": None,
            "conservative_total_loss_fallback": False,
            "source_economic_path_receipt_sha256": semantic_sha256(
                (origin.decision_session_date, security_id, "path")
            ),
        }
        paths.append(
            MassiveAdaptiveEconomicPathV1(
                **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
            )
        )
    return build_massive_adaptive_alpha_targets_v1(
        decision_session_date=origin.decision_session_date,
        built_at_ms=paths[0].available_at_ms[-1] + 1,
        paths=tuple(paths),
        exposure_panel=origin.exposure_panel,
        origin_receipt_sha256=origin.semantic_receipt_sha256,
        economic_accounting_receipt_sha256=semantic_sha256(
            (origin.decision_session_date, "accounting")
        ),
        fill_source_receipt_sha256=semantic_sha256(
            (origin.decision_session_date, "fills")
        ),
        terminal_authority_receipt_sha256=semantic_sha256(
            (origin.decision_session_date, "terminal")
        ),
        economic_coverage_receipt_sha256=semantic_sha256(
            (origin.decision_session_date, "coverage")
        ),
    )


def _changed_feature(feature):
    first = feature.rows[0]
    values = list(first.bars_values)
    values[0] += 10.0
    body = first.unsigned() | {"bars_values": tuple(values)}
    changed_row = MassiveProfitabilityOriginFeatureRowV2(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    rows = (changed_row, *feature.rows[1:])
    changed = replace(
        feature,
        rows=rows,
        row_inventory_sha256=semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256=semantic_sha256(
            (feature.decision_session_date, "changed-audit")
        ),
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    changed.validate()
    return changed


def test_create_only_tensor_reloads_nonauthorizing_and_replays_exactly(tmp_path) -> None:
    features = tuple(_feature(index) for index in range(2))
    origins = tuple(_origin(feature) for feature in features)

    committed = materialize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        artifact_id="canary",
        features=features,
        action_origins=origins,
        committed_at_ms=20_000,
    )

    assert committed.runtime_source_replayed
    assert committed.model_input_authorized
    assert not committed.development_training_authorized
    runtime = committed.runtime_tensor
    assert runtime is not None
    assert runtime.bars_values.shape == (2, 8, 19)
    assert runtime.tape_values.shape == (2, 8, 15)
    assert runtime.context_membership.all()
    assert runtime.action_mask[:, :7].all()
    assert not runtime.action_mask[:, 7:].any()
    assert torch.equal(
        runtime.bars_values[0, 0],
        torch.tensor(features[0].rows[0].bars_values, dtype=torch.float32),
    )
    model = MassiveAdaptiveAlphaTermStructureModelV1(
        replace(
            MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
            token_dimension=16,
            fast_window_sessions=2,
            maximum_context_sessions=2,
            maximum_intraday_intervals=4,
            market_latent_count=4,
            attention_heads=4,
            dropout_probability=0.0,
        )
    ).eval()
    output = model.forward_sequence(
        bars_values=runtime.bars_values.unsqueeze(0),
        bars_valid=runtime.bars_valid.unsqueeze(0),
        tape_values=runtime.tape_values.unsqueeze(0),
        tape_valid=runtime.tape_valid.unsqueeze(0),
        source_staleness=runtime.source_staleness.unsqueeze(0),
        context_membership=runtime.context_membership.unsqueeze(0),
        action_mask=runtime.action_mask.unsqueeze(0),
    )
    assert output.executable_score.shape == (1, 2, 8)
    assert torch.equal(output.valid, runtime.action_mask.unsqueeze(0))
    targets = tuple(_targets(origin) for origin in origins)
    batch = build_massive_adaptive_alpha_training_batch_v2(
        output=output,
        decision_tensor=committed,
        target_artifacts=(targets,),
        tensor_session_indices=torch.tensor(((0, 1),), dtype=torch.long),
        origin_indices=torch.tensor(((0, 1),), dtype=torch.long),
        split_start_inclusive=0,
        split_stop_exclusive=200,
        split_role="training",
        split_plan_receipt_sha256=semantic_sha256("split"),
    )
    assert not batch.target_valid[:, :, -1].any()
    assert torch.equal(batch.action_mask, output.valid)
    assert torch.allclose(batch.benchmark_weights.sum(dim=2), torch.ones(1, 2))
    loss = massive_adaptive_alpha_supervised_loss_v1(batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.residual_head.projection.weight.grad is not None

    generic = parse_massive_adaptive_decision_tensor_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )
    assert generic.runtime_tensor is None
    assert not generic.runtime_source_replayed
    assert not generic.model_input_authorized

    promoted = authorize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        tensor=generic,
        features=features,
        action_origins=origins,
    )
    assert promoted.source_array_receipts == committed.source_array_receipts
    assert promoted.tensor_inventory_sha256 == committed.tensor_inventory_sha256


def test_feature_corruption_and_missing_action_support_fail_replay(tmp_path) -> None:
    features = tuple(_feature(index) for index in range(2))
    origins = tuple(_origin(feature) for feature in features)
    committed = materialize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        artifact_id="corruption",
        features=features,
        action_origins=origins,
        committed_at_ms=30_000,
    )
    generic = parse_massive_adaptive_decision_tensor_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )

    changed = (_changed_feature(features[0]), features[1])
    with pytest.raises(
        MassiveAdaptiveDecisionTensorV1Error,
        match="does not replay",
    ):
        authorize_massive_adaptive_decision_tensor_v1(
            root=tmp_path,
            tensor=generic,
            features=changed,
            action_origins=origins,
        )

    unsupported = (
        _origin(features[0], action_ids=(*_SECURITIES[:7], "SEC-MISSING")),
        origins[1],
    )
    with pytest.raises(
        MassiveAdaptiveDecisionTensorV1Error,
        match="absent from the context",
    ):
        authorize_massive_adaptive_decision_tensor_v1(
            root=tmp_path,
            tensor=generic,
            features=features,
            action_origins=unsupported,
        )


def test_adaptive_tensor_has_no_duration_or_downstream_authority_fields() -> None:
    names = {
        field.name
        for field in fields(
            __import__(
                "rl_quant.features.massive_adaptive_decision_tensor_v1",
                fromlist=["MassiveAdaptiveDecisionTensorV1"],
            ).MassiveAdaptiveDecisionTensorV1
        )
    }
    assert "position_age" not in names
    assert "preferred_holding_sessions" not in names
    assert "reinforcement_learning_authorized" in names
