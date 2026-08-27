from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveProfitabilityResidualInputRowV1,
    build_massive_profitability_residual_scores_v1,
    evaluate_massive_profitability_fixed_tranches_v1,
    select_massive_profitability_tranches_v1,
)
from rl_quant.evaluation.massive_profitability_predictions_v3 import (
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET,
    MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256,
    MassiveProfitabilityPredictionsV3Error,
    authorize_massive_profitability_outer_predictions_v3,
    parse_massive_profitability_outer_predictions_v3,
    publish_massive_profitability_outer_predictions_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    authorize_massive_profitability_tournament_dataset_v3,
    materialize_massive_profitability_tournament_dataset_v3,
    parse_massive_profitability_tournament_dataset_v3,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    adapt_massive_profitability_training_fold_v2,
    materialize_massive_profitability_tournament_plan_v2,
)
from rl_quant.evaluation.massive_profitability_training_v4 import (
    authorize_massive_profitability_checkpoint_v3_from_roots,
    train_and_publish_massive_profitability_fold_v4,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256,
    MassiveProfitabilityDataGateV2,
    MassiveProfitabilityDateSupportGateV2,
    _MASSIVE_PROFITABILITY_DATA_GATE_V2_INPUT_SCHEMAS,
)
from rl_quant.features.massive_profitability_feature_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_fill_source_authority_v2 import (
    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_origin_features_v2 import (
    BARS_MIN_V2_FIELDS,
    TAPE_MIN_V2_FIELDS,
    MassiveProfitabilityOriginFeatureRowV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SPEC_SHA256,
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    materialize_massive_profitability_phase_plan_v2,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.features.massive_profitability_targets_v1 import (
    MassiveProfitabilityTargetRowV1,
    MassiveProfitabilityTargetSpecV1,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MASSIVE_PROFITABILITY_TARGETS_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TARGETS_V2_SPEC_SHA256,
    MassiveProfitabilityTargetsV2,
)
from rl_quant.features.massive_profitability_terminal_coverage_authority_v1 import (
    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MassiveProfitabilityTrainingConfigV1,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    bind_massive_profitability_trained_run_v3,
    load_massive_profitability_prediction_checkpoint_v2_from_v3,
    parse_massive_profitability_model_checkpoint_v3,
    publish_massive_profitability_model_checkpoint_v3,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    bind_massive_profitability_trained_run_v2,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from tests.test_massive_profitability_archive_freeze_v1 import _freeze, _inputs

_SECURITY_IDS = tuple(f"SEC-{index:02d}" for index in range(5))
_DIGESTS = tuple(semantic_sha256(("v6-vertical-slice", index)) for index in range(16))


def _feature_and_target(
    *,
    decision_session_date: str,
    source_session_date: str,
    input_session_dates: tuple[str, ...],
    date_index: int,
) -> tuple[MassiveProfitabilityOriginFeaturesV3, MassiveProfitabilityTargetsV2]:
    feature_rows = []
    target_rows = []
    for asset_index, security_id in enumerate(_SECURITY_IDS):
        bar_signal = (((asset_index * 7 + date_index * 3) % 17) - 8) / 8.0
        tape_signal = (((asset_index * 11 + date_index * 5) % 19) - 9) / 9.0
        bars = [0.0] * len(BARS_MIN_V2_FIELDS)
        tape = [0.0] * len(TAPE_MIN_V2_FIELDS)
        bars[0] = bar_signal
        bars[BARS_MIN_V2_FIELDS.index("reversal_5")] = -0.1 * bar_signal
        bars[BARS_MIN_V2_FIELDS.index("trend_21_minus_5")] = 0.2 * bar_signal
        tape[0] = tape_signal
        tape[TAPE_MIN_V2_FIELDS.index("signed_dollar_flow_fraction")] = tape_signal
        feature_body = {
            "decision_session_date": decision_session_date,
            "source_session_date": source_session_date,
            "security_id": security_id,
            "decision_membership_rank": asset_index + 1,
            "source_staleness_sessions": 2,
            "source_listed": True,
            "source_tradable": True,
            "source_observed_regular_trade": True,
            "source_halt_or_no_print": False,
            "bars_values": tuple(bars),
            "bars_valid": (True,) * len(bars),
            "tape_values": tuple(tape),
            "tape_valid": (True,) * len(tape),
            "source_panel_row_receipt_sha256": semantic_sha256(
                ("panel", decision_session_date, security_id)
            ),
            "feature_accounting_security_inventory_sha256": semantic_sha256(
                ("feature-accounting", decision_session_date, security_id)
            ),
            "tape_population_row_receipt_sha256": semantic_sha256(
                ("tape-population", decision_session_date, security_id)
            ),
        }
        feature_rows.append(
            MassiveProfitabilityOriginFeatureRowV2(
                **feature_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(feature_body),
            )
        )
        alpha = 0.02 * (0.5 * bar_signal + 0.5 * tape_signal)
        target_body = {
            "decision_session_date": decision_session_date,
            "security_id": security_id,
            "simple_returns": (alpha, alpha, alpha, alpha),
            "valid": (True, True, True, True),
            "terminal_zero_value": (False, False, False, False),
            "conservative_total_loss_fallback": False,
            "target_accounting_row_receipt_sha256": semantic_sha256(
                ("target-accounting", decision_session_date, security_id)
            ),
        }
        target_rows.append(
            MassiveProfitabilityTargetRowV1(
                **target_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(target_body),
            )
        )

    origin_receipt = semantic_sha256(("origin", decision_session_date))
    feature_accounting = semantic_sha256(("feature-accounting", decision_session_date))
    feature_body = {
        "origin_receipt_sha256": origin_receipt,
        "origin_plan_semantic_receipt_sha256": _DIGESTS[0],
        "decision_session_date": decision_session_date,
        "source_session_date": source_session_date,
        "feature_cutoff_at_ms": date_index + 1,
        "maximum_economic_input_at_ms": date_index + 1,
        "maximum_source_available_at_ms": date_index + 1,
        "source_staleness_sessions": 2,
        "input_session_dates": input_session_dates,
        "rows": tuple(feature_rows),
        "daily_input_authority_semantic_receipt_sha256": _DIGESTS[1],
        "feature_accounting_authority_semantic_receipt_sha256": feature_accounting,
        "accounting_freeze_semantic_receipt_sha256": _DIGESTS[2],
        "terminal_authority_semantic_receipt_sha256": _DIGESTS[3],
        "source_input_inventory_sha256": semantic_sha256(
            ("source-inputs", decision_session_date)
        ),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in feature_rows)
        ),
        "input_schemas": tuple(
            sorted(
                (
                    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SCHEMA,
                    MASSIVE_PROFITABILITY_FEATURE_ACCOUNTING_AUTHORITY_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
                    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
                )
            )
        ),
        "source_inputs_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_ORIGIN_FEATURES_V3_SOURCE_SHA256,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    provisional_feature = MassiveProfitabilityOriginFeaturesV3(
        **feature_body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256=semantic_sha256(("feature-audit", decision_session_date)),
    )
    feature = replace(
        provisional_feature,
        semantic_receipt_sha256=semantic_sha256(
            provisional_feature.semantic_unsigned()
        ),
    )
    feature.validate()

    target_spec = MassiveProfitabilityTargetSpecV1.build()
    target_body = {
        "decision_session_date": decision_session_date,
        "origin_receipt_sha256": origin_receipt,
        "origin_plan_semantic_receipt_sha256": _DIGESTS[0],
        "target_spec": target_spec,
        "target_accounting_authority_semantic_receipt_sha256": semantic_sha256(
            ("target-accounting-authority", decision_session_date)
        ),
        "accounting_freeze_semantic_receipt_sha256": _DIGESTS[2],
        "terminal_authority_semantic_receipt_sha256": _DIGESTS[3],
        "terminal_accounting_mode": "exact-provider",
        "rows": tuple(target_rows),
        "valid_counts_by_horizon": (len(_SECURITY_IDS),) * 4,
        "exact_provider_disposition_count": 0,
        "conservative_total_loss_count": 0,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in target_rows)
        ),
        "input_schemas": tuple(
            sorted(
                (
                    MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_FILL_SOURCE_AUTHORITY_V2_SCHEMA,
                    MASSIVE_PROFITABILITY_TERMINAL_COVERAGE_AUTHORITY_V1_SCHEMA,
                    MASSIVE_PROFITABILITY_ACCOUNTING_FREEZE_V1_SCHEMA,
                )
            )
        ),
        "fill_sources_qualified": True,
        "economic_values_data_qualified": True,
        "terminal_accounting_data_qualified": True,
        "source_inputs_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGETS_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGETS_V2_SOURCE_SHA256,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    provisional_target = MassiveProfitabilityTargetsV2(
        **target_body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        audit_receipt_sha256=semantic_sha256(("target-audit", decision_session_date)),
    )
    target = replace(
        provisional_target,
        semantic_receipt_sha256=semantic_sha256(provisional_target.semantic_unsigned()),
    )
    target.validate()
    return feature, target


def _gate(
    *,
    phase_plan,
    features: tuple[MassiveProfitabilityOriginFeaturesV3, ...],
    targets: tuple[MassiveProfitabilityTargetsV2, ...],
) -> MassiveProfitabilityDataGateV2:
    dates = tuple(row.decision_session_date for row in features)
    outer = tuple(
        value
        for fold in phase_plan.outer_folds
        for value in fold.outer_test_session_dates
    )
    support = []
    outer_set = set(outer)
    for session_date in dates:
        body = {
            "decision_session_date": session_date,
            "decision_member_count": len(_SECURITY_IDS),
            "required_common_valid_count": len(_SECURITY_IDS),
            "feature_row_count": len(_SECURITY_IDS),
            "target_common_valid_count": len(_SECURITY_IDS),
            "common_security_inventory_sha256": semantic_sha256(_SECURITY_IDS),
            "phase": "outer-test" if session_date in outer_set else "development",
            "passed": True,
        }
        support.append(
            MassiveProfitabilityDateSupportGateV2(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
        )
    gate_body = {
        "coverage_semantic_receipt_sha256": _DIGESTS[4],
        "archive_freeze_semantic_receipt_sha256": phase_plan.archive_freeze_semantic_receipt_sha256,
        "accounting_freeze_semantic_receipt_sha256": _DIGESTS[2],
        "origin_plan_semantic_receipt_sha256": _DIGESTS[0],
        "daily_input_authority_semantic_receipt_sha256": _DIGESTS[1],
        "fill_source_authority_semantic_receipt_sha256": _DIGESTS[5],
        "terminal_authority_semantic_receipt_sha256": _DIGESTS[3],
        "lockbox_seal_semantic_receipt_sha256": _DIGESTS[6],
        "gated_session_dates": dates,
        "outer_test_session_dates": outer,
        "excluded_lockbox_session_dates": phase_plan.lockbox_session_dates,
        "feature_receipts": tuple(row.semantic_receipt_sha256 for row in features),
        "target_receipts": tuple(row.semantic_receipt_sha256 for row in targets),
        "date_support_gates": tuple(support),
        "input_schemas": _MASSIVE_PROFITABILITY_DATA_GATE_V2_INPUT_SCHEMAS,
        "source_transport_qualified": True,
        "rank_bar_data_qualified": True,
        "exact_frozen_acquisition_complete": True,
        "exact_accounting_freeze_complete": True,
        "exact_origin_plan_membership_complete": True,
        "exact_feature_cutoff_complete": True,
        "exact_source_staleness_complete": True,
        "exact_64_session_rectangles_complete": True,
        "fill_source_complete": True,
        "economic_accounting_data_qualified": True,
        "terminal_accounting_complete": True,
        "common_model_support_complete": True,
        "package_rematerialization_complete": True,
        "future_mutation_invariance_complete": True,
        "lockbox_targets_sealed_and_excluded": True,
        "legacy_generations_rejected": True,
        "data_gate_passed": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_DATA_GATE_V2_SOURCE_SHA256,
        "development_training_authorized": True,
        "outer_prediction_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    provisional = MassiveProfitabilityDataGateV2(
        **gate_body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        component_audit_inventory_sha256=_DIGESTS[7],
        audit_receipt_sha256="0" * 64,
    )
    semantic = semantic_sha256(provisional.semantic_unsigned())
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic,
                "component_audit_inventory_sha256": _DIGESTS[7],
            }
        ),
    )
    result.validate()
    return result


def _target_authority(
    *,
    session_dates: tuple[str, ...],
    feature: MassiveProfitabilityOriginFeaturesV3,
    target: MassiveProfitabilityTargetsV2,
) -> MassiveProfitabilityTargetAccountingAuthorityV2:
    target_by_security = {row.security_id: row for row in target.rows}
    paths = []
    for security_id in _SECURITY_IDS:
        economic_return = target_by_security[security_id].simple_returns[0]
        values = (100.0,) + (100.0 * (1.0 + economic_return),) * 63
        path_body = {
            "security_id": security_id,
            "economic_at_ms": tuple(range(64)),
            "available_at_ms": tuple(range(64)),
            "values": values,
            "valid": (True,) * 64,
            "terminal": (False,) * 64,
            "mark_kinds": ("market",) * 64,
            "mark_receipts": tuple(
                semantic_sha256((feature.decision_session_date, security_id, offset))
                for offset in range(64)
            ),
            "unresolved_terminal_fallback_session_offset": None,
            "conservative_total_loss_fallback": False,
        }
        paths.append(
            MassiveProfitabilityTargetEconomicPathRowV1(
                **path_body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(path_body),
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in paths))
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SCHEMA,
        "origin_receipt_sha256": feature.origin_receipt_sha256,
        "origin_plan_semantic_receipt_sha256": feature.origin_plan_semantic_receipt_sha256,
        "decision_session_date": feature.decision_session_date,
        "session_dates": session_dates,
        "rows": tuple(asdict(row) for row in paths),
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "daily_input_authority_semantic_receipt_sha256": feature.daily_input_authority_semantic_receipt_sha256,
        "fill_source_authority_semantic_receipt_sha256": _DIGESTS[5],
        "terminal_authority_semantic_receipt_sha256": feature.terminal_authority_semantic_receipt_sha256,
        "economic_coverage_semantic_receipt_sha256": _DIGESTS[8],
        "scoped_economic_event_inventory_sha256": semantic_sha256(()),
        "row_inventory_sha256": row_inventory,
        "fill_sources_qualified": True,
        "economic_values_data_qualified": True,
        "terminal_accounting_complete": True,
        "conservative_total_loss_target_count": 0,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    economic_audit = semantic_sha256(("economic-audit", feature.decision_session_date))
    result = MassiveProfitabilityTargetAccountingAuthorityV2(
        **{**semantic, "rows": tuple(paths)},  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_receipt,
        economic_archive_audit_receipt_sha256=economic_audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_archive_audit_receipt_sha256": economic_audit,
            }
        ),
    )
    result.validate()
    return result


def _mean_net_by_cost(prediction, target_accounting) -> tuple[float, float, float]:
    inputs = []
    for index, row in enumerate(prediction.rows):
        body = {
            "decision_session_date": row.decision_session_date,
            "security_id": row.security_id,
            "raw_scores": row.mean,
            "exposures": (
                1.0,
                float(index % 8),
                float((index * 3) % 11),
                float((index * 5) % 13),
                float((index * 7) % 17),
                float((index * 11) % 19),
            ),
            "trailing_63_session_adv": 100_000_000.0,
            "prediction_row_receipt_sha256": row.receipt_sha256,
            "feature_row_receipt_sha256": semantic_sha256(("residual-feature", index)),
            "feature_accounting_row_inventory_sha256": semantic_sha256(
                ("residual-accounting", index)
            ),
        }
        inputs.append(
            MassiveProfitabilityResidualInputRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    research = build_massive_profitability_residual_scores_v1(
        setting_id=prediction.setting_id,
        fold_index=0,
        evaluation_plan_semantic_receipt_sha256=_DIGESTS[9],
        prediction_semantic_receipt_sha256=prediction.semantic_receipt_sha256,
        rows=inputs,
    )
    # This test is explicitly nonreportable. Promote only the deterministic
    # row utility into the existing selection guard; the production V6 path
    # performs this promotion from its frozen source bundle.
    promoted = replace(research, outer_evaluation_authorized=True)
    promoted = replace(
        promoted, semantic_receipt_sha256=semantic_sha256(promoted.unsigned())
    )
    promoted.validate()
    selected = select_massive_profitability_tranches_v1(
        residual_scores=promoted, target_accounting=target_accounting
    )
    pnl = evaluate_massive_profitability_fixed_tranches_v1(
        selected=selected, target_accounting=target_accounting
    )
    return tuple(
        sum(row.net_returns[cost_index] for row in pnl.rows) / len(pnl.rows)
        for cost_index in range(3)
    )  # type: ignore[return-value]


def _forged_checkpoint(
    *, root: Path, checkpoint, artifact_id: str, mutate: str, committed_at_ms: int
):
    source_run = checkpoint.run
    run_v1 = source_run.run_v2.run_v1
    trace = list(source_run.epoch_trace)
    if mutate == "weight":
        name, value = run_v1.model_state[0]
        state = ((name, value + 1.0),) + run_v1.model_state[1:]
        model_hash = state_dict_sha256(dict(state))
        run_v1 = replace(run_v1, model_state=state, model_state_sha256=model_hash)
        best = trace[run_v1.best_epoch]
        best = replace(best, model_state_sha256=model_hash, receipt_sha256="0" * 64)
        trace[run_v1.best_epoch] = replace(
            best, receipt_sha256=semantic_sha256(best.unsigned())
        )
    elif mutate == "validation":
        validation = tuple(value + 1.0 for value in run_v1.validation_rank_ic)
        run_v1 = replace(run_v1, validation_rank_ic=validation)
        best = trace[run_v1.best_epoch]
        best = replace(
            best,
            validation_rank_ic=validation,
            validation_mean_rank_ic=sum(validation) / len(validation),
            receipt_sha256="0" * 64,
        )
        trace[run_v1.best_epoch] = replace(
            best, receipt_sha256=semantic_sha256(best.unsigned())
        )
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mutate)
    run_receipt_body = {
        "setting_id": run_v1.setting_id,
        "fold_index": run_v1.fold_index,
        "seed": run_v1.seed,
        "best_epoch": run_v1.best_epoch,
        "completed_epochs": run_v1.completed_epochs,
        "validation_rank_ic": run_v1.validation_rank_ic,
        "normalization_receipt_sha256": run_v1.normalization.receipt_sha256,
        "model_state_sha256": run_v1.model_state_sha256,
        "fit_inventory_sha256": run_v1.fit_inventory_sha256,
        "validation_inventory_sha256": run_v1.validation_inventory_sha256,
        "training_source_receipt_sha256": run_v1.training_source_receipt_sha256,
        "tournament_plan_receipt_sha256": run_v1.tournament_plan_receipt_sha256,
        "training_config_receipt_sha256": run_v1.training_config_receipt_sha256,
        "outer_prediction_authorized": run_v1.outer_prediction_authorized,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    run_v1 = replace(run_v1, run_receipt_sha256=semantic_sha256(run_receipt_body))
    run_v2 = bind_massive_profitability_trained_run_v2(
        run_v1=run_v1,
        dataset_semantic_receipt_sha256=source_run.run_v2.dataset_semantic_receipt_sha256,
        dataset_source_receipt_sha256=source_run.run_v2.dataset_source_receipt_sha256,
        dataset_v2_receipt_sha256=source_run.run_v2.dataset_v2_receipt_sha256,
        tournament_plan_receipt_sha256=source_run.run_v2.tournament_plan_receipt_sha256,
        tournament_plan_source_receipt_sha256=source_run.run_v2.tournament_plan_source_receipt_sha256,
        phase_plan_semantic_receipt_sha256=source_run.run_v2.phase_plan_semantic_receipt_sha256,
        fold_receipt_sha256=source_run.run_v2.fold_receipt_sha256,
    )
    run_v3 = bind_massive_profitability_trained_run_v3(
        run_v2=run_v2,
        checkpoint_v2_source_receipt_sha256=source_run.checkpoint_v2_source_receipt_sha256,
        checkpoint_v2_payload_relative_path=source_run.checkpoint_v2_payload_relative_path,
        checkpoint_v2_verified_at_ms=source_run.checkpoint_v2_verified_at_ms,
        training_runtime=source_run.training_runtime,
        epoch_trace=tuple(trace),
    )
    return publish_massive_profitability_model_checkpoint_v3(
        root=root,
        artifact_id=artifact_id,
        run=run_v3,
        committed_at_ms=committed_at_ms,
    )


def _prediction_with_forged_mean(*, root: Path, prediction, committed_at_ms: int):
    payload = json.loads(
        read_loaded_massive_source_bytes(
            root=root, loaded_source=prediction.loaded_source
        )
    )
    payload["rows"][0]["mean"][0] += 1.0
    first_row = payload["rows"][0]
    first_row["receipt_sha256"] = semantic_sha256(
        {key: value for key, value in first_row.items() if key != "receipt_sha256"}
    )
    payload["row_inventory_sha256"] = semantic_sha256(
        tuple(row["receipt_sha256"] for row in payload["rows"])
    )
    payload["semantic_receipt_sha256"] = semantic_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "semantic_receipt_sha256"
        }
    )
    relative = "massive-profitability/predictions-v3/forged-mean.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PREDICTIONS_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PREDICTIONS_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=prediction.dataset_semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id="P0-PREDICTIONS-V3-forged-mean",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return parse_massive_profitability_outer_predictions_v3(
        root=root, loaded_source=loaded
    )


def test_one_fold_v6_source_to_net_pnl_vertical_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    freeze = _freeze(freeze_root)
    sessions, _, _, _ = _inputs()
    phase = materialize_massive_profitability_phase_plan_v2(
        root=tmp_path,
        archive_freeze=freeze,
        artifact_id="vertical-slice",
        committed_at_ms=freeze.data_freeze_at_ms + 2,
        entitlement_receipt_sha256=_DIGESTS[10],
    )
    fold = adapt_massive_profitability_training_fold_v2(phase.outer_folds[0])
    required_dates = tuple(
        sorted(
            set(fold.fit_session_dates)
            | set(fold.inner_validation_session_dates)
            | {
                value
                for row in phase.outer_folds
                for value in row.outer_test_session_dates
            }
            | set(phase.outer_to_lockbox_embargo_session_dates)
        )
    )
    session_dates = tuple(row.session_date for row in sessions.sessions)
    session_index = {value: index for index, value in enumerate(session_dates)}
    features = []
    targets = []
    for date_index, decision_date in enumerate(required_dates):
        decision_index = session_index[decision_date]
        source_index = decision_index - 2
        feature, target = _feature_and_target(
            decision_session_date=decision_date,
            source_session_date=session_dates[source_index],
            input_session_dates=session_dates[source_index - 63 : source_index + 1],
            date_index=date_index,
        )
        features.append(feature)
        targets.append(target)
    feature_inventory = tuple(features)
    target_inventory = tuple(targets)
    gate = _gate(phase_plan=phase, features=feature_inventory, targets=target_inventory)
    dataset = materialize_massive_profitability_tournament_dataset_v3(
        root=tmp_path,
        artifact_id="vertical-slice",
        data_gate=gate,
        phase_plan=phase,
        features=feature_inventory,
        targets=target_inventory,
        committed_at_ms=freeze.data_freeze_at_ms + 3,
    )
    generic_dataset = parse_massive_profitability_tournament_dataset_v3(
        root=tmp_path, loaded_source=dataset.loaded_source
    )
    assert generic_dataset.runtime_data_qualified is False
    assert generic_dataset.runtime_dataset is None
    promoted_dataset = authorize_massive_profitability_tournament_dataset_v3(
        root=tmp_path,
        dataset=generic_dataset,
        data_gate=gate,
        phase_plan=phase,
        features=feature_inventory,
        targets=target_inventory,
    )
    assert promoted_dataset.runtime_data_qualified is True
    assert (
        promoted_dataset.tensor_source_array_receipts
        == dataset.tensor_source_array_receipts
    )

    mutated_row = replace(
        feature_inventory[0].rows[0],
        bars_values=(99.0,) + feature_inventory[0].rows[0].bars_values[1:],
    )
    mutated_row = replace(
        mutated_row, receipt_sha256=semantic_sha256(mutated_row.unsigned())
    )
    mutated_feature = replace(
        feature_inventory[0],
        rows=(mutated_row,) + feature_inventory[0].rows[1:],
        row_inventory_sha256=semantic_sha256(
            (mutated_row.receipt_sha256,)
            + tuple(row.receipt_sha256 for row in feature_inventory[0].rows[1:])
        ),
    )
    mutated_feature = replace(
        mutated_feature,
        semantic_receipt_sha256=semantic_sha256(mutated_feature.semantic_unsigned()),
    )
    mutated_feature.validate()
    with pytest.raises(ValueError, match="feature or target inventory differs"):
        authorize_massive_profitability_tournament_dataset_v3(
            root=tmp_path,
            dataset=generic_dataset,
            data_gate=gate,
            phase_plan=phase,
            features=(mutated_feature,) + feature_inventory[1:],
            targets=target_inventory,
        )

    plan = materialize_massive_profitability_tournament_plan_v2(
        root=tmp_path,
        artifact_id="vertical-slice",
        data_gate=gate,
        phase_plan=phase,
        committed_at_ms=freeze.data_freeze_at_ms + 4,
    )
    fast_config = MassiveProfitabilityTrainingConfigV1(
        maximum_epochs=2,
        early_stopping_patience=1,
        complete_dates_per_batch=756,
    )
    for module_name in (
        "rl_quant.training.massive_profitability_tournament_v1",
        "rl_quant.training.massive_profitability_training_replay_v3",
        "rl_quant.evaluation.massive_profitability_training_v4",
        "rl_quant.training.massive_profitability_trained_run_v2",
        "rl_quant.evaluation.massive_profitability_predictions_v3",
    ):
        module = __import__(
            module_name, fromlist=["MassiveProfitabilityTrainingConfigV1"]
        )
        monkeypatch.setattr(
            module, "MassiveProfitabilityTrainingConfigV1", lambda: fast_config
        )

    checkpoints_v3 = {}
    checkpoints_v2 = {}
    for setting_id in ("MV02", "MV04", "MV04-SHUFFLE"):
        setting_checkpoints = []
        compat_checkpoints = []
        for seed in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1:
            checkpoint = train_and_publish_massive_profitability_fold_v4(
                root=tmp_path,
                artifact_id=f"vertical-{setting_id.lower()}-{seed}",
                dataset=dataset,
                data_gate=gate,
                phase_plan=phase,
                features=feature_inventory,
                targets=target_inventory,
                tournament_plan=plan,
                fold=fold,
                setting_id=setting_id,
                seed=seed,
                committed_at_ms=freeze.data_freeze_at_ms + 10 + seed,
            )
            setting_checkpoints.append(checkpoint)
            compat_checkpoints.append(
                load_massive_profitability_prediction_checkpoint_v2_from_v3(
                    root=tmp_path, checkpoint=checkpoint
                )
            )
        checkpoints_v3[setting_id] = tuple(setting_checkpoints)
        checkpoints_v2[setting_id] = tuple(compat_checkpoints)

    representative = checkpoints_v3["MV04"][0]
    generic_checkpoint = parse_massive_profitability_model_checkpoint_v3(
        root=tmp_path, loaded_source=representative.loaded_source
    )
    assert generic_checkpoint.run.runtime_training_replayed is False
    promoted_checkpoint = authorize_massive_profitability_checkpoint_v3_from_roots(
        root=tmp_path,
        checkpoint=generic_checkpoint,
        dataset=dataset,
        data_gate=gate,
        phase_plan=phase,
        features=feature_inventory,
        targets=target_inventory,
        tournament_plan=plan,
        fold=fold,
    )
    assert promoted_checkpoint.run.epoch_trace == representative.run.epoch_trace
    assert (
        promoted_checkpoint.run.run_v2.run_v1.model_state_sha256
        == representative.run.run_v2.run_v1.model_state_sha256
    )
    forged_weight = _forged_checkpoint(
        root=tmp_path,
        checkpoint=representative,
        artifact_id="forged-weight",
        mutate="weight",
        committed_at_ms=freeze.data_freeze_at_ms + 40,
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        authorize_massive_profitability_checkpoint_v3_from_roots(
            root=tmp_path,
            checkpoint=forged_weight,
            dataset=dataset,
            data_gate=gate,
            phase_plan=phase,
            features=feature_inventory,
            targets=target_inventory,
            tournament_plan=plan,
            fold=fold,
        )
    forged_validation = _forged_checkpoint(
        root=tmp_path,
        checkpoint=representative,
        artifact_id="forged-validation",
        mutate="validation",
        committed_at_ms=freeze.data_freeze_at_ms + 41,
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        authorize_massive_profitability_checkpoint_v3_from_roots(
            root=tmp_path,
            checkpoint=forged_validation,
            dataset=dataset,
            data_gate=gate,
            phase_plan=phase,
            features=feature_inventory,
            targets=target_inventory,
            tournament_plan=plan,
            fold=fold,
        )

    predictions = {}
    for setting_id in ("MV00", "MV02", "MV04", "MV04-SHUFFLE"):
        prediction = publish_massive_profitability_outer_predictions_v3(
            root=tmp_path,
            artifact_id=f"vertical-{setting_id.lower()}",
            dataset=dataset,
            data_gate=gate,
            phase_plan=phase,
            features=feature_inventory,
            targets=target_inventory,
            tournament_plan=plan,
            fold=fold,
            setting_id=setting_id,
            checkpoints=checkpoints_v2.get(setting_id, ()),
            committed_at_ms=freeze.data_freeze_at_ms + 30,
        )
        generic_prediction = parse_massive_profitability_outer_predictions_v3(
            root=tmp_path, loaded_source=prediction.loaded_source
        )
        assert generic_prediction.runtime_prediction_replayed is False
        promoted_prediction = authorize_massive_profitability_outer_predictions_v3(
            root=tmp_path,
            prediction=generic_prediction,
            dataset=dataset,
            data_gate=gate,
            phase_plan=phase,
            features=feature_inventory,
            targets=target_inventory,
            tournament_plan=plan,
            fold=fold,
            checkpoints=checkpoints_v2.get(setting_id, ()),
        )
        assert promoted_prediction.rows == prediction.rows
        predictions[setting_id] = promoted_prediction

    altered_prediction = _prediction_with_forged_mean(
        root=tmp_path,
        prediction=predictions["MV04"],
        committed_at_ms=freeze.data_freeze_at_ms + 50,
    )
    with pytest.raises(
        MassiveProfitabilityPredictionsV3Error,
        match="does not replay from its checkpoints",
    ):
        authorize_massive_profitability_outer_predictions_v3(
            root=tmp_path,
            prediction=altered_prediction,
            dataset=dataset,
            data_gate=gate,
            phase_plan=phase,
            features=feature_inventory,
            targets=target_inventory,
            tournament_plan=plan,
            fold=fold,
            checkpoints=checkpoints_v2["MV04"],
        )

    target_lookup = {row.decision_session_date: row for row in target_inventory}
    feature_lookup = {row.decision_session_date: row for row in feature_inventory}
    accounting = tuple(
        _target_authority(
            session_dates=session_dates[
                session_index[session_date] : session_index[session_date] + 64
            ],
            feature=feature_lookup[session_date],
            target=target_lookup[session_date],
        )
        for session_date in fold.outer_test_session_dates
    )
    for prediction in predictions.values():
        means = _mean_net_by_cost(prediction, accounting)
        assert all(math.isfinite(value) for value in means)
        assert means[0] > means[1] > means[2]
        assert prediction.profitability_reporting_authorized is False
        assert prediction.lockbox_access_authorized is False
