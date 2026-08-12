from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256,
    M03RV9HorizonBinding,
)
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import (
    M03RV9AlphaDistribution,
    M03RV9AlphaHeadIdentity,
)
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    M03RV9ProjectionError,
    M03RV9ProjectorManifest,
    bind_m03r_v9_projector_to_risk_source,
    build_m03r_v9_device_risk_state,
    freeze_m03r_v9_projector_manifest,
    load_m03r_v9_projector_manifest,
    project_m03r_v9_active_book,
    project_m03r_v9_signal_to_exposure_null,
    write_m03r_v9_projector_manifest,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03R_V9_PROJECTOR_EXPOSURE_FAMILIES,
    M03R_V9_PROJECTOR_EXPOSURE_NAMES,
    M03R_V9_SECTOR_EXPOSURE_NAMES,
    M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
    M03RV9MaterializedRiskSource,
    M03RV9WrittenRiskSource,
    _canonical_sha256,
)
from rl_quant.training.top2000_m03r_v9_risk_source import (
    M03RV9RiskSourceInventory,
    audit_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v9_runtime import run_m03r_v9_simple_sleeve
from rl_quant.training.top2000_m03r_v9_selection import (
    build_m03r_v9_simple_sleeve_fold_evidence,
)


def _source(
    *, states: int = 2, assets: int = 4
) -> tuple[M03RV9MaterializedRiskSource, M03RV9WrittenRiskSource]:
    names = M03R_V9_PROJECTOR_EXPOSURE_NAMES
    loadings = torch.zeros((states, assets, 1 + len(names)), dtype=torch.float32)
    loadings[:, 1:, 0] = 1.0
    sector_count = len(M03R_V9_SECTOR_EXPOSURE_NAMES)
    for asset in range(1, assets):
        loadings[:, asset, 1 + ((asset - 1) % sector_count)] = 1.0
    beta_index = 1 + sector_count
    loadings[:, 1:, beta_index] = torch.linspace(0.8, 1.2, assets - 1)
    style_start = beta_index + 1
    loadings[:, 1:, style_start:] = torch.stack(
        (
            torch.linspace(-0.5, 0.5, assets - 1),
            torch.linspace(0.2, 0.6, assets - 1),
            torch.linspace(1.0, 2.0, assets - 1),
        ),
        dim=-1,
    )
    source_receipt = "a" * 64
    axis = "b" * 64
    exposures = qualify_m03r_v9_origin_risk_exposures(
        state_start_index=0,
        cash_index=0,
        projector_exposure_names=names,
        projector_exposure_families=M03R_V9_PROJECTOR_EXPOSURE_FAMILIES,
        asset_axis_sha256=axis,
        source_receipt_sha256=source_receipt,
        exposure_loadings=loadings,
        regression_weights=torch.cat(
            (
                torch.zeros((states, 1), dtype=torch.float32),
                torch.ones((states, assets - 1), dtype=torch.float32),
            ),
            dim=1,
        ),
        decision_timestamp_ms=torch.arange(1, states + 1, dtype=torch.int64) * 100,
        exposure_available_timestamp_ms=(
            (torch.arange(1, states + 1, dtype=torch.int64) * 100 - 10)
            .view(states, 1, 1)
            .expand(states, assets, 3)
            .clone()
        ),
    )
    inventory = M03RV9RiskSourceInventory(
        source_id="test-pit-risk-source",
        source_schema_sha256="c" * 64,
        asset_axis_sha256=axis,
        source_columns=("sic", "daily-close", "daily-volume"),
        sector_exposure_names=M03R_V9_SECTOR_EXPOSURE_NAMES,
        style_risk_exposure_names=M03R_V9_STYLE_RISK_EXPOSURE_NAMES,
        active_beta_exposure_name="active-beta-63-session-to-c1-proxy",
        point_in_time_sector_receipt_sha256="d" * 64,
        point_in_time_style_risk_receipt_sha256="e" * 64,
        point_in_time_active_beta_receipt_sha256="f" * 64,
        origin_availability_receipt_sha256="1" * 64,
        projector_manifest_sha256=None,
        target_projector_exposure_names_match=False,
    )
    readiness = audit_m03r_v9_risk_source(inventory)
    provisional = M03RV9MaterializedRiskSource(
        exposures=exposures,
        inventory=inventory,
        readiness=readiness,
        cache_sha256="2" * 64,
        cache_identity="3" * 64,
        action_hash=axis,
        first_exchange_date="2022-01-03",
        last_exchange_date="2025-12-29",
        polygon_validation_receipt_file_sha256="4" * 64,
        polygon_validation_receipt_payload_sha256="5" * 64,
        sector_receipt_sha256="d" * 64,
        active_beta_receipt_sha256="f" * 64,
        style_risk_receipt_sha256="e" * 64,
        origin_availability_receipt_sha256="1" * 64,
        selected_overview_inventory_sha256="6" * 64,
        source_overview_file_inventory_sha256="7" * 64,
        raw_symbol_count=assets - 1,
        mapped_risky_action_count=assets - 1,
        unused_raw_symbols=(),
        selected_snapshot_count=assets - 1,
        explicit_unknown_state_asset_count=0,
        future_delta_rows_consumed=0,
        receipt_sha256="0" * 64,
    )
    source = replace(
        provisional,
        receipt_sha256=_canonical_sha256(provisional.canonical_payload()),
    )
    source.validate()
    written = M03RV9WrittenRiskSource(
        artifact_path=Path("risk-exposures.pt"),
        artifact_file_sha256="8" * 64,
        manifest_path=Path("risk-source-manifest.json"),
        manifest_file_sha256="9" * 64,
    )
    return source, written


def test_projector_closes_only_known_source_blockers() -> None:
    source, written = _source()
    projector = freeze_m03r_v9_projector_manifest()
    binding = bind_m03r_v9_projector_to_risk_source(source, written, projector)

    assert source.readiness.blocker_codes == (
        "missing-projector-manifest",
        "target-projector-exposure-name-mismatch",
    )
    assert binding.original_blocker_codes == source.readiness.blocker_codes
    assert binding.bound_readiness.predictive_worker_authorized
    assert not binding.bound_readiness.economic_panel_authorized
    assert binding.bound_readiness.blocker_codes == ()
    assert binding.bound_inventory.projector_manifest_sha256 == (
        projector.manifest_sha256
    )
    assert all(value != 0.0 for value in projector.exposure_lower_bounds)
    assert all(value != 0.0 for value in projector.exposure_upper_bounds)


def test_projector_rejects_name_drift_and_does_not_generalize_blockers() -> None:
    source, written = _source()
    projector = freeze_m03r_v9_projector_manifest()
    drifted_projector = replace(
        projector,
        exposure_names=(
            *projector.exposure_names[:-1],
            "style-drifted",
        ),
    )
    with pytest.raises(M03RV9ProjectionError, match="manifest drifted"):
        drifted_projector.validate()
    extra_blocker = replace(
        source,
        readiness=replace(
            source.readiness,
            blocker_codes=(*source.readiness.blocker_codes, "unexpected"),
        ),
    )
    with pytest.raises(ValueError):
        bind_m03r_v9_projector_to_risk_source(extra_blocker, written, projector)


def test_projector_manifest_is_no_clobber_and_externally_hash_bound(
    tmp_path: Path,
) -> None:
    source, written = _source()
    projector = freeze_m03r_v9_projector_manifest()
    binding = bind_m03r_v9_projector_to_risk_source(source, written, projector)
    path = tmp_path / "projector-manifest.json"
    file_sha = write_m03r_v9_projector_manifest(projector, binding, path)
    loaded_projector, loaded_binding = load_m03r_v9_projector_manifest(
        path,
        expected_file_sha256=file_sha,
    )
    assert loaded_projector.manifest_sha256 == projector.manifest_sha256
    assert loaded_binding.binding_sha256 == binding.binding_sha256
    with pytest.raises(FileExistsError):
        write_m03r_v9_projector_manifest(projector, binding, path)
    with pytest.raises(M03RV9ProjectionError, match="file hash"):
        load_m03r_v9_projector_manifest(
            path,
            expected_file_sha256="f" * 64,
        )


def _device_state_fixture(
    origin_state_indices: tuple[int, ...] = (70, 89),
) -> tuple[
    M03RV9MaterializedRiskSource,
    M03RV9ProjectorManifest,
    M03RV9DeviceRiskState,
    torch.Tensor,
    torch.Tensor,
]:
    source, written = _source(states=90, assets=102)
    projector = freeze_m03r_v9_projector_manifest()
    binding = bind_m03r_v9_projector_to_risk_source(source, written, projector)
    state = torch.arange(90, dtype=torch.float64).unsqueeze(1)
    asset = torch.arange(102, dtype=torch.float64).unsqueeze(0)
    returns = 0.0001 * torch.sin(state / 3.0 + asset / 7.0) + 0.00001 * asset
    returns[:, 0] = 0.0
    available = torch.ones_like(returns, dtype=torch.bool)
    available[:, 0] = False
    risk_state = build_m03r_v9_device_risk_state(
        source,
        binding,
        projector,
        daily_log_returns=returns,
        return_available=available,
        daily_returns_receipt_sha256="a" * 64,
        sequence_asset_axis_sha256=source.action_hash,
        checkpoint_asset_axis_sha256=source.action_hash,
        origin_state_indices=origin_state_indices,
        device=torch.device("cpu"),
    )
    return source, projector, risk_state, returns, available


def test_device_risk_state_is_once_qualified_and_asset_axis_bound() -> None:
    source, projector, risk_state, _, _ = _device_state_fixture()
    risk_state.validate()
    risk_state.require_fast_identity(
        sequence_asset_axis_sha256=source.action_hash,
        checkpoint_asset_axis_sha256=source.action_hash,
        expected_manifest_sha256=projector.manifest_sha256,
    )
    with pytest.raises(M03RV9ProjectionError, match="asset identity"):
        risk_state.require_fast_identity(
            sequence_asset_axis_sha256="f" * 64,
            checkpoint_asset_axis_sha256=source.action_hash,
            expected_manifest_sha256=projector.manifest_sha256,
        )


def test_projection_enforces_caps_and_detects_hot_path_tensor_mutation() -> None:
    source, projector, risk_state, _, _ = _device_state_fixture()
    benchmark = torch.full((1, 102), 0.98 / 101.0, dtype=torch.float64)
    benchmark[:, 0] = 0.02
    requested = benchmark.clone()
    requested[:, 1] -= 0.005
    requested[:, 2] += 0.005
    caps = torch.ones_like(benchmark)
    trade_mask = torch.ones_like(benchmark, dtype=torch.bool)
    result = project_m03r_v9_active_book(
        requested,
        benchmark,
        trade_mask,
        caps,
        torch.tensor([0.98], dtype=torch.float64),
        risk_state,
        origin_state_index=89,
        sequence_asset_axis_sha256=source.action_hash,
        checkpoint_asset_axis_sha256=source.action_hash,
        expected_manifest_sha256=projector.manifest_sha256,
    )
    assert result.radial_scale.item() < 1.0
    assert result.projected_weights[:, 1:].max().item() <= 0.01 + 2.0e-6
    assert 0.0 < result.requested_to_executed_retention.item() < 1.0

    risk_state.specific_variance.add_(1.0e-8)
    with pytest.raises(M03RV9ProjectionError, match="tensor version"):
        project_m03r_v9_active_book(
            benchmark,
            benchmark,
            trade_mask,
            caps,
            torch.tensor([0.98], dtype=torch.float64),
            risk_state,
            origin_state_index=89,
            sequence_asset_axis_sha256=source.action_hash,
            checkpoint_asset_axis_sha256=source.action_hash,
            expected_manifest_sha256=projector.manifest_sha256,
        )


def test_signal_is_projected_into_the_same_exposure_null_space() -> None:
    source, projector, risk_state, _, _ = _device_state_fixture()
    loadings = risk_state.exposure_loadings[1]
    prohibited_signal = loadings[:, 0].unsqueeze(0) + 0.25 * loadings[:, -1].unsqueeze(
        0
    )
    result = project_m03r_v9_signal_to_exposure_null(
        prohibited_signal,
        torch.ones_like(prohibited_signal, dtype=torch.bool),
        risk_state,
        origin_state_index=89,
        sequence_asset_axis_sha256=source.action_hash,
        checkpoint_asset_axis_sha256=source.action_hash,
        expected_manifest_sha256=projector.manifest_sha256,
    )
    assert result.signal_retention.item() < 1.0e-4
    assert result.projected_factor_component.abs().max().item() < 1.0e-5


def test_deterministic_simple_sleeve_carries_selected_distribution_to_gate() -> None:
    source, _projector, risk_state, _, _ = _device_state_fixture((70, 71, 72))
    transitions, assets = 3, 102
    benchmark = torch.full(
        (transitions + 1, 1, assets), 0.98 / 101.0, dtype=torch.float64
    )
    benchmark[:, :, 0] = 0.02
    time = torch.arange(transitions, dtype=torch.float64).view(-1, 1, 1)
    asset = torch.arange(assets, dtype=torch.float64).view(1, 1, -1)
    asset_returns = 0.002 * torch.sin(time + asset / 9.0)
    asset_returns[:, :, 0] = 0.0
    available = torch.ones((transitions + 1, 1, assets), dtype=torch.bool)
    caps = torch.ones_like(benchmark)
    caps[:, :, 1:] = 0.01
    sequence = Hold30Sequence(
        decision_state=torch.zeros(
            (transitions + 1, 1, assets, 1), dtype=torch.float64
        ),
        asset_returns=asset_returns,
        decision_available=available,
        fill_membership=available,
        fill_availability=available,
        benchmark_weights=benchmark,
        risk_asset_caps=caps,
        risk_gross_max=torch.full((transitions + 1, 1), 0.98, dtype=torch.float64),
        benchmark_net_returns=(benchmark[:-1] * asset_returns).sum(-1),
        initial_ledger=CohortLedger.from_staggered_endowment(
            benchmark[0],
            cash_index=0,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        axis_id=source.action_hash,
    )
    binding = M03RV9HorizonBinding(30, 30, 30)
    log_scale = torch.full(
        (1, assets, 4), torch.log(torch.tensor(0.01)).item(), dtype=torch.float64
    )
    distributions: list[M03RV9AlphaDistribution] = []
    for transition in range(transitions):
        base = 0.04 * torch.sin(
            torch.arange(assets, dtype=torch.float64) / 5.0 + transition
        )
        base[0] = 0.0
        means = torch.stack((0.25 * base, 0.75 * base, base, base), dim=-1).unsqueeze(0)
        distributions.append(
            M03RV9AlphaDistribution(
                mean_by_horizon=means,
                log_scale_by_horizon=log_scale,
                selected_horizon_sessions=30,
                selected_mean=means[..., 2],
                selected_scale=torch.exp(log_scale[..., 2]),
            )
        )
    identity = M03RV9AlphaHeadIdentity(
        selected_alpha_horizon=30,
        alpha_mean_head_state_sha256="a" * 64,
        alpha_scale_head_state_sha256="b" * 64,
        alpha_distribution_contract_sha256=(M03R_V9_ALPHA_DISTRIBUTION_CONTRACT_SHA256),
        horizon_binding_sha256=binding.receipt_sha256,
    )
    benchmark_gross = (benchmark[:-1] * asset_returns).sum(-1).squeeze(-1)
    trace = run_m03r_v9_simple_sleeve(
        sequence,
        tuple(distributions),
        risk_state,
        binding,
        identity,
        setting_id="V9-P0-factor-residual-ranked",
        fold_index=0,
        state_start_index=70,
        checkpoint_asset_axis_sha256=source.action_hash,
        source_receipt_sha256="c" * 64,
        benchmark_gross_returns=benchmark_gross,
        benchmark_one_way_turnover=torch.zeros(transitions, dtype=torch.float64),
    )
    trace.validate()
    assert not trace.learned_hazard_enabled
    assert trace.economic_optimizer_updates == 0
    assert trace.policy_one_way_turnover.sum().item() > 0.0
    assert trace.requested_weight_trace.shape == (transitions, assets)
    fold = build_m03r_v9_simple_sleeve_fold_evidence(
        setting_id="V9-P0-factor-residual-ranked",
        fold_index=0,
        horizon_binding=binding,
        policy_gross_returns=trace.policy_gross_returns,
        benchmark_gross_returns=trace.benchmark_gross_returns,
        policy_one_way_turnover=trace.policy_one_way_turnover,
        benchmark_one_way_turnover=trace.benchmark_one_way_turnover,
        requested_weight_trace=trace.requested_weight_trace,
        projected_weight_trace=trace.projected_weight_trace,
        signal_null_retention=trace.signal_null_retention,
        requested_to_executed_retention=(trace.requested_to_executed_retention),
        source_receipt_sha256=trace.trace_sha256,
    )
    assert fold.observation_count == transitions
