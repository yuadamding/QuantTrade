from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_graph_authority_v1 as graph_module,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MassiveAdaptiveRLTypedAuthorityInventoryV1,
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    build_massive_adaptive_rl_typed_authority_inventory_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    runtime_source_graph_authority_path_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1,
    MassiveAdaptiveRLRoleBoundSourceAuthorityV1,
    authorize_massive_adaptive_rl_source_bundle_v1,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    build_massive_adaptive_split_plan_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from test_massive_adaptive_rl_fit_forecast_v1 import _rl_fit_fixture
from test_massive_adaptive_profitability_v1_vertical_slice import _calibration_v2
from test_massive_adaptive_source_authorized_training_v1 import _sessions
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    run_massive_adaptive_rl_experiment_v2,
)


_GLOBAL = {
    "session-authority": "authorities/session-authority.json",
    "condition-authority": "authorities/condition-authority.json",
    "persisted-partition-inventory": "authorities/persisted-partition-inventory.json",
    "identity-authority": "authorities/identity-authority.json",
    "economic-event-archive": "authorities/economic-event-archive.json",
    "daily-input-authority": "authorities/daily-input-authority.json",
    "fill-source-authority": "authorities/fill-source-authority.json",
    "split-plan": "authorities/adaptive-split-plan.json",
}
_FOLD = {
    "training-window-inventory": "training-window-inventory.json",
    "supervised-checkpoint-inventory": "supervised-checkpoint-inventory.json",
    "calibration-inventory": "calibration-inventory.json",
    "fit-forecast-archive-inventory": "fit-forecast-archive-inventory.json",
    "decision-root-inventory": "decision-root-inventory.json",
    "context-origin-inventory": "context-origin-inventory.json",
}


@dataclass(frozen=True)
class _SyntheticRuntimeSource:
    semantic_receipt_sha256: str
    source_transport_qualified: bool = True
    daily_input_data_qualified: bool = True
    source_data_qualified: bool = True
    source_paths_replayed: bool = True
    candidate_source_data_qualified: bool = True
    source_geometry_replayed: bool = True

    def validate(self) -> None:
        assert len(self.semantic_receipt_sha256) == 64


@dataclass(frozen=True)
class _SyntheticPartition:
    source_session_date: str
    receipt_sha256: str
    schema: str = "synthetic-partition-v1"
    partition_spec_sha256: str = semantic_sha256("synthetic-partition-spec-v1")

    def validate(self) -> None:
        assert self.source_session_date
        assert len(self.receipt_sha256) == 64


def _persist_synthetic_source_graph(root: Path):
    runtimes = {}
    paths = {(role, None): path for role, path in _GLOBAL.items()}
    for fold_index in range(4):
        paths.update(
            {
                (role, fold_index): f"folds/fold-{fold_index}/{name}"
                for role, name in _FOLD.items()
            }
        )
    for key, relative_path in paths.items():
        receipt = semantic_sha256({"role": key[0], "fold_index": key[1]})
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_file_bytes({"semantic_receipt_sha256": receipt})
        )
        runtimes[key] = bind_massive_adaptive_rl_source_authority_v1(
            role=key[0],
            fold_index=key[1],
            authority=_SyntheticRuntimeSource(receipt),
            source_data_qualified=True,
            runtime_source_replayed=True,
        )
    return runtimes


def _authorized_source_bundle(root: Path, experiment_id: str):
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=experiment_id
    )
    runtimes = _persist_synthetic_source_graph(root)
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
        runtime_sources=runtimes,
    )
    generic = load_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
    )
    authorized = authorize_massive_adaptive_rl_source_bundle_v1(
        source_bundle=generic,
        runtime_sources=runtimes,
    )
    return manifest, runtimes, authorized


def _allow_synthetic_domain_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "_DOMAIN_RUNTIME_TYPES",
        {
            role: _SyntheticRuntimeSource
            for role in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1
        },
    )
    monkeypatch.setattr(
        graph_module,
        "_DIRECT_DOMAIN_SPECIFICATIONS",
        {role: None for role in graph_module._DIRECT_DOMAIN_SPECIFICATIONS},
    )
    monkeypatch.setattr(
        graph_module,
        "_validate_runtime_graph_contract",
        lambda **_kwargs: (
            (("synthetic-runtime-coverage",),),
            (("synthetic-runtime-edge", "test", "test"),),
        ),
    )


def _unchecked_bound(
    *, role: str, fold_index: int | None, authority: object
) -> MassiveAdaptiveRLRoleBoundSourceAuthorityV1:
    spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[role]
    receipt = getattr(
        authority,
        "semantic_receipt_sha256",
        getattr(authority, "receipt_sha256", semantic_sha256((role, fold_index))),
    )
    return MassiveAdaptiveRLRoleBoundSourceAuthorityV1(
        role=role,
        fold_index=fold_index,
        runtime_schema=spec.runtime_schema,
        role_specification_sha256=spec.specification_sha256,
        semantic_receipt_sha256=receipt,
        source_data_qualified=True,
        runtime_source_replayed=True,
        authority=authority,  # type: ignore[arg-type]
    )


def _unchecked_inventory(
    *, role: str, fold_index: int | None, items: tuple[object, ...]
) -> MassiveAdaptiveRLTypedAuthorityInventoryV1:
    receipt = semantic_sha256(
        (
            role,
            fold_index,
            tuple(
                getattr(
                    item,
                    "semantic_receipt_sha256",
                    getattr(item, "receipt_sha256", ""),
                )
                for item in items
            ),
        )
    )
    return MassiveAdaptiveRLTypedAuthorityInventoryV1(
        role=role,
        fold_index=fold_index,
        runtime_schema=MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[role].runtime_schema,
        item_type_name="test-contract-item",
        item_implementation_source_sha256=semantic_sha256("test-implementation"),
        item_schema="test-contract-item-v1",
        item_specification_sha256=None,
        item_bindings=(("test", receipt),),
        item_logical_keys=("test",),
        item_receipts=(receipt,),
        semantic_receipt_sha256=receipt,
        runtime_items=items,  # type: ignore[arg-type]
        runtime_source_replayed=True,
        source_data_qualified=True,
    )


def _runtime_contract_sources(
    *, block_sessions: int = 63
) -> tuple[dict[tuple[str, int | None], object], object]:
    sessions = _sessions()
    split_plan = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=tuple(
            session.session_date for session in sessions.sessions
        ),
        session_authority=sessions,
    )
    condition_receipt = semantic_sha256("graph-condition")
    identity_receipt = semantic_sha256("graph-identity")
    daily_receipt = semantic_sha256("graph-daily")
    partition_date = sessions.sessions[0].session_date
    direct = {
        "session-authority": sessions,
        "condition-authority": SimpleNamespace(receipt_sha256=condition_receipt),
        "identity-authority": SimpleNamespace(receipt_sha256=identity_receipt),
        "economic-event-archive": SimpleNamespace(
            identity_authority_receipt_sha256=identity_receipt,
            semantic_receipt_sha256=semantic_sha256("graph-events"),
        ),
        "daily-input-authority": SimpleNamespace(
            session_authority_receipt_sha256=sessions.receipt_sha256,
            condition_authority_receipt_sha256=condition_receipt,
            semantic_receipt_sha256=daily_receipt,
            sessions=(SimpleNamespace(source_session_date=partition_date),),
        ),
        "fill-source-authority": SimpleNamespace(
            daily_input_authority_semantic_receipt_sha256=daily_receipt,
            session_authority_receipt_sha256=sessions.receipt_sha256,
            condition_authority_receipt_sha256=condition_receipt,
            semantic_receipt_sha256=semantic_sha256("graph-fills"),
        ),
        "split-plan": split_plan,
    }
    runtime_sources: dict[tuple[str, int | None], object] = {
        (role, None): _unchecked_bound(
            role=role,
            fold_index=None,
            authority=authority,
        )
        for role, authority in direct.items()
    }
    partition = SimpleNamespace(
        source_session_date=partition_date,
        identity_authority_receipt_sha256=identity_receipt,
        receipt_sha256=semantic_sha256("graph-partition"),
    )
    partition_inventory = _unchecked_inventory(
        role="persisted-partition-inventory",
        fold_index=None,
        items=(partition,),
    )
    runtime_sources[("persisted-partition-inventory", None)] = _unchecked_bound(
        role="persisted-partition-inventory",
        fold_index=None,
        authority=partition_inventory,
    )

    blocks_per_source_fold = 126 // block_sessions
    for outer_fold_index in range(4):
        fit_count = 126 * (outer_fold_index + 1)
        expected_dates = split_plan.outer_folds[outer_fold_index].fit_session_dates[
            -fit_count:
        ]
        decisions = tuple(
            SimpleNamespace(
                decision_session_date=session_date,
                semantic_receipt_sha256=semantic_sha256(
                    ("decision", outer_fold_index, session_date)
                ),
                session_authority_receipt_sha256=sessions.receipt_sha256,
                context_origin_receipt_sha256=semantic_sha256(
                    ("context", outer_fold_index, session_date)
                ),
            )
            for session_date in expected_dates
        )
        contexts = tuple(
            SimpleNamespace(
                decision_session_date=session_date,
                semantic_receipt_sha256=semantic_sha256(
                    ("context", outer_fold_index, session_date)
                ),
                session_authority_receipt_sha256=sessions.receipt_sha256,
                identity_authority_receipt_sha256=identity_receipt,
            )
            for session_date in expected_dates
        )
        decision_by_date = {
            decision.decision_session_date: decision for decision in decisions
        }
        full_decision_inventory = semantic_sha256(
            tuple(
                decision_by_date[session_date].semantic_receipt_sha256
                for session_date in expected_dates
            )
        )
        windows = []
        checkpoints = []
        calibrations = []
        lineage = {}
        for source_fold_index in range(outer_fold_index + 1):
            first_block_index = source_fold_index * blocks_per_source_fold
            first_date = expected_dates[first_block_index * block_sessions]
            candidate_index = split_plan.candidate_session_dates.index(first_date)
            cutoff = split_plan.candidate_session_dates[candidate_index - 1]
            window_receipt = semantic_sha256(
                ("window", outer_fold_index, source_fold_index)
            )
            checkpoint_receipt = semantic_sha256(
                ("checkpoint", outer_fold_index, source_fold_index)
            )
            checkpoint_source_receipt = semantic_sha256(
                ("checkpoint-source", outer_fold_index, source_fold_index)
            )
            model_state_receipt = semantic_sha256(
                ("model-state", outer_fold_index, source_fold_index)
            )
            windows.append(
                SimpleNamespace(
                    fold_index=source_fold_index,
                    split_role="training",
                    rows=(SimpleNamespace(origin_session_date=cutoff),),
                    split_plan_receipt_sha256=(split_plan.semantic_receipt_sha256),
                    semantic_receipt_sha256=window_receipt,
                )
            )
            checkpoints.append(
                SimpleNamespace(
                    fold_index=source_fold_index,
                    selection_cutoff_session_date=cutoff,
                    training_window_plan_receipt_sha256=window_receipt,
                    selected_checkpoint_receipt_sha256=checkpoint_receipt,
                    selected_checkpoint_source_receipt_sha256=(
                        checkpoint_source_receipt
                    ),
                    selected_model_state_receipt_sha256=model_state_receipt,
                    semantic_receipt_sha256=semantic_sha256(
                        ("choice", outer_fold_index, source_fold_index)
                    ),
                )
            )
            calibrations.append(
                SimpleNamespace(
                    fold_index=source_fold_index,
                    calibration_fit_stop_session_date=cutoff,
                    training_window_plan_receipt_sha256=window_receipt,
                    checkpoint_receipt_sha256=checkpoint_receipt,
                    checkpoint_source_receipt_sha256=checkpoint_source_receipt,
                    model_state_receipt_sha256=model_state_receipt,
                    semantic_receipt_sha256=semantic_sha256(
                        ("calibration", outer_fold_index, source_fold_index)
                    ),
                )
            )
            lineage[source_fold_index] = (
                cutoff,
                window_receipt,
                checkpoint_receipt,
                checkpoint_source_receipt,
                model_state_receipt,
            )

        archives = []
        for block_index in range(fit_count // block_sessions):
            source_fold_index = block_index // blocks_per_source_fold
            start = block_index * block_sessions
            origin_dates = expected_dates[start : start + block_sessions]
            cutoff, window, checkpoint, checkpoint_source, model_state = lineage[
                source_fold_index
            ]
            archives.append(
                SimpleNamespace(
                    outer_fold_index=outer_fold_index,
                    source_fold_index=source_fold_index,
                    block_index=block_index,
                    block_sessions=block_sessions,
                    origin_session_dates=origin_dates,
                    training_window_plan_receipt_sha256=window,
                    checkpoint_receipt_sha256=checkpoint,
                    checkpoint_source_receipt_sha256=checkpoint_source,
                    model_state_receipt_sha256=model_state,
                    supervised_training_cutoff_session_date=cutoff,
                    split_plan_receipt_sha256=split_plan.semantic_receipt_sha256,
                    inference_full_decision_root_inventory_sha256=(
                        full_decision_inventory
                    ),
                    inference_origin_decision_root_inventory_sha256=semantic_sha256(
                        tuple(
                            decision_by_date[session_date].semantic_receipt_sha256
                            for session_date in origin_dates
                        )
                    ),
                    semantic_receipt_sha256=semantic_sha256(
                        ("archive", outer_fold_index, block_index)
                    ),
                )
            )

        fold_items = {
            "training-window-inventory": tuple(windows),
            "supervised-checkpoint-inventory": tuple(checkpoints),
            "calibration-inventory": tuple(calibrations),
            "fit-forecast-archive-inventory": tuple(archives),
            "decision-root-inventory": decisions,
            "context-origin-inventory": contexts,
        }
        for role, items in fold_items.items():
            inventory = _unchecked_inventory(
                role=role,
                fold_index=outer_fold_index,
                items=items,
            )
            runtime_sources[(role, outer_fold_index)] = _unchecked_bound(
                role=role,
                fold_index=outer_fold_index,
                authority=inventory,
            )
    return runtime_sources, split_plan


def test_runtime_source_graph_generic_reload_is_nonauthorizing_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-replay"
    )

    committed = materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    assert not committed.persisted_graph_replayed
    assert not committed.runtime_graph_replayed
    assert not committed.source_data_qualified

    generic_source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
    )
    generic = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=generic_source_bundle,
    )
    assert generic.persisted_graph_replayed
    assert not generic.runtime_graph_replayed
    assert not generic.source_data_qualified
    assert generic.runtime_authority_receipt_sha256 is None
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="concrete replay witness",
    ):
        generic.runtime_authority(role="session-authority", fold_index=None)

    forged = replace(
        generic,
        runtime_graph_replayed=True,
        source_data_qualified=True,
    )
    with pytest.raises(MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error):
        forged.validate()

    authorized = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic,
        source_bundle=generic_source_bundle,
        runtime_sources=runtimes,
    )
    assert authorized.runtime_graph_replayed
    assert authorized.source_data_qualified
    assert authorized.prequential_block_sessions == 63
    assert authorized.fold_fit_session_counts == (126, 252, 378, 504)
    assert (
        authorized.fold_candidate_schedule_receipts
        == manifest.base_manifest.fold_candidate_schedule_receipts
    )
    assert authorized.runtime_authority_receipt_sha256 is not None
    assert (
        authorized.runtime_authority_receipt_sha256
        != authorized.semantic_receipt_sha256
    )
    assert (
        authorized.runtime_authority(
            role="session-authority",
            fold_index=None,
        )
        is runtimes[("session-authority", None)].authority
    )
    assert (
        authorized.source_bundle_receipt_sha256 == source_bundle.semantic_receipt_sha256
    )
    assert len(authorized.rows) == 32
    assert not authorized.profitability_reporting_authorized
    assert not authorized.lockbox_access_authorized

    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    run = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        device="cpu",
        resume=False,
    )
    assert run.blocker_code == "runtime-source-graph-replay-required"
    assert (
        run.runtime_source_graph_authority_receipt_sha256
        == authorized.semantic_receipt_sha256
    )
    assert not run.runtime_source_graph_replayed

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="create-only",
    ):
        materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=source_bundle,
            runtime_sources=runtimes,
        )

    graph_path = runtime_source_graph_authority_path_v1(
        source_root=tmp_path,
        experiment_id=manifest.experiment_id,
    )
    assert not tuple(graph_path.parent.glob(f".{graph_path.name}.*.tmp"))
    backing_path = graph_path.with_name("runtime-source-graph-backing.json")
    graph_path.rename(backing_path)
    graph_path.symlink_to(backing_path.name)
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="symlink",
    ):
        load_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=generic_source_bundle,
        )


def test_runtime_source_graph_rejects_old_arbitrary_role_wrappers(
    tmp_path: Path,
) -> None:
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-domain-types"
    )

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="concrete domain type",
    ):
        materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=source_bundle,
            runtime_sources=runtimes,
        )


def test_typed_inventory_rejects_duplicate_logical_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = "persisted-partition-inventory"
    monkeypatch.setitem(
        graph_module._DOMAIN_INVENTORY_ITEM_TYPES,
        role,
        _SyntheticPartition,
    )
    monkeypatch.setitem(
        graph_module._DOMAIN_INVENTORY_ITEM_SCHEMAS,
        role,
        _SyntheticPartition.schema,
    )
    monkeypatch.setitem(
        graph_module._DOMAIN_INVENTORY_ITEM_SPECIFICATIONS,
        role,
        _SyntheticPartition.partition_spec_sha256,
    )
    first = _SyntheticPartition(
        source_session_date="2020-01-02",
        receipt_sha256=semantic_sha256("first"),
    )
    duplicate = _SyntheticPartition(
        source_session_date=first.source_session_date,
        receipt_sha256=semantic_sha256("duplicate"),
    )
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="identity or replay",
    ):
        build_massive_adaptive_rl_typed_authority_inventory_v1(
            role=role,
            fold_index=None,
            items=(first, duplicate),
        )

    second = _SyntheticPartition(
        source_session_date="2020-01-03",
        receipt_sha256=semantic_sha256("second"),
    )
    valid = build_massive_adaptive_rl_typed_authority_inventory_v1(
        role=role,
        fold_index=None,
        items=(first, second),
    )
    swapped = replace(
        valid,
        item_bindings=(
            (valid.item_bindings[0][0], valid.item_bindings[1][1]),
            (valid.item_bindings[1][0], valid.item_bindings[0][1]),
        ),
        semantic_receipt_sha256="0" * 64,
    )
    swapped = replace(
        swapped,
        semantic_receipt_sha256=semantic_sha256(swapped.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="key bindings",
    ):
        swapped.validate()


def test_typed_inventory_accepts_accumulated_source_fold_lineage(
    tmp_path: Path,
) -> None:
    checkpoint, window, tensor, _roots, _plan, _split_plan, _model_spec = (
        _rl_fit_fixture(
            tmp_path,
            outer_fold_index=0,
            block_index=0,
            block_sessions=21,
        )
    )
    choice = build_massive_adaptive_causal_checkpoint_choice_v1(
        checkpoints=(checkpoint,),
        training_window_plan=window,
    )
    choice = replace(
        choice,
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    choice = replace(
        choice,
        semantic_receipt_sha256=semantic_sha256(choice.semantic_unsigned()),
    )
    calibration = _calibration_v2(tensor.security_ids).calibration
    calibration = replace(
        calibration,
        fold_index=0,
        checkpoint_receipt_sha256=choice.selected_checkpoint_receipt_sha256,
        checkpoint_source_receipt_sha256=(
            choice.selected_checkpoint_source_receipt_sha256
        ),
        model_state_receipt_sha256=choice.selected_model_state_receipt_sha256,
        training_window_plan_receipt_sha256=window.semantic_receipt_sha256,
        calibration_fit_stop_session_date=choice.selection_cutoff_session_date,
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    calibration = replace(
        calibration,
        semantic_receipt_sha256=semantic_sha256(calibration.semantic_unsigned()),
    )
    calibration.validate()

    second_window = replace(
        window,
        fold_index=1,
        semantic_receipt_sha256="0" * 64,
    )
    second_window = replace(
        second_window,
        semantic_receipt_sha256=semantic_sha256(second_window.semantic_unsigned()),
    )
    second_choice = replace(
        choice,
        fold_index=1,
        training_window_plan_receipt_sha256=(second_window.semantic_receipt_sha256),
        semantic_receipt_sha256="0" * 64,
    )
    second_choice = replace(
        second_choice,
        semantic_receipt_sha256=semantic_sha256(second_choice.semantic_unsigned()),
    )
    second_calibration = replace(
        calibration,
        fold_index=1,
        training_window_plan_receipt_sha256=(second_window.semantic_receipt_sha256),
        semantic_receipt_sha256="0" * 64,
    )
    second_calibration = replace(
        second_calibration,
        semantic_receipt_sha256=semantic_sha256(second_calibration.semantic_unsigned()),
    )

    for role, items in (
        ("training-window-inventory", (window, second_window)),
        ("supervised-checkpoint-inventory", (choice, second_choice)),
        ("calibration-inventory", (calibration, second_calibration)),
    ):
        inventory = build_massive_adaptive_rl_typed_authority_inventory_v1(
            role=role,
            fold_index=1,
            items=items,
        )
        assert inventory.runtime_source_replayed
        assert tuple(item.fold_index for item in items) == (0, 1)


def test_runtime_graph_contract_accepts_all_accumulated_source_folds() -> None:
    runtime_sources, _split_plan = _runtime_contract_sources(block_sessions=63)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runtime-contract-fold-lineage",
        prequential_block_sessions=63,
    )

    coverage, edges = graph_module._validate_runtime_graph_contract(
        runtime_sources=runtime_sources,  # type: ignore[arg-type]
        prequential_block_sessions=manifest.base_manifest.prequential_block_sessions,
        fold_fit_session_counts=tuple(
            manifest.base_manifest.schedule(fold_index).rl_fit_session_count
            for fold_index in manifest.base_manifest.fold_indices
        ),
        fold_candidate_schedule_receipts=(
            manifest.base_manifest.fold_candidate_schedule_receipts
        ),
    )

    fold_one_inventory = runtime_sources[
        ("fit-forecast-archive-inventory", 1)
    ].authority
    fold_three_inventory = runtime_sources[
        ("fit-forecast-archive-inventory", 3)
    ].authority
    assert tuple(
        archive.source_fold_index for archive in fold_one_inventory.runtime_items
    ) == (0, 0, 1, 1)
    assert tuple(
        archive.source_fold_index for archive in fold_three_inventory.runtime_items
    ) == (0, 0, 1, 1, 2, 2, 3, 3)
    assert any(row[0] == "rl-fit-fold" and row[1] == 3 for row in coverage)
    assert any(edge[0] == "fit-forecast/training-window/3/7/3" for edge in edges)


def test_runtime_graph_contract_rejects_manifest_block_size_mismatch() -> None:
    runtime_sources, _split_plan = _runtime_contract_sources(block_sessions=63)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runtime-contract-block-size",
        prequential_block_sessions=21,
    )

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="block size differs from Manifest V3",
    ):
        graph_module._validate_runtime_graph_contract(
            runtime_sources=runtime_sources,  # type: ignore[arg-type]
            prequential_block_sessions=(
                manifest.base_manifest.prequential_block_sessions
            ),
            fold_fit_session_counts=tuple(
                manifest.base_manifest.schedule(fold_index).rl_fit_session_count
                for fold_index in manifest.base_manifest.fold_indices
            ),
            fold_candidate_schedule_receipts=(
                manifest.base_manifest.fold_candidate_schedule_receipts
            ),
        )


@pytest.mark.parametrize(
    ("role", "message"),
    (
        ("supervised-checkpoint-inventory", "checkpoint"),
        ("calibration-inventory", "calibration"),
    ),
)
def test_runtime_graph_contract_rejects_duplicate_training_lineage(
    role: str,
    message: str,
) -> None:
    runtime_sources, _split_plan = _runtime_contract_sources(block_sessions=63)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=f"runtime-contract-duplicate-{message}",
        prequential_block_sessions=63,
    )
    bound = runtime_sources[(role, 1)]
    inventory = bound.authority
    runtime_sources[(role, 1)] = replace(
        bound,
        authority=replace(
            inventory,
            runtime_items=inventory.runtime_items + (inventory.runtime_items[0],),
        ),
    )

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match=rf"{message} source-fold lineage is duplicated",
    ):
        graph_module._validate_runtime_graph_contract(
            runtime_sources=runtime_sources,  # type: ignore[arg-type]
            prequential_block_sessions=(
                manifest.base_manifest.prequential_block_sessions
            ),
            fold_fit_session_counts=tuple(
                manifest.base_manifest.schedule(fold_index).rl_fit_session_count
                for fold_index in manifest.base_manifest.fold_indices
            ),
            fold_candidate_schedule_receipts=(
                manifest.base_manifest.fold_candidate_schedule_receipts
            ),
        )


def test_runtime_graph_contract_rejects_wrong_archive_source_fold() -> None:
    runtime_sources, _split_plan = _runtime_contract_sources(block_sessions=63)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runtime-contract-wrong-source-fold",
        prequential_block_sessions=63,
    )
    bound = runtime_sources[("fit-forecast-archive-inventory", 1)]
    inventory = bound.authority
    archives = list(inventory.runtime_items)
    changed_archive = vars(archives[2]).copy()
    changed_archive["source_fold_index"] = 0
    archives[2] = SimpleNamespace(**changed_archive)
    runtime_sources[("fit-forecast-archive-inventory", 1)] = replace(
        bound,
        authority=replace(inventory, runtime_items=tuple(archives)),
    )

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="does not cover the exact fit prefix",
    ):
        graph_module._validate_runtime_graph_contract(
            runtime_sources=runtime_sources,  # type: ignore[arg-type]
            prequential_block_sessions=(
                manifest.base_manifest.prequential_block_sessions
            ),
            fold_fit_session_counts=tuple(
                manifest.base_manifest.schedule(fold_index).rl_fit_session_count
                for fold_index in manifest.base_manifest.fold_indices
            ),
            fold_candidate_schedule_receipts=(
                manifest.base_manifest.fold_candidate_schedule_receipts
            ),
        )


def test_runtime_source_graph_rejects_incomplete_and_changed_runtime_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-tamper"
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    generic_source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
    )
    generic = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=generic_source_bundle,
    )

    incomplete = dict(runtimes)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="does not replay",
    ):
        authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            authority=generic,
            source_bundle=generic_source_bundle,
            runtime_sources=incomplete,
        )

    changed = dict(runtimes)
    key = next(iter(changed))
    changed_receipt = semantic_sha256({"changed": True})
    changed[key] = bind_massive_adaptive_rl_source_authority_v1(
        role=key[0],
        fold_index=key[1],
        authority=_SyntheticRuntimeSource(changed_receipt),
        source_data_qualified=True,
        runtime_source_replayed=True,
    )
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="replay",
    ):
        authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            authority=generic,
            source_bundle=generic_source_bundle,
            runtime_sources=changed,
        )
