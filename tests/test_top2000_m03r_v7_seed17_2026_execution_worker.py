from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rl_quant.evaluation.top2000_m03r_v7_2026_execution import (
    Top2000M03RV72026ExecutionArtifactBinding,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SETTING_IDS,
    runtime_setting_id,
)
from rl_quant.workflows import (
    top2000_m03r_v7_seed17_2026_execution as worker,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_2026_ytd import (
    Top2000M03RV7Seed172026YTDCheckpointBinding,
)


def _digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding_receipt(path: str) -> Any:
    payload = json.loads(Path(path).read_bytes())
    payload["setting_index_map"] = tuple(payload["setting_index_map"])
    payload["compute_capability"] = tuple(payload["compute_capability"])
    return worker.Top2000M03RV72026FoldArtifactBindingReceipt(**payload)


def _checkpoint(setting_index: int, fold_index: int) -> Any:
    setting_id = M03R_SEED17_TOP2000_SETTING_IDS[setting_index]
    training_completion_index = 2
    training_root = (
        f"completion-{training_completion_index:02d}-setting-{setting_index:02d}/"
        "training"
    )
    return Top2000M03RV7Seed172026YTDCheckpointBinding(
        completion_index=training_completion_index,
        setting_index=setting_index,
        setting_id=setting_id,
        runtime_setting_id=runtime_setting_id(setting_id),
        training_fold_index=fold_index,
        seed=17,
        writer_rank=0,
        optimizer_steps=64,
        checkpoint_role="headline" if fold_index == 5 else "cutoff-sensitivity",
        training_root_relative_path=training_root,
        model_relative_path=(
            f"{training_root}/cells/fold-{fold_index:02d}-seed-17/"
            "model.rank-00.pt"
        ),
        model_file_sha256=_digest(f"model-file-{fold_index}"),
        model_state_sha256=_digest(f"model-state-{fold_index}"),
        cell_receipt_relative_path=(
            f"{training_root}/receipts/fold-{fold_index:02d}-seed-17.json"
        ),
        cell_receipt_file_sha256=_digest(f"cell-{fold_index}"),
        seed_validation_receipt_relative_path=(
            f"{training_root}/receipts/seed-validation/"
            f"fold-{fold_index:02d}-seed-17.json"
        ),
        seed_validation_receipt_file_sha256=_digest(f"validation-{fold_index}"),
        fold_execution_receipt_relative_path=(
            f"{training_root}/receipts/fold-execution/fold-{fold_index:02d}.json"
        ),
        fold_execution_receipt_file_sha256=_digest(f"execution-{fold_index}"),
        completion_receipt_relative_path=(
            f"{training_root}/completion-receipt.json"
        ),
        completion_receipt_file_sha256=_digest("training-completion"),
        training_plan_file_sha256=_digest("training-plan-file"),
        training_plan_receipt_sha256=_digest("training-plan-receipt"),
        source_protocol_sha256=M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    )


class _Fixture:
    def __init__(self, tmp_path: Path, *, setting_index: int = 6) -> None:
        self.setting_index = setting_index
        self.plan_file_sha256 = _digest("plan-file")
        self.plan_receipt_sha256 = _digest("plan-receipt")
        self.source_inventory_sha256 = _digest("source-inventory")
        self.pre_cache_sha256 = _digest("pre-cache")
        self.retrospective_cache = tmp_path / "retrospective.pt"
        self.retrospective_cache.write_bytes(b"retrospective-cache")
        self.retrospective_cache_sha256 = _file_sha256(self.retrospective_cache)
        self.plan_path = tmp_path / "frozen-plan.json"
        self.plan_path.write_text("frozen", encoding="utf-8")
        self.training_root = tmp_path / "completed-training"
        self.training_root.mkdir()
        self.source_root = tmp_path / "source"
        self.source_root.mkdir()
        self.output_root = tmp_path / "evaluation-output"
        self.checkpoints = tuple(
            _checkpoint(setting_index, fold_index) for fold_index in range(6)
        )
        self.plan = SimpleNamespace(
            checkpoints=self.checkpoints,
            evaluation_source_sha256=self.source_inventory_sha256,
            evaluation_source=SimpleNamespace(source_root=str(self.source_root)),
            pre2026_cache=SimpleNamespace(
                cache_file_sha256=self.pre_cache_sha256,
                cache_path=str(tmp_path / "pre-cache.pt"),
            ),
            source_training_output_root=str(self.training_root),
        )
        self.session = SimpleNamespace(name="resident-session")
        self.prepared: list[dict[str, Any]] = []
        self.executed: list[tuple[int, object]] = []
        self.loaded: dict[int, Any] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            worker,
            "load_top2000_m03r_v7_seed17_2026_ytd_plan",
            lambda *args, **kwargs: self.plan,
        )

        def _prepare(**kwargs: Any) -> Any:
            self.prepared.append(kwargs)
            return self.session

        monkeypatch.setattr(
            worker,
            "prepare_top2000_m03r_v7_seed17_2026_execution_session",
            _prepare,
        )

        def _run(
            session: object,
            checkpoint: Any,
            *,
            training_output_root: Path,
            output_path: Path,
        ) -> Any:
            assert training_output_root == self.training_root.resolve()
            self.executed.append((checkpoint.training_fold_index, session))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(
                f"artifact-{checkpoint.training_fold_index}".encode()
            )
            artifact_sha256 = _file_sha256(output_path)
            execution_receipt_sha256 = _digest(
                f"execution-receipt-{checkpoint.training_fold_index}"
            )
            self.loaded[checkpoint.training_fold_index] = self._loaded(checkpoint)
            return Top2000M03RV72026ExecutionArtifactBinding(
                artifact_path=str(output_path),
                artifact_file_sha256=artifact_sha256,
                execution_receipt_sha256=execution_receipt_sha256,
                setting_index=self.setting_index,
                training_fold_index=checkpoint.training_fold_index,
            )

        monkeypatch.setattr(
            worker,
            "run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session",
            _run,
        )

        def _load(path: str | Path, **kwargs: Any) -> Any:
            fold_index = int(Path(path).name.split("-")[1].split(".")[0])
            assert kwargs["expected_file_sha256"] == _file_sha256(Path(path))
            return self.loaded.setdefault(
                fold_index,
                self._loaded(self.checkpoints[fold_index]),
            )

        monkeypatch.setattr(
            worker,
            "load_top2000_m03r_v7_seed17_2026_execution_artifact",
            _load,
        )

    def _loaded(self, checkpoint: Any) -> Any:
        fold_index = checkpoint.training_fold_index
        execution_receipt_sha256 = _digest(f"execution-receipt-{fold_index}")
        execution = SimpleNamespace(
            receipt_sha256=execution_receipt_sha256,
            policy_model_state_sha256_before=checkpoint.model_state_sha256,
            policy_model_state_sha256_after=checkpoint.model_state_sha256,
            elapsed_wall_seconds=float(fold_index + 1),
        )
        return SimpleNamespace(
            execution_receipt=execution,
            cuda_proof=SimpleNamespace(
                visible_cuda_device_count=1,
                gpu_name="NVIDIA H100 80GB HBM3",
                gpu_total_memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                peak_allocated_bytes=1000 + fold_index,
                peak_reserved_bytes=2000 + fold_index,
                allocator_oom_count_delta=0,
                allocator_retry_count_delta=0,
            ),
            chronology_identity=SimpleNamespace(
                receipt_sha256=_digest("chronology")
            ),
            economic_execution_receipt=SimpleNamespace(
                pre2026_cache_sha256=self.pre_cache_sha256
            ),
            checkpoint_load_receipt=SimpleNamespace(
                frozen_checkpoint_binding_sha256=checkpoint.receipt_sha256,
                model_file_sha256=checkpoint.model_file_sha256,
                model_state_sha256=checkpoint.model_state_sha256,
            ),
        )

    def run(
        self,
        *,
        setting_index_map: tuple[int, ...] = tuple(range(12)),
    ) -> Any:
        local_index = setting_index_map.index(self.setting_index)
        return worker.run_top2000_m03r_v7_seed17_2026_setting_worker(
            plan_path=self.plan_path,
            expected_plan_file_sha256=self.plan_file_sha256,
            expected_plan_receipt_sha256=self.plan_receipt_sha256,
            expected_execution_source_inventory_sha256=(
                self.source_inventory_sha256
            ),
            retrospective_cache_path=self.retrospective_cache,
            expected_retrospective_cache_file_sha256=(
                self.retrospective_cache_sha256
            ),
            output_root=self.output_root,
            completion_index=local_index,
            setting_index_map=setting_index_map,
            environment={},
        )


def test_direct_index_one_session_fold_order_and_retry_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Fixture(tmp_path)
    fixture.install(monkeypatch)

    completed = fixture.run()

    assert [fold for fold, _session in fixture.executed] == [5, 0, 1, 2, 3, 4]
    assert {id(session) for _fold, session in fixture.executed} == {
        id(fixture.session)
    }
    assert len(fixture.prepared) == 1
    assert fixture.prepared[0]["device"] == "cuda:0"
    assert completed.receipt.local_completion_index == fixture.setting_index
    assert completed.receipt.setting_index == fixture.setting_index
    assert completed.receipt.setting_index_map == tuple(range(12))
    assert {
        row.training_completion_index
        for row in (
            _binding_receipt(item.artifact_binding_path)
            for item in completed.receipt.fold_artifacts
        )
    } == {2}
    assert completed.receipt.maximum_peak_allocated_bytes == 1005
    assert completed.receipt.maximum_peak_reserved_bytes == 2005
    assert not completed.receipt.panel_aggregation_performed
    assert not completed.receipt.policy_training_authorized

    fixture.prepared.clear()
    fixture.executed.clear()
    retried = fixture.run()
    assert retried == completed
    assert fixture.prepared == []
    assert fixture.executed == []


def test_partial_valid_retry_runs_only_wholly_absent_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Fixture(tmp_path)
    fixture.install(monkeypatch)
    first = fixture.run()
    Path(first.completion_path).unlink()
    missing = first.receipt.fold_artifacts[3]
    Path(missing.artifact_binding_path).unlink()
    Path(missing.artifact_path).unlink()
    fixture.prepared.clear()
    fixture.executed.clear()

    resumed = fixture.run()

    assert [fold for fold, _session in fixture.executed] == [2]
    assert len(fixture.prepared) == 1
    assert resumed.receipt.fold_execution_order == (5, 0, 1, 2, 3, 4)


def test_orphaned_or_tampered_fold_fails_before_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Fixture(tmp_path)
    fixture.install(monkeypatch)
    artifact, _binding = worker._artifact_paths(
        fixture.output_root.resolve(), fixture.setting_index, 5
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"orphan")

    with pytest.raises(
        worker.Top2000M03RV72026SettingWorkerError,
        match="partial artifact/binding pair",
    ):
        fixture.run()
    assert fixture.prepared == []

    artifact.unlink()
    completed = fixture.run()
    binding_path = Path(completed.receipt.fold_artifacts[0].artifact_binding_path)
    payload = json.loads(binding_path.read_bytes())
    payload["execution_source_inventory_sha256"] = _digest("tampered-source")
    binding_path.chmod(0o644)
    binding_path.write_bytes(worker._canonical_json(payload))

    with pytest.raises(
        worker.Top2000M03RV72026SettingWorkerError,
        match="does not match the frozen worker inputs",
    ):
        fixture.run()


def test_public_completion_loader_requires_external_plan_source_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Fixture(tmp_path)
    fixture.install(monkeypatch)
    completed = fixture.run()

    restored = worker.load_top2000_m03r_v7_seed17_2026_setting_completion(
        completed.completion_path,
        expected_completion_file_sha256=completed.completion_file_sha256,
        plan_path=fixture.plan_path,
        expected_plan_file_sha256=fixture.plan_file_sha256,
        expected_plan_receipt_sha256=fixture.plan_receipt_sha256,
        expected_execution_source_inventory_sha256=(
            fixture.source_inventory_sha256
        ),
        expected_retrospective_cache_file_sha256=(
            fixture.retrospective_cache_sha256
        ),
        output_root=fixture.output_root,
        completion_index=fixture.setting_index,
        environment={},
    )
    assert restored == completed

    with pytest.raises(
        worker.Top2000M03RV72026SettingWorkerError,
        match="completion file SHA-256 drifted",
    ):
        worker.load_top2000_m03r_v7_seed17_2026_setting_completion(
            completed.completion_path,
            expected_completion_file_sha256=_digest("wrong-completion"),
            plan_path=fixture.plan_path,
            expected_plan_file_sha256=fixture.plan_file_sha256,
            expected_plan_receipt_sha256=fixture.plan_receipt_sha256,
            expected_execution_source_inventory_sha256=(
                fixture.source_inventory_sha256
            ),
            expected_retrospective_cache_file_sha256=(
                fixture.retrospective_cache_sha256
            ),
            output_root=fixture.output_root,
            completion_index=fixture.setting_index,
            environment={},
        )


@pytest.mark.parametrize("value", [0, 5, 11])
def test_completion_index_maps_directly_to_setting(value: int) -> None:
    resolution = worker.resolve_top2000_m03r_v7_2026_setting_index(
        environment={"JOB_COMPLETION_INDEX": str(value)}
    )
    assert resolution.local_completion_index == value
    assert resolution.setting_index == value
    assert resolution.setting_index_map == tuple(range(12))


def test_remainder_map_resolves_local_zero_to_setting_one() -> None:
    resolution = worker.resolve_top2000_m03r_v7_2026_setting_index(
        environment={"JOB_COMPLETION_INDEX": "0"},
        setting_index_map=tuple(range(1, 12)),
    )
    assert resolution.local_completion_index == 0
    assert resolution.setting_index == 1
    assert resolution.setting_index_map_sha256 == worker._setting_index_map_sha256(
        tuple(range(1, 12))
    )


def test_remainder_worker_receipt_binds_local_and_scientific_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _Fixture(tmp_path, setting_index=1)
    fixture.install(monkeypatch)

    completed = fixture.run(setting_index_map=tuple(range(1, 12)))

    assert completed.receipt.local_completion_index == 0
    assert completed.receipt.setting_index == 1
    assert completed.receipt.setting_index_map == tuple(range(1, 12))


def test_completion_index_rejects_disagreement_and_out_of_range() -> None:
    with pytest.raises(
        worker.Top2000M03RV72026SettingWorkerError,
        match="disagrees",
    ):
        worker.resolve_top2000_m03r_v7_2026_setting_index(
            4, environment={"JOB_COMPLETION_INDEX": "5"}
        )
    with pytest.raises(
        worker.Top2000M03RV72026SettingWorkerError,
        match="outside the explicit",
    ):
        worker.resolve_top2000_m03r_v7_2026_setting_index(
            environment={"JOB_COMPLETION_INDEX": "12"}
        )


def test_completion_receipt_is_exact_nonpromotable_inventory(tmp_path: Path) -> None:
    rows = tuple(
        worker.Top2000M03RV72026SettingFoldCompletion(
            training_fold_index=fold,
            artifact_binding_path=str(tmp_path / f"binding-{fold}"),
            artifact_binding_file_sha256=_digest(f"binding-file-{fold}"),
            artifact_binding_receipt_sha256=_digest(f"binding-receipt-{fold}"),
            artifact_path=str(tmp_path / f"artifact-{fold}"),
            artifact_file_sha256=_digest(f"artifact-file-{fold}"),
            execution_receipt_sha256=_digest(f"execution-{fold}"),
            elapsed_wall_seconds=float(fold + 1),
            visible_cuda_device_count=1,
            gpu_name="NVIDIA H100 80GB HBM3",
            gpu_total_memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            peak_allocated_bytes=100 + fold,
            peak_reserved_bytes=200 + fold,
        )
        for fold in worker.TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER
    )
    receipt = worker.Top2000M03RV72026SettingCompletionReceipt(
        local_completion_index=0,
        setting_index_map=tuple(range(12)),
        setting_index_map_sha256=worker._setting_index_map_sha256(tuple(range(12))),
        setting_index=0,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        runtime_setting_id=runtime_setting_id(M03R_SEED17_TOP2000_SETTING_IDS[0]),
        frozen_plan_path=str(tmp_path / "plan"),
        frozen_plan_file_sha256=_digest("plan-file"),
        frozen_plan_receipt_sha256=_digest("plan-receipt"),
        execution_source_inventory_sha256=_digest("source"),
        pre2026_cache_file_sha256=_digest("pre-cache"),
        retrospective_cache_file_sha256=_digest("retro-cache"),
        retrospective_chronology_receipt_sha256=_digest("chronology"),
        fold_execution_order=worker.TOP2000_M03R_V7_2026_SETTING_FOLD_ORDER,
        fold_artifacts=rows,
        completed_fold_count=6,
        total_elapsed_wall_seconds=sum(row.elapsed_wall_seconds for row in rows),
        visible_cuda_device_count=1,
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_total_memory_bytes=80 * 1024**3,
        compute_capability=(9, 0),
        maximum_peak_allocated_bytes=max(row.peak_allocated_bytes for row in rows),
        maximum_peak_reserved_bytes=max(row.peak_reserved_bytes for row in rows),
    )
    assert asdict(receipt)["promotion_eligible"] is False
    assert receipt.receipt_sha256 == worker._sha256(asdict(receipt))
