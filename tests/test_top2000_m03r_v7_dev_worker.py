from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
    build_m03r_v7_top2000_package_plan,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_SEEDS,
    Top2000M03RV7DevelopmentTrainingPlan,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.workflows import top2000_m03r_v7_dev as worker


def _plan(
    tmp_path: Path,
    *,
    optimizer_steps: int = 2,
    setting_index: int = 0,
    activation_checkpointing: bool = False,
) -> Top2000M03RV7DevelopmentTrainingPlan:
    return Top2000M03RV7DevelopmentTrainingPlan(
        setting_index=setting_index,
        setting_id=M03R_TOP2000_DEV_SETTING_IDS[setting_index],
        cache_path=str(tmp_path / "cache.pt"),
        cache_sha256="a" * 64,
        output_root=str(tmp_path / "output"),
        total_optimizer_steps_per_fold_seed=optimizer_steps,
        token_dim=8,
        raw_stock_chunk=8,
        activation_checkpointing=activation_checkpointing,
    )


def test_plan_round_trip_is_content_pinned(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    file_sha256 = worker.render_training_plan(path, plan)
    loaded, loaded_sha256 = worker.load_training_plan(
        path,
        expected_sha256=file_sha256,
        expected_setting_index=0,
    )
    assert loaded == plan
    assert loaded_sha256 == file_sha256

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["learning_rate"] = 2.0e-4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(worker.Top2000M03RV7WorkerError, match="SHA-256 mismatch"):
        worker.load_training_plan(
            path,
            expected_sha256=file_sha256,
            expected_setting_index=0,
        )


def test_new_cell_applies_content_bound_activation_checkpointing(
    tmp_path: Path,
) -> None:
    retained_plan = _plan(tmp_path / "retained")
    retained, _, _ = worker._new_cell(
        retained_plan,
        seed=17,
        device=torch.device("cpu"),
    )
    checkpointed_plan = _plan(
        tmp_path / "checkpointed",
        activation_checkpointing=True,
    )
    checkpointed, _, _ = worker._new_cell(
        checkpointed_plan,
        seed=17,
        device=torch.device("cpu"),
    )

    assert retained_plan.receipt_sha256 != checkpointed_plan.receipt_sha256
    assert not retained.core.config.grad_checkpoint
    assert not retained.core.raw_encoder.grad_checkpoint
    assert checkpointed.core.config.grad_checkpoint
    assert checkpointed.core.raw_encoder.grad_checkpoint


def test_episode_schedule_is_deterministic_bounded_and_step_sensitive() -> None:
    folds = render_top2000_m03r_v7_development_folds(1001)
    first_fold = folds[0]
    assert worker.deterministic_episode_start(
        episode_schedule_sha256="b" * 64,
        fold=first_fold,
        seed=TOP2000_M03R_V7_DEV_SEEDS[0],
        optimizer_step=0,
    ) == 0
    fold = folds[-1]
    values = [
        worker.deterministic_episode_start(
            episode_schedule_sha256="b" * 64,
            fold=fold,
            seed=TOP2000_M03R_V7_DEV_SEEDS[0],
            optimizer_step=step,
        )
        for step in range(8)
    ]
    assert values == [
        worker.deterministic_episode_start(
            episode_schedule_sha256="b" * 64,
            fold=fold,
            seed=TOP2000_M03R_V7_DEV_SEEDS[0],
            optimizer_step=step,
        )
        for step in range(8)
    ]
    assert len(set(values)) > 1
    assert all(
        fold.training_state_start
        <= value
        <= fold.training_state_stop_exclusive
        - TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
        for value in values
    )


def test_episode_schedule_is_paired_across_settings_and_run_paths(
    tmp_path: Path,
) -> None:
    left = _plan(tmp_path / "run-a", setting_index=0)
    right = _plan(tmp_path / "run-b", setting_index=1)
    assert left.receipt_sha256 != right.receipt_sha256
    assert left.episode_schedule_sha256 == right.episode_schedule_sha256

    for fold in render_top2000_m03r_v7_development_folds(1001):
        for seed in TOP2000_M03R_V7_DEV_SEEDS:
            left_starts = tuple(
                worker.deterministic_episode_start(
                    episode_schedule_sha256=left.episode_schedule_sha256,
                    fold=fold,
                    seed=seed,
                    optimizer_step=step,
                )
                for step in range(8)
            )
            right_starts = tuple(
                worker.deterministic_episode_start(
                    episode_schedule_sha256=right.episode_schedule_sha256,
                    fold=fold,
                    seed=seed,
                    optimizer_step=step,
                )
                for step in range(8)
            )
            assert left_starts == right_starts


def test_package_plan_uses_completion_env_without_shell_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "package-plan.json"
    artifacts = M03RV7Top2000ArtifactBindings(
        source_archive_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        dependency_lock_sha256="3" * 64,
        cache_artifact_sha256="4" * 64,
        cache_manifest_sha256="5" * 64,
        data_manifest_sha256="6" * 64,
        execution_model_sha256="7" * 64,
        image_reference=f"registry/research@sha256:{'8' * 64}",
        image_digest_sha256="8" * 64,
    )
    package = build_m03r_v7_top2000_package_plan(
        artifacts=artifacts,
        plan_artifact_path=str(package_path),
    )
    package_path.write_text(
        json.dumps(asdict(package), sort_keys=True),
        encoding="utf-8",
    )
    loaded = worker.load_package_plan(
        package_path,
        expected_package_plan_sha256=package.package_plan_sha256,
    )
    assert loaded == package
    with pytest.raises(worker.Top2000M03RV7WorkerError, match="admitted"):
        worker.load_package_plan(
            package_path,
            expected_package_plan_sha256="9" * 64,
        )

    monkeypatch.setenv("JOB_COMPLETION_INDEX", "3")
    assert worker.resolve_completion_index(None) == 3
    with pytest.raises(worker.Top2000M03RV7WorkerError, match="disagrees"):
        worker.resolve_completion_index(4)
    derived, binding_sha256 = worker.plan_from_package_completion(
        loaded,
        package_plan_path=package_path,
        completion_index=3,
        output_root=tmp_path / "outputs",
    )
    assert derived.setting_index == loaded.indices[3].setting_index
    assert derived.cache_path == str(tmp_path / "cache.pt")
    assert derived.total_optimizer_steps_per_fold_seed == loaded.runtime_profile.optimizer_steps_per_fold_seed
    assert derived.max_origin_batch == loaded.runtime_profile.max_origin_batch
    assert derived.learning_rate == loaded.runtime_profile.learning_rate
    assert derived.weight_decay == loaded.runtime_profile.weight_decay
    assert derived.grad_clip == loaded.runtime_profile.grad_clip
    assert derived.token_dim == loaded.runtime_profile.token_dim
    assert derived.raw_stock_chunk == loaded.runtime_profile.raw_stock_chunk
    assert derived.expected_world_size == loaded.runtime_profile.expected_world_size
    assert (
        derived.activation_checkpointing
        == loaded.runtime_profile.activation_checkpointing
    )
    assert derived.mixed_precision == loaded.runtime_profile.mixed_precision
    assert len(binding_sha256) == 64
    binding_path = Path(derived.output_root) / "execution-plan-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["episode_schedule_sha256"] == derived.episode_schedule_sha256


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def bind_episode_factor_loadings(self, value: torch.Tensor) -> None:
        assert value.shape == (2, 1)


@dataclass(frozen=True)
class _FakeBuilt:
    identity: Any


def test_qualification_resumes_exact_update_cursor_and_publishes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, optimizer_steps=2)
    plan_path = tmp_path / "plan.json"
    plan_file_sha256 = worker.render_training_plan(plan_path, plan)
    fake_cache = SimpleNamespace(
        daily_ohlcv=torch.zeros((1001, 2, 5)),
        cache_identity="c" * 64,
        search_identity="d" * 64,
        action_hash="e" * 64,
    )
    monkeypatch.setattr(
        worker,
        "load_verified_top2000_hold30_development_cache",
        lambda *_args, **_kwargs: fake_cache,
    )
    monkeypatch.setattr(
        worker,
        "_build_episode",
        lambda *_args, start, **_kwargs: (
            _FakeBuilt(SimpleNamespace(receipt_sha256=f"{start:064x}")),
            object(),
            SimpleNamespace(
                loadings=torch.zeros((2, 1)),
                receipt_sha256=f"{start + 1:064x}",
            ),
        ),
    )
    provider = SimpleNamespace(
        inputs=SimpleNamespace(daily_ohlcv=torch.zeros((1, 2, 2, 5)))
    )
    monkeypatch.setattr(
        worker,
        "bind_top2000_m03r_v7_runtime_sequence",
        lambda sequence, _policy: (sequence, provider),
    )
    monkeypatch.setattr(
        worker,
        "_new_cell",
        lambda _plan, *, seed, device: (
            _FakePolicy().to(device),
            torch.optim.AdamW(_FakePolicy().parameters()),
            None,
        ),
    )

    # Use one coherent fake model/optimizer pair despite the compact lambda
    # above being unsuitable for state restoration.
    def new_cell(
        _plan: Top2000M03RV7DevelopmentTrainingPlan,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[_FakePolicy, torch.optim.Optimizer, None]:
        del _plan, seed
        model = _FakePolicy().to(device)
        return model, torch.optim.AdamW(model.parameters(), lr=1.0e-3), None

    monkeypatch.setattr(worker, "_new_cell", new_cell)
    calls: list[int] = []
    fail_once = {"enabled": True}

    def train_update(
        policy: _FakePolicy,
        _sequence: object,
        _provider: object,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        step = int(kwargs["completed_optimizer_steps"])
        calls.append(step)
        if step == 1 and fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("injected restart boundary")
        optimizer.zero_grad(set_to_none=True)
        policy.weight.sum().backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return {"objective": float(step), "development_only": True}

    monkeypatch.setattr(
        worker,
        "train_top2000_m03r_v7_development_update",
        train_update,
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="injected restart boundary"):
        worker.run_worker(
            plan,
            plan_file_sha256=plan_file_sha256,
            qualification_only=True,
            qualification_steps=2,
        )
    assert calls == [0, 1]

    terminal = worker.run_worker(
        plan,
        plan_file_sha256=plan_file_sha256,
        qualification_only=True,
        qualification_steps=2,
    )
    assert calls == [0, 1, 1]
    assert terminal is not None
    assert terminal["complete"] is True
    assert terminal["completed_cells"] == 1
    receipt = Path(terminal["receipt_path"])
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == terminal["receipt_sha256"]

    # Emulate a crash after the cell receipt was committed but before its
    # cursor-advance manifest.  The prior slot still owns the exact step-2
    # state; resume must accept the immutable model/receipt without rewriting.
    run_root = Path(plan.output_root) / "qualification"
    manifest_path = run_root / "checkpoints" / "progress-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    step_two = run_root / "checkpoints" / "progress.slot-1.rank-00.pt"
    manifest["cell_index"] = 0
    manifest["completed_steps"] = 2
    manifest["rank_checkpoints"] = [
        {
            "rank": 0,
            "checkpoint": step_two.name,
            "checkpoint_sha256": hashlib.sha256(step_two.read_bytes()).hexdigest(),
        }
    ]
    worker._atomic_write_json(manifest_path, manifest)
    model_path = run_root / "cells" / "fold-00-seed-17" / "model.rank-00.pt"
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # A completed rerun is idempotent and performs no optimizer update.
    repeated = worker.run_worker(
        plan,
        plan_file_sha256=plan_file_sha256,
        qualification_only=True,
        qualification_steps=2,
    )
    assert calls == [0, 1, 1]
    assert repeated is not None
    assert repeated["receipt_sha256"] == terminal["receipt_sha256"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_sha256
    canonical_payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    assert canonical_payload["overlay_optimizer_required"] is False
    assert canonical_payload["overlay_optimizer_state_dict"] is None
    assert canonical_payload["overlay_optimizer_state_sha256"] is None
    assert worker.optimizer_state_dict_sha256(
        canonical_payload["alpha_core_optimizer_state_dict"]
    ) == canonical_payload["alpha_core_optimizer_state_sha256"]


def test_a06_interruption_restores_separately_bound_overlay_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting_index = M03R_TOP2000_DEV_SETTING_IDS.index(
        "A06-sharpe-overlay-top2000-dev-v1"
    )
    plan = _plan(
        tmp_path,
        optimizer_steps=2,
        setting_index=setting_index,
    )
    plan_path = tmp_path / "a06-plan.json"
    plan_file_sha256 = worker.render_training_plan(plan_path, plan)
    fake_cache = SimpleNamespace(
        daily_ohlcv=torch.zeros((1001, 2, 5)),
        cache_identity="c" * 64,
        search_identity="d" * 64,
        action_hash="e" * 64,
    )
    monkeypatch.setattr(
        worker,
        "load_verified_top2000_hold30_development_cache",
        lambda *_args, **_kwargs: fake_cache,
    )
    monkeypatch.setattr(
        worker,
        "_build_episode",
        lambda *_args, start, **_kwargs: (
            _FakeBuilt(SimpleNamespace(receipt_sha256=f"{start:064x}")),
            object(),
            SimpleNamespace(
                loadings=torch.zeros((2, 1)),
                receipt_sha256=f"{start + 1:064x}",
            ),
        ),
    )
    provider = SimpleNamespace(
        inputs=SimpleNamespace(daily_ohlcv=torch.zeros((1, 2, 2, 5)))
    )
    monkeypatch.setattr(
        worker,
        "bind_top2000_m03r_v7_runtime_sequence",
        lambda sequence, _policy: (sequence, provider),
    )
    calls: list[int] = []
    fail_once = {"enabled": True}

    def train_update(
        policy: Any,
        _sequence: object,
        _provider: object,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        overlay_optimizer = kwargs.get("overlay_optimizer")
        assert isinstance(overlay_optimizer, torch.optim.Optimizer)
        step = int(kwargs["completed_optimizer_steps"])
        calls.append(step)
        if step == 1 and fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("injected A06 restart boundary")
        optimizer.zero_grad(set_to_none=True)
        policy.alpha_core_parameters()[0].square().sum().backward()
        optimizer.step()
        overlay_optimizer.zero_grad(set_to_none=True)
        policy.total_risk_overlay_parameters()[0].square().sum().backward()
        overlay_optimizer.step()
        return {
            "objective": float(step),
            "alpha_core_optimizer_steps": 1,
            "overlay_optimizer_steps": 1,
            "development_only": True,
        }

    monkeypatch.setattr(
        worker,
        "train_top2000_m03r_v7_development_update",
        train_update,
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="injected A06 restart boundary"):
        worker.run_worker(
            plan,
            plan_file_sha256=plan_file_sha256,
            qualification_only=True,
            qualification_steps=2,
        )
    assert calls == [0, 1]

    terminal = worker.run_worker(
        plan,
        plan_file_sha256=plan_file_sha256,
        qualification_only=True,
        qualification_steps=2,
    )
    assert terminal is not None
    assert terminal["complete"] is True
    assert calls == [0, 1, 1]

    run_root = Path(plan.output_root) / "qualification"
    model_path = run_root / "cells" / "fold-00-seed-17" / "model.rank-00.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == worker.CELL_MODEL_SCHEMA
    assert payload["overlay_optimizer_required"] is True
    assert worker.optimizer_state_dict_sha256(
        payload["alpha_core_optimizer_state_dict"]
    ) == payload["alpha_core_optimizer_state_sha256"]
    assert worker.optimizer_state_dict_sha256(
        payload["overlay_optimizer_state_dict"]
    ) == payload["overlay_optimizer_state_sha256"]

    cell_receipt = json.loads(
        (
            run_root / "receipts" / "fold-00-seed-17.json"
        ).read_text(encoding="utf-8")
    )
    assert cell_receipt["overlay_optimizer_required"] is True
    assert cell_receipt["alpha_core_optimizer_steps"] == 2
    assert cell_receipt["overlay_optimizer_steps"] == 2
    assert cell_receipt["rank_alpha_core_optimizer_state_sha256"] == [
        payload["alpha_core_optimizer_state_sha256"]
    ]
    assert cell_receipt["rank_overlay_optimizer_state_sha256"] == [
        payload["overlay_optimizer_state_sha256"]
    ]

    repeated = worker.run_worker(
        plan,
        plan_file_sha256=plan_file_sha256,
        qualification_only=True,
        qualification_steps=2,
    )
    assert repeated is not None
    assert repeated["receipt_sha256"] == terminal["receipt_sha256"]
    assert calls == [0, 1, 1]


def test_retry_reuses_cell_model_after_seed_receipt_publication_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, optimizer_steps=1)
    # Exercise the full-only seed-validation boundary in one CPU test process.
    # The production plan remains two-rank; only this injected context is local.
    object.__setattr__(plan, "expected_world_size", 1)
    plan_path = tmp_path / "full-plan.json"
    plan_file_sha256 = worker.render_training_plan(plan_path, plan)
    context = worker._DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        owns_process_group=False,
    )
    monkeypatch.setattr(worker, "_distributed_context", lambda **_kwargs: context)
    fake_cache = SimpleNamespace(
        daily_ohlcv=torch.zeros((1001, 2, 5)),
        cache_identity="c" * 64,
        search_identity="d" * 64,
        action_hash="e" * 64,
    )
    monkeypatch.setattr(
        worker,
        "load_verified_top2000_hold30_development_cache",
        lambda *_args, **_kwargs: fake_cache,
    )
    monkeypatch.setattr(
        worker,
        "_build_episode",
        lambda *_args, start, **_kwargs: (
            _FakeBuilt(SimpleNamespace(receipt_sha256=f"{start:064x}")),
            object(),
            SimpleNamespace(
                loadings=torch.zeros((2, 1)),
                receipt_sha256=f"{start + 1:064x}",
            ),
        ),
    )
    provider = SimpleNamespace(
        inputs=SimpleNamespace(daily_ohlcv=torch.zeros((1, 2, 2, 5)))
    )
    monkeypatch.setattr(
        worker,
        "bind_top2000_m03r_v7_runtime_sequence",
        lambda sequence, _policy: (sequence, provider),
    )

    def new_cell(
        _plan: Top2000M03RV7DevelopmentTrainingPlan,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[_FakePolicy, torch.optim.Optimizer, None]:
        del _plan
        if seed != TOP2000_M03R_V7_DEV_SEEDS[0]:
            raise RuntimeError("stop after recovered finalization")
        model = _FakePolicy().to(device)
        return model, torch.optim.AdamW(model.parameters(), lr=1.0e-3), None

    monkeypatch.setattr(worker, "_new_cell", new_cell)
    updates: list[int] = []

    def train_update(
        policy: _FakePolicy,
        _sequence: object,
        _provider: object,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updates.append(int(kwargs["completed_optimizer_steps"]))
        optimizer.zero_grad(set_to_none=True)
        policy.weight.sum().backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return {"objective": 1.0, "development_only": True}

    monkeypatch.setattr(
        worker,
        "train_top2000_m03r_v7_development_update",
        train_update,
    )
    seed_hashes: list[str] = []

    def evaluate_seed(
        _cache: Any,
        fold: Any,
        _policy: Any,
        *,
        run_root: Path,
        seed: int,
        checkpoint_file_sha256: str,
        **_kwargs: Any,
    ) -> tuple[Path, str]:
        receipt_path = worker._seed_validation_receipt_path(
            run_root,
            fold_index=fold.fold_index,
            seed=seed,
        )
        payload = {
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "seed": seed,
        }
        receipt_sha256 = worker._write_immutable_json(receipt_path, payload)
        seed_hashes.append(checkpoint_file_sha256)
        return receipt_path, receipt_sha256

    monkeypatch.setattr(worker, "_evaluate_seed_checkpoint", evaluate_seed)
    original_write_immutable_json = worker._write_immutable_json
    crash_once = {"enabled": True}

    def crash_between_receipts(
        path: Path,
        payload: dict[str, Any],
    ) -> str:
        is_cell_receipt = (
            path.parent.name == "receipts"
            and path.name == "fold-00-seed-17.json"
        )
        if is_cell_receipt and crash_once["enabled"]:
            crash_once["enabled"] = False
            seed_receipt = worker._seed_validation_receipt_path(
                path.parents[1],
                fold_index=0,
                seed=TOP2000_M03R_V7_DEV_SEEDS[0],
            )
            assert seed_receipt.is_file()
            raise RuntimeError("injected post-seed-receipt crash")
        return original_write_immutable_json(path, payload)

    monkeypatch.setattr(worker, "_write_immutable_json", crash_between_receipts)
    with pytest.raises(RuntimeError, match="post-seed-receipt"):
        worker.run_worker(
            plan,
            plan_file_sha256=plan_file_sha256,
            qualification_only=False,
        )

    run_root = Path(plan.output_root) / "training"
    model_path = run_root / "cells" / "fold-00-seed-17" / "model.rank-00.pt"
    first_model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert updates == [0]
    assert seed_hashes == [first_model_sha256]
    assert not (run_root / "receipts" / "fold-00-seed-17.json").exists()

    with pytest.raises(RuntimeError, match="stop after recovered finalization"):
        worker.run_worker(
            plan,
            plan_file_sha256=plan_file_sha256,
            qualification_only=False,
        )
    assert updates == [0]
    assert seed_hashes == [first_model_sha256, first_model_sha256]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == first_model_sha256
    cell_receipt = json.loads(
        (run_root / "receipts" / "fold-00-seed-17.json").read_text(
            encoding="utf-8"
        )
    )
    assert cell_receipt["rank_model_sha256"] == [first_model_sha256]


def test_full_worker_rejects_single_rank_before_reading_cache(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(worker.Top2000M03RV7WorkerError, match="nproc_per_node=2"):
        worker.run_worker(
            plan,
            plan_file_sha256="f" * 64,
            qualification_only=False,
        )


def test_full_completion_requires_seed_and_six_ensemble_receipts(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    folds = render_top2000_m03r_v7_development_folds(1001)
    cells = tuple(
        (fold_index, seed)
        for fold_index in range(6)
        for seed in TOP2000_M03R_V7_DEV_SEEDS
    )
    with pytest.raises(
        worker.Top2000M03RV7WorkerError,
        match="exact thirty seed validation receipts",
    ):
        worker._collect_full_validation_evidence(
            tmp_path / "training",
            plan=plan,
            folds=folds,
            cells=cells,
        )
