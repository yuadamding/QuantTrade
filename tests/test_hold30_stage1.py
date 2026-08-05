"""Frozen contract, censorship, and durable-resume tests for Hold-30 Stage 1."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.training.context_pretrain import train_context_encoder
from rl_quant.training.hold30_stage1 import (
    HOLD30_STAGE1_ACCUMULATION,
    HOLD30_STAGE1_BATCH_SIZE,
    HOLD30_STAGE1_CHECKPOINT_EVERY,
    HOLD30_STAGE1_EFFECTIVE_BATCH,
    HOLD30_STAGE1_EXECUTION_STRATEGY,
    HOLD30_STAGE1_HORIZON,
    HOLD30_STAGE1_LR,
    HOLD30_STAGE1_SHARING_SCOPE,
    HOLD30_STAGE1_STEPS,
    HOLD30_STAGE1_WARMUP_STEPS,
    Hold30Stage1Error,
    Hold30Stage1FreezeBlocker,
    Hold30Stage1FreezeDecision,
    Hold30Stage1Identity,
    Hold30Stage1TrainingData,
    derive_hold30_stage1_seed,
    hold30_stage1_contract,
    hold30_stage1_freeze_status,
    hold30_stage1_normalization_schedule_sha256,
    hold30_stage1_optimizer_schedule_sha256,
    hold30_stage1_training_tensor_sha256,
    materialize_hold30_stage1_schedule,
    run_hold30_stage1,
    verify_hold30_stage1_run,
)

_DIGEST = "a" * 64


class _ParityEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)

    def forward(self, bars, _mask, cov):
        del cov
        per_stock = self.proj(bars[:, :, :1]).transpose(1, 2)
        return per_stock, per_stock.mean(dim=2)


class _ParityDailyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)

    def forward(self, value):
        return self.proj(value).squeeze(-1)


def _parity_days():
    days, targets = [], []
    for index in range(36):
        ret_valid = torch.ones(1, 2, dtype=torch.bool)
        daily_valid = torch.tensor([False, index % 5 != 0])
        days.append(
            {
                "bars": torch.tensor([[[0.0]], [[float(index + 1) / 36.0]]]),
                "bar_mask": torch.ones(2, 1, dtype=torch.bool),
                "cov_blocks": torch.empty(1, 2, 0),
                "ret": torch.tensor([[0.0, float(index - 18) / 10_000.0]]),
                "ret_valid": ret_valid,
                "session_close_block": 0,
            }
        )
        targets.append(
            (
                torch.tensor([0.0, float(18 - index) / 1_000.0]),
                daily_valid,
            )
        )
    return days, targets


def _parity_modules(initial_state=None):
    torch.manual_seed(37)
    modules = (_ParityEncoder(), torch.nn.Linear(1, 2), _ParityDailyHead())
    if initial_state is not None:
        for module, state in zip(modules, initial_state, strict=True):
            module.load_state_dict(state)
    return modules


def _parity_train(rank: int, world: int, initial_state, reduce=None):
    encoder, market_head, daily_head = _parity_modules(initial_state)
    parameters = [*encoder.parameters(), *market_head.parameters(), *daily_head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-2)
    train_context_encoder(
        encoder,
        market_head,
        _parity_days()[0],
        device=torch.device("cpu"),
        daily_head=daily_head,
        daily_targets=_parity_days()[1],
        daily_coef=1.0,
        perstock_coef=0.0,
        steps=1,
        lr=2e-4,
        weight_decay=1e-2,
        batch_size=3,
        accum_steps=12,
        schedule="constant",
        optimizer=optimizer,
        effective_index_schedule=(tuple(range(36)),),
        distributed_rank=rank,
        distributed_world_size=world,
        global_valid_normalization=True,
        grad_reduce=reduce,
        grad_reduce_mode="sum" if world == 2 else None,
    )
    return tuple(module.state_dict() for module in (encoder, market_head, daily_head))


def _gloo_parity_worker(rank: int, init_method: str, initial_state, output_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )

    def reduce(parameters):
        for parameter in parameters:
            if parameter.grad is not None:
                torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)

    try:
        state = _parity_train(rank, 2, initial_state, reduce)
        torch.save(state, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def _gloo_driver_worker(
    rank: int,
    init_method: str,
    root: str,
    identity: Hold30Stage1Identity,
    data: Hold30Stage1TrainingData,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        progress = run_hold30_stage1(
            root,
            identity,
            data,
            device=torch.device("cpu"),
            world_size=2,
            rank=rank,
            _train_fn=_fake_train,
        )
        assert progress.complete
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _join_processes(processes: list[mp.Process], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)


def _freeze(data: Hold30Stage1TrainingData | None = None) -> Hold30Stage1FreezeDecision:
    bound_data = _data() if data is None else data
    normalization = (0, 7, 14, 21, 28, 35)
    return Hold30Stage1FreezeDecision(
        stage1_seed=derive_hold30_stage1_seed(0),
        sharing_scope=HOLD30_STAGE1_SHARING_SCOPE,
        sharing_manifest_sha256="b" * 64,
        normalization_day_indices=normalization,
        normalization_schedule_sha256=hold30_stage1_normalization_schedule_sha256(
            bound_data.day_ids, normalization
        ),
        optimizer_schedule_sha256=hold30_stage1_optimizer_schedule_sha256(
            bound_data.optimizer_day_schedule
        ),
        checkpoint_every=HOLD30_STAGE1_CHECKPOINT_EVERY,
        execution_strategy=HOLD30_STAGE1_EXECUTION_STRATEGY,
        approval_receipt_sha256="c" * 64,
    )


def _identity(data: Hold30Stage1TrainingData | None = None) -> Hold30Stage1Identity:
    bound_data = _data() if data is None else data
    return Hold30Stage1Identity(
        fold_index=0,
        executable_manifest_sha256=_DIGEST,
        source_archive_sha256=_DIGEST,
        data_snapshot_sha256=_DIGEST,
        data_qualification_sha256=_DIGEST,
        fold_sha256=_DIGEST,
        training_tensor_sha256=hold30_stage1_training_tensor_sha256(bound_data),
        freeze=_freeze(bound_data),
    )


def _day(index: int, *, actions: int = 3) -> dict[str, object]:
    bars = torch.zeros(actions, 390, 5, dtype=torch.float32)
    bars[..., 0] = 10.0 + index
    bars[..., 1] = 11.0 + index
    bars[..., 2] = 9.0 + index
    bars[..., 3] = 10.5 + index
    bars[..., 4] = 1_000.0 + index
    ret = torch.zeros(78, actions, dtype=torch.float32)
    ret[:, 1] = 0.001 * (index + 1)
    ret[:, 2] = -0.0005 * (index + 1)
    return {
        "bars": bars,
        "bar_mask": torch.ones(actions, 390, dtype=torch.bool),
        "cov_blocks": torch.empty(78, actions, 0),
        "ret": ret,
        "ret_valid": torch.ones(78, actions, dtype=torch.bool),
        "session_close_block": 77,
    }


def _data() -> Hold30Stage1TrainingData:
    days = tuple(_day(index) for index in range(36))
    close = torch.ones(36, 3, dtype=torch.float32)
    close[:, 1] = torch.linspace(100.0, 135.0, 36)
    close[:, 2] = torch.linspace(80.0, 115.0, 36)
    schedule = materialize_hold30_stage1_schedule(
        36, derive_hold30_stage1_seed(0)
    )
    return Hold30Stage1TrainingData(
        fold_index=0,
        day_ids=tuple(f"2020-02-{index + 1:02d}" for index in range(28))
        + tuple(f"2020-03-{index + 1:02d}" for index in range(8)),
        train_days=days,
        day_close=close,
        optimizer_day_schedule=schedule,
    )


def _fake_train(encoder, _market_head, _days, **kwargs):
    assert kwargs["steps"] <= HOLD30_STAGE1_STEPS
    assert kwargs["lr"] == HOLD30_STAGE1_LR
    assert kwargs["batch_size"] == HOLD30_STAGE1_BATCH_SIZE
    assert kwargs["accum_steps"] == HOLD30_STAGE1_ACCUMULATION
    assert kwargs["warmup_steps"] == HOLD30_STAGE1_WARMUP_STEPS
    assert kwargs["daily_coef"] == 1.0
    assert kwargs["perstock_coef"] == 0.0
    assert kwargs["global_valid_normalization"] is True
    assert any(bool(mask.any()) for _, mask in kwargs["daily_targets"][:-31])
    assert not any(bool(mask.any()) for _, mask in kwargs["daily_targets"][-31:])
    assert len(kwargs["effective_index_schedule"]) == kwargs["steps"]
    parameter = next(encoder.parameters())
    for step in range(kwargs["start_step"], kwargs["steps"]):
        with torch.no_grad():
            parameter.reshape(-1)[0].add_(float(step + 1) / 1_000_000.0)
        if (step + 1) % kwargs["checkpoint_every"] == 0:
            kwargs["on_checkpoint"](step + 1, kwargs["optimizer"])
    return kwargs["optimizer"]


def test_stage1_contract_is_the_exact_compact_daily_target_design() -> None:
    contract = hold30_stage1_contract()

    assert contract["model"] | {
        "bar_feature_dim": 5,
        "covariate_dim": 0,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "feedforward_dim": 256,
    } == contract["model"]
    assert contract["optimizer"]["optimizer_steps"] == 1_000
    assert contract["optimizer"]["micro_batch_days"] == 3
    assert contract["optimizer"]["accumulation_micro_batches"] == 12
    assert contract["optimizer"]["effective_batch_days"] == 36
    assert contract["optimizer"]["world_size"] == 2
    assert contract["optimizer"]["global_microbatch_rank_counts"] == {
        "even_microbatch": [2, 1],
        "odd_microbatch": [1, 2],
    }
    assert contract["optimizer"]["dates_per_rank_per_update"] == 18
    assert contract["optimizer"]["gradient_reduction"] == "SUM"
    assert contract["targets"]["daily_horizon_sessions"] == HOLD30_STAGE1_HORIZON
    assert contract["targets"]["daily_cross_sectional_coefficient"] == 1.0
    assert contract["targets"]["intraday_per_stock_coefficient"] == 0.0
    assert contract["targets"]["covariate_axis_width"] == 0
    assert contract["input"]["news_enabled"] is False


def test_stage1_schedule_is_exact_replayable_and_uses_distinct_effective_days() -> None:
    assert [derive_hold30_stage1_seed(index) for index in range(6)] == [
        12_878_953_054_537_996_694,
        11_616_912_469_286_781_817,
        2_264_784_845_887_035_364,
        6_068_831_580_819_448_648,
        6_402_213_304_234_286_745,
        17_743_182_390_287_110_661,
    ]
    first = materialize_hold30_stage1_schedule(47, 211)
    second = materialize_hold30_stage1_schedule(47, 211)

    assert first == second
    assert len(first) == HOLD30_STAGE1_STEPS
    assert all(len(row) == HOLD30_STAGE1_EFFECTIVE_BATCH for row in first)
    assert all(len(set(row)) == HOLD30_STAGE1_EFFECTIVE_BATCH for row in first)
    assert materialize_hold30_stage1_schedule(47, 212) != first


def test_stage1_missing_seed_sharing_or_execution_choice_is_a_freeze_blocker() -> None:
    missing_status = hold30_stage1_freeze_status(None)
    assert missing_status["freeze_decision_complete"] is False
    assert missing_status["launch_authorized"] is False
    assert "missing:stage1_seed" in missing_status["qualification_blockers"]
    assert missing_status["two_h100_stage1_qualified"] is False

    with pytest.raises(Hold30Stage1FreezeBlocker, match="stage1_seed"):
        Hold30Stage1FreezeDecision(
            stage1_seed=True,
            sharing_scope="shared",
            sharing_manifest_sha256=_DIGEST,
            normalization_day_indices=(0,),
            normalization_schedule_sha256=_DIGEST,
            optimizer_schedule_sha256=_DIGEST,
            checkpoint_every=HOLD30_STAGE1_CHECKPOINT_EVERY,
            execution_strategy=HOLD30_STAGE1_EXECUTION_STRATEGY,
            approval_receipt_sha256=_DIGEST,
        )

    ready_status = hold30_stage1_freeze_status(_freeze())
    assert ready_status["freeze_decision_complete"] is True
    assert ready_status["receipt_driver_ready"] is True
    assert ready_status["two_rank_software_contract_ready"] is True
    assert ready_status["two_h100_stage1_qualified"] is False
    assert ready_status["launch_authorized"] is False
    with pytest.raises(Hold30Stage1FreezeBlocker, match="sharing_scope"):
        Hold30Stage1FreezeDecision(
            stage1_seed=1,
            sharing_scope="",
            sharing_manifest_sha256=_DIGEST,
            normalization_day_indices=(0,),
            normalization_schedule_sha256=_DIGEST,
            optimizer_schedule_sha256=_DIGEST,
            checkpoint_every=HOLD30_STAGE1_CHECKPOINT_EVERY,
            execution_strategy=HOLD30_STAGE1_EXECUTION_STRATEGY,
            approval_receipt_sha256=_DIGEST,
        )
    with pytest.raises(Hold30Stage1FreezeBlocker, match="distributed/sharding"):
        Hold30Stage1FreezeDecision(
            stage1_seed=derive_hold30_stage1_seed(0),
            sharing_scope=HOLD30_STAGE1_SHARING_SCOPE,
            sharing_manifest_sha256=_DIGEST,
            normalization_day_indices=(0,),
            normalization_schedule_sha256=_DIGEST,
            optimizer_schedule_sha256=_DIGEST,
            checkpoint_every=HOLD30_STAGE1_CHECKPOINT_EVERY,
            execution_strategy="guessed-two-rank-data-parallel",
            approval_receipt_sha256=_DIGEST,
        )


def test_stage1_rejects_outer_or_news_fields_before_writing(tmp_path: Path) -> None:
    source = _data()
    days = [dict(day) for day in source.train_days]
    days[0]["outer_score"] = torch.ones(1)
    contaminated = Hold30Stage1TrainingData(
        fold_index=source.fold_index,
        day_ids=source.day_ids,
        train_days=days,
        day_close=source.day_close,
        optimizer_day_schedule=source.optimizer_day_schedule,
    )

    root = tmp_path / "blocked"
    with pytest.raises(Hold30Stage1Error, match="forbidden non-training fields"):
        run_hold30_stage1(
            root,
            _identity(),
            contaminated,
            device=torch.device("cpu"),
            _train_fn=_fake_train,
        )
    assert not root.exists()


def test_stage1_append_only_resume_closes_receipt_chain(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    partial = run_hold30_stage1(
        root,
        _identity(),
        _data(),
        device=torch.device("cpu"),
        max_new_steps=500,
        _train_fn=_fake_train,
    )
    assert partial.complete is False
    assert partial.completed_steps == 500
    assert partial.run_receipt is None
    optimizer_schedule = json.loads(
        (root / "optimizer-date-schedule.json").read_text(encoding="utf-8")
    )
    normalization_schedule = json.loads(
        (root / "normalization-date-schedule.json").read_text(encoding="utf-8")
    )
    assert len(optimizer_schedule["rows"]) == 1_000
    assert all(len(row) == 36 for row in optimizer_schedule["rows"])
    assert normalization_schedule["fold_training_only"] is True
    assert optimizer_schedule["previous_receipt_sha256"] == normalization_schedule[
        "receipt_sha256"
    ]

    complete = run_hold30_stage1(
        root,
        _identity(),
        _data(),
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    assert complete.complete is True
    assert complete.completed_steps == 1_000
    receipt = verify_hold30_stage1_run(root, expected_identity=_identity())
    assert receipt["outer_data_exposed"] is False
    assert receipt["complete"] is True

    # A completed run is idempotently reusable and never overwrites artifacts.
    repeated = run_hold30_stage1(
        root,
        _identity(),
        _data(),
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    assert repeated == complete

    uninterrupted_root = tmp_path / "uninterrupted"
    run_hold30_stage1(
        uninterrupted_root,
        _identity(),
        _data(),
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    resumed_encoder = json.loads(
        (root / "frozen-encoder.receipt.json").read_text(encoding="utf-8")
    )
    uninterrupted_encoder = json.loads(
        (uninterrupted_root / "frozen-encoder.receipt.json").read_text(encoding="utf-8")
    )
    assert resumed_encoder["encoder_state_sha256"] == uninterrupted_encoder["encoder_state_sha256"]


def test_stage1_verifier_rejects_receipt_tampering(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    run_hold30_stage1(
        root,
        _identity(),
        _data(),
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    path = root / "checkpoints" / "step-000500.receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["step"] = 501
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(Hold30Stage1Error, match="receipt hash mismatch"):
        verify_hold30_stage1_run(root)


def test_stage1_resume_attaches_only_exact_missing_receipts(tmp_path: Path) -> None:
    identity, data = _identity(), _data()
    partial_root = tmp_path / "checkpoint-attach"
    run_hold30_stage1(
        partial_root,
        identity,
        data,
        device=torch.device("cpu"),
        max_new_steps=50,
        _train_fn=_fake_train,
    )
    checkpoint_receipt = (
        partial_root / "checkpoints" / "step-000050.receipt.json"
    )
    checkpoint_receipt.unlink()
    resumed = run_hold30_stage1(
        partial_root,
        identity,
        data,
        device=torch.device("cpu"),
        max_new_steps=0,
        _train_fn=_fake_train,
    )
    assert resumed.completed_steps == 50
    assert checkpoint_receipt.is_file()

    complete_root = tmp_path / "final-attach"
    run_hold30_stage1(
        complete_root,
        identity,
        data,
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    (complete_root / "run-receipt.json").unlink()
    (complete_root / "frozen-encoder.receipt.json").unlink()
    reattached = run_hold30_stage1(
        complete_root,
        identity,
        data,
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    assert reattached.complete
    verify_hold30_stage1_run(complete_root, expected_identity=identity)


def test_stage1_verifier_rejects_reuse_under_changed_bindings(tmp_path: Path) -> None:
    root = tmp_path / "binding-change"
    base = _identity()
    run_hold30_stage1(
        root,
        base,
        _data(),
        device=torch.device("cpu"),
        _train_fn=_fake_train,
    )
    variants = (
        replace(base, source_archive_sha256="d" * 64),
        replace(base, fold_sha256="e" * 64),
        replace(
            base,
            freeze=replace(base.freeze, normalization_day_indices=(0, 8, 16, 24, 32)),
        ),
        replace(
            base,
            freeze=replace(base.freeze, optimizer_schedule_sha256="f" * 64),
        ),
    )
    for changed in variants:
        with pytest.raises(Hold30Stage1Error, match="identity differs"):
            verify_hold30_stage1_run(root, expected_identity=changed)


def test_context_pretrainer_obeys_explicit_effective_index_schedule() -> None:
    class TinyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.seen: list[int] = []

        def forward(self, bars, _mask, cov):
            del cov
            self.seen.extend(int(value) for value in bars[:, 0, 0, 0].tolist())
            batch, actions = bars.shape[:2]
            per_stock = bars[:, :, :1, :1].reshape(batch, 1, actions, 1) * self.scale
            return per_stock, per_stock.mean(dim=2)

    days = []
    for index in range(4):
        days.append(
            {
                "bars": torch.full((2, 1, 1), float(index)),
                "bar_mask": torch.ones(2, 1, dtype=torch.bool),
                "cov_blocks": torch.empty(1, 2, 0),
                "ret": torch.zeros(1, 2),
                "ret_valid": torch.ones(1, 2, dtype=torch.bool),
                "session_close_block": 0,
            }
        )
    encoder = TinyEncoder()
    train_context_encoder(
        encoder,
        torch.nn.Linear(1, 2),
        days,
        device=torch.device("cpu"),
        steps=2,
        batch_size=2,
        accum_steps=1,
        schedule="constant",
        effective_index_schedule=((3, 1), (2, 0)),
    )
    assert encoder.seen == [3, 1, 2, 0]

    with pytest.raises(ValueError, match="must use distinct days"):
        train_context_encoder(
            TinyEncoder(),
            torch.nn.Linear(1, 2),
            days,
            device=torch.device("cpu"),
            steps=1,
            batch_size=2,
            accum_steps=1,
            effective_index_schedule=((1, 1),),
        )


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU/Gloo is unavailable",
)
def test_stage1_alternating_two_rank_update_matches_one_rank(tmp_path: Path) -> None:
    initial_modules = _parity_modules()
    initial_state = tuple(module.state_dict() for module in initial_modules)
    expected = _parity_train(0, 1, initial_state)

    context = mp.get_context("spawn")
    init_method = f"file://{tmp_path / 'gloo-init'}"
    processes = [
        context.Process(
            target=_gloo_parity_worker,
            args=(rank, init_method, initial_state, str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    _join_processes(processes, timeout=30)

    actual = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for left_module, right_module in zip(actual[0], actual[1], strict=True):
        for name in left_module:
            torch.testing.assert_close(left_module[name], right_module[name], rtol=0, atol=0)
    for expected_module, actual_module in zip(expected, actual[0], strict=True):
        for name in expected_module:
            torch.testing.assert_close(
                expected_module[name],
                actual_module[name],
                rtol=2e-6,
                atol=2e-7,
            )


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU/Gloo is unavailable",
)
def test_two_rank_stage1_driver_has_one_receipt_complete_writer(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    root = tmp_path / "two-rank-run"
    init_method = f"file://{tmp_path / 'driver-gloo-init'}"
    identity, data = _identity(), _data()
    processes = [
        context.Process(
            target=_gloo_driver_worker,
            args=(rank, init_method, str(root), identity, data),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    _join_processes(processes, timeout=60)

    receipt = verify_hold30_stage1_run(root, expected_identity=identity)
    assert receipt["two_rank_software_contract"] is True
    assert receipt["h100_qualified"] is False
    identity_receipt = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    assert identity_receipt["execution"]["world_size"] == 2
    assert identity_receipt["execution"]["backend"] == "gloo"
    assert len(list((root / "checkpoints").glob("*.pt"))) == 20
