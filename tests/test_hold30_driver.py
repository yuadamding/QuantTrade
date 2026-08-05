from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from rl_quant.training.hold30 import Hold30CanonicalRow, Hold30OriginReplay
from rl_quant.training.hold30_driver import (
    Hold30ArtifactError,
    Hold30StateProviderBinding,
    Hold30TrainingSweep,
    Hold30TrialIdentity,
    run_hold30_trial as _run_hold30_trial,
    verify_hold30_run as _verify_hold30_run,
)
from rl_quant.training.hold30_state import (
    Hold30DailyPolicyInputs,
    Hold30DailyPolicyStateProvider,
)
from rl_quant.training.hold30_coordinator import (
    Hold30CohortIdentity,
    Hold30CoordinationError,
    Hold30ValidationScore,
    checkpoint_reference_from_trial,
    coordinate_hold30_seed_cohort,
    publish_hold30_cohort_finalization,
)
from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(*, seed: int = 17, setting_id: str = "hold30-m02-age-hazard") -> Hold30TrialIdentity:
    return Hold30TrialIdentity(
        setting_id=setting_id,
        fold_index=2,
        seed=seed,
        executable_manifest_sha256=_digest("executable-manifest"),
        fold_sha256=_digest("fold-2"),
    )


def _sweeps(count: int = 3) -> tuple[Hold30TrainingSweep, ...]:
    sequence_sha256 = _digest("shared-fold-sequence")
    return tuple(
        Hold30TrainingSweep(
            sweep_index=index,
            sweep_id=f"fold-2-chronological-{index:03d}",
            sequence_sha256=sequence_sha256,
            sequence={"index": index},
            n_positions=95,
        )
        for index in range(count)
    )


def _provider_binding() -> Hold30StateProviderBinding:
    config = {"causal_input": "tiny-raw-fixture", "encoder": "trainable"}
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return Hold30StateProviderBinding(
        provider_id="test-functional-provider-v1",
        provider_config=config,
        provider_config_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def run_hold30_trial(*args, **kwargs):
    kwargs.setdefault("qualification_update_override", len(args[4]))
    return _run_hold30_trial(*args, **kwargs)


def verify_hold30_run(*args, **kwargs):
    kwargs.setdefault("allow_qualification_only", True)
    return _verify_hold30_run(*args, **kwargs)


class _Policy(torch.nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))


class _RandomReplayAdapter:
    """Tiny adapter that makes global RNG restoration observable in weights."""

    def __init__(self) -> None:
        self.canonical_sweeps: list[int] = []
        self.state_provider = type(
            "TrainableFixtureProvider",
            (),
            {
                "trains_upstream_encoder": True,
                "hold30_provider_config": {
                    "causal_input": "tiny-raw-fixture",
                    "encoder": "trainable",
                },
            },
        )()
        self.require_trainable_state_provider = True

    def canonical_pass(self, policy, sequence, roles):
        del policy
        assert not torch.is_grad_enabled()
        self.canonical_sweeps.append(sequence["index"])
        rows = [
            Hold30CanonicalRow(0.0, discretionary_turnover=0.02)
            for _ in range(roles.n_positions - 1)
        ]
        return {"sequence": sequence["index"]}, rows

    def replay_origins(self, policy, sequence, canonical_state, origins, roles):
        del roles
        assert canonical_state["sequence"] == sequence["index"]
        result = []
        for origin in origins.tolist():
            random_scale = torch.rand((), dtype=policy.score.dtype)
            utility = (policy.score * random_scale).expand(31) / 31.0
            zero = policy.score * 0.0
            result.append(
                Hold30OriginReplay(
                    origin=origin,
                    utility_rows=utility,
                    discretionary_turnover=zero + 0.02,
                    early_sale_mass=zero,
                    gate=zero,
                    gate_entropy=zero,
                )
            )
        return result


def _trainables(value: float = 0.25) -> tuple[_Policy, torch.optim.Optimizer]:
    policy = _Policy(value)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-5
    )
    return policy, optimizer


def test_trial_identity_and_sweep_plan_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown Hold-30 setting"):
        _identity(setting_id="hold30-unknown")
    with pytest.raises(ValueError, match="seed must be one of"):
        _identity(seed=18)
    with pytest.raises(ValueError, match="fold_index"):
        Hold30TrialIdentity(
            "hold30-m02-age-hazard",
            6,
            17,
            _digest("manifest"),
            _digest("fold"),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        Hold30TrainingSweep(0, "sweep", "BAD", object(), 95)

    duplicate_id = list(_sweeps(2))
    duplicate_id[1] = Hold30TrainingSweep(
        1,
        duplicate_id[0].sweep_id,
        duplicate_id[1].sequence_sha256,
        duplicate_id[1].sequence,
        95,
    )
    policy, optimizer = _trainables()
    with pytest.raises(ValueError, match="duplicate sweep_id"):
        run_hold30_trial(
            policy,
            optimizer,
            _RandomReplayAdapter(),
            _identity(),
            duplicate_id,
            Path("unused-duplicate-driver-test"),
            state_provider_binding=_provider_binding(),
        )


def test_static_precomputed_state_provider_is_rejected_before_writing(tmp_path: Path) -> None:
    adapter = _RandomReplayAdapter()
    adapter.state_provider.trains_upstream_encoder = False
    policy, optimizer = _trainables()
    root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="rejects static precomputed decision state"):
        run_hold30_trial(
            policy,
            optimizer,
            adapter,
            _identity(),
            _sweeps(1),
            root,
            state_provider_binding=_provider_binding(),
        )
    assert not root.exists()


def test_repeated_updates_may_bind_the_same_chronological_sequence(tmp_path: Path) -> None:
    shared_hash = _digest("same-chronological-fold-sequence")
    sweeps = tuple(
        Hold30TrainingSweep(
            index,
            f"optimizer-update-{index:03d}",
            shared_hash,
            {"index": index},
            95,
        )
        for index in range(3)
    )
    policy, optimizer = _trainables()
    progress = run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        tmp_path / "repeated-sequence",
        state_provider_binding=_provider_binding(),
    )
    assert progress.complete
    receipt = verify_hold30_run(tmp_path / "repeated-sequence")
    hashes = [row["sequence_sha256"] for row in json.loads(
        (tmp_path / "repeated-sequence/identity.json").read_text(encoding="utf-8")
    )["sweep_plan"]]
    assert hashes == [shared_hash] * 3
    assert receipt["optimizer_updates"] == 3
    assert receipt["qualification_only"] is True
    assert receipt["production_update_contract"] is False
    with pytest.raises(Hold30ArtifactError, match="cannot pass production verification"):
        _verify_hold30_run(tmp_path / "repeated-sequence")


def test_production_count_optimizer_and_actual_provider_config_are_enforced(
    tmp_path: Path,
) -> None:
    policy, optimizer = _trainables()
    with pytest.raises(ValueError, match="exactly 128 optimizer-update sweeps"):
        _run_hold30_trial(
            policy,
            optimizer,
            _RandomReplayAdapter(),
            _identity(),
            _sweeps(3),
            tmp_path / "short-production-plan",
            state_provider_binding=_provider_binding(),
        )

    policy = _Policy()
    wrong_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=2e-4, weight_decay=1e-4, eps=1e-5
    )
    with pytest.raises(ValueError, match="requires lr=0.0001"):
        _run_hold30_trial(
            policy,
            wrong_optimizer,
            _RandomReplayAdapter(),
            _identity(),
            _sweeps(3),
            tmp_path / "wrong-optimizer",
            state_provider_binding=_provider_binding(),
            qualification_update_override=3,
        )


def test_production_plan_keeps_all_128_updates_when_chunk_stops_early(
    tmp_path: Path,
) -> None:
    shared_hash = _digest("production-fold-sequence")
    sweeps = tuple(
        Hold30TrainingSweep(
            index,
            f"production-update-{index:03d}",
            shared_hash,
            {"index": index},
            95,
        )
        for index in range(128)
    )
    policy, optimizer = _trainables()
    progress = _run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        tmp_path / "production-prefix",
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    assert not progress.complete
    assert progress.completed_sweeps == 1
    assert progress.total_sweeps == 128
    identity = json.loads(
        (tmp_path / "production-prefix/identity.json").read_text(encoding="utf-8")
    )
    assert len(identity["sweep_plan"]) == 128
    assert identity["driver_contract"]["production_update_contract"] is True
    assert identity["driver_contract"]["qualification_only"] is False


def test_driver_accepts_and_binds_the_package_daily_policy_provider(tmp_path: Path) -> None:
    decisions, assets = 94, 2
    dtype = torch.float64
    inputs = Hold30DailyPolicyInputs(
        market_context=torch.zeros((1, decisions, 1), dtype=dtype),
        stock_context=torch.zeros((1, decisions, assets, 1), dtype=dtype),
        news_raw=torch.zeros((1, decisions, assets, 1, 1), dtype=dtype),
        news_mask=torch.zeros((1, decisions, assets, 1), dtype=torch.bool),
        available=torch.ones((1, decisions, assets), dtype=torch.bool),
        past_return=torch.zeros((1, decisions, assets), dtype=dtype),
        past_return_valid=torch.ones((1, decisions, assets), dtype=torch.bool),
        day_bars_fn=lambda _index: (
            torch.zeros((1, assets, 1, 1), dtype=dtype),
            torch.ones((1, assets, 1), dtype=torch.bool),
        ),
        source_axis_id=_digest("axis"),
        raw_bars_sha256=_digest("raw-bars"),
        frozen_context_sha256=_digest("context"),
    )
    provider = Hold30DailyPolicyStateProvider(inputs)
    binding = Hold30StateProviderBinding.from_provider(
        "package-daily-policy-provider",
        provider,
    )
    adapter = _RandomReplayAdapter()
    adapter.state_provider = provider
    policy, optimizer = _trainables()
    root = tmp_path / "package-provider"
    _run_hold30_trial(
        policy,
        optimizer,
        adapter,
        _identity(),
        _sweeps(1),
        root,
        state_provider_binding=binding,
        qualification_update_override=1,
    )
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    bound = identity["driver_contract"]["state_provider"]
    assert bound["provider_class"].endswith("Hold30DailyPolicyStateProvider")
    assert bound["provider_config"] == provider.binding_config


def test_receipt_complete_five_seed_early_stop_finalization_and_resume(
    tmp_path: Path,
) -> None:
    roots = {seed: tmp_path / f"seed-{seed}" for seed in HOLD30_SEEDS}
    sweeps = _sweeps(128)
    for seed in HOLD30_SEEDS:
        policy, optimizer = _trainables()
        progress = _run_hold30_trial(
            policy,
            optimizer,
            _RandomReplayAdapter(),
            _identity(seed=seed),
            sweeps,
            roots[seed],
            state_provider_binding=_provider_binding(),
            max_sweeps=40,
        )
        assert not progress.complete
        assert progress.completed_sweeps == 40

    cohort_identity = Hold30CohortIdentity(
        setting_id="hold30-m02-age-hazard",
        fold_index=2,
        executable_manifest_sha256=_identity().executable_manifest_sha256,
        fold_sha256=_identity().fold_sha256,
        inner_validation_sequence_sha256=_digest("inner-validation"),
    )

    def refs(update: int):
        return tuple(
            checkpoint_reference_from_trial(roots[seed], update)
            for seed in HOLD30_SEEDS
        )

    def score(update: int, _references):
        return Hold30ValidationScore(
            update=update,
            active_log_wealth=0.001 if update == 8 else 0.0,
            discretionary_turnover=0.02,
            trace_sha256=_digest(f"validation-trace-{update}"),
            inner_validation_sequence_sha256=cohort_identity.inner_validation_sequence_sha256,
        )

    outcome = coordinate_hold30_seed_cohort(
        cohort_identity,
        refs(0),
        advance_cohort=refs,
        validate_ensemble=score,
    )
    assert outcome.stopped_update == 40
    assert outcome.selected_validation.update == 32
    assert outcome.stop_reason == "validation_patience_exhausted"

    receipt_path = tmp_path / "cohort/cohort-finalization.json"
    receipt = publish_hold30_cohort_finalization(outcome, roots, receipt_path)
    assert receipt["selected_update"] == 32
    assert receipt["stopped_update"] == 40
    assert [row["seed"] for row in receipt["trial_artifacts"]] == list(HOLD30_SEEDS)
    verified = _verify_hold30_run(
        roots[HOLD30_SEEDS[0]],
        expected_identity=_identity(seed=HOLD30_SEEDS[0]),
    )
    assert verified["cohort_early_stop_finalized"] is True
    assert verified["selected_update"] == 32
    assert verified["stopped_update"] == 40
    assert verified["validation_checkpoint_selected"] is True
    assert all((roots[seed] / "cohort-finalization.json").is_file() for seed in HOLD30_SEEDS)

    # Publication is resumable only when every existing byte is safe and
    # identical.  A missing marker is completed without rewriting the common
    # receipt or the other seed markers.
    before = receipt_path.read_bytes()
    missing_marker = roots[HOLD30_SEEDS[-1]] / "cohort-finalization.json"
    missing_marker.unlink()
    assert publish_hold30_cohort_finalization(outcome, roots, receipt_path) == receipt
    assert receipt_path.read_bytes() == before
    assert missing_marker.is_file()

    policy, optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="cannot resume optimizer updates"):
        _run_hold30_trial(
            policy,
            optimizer,
            _RandomReplayAdapter(),
            _identity(seed=HOLD30_SEEDS[0]),
            sweeps,
            roots[HOLD30_SEEDS[0]],
            state_provider_binding=_provider_binding(),
            resume=True,
        )

    tampered_marker = roots[HOLD30_SEEDS[0]] / "cohort-finalization.json"
    original_marker = tampered_marker.read_bytes()
    tampered = json.loads(tampered_marker.read_text())
    tampered["stop_reason"] = "maximum_updates"
    tampered_marker.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(Hold30CoordinationError, match="unsafe existing cohort marker"):
        publish_hold30_cohort_finalization(outcome, roots, receipt_path)
    with pytest.raises(Hold30ArtifactError, match="receipt hash mismatch"):
        _verify_hold30_run(roots[HOLD30_SEEDS[0]], expected_identity=_identity(seed=HOLD30_SEEDS[0]))
    tampered_marker.write_bytes(original_marker)
    final_checkpoint = roots[HOLD30_SEEDS[1]] / "checkpoints/update-000040.pt"
    with final_checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(Hold30ArtifactError, match="cohort-finalization receipt verification failed"):
        _verify_hold30_run(roots[HOLD30_SEEDS[2]], expected_identity=_identity(seed=HOLD30_SEEDS[2]))

    wrong_config = {"causal_input": "caller-asserted-only", "encoder": "trainable"}
    encoded = json.dumps(wrong_config, sort_keys=True, separators=(",", ":")).encode()
    wrong_binding = Hold30StateProviderBinding(
        provider_id="wrong-config",
        provider_config=wrong_config,
        provider_config_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    policy, optimizer = _trainables()
    with pytest.raises(ValueError, match="differs from the actual provider"):
        _run_hold30_trial(
            policy,
            optimizer,
            _RandomReplayAdapter(),
            _identity(),
            _sweeps(3),
            tmp_path / "wrong-provider-config",
            state_provider_binding=wrong_binding,
            qualification_update_override=3,
        )


def test_uninterrupted_and_resumed_training_are_exactly_equal(tmp_path: Path) -> None:
    sweeps = _sweeps()
    identity = _identity()

    uninterrupted_policy, uninterrupted_optimizer = _trainables()
    uninterrupted_adapter = _RandomReplayAdapter()
    uninterrupted = run_hold30_trial(
        uninterrupted_policy,
        uninterrupted_optimizer,
        uninterrupted_adapter,
        identity,
        sweeps,
        tmp_path / "uninterrupted",
        state_provider_binding=_provider_binding(),
    )
    assert uninterrupted.complete
    assert uninterrupted.completed_sweeps == 3
    assert uninterrupted_adapter.canonical_sweeps == [0, 1, 2]

    partial_policy, partial_optimizer = _trainables()
    first_adapter = _RandomReplayAdapter()
    partial = run_hold30_trial(
        partial_policy,
        partial_optimizer,
        first_adapter,
        identity,
        sweeps,
        tmp_path / "resumed",
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    assert not partial.complete
    assert partial.completed_sweeps == 1
    assert partial.run_receipt is None
    assert first_adapter.canonical_sweeps == [0]

    # Resume into newly constructed objects.  Their current value is ignored
    # only after the exact architecture and optimizer contract have matched.
    resumed_policy, resumed_optimizer = _trainables(value=99.0)
    second_adapter = _RandomReplayAdapter()
    resumed = run_hold30_trial(
        resumed_policy,
        resumed_optimizer,
        second_adapter,
        identity,
        sweeps,
        tmp_path / "resumed",
        state_provider_binding=_provider_binding(),
        resume=True,
    )
    assert resumed.complete
    assert second_adapter.canonical_sweeps == [1, 2]
    assert torch.equal(resumed_policy.score, uninterrupted_policy.score)
    assert resumed_optimizer.state_dict() == uninterrupted_optimizer.state_dict()

    uninterrupted_metrics = json.loads(
        (tmp_path / "uninterrupted/metrics.json").read_text(encoding="utf-8")
    )
    resumed_metrics = json.loads(
        (tmp_path / "resumed/metrics.json").read_text(encoding="utf-8")
    )
    assert resumed_metrics["metrics"] == uninterrupted_metrics["metrics"]
    receipt = verify_hold30_run(tmp_path / "resumed", expected_identity=identity)
    assert receipt["completed_sweeps"] == 3
    assert receipt["optimizer_updates"] == 3
    assert receipt["optimization_sweeps_complete"] is True
    assert receipt["end_to_end_validation_complete"] is False
    assert receipt["validation_checkpoint_selected"] is False
    assert receipt["scientific_qualification"] is False
    assert receipt["promotion_authorized"] is False
    assert receipt["gpu_launch_performed"] is False
    assert len(receipt["artifact_graph"]["checkpoints"]) == 3


def test_resume_rejects_mismatched_identity_partial_pairs_and_duplicates(tmp_path: Path) -> None:
    sweeps = _sweeps(2)

    mismatch_root = tmp_path / "mismatch"
    policy, optimizer = _trainables()
    run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        mismatch_root,
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    fresh_policy, fresh_optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="setting/fold/seed identity mismatch"):
        run_hold30_trial(
            fresh_policy,
            fresh_optimizer,
            _RandomReplayAdapter(),
            _identity(seed=29),
            sweeps,
            mismatch_root,
            state_provider_binding=_provider_binding(),
            resume=True,
        )

    orphan_root = tmp_path / "orphan"
    policy, optimizer = _trainables()
    run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        orphan_root,
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    (orphan_root / "checkpoints/update-000002.pt").write_bytes(b"orphan")
    fresh_policy, fresh_optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="not complete pairs"):
        run_hold30_trial(
            fresh_policy,
            fresh_optimizer,
            _RandomReplayAdapter(),
            _identity(),
            sweeps,
            orphan_root,
            state_provider_binding=_provider_binding(),
            resume=True,
        )

    duplicate_root = tmp_path / "duplicate"
    policy, optimizer = _trainables()
    run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        duplicate_root,
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    (duplicate_root / "checkpoints/update-000001-copy.pt").write_bytes(b"duplicate")
    fresh_policy, fresh_optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="unknown or partial checkpoint"):
        run_hold30_trial(
            fresh_policy,
            fresh_optimizer,
            _RandomReplayAdapter(),
            _identity(),
            sweeps,
            duplicate_root,
            state_provider_binding=_provider_binding(),
            resume=True,
        )


def test_complete_graph_rejects_tampering_and_never_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "complete"
    policy, optimizer = _trainables()
    identity = _identity()
    sweeps = _sweeps(2)
    run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        identity,
        sweeps,
        root,
        state_provider_binding=_provider_binding(),
    )
    verify_hold30_run(root, expected_identity=identity)

    new_policy, new_optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="new trials require an empty"):
        run_hold30_trial(
            new_policy,
            new_optimizer,
            _RandomReplayAdapter(),
            identity,
            sweeps,
            root,
            state_provider_binding=_provider_binding(),
        )
    with pytest.raises(Hold30ArtifactError, match="completed trial cannot be resumed"):
        run_hold30_trial(
            new_policy,
            new_optimizer,
            _RandomReplayAdapter(),
            identity,
            sweeps,
            root,
            state_provider_binding=_provider_binding(),
            resume=True,
        )

    with (root / "final-model.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(Hold30ArtifactError, match="artifact_graph mismatch"):
        verify_hold30_run(root, expected_identity=identity)


def test_partial_final_artifact_set_is_not_resumable(tmp_path: Path) -> None:
    root = tmp_path / "partial-final"
    policy, optimizer = _trainables()
    sweeps = _sweeps(2)
    run_hold30_trial(
        policy,
        optimizer,
        _RandomReplayAdapter(),
        _identity(),
        sweeps,
        root,
        state_provider_binding=_provider_binding(),
        max_sweeps=1,
    )
    (root / "final-model.pt").write_bytes(b"crash-before-final-receipt")
    fresh_policy, fresh_optimizer = _trainables()
    with pytest.raises(Hold30ArtifactError, match="final artifact set is partial"):
        run_hold30_trial(
            fresh_policy,
            fresh_optimizer,
            _RandomReplayAdapter(),
            _identity(),
            sweeps,
            root,
            state_provider_binding=_provider_binding(),
            resume=True,
        )
