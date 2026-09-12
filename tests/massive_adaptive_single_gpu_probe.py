"""Single-CUDA-device regression of the existing PPO trainer, not real-data training.

Synthetic upstream sources come from the existing numerical market fixture.
The actor, collector, compiler, execution kernel, updates and checkpoint parser
run unmodified. No output from this probe authorizes QT200 or native V5.
"""

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import stat
import sys

import numpy as np
import torch

from rl_quant.rl.massive_adaptive_ppo_policy_v1 import build_seeded_massive_adaptive_ppo_model_v1
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1, MassiveAdaptivePPOTrainerV1, _state_receipt,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    _checkpoint_payload, _parse_checkpoint,
)
from test_massive_adaptive_rl_v5_vertical import _market_fixture

MAX_BYTES = 32 * 1024 * 1024


def write_json(path, value):
    body = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    write_bytes(path, body)


def write_bytes(path, body):
    assert len(body) <= MAX_BYTES
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def load_checkpoint(path, expected_sha256):
    assert path.is_absolute() and path.resolve() == path
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES
        body = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    assert all(getattr(before, k) == getattr(after, k) for k in (
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
    if len(body) != before.st_size or hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("Single-GPU checkpoint bytes changed")
    checkpoint = _parse_checkpoint(torch.load(BytesIO(body), map_location="cpu", weights_only=True))
    assert not checkpoint.source_data_qualified and not checkpoint.development_rl_training_authorized
    return checkpoint


def make_trainer(*, seed=17):
    config = MassiveAdaptivePPOConfigV1(seed=seed, rollout_length=63, minibatch_size=63)
    return MassiveAdaptivePPOTrainerV1(
        environment=_market_fixture(0).environment(20.0),
        model=build_seeded_massive_adaptive_ppo_model_v1(seed=seed),
        config=config, device="cuda:0")


def rng_sample():
    return dict(torch=torch.rand(3).tolist(), cuda=torch.rand(3, device="cuda:0").cpu().tolist(),
                numpy=np.random.random(3).tolist(), python=[random.random() for _ in range(3)])


def tensor_groups(model):
    states = model.state_dict()
    return {family: _state_receipt({name: value for name, value in states.items()
            if name.startswith(prefixes)}) for family, prefixes in (
                ("actor", ("actor",)), ("critic", ("critic", "value_head")))}


def step(trainer):
    rollout = trainer.collect_rollout(steps=63)
    assert rollout.observations.shape == (63, 90) and rollout.actions.shape == (63, 10)
    for name in ("observations", "actions", "old_log_probabilities", "old_values",
                 "rewards", "advantages", "returns", "terminated"):
        assert getattr(rollout, name).device == torch.device("cuda:0")
    metrics = trainer.update(rollout)
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(p.device == torch.device("cuda:0") and p.dtype == torch.float32
               for p in trainer.model.parameters())
    for optimizer in (trainer.actor_optimizer, trainer.critic_optimizer):
        states = optimizer.state_dict()["state"]
        assert states and all(float(row["step"]) == 4 * trainer.update_index for row in states.values())
        assert all(row[key].device == torch.device("cuda:0") for row in states.values()
                   for key in ("exp_avg", "exp_avg_sq"))
    return dict(metrics=metrics, model_sha256=_state_receipt(trainer.model.state_dict()),
        actor_optimizer_sha256=_state_receipt(trainer.actor_optimizer.state_dict()),
        critic_optimizer_sha256=_state_receipt(trainer.critic_optimizer.state_dict()),
        actions_sha256=_state_receipt(rollout.actions), transitions=list(rollout.transition_receipts),
        environment_sha256=trainer.environment.state.semantic_receipt_sha256,
        update_index=trainer.update_index, chronology_cursor=trainer.environment.state.chronology_cursor)


def main(mode, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if mode == "cpu":
        assert not torch.cuda.is_available() and torch.cuda.device_count() == 0
        reference = json.loads((root / "reference.json").read_text())
        checkpoint = load_checkpoint(root / "single.pt", reference["checkpoint_sha256"])
        model = build_seeded_massive_adaptive_ppo_model_v1(seed=17).eval()
        model.load_state_dict(checkpoint.model_state)
        with torch.no_grad():
            actual = model({"adaptive_state": torch.tensor(reference["inference_observation"],
                dtype=torch.float32).unsqueeze(0)}).distribution.deterministic_action()
        torch.testing.assert_close(actual, torch.tensor(reference["inference_actions"]), atol=2e-6, rtol=2e-5)
        assert checkpoint.environment_state.chronology_cursor == 63
        print(json.dumps(dict(cpu_inference=True, gpu_visible=False, economic_cursor=63,
            source_data_qualified=False, native_v5_qualified=False)), flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        python=sys.version, gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)), visible_devices=1,
        deterministic_algorithms=True, tf32=False, autocast=False)
    trainer = make_trainer(seed=29 if mode == "wrong-config" else 17)
    if mode in ("resume", "wrong-config"):
        reference = json.loads((root / "reference.json").read_text())
        assert runtime == reference["runtime"]
        trainer.restore(load_checkpoint(root / "single.pt", reference["checkpoint_sha256"]))
        if mode == "wrong-config":
            raise AssertionError("Wrong seed/configuration was accepted")
        assert rng_sample() == reference["rng_after_save"]
        actual = step(trainer)
        assert actual == reference["second_update"]
        assert len(set(trainer.transition_decision_session_dates)) == 126
        write_json(root / "resume.json", dict(fresh_process=True, exact_model=True,
            exact_optimizers=True, exact_rng=True, exact_actions=True,
            exact_transitions=True, exact_economic_state=True, native_v5_qualified=False))
        return
    assert mode == "qualify"
    before = tensor_groups(trainer.model)
    first = step(trainer)
    after = tensor_groups(trainer.model)
    assert all(before[key] != after[key] for key in before)
    assert trainer.environment.state.chronology_cursor == 63
    checkpoint = trainer.checkpoint()
    buffer = BytesIO()
    torch.save(_checkpoint_payload(checkpoint, source_data_qualified=False), buffer)
    checkpoint_bytes = buffer.getvalue()
    write_bytes(root / "single.pt", checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    assert load_checkpoint(root / "single.pt", digest).semantic_receipt_sha256 == checkpoint.semantic_receipt_sha256
    observation = trainer._ensure_observation().values
    with torch.no_grad():
        inference = trainer.model({"adaptive_state": torch.tensor(observation,
            dtype=torch.float32, device="cuda:0").unsqueeze(0)}).distribution.deterministic_action().cpu().tolist()
    probe = rng_sample()
    second = step(trainer)
    assert len(trainer.transition_receipts) == len(set(trainer.transition_decision_session_dates)) == 126
    assert first["model_sha256"] != second["model_sha256"]
    assert hashlib.sha256((root / "single.pt").read_bytes()).hexdigest() == digest
    write_json(root / "reference.json", dict(runtime=runtime, first_update=first, second_update=second,
        checkpoint_sha256=digest, rng_after_save=probe, inference_observation=list(observation),
        inference_actions=inference, actor_updated=True, critic_updated=True,
        optimizer_steps_per_update=4, global_batch=63, total_economic_transitions=126,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
        real_sources_opened=False, real_training_authorized=False, native_v5_qualified=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
