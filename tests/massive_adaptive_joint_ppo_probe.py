"""Real two-CUDA-rank optimizer regression, not a native/source qualification.

The existing numerical market fixture supplies synthetic upstream observations.
Actors, PPO updates, the collector, compiler, fills and checkpoint code run
unchanged. No result from this file authorizes real data or an outer study.
"""

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import distributed as dist

from rl_quant.rl.massive_adaptive_ppo_policy_v1 import build_seeded_massive_adaptive_ppo_model_v1
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1, MassiveAdaptivePPOTrainerV1, _state_receipt,
)
from rl_quant.training.massive_adaptive_joint_ppo_v1 import (
    MassiveAdaptiveJointPPOV1, load_joint_ppo_cpu_checkpoint, weighted_rank_loss,
)
from test_massive_adaptive_rl_v5_vertical import _market_fixture


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")


def collector(config):
    return MassiveAdaptivePPOTrainerV1(environment=_market_fixture(0).environment(20.0),
        model=build_seeded_massive_adaptive_ppo_model_v1(seed=config.seed), config=config, device="cpu")


def rng_sample():
    return dict(cpu=torch.rand(3).tolist(), cuda=torch.rand(3, device="cuda").cpu().tolist(),
                numpy=np.random.random(3).tolist(), python=[random.random() for _ in range(3)])


def gradient_probe(adapter, change_rank_one):
    """Independent full-batch derivative vs both actual DDP actor/critic ranks."""
    device = adapter.device
    full_x = torch.sin(torch.arange(63 * 90, device=device).reshape(63, 90).float() / 97)
    full_actions = torch.tanh(torch.cos(torch.arange(630, device=device).reshape(63, 10).float()) * 0.2)
    labels = torch.arange(63, device=device).float() / 63
    if change_rank_one:
        labels[1::2] += 2.0
    idx = torch.arange(adapter.rank, 63, 2, device=device)
    adapter.model.zero_grad(set_to_none=True)
    logp, entropy, value = adapter.ddp(full_x[idx], full_actions[idx])
    weighted_rank_loss(0.1 * logp + (value - labels[idx]).square() + 0.01 * entropy).backward()
    actual = torch.cat([p.grad.detach().flatten() for p in adapter.model.parameters()])
    identity = _state_receipt(actual)
    identities = [None, None]
    dist.all_gather_object(identities, identity)
    assert identities[0] == identities[1]
    if adapter.rank == 0:
        reference = build_seeded_massive_adaptive_ppo_model_v1(seed=adapter.config.seed).to(device)
        reference.load_state_dict(adapter.model.state_dict())
        output = reference({"adaptive_state": full_x})
        # Do not reuse the adapter's loss-normalization helper for the oracle.
        loss = (0.1 * output.distribution.log_prob(full_actions)
                + (output.value - labels).square() + 0.01 * output.distribution.entropy()).sum() / 63
        loss.backward()
        expected = torch.cat([p.grad.detach().flatten() for p in reference.parameters()])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    return actual.clone()


def main(mode, directory):
    root = Path(directory)
    if mode == "cpu":
        info = json.loads((root / "reference.json").read_text())
        checkpoint = load_joint_ppo_cpu_checkpoint(root / "joint.pt", info["checkpoint_sha256"])
        model = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
        model.load_state_dict(checkpoint.model_state)
        with torch.no_grad():
            actions = model({"adaptive_state": torch.zeros(1, 90)}).distribution.deterministic_action()
        assert actions.shape == (1, 10) and torch.isfinite(actions).all()
        assert not checkpoint.source_data_qualified and not checkpoint.development_rl_training_authorized
        print(json.dumps(dict(cpu_reload=True, model_sha256=checkpoint.model_state_receipt_sha256,
                              economic_cursor=checkpoint.environment_state.chronology_cursor,
                              native_v5_qualified=False)), flush=True)
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        config = MassiveAdaptivePPOConfigV1(rollout_length=63, minibatch_size=63)
        cpu = collector(config) if rank == 0 else None
        adapter = MassiveAdaptiveJointPPOV1(config=config, trial_id="synthetic-joint-17", fold_index=0,
                                          collector=cpu)
        if mode == "failure":
            dist.barrier()
            if rank == 1:
                os._exit(23)
            adapter.step()
            write(root / "incorrectly-survived.json", {"survived": True})
            raise AssertionError("Rank zero survived rank-one failure")
        if mode == "resume":
            expected = json.loads((root / "reference.json").read_text())
            adapter.restore(root / "joint.pt", expected["checkpoint_sha256"])
            probe = rng_sample()
            probes = [None, None]
            dist.all_gather_object(probes, probe)
            assert probes == expected["rng_after_save"]
            result = adapter.step()
            assert result["model_sha256"] == expected["second_update"]["model_sha256"]
            assert result["metrics"] == expected["second_update"]["metrics"]
            if rank == 0:
                state = cpu.environment.state
                assert state.semantic_receipt_sha256 == expected["second_environment_sha256"]
                assert list(adapter.last_rollout.transition_receipts) == expected["second_transition_receipts"]
                assert _state_receipt(adapter.last_rollout.actions) == expected["second_actions_sha256"]
                assert _state_receipt(adapter.actor_optimizer.state_dict()) == expected["second_actor_optimizer_sha256"]
                assert _state_receipt(adapter.critic_optimizer.state_dict()) == expected["second_critic_optimizer_sha256"]
                write(root / "resume.json", dict(fresh_process=True, exact_model=True, exact_optimizers=True,
                    exact_rank_rng=True, exact_actions=True, exact_transitions=True, exact_economic_state=True,
                    native_v5_qualified=False))
            return
        assert mode == "qualify"
        baseline = gradient_probe(adapter, False)
        changed = gradient_probe(adapter, True)
        assert float((baseline - changed).abs().max()) > 0.01
        first = adapter.step()

        def reference_update():
            model = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
            reference = MassiveAdaptivePPOTrainerV1(environment=_market_fixture(0).environment(20.0),
                                                    model=model, config=config, device="cuda:0")
            fields = {key: getattr(adapter.last_rollout, key).to("cuda:0") for key in (
                "observations", "actions", "old_log_probabilities", "old_values", "rewards",
                "advantages", "returns", "terminated")}
            reference.update(replace(adapter.last_rollout, **fields))
            maximum = 0.0
            for name, tensor in reference.model.state_dict().items():
                other = adapter.model.state_dict()[name]
                torch.testing.assert_close(other, tensor, rtol=3e-4, atol=2e-5)
                maximum = max(maximum, float((other - tensor).abs().max()))
            assert first["model_sha256"] != _state_receipt(build_seeded_massive_adaptive_ppo_model_v1(seed=17).state_dict())
            assert cpu.environment.state.chronology_cursor == 63
            assert len(set(cpu.transition_decision_session_dates)) == 63
            assert all(float(row["step"]) == 4 for row in adapter.actor_optimizer.state_dict()["state"].values())
            assert all(float(row["step"]) == 4 for row in adapter.critic_optimizer.state_dict()["state"].values())
            return maximum

        maximum_error = adapter._root(reference_update)
        digest = adapter.save(root / "joint.pt")
        try:
            adapter.save(root / "joint.pt")
        except ValueError as exc:
            assert "FileExistsError" in str(exc)
        else:
            raise AssertionError("Checkpoint was overwritten")
        probes = [None, None]
        dist.all_gather_object(probes, rng_sample())
        second = adapter.step()
        if rank == 0:
            assert hashlib.sha256((root / "joint.pt").read_bytes()).hexdigest() == digest
            assert len(cpu.transition_receipts) == 126
            assert len(set(cpu.transition_decision_session_dates)) == 126
            write(root / "reference.json", dict(checkpoint_sha256=digest, first_update=first,
                second_update=second, rank_one_changes_gradient=True, replicas_equal=True,
                full_batch_parameter_max_error=maximum_error, rng_after_save=probes,
                second_environment_sha256=cpu.environment.state.semantic_receipt_sha256,
                second_transition_receipts=list(adapter.last_rollout.transition_receipts),
                second_actions_sha256=_state_receipt(adapter.last_rollout.actions),
                second_actor_optimizer_sha256=_state_receipt(adapter.actor_optimizer.state_dict()),
                second_critic_optimizer_sha256=_state_receipt(adapter.critic_optimizer.state_dict()),
                total_economic_transitions=126, real_training_authorized=False, native_v5_qualified=False))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
