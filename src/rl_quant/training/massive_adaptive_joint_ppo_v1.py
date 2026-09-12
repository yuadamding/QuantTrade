"""Two-rank FP32 PPO engineering adapter; never a V5 training authority.

One rank-zero CPU collector owns the chronological economic episode and GAE.
DDP partitions optimization samples, not economic time. This adapter is kept
out of the authorizing V5 runner until its source, H100 and study gates pass.
All methods that touch distributed state must be called by BOTH ranks.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from io import BytesIO
import os
from pathlib import Path
import random
import re
import stat

import numpy as np
import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from rl_quant.protocol.canonical_artifact import file_sha256
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
    build_seeded_massive_adaptive_ppo_model_v1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPORolloutV1,
    MassiveAdaptivePPOTrainerV1,
    MassiveAdaptiveRLCheckpointV1,
    _state_receipt,
)
from rl_quant.training.massive_adaptive_rl_checkpoint_authority_v1 import (
    _checkpoint_payload,
    _numpy_rng_payload,
    _parse_checkpoint,
)

SCHEMA = "rl-quant.massive-adaptive-joint-ppo-engineering-v1"
_METRICS = ("policy_loss", "value_loss", "entropy", "approximate_kl", "gradient_norm")
_MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def rank_sample_indices(order: torch.Tensor, rank: int) -> torch.Tensor:
    """Exact 63-sample partition: 32 on rank zero, 31 on rank one."""
    if (type(rank) is not int or rank not in (0, 1) or order.device.type != "cpu"
            or order.dtype != torch.int64 or order.shape != (63,)
            or not torch.equal(order.sort().values, torch.arange(63))):
        raise ValueError("Joint PPO needs one permutation of all 63 samples")
    return order[rank::2]


def weighted_rank_loss(per_sample: torch.Tensor) -> torch.Tensor:
    """DDP averages ranks: 2/63 times a local SUM gives the global MEAN."""
    if per_sample.ndim != 1 or per_sample.numel() not in (31, 32):
        raise ValueError("Joint PPO local sample count must be 32 or 31")
    return per_sample.sum() * (2.0 / 63.0)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


class _Objective(nn.Module):
    """Return tensors through DDP, not a custom distribution object."""

    def __init__(self, model: MassiveAdaptivePPOActorCriticV1) -> None:
        super().__init__()
        self.model = model

    def forward(self, observations, actions):
        output = self.model({"adaptive_state": observations})
        return (output.distribution.log_prob(actions),
                output.distribution.entropy(), output.value)


def _read_checkpoint(path: Path, expected_sha256: str) -> dict:
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or not path.is_absolute() or path.resolve() != path):
        raise ValueError("Joint checkpoint path or expected digest differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Joint checkpoint is not a bounded regular file")
        raw = stream.read(_MAX_CHECKPOINT_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key)
            for key in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("Joint checkpoint bytes changed")
    body = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(body, dict):
        raise ValueError("Joint checkpoint payload differs")
    receipt = body.pop("semantic_sha256", None)
    if (receipt != _state_receipt(body) or body.get("schema") != SCHEMA
            or body.get("source_sha256") != file_sha256(Path(__file__))
            or body.get("real_training_authorized") is not False
            or body.get("native_v5_qualified") is not False
            or body.get("world_size") != 2 or len(body.get("rank_rng", ())) != 2):
        raise ValueError("Joint checkpoint identity or authorization differs")
    checkpoint = _parse_checkpoint(body["collector"])
    if (checkpoint.source_data_qualified or checkpoint.development_rl_training_authorized
            or checkpoint.update_index != body["update_index"]
            or checkpoint.model_state_receipt_sha256 != _state_receipt(body["model"])
            or checkpoint.actor_optimizer_state_receipt_sha256 != _state_receipt(body["actor_optimizer"])
            or checkpoint.critic_optimizer_state_receipt_sha256 != _state_receipt(body["critic_optimizer"])
            or not torch.equal(checkpoint.minibatch_rng_state, body["shuffle_rng"])):
        raise ValueError("Joint checkpoint and economic collector disagree")
    return body


def load_joint_ppo_cpu_checkpoint(path: Path, expected_sha256: str) -> MassiveAdaptiveRLCheckpointV1:
    """Read-only CPU inference/collector state, with NO native authorization."""
    return _parse_checkpoint(_read_checkpoint(path, expected_sha256)["collector"])


class MassiveAdaptiveJointPPOV1:
    """Same actor/critic, seed and fold on two CUDA ranks; no real-data gate.

    ``collector`` is the existing CPU economic trainer on rank zero, and None
    on rank one. Its optimizers are synchronized for checkpoint compatibility
    but are never stepped on the CPU. Caller-owned authority flags are not
    accepted. The adapter currently rejects authorizing collector roots.
    """

    def __init__(self, *, config: MassiveAdaptivePPOConfigV1, trial_id: str,
                 fold_index: int, collector: MassiveAdaptivePPOTrainerV1 | None):
        if (not dist.is_initialized() or dist.get_world_size() != 2
                or dist.get_backend() != "nccl"):
            raise ValueError("Joint PPO requires an initialized two-rank NCCL group")
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", self.rank)
        if (os.environ.get("LOCAL_RANK") != str(self.rank)
                or os.environ.get("LOCAL_WORLD_SIZE") != "2"
                or torch.cuda.device_count() != 2 or torch.cuda.current_device() != self.rank):
            raise ValueError("Joint PPO requires one local rank per assigned CUDA device")
        self.config = config
        self.collector = collector
        error = None
        try:
            config.validate()
            if (any(type(value) is not int for value in (config.rollout_length,
                    config.minibatch_size, config.epochs_per_rollout))
                    or config.rollout_length != 63 or config.minibatch_size != 63
                    or config.epochs_per_rollout != 4
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", trial_id)
                    or type(fold_index) is not int or fold_index not in range(4)):
                raise ValueError("Joint PPO trial, fold or 63-sample/four-epoch budget differs")
            if (not torch.are_deterministic_algorithms_enabled()
                    or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                    or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic
                    or torch.get_num_threads() != 1):
                raise ValueError("Joint PPO deterministic FP32 runtime differs")
            if self.rank == 0:
                if (type(collector) is not MassiveAdaptivePPOTrainerV1
                        or collector.device.type != "cpu" or collector.config != config
                        or collector.update_index != 0 or collector.loss_trace or collector.transition_receipts
                        or collector.training_forecast_authority is not None
                        or collector.fit_environment_authority is not None):
                    raise ValueError("Joint PPO needs a fresh nonauthorizing rank-zero CPU collector")
            elif collector is not None:
                raise ValueError("Only rank zero may own the economic collector")
        except Exception as exc:
            error = str(exc)
        bindings = [None, None]
        dist.all_gather_object(bindings, (error, trial_id, fold_index, asdict(config)))
        if any(row[0] for row in bindings) or bindings[0][1:] != bindings[1][1:]:
            raise ValueError("Joint PPO rank contracts disagree: " + str([row[0] for row in bindings]))
        self.trial_id, self.fold_index = trial_id, fold_index
        self.model = build_seeded_massive_adaptive_ppo_model_v1(seed=config.seed).to(self.device)
        self.ddp = DistributedDataParallel(_Objective(self.model), device_ids=[self.rank],
                                           output_device=self.rank, broadcast_buffers=False)
        self.actor_optimizer = torch.optim.Adam([
            *self.model.actor.parameters(), *self.model.actor_mean.parameters(), self.model.actor_log_std,
        ], lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam([
            *self.model.critic.parameters(), *self.model.value_head.parameters(),
        ], lr=config.critic_learning_rate)
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        self.update_index = 0
        self.loss_trace = []
        self.last_rollout = None
        self._poisoned = False
        prop = torch.cuda.get_device_properties(self.rank)
        profiles = [None, None]
        dist.all_gather_object(profiles, dict(name=prop.name, capability=[prop.major, prop.minor],
                                            memory=prop.total_memory))
        self.runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                            devices=profiles, dtype="float32", tf32=False, autocast=False)
        self._root(self._check_initial_collector)
        self.assert_replicas_equal()

    def _root(self, operation):
        """Propagate rank-zero validation/I/O failures before the next collective."""
        message = [None]
        if self.rank == 0:
            try:
                message[0] = (None, operation())
            except Exception as exc:
                message[0] = (type(exc).__name__ + ": " + str(exc), None)
        dist.broadcast_object_list(message, src=0, device=self.device)
        error, value = message[0]
        if error is not None:
            raise ValueError("Rank-zero joint PPO operation failed: " + error)
        return value

    def _check_initial_collector(self):
        if _state_receipt(self.collector.model.state_dict()) != _state_receipt(self.model.state_dict()):
            raise ValueError("Collector weights differ from the seeded replica")

    def assert_replicas_equal(self) -> str:
        identity = (_state_receipt(self.model.state_dict()),
                    _state_receipt(self.actor_optimizer.state_dict()),
                    _state_receipt(self.critic_optimizer.state_dict()), self.update_index)
        identities = [None, None]
        dist.all_gather_object(identities, identity)
        if identities[0] != identities[1]:
            raise ValueError("Joint PPO replicas or optimizer trajectories differ")
        return identity[0]

    def _collect(self):
        if (self.collector.update_index != self.update_index
                or _state_receipt(self.collector.model.state_dict()) != _state_receipt(self.model.state_dict())):
            raise ValueError("Collector is not synchronized with joint weights")
        self.last_rollout = self.collector.collect_rollout(steps=63)
        return self.last_rollout

    def step(self) -> dict:
        """One chronological CPU trajectory; exactly four joint optimizer steps."""
        self._require_healthy()
        try:
            rollout = self._root(self._collect)
            return self.update(rollout if self.rank == 0 else None)
        except BaseException:
            self._poisoned = True
            raise

    def _require_healthy(self):
        flags = [None, None]
        dist.all_gather_object(flags, self._poisoned)
        if any(flags):
            raise ValueError("Failed joint update requires restoration, not checkpoint publication")

    def update(self, rollout: MassiveAdaptivePPORolloutV1 | None) -> dict:
        """Optimize a rank-zero collector rollout; no second-rank data duplication.

        This numerical surface is not an evidence authority. Production callers
        must use a future source-qualified integration, not supply outcome arrays.
        """
        self._require_healthy()
        try:
            return self._update(rollout)
        except BaseException:
            self._poisoned = True
            raise

    def _update(self, rollout):
        supplied = [None, None]
        profile_ok = (not torch.is_autocast_enabled() and torch.are_deterministic_algorithms_enabled()
                      and not torch.backends.cuda.matmul.allow_tf32
                      and not torch.backends.cudnn.allow_tf32
                      and all(p.dtype == torch.float32 and p.device == self.device for p in self.model.parameters()))
        dist.all_gather_object(supplied, (rollout is not None, profile_ok))
        if supplied != [(True, True), (False, True)]:
            raise ValueError("Only rank zero supplies the global rollout")

        def prepare():
            if type(rollout) is not MassiveAdaptivePPORolloutV1:
                raise ValueError("Joint rollout type differs")
            rollout.validate()
            if (rollout.observations.shape != (63, 90) or rollout.rewards.shape != (63,)
                    or any(value.device.type != "cpu" or value.dtype != torch.float32
                           for value in (rollout.observations, rollout.actions, rollout.advantages,
                                         rollout.old_log_probabilities, rollout.old_values, rollout.returns))
                    or bool(rollout.terminated[:-1].any())
                    or rollout.transition_decision_session_dates != tuple(sorted(set(rollout.transition_decision_session_dates)))):
                raise ValueError("Joint rollout must be one chronological 63-session CPU trajectory")
            advantage = rollout.advantages
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
            return dict(observations=rollout.observations, actions=rollout.actions,
                        old_log_probabilities=rollout.old_log_probabilities, old_values=rollout.old_values,
                        returns=rollout.returns, advantages=advantage)

        batch = {key: tensor.to(self.device) for key, tensor in self._root(prepare).items()}
        self.ddp.train()
        total = torch.zeros(5, dtype=torch.float64, device=self.device)
        for _ in range(4):
            order, shuffle_state = self._root(lambda: (
                torch.randperm(63, generator=self.shuffle_rng), self.shuffle_rng.get_state()))
            self.shuffle_rng.set_state(shuffle_state)
            indices = rank_sample_indices(order, self.rank).to(self.device)
            logp, entropy, value = self.ddp(batch["observations"][indices], batch["actions"][indices])
            old_logp, old_value = batch["old_log_probabilities"][indices], batch["old_values"][indices]
            ratio = (logp - old_logp).clamp(-20.0, 20.0).exp()
            advantage, target = batch["advantages"][indices], batch["returns"][indices]
            policy_loss = -torch.minimum(ratio * advantage,
                ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * advantage)
            clipped_value = old_value + (value - old_value).clamp(
                -self.config.value_clip_range, self.config.value_clip_range)
            value_loss = 0.5 * torch.maximum((value - target).square(), (clipped_value - target).square())
            loss = weighted_rank_loss(policy_loss - self.config.entropy_coefficient * entropy
                                      + self.config.value_coefficient * value_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()  # DDP reduction completes before clipping or either optimizer step.
            finite = torch.tensor(int(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                         for p in self.model.parameters())), device=self.device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise ValueError("Joint PPO has a missing or nonfinite synchronized gradient")
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.maximum_gradient_norm,
                                            error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            sums = torch.stack([x.detach().double().sum() for x in
                                (policy_loss, value_loss, entropy, old_logp - logp)])
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            total[:4] += sums / 63
            total[4] += norm.detach().double()
        metrics = dict(zip(_METRICS, (total / 4).cpu().tolist(), strict=True))
        self.update_index += 1
        self.loss_trace.append(tuple(metrics.values()))
        model_sha = self.assert_replicas_equal()
        self._root(self._sync_collector)
        return dict(metrics=metrics, update_index=self.update_index, global_samples=63,
                    samples_per_rank=[32, 31], optimizer_steps=4, model_sha256=model_sha,
                    real_training_authorized=False, native_v5_qualified=False)

    def _sync_collector(self):
        self.collector.model.load_state_dict(_cpu(self.model.state_dict()))
        self.collector.actor_optimizer.load_state_dict(_cpu(self.actor_optimizer.state_dict()))
        self.collector.critic_optimizer.load_state_dict(_cpu(self.critic_optimizer.state_dict()))
        self.collector.minibatch_rng.set_state(self.shuffle_rng.get_state())
        self.collector.update_index = self.update_index
        self.collector.loss_trace = list(self.loss_trace)

    def save(self, path: Path) -> str:
        """Rank-zero-only, create-only safe-torch checkpoint after a full update."""
        self._require_healthy()
        self.assert_replicas_equal()
        rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device),
                   python=random.getstate(), numpy=_numpy_rng_payload(np.random.get_state()))
        ranks = [None, None]
        dist.all_gather_object(ranks, rng)

        def publish():
            checkpoint = self.collector.checkpoint()
            if checkpoint.source_data_qualified:
                raise ValueError("Engineering adapter cannot publish a native authorizing checkpoint")
            body = dict(schema=SCHEMA, source_sha256=file_sha256(Path(__file__)), world_size=2,
                config=asdict(self.config), trial_id=self.trial_id, fold_index=self.fold_index,
                runtime=self.runtime, model=_cpu(self.model.state_dict()),
                actor_optimizer=_cpu(self.actor_optimizer.state_dict()),
                critic_optimizer=_cpu(self.critic_optimizer.state_dict()), rank_rng=ranks,
                shuffle_rng=self.shuffle_rng.get_state(), update_index=self.update_index,
                collector=_checkpoint_payload(checkpoint, source_data_qualified=False),
                real_training_authorized=False, native_v5_qualified=False)
            body["semantic_sha256"] = _state_receipt(body)
            buffer = BytesIO()
            torch.save(body, buffer)
            raw = buffer.getvalue()
            if len(raw) > _MAX_CHECKPOINT_BYTES or not path.is_absolute() or path.resolve() != path:
                raise ValueError("Joint checkpoint path or size differs")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return hashlib.sha256(raw).hexdigest()

        return self._root(publish)

    def restore(self, path: Path, expected_sha256: str) -> None:
        """Restore identical hardware/runtime, both rank RNGs, and economic state."""
        body = self._root(lambda: _read_checkpoint(path, expected_sha256))
        if (body["config"] != asdict(self.config) or body["trial_id"] != self.trial_id
                or body["fold_index"] != self.fold_index or body["runtime"] != self.runtime):
            raise ValueError("Joint checkpoint trial, fold, configuration or runtime differs")
        self.model.load_state_dict(body["model"])
        self.actor_optimizer.load_state_dict(body["actor_optimizer"])
        self.critic_optimizer.load_state_dict(body["critic_optimizer"])
        self.shuffle_rng.set_state(body["shuffle_rng"])
        checkpoint = _parse_checkpoint(body["collector"])
        self.update_index = body["update_index"]
        self.loss_trace = list(checkpoint.loss_trace)
        self._root(lambda: self.collector.restore(checkpoint))
        rng = body["rank_rng"][self.rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state["bit_generator"], state["keys"].numpy().astype(np.uint32),
                             state["position"], state["has_gauss"], state["cached_gaussian"]))
        self.assert_replicas_equal()
        self._poisoned = False
