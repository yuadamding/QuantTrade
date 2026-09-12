"""Portable, immutable raw-second inference, separate from exact PPO resume."""

from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondCatalog, _read, _write, digest
from rl_quant.models.raw_second_policy_v1 import RawSecondActorCritic, RawSecondModelConfig
from rl_quant.rl.types import ActionBatch, ObservationBatch

POLICY_SCHEMA = "rl-quant.raw-second-frozen-policy-v1"


def parameter_hash(model: RawSecondActorCritic) -> str:
    value = sha256()
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite/non-FP32 frozen parameter")
        value.update(digest((name, list(tensor.shape), str(tensor.dtype))).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def write_frozen_raw_second_policy(model: RawSecondActorCritic, path: Path, *, training_provenance: dict) -> str:
    if training_provenance["catalog_sha256"] != model.catalog.identity:
        raise ValueError("Training provenance differs from model catalog")
    state = dict(schema=POLICY_SCHEMA, policy_contract=model.policy_contract(),
                 training_provenance=training_provenance, parameter_sha256=parameter_hash(model),
                 weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k != "_extra_state"})
    stream = io.BytesIO()
    torch.save(state, stream)
    body = stream.getvalue()
    _write(path, body)
    return sha256(body).hexdigest()


class FrozenRawSecondPolicy:
    """No optimizer or update method; detect parameter mutation before acting."""

    def __init__(self, model: RawSecondActorCritic, *, artifact_sha256: str,
                 expected_parameters: str, provenance: dict):
        model.freeze_for_evaluation()
        self.model = model
        self.artifact_sha256 = artifact_sha256
        self.parameter_sha256 = expected_parameters
        self.training_provenance = provenance
        self.validate_unchanged()

    def validate_unchanged(self) -> None:
        if (self.model.training or not self.model._frozen_evaluation
                or any(p.requires_grad for p in self.model.parameters())
                or parameter_hash(self.model) != self.parameter_sha256):
            raise ValueError("Frozen policy changed during evaluation")

    @torch.inference_mode()
    def act(self, observation: ObservationBatch) -> ActionBatch:
        self.validate_unchanged()
        result = self.model(observation.tensors, action_mask=observation.action_mask)
        requested = result.distribution.mode()
        return ActionBatch(requested, log_prob=result.distribution.log_prob(requested),
                           entropy=result.distribution.entropy(), extras={"value": result.value})


def load_frozen_raw_second_policy(path: Path, expected_sha256: str, *,
                                  catalog: RawSecondCatalog, device: str | torch.device) -> FrozenRawSecondPolicy:
    state = torch.load(io.BytesIO(_read(path, expected_sha256)), map_location="cpu", weights_only=True)
    if set(state) != {"schema", "policy_contract", "training_provenance", "parameter_sha256", "weights"} or state["schema"] != POLICY_SCHEMA:
        raise ValueError("Expected a frozen policy, not a resume checkpoint")
    # Constructing an inference replica must not perturb future collection RNG.
    with torch.random.fork_rng(devices=[]):
        model = RawSecondActorCritic(catalog, RawSecondModelConfig(**state["policy_contract"]["model"]))
    if state["policy_contract"] != model.policy_contract():
        raise ValueError("Frozen policy architecture, raw semantics or instrument order differs")
    if any(not isinstance(v, torch.Tensor) for v in state["weights"].values()) or "_extra_state" in state["weights"]:
        raise ValueError("Frozen policy contains non-weight state")
    # Only after exact portable compatibility checks, bind the new inference
    # catalog. The original training catalog remains immutable provenance.
    model.load_state_dict({**state["weights"], "_extra_state": model.get_extra_state()}, strict=True)
    model.to(device)
    return FrozenRawSecondPolicy(model, artifact_sha256=expected_sha256,
                                 expected_parameters=state["parameter_sha256"], provenance=state["training_provenance"])
