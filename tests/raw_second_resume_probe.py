"""Fresh-process reconstruction on an assigned LSF node (never local tests)."""

from pathlib import Path
import sys

import torch

from raw_second_fixture import configure, make_catalog, reopen_catalog, save_catalog, trainer
from rl_quant.evaluation.raw_second_policy_v1 import load_frozen_raw_second_policy


def main():
    configure()
    if sys.argv[1] == "prepare":
        import json
        root = Path(sys.argv[2])
        catalog = make_catalog(root / "sources")
        save_catalog(catalog, root / "catalog.json")
        agent = trainer(catalog)
        agent.update(agent.collect(steps=2))
        checkpoint_sha = agent.save(root / "resume.pt")
        frozen_sha = agent.freeze(root / "frozen.pt")
        deterministic = agent.algorithm.act(agent.environment.observation(), deterministic=True).action.cpu().tolist()
        batch = agent.collect(steps=2)
        metrics = agent.update(batch)
        torch.save(dict(model={k: p.detach().cpu() for k, p in agent.algorithm.model.named_parameters()},
                        rewards=batch.as_batch().rewards.cpu(), actions=batch.as_batch().actions.cpu(),
                        metrics=metrics, deterministic=deterministic), root / "reference.pt")
        (root / "hashes.json").write_text(json.dumps(dict(checkpoint_sha=checkpoint_sha, frozen_sha=frozen_sha)))
        return
    catalog_path, checkpoint, expected_sha, reference_path, mode, frozen, frozen_sha = sys.argv[1:]
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    agent = trainer(reopen_catalog(Path(catalog_path)), device="cpu" if mode == "cpu-inference" else "cuda:0")
    if mode == "cpu-inference":
        assert not torch.cuda.is_available()
        policy = load_frozen_raw_second_policy(Path(frozen), frozen_sha, catalog=agent.environment.catalog, device="cpu")
        agent.environment.index = 2
        # Load account state from the exact checkpoint, without restoring a GPU
        # optimizer onto CPU. This branch only verifies frozen policy inference.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["environment"]
        obs = agent.environment.observation()
        account = [float(state["cash"]), *(float(dict(state["holdings"]).get(a, 0)) for a in agent.environment.catalog.asset_ids)]
        tensors = {**obs.tensors, "account_state": torch.tensor([account], dtype=torch.float32)}
        with torch.no_grad():
            actual = policy.model(tensors, action_mask=obs.action_mask).distribution.mode()
        torch.testing.assert_close(actual, torch.tensor(reference["deterministic"]), atol=2e-5, rtol=2e-4)
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        agent.load(Path(checkpoint), expected_sha)
        batch = agent.collect(steps=2)
        metrics = agent.update(batch)
        assert metrics == reference["metrics"]
        torch.testing.assert_close(batch.as_batch().actions.cpu(), reference["actions"], atol=0, rtol=0)
        torch.testing.assert_close(batch.as_batch().rewards.cpu(), reference["rewards"], atol=0, rtol=0)
        for name, parameter in agent.algorithm.model.named_parameters():
            torch.testing.assert_close(parameter.detach().cpu(), reference["model"][name], atol=0, rtol=0)
    print(mode + " passed")


if __name__ == "__main__":
    main()
