"""Fresh-process reconstruction on an assigned LSF node (never local tests)."""

from pathlib import Path
import sys

import torch

from raw_second_fixture import configure, reopen_catalog, trainer


def main():
    configure()
    catalog_path, checkpoint, expected_sha, reference_path, mode = sys.argv[1:]
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    agent = trainer(reopen_catalog(Path(catalog_path)), device="cpu" if mode == "cpu-inference" else "cuda:0")
    if mode == "cpu-inference":
        assert not torch.cuda.is_available()
        agent.load(Path(checkpoint), expected_sha, inference_only=True)
        agent.environment.index = 2
        # Load account state from the exact checkpoint, without restoring a GPU
        # optimizer onto CPU. This branch only verifies frozen policy inference.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["environment"]
        obs = agent.environment.observation()
        account = [float(state["cash"]), *(float(dict(state["holdings"]).get(a, 0)) for a in agent.environment.catalog.asset_ids)]
        tensors = {**obs.tensors, "account_state": torch.tensor([account], dtype=torch.float32)}
        with torch.no_grad():
            actual = agent.algorithm.model(tensors, action_mask=obs.action_mask).distribution.mode()
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
