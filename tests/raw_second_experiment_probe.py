"""Fresh-process persisted experiment probe; called only by LSF acceptance."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from raw_second_fixture import START, configure, event_coverage, make_catalog, reopen_catalog, save_catalog, trainer
from rl_quant.datasets.raw_second_economics_v1 import SecondEconomicInputs
from rl_quant.workflows.raw_second_experiment_v1 import run_raw_second_experiment, verify_raw_second_experiment


def main():
    configure()
    mode, location = sys.argv[1:]
    root = Path(location)
    if mode == "prepare":
        catalogs = tuple(make_catalog(root / role, start=START + i * 86_400_000, falling=role == "test")
                         for i, role in enumerate(("train", "validation", "test")))
        for role, catalog in zip(("train", "validation", "test"), catalogs, strict=True):
            save_catalog(catalog, root / (role + ".json"))
        events = event_coverage(root / "events", catalogs)
        agent = trainer(catalogs[0])
        result = run_raw_second_experiment(training=catalogs[0], validation=catalogs[1], test=catalogs[2],
            economic_inputs=events, output=root / "run", device="cuda:0", model_config=agent.algorithm.model.config,
            execution_config=replace(agent.environment.config, execution_session="regular-only", order_expiry="session-close"),
            ppo_config=agent.algorithm.config, rollout_steps=2)
        (root / "replay.json").write_text(json.dumps(dict(report_sha256=result["report_sha256"], events=asdict(events))))
        assert result["execution_complete"] and not result["positive_test_net_return"]
    elif mode == "verify":
        config = json.loads((root / "replay.json").read_text())
        result = verify_raw_second_experiment(output=root / "run", expected_report_sha256=config["report_sha256"],
            training=reopen_catalog(root / "train.json"), validation=reopen_catalog(root / "validation.json"),
            test=reopen_catalog(root / "test.json"), economic_inputs=SecondEconomicInputs(**config["events"]), device="cuda:0")
        assert result["nonmaterializing"] and result["frozen_report_replayed"] and not result["positive_test_net_return"]
        print(json.dumps(result, sort_keys=True))
    else:
        raise ValueError("Unknown probe mode")


if __name__ == "__main__":
    main()
