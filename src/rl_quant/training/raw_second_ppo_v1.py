"""Raw-reference PPO integration using the existing clipped-PPO implementation.

This bounded engineering route does not grant native-V5 data authorization or
issue a profitability result. All market encoder parameters join the optimizer.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import io
from pathlib import Path

import torch

from rl_quant.datasets.massive_raw_seconds_v1 import SCHEMA, _read, _write
from rl_quant.envs.raw_second_portfolio_v1 import LEDGER_SCHEMA, RawSecondPortfolioEnv, SecondFill, SecondLedgerMark
from rl_quant.execution.qt200_aggregate_execution_v1 import Book, Receivable
from rl_quant.models.raw_second_policy_v1 import RawSecondActorCritic
from rl_quant.rl.ppo import PPOConfig, RecurrentPPO
from rl_quant.rl.trajectory import OnPolicyTrajectoryBuffer


class RawSecondPPOTrainer:
    def __init__(self, model: RawSecondActorCritic, environment: RawSecondPortfolioEnv,
                 config: PPOConfig | None = None):
        if model._frozen_evaluation or any(not p.requires_grad for p in model.parameters()):
            raise ValueError("Frozen evaluation policies have no optimizer/update route")
        if model.catalog.identity != environment.catalog.identity:
            raise ValueError("Model and ledger must share the exact raw references")
        self.algorithm = RecurrentPPO(model, config)
        self.environment = environment
        if self.algorithm.device != environment.device:
            raise ValueError("Collection and optimizer devices differ")
        if {id(p) for g in self.algorithm.optimizer.param_groups for p in g["params"]} != {id(p) for p in model.parameters()}:
            raise ValueError("PPO must optimize the entire encoder, actor and critic")

    def runtime_profile(self) -> dict:
        device = self.algorithm.device
        return dict(torch=str(torch.__version__), cuda=torch.version.cuda, device_type=device.type,
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    capability=list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    tf32=torch.backends.cuda.matmul.allow_tf32, threads=torch.get_num_threads())

    def collect(self, *, steps: int, deterministic: bool = False) -> OnPolicyTrajectoryBuffer:
        if type(steps) is not int or steps <= 0 or self.environment.index + steps >= len(self.environment.catalog.windows):
            raise ValueError("Rollout exceeds the raw episode")
        buffer = OnPolicyTrajectoryBuffer(horizon=steps, num_envs=1)
        for _ in range(steps):
            observation = self.environment.observation()
            action = self.algorithm.act(observation, deterministic=deterministic)
            transition = self.environment.step(action)
            next_value = torch.zeros(1, device=self.algorithm.device) if bool(transition.terminated.all()) else self.algorithm.value(transition.next_observation)
            buffer.add(transition, value=action.extras["value"], next_value=next_value)
        buffer.compute_gae(gae_lambda=0.95)
        return buffer

    def update(self, buffer: OnPolicyTrajectoryBuffer) -> dict:
        # One decision per learning sequence keeps raw recomputation bounded;
        # advantages were already calculated on the full chronological rollout.
        return dict(self.algorithm.update(buffer.recurrent_sequences(sequence_length=1)))

    def save(self, path: Path) -> str:
        env = self.environment
        state = dict(schema=SCHEMA, ledger_schema=LEDGER_SCHEMA, catalog=env.catalog.identity, model_contract=self.algorithm.model.get_extra_state(),
                     execution_config=asdict(env.config), splits=[asdict(e) for e in env.splits],
                     dividends=[asdict(e) for e in env.dividends], sessions=[asdict(s) for s in env.sessions], algorithm=self.algorithm.state_dict(),
                     cpu_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(self.algorithm.device) if self.algorithm.device.type == "cuda" else None,
                     device_type=self.algorithm.device.type, runtime=self.runtime_profile(),
                     environment=dict(index=env.index, cash=str(env.book.cash),
                                      holdings=[(a, str(q)) for a, q in env.book.holdings],
                                      receivables=[dict(event_id=r.event_id, instrument=r.instrument, payable_date=r.payable_date, amount=str(r.amount)) for r in env.book.receivables],
                                      applied_events=env.book.applied_events, action_date=env.book.action_date,
                                      accounting_mark_state={a: {**asdict(m), "price": str(m.price)} for a, m in env.accounting_mark_state.items()},
                                      known_mark_state={a: {**asdict(m), "price": str(m.price)} for a, m in env.known_mark_state.items()},
                                      peak=str(env.peak), risk_halted=env.risk_halted, equity=str(env.current_equity),
                                      fills=[asdict(f) for f in env.fills], audit=env.audit))
        # Decimal event terms are serialized as strings, not pickle globals.
        for family in ("splits", "dividends"):
            state[family] = [{k: str(v) if isinstance(v, Decimal) else v for k, v in e.items()} for e in state[family]]
        stream = io.BytesIO()
        torch.save(state, stream)
        body = stream.getvalue()
        _write(path, body)
        return sha256(body).hexdigest()

    def load(self, path: Path, expected_sha256: str, *, inference_only: bool = False) -> None:
        if inference_only:
            raise ValueError("Use load_frozen_raw_second_policy; resume is never an evaluation interface")
        state = torch.load(io.BytesIO(_read(path, expected_sha256)), map_location=self.algorithm.device, weights_only=True)
        env = self.environment
        if (state["schema"] != SCHEMA or state.get("ledger_schema") != LEDGER_SCHEMA or state["catalog"] != env.catalog.identity
                or state["model_contract"] != self.algorithm.model.get_extra_state()
                or state["execution_config"] != asdict(env.config)
                or state.get("sessions") != [asdict(s) for s in env.sessions]):
            raise ValueError("Checkpoint source/model/ledger configuration differs")
        for family in ("splits", "dividends"):
            expected = [{k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(e).items()} for e in getattr(env, family)]
            if state[family] != expected:
                raise ValueError("Checkpoint corporate-action population differs")
        if state["runtime"] != self.runtime_profile():
            raise ValueError("Optimizer resume requires the frozen device profile; CPU is inference-only")
        state["algorithm"]["minibatch_rng_state"] = state["algorithm"]["minibatch_rng_state"].cpu()
        self.algorithm.load_state_dict(state["algorithm"])
        book = state["environment"]
        env.index = book["index"]
        env.book = Book(Decimal(book["cash"]), tuple((a, Decimal(q)) for a, q in book["holdings"]),
                        tuple(Receivable(r["event_id"], r["instrument"], r["payable_date"], Decimal(r["amount"])) for r in book["receivables"]),
                        tuple(book["applied_events"]), book["action_date"])
        env.accounting_mark_state = {a: SecondLedgerMark(**{**m, "price": Decimal(m["price"])}) for a, m in book["accounting_mark_state"].items()}
        env.known_mark_state = {a: SecondLedgerMark(**{**m, "price": Decimal(m["price"])}) for a, m in book["known_mark_state"].items()}
        env.peak, env.current_equity = Decimal(book["peak"]), Decimal(book["equity"])
        env.risk_halted = book["risk_halted"]
        env.fills = [SecondFill(**f) for f in book["fills"]]
        env.audit = book["audit"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        if self.algorithm.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.algorithm.device)
        env.observation()  # read-only source revalidation before further work

    def freeze(self, path: Path) -> str:
        """Export tensor-only policy weights; no optimizer, RNG or ledger state."""
        from rl_quant.evaluation.raw_second_policy_v1 import write_frozen_raw_second_policy

        return write_frozen_raw_second_policy(self.algorithm.model, path,
            training_provenance=dict(catalog_sha256=self.environment.catalog.identity,
                last_reward_timestamp_ms=self.environment.catalog.windows[self.environment.index].decision_ms,
                optimizer_updates=self.algorithm.update_count, ppo_config=asdict(self.algorithm.config)))
