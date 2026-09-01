"""Package-owned multi-block PPO orchestration for adaptive profitability.

The single-environment trainer is intentionally small.  This runner is the
authoritative layer that consumes every chronological block committed by a
causal RL forecast authority, binds each block to its matching environment,
and carries only learning state across non-contiguous forecast archives.
Economic books carry across block boundaries only while the blocks are exact
consecutive slices of the same environment chronology.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, replace
import random

import numpy as np
import torch

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MassiveAdaptivePPOActorCriticV1,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
    MassiveAdaptivePPORolloutV1,
    MassiveAdaptivePPOTrainerV1,
    MassiveAdaptiveRLCheckpointV1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_protocol_v1 import (
    MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
)
from rl_quant.training.massive_adaptive_rl_fit_environment_authority_v1 import (
    MassiveAdaptiveRLFitEnvironmentAuthorityV1,
)
from rl_quant.training.massive_adaptive_economic_continuity_authority_v1 import (
    MassiveAdaptiveEconomicContinuityAuthorityV1,
    build_massive_adaptive_economic_continuity_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)


MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA = (
    "rl-quant.massive-adaptive-ppo-block-runtime-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prequential-ppo-checkpoint-v1"
)
MASSIVE_ADAPTIVE_PPO_TRAINING_RUN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-ppo-training-run-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SOURCE_SHA256 = file_sha256(__file__)
MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SPEC_SHA256 = semantic_sha256(
    {
        "blocks": "every-authorized-block-in-order",
        "learning_state": "carried-across-all-blocks",
        "economic_state": "carried-across-compatible-consecutive-forecast-refits",
        "internal_liquidation": False,
        "checkpoint": (
            "block-index-cursor-completed-continuity-inventory-and-ppo-state"
        ),
        "validation_access": False,
        "outer_access": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptivePrequentialPPORunnerV1Error(ValueError):
    """The block order, environment lineage, or exact runner state differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptivePrequentialPPORunnerV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePPOBlockRuntimeV1:
    block_index: int
    block_receipt_sha256: str
    forecast_archive_receipt_sha256: str
    inference_plan_receipt_sha256: str
    calibration_receipt_sha256: str
    environment_source_inventory_sha256: str
    fit_environment_authority_receipt_sha256: str | None
    forecast_session_dates: tuple[str, ...]
    environment_start_cursor: int
    environment_stop_cursor: int
    semantic_receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA
            or self.block_index < 0
            or not self.forecast_session_dates
            or self.forecast_session_dates
            != tuple(sorted(set(self.forecast_session_dates)))
            or self.environment_start_cursor < 0
            or self.environment_stop_cursor
            != self.environment_start_cursor + len(self.forecast_session_dates)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "adaptive PPO block runtime differs"
            )
        for value in (
            self.block_receipt_sha256,
            self.forecast_archive_receipt_sha256,
            self.inference_plan_receipt_sha256,
            self.calibration_receipt_sha256,
            self.environment_source_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive PPO block runtime", value)
        if self.fit_environment_authority_receipt_sha256 is not None:
            _digest(
                "adaptive PPO fit environment authority",
                self.fit_environment_authority_receipt_sha256,
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePrequentialPPOCheckpointV1:
    training_forecast_authority_receipt_sha256: str
    rl_chronology_authority_receipt_sha256: str
    block_inventory_sha256: str
    block_runtime_inventory_sha256: str
    continuity_authority_receipts: tuple[str, ...]
    continuity_authority_inventory_sha256: str
    current_block_index: int
    current_block_receipt_sha256: str
    current_calibration_receipt_sha256: str
    current_environment_source_inventory_sha256: str
    within_block_chronology_cursor: int
    completed_block_receipts: tuple[str, ...]
    completed_block_inventory_sha256: str
    transition_receipts: tuple[str, ...]
    transition_source_data_qualified: tuple[bool, ...]
    transition_inventory_sha256: str
    fit_environment_authority_receipts: tuple[str, ...]
    fit_environment_authority_inventory_sha256: str
    ppo_checkpoint: MassiveAdaptiveRLCheckpointV1
    training_complete: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    exact_resume_authorized: bool
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "ppo_checkpoint",
                "semantic_receipt_sha256",
                "exact_resume_authorized",
                "development_rl_training_authorized",
            }
        } | {
            "ppo_checkpoint_receipt_sha256": self.ppo_checkpoint.semantic_receipt_sha256
        }

    def validate(self) -> None:
        self.ppo_checkpoint.validate()
        expected_authorized = self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_V1_SCHEMA
            or self.current_block_index < 0
            or self.within_block_chronology_cursor < 0
            or self.completed_block_inventory_sha256
            != semantic_sha256(self.completed_block_receipts)
            or self.transition_inventory_sha256
            != semantic_sha256(self.transition_receipts)
            or len(self.transition_source_data_qualified)
            != len(self.transition_receipts)
            or any(
                not isinstance(value, bool)
                for value in self.transition_source_data_qualified
            )
            or self.fit_environment_authority_inventory_sha256
            != semantic_sha256(self.fit_environment_authority_receipts)
            or self.fit_environment_authority_receipts
            != tuple(dict.fromkeys(self.fit_environment_authority_receipts))
            or self.continuity_authority_inventory_sha256
            != semantic_sha256(self.continuity_authority_receipts)
            or self.ppo_checkpoint.training_forecast_authority_receipt_sha256
            != self.training_forecast_authority_receipt_sha256
            or self.ppo_checkpoint.environment_state.chronology_cursor
            != self.within_block_chronology_cursor
            or self.exact_resume_authorized != expected_authorized
            or self.development_rl_training_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "adaptive prequential PPO checkpoint differs"
            )
        for value in (
            self.training_forecast_authority_receipt_sha256,
            self.rl_chronology_authority_receipt_sha256,
            self.block_inventory_sha256,
            self.block_runtime_inventory_sha256,
            *self.continuity_authority_receipts,
            self.continuity_authority_inventory_sha256,
            self.current_block_receipt_sha256,
            self.current_calibration_receipt_sha256,
            self.current_environment_source_inventory_sha256,
            *self.completed_block_receipts,
            self.completed_block_inventory_sha256,
            *self.transition_receipts,
            self.transition_inventory_sha256,
            *self.fit_environment_authority_receipts,
            self.fit_environment_authority_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive prequential PPO checkpoint", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePPOTrainingRunV1:
    training_forecast_authority_receipt_sha256: str
    rl_chronology_authority_receipt_sha256: str
    block_runtime_inventory_sha256: str
    continuity_authority_receipts: tuple[str, ...]
    continuity_authority_inventory_sha256: str
    completed_block_receipts: tuple[str, ...]
    completed_block_inventory_sha256: str
    transition_receipts: tuple[str, ...]
    transition_source_data_qualified: tuple[bool, ...]
    transition_inventory_sha256: str
    fit_environment_authority_receipts: tuple[str, ...]
    fit_environment_authority_inventory_sha256: str
    final_checkpoint_receipt_sha256: str
    update_count: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PPO_TRAINING_RUN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {"semantic_receipt_sha256", "development_rl_training_authorized"}
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_PPO_TRAINING_RUN_V1_SCHEMA
            or not self.completed_block_receipts
            or self.completed_block_inventory_sha256
            != semantic_sha256(self.completed_block_receipts)
            or not self.transition_receipts
            or self.transition_inventory_sha256
            != semantic_sha256(self.transition_receipts)
            or len(self.transition_source_data_qualified)
            != len(self.transition_receipts)
            or any(
                not isinstance(value, bool)
                for value in self.transition_source_data_qualified
            )
            or self.fit_environment_authority_inventory_sha256
            != semantic_sha256(self.fit_environment_authority_receipts)
            or self.fit_environment_authority_receipts
            != tuple(dict.fromkeys(self.fit_environment_authority_receipts))
            or self.continuity_authority_inventory_sha256
            != semantic_sha256(self.continuity_authority_receipts)
            or self.update_count <= 0
            or self.development_rl_training_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "adaptive PPO training run differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


class MassiveAdaptivePrequentialPPORunnerV1:
    """Train one policy over every block in one causal forecast authority."""

    def __init__(
        self,
        *,
        training_authority: MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
        chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
        environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
        fit_environment_authorities: (
            Mapping[str, MassiveAdaptiveRLFitEnvironmentAuthorityV1] | None
        ) = None,
        model: MassiveAdaptivePPOActorCriticV1,
        config: MassiveAdaptivePPOConfigV1 | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        training_authority.validate()
        chronology_authority.validate()
        if not training_authority.reinforcement_learning_authorized:
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "causal training authority does not authorize PPO fitting"
            )
        if (
            not chronology_authority.development_rl_training_authorized
            or chronology_authority.training_forecast_authority_receipt_sha256
            != training_authority.semantic_receipt_sha256
            or chronology_authority.rl_fit_origin_dates
            != training_authority.origin_session_dates
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "PPO fitting chronology differs from its causal forecast authority"
            )
        self.training_authority = training_authority
        self.chronology_authority = chronology_authority
        self.environments = dict(environments)
        self.fit_environment_authorities = (
            {} if fit_environment_authorities is None else dict(fit_environment_authorities)
        )
        if self.fit_environment_authorities:
            if set(self.fit_environment_authorities) != set(self.environments):
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "PPO fit environment authority registry coverage differs"
                )
            for receipt, authority in self.fit_environment_authorities.items():
                if type(authority) is not MassiveAdaptiveRLFitEnvironmentAuthorityV1:
                    raise MassiveAdaptivePrequentialPPORunnerV1Error(
                        "PPO fit environment authority type differs"
                    )
                authority.validate()
                environment = self.environments[receipt]
                authority.validate_environment(environment)
                if (
                    not authority.runtime_environment_replayed
                    or not authority.source_data_qualified
                    or authority.forecast_archive_receipt_sha256 != receipt
                    or authority.environment_source_inventory_sha256
                    != environment.source_inventory_sha256
                ):
                    raise MassiveAdaptivePrequentialPPORunnerV1Error(
                        "PPO fit environment authority registry differs"
                    )
        self.model = model
        self.config = config or MassiveAdaptivePPOConfigV1()
        self.config.validate()
        self.device = torch.device(device)
        self.block_runtimes = self._bind_block_runtimes()
        self.block_runtime_inventory_sha256 = semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in self.block_runtimes)
        )
        self.continuity_authorities = self._bind_continuity_authorities()
        self.continuity_authority_inventory_sha256 = semantic_sha256(
            tuple(
                row.semantic_receipt_sha256
                for row in self.continuity_authorities.values()
            )
        )
        self.current_block_index = 0
        self.completed_block_receipts: list[str] = []
        self.transition_receipts: list[str] = []
        self.transition_source_data_qualified: list[bool] = []
        self._trainer = self._new_trainer(self._environment_for_block(0))
        self._trainer._ensure_observation()

    def _environment_for_block(self, index: int) -> MassiveAdaptiveProfitabilityEnvV1:
        receipt = self.training_authority.blocks[
            index
        ].source_forecast_archive_receipt_sha256
        try:
            return self.environments[receipt]
        except KeyError as error:
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "authorized PPO block has no matching environment"
            ) from error

    def _bind_block_runtimes(self) -> tuple[MassiveAdaptivePPOBlockRuntimeV1, ...]:
        grouped_dates: dict[str, list[str]] = {}
        for block in self.training_authority.blocks:
            grouped_dates.setdefault(
                block.source_forecast_archive_receipt_sha256, []
            ).extend(block.forecast_session_dates)
        runtimes: list[MassiveAdaptivePPOBlockRuntimeV1] = []
        cursors: dict[str, int] = {}
        for block in self.training_authority.blocks:
            environment = self._environment_for_block(block.block_index)
            environment.forecast_archive.validate()
            environment.calibration.validate()
            environment.inference_plan.validate()
            receipt = block.source_forecast_archive_receipt_sha256
            plan_dates = tuple(
                row.decision_session_date for row in environment.inference_plan.rows
            )
            if (
                environment.forecast_archive.semantic_receipt_sha256 != receipt
                or environment.calibration.semantic_receipt_sha256
                != block.calibration_receipt_sha256
                or tuple(grouped_dates[receipt]) != plan_dates
                or tuple(environment.forecasts) != plan_dates
            ):
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "authorized PPO block environment provenance differs"
                )
            start = cursors.get(receipt, 0)
            stop = start + len(block.forecast_session_dates)
            if plan_dates[start:stop] != block.forecast_session_dates:
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "authorized PPO block is not the next environment chronology"
                )
            body = {
                "schema": MASSIVE_ADAPTIVE_PPO_BLOCK_RUNTIME_V1_SCHEMA,
                "block_index": block.block_index,
                "block_receipt_sha256": block.semantic_receipt_sha256,
                "forecast_archive_receipt_sha256": receipt,
                "inference_plan_receipt_sha256": (
                    environment.inference_plan.semantic_receipt_sha256
                ),
                "calibration_receipt_sha256": block.calibration_receipt_sha256,
                "environment_source_inventory_sha256": (
                    environment.source_inventory_sha256
                ),
                "fit_environment_authority_receipt_sha256": (
                    None
                    if receipt not in self.fit_environment_authorities
                    else self.fit_environment_authorities[
                        receipt
                    ].semantic_receipt_sha256
                ),
                "forecast_session_dates": block.forecast_session_dates,
                "environment_start_cursor": start,
                "environment_stop_cursor": stop,
            }
            runtime = MassiveAdaptivePPOBlockRuntimeV1(
                **body,  # type: ignore[arg-type]
                semantic_receipt_sha256=semantic_sha256(body),
            )
            runtime.validate()
            runtimes.append(runtime)
            cursors[receipt] = stop
        if set(self.environments) != set(grouped_dates):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "PPO environment registry contains missing or extra forecasts"
            )
        return tuple(runtimes)

    def _bind_continuity_authorities(
        self,
    ) -> dict[int, MassiveAdaptiveEconomicContinuityAuthorityV1]:
        """Bind every boundary whose next block uses a different archive."""

        authorities: dict[int, MassiveAdaptiveEconomicContinuityAuthorityV1] = {}
        for next_index in range(1, len(self.block_runtimes)):
            previous = self.block_runtimes[next_index - 1]
            current = self.block_runtimes[next_index]
            if (
                previous.forecast_archive_receipt_sha256
                == current.forecast_archive_receipt_sha256
            ):
                continue
            authority = build_massive_adaptive_economic_continuity_authority_v1(
                previous_block_receipt_sha256=previous.block_receipt_sha256,
                next_block_receipt_sha256=current.block_receipt_sha256,
                previous_environment=self._environment_for_block(next_index - 1),
                next_environment=self._environment_for_block(next_index),
                source_data_qualified=bool(
                    self.training_authority.source_data_qualified
                    and previous.fit_environment_authority_receipt_sha256 is not None
                    and current.fit_environment_authority_receipt_sha256 is not None
                ),
            )
            if (
                authority.exchange_sessions_consecutive
                and not authority.carry_books_authorized
            ):
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "consecutive forecast refits changed economic accounting roots"
                )
            authorities[next_index] = authority
        return authorities

    def _new_trainer(
        self, environment: MassiveAdaptiveProfitabilityEnvV1
    ) -> MassiveAdaptivePPOTrainerV1:
        forecast_receipt = environment.forecast_archive.semantic_receipt_sha256
        return MassiveAdaptivePPOTrainerV1(
            environment=environment,
            model=self.model,
            config=self.config,
            device=self.device,
            training_forecast_authority=self.training_authority,
            fit_environment_authority=self.fit_environment_authorities.get(
                forecast_receipt
            ),
        )

    def _bootstrap_continuation_rollout(
        self,
        rollout: MassiveAdaptivePPORolloutV1,
        *,
        next_block_index: int,
    ) -> MassiveAdaptivePPORolloutV1:
        """Bootstrap GAE from the first observation under the next refit."""

        preview = copy.copy(self._environment_for_block(next_block_index))
        preview.restore_continuation(self._trainer.environment.state)
        observation = torch.tensor(
            preview.current_observation.values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            bootstrap = self.model({"adaptive_state": observation}).value[0].detach()
        next_values = torch.cat((rollout.old_values[1:], bootstrap.unsqueeze(0)))
        advantages = torch.zeros_like(rollout.rewards)
        running = torch.zeros((), dtype=torch.float32, device=self.device)
        for index in range(rollout.rewards.shape[0] - 1, -1, -1):
            continuation = 0.0 if bool(rollout.terminated[index].item()) else 1.0
            delta = (
                rollout.rewards[index]
                + self.config.gamma * next_values[index] * continuation
                - rollout.old_values[index]
            )
            running = (
                delta
                + self.config.gamma * self.config.gae_lambda * continuation * running
            )
            advantages[index] = running
        result = replace(
            rollout,
            advantages=advantages,
            returns=advantages + rollout.old_values,
        )
        result.validate()
        return result

    @staticmethod
    def _restore_learning_state(
        trainer: MassiveAdaptivePPOTrainerV1,
        checkpoint: MassiveAdaptiveRLCheckpointV1,
    ) -> None:
        checkpoint.validate()
        trainer.model.load_state_dict(checkpoint.model_state)
        trainer.actor_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.actor_optimizer_state)
        )
        trainer.critic_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.critic_optimizer_state)
        )
        torch.set_rng_state(checkpoint.torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(checkpoint.cuda_rng_states))
        random.setstate(checkpoint.python_rng_state)
        np.random.set_state(checkpoint.numpy_rng_state)
        trainer.minibatch_rng.set_state(checkpoint.minibatch_rng_state)
        trainer.update_index = checkpoint.update_index
        trainer.loss_trace = list(checkpoint.loss_trace)
        trainer.transition_receipts = list(checkpoint.transition_receipts)
        trainer.transition_source_data_qualified = list(
            checkpoint.transition_source_data_qualified
        )
        trainer.fit_environment_authority_receipts = list(
            checkpoint.fit_environment_authority_receipts
        )
        if trainer.fit_environment_authority is not None:
            receipt = trainer.fit_environment_authority.semantic_receipt_sha256
            if receipt not in trainer.fit_environment_authority_receipts:
                trainer.fit_environment_authority_receipts.append(receipt)
        trainer._observation = None

    @property
    def trainer(self) -> MassiveAdaptivePPOTrainerV1:
        return self._trainer

    @property
    def training_complete(self) -> bool:
        return len(self.completed_block_receipts) == len(self.block_runtimes)

    def _advance_block(
        self, learning_checkpoint: MassiveAdaptiveRLCheckpointV1
    ) -> None:
        self.current_block_index += 1
        if self.current_block_index >= len(self.block_runtimes):
            return
        previous = self.block_runtimes[self.current_block_index - 1]
        current = self.block_runtimes[self.current_block_index]
        if (
            current.forecast_archive_receipt_sha256
            == previous.forecast_archive_receipt_sha256
        ):
            if (
                self._trainer.environment.state.chronology_cursor
                != current.environment_start_cursor
            ):
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "consecutive PPO block did not preserve its economic cursor"
                )
            return
        environment = self._environment_for_block(self.current_block_index)
        continuity = self.continuity_authorities[self.current_block_index]
        if continuity.carry_books_authorized:
            previous_state = self._trainer.environment.state
            if (
                not previous_state.done
                or previous_state.strategy_book.decision_session_date
                != continuity.next_initial_decision_session_date
            ):
                raise MassiveAdaptivePrequentialPPORunnerV1Error(
                    "consecutive forecast refit did not preserve its terminal books"
                )
            environment.restore_continuation(previous_state)
        else:
            environment.reset()
        self._trainer = self._new_trainer(environment)
        self._restore_learning_state(self._trainer, learning_checkpoint)
        self._trainer._observation = environment.current_observation

    def run_next_update(self) -> dict[str, float]:
        if self.training_complete:
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "prequential PPO training is complete"
            )
        runtime = self.block_runtimes[self.current_block_index]
        cursor = self._trainer.environment.state.chronology_cursor
        if (
            not runtime.environment_start_cursor
            <= cursor
            < runtime.environment_stop_cursor
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "PPO trainer cursor lies outside its authorized block"
            )
        steps = min(
            self.config.rollout_length,
            runtime.environment_stop_cursor - cursor,
        )
        next_index = self.current_block_index + 1
        continuity = self.continuity_authorities.get(next_index)
        continue_economic_episode = bool(
            next_index < len(self.block_runtimes)
            and continuity is not None
            and continuity.carry_books_authorized
            and steps == runtime.environment_stop_cursor - cursor
        )
        rollout = self._trainer.collect_rollout(
            steps=steps,
            continue_economic_episode_at_end=continue_economic_episode,
        )
        if continue_economic_episode:
            rollout = self._bootstrap_continuation_rollout(
                rollout,
                next_block_index=next_index,
            )
        metrics = self._trainer.update(rollout)
        self.transition_receipts.extend(rollout.transition_receipts)
        self.transition_source_data_qualified.extend(
            rollout.transition_source_data_qualified
        )
        if (
            self._trainer.environment.state.chronology_cursor
            == runtime.environment_stop_cursor
        ):
            self.completed_block_receipts.append(runtime.block_receipt_sha256)
            learning_checkpoint = self._trainer.checkpoint()
            self._advance_block(learning_checkpoint)
        return metrics

    def checkpoint(self) -> MassiveAdaptivePrequentialPPOCheckpointV1:
        if self.training_complete:
            index = len(self.block_runtimes) - 1
        else:
            index = self.current_block_index
        runtime = self.block_runtimes[index]
        ppo_checkpoint = self._trainer.checkpoint()
        environment_authority_receipts = tuple(
            dict.fromkeys(
                row.fit_environment_authority_receipt_sha256
                for row in self.block_runtimes
                if row.fit_environment_authority_receipt_sha256 is not None
            )
        )
        source_qualified = bool(
            self.training_authority.source_data_qualified
            and getattr(self.chronology_authority, "source_data_qualified", False)
            and len(environment_authority_receipts) == len(self.block_runtimes)
            and self.transition_receipts
            and len(self.transition_receipts)
            == len(self.transition_source_data_qualified)
            and all(self.transition_source_data_qualified)
            and ppo_checkpoint.source_data_qualified
        )
        body = {
            "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_CHECKPOINT_V1_SCHEMA,
            "training_forecast_authority_receipt_sha256": (
                self.training_authority.semantic_receipt_sha256
            ),
            "rl_chronology_authority_receipt_sha256": (
                self.chronology_authority.semantic_receipt_sha256
            ),
            "block_inventory_sha256": self.training_authority.block_inventory_sha256,
            "block_runtime_inventory_sha256": self.block_runtime_inventory_sha256,
            "continuity_authority_receipts": tuple(
                row.semantic_receipt_sha256
                for row in self.continuity_authorities.values()
            ),
            "continuity_authority_inventory_sha256": (
                self.continuity_authority_inventory_sha256
            ),
            "current_block_index": index,
            "current_block_receipt_sha256": runtime.block_receipt_sha256,
            "current_calibration_receipt_sha256": runtime.calibration_receipt_sha256,
            "current_environment_source_inventory_sha256": (
                runtime.environment_source_inventory_sha256
            ),
            "within_block_chronology_cursor": (
                ppo_checkpoint.environment_state.chronology_cursor
            ),
            "completed_block_receipts": tuple(self.completed_block_receipts),
            "completed_block_inventory_sha256": semantic_sha256(
                tuple(self.completed_block_receipts)
            ),
            "transition_receipts": tuple(self.transition_receipts),
            "transition_source_data_qualified": tuple(
                self.transition_source_data_qualified
            ),
            "transition_inventory_sha256": semantic_sha256(
                tuple(self.transition_receipts)
            ),
            "fit_environment_authority_receipts": (
                environment_authority_receipts
            ),
            "fit_environment_authority_inventory_sha256": semantic_sha256(
                environment_authority_receipts
            ),
            "ppo_checkpoint": ppo_checkpoint,
            "training_complete": self.training_complete,
            "source_data_qualified": source_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
            "specification_sha256": (
                MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SPEC_SHA256
            ),
            "implementation_source_sha256": (
                MASSIVE_ADAPTIVE_PREQUENTIAL_PPO_RUNNER_V1_SOURCE_SHA256
            ),
        }
        provisional = MassiveAdaptivePrequentialPPOCheckpointV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256="0" * 64,
            exact_resume_authorized=source_qualified,
            development_rl_training_authorized=(
                source_qualified
            ),
        )
        result = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        )
        result.validate()
        return result

    def restore(self, checkpoint: MassiveAdaptivePrequentialPPOCheckpointV1) -> None:
        checkpoint.validate()
        if (
            checkpoint.training_forecast_authority_receipt_sha256
            != self.training_authority.semantic_receipt_sha256
            or checkpoint.rl_chronology_authority_receipt_sha256
            != self.chronology_authority.semantic_receipt_sha256
            or checkpoint.block_inventory_sha256
            != self.training_authority.block_inventory_sha256
            or checkpoint.block_runtime_inventory_sha256
            != self.block_runtime_inventory_sha256
            or checkpoint.continuity_authority_inventory_sha256
            != self.continuity_authority_inventory_sha256
            or checkpoint.continuity_authority_receipts
            != tuple(
                row.semantic_receipt_sha256
                for row in self.continuity_authorities.values()
            )
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "prequential PPO checkpoint and runner roots differ"
            )
        index = checkpoint.current_block_index
        runtime = self.block_runtimes[index]
        if (
            checkpoint.current_block_receipt_sha256 != runtime.block_receipt_sha256
            or checkpoint.current_calibration_receipt_sha256
            != runtime.calibration_receipt_sha256
            or checkpoint.current_environment_source_inventory_sha256
            != runtime.environment_source_inventory_sha256
            or checkpoint.completed_block_receipts
            != tuple(
                row.block_receipt_sha256
                for row in self.block_runtimes[
                    : len(checkpoint.completed_block_receipts)
                ]
            )
        ):
            raise MassiveAdaptivePrequentialPPORunnerV1Error(
                "prequential PPO checkpoint block lineage differs"
            )
        environment = self._environment_for_block(index)
        self.current_block_index = index
        self.completed_block_receipts = list(checkpoint.completed_block_receipts)
        self.transition_receipts = list(checkpoint.transition_receipts)
        self.transition_source_data_qualified = list(
            checkpoint.transition_source_data_qualified
        )
        self._trainer = self._new_trainer(environment)
        self._trainer.restore(checkpoint.ppo_checkpoint)

    def run_to_completion(self) -> MassiveAdaptivePPOTrainingRunV1:
        while not self.training_complete:
            self.run_next_update()
        checkpoint = self.checkpoint()
        source_qualified = checkpoint.source_data_qualified
        body = {
            "schema": MASSIVE_ADAPTIVE_PPO_TRAINING_RUN_V1_SCHEMA,
            "training_forecast_authority_receipt_sha256": (
                self.training_authority.semantic_receipt_sha256
            ),
            "rl_chronology_authority_receipt_sha256": (
                self.chronology_authority.semantic_receipt_sha256
            ),
            "block_runtime_inventory_sha256": self.block_runtime_inventory_sha256,
            "continuity_authority_receipts": tuple(
                row.semantic_receipt_sha256
                for row in self.continuity_authorities.values()
            ),
            "continuity_authority_inventory_sha256": (
                self.continuity_authority_inventory_sha256
            ),
            "completed_block_receipts": tuple(self.completed_block_receipts),
            "completed_block_inventory_sha256": semantic_sha256(
                tuple(self.completed_block_receipts)
            ),
            "transition_receipts": tuple(self.transition_receipts),
            "transition_source_data_qualified": tuple(
                self.transition_source_data_qualified
            ),
            "transition_inventory_sha256": semantic_sha256(
                tuple(self.transition_receipts)
            ),
            "fit_environment_authority_receipts": (
                checkpoint.fit_environment_authority_receipts
            ),
            "fit_environment_authority_inventory_sha256": (
                checkpoint.fit_environment_authority_inventory_sha256
            ),
            "final_checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
            "update_count": checkpoint.ppo_checkpoint.update_index,
            "source_data_qualified": source_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        }
        provisional = MassiveAdaptivePPOTrainingRunV1(
            **body,  # type: ignore[arg-type]
            semantic_receipt_sha256="0" * 64,
            development_rl_training_authorized=(
                source_qualified
            ),
        )
        result = replace(
            provisional,
            semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        )
        result.validate()
        return result


def train_massive_adaptive_prequential_ppo_v1(
    *,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityProtocol,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    fit_environment_authorities: (
        Mapping[str, MassiveAdaptiveRLFitEnvironmentAuthorityV1] | None
    ) = None,
    policy: MassiveAdaptivePPOActorCriticV1,
    config: MassiveAdaptivePPOConfigV1 | None = None,
    device: torch.device | str = "cpu",
) -> MassiveAdaptivePPOTrainingRunV1:
    """Run the complete causal block inventory without caller-selected omissions."""

    runner = MassiveAdaptivePrequentialPPORunnerV1(
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        fit_environment_authorities=fit_environment_authorities,
        model=policy,
        config=config,
        device=device,
    )
    return runner.run_to_completion()


__all__ = [
    "MassiveAdaptivePPOBlockRuntimeV1",
    "MassiveAdaptivePPOTrainingRunV1",
    "MassiveAdaptivePrequentialPPOCheckpointV1",
    "MassiveAdaptivePrequentialPPORunnerV1",
    "MassiveAdaptivePrequentialPPORunnerV1Error",
    "train_massive_adaptive_prequential_ppo_v1",
]
