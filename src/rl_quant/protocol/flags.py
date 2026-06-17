"""Flag registry (governance) -- the opt-in flags that move results, with the metadata the promotion gate
requires, encoded as data so flags cannot accumulate ambiguously.

Every result-moving capability in QuantTrade ships behind a default-OFF flag whose default reproduces the old
behaviour byte-for-byte; only flipping the default moves a reported number, and that is a separate, A/B-gated,
one-flag-rollback act (see architecture_migration_plan.md and engineering_thinking.md). This module records,
for each such flag: its default, WHAT flipping it moves, the A/B metrics a latest-period experiment must
report before a flip, the flip criterion, and the delete criterion (so the flag doesn't live forever). Pure
data + helpers; imports nothing else in rl_quant and changes no result. A gate test asserts the registry is
well-formed and that the recorded defaults match the actual config dataclasses (drift guard)."""

from __future__ import annotations

from dataclasses import dataclass

# Categories of impact a flag flip can have. A flag whose `moves` intersects this set is "result-moving" and
# must carry A/B metadata before any default flip.
RESULT_MOVING_IMPACTS = ("pnl", "reward", "model_inputs", "replay_schema", "td_targets")

# The latest-period A/B metrics a result-moving flip must report (the promotion gate's reporting contract).
_STANDARD_AB_METRICS = (
    "total_return",
    "turnover",
    "exposure",
    "cash_share",
    "cost_drag",
    "max_drawdown",
    "sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
)


@dataclass(frozen=True)
class FlagSpec:
    """Promotion metadata for one opt-in flag."""

    name: str
    default: bool
    moves: tuple[str, ...]
    required_ab: tuple[str, ...]
    flip_criterion: str
    delete_criterion: str

    @property
    def is_result_moving(self) -> bool:
        return any(impact in RESULT_MOVING_IMPACTS for impact in self.moves)


FLAG_REGISTRY: dict[str, FlagSpec] = {
    "use_dynamic_transition_features": FlagSpec(
        name="use_dynamic_transition_features",
        default=False,
        moves=("model_inputs", "replay_schema", "td_targets"),
        required_ab=_STANDARD_AB_METRICS,
        flip_criterion=(
            "latest-period A/B improves the promotion gate (return/cost/drawdown) without materially degrading "
            "the cost-stress grid; test block untouched; clean-perturbation verified at init"
        ),
        delete_criterion="remove the flag (make it the only path) after two stable cycles at the new default",
    ),
    "use_transition_features": FlagSpec(
        name="use_transition_features",
        default=False,
        moves=("model_inputs",),
        required_ab=_STANDARD_AB_METRICS,
        flip_criterion=(
            "latest-period A/B improves the promotion gate; transition encoder is a clean perturbation "
            "(bit-identical backbone at init); test block untouched"
        ),
        delete_criterion="remove the flag after two stable cycles at the new default",
    ),
    # Declared AHEAD of the PR-3/PR-4 execution-engine wiring (see docs/execution_wiring_design.md) so the
    # governance contract exists before the code does. These are not yet config fields; the drift guard in the
    # gate test only cross-checks the two flags above against MinuteToHourTrainingConfig.
    "execution_env_reward_shadow": FlagSpec(
        name="execution_env_reward_shadow",
        default=False,
        # Label-changing only: computes + logs the execution-engine reward/cost ALONGSIDE the legacy reward;
        # training is unchanged, so this moves artifacts/metrics, NOT a reported P&L number.
        moves=("metrics", "manifest", "artifact_schema"),
        required_ab=(),
        flip_criterion="no default flip -- diagnostic shadow only, unless promoted into PR-4",
        delete_criterion="remove once use_execution_env_reward replaces it, or if abandoned",
    ),
    "use_execution_env_reward": FlagSpec(
        name="use_execution_env_reward",
        default=False,
        # Result-moving: makes the env TRAIN on the execution-engine reward instead of the legacy reward.
        moves=("reward", "pnl", "td_targets"),
        required_ab=_STANDARD_AB_METRICS,
        flip_criterion="latest-period A/B improves the promotion gate without degrading reportability; test block untouched",
        delete_criterion="remove after two stable cycles at the new default",
    ),
}


def result_moving_flags() -> tuple[str, ...]:
    """Names of registered flags whose flip moves a reported result (need A/B before a default flip)."""
    return tuple(name for name, spec in FLAG_REGISTRY.items() if spec.is_result_moving)
