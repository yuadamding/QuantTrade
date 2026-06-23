"""Stage 2 — DECISION-POLICY learning over the frozen context embeddings.

The context encoder (Stage 1, ``training.context_pretrain``) is trained self-supervised and FROZEN; its hourly
embeddings are precomputed per decision (``precompute_context_embeddings``). This stage learns the trading policy
by double-DQN on those compact embeddings + the policy state (held action, constraint state, optional per-action
features), using ``rl_quant.models.decision_policy.DecisionPolicyQNetwork``.

Why this is cheap and clean: ``VectorizedSecondToHourEnv`` already serves data BY DECISION INDEX (its ``step`` /
replay carry ``indices`` / ``next_indices``, not raw tensors), so Stage-2 reuses the env, reward ledger, constraint
masks, replay buffer, leg-aware hysteresis and ``dqn_td_target`` VERBATIM -- the only change is that the model's
observation is ``embeddings[indices]`` (a ``[B, d_model]`` gather) instead of the per-second ``(second, mask,
hour)`` window, and the model is the small policy head (no transformer in the hot loop). The heavy per-second
encoding is paid once in Stage 1, so reward/constraint/policy iteration here never re-touches the seconds.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl_quant.core import (
    TensorDictReplayBuffer,
    dqn_td_target,
    epsilon_by_step,
    safe_next_row_indices,
)
from rl_quant.datasets.hour_from_second import HourFromMinuteDataSplit
from rl_quant.envs.second_to_hour import SecondToHourEnvConfig, VectorizedSecondToHourEnv
from rl_quant.models.decision_policy import DecisionPolicyQNetwork
from rl_quant.models.second_to_hour import SecondToHourContextEncoder
from rl_quant.protocol.constraints import CONSTRAINT_FEATURE_DIM
from rl_quant.trading_constraints import apply_leg_aware_hysteresis, sample_valid_actions
from rl_quant.training.context_pretrain import encode_split


@dataclass(frozen=True)
class DecisionPolicyConfig:
    # model: a SMALL head over the frozen context embedding (the heavy per-second encoder is Stage-1 + frozen,
    # so the policy has few params -> far less reward-overfit than the joint trainer). d_model/feedforward/dropout
    # must match the Stage-1 encoder to load into the SecondToHourPolicyQNetwork bridge for OOS eval.
    d_model: int = 128
    action_embedding_dim: int = 32
    feedforward_dim: int = 256
    dropout: float = 0.15
    action_feature_dim: int = 0  # set to the news/covariate width (e.g. 56) to feed those features at the policy
    transition_feature_dim: int = 0
    dynamic_feature_dim: int = 0
    # env / DQN -- scaled to the small per-window data + stronger weight decay to regularize the head
    num_envs: int = 24
    episode_length: int = 8
    reward_scale: float = 10_000.0
    train_steps: int = 2_000
    batch_size: int = 64
    warmup_steps: int = 200
    replay_capacity: int = 20_000
    gamma: float = 0.99
    lr: float = 3e-4
    weight_decay: float = 3e-2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    target_update_interval: int = 150


@contextmanager
def _eval_for_action_selection(module):
    """Temporarily eval() a module for a no-grad action-selection / bootstrap-target forward so dropout cannot
    perturb the greedy argmax used to COLLECT transitions or the double-DQN next-action argmax, then restore
    its prior mode. The gradient-bearing forward stays in train()."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


def freeze_encoder(encoder: SecondToHourContextEncoder) -> SecondToHourContextEncoder:
    """Freeze the Stage-1 context encoder so Stage-2 only trains the policy on its embeddings."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def precompute_context_embeddings(
    encoder: SecondToHourContextEncoder, data: HourFromMinuteDataSplit, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Frozen hourly embedding ``[D, d_model]`` per decision row -- the Stage-2 observation source."""
    embeddings = encode_split(freeze_encoder(encoder), data, device=device)
    # Stamp the split identity so train_decision_policy_dqn can reject embeddings precomputed for a
    # different (same-row-count) split. Opt-in attribute -> non-breaking for plain encode_split callers.
    try:
        embeddings._qt_split_key = tuple(data.decision_timestamps)
    except (AttributeError, TypeError):
        pass
    return embeddings


def build_decision_policy(
    *, action_count: int, config: DecisionPolicyConfig = DecisionPolicyConfig(),
    transition_table: torch.Tensor | None = None,
) -> DecisionPolicyQNetwork:
    """Construct the Stage-2 policy Q-network that scores actions from a context embedding + policy state."""
    return DecisionPolicyQNetwork(
        d_model=config.d_model, action_count=action_count, action_embedding_dim=config.action_embedding_dim,
        constraint_feature_dim=CONSTRAINT_FEATURE_DIM, feedforward_dim=config.feedforward_dim,
        dropout=config.dropout, action_feature_dim=config.action_feature_dim,
        transition_feature_dim=config.transition_feature_dim, transition_table=transition_table,
        dynamic_feature_dim=config.dynamic_feature_dim,
    )


def train_decision_policy_dqn(
    embeddings: torch.Tensor,
    data: HourFromMinuteDataSplit,
    config: DecisionPolicyConfig = DecisionPolicyConfig(),
    *,
    device: torch.device | str = "cpu",
    transition_table: torch.Tensor | None = None,
) -> tuple[DecisionPolicyQNetwork, dict]:
    """Double-DQN over the precomputed context ``embeddings [D, d_model]``. Reuses the second->hour env (state
    machine + reward + constraint masks) and the shared DQN helpers; the observation is ``embeddings[indices]``
    and the model is a ``DecisionPolicyQNetwork`` (no transformer in the loop). Returns ``(policy, metrics)``."""
    device = torch.device(device)
    data = data.to(device)
    split_key = getattr(embeddings, "_qt_split_key", None)  # captured before .to() drops custom attrs
    embeddings = embeddings.to(device)
    if embeddings.shape[0] != int(data.action_returns.shape[0]):
        raise ValueError("embeddings rows must equal the number of decision rows in the split.")
    # Identity (not just row count): embeddings precomputed for a different split with the same row count
    # would gather misaligned per-row context. precompute_context_embeddings stamps this key.
    if split_key is not None and split_key != tuple(data.decision_timestamps):
        raise ValueError(
            "embeddings were precomputed for a DIFFERENT split (decision_timestamps mismatch); row counts "
            "match but the per-row context would be misaligned. Re-run precompute_context_embeddings on this split."
        )
    # Stage-2 does not thread the joint trainer's dynamic-position state or transition table, and must not
    # silently drop per-action features the split actually carries. Fail fast on misconfiguration.
    if config.dynamic_feature_dim > 0:
        raise ValueError(
            "DecisionPolicyConfig.dynamic_feature_dim > 0 is not supported by train_decision_policy_dqn: the "
            "Stage-2 env/replay do not carry dynamic position state. Use the joint trainer if you need it."
        )
    if config.transition_feature_dim > 0 and transition_table is None:
        raise ValueError("DecisionPolicyConfig.transition_feature_dim > 0 requires a transition_table.")
    data_action_feature_dim = 0 if data.action_features is None else int(data.action_features.shape[-1])
    if config.action_feature_dim and config.action_feature_dim != data_action_feature_dim:
        raise ValueError(
            f"action_feature_dim={config.action_feature_dim} but the split carries "
            f"{data_action_feature_dim} action-feature columns."
        )
    if config.action_feature_dim == 0 and data_action_feature_dim > 0:
        raise ValueError(
            f"The split carries {data_action_feature_dim} action-feature columns but action_feature_dim=0 would "
            f"silently drop them; set DecisionPolicyConfig.action_feature_dim={data_action_feature_dim}."
        )
    action_count = len(data.action_names)
    use_action_features = config.action_feature_dim > 0

    policy = build_decision_policy(action_count=action_count, config=config, transition_table=transition_table).to(device)
    target = deepcopy(policy).to(device)
    target.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    env = VectorizedSecondToHourEnv(
        data,
        SecondToHourEnvConfig(num_envs=config.num_envs, episode_length=config.episode_length,
                              reward_scale=config.reward_scale),
        device,
    )
    c = env.constraints  # normalized constraints (canonical types)
    replay = TensorDictReplayBuffer(
        capacity=config.replay_capacity, device=device,
        fields={
            "indices": ((), torch.long), "previous_actions": ((), torch.long),
            "constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
            "action_mask": ((action_count,), torch.bool), "actions": ((), torch.long),
            "rewards": ((), torch.float32), "next_indices": ((), torch.long),
            "next_previous_actions": ((), torch.long),
            "next_constraint_features": ((CONSTRAINT_FEATURE_DIM,), torch.float32),
            "next_action_mask": ((action_count,), torch.bool), "terminated": ((), torch.float32),
        },
    )

    def hysteresis(q, prev, mask):
        return apply_leg_aware_hysteresis(
            q, prev, mask, one_way_cost_bps=c.one_way_cost_bps, extra_switch_penalty_bps=c.extra_switch_penalty_bps,
            q_switch_margin_bps=c.q_switch_margin_bps, cash_index=env.cash_index, reward_scale=config.reward_scale,
            count_etf_to_etf_as_two_legs=c.count_etf_to_etf_as_two_legs,
        )

    def action_feats(indices):
        return data.action_feature_state(indices) if use_action_features else None

    reward_trace: list[float] = []
    loss_trace: list[float] = []
    for step in range(1, config.train_steps + 1):
        _, _, _, _, previous_actions, constraint_features, action_mask = env.observe()
        context = embeddings[env.indices]
        epsilon = epsilon_by_step(step=step, train_steps=config.train_steps,
                                  start=config.epsilon_start, end=config.epsilon_end)
        with torch.no_grad(), _eval_for_action_selection(policy):
            q = policy(context, previous_actions, constraint_features, action_features=action_feats(env.indices))
            greedy = hysteresis(q, previous_actions, action_mask)
            explore = torch.rand(greedy.shape, device=device) < epsilon
            actions = torch.where(explore, sample_valid_actions(action_mask), greedy)
        transition = env.step(actions)
        replay.add(**{k: v for k, v in transition.items() if k in replay.storage})
        reward_trace.append(float(transition["rewards"].mean().item()))
        env.reset(transition["resets"].bool())

        if replay.size >= max(config.warmup_steps, config.batch_size):
            batch = replay.sample(config.batch_size)
            n_rows = int(data.action_returns.shape[0])
            safe_next = safe_next_row_indices(
                batch["next_indices"], batch["terminated"], min_index=0, max_index=n_rows - 1,
                valid_index_mask=data.valid_index_mask,
            )
            ctx = embeddings[batch["indices"]]
            next_ctx = embeddings[safe_next]
            q = policy(ctx, batch["previous_actions"], batch["constraint_features"],
                       action_features=action_feats(batch["indices"]))
            chosen_q = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
            with torch.no_grad(), _eval_for_action_selection(policy):
                next_online = policy(next_ctx, batch["next_previous_actions"], batch["next_constraint_features"],
                                     action_features=action_feats(safe_next))
                next_actions = hysteresis(next_online, batch["next_previous_actions"], batch["next_action_mask"])
                next_target = target(next_ctx, batch["next_previous_actions"], batch["next_constraint_features"],
                                     action_features=action_feats(safe_next))
                next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = dqn_td_target(batch["rewards"], config.gamma, batch["terminated"], next_q)
            loss = F.smooth_l1_loss(chosen_q.float(), target_q.float())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_trace.append(float(loss.detach()))
            if step % config.target_update_interval == 0:
                target.load_state_dict(policy.state_dict())

    return policy, {
        "train_steps": config.train_steps,
        "final_loss": loss_trace[-1] if loss_trace else None,
        "mean_reward_last_100": (sum(reward_trace[-100:]) / len(reward_trace[-100:])) if reward_trace else None,
    }
