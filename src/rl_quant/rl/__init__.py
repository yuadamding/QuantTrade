"""Algorithm-neutral deep reinforcement-learning substrate."""

from rl_quant.rl.algorithm import ActionValueCritic, Actor, Algorithm, Critic, RecurrentState
from rl_quant.rl.environment import VectorEnvironment
from rl_quant.rl.iql import (
    IQLConfig,
    ImplicitQLearning,
    OfflineActorCritic,
    RegimeMixtureIQLActorCritic,
    VectorIQLActorCritic,
)
from rl_quant.rl.mixture import (
    MixtureActionDistribution,
    RegimeRouter,
    RoutedAction,
    RouterOutput,
)
from rl_quant.rl.offline import (
    OfflineTrainer,
    OfflineTrainingConfig,
    OfflineTrainingSummary,
    ReplayCollectionContinuation,
    ReplayCollectionMetrics,
    ReplayCollectionResult,
    ReplayRolloutCollector,
)
from rl_quant.rl.ppo import (
    ActionDistribution,
    DiagonalNormal,
    MaskedCategorical,
    MaskedDirichlet,
    PPOActorCritic,
    PPOConfig,
    PPOModelOutput,
    RecurrentActorCritic,
    RecurrentPPO,
)
from rl_quant.rl.replay import ReplayBatch, TransitionReplayBuffer, align_replay_batches
from rl_quant.rl.robust import (
    AbstentionResult,
    AdverseRewardTransform,
    ObservationAffineTransform,
    TransitionTransform,
    UncertaintyAbstention,
)
from rl_quant.rl.rollout import (
    OnPolicyAgent,
    OnPolicyRolloutCoordinator,
    RolloutContinuation,
    RolloutMetrics,
    RolloutResult,
)
from rl_quant.rl.specs import ActionKind, ActionSpec, TensorSpec
from rl_quant.rl.trajectory import (
    OnPolicyTrajectoryBuffer,
    RecurrentSequenceBatch,
    TrajectoryBatch,
)
from rl_quant.rl.types import ActionBatch, ObservationBatch, RewardComponents, TransitionBatch

__all__ = [
    "ActionBatch",
    "ActionKind",
    "ActionSpec",
    "ActionValueCritic",
    "ActionDistribution",
    "AbstentionResult",
    "AdverseRewardTransform",
    "Actor",
    "Algorithm",
    "Critic",
    "DiagonalNormal",
    "IQLConfig",
    "ImplicitQLearning",
    "MaskedCategorical",
    "MaskedDirichlet",
    "MixtureActionDistribution",
    "ObservationBatch",
    "ObservationAffineTransform",
    "OfflineActorCritic",
    "OfflineTrainer",
    "OfflineTrainingConfig",
    "OfflineTrainingSummary",
    "OnPolicyTrajectoryBuffer",
    "OnPolicyAgent",
    "OnPolicyRolloutCoordinator",
    "PPOActorCritic",
    "PPOConfig",
    "PPOModelOutput",
    "RecurrentSequenceBatch",
    "RecurrentActorCritic",
    "RecurrentPPO",
    "RecurrentState",
    "RegimeRouter",
    "RegimeMixtureIQLActorCritic",
    "ReplayBatch",
    "ReplayCollectionContinuation",
    "ReplayCollectionMetrics",
    "ReplayCollectionResult",
    "ReplayRolloutCollector",
    "RewardComponents",
    "RoutedAction",
    "RouterOutput",
    "RolloutContinuation",
    "RolloutMetrics",
    "RolloutResult",
    "TensorSpec",
    "TrajectoryBatch",
    "TransitionBatch",
    "TransitionReplayBuffer",
    "TransitionTransform",
    "UncertaintyAbstention",
    "VectorEnvironment",
    "VectorIQLActorCritic",
    "align_replay_batches",
]
