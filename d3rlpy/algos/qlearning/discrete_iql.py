import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_discrete_q_function,
    create_value_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import MeanQFunctionFactory
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.discrete_iql_impl import DiscreteIQLImpl, DiscreteIQLModules

__all__ = ["DiscreteIQLConfig", "DiscreteIQL"]


@dataclasses.dataclass()
class DiscreteIQLConfig(LearnableConfig):
    r"""Discrete Implicit Q-Learning algorithm.

    Discrete IQL is an adaptation of the IQL algorithm for discrete action spaces.
    Similar to the continuous version, it avoids querying values of unseen actions
    while still performing multi-step dynamic programming updates.

    The state-value function is trained via expectile regression:

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [L_2^\tau (Q_\theta (s, a) - V_\psi (s))]

    where :math:`L_2^\tau (u) = |\tau - \mathbb{1}(u < 0)|u^2`.

    The Q-function is trained with the state-value function:

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    The policy is trained using advantage weighted behavioral cloning with
    a categorical distribution:

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [-\exp(\beta (Q_\theta(s, a) - V_\psi(s))) \log \pi_\phi(a|s)]

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the value function.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        expectile (float): Expectile value for value function training.
        weight_temp (float): Inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): Maximum advantage weight value to clip.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteIQL":
        return DiscreteIQL(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_iql"


class DiscreteIQL(QLearningAlgoBase[DiscreteIQLImpl, DiscreteIQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        # Import torch here to create the policy network
        import torch.nn as nn
        
        # Create encoder for policy
        policy_encoder = self._config.actor_encoder_factory.create(
            observation_shape, self._device
        )
        
        # Create simple categorical policy network
        class CategoricalPolicy(nn.Module):
            def __init__(self, encoder, action_size):
                super().__init__()
                self.encoder = encoder
                self.fc = nn.Linear(encoder.get_feature_size(), action_size)
                
            def forward(self, x):
                h = self.encoder(x)
                return self.fc(h)  # Return logits
        
        policy = CategoricalPolicy(policy_encoder, action_size).to(self._device)
        
        # Create Q-functions
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # Create target Q-functions
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # Create value function
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        # Create optimizers
        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        
        # Combine Q-function and value function parameters for critic optimizer
        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params,
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        # Create modules
        modules = DiscreteIQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        # Create implementation
        self._impl = DiscreteIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(DiscreteIQLConfig)