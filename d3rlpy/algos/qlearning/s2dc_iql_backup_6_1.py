"""S2DC-IQL: IQL with Self-Supervised Distribution Correction."""

import dataclasses
from typing import Optional

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_value_function,
    create_parameter,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import MeanQFunctionFactory
from ...models.torch import Parameter, get_parameter
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...preprocessing import RewardScaler, make_reward_scaler_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.s2dc_iql_impl import S2DCIQLImpl, S2DCIQLModules

__all__ = ["S2DCIQLConfig", "S2DCIQL"]


@dataclasses.dataclass()
class S2DCIQLConfig(LearnableConfig):
    r"""Config for IQL with S2DC (Self-Supervised Distribution Correction).
    
    S2DC-IQL extends IQL by learning importance weights based on Q-function 
    improvement and Bellman consistency. It maintains two sets of Q-functions:
    - Q_RS: Trained with resampled data using S2DC weights
    - Q_D: Trained with uniform sampling as baseline
    
    Two difference networks (N1, N2) learn to predict:
    - N1: Q_RS - Bellman(Q_RS)
    - N2: Q_RS - Q_D
    
    These predictions are used to compute importance weights:
    w = exp(-n2/tau) * |n1|
    
    Args:
        actor_learning_rate: Learning rate for policy function.
        critic_learning_rate: Learning rate for Q functions and value function.
        actor_optim_factory: Optimizer factory for the actor.
        critic_optim_factory: Optimizer factory for the critic.
        actor_encoder_factory: Encoder factory for the actor.
        critic_encoder_factory: Encoder factory for the critic.
        value_encoder_factory: Encoder factory for the value function.
        delta_encoder_factory: Encoder factory for difference networks.
        batch_size: Mini-batch size.
        gamma: Discount factor.
        tau: Target network synchronization coefficient.
        n_critics: Number of Q functions for ensemble.
        expectile: Expectile value for value function training.
        weight_temp: Inverse temperature for advantage weighting.
        max_weight: Maximum advantage weight value to clip.
        initial_tau_s2dc: Initial temperature for S2DC weighting.
        min_tau_s2dc: Minimum tau value.
        max_tau_s2dc: Maximum tau value.
        tau_decay: Decay rate for tau.
        delta_learning_rate: Learning rate for difference networks.
        delta_optim_factory: Optimizer factory for difference networks.
        warm_up_steps: Steps before applying S2DC weights.
        blend_steps: Steps to blend from uniform to S2DC weights.
        kl_div_threshold: KL divergence threshold for tau adjustment.
        tau_increase_rate: Rate to increase tau when KL is high.
        weight_clip_percentile: Percentile for weight clipping.
        gradient_clip_norm: Gradient clipping norm.
        reward_scaler: Reward preprocessor.
        compile_graph: Flag to enable JIT compilation and CUDAGraph.
    """
    
    # IQL parameters
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
    
    # S2DC specific parameters
    initial_tau_s2dc: float = 1.0
    min_tau_s2dc: float = 0.02
    max_tau_s2dc: float = 10.0
    tau_decay: float = 0.995
    delta_learning_rate: float = 3e-4
    delta_optim_factory: OptimizerFactory = make_optimizer_field()
    delta_encoder_factory: EncoderFactory = make_encoder_field()
    warm_up_steps: int = 5000
    blend_steps: int = 5000
    kl_div_threshold: float = 0.5
    tau_increase_rate: float = 1.0001
    weight_clip_percentile: float = 99.0
    gradient_clip_norm: float = 10.0
    
    # Additional features
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()
    compile_graph: bool = False

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "S2DCIQL":
        return S2DCIQL(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "s2dc_iql"


class S2DCIQL(QLearningAlgoBase[S2DCIQLImpl, S2DCIQLConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        # Create policy (same as IQL)
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        actor_d = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_d_optim = self._config.actor_optim_factory.create(
            actor_d.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )        
        # Create Q-functions for QRS (resampled)
        q_funcs_rs, q_func_forwarder_rs = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs_rs, targ_q_func_forwarder_rs = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # Create Q-functions for QD (uniform sampling)
        q_funcs_d, q_func_forwarder_d = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs_d, targ_q_func_forwarder_d = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # Create value function (same as IQL)
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        value_func_d = create_value_function(
        observation_shape,
        self._config.value_encoder_factory,  # Can use same or different encoder
        device=self._device,
        enable_ddp=self._enable_ddp,
       )   
        # Create difference networks for S2DC
        # N1: predicts Q_RS - Bellman(Q_RS)
        n1 = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.delta_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=1,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )[0]  # Just get the ModuleList
        
        # N2: predicts Q_RS - Q_D
        n2 = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.delta_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=1,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )[0]  # Just get the ModuleList
        
        # Create old Q-functions for tracking
        q_funcs_rs_old, q_func_forwarder_rs_old = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs_rs_old, targ_q_func_forwarder_rs_old = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        
        # Create tau parameter for S2DC
        import math
        tau_s2dc = create_parameter(
            (1,),
            math.log(self._config.initial_tau_s2dc),
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        # Ensure tau_s2dc doesn't require gradients since it's adjusted programmatically
     #   get_parameter(tau_s2dc).requires_grad_(False)
        
        # Create optimizers
        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        
        # Optimizer for QRS and value function
        q_rs_params = list(q_funcs_rs.named_modules())
        v_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_rs_params + list(value_func.named_modules()),  # Q_RS + V_RS
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        # Optimizer for QD
        critic_d_optim = self._config.critic_optim_factory.create(
            list(q_funcs_d.named_modules()) + list(value_func_d.named_modules()),  # Q_D + V_D
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
            # Optimizer for difference networks
        n1_params = list(n1.named_modules())
        n2_params = list(n2.named_modules())
        delta_optim = self._config.delta_optim_factory.create(
            n1_params + n2_params,
            lr=self._config.delta_learning_rate,
            compiled=self.compiled,
        )
        
        modules = S2DCIQLModules(
            policy=policy,
            q_funcs=q_funcs_rs,  # Use standard name for base class
            targ_q_funcs=targ_q_funcs_rs,  # Use standard name for base class
            q_funcs_d=q_funcs_d,
            targ_q_funcs_d=targ_q_funcs_d,
            value_func=value_func,
            value_func_d=value_func_d,  # NEW

            n1=n1,
            n2=n2,
            q_funcs_rs_old=q_funcs_rs_old,
            targ_q_funcs_rs_old=targ_q_funcs_rs_old,
            tau_s2dc=tau_s2dc,
            actor_optim=actor_optim,  # Use standard name for base class
            critic_optim=critic_optim,  # Use standard name for base class
            critic_d_optim=critic_d_optim,
            delta_optim=delta_optim,
            actor_d=actor_d,         # ADD THIS LINE
            actor_d_optim=actor_d_optim,  # ADD THIS LINE           
        )
                
        self._impl = S2DCIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder_rs=q_func_forwarder_rs,
            targ_q_func_forwarder_rs=targ_q_func_forwarder_rs,
            q_func_forwarder_d=q_func_forwarder_d,
            targ_q_func_forwarder_d=targ_q_func_forwarder_d,
            q_func_forwarder_rs_old=q_func_forwarder_rs_old,
            targ_q_func_forwarder_rs_old=targ_q_func_forwarder_rs_old,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            initial_tau_s2dc=self._config.initial_tau_s2dc,
            min_tau_s2dc=self._config.min_tau_s2dc,
            max_tau_s2dc=self._config.max_tau_s2dc,
            tau_decay=self._config.tau_decay,
            warm_up_steps=self._config.warm_up_steps,
            blend_steps=self._config.blend_steps,
            kl_div_threshold=self._config.kl_div_threshold,
            tau_increase_rate=self._config.tau_increase_rate,
            weight_clip_percentile=self._config.weight_clip_percentile,
            gradient_clip_norm=self._config.gradient_clip_norm,
            reward_scaler=self._config.reward_scaler,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(S2DCIQLConfig)