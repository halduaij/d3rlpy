import dataclasses
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from ....models.torch import (
    DiscreteEnsembleQFunctionForwarder,
)
from ....torch_utility import (
    Modules,
    TorchMiniBatch,
    hard_sync,
)
from ....optimizers.optimizers import OptimizerWrapper
from ....types import Shape, TorchObservation
from .dqn_impl import DoubleDQNImpl, DQNLoss

__all__ = ["DiscreteIQLImpl", "DiscreteIQLModules", "DiscreteIQLLoss"]


@dataclasses.dataclass(frozen=True)
class DiscreteIQLModules(Modules):
    policy: nn.Module  # CategoricalPolicy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    value_func: nn.Module  # ValueFunction
    actor_optim: OptimizerWrapper
    critic_optim: OptimizerWrapper


@dataclasses.dataclass(frozen=True)
class DiscreteIQLLoss(DQNLoss):
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class DiscreteIQLImpl(DoubleDQNImpl):
    _modules: DiscreteIQLModules
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteIQLModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        compiled: bool,
        device: str,
    ):
        # Initialize parent class attributes directly
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._modules = modules
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._gamma = gamma
        self._tau = tau
        self._compiled = compiled
        self._device = device
        
        # IQL specific parameters
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight

        # Initialize grad steps counter
        self._grad_step = 0
        
        # Sync target networks
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> DiscreteIQLLoss:
        # Compute Q-function loss
        q_loss = self._compute_q_function_loss(batch, q_tpn)
        
        # Compute value function loss
        v_loss = self._compute_value_loss(batch)
        
        # Compute actor loss
        actor_loss = self._compute_actor_loss(batch)
        
        critic_loss = q_loss + v_loss
        
        return DiscreteIQLLoss(
            loss=critic_loss + actor_loss,  # Total loss
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            q_loss=q_loss,
            v_loss=v_loss,
        )

    def _compute_q_function_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-function loss with discrete actions."""
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def _compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Compute value function loss using expectile regression."""
        # Get Q-values for taken actions
        q_values = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, reduction="min"
        )
        q_t = q_values.gather(1, batch.actions.long().unsqueeze(1)).squeeze(1)
        
        # Compute value function output
        v_t = self._modules.value_func(batch.observations).squeeze(-1)
        
        # Expectile regression
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        
        return (weight * (diff**2)).mean()

    def _compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Compute actor loss using advantage weighted behavioral cloning."""
        # Get logits from policy
        logits = self._modules.policy(batch.observations)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_taken = log_probs.gather(1, batch.actions.long().unsqueeze(1)).squeeze(1)
        
        # Compute advantage weights
        with torch.no_grad():
            weight = self._compute_weight(batch)
        
        # Advantage weighted loss
        return -(weight * log_prob_taken).mean()

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Compute advantage weights for policy training."""
        # Get Q-values for taken actions
        q_values = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, reduction="min"
        )
        q_t = q_values.gather(1, batch.actions.long().unsqueeze(1)).squeeze(1)
        
        # Get value estimates
        v_t = self._modules.value_func(batch.observations).squeeze(-1)
        
        # Compute advantage
        adv = q_t - v_t
        
        # Compute weights with temperature and clipping
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Compute target values using the value function."""
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations).squeeze(-1)

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        """Predict best action using the learned policy."""
        logits = self._modules.policy(x)
        return logits.argmax(dim=1)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        """Sample action from the learned policy."""
        logits = self._modules.policy(x)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        """Update all components of discrete IQL."""
        self._grad_step = grad_step
        
        # Compute targets
        q_tpn = self.compute_target(batch)
        
        # Compute losses
        loss = self.compute_loss(batch, q_tpn)
        
        # Update critic (Q-functions and value function)
        self._modules.critic_optim.zero_grad()
        loss.critic_loss.backward(retain_graph=True)
        self._modules.critic_optim.step()
        
        # Update actor
        self._modules.actor_optim.zero_grad()
        loss.actor_loss.backward()
        self._modules.actor_optim.step()
        
        # Update target networks with soft update
        if grad_step % 1 == 0:  # Update every step with tau
            self.update_target()
        
        return {
            "actor_loss": float(loss.actor_loss.cpu().detach()),
            "critic_loss": float(loss.critic_loss.cpu().detach()),
            "q_loss": float(loss.q_loss.cpu().detach()),
            "v_loss": float(loss.v_loss.cpu().detach()),
        }

    def update_target(self) -> None:
        """Soft update for target networks."""
        # Soft update Q networks
        for targ_q, q in zip(self._modules.targ_q_funcs, self._modules.q_funcs):
            for targ_param, param in zip(targ_q.parameters(), q.parameters()):
                targ_param.data.copy_(
                    self._tau * param.data + (1.0 - self._tau) * targ_param.data
                )