# d3rlpy/algos/qlearning/torch/s2dc_iql_impl.py
"""PyTorch implementation of S2DC-IQL."""

import dataclasses
import math
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    Parameter,
    build_gaussian_distribution,
    get_parameter,
)
from ....optimizers import OptimizerWrapper
from ....preprocessing import RewardScaler
from ....torch_utility import TorchMiniBatch, soft_sync, hard_sync
from ....types import Shape, TorchObservation
from .ddpg_impl import DDPGBaseImpl, DDPGBaseModules


@dataclasses.dataclass(frozen=True)
class S2DCIQLModules(DDPGBaseModules):
    value_func: ValueFunction
    value_func_d: ValueFunction  # NEW: For Q_D

    # Additional Q-functions for S2DC
    q_funcs_d: nn.ModuleList
    targ_q_funcs_d: nn.ModuleList
    n1: nn.ModuleList  # Difference network 1
    n2: nn.ModuleList  # Difference network 2
    q_funcs_rs_old: nn.ModuleList
    targ_q_funcs_rs_old: nn.ModuleList
    tau_s2dc: Parameter
    # Additional optimizers
    critic_d_optim: OptimizerWrapper
    delta_optim: OptimizerWrapper
    critic_d_optim: OptimizerWrapper
    delta_optim: OptimizerWrapper
    actor_d: NormalPolicy  # NEW: Policy trained with Q_D
    actor_d_optim: OptimizerWrapper  # NEW: Optimizer for actor_d


class S2DCIQLImpl(DDPGBaseImpl):
    _modules: S2DCIQLModules
    _q_func_forwarder_rs: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder_rs: ContinuousEnsembleQFunctionForwarder
    _q_func_forwarder_d: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder_d: ContinuousEnsembleQFunctionForwarder
    _q_func_forwarder_rs_old: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder_rs_old: ContinuousEnsembleQFunctionForwarder
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _initial_tau_s2dc: float
    _min_tau_s2dc: float
    _max_tau_s2dc: float
    _tau_decay: float
    _warm_up_steps: int
    _blend_steps: int
    _kl_div_threshold: float
    _tau_increase_rate: float
    _weight_clip_percentile: float
    _gradient_clip_norm: float
    _reward_scaler: Optional[RewardScaler]
    _step: int
   
    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: S2DCIQLModules,
        q_func_forwarder_rs: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder_rs: ContinuousEnsembleQFunctionForwarder,
        q_func_forwarder_d: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder_d: ContinuousEnsembleQFunctionForwarder,
        q_func_forwarder_rs_old: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder_rs_old: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        initial_tau_s2dc: float,
        min_tau_s2dc: float,
        max_tau_s2dc: float,
        tau_decay: float,
        warm_up_steps: int,
        blend_steps: int,
        kl_div_threshold: float,
        tau_increase_rate: float,
        weight_clip_percentile: float,
        gradient_clip_norm: float,
        reward_scaler: Optional[RewardScaler],
        compiled: bool,
        device: str,
        
    ):
        # Use q_func_forwarder_rs as the main q_func_forwarder for parent class
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder_rs,
            targ_q_func_forwarder=targ_q_func_forwarder_rs,
            gamma=gamma,
            tau=tau,
            compiled=compiled,
            device=device,
        )
        self._q_func_forwarder_rs = q_func_forwarder_rs
        self._targ_q_func_forwarder_rs = targ_q_func_forwarder_rs
        self._q_func_forwarder_d = q_func_forwarder_d
        self._targ_q_func_forwarder_d = targ_q_func_forwarder_d
        self._q_func_forwarder_rs_old = q_func_forwarder_rs_old
        self._targ_q_func_forwarder_rs_old = targ_q_func_forwarder_rs_old
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._initial_tau_s2dc = initial_tau_s2dc
        self._min_tau_s2dc = min_tau_s2dc
        self._max_tau_s2dc = max_tau_s2dc
        self._tau_decay = tau_decay
        self._warm_up_steps = warm_up_steps
        self._blend_steps = blend_steps
        self._kl_div_threshold = kl_div_threshold
        self._tau_increase_rate = tau_increase_rate
        self._weight_clip_percentile = weight_clip_percentile
        self._gradient_clip_norm = gradient_clip_norm
        self._reward_scaler = reward_scaler
                
        # Initialize old networks with current values
        hard_sync(modules.q_funcs_rs_old, modules.q_funcs)
        hard_sync(modules.targ_q_funcs_rs_old, modules.targ_q_funcs)
        
        self._step = 0
        self._prev_weights = None  # For EMA smoothing
        self._tau_ema_rate = 0.005  # Add this line

    def _get_batch_size(self, observations: TorchObservation) -> int:
        """Get batch size handling multi-modal observations."""
        if isinstance(observations, torch.Tensor):
            return observations.shape[0]
        elif isinstance(observations, (list, tuple)):
            return observations[0].shape[0]
        else:
            raise ValueError(f"Unsupported observation type: {type(observations)}")
        
    def compute_s2dc_weights(
        self, 
        batch: TorchMiniBatch,
        n1_pred: torch.Tensor,
        n2_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute S2DC importance weights with KL tracking."""
        # Get current tau value (already a float)
        tau = self._get_current_tau()
        
        # Ensure predictions are 1D
        n1_pred = n1_pred.squeeze()
        n2_pred = n2_pred.squeeze()
        
        # Compute log p_rs using the S2DC formula
        # Soften n2 to ensure stability
        softened_n2 = 0.5 + 0.5 * torch.abs(n2_pred)
        
        # Add numerical stability
        log_p_rs = -softened_n2 / (tau + 1e-8) + torch.log(torch.abs(n1_pred) + 1e-8)
        
        # Clamp log values for numerical stability
        log_p_rs = torch.clamp(log_p_rs, min=-20, max=20)
        
        # Get unnormalized p_rs
        p_rs_unnorm = torch.exp(log_p_rs)
        
        # For IQL, we don't have explicit density p_d, so we assume uniform
        batch_size = self._get_batch_size(batch.observations)
        p_d = torch.ones(batch_size, device=self._device) / batch_size
        
        # Compute normalized p_rs
        numerator = p_rs_unnorm * p_d
        denom = numerator.sum() + 1e-8
        p_rs = numerator / denom
        
        # Compute importance ratio
        ratio = p_rs / (p_d + 1e-8)
        
        # Clip weights at percentile
        if self._weight_clip_percentile < 100:
            percentile_val = torch.quantile(
                ratio, self._weight_clip_percentile / 100.0
            )
            ratio_clipped = torch.minimum(ratio, percentile_val)
        else:
            ratio_clipped = ratio
            
        weights = ratio_clipped
        
        # Compute KL divergence for monitoring
        kl_div = torch.sum(
            p_rs * (torch.log(p_rs + 1e-8) - torch.log(p_d + 1e-8))
        )
        
        return weights, kl_div, tau
    def compute_actor_d_loss(self, batch: TorchMiniBatch) -> Dict[str, torch.Tensor]:
        """Compute actor loss using Q_D (uniform sampling)."""
        # Get policy distribution from actor_d
        dist = build_gaussian_distribution(self._modules.actor_d(batch.observations))
        log_probs = dist.log_prob(batch.actions)
        
        # Compute advantage weight using Q_D
        with torch.no_grad():
            q_t = self._q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            v_t = self._modules.value_func_d(batch.observations)
            
            q_t = q_t.squeeze()
            v_t = v_t.squeeze()
            
            adv = q_t - v_t
            weight = (self._weight_temp * adv).exp().clamp(
                max=self._max_weight
            )
        
        actor_d_loss = -(weight * log_probs).mean()
        
        return {
            "actor_d_loss": actor_d_loss,
        }       
    def _get_current_tau(self) -> float:
        """Get current tau value (no decay, just return current value)."""
        tau_tensor = get_parameter(self._modules.tau_s2dc).exp()
        tau = tau_tensor.item() if tau_tensor.numel() == 1 else tau_tensor[0].item()
        return tau
 
    
    def _update_tau_adaptive(self, n1_values: torch.Tensor, n2_values: torch.Tensor) -> None:
        """Adapt tau using DisCor-style EMA of appropriate scale."""
        with torch.no_grad():
            # Option 1: Use n1 (Bellman error) as the scale signal
            # This represents the "uncertainty" in Q-values
            error_scale = n1_values.abs().mean()
            
            # Option 2: Use a combination of n1 and n2
            # error_scale = (n1_values.abs().mean() + 0.1 * n2_values.abs().mean()) / 1.1
            
            # Option 3: Use percentile of n2 to be robust to outliers
            # error_scale = torch.quantile(n2_values.abs(), 0.75)
            
            current_tau = self._get_current_tau()
            
            # DisCor-style EMA update
            new_tau = (1 - self._tau_ema_rate) * current_tau + self._tau_ema_rate * error_scale
            
            # Clamp to bounds
            new_tau = max(self._min_tau_s2dc, min(self._max_tau_s2dc, new_tau))
            
            # Update log-space parameter
            new_log_tau = torch.tensor([math.log(new_tau)], device=self._device)
            get_parameter(self._modules.tau_s2dc).copy_(new_log_tau)
 
 
    def _adjust_tau_based_on_kl(self, kl_div: torch.Tensor) -> None:
        """Adjust tau if KL divergence is too high."""
        if kl_div > self._kl_div_threshold:
            with torch.no_grad():
                current_log_tau = get_parameter(self._modules.tau_s2dc)
                new_log_tau = torch.minimum(
                    torch.tensor([math.log(self._max_tau_s2dc)], device=self._device),
                    current_log_tau + math.log(self._tau_increase_rate)
                )
                get_parameter(self._modules.tau_s2dc).copy_(new_log_tau)
            
    def _blend_weights(
        self, uniform_w: torch.Tensor, s2dc_w: torch.Tensor
    ) -> torch.Tensor:
        """Blend uniform and S2DC weights during warm-up/transition."""
        if self._step < self._warm_up_steps:
            return uniform_w
        elif self._step < self._warm_up_steps + self._blend_steps:
            # Linear blend
            blend_progress = (
                self._step - self._warm_up_steps
            ) / self._blend_steps
            return (1 - blend_progress) * uniform_w + blend_progress * s2dc_w
        else:
            return s2dc_w
            
    def compute_actor_loss(self, batch: TorchMiniBatch) -> Dict[str, torch.Tensor]:
        """Compute IQL actor loss (standard AWR without S2DC weighting)."""
        # Get policy distribution
        dist = build_gaussian_distribution(self._modules.policy(batch.observations))
        log_probs = dist.log_prob(batch.actions)
        
        # Compute advantage weight (IQL style)
        with torch.no_grad():
            q_t = self._q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            v_t = self._modules.value_func(batch.observations)
            
            # Ensure shapes are consistent
            q_t = q_t.squeeze()
            v_t = v_t.squeeze()
            
            adv = q_t - v_t
            weight = (self._weight_temp * adv).exp().clamp(
                max=self._max_weight
            )
        
        # Actor loss with standard IQL weighting (no S2DC weights)
        actor_loss = -(weight * log_probs).mean()
        
        return {
            "actor_loss": actor_loss,
        }
        
    def compute_value_loss(self, batch: TorchMiniBatch) -> Dict[str, torch.Tensor]:
        """Compute value losses for both V_RS and V_D"""
        losses = {}
        
        # V_RS loss (trained on Q_RS)
        with torch.no_grad():
            q_rs = self._q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "min"
            ).squeeze()
        
        v_rs = self._modules.value_func(batch.observations).squeeze()
        diff_rs = q_rs - v_rs
        weight_rs = (self._expectile - (diff_rs < 0.0).float()).abs()
        losses['value_loss_rs'] = (weight_rs * (diff_rs ** 2)).mean()
        
        # V_D loss (trained on Q_D) - can use different expectile
        with torch.no_grad():
            q_d = self._q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            ).squeeze()
        
        v_d = self._modules.value_func_d(batch.observations).squeeze()
        diff_d = q_d - v_d
        
        # Option: Use different expectile for V_D
        expectile_d = 0.7  # More conservative for Q_D
        weight_d = (expectile_d - (diff_d < 0.0).float()).abs()
        losses['value_loss_d'] = (weight_d * (diff_d ** 2)).mean()
        
        return losses
        
    def compute_critic_loss(
        self, batch: TorchMiniBatch
    ) -> Dict[str, torch.Tensor]:
        """Compute Q-function losses for both QRS and QD."""
        with torch.no_grad():
            # Q_RS uses its own value function
            target_v_rs = self._modules.value_func(batch.next_observations)
            
            # Q_D uses its own value function
            target_v_d = self._modules.value_func_d(batch.next_observations)
        
        # Q_D loss with its own value target
        q_loss_d = self._q_func_forwarder_d.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=target_v_d,  # Use V_D
            terminals=batch.terminals,
            gamma=self._gamma ** batch.intervals,
        )
        
        
        # For QRS with S2DC weights, we need custom computation
        # Get S2DC weights - use networks directly
        with torch.no_grad():
            n1_out = self._modules.n1[0](batch.observations, batch.actions)
            n2_out = self._modules.n2[0](batch.observations, batch.actions)
            n1_pred = n1_out.q_value.squeeze()
            n2_pred = n2_out.q_value.squeeze()
        
        s2dc_weights, kl_div, tau = self.compute_s2dc_weights(batch, n1_pred, n2_pred)
        
        batch_size = self._get_batch_size(batch.observations)
        uniform_weights = torch.ones(batch_size, device=self._device)
        weights = self._blend_weights(uniform_weights, s2dc_weights)
        
        # Optional: Apply EMA smoothing to weights for stability
#        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
#            weights = 0.9 * self._prev_weights + 0.1 * weights
        self._prev_weights = weights.detach().clone()
        
        # Compute weighted loss manually since we need per-sample weights
        with torch.no_grad():
            # Ensure rewards and target_v are properly shaped
            rewards = batch.rewards.squeeze() if batch.rewards.numel() == batch.rewards.shape[0] else batch.rewards.squeeze(-1)
            target_v_rs_squeezed = target_v_rs.squeeze() if target_v_rs.numel() == target_v_rs.shape[0] else target_v_rs.squeeze(-1)
            terminals = batch.terminals.squeeze() if batch.terminals.numel() == batch.terminals.shape[0] else batch.terminals.squeeze(-1)
            
            # Handle intervals - ensure it's 1D
            intervals = batch.intervals
            if intervals.dim() > 1:
                intervals = intervals.squeeze()
            
            td_target = rewards + (self._gamma ** intervals) * target_v_rs.squeeze() * (1 - terminals)
        
        # Get Q-values using forwarder with "none" reduction to get all critics
        q_values_all = self._q_func_forwarder_rs.compute_expected_q(
            batch.observations, batch.actions, "none"
        )  # Shape: [n_critics, batch_size]
        
        # Ensure shapes match for broadcasting
        if q_values_all.dim() == 3:  # [n_critics, batch_size, 1]
            q_values_all = q_values_all.squeeze(-1)  # [n_critics, batch_size]
        
        # Expand target to match Q-values shape [n_critics, batch_size]
        td_target_expanded = td_target.unsqueeze(0).expand_as(q_values_all)
        
        # Compute weighted MSE loss
        td_errors = (q_values_all - td_target_expanded) ** 2  # [n_critics, batch_size]
        weighted_td_errors = td_errors * weights.unsqueeze(0)  # [n_critics, batch_size]
        q_loss_rs = weighted_td_errors.mean()
            
        return {
            "q_loss_rs": q_loss_rs,
            "q_loss_d": q_loss_d,
            "total_critic_loss": q_loss_rs + q_loss_d,
            "kl_div": kl_div.detach(),
            "tau": tau,
        }
        
    def compute_delta_loss(self, batch: TorchMiniBatch) -> Dict[str, torch.Tensor]:
        """Compute loss for difference networks N1 and N2."""
        with torch.no_grad():
            # Get current Q values
            q_rs = self._q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            q_d = self._q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            
            # Target for N1: Q_RS - Bellman(Q_RS)
            # Using value function for Bellman target (IQL style)
            v_next = self._modules.value_func(batch.next_observations)
            
            # Ensure proper shapes
            rewards = batch.rewards.squeeze() if batch.rewards.numel() == batch.rewards.shape[0] else batch.rewards.squeeze(-1)
            v_next_squeezed = v_next.squeeze() if v_next.numel() == v_next.shape[0] else v_next.squeeze(-1)
            terminals = batch.terminals.squeeze() if batch.terminals.numel() == batch.terminals.shape[0] else batch.terminals.squeeze(-1)
            
            # Handle intervals - ensure it's 1D
            intervals = batch.intervals
            if intervals.dim() > 1:
                intervals = intervals.squeeze()
            
            bellman_target = rewards + (self._gamma ** intervals) * v_next_squeezed * (1 - terminals)
            n1_target = q_rs.squeeze() - bellman_target
            
            # Target for N2: Q_RS - Q_D
            n2_target = q_rs.squeeze() - q_d.squeeze()
            
        # Get predictions directly from the first (and only) network in the ModuleList
        n1_out = self._modules.n1[0](batch.observations, batch.actions)
        n2_out = self._modules.n2[0](batch.observations, batch.actions)
        
        # Extract q_value from QFunctionOutput and ensure 1D
        n1_pred = n1_out.q_value.squeeze()
        n2_pred = n2_out.q_value.squeeze()
        
        # MSE losses
        n1_loss = F.mse_loss(n1_pred, n1_target)
        n2_loss = F.mse_loss(n2_pred, n2_target)
        
        return {
            "n1_loss": n1_loss,
            "n2_loss": n2_loss,
            "delta_loss": n1_loss + n2_loss,
        }
    def predict_best_action_qd(self, x: TorchObservation) -> torch.Tensor:
        """Predict best action using Q_D instead of Q_RS."""
        # Temporarily swap Q-function forwarders
        original_q_func_forwarder = self._q_func_forwarder
        self._q_func_forwarder = self._q_func_forwarder_d
        
        # Get action using Q_D
        action = self.inner_predict_best_action(x)
        
        # Restore original
        self._q_func_forwarder = original_q_func_forwarder
        
        return action       
    def inner_update(self, batch: TorchMiniBatch, grad_step: int) -> Dict[str, float]:
        """Main update function."""
        self._step = grad_step
        metrics = {}
        
        # Update difference networks less frequently for stability
        if grad_step % 10 == 0:  # Update every 10 steps instead of every step
            self._modules.delta_optim.zero_grad()
            delta_losses = self.compute_delta_loss(batch)
            delta_losses["delta_loss"].backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self._modules.n1.parameters()) + list(self._modules.n2.parameters()),
                self._gradient_clip_norm,
            )
            self._modules.delta_optim.step()
            with torch.no_grad():
                n1_out = self._modules.n1[0](batch.observations, batch.actions)
                n2_out = self._modules.n2[0](batch.observations, batch.actions)
                n1_values = n1_out.q_value.squeeze()
                n2_values = n2_out.q_value.squeeze()
                
                self._update_tau_adaptive(n1_values, n2_values)
            
            metrics.update({k: v.item() for k, v in delta_losses.items()})
        else:
            # Add placeholder values when not updating
            metrics.update({
                "n1_loss": 0.0,
                "n2_loss": 0.0,
                "delta_loss": 0.0,
            })
            
        # Update critics (QRS, QD, and value function)
        self._modules.critic_optim.zero_grad()
        self._modules.critic_d_optim.zero_grad()
            

        critic_losses = self.compute_critic_loss(batch)
        value_losses = self.compute_value_loss(batch)
        
        # Q_RS + V_RS update
        total_rs_loss = critic_losses["q_loss_rs"] + value_losses["value_loss_rs"]
        total_rs_loss.backward()
        torch.nn.utils.clip_grad_norm_(
        list(self._modules.q_funcs.parameters()) + 
        list(self._modules.value_func.parameters()),
        self._gradient_clip_norm,)
          
        self._modules.critic_optim.step()
        

        # Q_D + V_D update - recompute to avoid graph reuse
        self._modules.critic_d_optim.zero_grad()

        # Recompute Q_D loss
        q_loss_d_fresh = self._q_func_forwarder_d.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=self._modules.value_func_d(batch.next_observations),
            terminals=batch.terminals,
            gamma=self._gamma ** batch.intervals,
        )

        # Recompute V_D loss
        with torch.no_grad():
            q_t_d = self._q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            ).squeeze()
        v_t_d = self._modules.value_func_d(batch.observations).squeeze()
        diff_d = q_t_d - v_t_d
        weight_d = (self._expectile - (diff_d < 0.0).float()).abs()
        value_loss_d_fresh = (weight_d * (diff_d ** 2)).mean()

        # Combined backward
        total_d_loss_fresh = q_loss_d_fresh + value_loss_d_fresh
        total_d_loss_fresh.backward()

        # Gradient clipping for both Q_D and V_D
        torch.nn.utils.clip_grad_norm_(
            list(self._modules.q_funcs_d.parameters()) +
            list(self._modules.value_func_d.parameters()),
            self._gradient_clip_norm,
        )
        self._modules.critic_d_optim.step()

        
        metrics.update({
            "q_loss_rs": critic_losses["q_loss_rs"].item(),
            "q_loss_d": critic_losses["q_loss_d"].item(),
            "value_loss_rs": value_losses["value_loss_rs"].item(),
            "value_loss_d": value_losses["value_loss_d"].item(),
            "kl_div": critic_losses["kl_div"].item(),
            "tau": critic_losses["tau"],
        })
        with torch.no_grad():
            # Get Q values for the current batch
            q_rs_values = self._q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "mean"
            )
            q_d_values = self._q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "mean"
            )
            
            # Compute difference
            q_diff = q_rs_values - q_d_values
            
            # Add statistics to metrics
            metrics["q_rs_minus_q_d_mean"] = q_diff.mean().item()
            metrics["q_rs_minus_q_d_std"] = q_diff.std().item()
            metrics["q_rs_minus_q_d_max"] = q_diff.max().item()
            metrics["q_rs_minus_q_d_min"] = q_diff.min().item()       
        # Optionally disable KL-based adjustment (comment out if needed)
        # self._adjust_tau_based_on_kl(critic_losses["kl_div"])
        
        # Update actor
        self._modules.actor_optim.zero_grad()
        actor_losses = self.compute_actor_loss(batch)
        actor_losses["actor_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self._modules.policy.parameters(),
            self._gradient_clip_norm,
        )
        self._modules.actor_optim.step()
        
        metrics.update({
            "actor_loss": actor_losses["actor_loss"].item(),
        })
        self._modules.actor_d_optim.zero_grad()
        actor_d_losses = self.compute_actor_d_loss(batch)
        actor_d_losses["actor_d_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self._modules.actor_d.parameters(),
            self._gradient_clip_norm,
        )
        self._modules.actor_d_optim.step()

        metrics.update({
            "actor_d_loss": actor_d_losses["actor_d_loss"].item(),
        })        
        # Soft updates
        if grad_step % 1 == 0:  # Update every step
            soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, self._tau)
            soft_sync(self._modules.targ_q_funcs_d, self._modules.q_funcs_d, self._tau)
            
            # Periodically update old networks (less frequently)
            if grad_step % 100 == 0:  # Changed from 10 to 100
                hard_sync(self._modules.q_funcs_rs_old, self._modules.q_funcs)
                hard_sync(self._modules.targ_q_funcs_rs_old, self._modules.targ_q_funcs)
                
        return metrics
    def inner_predict_best_action_d(self, x: TorchObservation) -> torch.Tensor:
        """Predict best action using actor_d."""
        return self._modules.actor_d(x).squashed_mu        
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        """Compute target value for Q-learning."""
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)
        
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        """Predict best action (deterministic)."""
        return self._modules.policy(x).squashed_mu
        
    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        """Sample action from policy."""
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()
        
    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict Q-value."""
        return self._q_func_forwarder_rs.compute_expected_q(
            x, action, reduction="mean"
        ).reshape(-1)
        
    @property
    def policy(self) -> NormalPolicy:
        return self._modules.policy
        
    @property
    def policy_optim(self) -> OptimizerWrapper:
        return self._modules.actor_optim
        
    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs
        
    @property
    def q_function_optim(self) -> OptimizerWrapper:
        return self._modules.critic_optim