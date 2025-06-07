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
            q_t = self._targ_q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            v_t = self._modules.value_func_d(batch.observations)
            
            
            adv = q_t - v_t
            weight = (self._weight_temp * adv).exp().clamp(
                max=self._max_weight
            )
        
        actor_d_loss = -(weight * log_probs).mean()
        
        return {
            "actor_d_loss": actor_d_loss,
        }       
    def _get_current_tau(self) -> float:
        """Get current tau value with decay."""
        tau_tensor = get_parameter(self._modules.tau_s2dc).exp()
        tau = tau_tensor.item() if tau_tensor.numel() == 1 else tau_tensor[0].item()
        
        # Apply decay based on step count
        decay_factor = self._tau_decay ** (self._step // 1000)
        tau_decayed = max(self._min_tau_s2dc, tau * decay_factor)
        
        return tau_decayed
        
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
            q_t = self._targ_q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
            v_t = self._modules.value_func(batch.observations)
            
            # Ensure shapes are consistent

            
            adv = q_t - v_t
            weight = (self._weight_temp * adv).exp().clamp(
                max=self._max_weight
            )
        
        # Actor loss with standard IQL weighting (no S2DC weights)
        actor_loss = -(weight * log_probs).mean()
        
        return {
            "actor_loss": actor_loss,
        }
            
        
    def compute_value_loss_rs(self, batch):
        with torch.no_grad():
            q_t = self._targ_q_func_forwarder_rs.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
        v_t = self._modules.value_func(batch.observations)
        diff   = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0).float()).abs()
        return (weight * diff.pow(2)).mean()

    def compute_value_loss_d(self, batch):
        with torch.no_grad():
            q_t = self._targ_q_func_forwarder_d.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
        v_t = self._modules.value_func_d(batch.observations)
        diff   = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0).float()).abs()
        return (weight * diff.pow(2)).mean()

        
    def compute_critic_loss(
        self, batch: TorchMiniBatch
    ) -> Dict[str, torch.Tensor]:
        """Compute Q-function losses for both QRS and QD."""
        with torch.no_grad():
            # IQL style target using value function
            target_v_rs = self._modules.value_func(batch.next_observations)
            target_v_d = self._modules.value_func_d(batch.next_observations)            
        # Standard loss for QD first (simpler)
        q_loss_d = self._q_func_forwarder_d.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=target_v_d,
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
        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            weights = 0.9 * self._prev_weights + 0.1 * weights
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
            
            td_target = rewards + (self._gamma ** intervals) * target_v_rs_squeezed * (1 - terminals)
        
        # Get Q-values using forwarder with "none" reduction to get all critics
        q_values_all = self._q_func_forwarder_rs.compute_expected_q(
            batch.observations, batch.actions, "none"
        )  # Shape: [n_critics, batch_size]
        
        # Ensure shapes match for broadcasting
        if q_values_all.dim() == 3:  # [n_critics, batch_size, 1]
            q_values_all = q_values_all.squeeze(-1)  # [n_critics, batch_size]
        
        # Expand target to match Q-values shape [n_critics, batch_size]
        td_target_expanded = td_target.unsqueeze(0).expand_as(q_values_all)
        q_d_values = self._q_func_forwarder_d.compute_expected_q(
            batch.observations, batch.actions, "mean"   # shape [B, 1]
        ).squeeze(-1)                                   # -> shape [B]
        
        # Compute weighted MSE loss
        td_errors = (q_values_all - td_target_expanded) ** 2  # [n_critics, batch_size]
        weighted_td_errors = td_errors * weights.unsqueeze(0)  # [n_critics, batch_size]
        q_loss_rs = weighted_td_errors.mean(dim=1).sum()  # mean per critic, then sum
            
        return {
            "q_loss_rs": q_loss_rs,
            "q_loss_d": q_loss_d,
            "total_critic_loss": q_loss_rs + q_loss_d,
            "kl_div": kl_div.detach(),
            "tau": tau,
            "final_weights": weights,  # ADD THIS LINE
            "td_target": td_target,               
            "q_rs_values": q_values_all.mean(0).detach(),  # shape [B]
            "q_d_values":  q_d_values.detach(),            # <= add this
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
      #      torch.nn.utils.clip_grad_norm_(
       #         list(self._modules.n1.parameters()) + list(self._modules.n2.parameters()),
        #        self._gradient_clip_norm,
         #   )
            self._modules.delta_optim.step()
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
        weights = critic_losses["final_weights"]  # ADD THIS LINE
        td_target = critic_losses["td_target"]  # ADD THIS LINE
        v_loss_rs = self.compute_value_loss_rs(batch)
        v_loss_d  = self.compute_value_loss_d(batch)

                # Combined RS loss
        total_rs_loss = critic_losses["q_loss_rs"] + v_loss_rs
        total_rs_loss.backward()       
        # Combined loss for QRS and value

        # Gradient clipping
    #    torch.nn.utils.clip_grad_norm_(
     #       list(self._modules.q_funcs.parameters()) + 
      #      list(self._modules.value_func.parameters()),
       #     self._gradient_clip_norm,
       # )
        self._modules.critic_optim.step()
        
        # Update QD separately
        
        total_d_loss = critic_losses["q_loss_d"] + v_loss_d
        self._modules.critic_d_optim.zero_grad()
        total_d_loss.backward()
        self._modules.critic_d_optim.step()
        
        metrics.update({
            "q_loss_rs": critic_losses["q_loss_rs"].item(),
            "q_loss_d": critic_losses["q_loss_d"].item(),
            "value_loss_rs": v_loss_rs.item(),
            "value_loss_d":  v_loss_d.item(),
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
            q_x= ( q_rs_values - q_d_values).square().mean()
            # Add statistics to metrics
            metrics["q_rs_minus_q_d_mean"] =q_x.item()
            metrics["q_rs_minus_q_d_std"] = q_diff.std().item()
            metrics["q_rs_minus_q_d_max"] = q_diff.max().item()
            metrics["q_rs_minus_q_d_min"] = q_diff.min().item()       
        # Optionally disable KL-based adjustment (comment out if needed)
        # self._adjust_tau_based_on_kl(critic_losses["kl_div"])
# -------- improved diagnostics (≈20 lines) --------
# ---------- diagnostics: no extra forward, no BN updates ----------
        with torch.no_grad():
            # q_rs_values, q_d_values, td_target, weights were returned from compute_critic_loss
            q_rs_vals = critic_losses["q_rs_values"].squeeze(-1)   # shape [B]
            q_d_vals  = critic_losses["q_d_values"].squeeze(-1)    # shape [B]
            td_tgt    = critic_losses["td_target"].squeeze(-1)     # shape [B]
            w_raw     = critic_losses["final_weights"].squeeze()   # shape [B]

            # normalise weights once
            w = w_raw / (w_raw.sum() + 1e-8)

            # 0) ESS
            ess = 1.0 / (w.pow(2).sum() + 1e-8)
            metrics["ess_frac"] = ess.item() / w.shape[0]

            # 1) weighted Bellman error + amplification
            td_err = (q_rs_vals - td_tgt).abs()
            bell_w = (td_err * w).sum()
            metrics["bellman_err_w"]  = bell_w.item()
            metrics["bellman_amplif"] = bell_w.item() / (td_err.mean().item() + 1e-8)

            # 2) weighted |Q_RS − Q_D| + amplification
            gap = (q_rs_vals - q_d_vals).abs()
            gap_w = (gap * w).sum()
            metrics["value_err_l1_w"] = gap_w.item()
            metrics["value_amplif"]   = gap_w.item() / (gap.mean().item() + 1e-8)
            self._modules.policy.eval()           # ADD: Also freeze policy
            self._modules.actor_d.eval()          # ADD: Also freeze actor_d
            self._modules.q_funcs.eval()
            self._modules.q_funcs_d.eval()
            
            # 3) on-policy signed advantage gap (freeze BN/DO during these extra passes)
            self._modules.q_funcs.eval()
            self._modules.q_funcs_d.eval()
            a_rs = self._modules.policy(batch.observations).squashed_mu
            a_d  = self._modules.actor_d(batch.observations).squashed_mu
            q_rs_pi = self._q_func_forwarder_rs.compute_expected_q(batch.observations, a_rs, "mean")
            q_d_pi  = self._q_func_forwarder_d.compute_expected_q(batch.observations, a_d,  "mean")
            metrics["adv_gap"] = (q_rs_pi - q_d_pi).mean().item()
            self._modules.q_funcs.train()
            self._modules.q_funcs_d.train()
                    # ------------------------------------------------------------------
            self._modules.policy.train()
            self._modules.actor_d.train()


        # ---------------------------------------------------
        # Update actor
        self._modules.q_funcs.eval()

        self._modules.actor_optim.zero_grad()
        actor_losses = self.compute_actor_loss(batch)
        actor_losses["actor_loss"].backward()
        #torch.nn.utils.clip_grad_norm_(
        #    self._modules.policy.parameters(),
        #    self._gradient_clip_norm,
        #)
        self._modules.actor_optim.step()
        self._modules.q_funcs.train()  # Set back to train mode

        metrics.update({
            "actor_loss": actor_losses["actor_loss"].item(),
        })
        self._modules.q_funcs_d.eval()

        self._modules.actor_d_optim.zero_grad()
        actor_d_losses = self.compute_actor_d_loss(batch)
        actor_d_losses["actor_d_loss"].backward()
        #torch.nn.utils.clip_grad_norm_(
        #    self._modules.actor_d.parameters(),
        #    self._gradient_clip_norm,
        #)
        self._modules.actor_d_optim.step()

        metrics.update({
            "actor_d_loss": actor_d_losses["actor_d_loss"].item(),
        })        
        # Soft updates
        self._modules.q_funcs_d.train()

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