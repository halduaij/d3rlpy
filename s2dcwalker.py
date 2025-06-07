import argparse
import d3rlpy
import torch
import numpy as np
class ActorDEvaluator:
    def __init__(self, env, n_trials=10):
        self.env = env
        self.n_trials = n_trials
    
    def __call__(self, algo, dataset) -> float:
        """Evaluate using actor_d (trained with Q_D)."""
        impl = algo._impl
        
        scores = []
        for _ in range(self.n_trials):
            obs, _ = self.env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                torch_obs = torch.from_numpy(obs.reshape(1, -1)).float().to(impl._device)
                
                with torch.no_grad():
                    # Use actor_d instead of main actor
                    action = impl._modules.actor_d(torch_obs).squashed_mu
                    action_np = action.cpu().numpy()[0]
                
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                total_reward += float(reward)
                done = terminated or truncated
                
            scores.append(total_reward)
            
        return float(np.mean(scores))
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="walker2d-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # Create reward scaler
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    # Create S2DC-IQL with tuned hyperparameters
    s2dciql = d3rlpy.algos.S2DCIQLConfig(
        # IQL parameters
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,  # Lowered from 3e-4
        actor_optim_factory=d3rlpy.optimizers.AdamFactory(),
      #  critic_optim_factory=d3rlpy.optimizers.AdamFactory(
      #      lr_scheduler_factory=d3rlpy.optimizers.CosineAnnealingLRFactory(
      #          T_max=500000
      #      ),
      #  ),
        batch_size=256,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        
        # S2DC specific - TUNED VALUES
        initial_tau_s2dc=1,      # Lowered from 1.0
        min_tau_s2dc=0.25,          # Raised from 0.01
        max_tau_s2dc=2.0,          # Lowered from 10.0
        tau_decay=0.99,            # Faster decay from 0.995
        warm_up_steps=20000,       # Extended from 5000
        blend_steps=50000,         # Extended from 5000
        kl_div_threshold=100.0,    # Effectively disable KL-based adjustment
        tau_increase_rate=1.0,     # No increase (disable KL adjustment)
        weight_clip_percentile=95.0,  # More aggressive clipping from 99.0
        gradient_clip_norm=999.0,    # Added gradient clipping
        
        # Delta network specific
        delta_learning_rate=3e-4,  # Lowered from default 3e-4
        delta_optim_factory=d3rlpy.optimizers.AdamFactory(
            lr_scheduler_factory=d3rlpy.optimizers.CosineAnnealingLRFactory(
                T_max=500000
            ),
        ),
        
        # Features
        reward_scaler=reward_scaler,
        compile_graph=args.compile,
    ).create(device='cuda')

    s2dciql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={
    "environment": d3rlpy.metrics.EnvironmentEvaluator(env),
    "environment_qd": ActorDEvaluator(env),},
        experiment_name=f"S2DC-IQL_tuned_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()