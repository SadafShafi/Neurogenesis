import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import torch

def train_all_together():
    # Force GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 0. Path Setup
    base_path = "./allTogether/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    stages = [
        {
            "id": "MiniGrid-Empty-5x5-v0",
            "timesteps": 40000,
            "lr": 3e-4,
            "batch_size": 64,
            "ent_coef": 0.0,
        },
        {
            "id": "MiniGrid-Empty-8x8-v0",
            "timesteps": 80000,
            "lr": 1e-3,
            "batch_size": 64,
            "ent_coef": 0.01,
        },
        {
            "id": "MiniGrid-Empty-16x16-v0",
            "timesteps": 300000,
            "lr": 5e-4,
            "batch_size": 128,
            "ent_coef": 0.05,
        },
        {
            "id": "MiniGrid-Empty-Random-6x6-v0",
            "timesteps": 60000,
            "lr": 3e-4,
            "batch_size": 64,
            "ent_coef": 0.01,
        }
    ]

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    model = None
    stage_results = []

    results_md_path = os.path.join(base_path, "results.md")
    with open(results_md_path, "w") as f:
        f.write("# Continual Learning Results - allTogether\n\n")

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"\n--- Starting Stage {i+1}: {env_id} ---")
        
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = Monitor(env)

        if model is None:
            # Initialize for the first time
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(log_path, "ppo_minigrid_tensorboard/"),
                learning_rate=stage["lr"],
                batch_size=stage["batch_size"],
                ent_coef=stage["ent_coef"],
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                device=device,
            )
        else:
            # Re-load model with NEW hyperparameters to ensure they take effect
            # In SB3, many attributes are baked into the optimizer/scheduler and 
            # simply setting 'model.learning_rate = ...' on an existing object is not enough.
            print(f"Transitioning to Stage {i+1}: Applying new hyperparameters (LR: {stage['lr']}, Batch: {stage['batch_size']}, Ent: {stage['ent_coef']})")
            temp_model_path = os.path.join(model_path, "temp_transition_model")
            model.save(temp_model_path)
            model = PPO.load(
                temp_model_path, 
                env=env, 
                device=device,
                custom_objects={
                    "learning_rate": stage["lr"],
                    "batch_size": stage["batch_size"],
                    "ent_coef": stage["ent_coef"]
                }
            )

        print(f"Training for {stage['timesteps']} steps...")

        print(f"Training for {stage['timesteps']} steps...")
        model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=False)
        
        # Test on CURRENT stage
        print(f"Evaluating on {env_id}...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Stage {i+1} Mean Reward: {mean_reward:.2f}")
        stage_results.append((env_id, mean_reward, std_reward))

        # Update results.md
        with open(results_md_path, "a") as f:
            f.write(f"## Stage {i+1}: {env_id}\n")
            f.write(f"### Performance\n")
            f.write(f"- **Mean Reward:** {mean_reward:.2f} +/- {std_reward:.2f}\n")
            f.write(f"- **Timesteps:** {stage['timesteps']}\n\n")
            f.write(f"### Configuration\n")
            f.write(f"- **Learning Rate:** {stage['lr']}\n")
            f.write(f"- **Batch Size:** {stage['batch_size']}\n")
            f.write(f"- **Ent Coef:** {stage['ent_coef']}\n")
            f.write(f"- **Architecture:** {policy_kwargs['net_arch']}\n\n")

    # Final Evaluation on ALL stages
    print("\n--- Final Evaluation on ALL Stages ---")
    with open(results_md_path, "a") as f:
        f.write("## Final Evaluation (Catastrophic Forgetting Check)\n")
        f.write("| Stage | Environment | Mean Reward | Std Reward |\n")
        f.write("|-------|-------------|-------------|------------|\n")

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"Final testing on {env_id}...")
        test_env = gym.make(env_id, render_mode="rgb_array")
        test_env = FlatObsWrapper(test_env)
        test_env = Monitor(test_env)
        
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
        print(f"Final Reward on {env_id}: {mean_reward:.2f}")
        
        with open(results_md_path, "a") as f:
            f.write(f"| {i+1} | {env_id} | {mean_reward:.2f} | {std_reward:.2f} |\n")
        
        test_env.close()

    # Save final model
    final_model_path = os.path.join(model_path, "ppo_minigrid_allTogether_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Results summary saved to {results_md_path}")

if __name__ == "__main__":
    train_all_together()
