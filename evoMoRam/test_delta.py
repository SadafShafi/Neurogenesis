import os
import sys
import torch
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import numpy as np

# Add parent directory to path to import evoMoRam
sys.path.insert(0, "/user/sshafi/u20715/neurogenesis")
from evoMoRam.evo_moram import EvoPPO, pad_state, evaluate_policy, EvoMoRAMLinear

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stages = [
        {"id": "MiniGrid-Empty-5x5-v0"},
        {"id": "MiniGrid-Empty-8x8-v0"},
        {"id": "MiniGrid-Empty-16x16-v0"},
        {"id": "MiniGrid-Empty-Random-6x6-v0"}
    ]

    max_obs_dim = 0 
    for stage in stages:
        temp_env = FlatObsWrapper(gym.make(stage["id"]))
        obs_dim = temp_env.observation_space.shape[0]
        max_obs_dim = max(max_obs_dim, obs_dim)
        temp_env.close()

    print(f"Max Observation Dimension: {max_obs_dim}")

    # 1. Recreate agent and add tasks to match the final model architecture
    temp_env = FlatObsWrapper(gym.make(stages[0]["id"]))
    action_dim = temp_env.action_space.n
    temp_env.close()

    agent = EvoPPO(max_obs_dim, action_dim, lr=1e-3, gamma=0.99, epochs=10, eps_clip=0.2, ent_coef=0.01, device=device)
    
    # Note: You need to match the exact architecture of the trained model.
    # If neurogenesis was triggered for some tasks, those will have r_full=16.
    # If probes were sufficient, they'll have r_probe=2.
    # For simplicity, we add full tasks for all 3 stages (worst-case architecture).
    for _ in range(3):
        agent.add_new_task(new_lr=1e-3)

    # 2. Load the final model state dict
    model_path = "/user/sshafi/u20715/neurogenesis/evoMoRam/models/ppo_minigrid_evomoram_final.pt"
    agent.policy.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded model weights successfully.")

    # 3. Evaluate under different delta thresholds
    deltas = [0.15, 0.05, 0.0]
    
    for delta in deltas:
        print(f"\n==========================================")
        print(f"Evaluating with EvoMoRAM delta = {delta}")
        print(f"==========================================")
        
        # Set delta for all EvoMoRAM layers
        for module in agent.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.delta = delta

        for i, stage in enumerate(stages):
            env_id = stage["id"]
            env = gym.make(env_id)
            env = FlatObsWrapper(env)
            mean_reward, std_reward = evaluate_policy(agent, env, max_obs_dim, n_eval_episodes=20)
            print(f"Stage {i+1} ({env_id}) | Mean Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
            env.close()

if __name__ == "__main__":
    run_test()
