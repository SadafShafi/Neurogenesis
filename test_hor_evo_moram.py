import os
import torch
import numpy as np
import gymnasium as gym

from hor_evo_moram import (
    ActorCritic, 
    EvoMoRAMLinear, 
    evaluate_policy, 
    make_wrapped_env
)

class DummyAgent:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

def reconstruct_and_load(policy, state_dict_path, device):
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # Analyze state_dict to find A_list and B_list shapes for each EvoMoRAMLinear
    for module_name, module in policy.named_modules():
        if isinstance(module, EvoMoRAMLinear):
            idx = 0
            while True:
                a_key = f"{module_name}.A_list.{idx}"
                b_key = f"{module_name}.B_list.{idx}"
                if a_key in state_dict and b_key in state_dict:
                    a_shape = state_dict[a_key].shape
                    b_shape = state_dict[b_key].shape
                    
                    # Append appropriately sized parameters to recreate the architecture
                    module.A_list.append(torch.nn.Parameter(torch.empty(a_shape)))
                    module.B_list.append(torch.nn.Parameter(torch.empty(b_shape)))
                    idx += 1
                else:
                    break
                    
    # Analyze state_dict to find how many heads we have in actor_out and critic_out
    head_idx = 1
    while True:
        actor_key = f"actor_out.{head_idx}.weight"
        critic_key = f"critic_out.{head_idx}.weight"
        
        has_actor = actor_key in state_dict
        has_critic = critic_key in state_dict
        
        if not has_actor and not has_critic:
            break
            
        if has_actor:
            out_features, in_features = state_dict[actor_key].shape
            policy.actor_out.append(torch.nn.Linear(in_features, out_features))
            
        if has_critic:
            out_features, in_features = state_dict[critic_key].shape
            policy.critic_out.append(torch.nn.Linear(in_features, out_features))
            
        head_idx += 1
                    
    policy.load_state_dict(state_dict)
    policy.to(device)
    return policy

def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_path = "./HorEvoMoRam/"
    model_path = os.path.join(base_path, "models", "ppo_minigrid_horevomoram_final.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure the training job has completed and saved the final model.")
        return

    stages = [
        {"id": "MiniGrid-Empty-5x5-v0"},
        {"id": "MiniGrid-Empty-Random-6x6-v0"},
        {"id": "MiniGrid-Empty-8x8-v0"},
        {"id": "MiniGrid-Empty-16x16-v0"},
        {"id": "MiniGrid-LavaGapS5-v0"},
        {"id": "MiniGrid-LavaGapS7-v0"},
        {"id": "MiniGrid-Dynamic-Obstacles-5x5-v0"},
        {"id": "MiniGrid-Dynamic-Obstacles-8x8-v0"},
        {"id": "MiniGrid-Dynamic-Obstacles-16x16-v0"},
        {"id": "MiniGrid-FourRooms-v0"}
    ]

    max_obs_dim = 0 
    max_action_dim = 0

    print("Calculating observation and action spaces...")
    for stage in stages:
        temp_env = make_wrapped_env(stage["id"])
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.n
        max_obs_dim = max(max_obs_dim, obs_dim)
        max_action_dim = max(max_action_dim, act_dim)
        temp_env.close()

    print(f"Max Observation Dimension: {max_obs_dim}")
    print(f"Max Action Dimension: {max_action_dim}")

    # Initialize empty policy
    policy = ActorCritic(max_obs_dim, max_action_dim).to(device)
    
    # Reconstruct architecture and load weights
    print(f"Loading weights from {model_path}...")
    policy = reconstruct_and_load(policy, model_path, device)
    
    agent = DummyAgent(policy, device)
    
    print(f"\n{'='*60}")
    print("--- Running Final Evaluation ---")
    print(f"{'='*60}")
    
    results = []
    
    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"Testing on {env_id}...")
        test_env = make_wrapped_env(env_id)
        
        # We pass task_id=None because we modified evaluate_policy to not use it
        mean_reward, std_reward = evaluate_policy(agent, test_env, max_obs_dim, task_id=None, n_eval_episodes=20)
        print(f"Result for {env_id}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        results.append((env_id, mean_reward, std_reward))
        test_env.close()

    print(f"\n{'='*60}")
    print("--- Summary of Results ---")
    print(f"{'='*60}")
    print(f"| {'Stage':<5} | {'Environment':<35} | {'Mean Reward':<12} | {'Std Reward':<10} |")
    print(f"|{'-'*7}|{'-'*37}|{'-'*14}|{'-'*12}|")
    for i, (env_id, mean, std) in enumerate(results):
        print(f"| {i+1:<5} | {env_id:<35} | {mean:<12.2f} | {std:<10.2f} |")

if __name__ == "__main__":
    run_tests()
