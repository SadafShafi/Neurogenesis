import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from minigrid.wrappers import FlatObsWrapper

# -----------------------------------------
# 1. Memory Buffer
# -----------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# -----------------------------------------
# 2. MoRAM Layer Implementation
# -----------------------------------------
class MoRAMLinear(nn.Module):
    """
    Mixture of Rank-1 Associative Memory (MoRAM) Layer
    Acts as a standard linear layer initially, then expands with 
    rank-1 experts (adapters) for continual learning.
    """
    def __init__(self, in_features, out_features, r_per_task=16, k=4, tau=0.01, delta=0.05):
        super(MoRAMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_per_task = r_per_task
        
        # Hyperparameters from the paper
        self.k = k
        self.tau = tau
        self.delta = delta

        # W_0: The base knowledge (Stage 1)
        self.base_layer = nn.Linear(in_features, out_features)

        # MoRAM associative memory pools (Lists of Tensors)
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()

    def add_task(self):
        """Freezes existing weights and adds a new set of rank-1 experts."""
        # Freeze base knowledge and all previously added tasks
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        for param in self.A_list:
            param.requires_grad = False
        for param in self.B_list:
            param.requires_grad = False

        # Initialize new rank-1 keys (A) and values (B)
        # A shape: [r, in_features], B shape: [out_features, r]
        device = self.base_layer.weight.device
        new_A = nn.Parameter(torch.randn(self.r_per_task, self.in_features, device=device) / np.sqrt(self.in_features))
        new_B = nn.Parameter(torch.zeros(self.out_features, self.r_per_task, device=device)) # Zero init to start neutral
        
        self.A_list.append(new_A)
        self.B_list.append(new_B)

    def forward(self, x):
        base_out = self.base_layer(x)
        
        # Support task masking during evaluation
        active_tasks = getattr(self, "active_tasks", None)
        if active_tasks is None:
            active_A_list = list(self.A_list)
            active_B_list = list(self.B_list)
        else:
            active_A_list = list(self.A_list)[:active_tasks]
            active_B_list = list(self.B_list)[:active_tasks]

        # If no adapters are active
        if len(active_A_list) == 0:
            return base_out

        # Concatenate active memory atoms
        A_cat = torch.cat(active_A_list, dim=0) # Shape: [total_r, in_features]
        B_cat = torch.cat(active_B_list, dim=1) # Shape: [out_features, total_r]

        total_r = A_cat.shape[0]

        # 1. Raw activations (Intrinsic Keys evaluating the input)
        raw_act = F.linear(x, A_cat) # Shape: [batch, total_r]

        # 2. Relevance Scoring (L2 Normalization)
        norm_act = raw_act / (torch.norm(raw_act, p=2, dim=-1, keepdim=True) + 1e-8)

        # 3. Sparsity (Top-k masking)
        current_k = min(self.k, total_r)
        topk_vals, topk_indices = torch.topk(norm_act, current_k, dim=-1)
        mask = torch.full_like(norm_act, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)

        # 4. Sharpness Enhancement (Temperature scaled softmax)
        w = torch.softmax(mask / self.tau, dim=-1)

        # 5. Threshold-based expert selection (applied only during inference/evaluation)
        if not self.training:
            threshold_mask = (norm_act >= self.delta).float()
            w = w * threshold_mask

        # 6. Retrieve Values and Apply (w * raw_act) @ B_cat^T
        adapter_out = F.linear(w * raw_act, B_cat)

        return base_out + adapter_out

# -----------------------------------------
# 3. Actor-Critic Network
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # ACTOR NETWORK - Using MoRAM Layers
        self.actor_fc1 = MoRAMLinear(obs_dim, 128, r_per_task=16, k=4)
        self.actor_fc2 = MoRAMLinear(128, 64, r_per_task=16, k=4)
        self.actor_out = MoRAMLinear(64, action_dim, r_per_task=16, k=4)
        
        # CRITIC NETWORK - Using MoRAM Layers
        self.critic_fc1 = MoRAMLinear(obs_dim, 128, r_per_task=16, k=4)
        self.critic_fc2 = MoRAMLinear(128, 64, r_per_task=16, k=4)
        self.critic_out = MoRAMLinear(64, 1, r_per_task=16, k=4)
        
        self.activation = nn.Tanh()

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # Actor
        x_a = self.activation(self.actor_fc1(state))
        x_a = self.activation(self.actor_fc2(x_a))
        action_logits = self.actor_out(x_a)
        action_probs = Categorical(logits=action_logits)
        action = action_probs.sample()
        
        # Critic
        x_c = self.activation(self.critic_fc1(state))
        x_c = self.activation(self.critic_fc2(x_c))
        state_value = self.critic_out(x_c)
        
        return action, action_probs.log_prob(action), state_value.squeeze(-1)

    def evaluate(self, state, action):
        # Actor
        x_a = self.activation(self.actor_fc1(state))
        x_a = self.activation(self.actor_fc2(x_a))
        action_logits = self.actor_out(x_a)
        action_probs = Categorical(logits=action_logits)
        
        # Critic
        x_c = self.activation(self.critic_fc1(state))
        x_c = self.activation(self.critic_fc2(x_c))
        state_value = self.critic_out(x_c)
        
        action_logprobs = action_probs.log_prob(action)
        dist_entropy = action_probs.entropy()
        
        return action_logprobs, state_value.squeeze(-1), dist_entropy

# -----------------------------------------
# 4. PPO Agent
# -----------------------------------------
class PPO:
    def __init__(self, obs_dim, action_dim, lr, gamma, epochs, eps_clip, ent_coef, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.device = device
        self.lr = lr

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def add_new_task(self, new_lr):
        """Triggers MoRAM layer expansion and rebuilds the optimizer."""
        # 1. Freeze output heads (actor_out and critic_out) if they are standard linear layers
        if not isinstance(self.policy.actor_out, MoRAMLinear):
            self.policy.actor_out.weight.requires_grad = False
            if self.policy.actor_out.bias is not None:
                self.policy.actor_out.bias.requires_grad = False
            
        if not isinstance(self.policy.critic_out, MoRAMLinear):
            self.policy.critic_out.weight.requires_grad = False
            if self.policy.critic_out.bias is not None:
                self.policy.critic_out.bias.requires_grad = False

        # 2. Expand network architecture by adding new rank-1 experts
        for module in self.policy.modules():
            if isinstance(module, MoRAMLinear):
                module.add_task()
                
        # 3. Sync the old policy architecture
        import copy
        self.policy_old = copy.deepcopy(self.policy)
        
        # 4. Rebuild the optimizer with only trainable parameters
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=new_lr)
        
        print(f"--> Added MoRAM Task. Optimizing {len(trainable_params)} parameter tensors (rest are frozen).")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.cpu().numpy()

    def update(self, next_state, next_done, batch_size):
        # Calculate next state value for GAE
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            x_c = self.policy_old.activation(self.policy_old.critic_fc1(next_state_t))
            x_c = self.policy_old.activation(self.policy_old.critic_fc2(x_c))
            next_value = self.policy_old.critic_out(x_c).squeeze(-1) # shape: [num_envs]
            
            next_done_t = torch.FloatTensor(next_done).to(self.device)
            next_value = next_value * (1.0 - next_done_t)

        advantages = []
        num_envs = next_value.shape[0]
        gae = torch.zeros(num_envs, device=self.device)
        next_val = next_value
        
        for step in reversed(range(len(self.buffer.rewards))):
            done = self.buffer.is_terminals[step]
            val = self.buffer.state_values[step]
            reward = self.buffer.rewards[step]
            
            mask = 1.0 - done.float()
            delta = reward + self.gamma * next_val * mask - val
            gae = delta + self.gamma * 0.95 * mask * gae  # lambda = 0.95
            advantages.insert(0, gae)
            next_val = val
            
        advantages = torch.stack(advantages, dim=0) # [T, num_envs]
        returns = advantages + torch.stack(self.buffer.state_values, dim=0) # [T, num_envs]
        
        advantages = advantages.flatten().detach()
        returns = returns.flatten().detach()
        
        states_stacked = torch.stack(self.buffer.states, dim=0) # [T, num_envs, obs_dim]
        obs_dim = states_stacked.shape[-1]
        old_states = states_stacked.view(-1, obs_dim).detach()
        
        old_actions = torch.stack(self.buffer.actions, dim=0).flatten().detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).flatten().detach()
        
        dataset_size = old_states.size(0)
        
        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = permutation[start_idx:start_idx + batch_size]
                
                b_states = old_states[batch_indices]
                b_actions = old_actions[batch_indices]
                b_logprobs = old_logprobs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                
                logprobs, state_values, dist_entropy = self.policy.evaluate(b_states, b_actions)
                ratios = torch.exp(logprobs - b_logprobs)
                
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advantages
                
                loss_policy = -torch.min(surr1, surr2).mean()
                loss_value = 0.5 * self.mse_loss(state_values, b_returns)
                loss_entropy = - self.ent_coef * dist_entropy.mean()
                
                loss = loss_policy + loss_value + loss_entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# -----------------------------------------
# Helper Functions
# -----------------------------------------
def pad_state(state, max_obs_dim):
    padded = np.zeros(max_obs_dim)
    padded[:len(state)] = state
    return padded

def pad_state_batch(states, max_obs_dim):
    num_envs, obs_dim = states.shape
    if obs_dim == max_obs_dim:
        return states
    padded = np.zeros((num_envs, max_obs_dim), dtype=states.dtype)
    padded[:, :obs_dim] = states
    return padded


def evaluate_policy(agent, env, max_obs_dim, task_id=None, n_eval_episodes=10):
    agent.policy.eval()
    
    # Configure MoRAM layers for the specific task/stage
    for module in agent.policy.modules():
        if isinstance(module, MoRAMLinear):
            module.active_tasks = task_id

    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            padded_state = pad_state(state, max_obs_dim)
            state_t = torch.FloatTensor(padded_state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                x_a = agent.policy.activation(agent.policy.actor_fc1(state_t))
                x_a = agent.policy.activation(agent.policy.actor_fc2(x_a))
                action_logits = agent.policy.actor_out(x_a)
                action_probs = F.softmax(action_logits, dim=-1)
                action = Categorical(probs=action_probs).sample().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_rewards.append(ep_reward)
        
    # Reset active_tasks back to None for training
    for module in agent.policy.modules():
        if isinstance(module, MoRAMLinear):
            module.active_tasks = None

    agent.policy.train()
    return np.mean(episode_rewards), np.std(episode_rewards)

# -----------------------------------------
# 5. Training Loop
# -----------------------------------------
def train_all_together():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. Path Setup
    base_path = "./MeMoRam/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    stages = [
        {"id": "MiniGrid-Empty-5x5-v0", "timesteps": 40000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
        {"id": "MiniGrid-Empty-8x8-v0", "timesteps": 80000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
        {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},
        {"id": "MiniGrid-Empty-Random-6x6-v0", "timesteps": 60000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01}
    ]

    update_timestep = 2048 
    gamma = 0.99
    epochs = 10
    eps_clip = 0.2

    agent = None
    max_obs_dim = 0 

    # Pre-calculate max observation dimension
    for stage in stages:
        temp_env = FlatObsWrapper(gym.make(stage["id"]))
        obs_dim = temp_env.observation_space.shape[0]
        max_obs_dim = max(max_obs_dim, obs_dim)
        temp_env.close()

    print(f"Max Observation Dimension across stages: {max_obs_dim}")

    results_md_path = os.path.join(base_path, "results.md")
    with open(results_md_path, "w") as f:
        f.write("# Continual Learning Results - MoRAM Implementation\n\n")

    stage_results = []

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"\n--- Starting Stage {i+1}: {env_id} ---")
        
        eval_env = gym.make(env_id)
        eval_env = FlatObsWrapper(eval_env)
        action_dim = eval_env.action_space.n

        # Vectorized environment for training
        num_envs = 8
        def make_env():
            return FlatObsWrapper(gym.make(env_id))
        env = SyncVectorEnv([make_env for _ in range(num_envs)])

        if agent is None:
            # Stage 1: Initialize base PPO agent
            agent = PPO(max_obs_dim, action_dim, stage["lr"], gamma, epochs, eps_clip, stage["ent_coef"], device)
            print(f"Stage 1: Training Base Knowledge (W_0). No Adapters yet.")
        else:
            # Stage 2+: Trigger MoRAM Expansion
            print(f"Transitioning to Stage {i+1}: Adding MoRAM Adapters & Rebuilding Optimizer.")
            agent.ent_coef = stage["ent_coef"]
            agent.add_new_task(stage["lr"])

        time_step = 0
        state, _ = env.reset()
        padded_state = pad_state_batch(state, max_obs_dim)

        current_ep_rewards = np.zeros(num_envs)
        ep_rewards = []

        print(f"Training for {stage['timesteps']} steps...")
        while time_step < stage["timesteps"]:
            action = agent.select_action(padded_state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            
            agent.buffer.rewards.append(torch.FloatTensor(reward).to(agent.device))
            agent.buffer.is_terminals.append(torch.FloatTensor(done).to(agent.device))

            current_ep_rewards += reward
            for env_idx in range(num_envs):
                if done[env_idx]:
                    ep_rewards.append(current_ep_rewards[env_idx])
                    current_ep_rewards[env_idx] = 0.0

            time_step += num_envs

            # Update to next state s_{t+1}
            padded_state = pad_state_batch(state, max_obs_dim)

            if time_step % update_timestep == 0:
                agent.update(padded_state, done, stage["batch_size"])
                avg_ep_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 0 else 0.0
                print(f"  Step {time_step}/{stage['timesteps']} | Recent Ep Reward: {avg_ep_reward:.2f}")

        print(f"Finished {stage['timesteps']} steps for {env_id}.")
        
        # Test on CURRENT stage
        print(f"Evaluating on {env_id}...")
        mean_reward, std_reward = evaluate_policy(agent, eval_env, max_obs_dim, task_id=i, n_eval_episodes=10)
        print(f"Stage {i+1} Mean Reward: {mean_reward:.2f}")
        stage_results.append((env_id, mean_reward, std_reward))

        # Dynamically determine the ranks added and total active ranks
        fc1 = agent.policy.actor_fc1
        total_ranks = len(fc1.A_list) * fc1.r_per_task
        ranks_added = fc1.r_per_task if i > 0 else 0

        with open(results_md_path, "a") as f:
            f.write(f"## Stage {i+1}: {env_id}\n")
            f.write(f"### Performance\n")
            f.write(f"- **Mean Reward:** {mean_reward:.2f} +/- {std_reward:.2f}\n")
            f.write(f"- **Timesteps:** {stage['timesteps']}\n\n")
            f.write(f"### Configuration\n")
            f.write(f"- **Learning Rate:** {stage['lr']}\n")
            f.write(f"- **Batch Size:** {stage['batch_size']}\n")
            f.write(f"- **Ent Coef:** {stage['ent_coef']}\n")
            f.write(f"- **MoRAM Ranks Added (This Stage):** {ranks_added}\n")
            f.write(f"- **Total Active MoRAM Ranks:** {total_ranks}\n\n")

        env.close()
        eval_env.close()

    # Final Evaluation on ALL stages
    print("\n--- Final Evaluation on ALL Stages ---")
    with open(results_md_path, "a") as f:
        f.write("## Final Evaluation (Catastrophic Forgetting Check)\n")
        f.write("| Stage | Environment | Mean Reward | Std Reward |\n")
        f.write("|-------|-------------|-------------|------------|\n")

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"Final testing on {env_id}...")
        test_env = gym.make(env_id)
        test_env = FlatObsWrapper(test_env)
        
        mean_reward, std_reward = evaluate_policy(agent, test_env, max_obs_dim, task_id=i, n_eval_episodes=10)
        print(f"Final Reward on {env_id}: {mean_reward:.2f}")
        
        with open(results_md_path, "a") as f:
            f.write(f"| {i+1} | {env_id} | {mean_reward:.2f} | {std_reward:.2f} |\n")
        
        test_env.close()

    # Save final model
    final_model_path = os.path.join(model_path, "ppo_minigrid_moram_final.pt")
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")
    print(f"Results summary saved to {results_md_path}")

if __name__ == "__main__":
    train_all_together()