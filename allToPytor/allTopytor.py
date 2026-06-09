import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
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
# 2. Actor-Critic Network
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # ACTOR NETWORK
        # Structured explicitly so we can later replace these with MoRAM layers
        self.actor_fc1 = nn.Linear(obs_dim, 128)
        self.actor_fc2 = nn.Linear(128, 64)
        self.actor_out = nn.Linear(64, action_dim)
        
        # CRITIC NETWORK
        self.critic_fc1 = nn.Linear(obs_dim, 128)
        self.critic_fc2 = nn.Linear(128, 64)
        self.critic_out = nn.Linear(64, 1)
        
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
# 3. PPO Agent
# -----------------------------------------
class PPO:
    def __init__(self, obs_dim, action_dim, lr, gamma, epochs, eps_clip, ent_coef, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.device = device

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action.item())
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def update(self, next_state, next_done, batch_size):
        # Calculate next state value for GAE
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            x_c = self.policy_old.activation(self.policy_old.critic_fc1(next_state_t))
            x_c = self.policy_old.activation(self.policy_old.critic_fc2(x_c))
            next_value = self.policy_old.critic_out(x_c).squeeze(-1).item()
            if next_done:
                next_value = 0.0

        # Calculate generalized advantages (GAE) and returns
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        state_values = [v.item() for v in self.buffer.state_values]
        
        advantages = []
        gae = 0.0
        next_val = next_value
        
        for step in reversed(range(len(rewards))):
            done = is_terminals[step]
            val = state_values[step]
            reward = rewards[step]
            
            mask = 1.0 - float(done)
            delta = reward + self.gamma * next_val * mask - val
            gae = delta + self.gamma * 0.95 * mask * gae  # lambda = 0.95
            advantages.insert(0, gae)
            next_val = val
            
        returns = [adv + val for adv, val in zip(advantages, state_values)]
        
        # Convert to tensors
        old_states = torch.stack(self.buffer.states, dim=0).detach() # shape: (N, max_obs_dim)
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device).detach() # shape: (N,)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach() # shape: (N,)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device).detach() # shape: (N,)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).detach() # shape: (N,)
        
        # Optimize policy for K epochs using mini-batches
        dataset_size = old_states.size(0)
        
        for _ in range(self.epochs):
            # Shuffle indices
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = permutation[start_idx:start_idx + batch_size]
                
                b_states = old_states[batch_indices]
                b_actions = old_actions[batch_indices]
                b_logprobs = old_logprobs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                
                # Normalize advantages batch-wise
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                
                # Evaluate
                logprobs, state_values, dist_entropy = self.policy.evaluate(b_states, b_actions)
                ratios = torch.exp(logprobs - b_logprobs)
                
                # Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advantages
                
                loss_policy = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                loss_value = 0.5 * self.mse_loss(state_values, b_returns)
                
                # Entropy Loss
                loss_entropy = - self.ent_coef * dist_entropy.mean()
                
                loss = loss_policy + loss_value + loss_entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# -----------------------------------------
# Helper Functions
# -----------------------------------------
def pad_state(state, max_obs_dim):
    padded = np.zeros(max_obs_dim)
    padded[:len(state)] = state
    return padded

def evaluate_policy(agent, env, max_obs_dim, n_eval_episodes=10):
    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            padded_state = pad_state(state, max_obs_dim)
            state_t = torch.FloatTensor(padded_state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                # select deterministic action
                x_a = agent.policy.activation(agent.policy.actor_fc1(state_t))
                x_a = agent.policy.activation(agent.policy.actor_fc2(x_a))
                action_logits = agent.policy.actor_out(x_a)
                action = torch.argmax(action_logits, dim=-1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_rewards.append(ep_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

# -----------------------------------------
# 4. Training Loop
# -----------------------------------------
def train_all_together():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. Path Setup
    base_path = "./allToPytor/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    stages = [
        {"id": "MiniGrid-Empty-5x5-v0", "timesteps": 40000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.0},
        {"id": "MiniGrid-Empty-8x8-v0", "timesteps": 80000, "lr": 1e-3, "batch_size": 64, "ent_coef": 0.01},
        {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},
        {"id": "MiniGrid-Empty-Random-6x6-v0", "timesteps": 60000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01}
    ]

    # Global agent parameters
    update_timestep = 2048 
    gamma = 0.99
    epochs = 10
    eps_clip = 0.2

    agent = None
    max_obs_dim = 0  # To ensure network handles largest grid size

    # Pre-calculate max observation dimension to fix network size
    for stage in stages:
        temp_env = FlatObsWrapper(gym.make(stage["id"]))
        obs_dim = temp_env.observation_space.shape[0]
        max_obs_dim = max(max_obs_dim, obs_dim)
        temp_env.close()

    print(f"Max Observation Dimension across stages: {max_obs_dim}")

    results_md_path = os.path.join(base_path, "results.md")
    with open(results_md_path, "w") as f:
        f.write("# Continual Learning Results - allToPytor\n\n")

    stage_results = []

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"\n--- Starting Stage {i+1}: {env_id} ---")
        
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        action_dim = env.action_space.n

        if agent is None:
            # Initialize agent on first stage
            agent = PPO(max_obs_dim, action_dim, stage["lr"], gamma, epochs, eps_clip, stage["ent_coef"], device)
        else:
            # Update optimizer LR and entropy coefficient for new stage
            print(f"Transitioning to Stage {i+1}: Applying new hyperparameters (LR: {stage['lr']}, Batch: {stage['batch_size']}, Ent: {stage['ent_coef']})")
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = stage["lr"]
            agent.ent_coef = stage["ent_coef"]

        time_step = 0
        state, _ = env.reset()
        padded_state = pad_state(state, max_obs_dim)

        ep_reward = 0.0
        ep_rewards = []

        print(f"Training for {stage['timesteps']} steps...")
        while time_step < stage["timesteps"]:
            action = agent.select_action(padded_state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            ep_reward += reward
            time_step += 1

            if time_step % update_timestep == 0:
                agent.update(padded_state, done, stage["batch_size"])
                avg_ep_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 0 else 0.0
                print(f"  Step {time_step}/{stage['timesteps']} | Recent Ep Reward: {avg_ep_reward:.2f}")

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()

            padded_state = pad_state(state, max_obs_dim)

        print(f"Finished {stage['timesteps']} steps for {env_id}.")
        
        # Test on CURRENT stage
        print(f"Evaluating on {env_id}...")
        mean_reward, std_reward = evaluate_policy(agent, env, max_obs_dim, n_eval_episodes=10)
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
            f.write(f"- **Architecture:** {{'pi': [128, 64], 'vf': [128, 64]}}\n\n")

        env.close()

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
        
        mean_reward, std_reward = evaluate_policy(agent, test_env, max_obs_dim, n_eval_episodes=10)
        print(f"Final Reward on {env_id}: {mean_reward:.2f}")
        
        with open(results_md_path, "a") as f:
            f.write(f"| {i+1} | {env_id} | {mean_reward:.2f} | {std_reward:.2f} |\n")
        
        test_env.close()

    # Save final model
    final_model_path = os.path.join(model_path, "ppo_minigrid_allToPytor_final.pt")
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")
    print(f"Results summary saved to {results_md_path}")

if __name__ == "__main__":
    train_all_together()