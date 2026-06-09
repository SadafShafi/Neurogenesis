import os
import copy
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
# 2. Actor-Critic Network (Plain MLP — No Adapters)
# -----------------------------------------
class ActorCritic(nn.Module):
    """
    Same architecture as EvoMoRAM's ActorCritic but using plain nn.Linear
    layers instead of EvoMoRAMLinear.  No adapters, no rank expansion.
    
    Architecture matches the "Smaller Network 8x4" variant:
        Actor:  obs_dim → 4 → 2 → action_dim
        Critic: obs_dim → 4 → 2 → 1
    """
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        print('Ablation: Plain MLP (no adapters) — SMaller Network 8x4')
        # ACTOR NETWORK — Plain Linear Layers (same sizes as EvoMoRAM)
        self.actor_fc1 = nn.Linear(obs_dim, 4)
        self.actor_fc2 = nn.Linear(4, 2)
        self.actor_out = nn.Linear(2, action_dim)

        # CRITIC NETWORK — Plain Linear Layers (same sizes as EvoMoRAM)
        self.critic_fc1 = nn.Linear(obs_dim, 4)
        self.critic_fc2 = nn.Linear(4, 2)
        self.critic_out = nn.Linear(2, 1)

        self.activation = nn.Tanh()

    def forward(self):
        raise NotImplementedError

    def _apply_action_mask(self, action_logits):
        if hasattr(self, 'current_action_dim') and self.current_action_dim is not None:
            mask = torch.ones_like(action_logits, dtype=torch.bool)
            mask[..., :self.current_action_dim] = False
            action_logits = action_logits.masked_fill(mask, -1e10)
        return action_logits

    def act(self, state):
        # Actor
        x_a = self.activation(self.actor_fc1(state))
        x_a = self.activation(self.actor_fc2(x_a))
        action_logits = self.actor_out(x_a)
        action_logits = self._apply_action_mask(action_logits)
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
        action_logits = self._apply_action_mask(action_logits)
        action_probs = Categorical(logits=action_logits)

        # Critic
        x_c = self.activation(self.critic_fc1(state))
        x_c = self.activation(self.critic_fc2(x_c))
        state_value = self.critic_out(x_c)

        action_logprobs = action_probs.log_prob(action)
        dist_entropy = action_probs.entropy()

        return action_logprobs, state_value.squeeze(-1), dist_entropy

# -----------------------------------------
# 3. PPO Agent (Single-Task, No Adapter Logic)
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
            next_value = self.policy_old.critic_out(x_c).squeeze(-1)

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

        advantages = torch.stack(advantages, dim=0)
        returns = advantages + torch.stack(self.buffer.state_values, dim=0)

        advantages = advantages.flatten().detach()
        returns = returns.flatten().detach()

        states_stacked = torch.stack(self.buffer.states, dim=0)
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
                loss_entropy = -self.ent_coef * dist_entropy.mean()

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


def evaluate_policy(agent, env, max_obs_dim, n_eval_episodes=10):
    """Evaluate a single-task agent on its environment."""
    agent.policy.eval()

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
                if hasattr(agent.policy, 'current_action_dim') and agent.policy.current_action_dim is not None:
                    mask = torch.ones_like(action_logits, dtype=torch.bool)
                    mask[..., :agent.policy.current_action_dim] = False
                    action_logits = action_logits.masked_fill(mask, -1e10)
                action_probs = F.softmax(action_logits, dim=-1)
                action = Categorical(probs=action_probs).sample().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_rewards.append(ep_reward)

    agent.policy.train()
    return np.mean(episode_rewards), np.std(episode_rewards)

# -----------------------------------------
# 4. Training Loop — Separate Model Per Problem
# -----------------------------------------
def make_wrapped_env(env_id):
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        from minigrid.wrappers import FlatObsWrapper
        env = FlatObsWrapper(env)
    return env

def train_ablation():
    """
    ABLATION: Train a separate, independent model for each problem.
    
    No adapters, no continual learning, no neurogenesis.
    Each problem gets a fresh ActorCritic with plain nn.Linear layers,
    trained from scratch, using the exact same hyperparameters as EvoMoRAM.
    
    This establishes the single-task baseline to compare against.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. Path Setup
    base_path = "./evoMoRam_ablation/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Same problems and hyperparameters as EvoMoRAM
    stages = [
    # --- Empty Environments (Simplest baselines) ---
    {"id": "MiniGrid-Empty-5x5-v0", "timesteps": 40000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    {"id": "MiniGrid-Empty-Random-6x6-v0", "timesteps": 60000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    {"id": "MiniGrid-Empty-8x8-v0", "timesteps": 80000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},

    # --- Lava Gap (Static, discoverable hazards) ---
    {"id": "MiniGrid-LavaGapS5-v0", "timesteps": 150000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.02},
    {"id": "MiniGrid-LavaGapS7-v0", "timesteps": 250000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.02},

    # --- Dynamic Obstacles (Pure reactive avoidance) ---
    {"id": "MiniGrid-Dynamic-Obstacles-5x5-v0", "timesteps": 200000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.03},
    {"id": "MiniGrid-Dynamic-Obstacles-8x8-v0", "timesteps": 500000, "lr": 3e-4, "batch_size": 256, "ent_coef": 0.03},
    {"id": "MiniGrid-Dynamic-Obstacles-16x16-v0", "timesteps": 1000000, "lr": 5e-4, "batch_size": 256, "ent_coef": 0.05},

    # --- Four Rooms (Requires more systematic exploration heuristic) ---
    {"id": "MiniGrid-FourRooms-v0", "timesteps": 500000, "lr": 3e-4, "batch_size": 256, "ent_coef": 0.05}
    ]

    update_timestep = 2048
    gamma = 0.99
    epochs = 10
    eps_clip = 0.2

    # Pre-calculate max observation dimension (for padding consistency with EvoMoRAM)
    max_obs_dim = 0
    max_action_dim = 0
    for stage in stages:
        temp_env = make_wrapped_env(stage["id"])
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.n
        max_obs_dim = max(max_obs_dim, obs_dim)
        max_action_dim = max(max_action_dim, act_dim)
        temp_env.close()

    print(f"Max Observation Dimension across stages: {max_obs_dim}")
    print(f"Max Action Dimension across stages: {max_action_dim}")

    results_md_path = os.path.join(base_path, "results.md")
    with open(results_md_path, "w") as f:
        f.write("# Ablation Results — Plain MLP (No Adapters, Separate Model Per Problem)\n\n")
        f.write("Each problem is trained with a **fresh, independent model** from scratch.\n")
        f.write("Same architecture as EvoMoRAM base network (obs→4→2→action) but using plain nn.Linear.\n")
        f.write("No adapters, no rank expansion, no neurogenesis.\n\n")

    all_results = []

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"\n{'='*60}")
        print(f"--- Problem {i+1}/{len(stages)}: {env_id} ---")
        print(f"--- Training FRESH model from scratch (no adapters) ---")
        print(f"{'='*60}")

        eval_env = make_wrapped_env(env_id)
        action_dim = eval_env.action_space.n

        # Vectorized environment for training
        num_envs = 8
        def make_env():
            return make_wrapped_env(env_id)
        env = SyncVectorEnv([make_env for _ in range(num_envs)])

        # Create a FRESH agent for this problem
        agent = PPO(max_obs_dim, max_action_dim, stage["lr"], gamma, epochs, eps_clip, stage["ent_coef"], device)
        agent.policy.current_action_dim = action_dim
        agent.policy_old.current_action_dim = action_dim

        # Count parameters
        total_params = sum(p.numel() for p in agent.policy.parameters())
        trainable_params = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params} total, {trainable_params} trainable")

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

            # Update to next state
            padded_state = pad_state_batch(state, max_obs_dim)

            if time_step % update_timestep == 0:
                agent.update(padded_state, done, stage["batch_size"])
                avg_ep_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 0 else 0.0
                print(f"  Step {time_step}/{stage['timesteps']} | "
                      f"Recent Ep Reward: {avg_ep_reward:.2f}")

        print(f"Finished {stage['timesteps']} steps for {env_id}.")

        # Evaluate
        print(f"Evaluating on {env_id}...")
        mean_reward, std_reward = evaluate_policy(agent, eval_env, max_obs_dim, n_eval_episodes=10)
        print(f"Problem {i+1} ({env_id}) Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        all_results.append((env_id, mean_reward, std_reward, total_params, trainable_params))

        # Log results
        with open(results_md_path, "a") as f:
            f.write(f"## Problem {i+1}: {env_id}\n")
            f.write(f"### Performance\n")
            f.write(f"- **Mean Reward:** {mean_reward:.2f} +/- {std_reward:.2f}\n")
            f.write(f"- **Timesteps:** {stage['timesteps']}\n\n")
            f.write(f"### Configuration\n")
            f.write(f"- **Learning Rate:** {stage['lr']}\n")
            f.write(f"- **Batch Size:** {stage['batch_size']}\n")
            f.write(f"- **Ent Coef:** {stage['ent_coef']}\n")
            f.write(f"- **Model:** Fresh plain MLP (no adapters)\n")
            f.write(f"- **Total Parameters:** {total_params}\n")
            f.write(f"- **Trainable Parameters:** {trainable_params}\n\n")

        # Save model for this problem
        model_save_path = os.path.join(model_path, f"ppo_ablation_{env_id.replace('/', '_')}.pt")
        torch.save(agent.policy.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        env.close()
        eval_env.close()

    # Final Summary Table
    print(f"\n{'='*60}")
    print("--- ABLATION SUMMARY: All Problems ---")
    print(f"{'='*60}")

    with open(results_md_path, "a") as f:
        f.write("## Summary: All Problems\n")
        f.write("| # | Environment | Mean Reward | Std Reward | Parameters |\n")
        f.write("|---|-------------|-------------|------------|------------|\n")

    for idx, (env_id, mean_r, std_r, total_p, train_p) in enumerate(all_results):
        print(f"  Problem {idx+1}: {env_id} | Reward: {mean_r:.2f} +/- {std_r:.2f} | Params: {total_p}")
        with open(results_md_path, "a") as f:
            f.write(f"| {idx+1} | {env_id} | {mean_r:.2f} | {std_r:.2f} | {total_p} |\n")

    print(f"\nResults saved to {results_md_path}")


if __name__ == "__main__":
    train_ablation()
