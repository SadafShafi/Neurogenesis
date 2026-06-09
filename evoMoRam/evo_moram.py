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
# 2. Stress Monitor — Biological Stress Signal
# -----------------------------------------
class StressMonitor:
    """
    Tracks learning health and signals when the model is struggling.
    
    Biological analogy: When we attempt a new skill with existing neural
    pathways and repeatedly fail, stress hormones (cortisol) accumulate.
    Chronic stress triggers neurogenesis — the growth of new neural
    pathways to accommodate the new skill.
    
    Stress accumulates when:
      - Reward is BELOW success_threshold AND improvement stagnates
        for `patience` consecutive update cycles
    Stress does NOT accumulate when:
      - Reward is already above success_threshold (task is mastered)
    Stress decreases when:
      - The model shows meaningful reward improvement (relief)
      - The model reaches the success_threshold (mastery)
    Neurogenesis fires when:
      - stress_level >= stress_threshold
    """
    def __init__(self, patience=5, min_improvement=0.01, stress_threshold=3, success_threshold=0.97):
        self.patience = patience              # stagnant cycles before stress++
        self.min_improvement = min_improvement # min reward delta for "improvement"
        self.stress_threshold = stress_threshold # stress level to trigger neurogenesis
        self.success_threshold = success_threshold # reward above this = task mastered, no stress
        
        self.stress_level = 0
        self.stagnation_counter = 0
        self.best_reward = -float('inf')
        self.reward_history = []
    
    def update(self, avg_reward):
        """Update stress state with latest average reward. Returns current stress level."""
        self.reward_history.append(avg_reward)
        
        # If the model is already performing well, no stress — task is mastered
        if avg_reward >= self.success_threshold:
            self.stagnation_counter = 0
            self.stress_level = max(0, self.stress_level - 1)  # relief
            self.best_reward = max(self.best_reward, avg_reward)
            return self.stress_level
        
        if avg_reward > self.best_reward + self.min_improvement:
            # Meaningful improvement — relief
            self.best_reward = avg_reward
            self.stagnation_counter = 0
            self.stress_level = max(0, self.stress_level - 1)
        else:
            # Stagnation while performing poorly — stress builds
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.patience:
                self.stress_level += 1
                self.stagnation_counter = 0  # reset counter, keep stress
        
        return self.stress_level
    
    def is_stressed(self):
        """Returns True when stress has exceeded the neurogenesis threshold."""
        return self.stress_level >= self.stress_threshold
    
    def reset(self):
        """Reset after a neurogenesis event."""
        self.stress_level = 0
        self.stagnation_counter = 0
        self.best_reward = -float('inf')
        self.reward_history = []

# -----------------------------------------
# 3. EvoMoRAM Layer — Adaptive Rank Expansion
# -----------------------------------------
class EvoMoRAMLinear(nn.Module):
    """
    Evolutionary Mixture of Rank-1 Associative Memory (EvoMoRAM) Layer.
    
    Extends MoRAM with stress-triggered neurogenesis:
      1. On a new task, a minimal "probe" adapter (r_probe=2) is added first
      2. If stress is detected, neurogenesis expands the probe to full rank (r_full=16)
      3. The probe's learned weights are PRESERVED and augmented with new capacity
    """
    def __init__(self, in_features, out_features, r_probe=2, r_full=16, k=4, tau=0.01, delta=0.05):
        super(EvoMoRAMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_probe = r_probe
        self.r_full = r_full
        
        # Hyperparameters
        self.k = k
        self.tau = tau
        self.delta = delta

        # W_0: The base knowledge (Stage 1)
        self.base_layer = nn.Linear(in_features, out_features)

        # EvoMoRAM associative memory pools
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        
        # Track adapter state per task: 'probe' or 'full'
        self.adapter_states = []  # list of strings: 'probe' or 'full'

    def _freeze_existing(self):
        """Freeze base knowledge and all previously added adapters."""
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        for param in self.A_list:
            param.requires_grad = False
        for param in self.B_list:
            param.requires_grad = False

    def add_probe(self):
        """
        Add a minimal probe adapter to test if existing knowledge transfers.
        The probe has low rank (r_probe) to minimize added capacity.
        """
        self._freeze_existing()
        
        device = self.base_layer.weight.device
        new_A = nn.Parameter(
            torch.randn(self.r_probe, self.in_features, device=device) / np.sqrt(self.in_features)
        )
        new_B = nn.Parameter(
            torch.zeros(self.out_features, self.r_probe, device=device)
        )
        
        self.A_list.append(new_A)
        self.B_list.append(new_B)
        self.adapter_states.append('probe')

    def trigger_neurogenesis(self):
        """
        Expand the current probe adapter to full rank.
        
        The probe's learned r_probe weights are PRESERVED.
        New r_full - r_probe additional rank-1 experts are appended.
        """
        if len(self.adapter_states) == 0 or self.adapter_states[-1] != 'probe':
            raise RuntimeError("Cannot trigger neurogenesis: no active probe adapter found.")
        
        device = self.base_layer.weight.device
        task_idx = len(self.A_list) - 1
        
        # Current probe weights (preserve these)
        old_A = self.A_list[task_idx]  # [r_probe, in_features]
        old_B = self.B_list[task_idx]  # [out_features, r_probe]
        
        r_expand = self.r_full - self.r_probe
        
        # New capacity to add
        expand_A = torch.randn(r_expand, self.in_features, device=device) / np.sqrt(self.in_features)
        expand_B = torch.zeros(self.out_features, r_expand, device=device)
        
        # Concatenate: preserve probe + add new capacity
        new_A = nn.Parameter(torch.cat([old_A.data, expand_A], dim=0))  # [r_full, in_features]
        new_B = nn.Parameter(torch.cat([old_B.data, expand_B], dim=1))  # [out_features, r_full]
        
        self.A_list[task_idx] = new_A
        self.B_list[task_idx] = new_B
        self.adapter_states[-1] = 'full'

    def add_task(self):
        """Legacy method: directly add a full-rank adapter (same as original MoRAM)."""
        self._freeze_existing()
        
        device = self.base_layer.weight.device
        new_A = nn.Parameter(
            torch.randn(self.r_full, self.in_features, device=device) / np.sqrt(self.in_features)
        )
        new_B = nn.Parameter(
            torch.zeros(self.out_features, self.r_full, device=device)
        )
        
        self.A_list.append(new_A)
        self.B_list.append(new_B)
        self.adapter_states.append('full')

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

        if len(active_A_list) == 0:
            return base_out

        # Concatenate active memory atoms
        A_cat = torch.cat(active_A_list, dim=0)  # [total_r, in_features]
        B_cat = torch.cat(active_B_list, dim=1)  # [out_features, total_r]

        total_r = A_cat.shape[0]

        # 1. Raw activations
        raw_act = F.linear(x, A_cat)  # [batch, total_r]

        # 2. Relevance Scoring (L2 Normalization)
        norm_act = raw_act / (torch.norm(raw_act, p=2, dim=-1, keepdim=True) + 1e-8)

        # 3. Sparsity (Top-k masking)
        current_k = min(self.k, total_r)
        topk_vals, topk_indices = torch.topk(norm_act, current_k, dim=-1)
        mask = torch.full_like(norm_act, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)

        # 4. Sharpness Enhancement (Temperature scaled softmax)
        w = torch.softmax(mask / self.tau, dim=-1)

        # 5. Threshold-based expert selection (inference only)
        if not self.training:
            threshold_mask = (norm_act >= self.delta).float()
            w = w * threshold_mask

        # 6. Retrieve values
        adapter_out = F.linear(w * raw_act, B_cat)

        return base_out + adapter_out

# -----------------------------------------
# 4. Actor-Critic Network
# -----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # # ACTOR NETWORK — Using EvoMoRAM Layers
        # self.actor_fc1 = EvoMoRAMLinear(obs_dim, 128, r_probe=2, r_full=16, k=4)
        # self.actor_fc2 = EvoMoRAMLinear(128, 64, r_probe=2, r_full=16, k=4)
        # self.actor_out = EvoMoRAMLinear(64, action_dim, r_probe=2, r_full=16, k=4)
        
        # # CRITIC NETWORK — Using EvoMoRAM Layers
        # self.critic_fc1 = EvoMoRAMLinear(obs_dim, 128, r_probe=2, r_full=16, k=4)
        # self.critic_fc2 = EvoMoRAMLinear(128, 64, r_probe=2, r_full=16, k=4)
        # self.critic_out = EvoMoRAMLinear(64, 1, r_probe=2, r_full=16, k=4)
        
        # self.activation = nn.Tanh()

        print('SMaller Network 8x4')
        # ACTOR NETWORK — Using EvoMoRAM Layers
        self.actor_fc1 = EvoMoRAMLinear(obs_dim, 4, r_probe=2, r_full=16, k=4)
        self.actor_fc2 = EvoMoRAMLinear(4, 2, r_probe=2, r_full=16, k=4)
        self.actor_out = EvoMoRAMLinear(2, action_dim, r_probe=2, r_full=16, k=4)
        
        # CRITIC NETWORK — Using EvoMoRAM Layers
        self.critic_fc1 = EvoMoRAMLinear(obs_dim, 4, r_probe=2, r_full=16, k=4)
        self.critic_fc2 = EvoMoRAMLinear(4, 2, r_probe=2, r_full=16, k=4)
        self.critic_out = EvoMoRAMLinear(2, 1, r_probe=2, r_full=16, k=4)
        
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
# 5. Evolutionary PPO Agent
# -----------------------------------------
class EvoPPO:
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

    def _rebuild_optimizer(self, lr):
        """Rebuild optimizer with only trainable parameters."""
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=lr)
        return len(trainable_params)

    def attempt_new_task(self, new_lr):
        """
        ATTEMPT PHASE: Add minimal probe adapters and try to learn.
        
        The model first tries to solve the new task with minimal architecture
        change (r=2 probe), preserving as much of the existing structure as
        possible. Only if this fails will neurogenesis be triggered.
        """
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.add_probe()
        
        # Sync old policy
        self.policy_old = copy.deepcopy(self.policy)
        
        # Rebuild optimizer with only trainable params (the probes)
        n_params = self._rebuild_optimizer(new_lr)
        print(f"  --> ATTEMPT PHASE: Added probe adapters (r=2). "
              f"Optimizing {n_params} parameter tensors.")

    def trigger_neurogenesis(self, new_lr):
        """
        NEUROGENESIS: Expand probe adapters to full rank.
        
        Called when the StressMonitor detects learning stagnation.
        Probe weights are preserved and augmented with new capacity.
        """
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.trigger_neurogenesis()
        
        # Sync old policy
        self.policy_old = copy.deepcopy(self.policy)
        
        # Rebuild optimizer with expanded parameters
        n_params = self._rebuild_optimizer(new_lr)
        print(f"  --> 🧠 NEUROGENESIS: Expanded to full adapters (r=16). "
              f"Optimizing {n_params} parameter tensors.")

    def add_new_task(self, new_lr):
        """Legacy: directly add full-rank adapters (same as original MoRAM)."""
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.add_task()
        
        self.policy_old = copy.deepcopy(self.policy)
        n_params = self._rebuild_optimizer(new_lr)
        print(f"  --> Added full MoRAM task. Optimizing {n_params} parameter tensors.")

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


def evaluate_policy(agent, env, max_obs_dim, task_id=None, n_eval_episodes=10):
    agent.policy.eval()
    
    # Configure EvoMoRAM layers for the specific task/stage
    for module in agent.policy.modules():
        if isinstance(module, EvoMoRAMLinear):
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
        
    # Reset active_tasks back to None for training
    for module in agent.policy.modules():
        if isinstance(module, EvoMoRAMLinear):
            module.active_tasks = None

    agent.policy.train()
    return np.mean(episode_rewards), np.std(episode_rewards)

# -----------------------------------------
# 6. Training Loop — Stress-Triggered Neurogenesis
# -----------------------------------------
def make_wrapped_env(env_id):
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        from minigrid.wrappers import FlatObsWrapper
        env = FlatObsWrapper(env)
    return env

def train_all_together():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. Path Setup
    base_path = "./evoMoRam/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # stages = [
    #     {"id": "MiniGrid-Empty-5x5-v0", "timesteps": 40000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    #     {"id": "MiniGrid-Empty-8x8-v0", "timesteps": 80000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    #     {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},
    #     {"id": "MiniGrid-Empty-Random-6x6-v0", "timesteps": 60000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01}
    # ]

    # stages = [
    # # --- PHASE 1: Spatial Navigation (Your existing tasks) ---
    # {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},
    # {"id": "MiniGrid-Empty-5x5-v0", "timesteps": 40000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    # {"id": "MiniGrid-Empty-8x8-v0", "timesteps": 80000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    # # {"id": "MiniGrid-Empty-16x16-v0", "timesteps": 300000, "lr": 5e-4, "batch_size": 128, "ent_coef": 0.05},
    # {"id": "MiniGrid-Empty-Random-6x6-v0", "timesteps": 60000, "lr": 3e-4, "batch_size": 64, "ent_coef": 0.01},
    
    # # --- PHASE 2: Object Interaction & Emergent Behaviors ---
    # # Will likely trigger neurogenesis due to needing "pickup" and "toggle" actions
    # {"id": "MiniGrid-DoorKey-6x6-v0", "timesteps": 250000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.05},
    
    # # --- PHASE 3: Risk & Dynamic Adaptation ---
    # # Tests value function stability under sudden failure states
    # {"id": "MiniGrid-LavaGapS5-v0", "timesteps": 150000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.02},
    # {"id": "MiniGrid-Dynamic-Obstacles-8x8-v0", "timesteps": 300000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.05},
    
    # # --- PHASE 4: Extreme Domain Shift ---
    # # Shifts from spatial/categorical arrays to floating-point kinematics
    # {"id": "LunarLander-v3", "timesteps": 500000, "lr": 3e-4, "batch_size": 128, "ent_coef": 0.01}
    # ]

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

    # Stress monitor hyperparameters
    stress_patience = 5        # stagnant update cycles before stress++
    stress_min_improvement = 0.01  # minimum reward improvement
    stress_threshold = 3       # stress level triggering neurogenesis
    success_threshold = 0.97   # reward above this = task mastered, no stress

    update_timestep = 2048 
    gamma = 0.99
    epochs = 10
    eps_clip = 0.2

    agent = None
    max_obs_dim = 0 
    max_action_dim = 0

    # Pre-calculate max observation dimension
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
        f.write("# Continual Learning Results — EvoMoRAM (Stress-Triggered Neurogenesis)\n\n")

    stage_results = []

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"\n{'='*60}")
        print(f"--- Starting Stage {i+1}: {env_id} ---")
        print(f"{'='*60}")
        
        eval_env = make_wrapped_env(env_id)
        action_dim = eval_env.action_space.n

        # Vectorized environment for training
        num_envs = 8
        def make_env():
            return make_wrapped_env(env_id)
        env = SyncVectorEnv([make_env for _ in range(num_envs)])

        neurogenesis_triggered = False
        neurogenesis_timestep = None
        stress_history = []

        if agent is None:
            # ========================================
            # STAGE 1: Train base model (W_0)
            # ========================================
            agent = EvoPPO(max_obs_dim, max_action_dim, stage["lr"], gamma, epochs, eps_clip, stage["ent_coef"], device)
            print(f"Stage 1: Training Base Knowledge (W_0). No adapters yet.")
            stress_monitor = None  # No stress monitoring for Stage 1
        else:
            # ========================================
            # STAGE 2+: ATTEMPT PHASE — Try with probe first
            # ========================================
            print(f"Transitioning to Stage {i+1}: ATTEMPT PHASE (probe adapters).")
            agent.ent_coef = stage["ent_coef"]
            agent.attempt_new_task(stage["lr"])
            
            # Initialize stress monitor for this stage
            stress_monitor = StressMonitor(
                patience=stress_patience,
                min_improvement=stress_min_improvement,
                stress_threshold=stress_threshold,
                success_threshold=success_threshold
            )

        agent.policy.current_action_dim = action_dim
        agent.policy_old.current_action_dim = action_dim

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
                
                # --- Stress Monitoring (Stage 2+ only) ---
                if stress_monitor is not None:
                    current_stress = stress_monitor.update(avg_ep_reward)
                    stress_history.append({
                        "timestep": time_step,
                        "avg_reward": avg_ep_reward,
                        "stress_level": current_stress
                    })
                    
                    stress_indicator = "🟢" if current_stress == 0 else "🟡" if current_stress < stress_threshold else "🔴"
                    adapter_state = "PROBE" if not neurogenesis_triggered else "FULL"
                    print(f"  Step {time_step}/{stage['timesteps']} | "
                          f"Reward: {avg_ep_reward:.2f} | "
                          f"Stress: {current_stress}/{stress_threshold} {stress_indicator} | "
                          f"Adapters: {adapter_state}")
                    
                    # Check for neurogenesis trigger
                    if stress_monitor.is_stressed() and not neurogenesis_triggered:
                        print(f"\n  {'!'*50}")
                        print(f"  🧠 STRESS DETECTED at step {time_step}!")
                        print(f"  Triggering NEUROGENESIS — expanding adapters...")
                        print(f"  {'!'*50}\n")
                        
                        agent.trigger_neurogenesis(stage["lr"])
                        neurogenesis_triggered = True
                        neurogenesis_timestep = time_step
                        stress_monitor.reset()
                else:
                    # Stage 1: simple logging
                    print(f"  Step {time_step}/{stage['timesteps']} | "
                          f"Recent Ep Reward: {avg_ep_reward:.2f}")

        print(f"Finished {stage['timesteps']} steps for {env_id}.")
        
        # Test on CURRENT stage
        print(f"Evaluating on {env_id}...")
        mean_reward, std_reward = evaluate_policy(agent, eval_env, max_obs_dim, task_id=i, n_eval_episodes=10)
        print(f"Stage {i+1} Mean Reward: {mean_reward:.2f}")
        stage_results.append((env_id, mean_reward, std_reward))

        # Determine adapter info for logging
        fc1 = agent.policy.actor_fc1
        total_ranks = sum(
            a.shape[0] for a in fc1.A_list
        )
        if i > 0:
            current_adapter = fc1.adapter_states[-1]
            ranks_this_stage = fc1.A_list[-1].shape[0]
        else:
            current_adapter = "none (base)"
            ranks_this_stage = 0

        with open(results_md_path, "a") as f:
            f.write(f"## Stage {i+1}: {env_id}\n")
            f.write(f"### Performance\n")
            f.write(f"- **Mean Reward:** {mean_reward:.2f} +/- {std_reward:.2f}\n")
            f.write(f"- **Timesteps:** {stage['timesteps']}\n\n")
            f.write(f"### Configuration\n")
            f.write(f"- **Learning Rate:** {stage['lr']}\n")
            f.write(f"- **Batch Size:** {stage['batch_size']}\n")
            f.write(f"- **Ent Coef:** {stage['ent_coef']}\n")
            f.write(f"- **Adapter State:** {current_adapter}\n")
            f.write(f"- **Ranks This Stage:** {ranks_this_stage}\n")
            f.write(f"- **Total Active Ranks:** {total_ranks}\n")
            if neurogenesis_triggered:
                f.write(f"- **🧠 Neurogenesis Triggered:** Yes (at step {neurogenesis_timestep})\n")
            elif i > 0:
                f.write(f"- **🧠 Neurogenesis Triggered:** No (probe was sufficient)\n")
            f.write(f"\n")
            
            # Log stress history if available
            if stress_history:
                f.write(f"### Stress History\n")
                f.write(f"| Timestep | Avg Reward | Stress Level |\n")
                f.write(f"|----------|------------|-------------|\n")
                for entry in stress_history:
                    f.write(f"| {entry['timestep']} | {entry['avg_reward']:.2f} | "
                            f"{entry['stress_level']}/{stress_threshold} |\n")
                f.write(f"\n")

        env.close()
        eval_env.close()

    # Final Evaluation on ALL stages
    print(f"\n{'='*60}")
    print("--- Final Evaluation on ALL Stages ---")
    print(f"{'='*60}")
    
    with open(results_md_path, "a") as f:
        f.write("## Final Evaluation (Catastrophic Forgetting Check)\n")
        f.write("| Stage | Environment | Mean Reward | Std Reward |\n")
        f.write("|-------|-------------|-------------|------------|\n")

    for i, stage in enumerate(stages):
        env_id = stage["id"]
        print(f"Final testing on {env_id}...")
        test_env = make_wrapped_env(env_id)
        
        agent.policy.current_action_dim = test_env.action_space.n
        
        mean_reward, std_reward = evaluate_policy(agent, test_env, max_obs_dim, task_id=i, n_eval_episodes=10)
        print(f"Final Reward on {env_id}: {mean_reward:.2f}")
        
        with open(results_md_path, "a") as f:
            f.write(f"| {i+1} | {env_id} | {mean_reward:.2f} | {std_reward:.2f} |\n")
        
        test_env.close()

    # Save final model
    final_model_path = os.path.join(model_path, "ppo_minigrid_evomoram_final.pt")
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")
    print(f"Results summary saved to {results_md_path}")

    # Summary of neurogenesis events
    print(f"\n{'='*60}")
    print("--- EvoMoRAM Neurogenesis Summary ---")
    print(f"{'='*60}")
    fc1 = agent.policy.actor_fc1
    for idx, state in enumerate(fc1.adapter_states):
        rank = fc1.A_list[idx].shape[0]
        print(f"  Task {idx+1}: adapter_state={state}, rank={rank}")
    total_ranks = sum(a.shape[0] for a in fc1.A_list)
    print(f"  Total ranks used: {total_ranks}")


if __name__ == "__main__":
    train_all_together()
