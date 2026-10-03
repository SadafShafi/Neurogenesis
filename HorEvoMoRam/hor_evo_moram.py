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
# 2b. Adrenaline Response — Hormonal Exploration Injection
# -----------------------------------------
class AdrenalineResponse:
    """
    Simulates the hormonal "adrenaline" response that accompanies neurogenesis.
    
    Biological analogy: When stress triggers growth of new neural pathways,
    accompanying hormones (adrenaline, cortisol) drastically increase
    behavioral variability. The organism MUST try new actions to generate
    the novel experiences needed to train those new pathways.
    
    In PPO terms: spike the entropy coefficient after neurogenesis to force
    exploration, then decay it back to baseline as the new adapters learn.
    """
    def __init__(self, spike_multiplier=5.0, decay_steps=15):
        self.spike_multiplier = spike_multiplier  # how much to multiply ent_coef
        self.decay_steps = decay_steps            # update cycles to decay back
        self.active = False
        self.steps_remaining = 0
        self.base_ent_coef = None
        self.peak_ent_coef = None
    
    def trigger(self, current_ent_coef):
        """Spike the entropy coefficient — called at neurogenesis."""
        self.active = True
        self.base_ent_coef = current_ent_coef
        self.peak_ent_coef = min(current_ent_coef * self.spike_multiplier, 0.15)
        self.steps_remaining = self.decay_steps
        return self.peak_ent_coef
    
    def decay(self):
        """
        Decay the entropy coefficient one step toward baseline.
        Returns the current ent_coef after decay.
        Called once per PPO update cycle.
        """
        if not self.active:
            return None
        
        self.steps_remaining -= 1
        
        if self.steps_remaining <= 0:
            # Adrenaline has fully worn off
            self.active = False
            return self.base_ent_coef
        
        # Linear decay from peak to base
        progress = 1.0 - (self.steps_remaining / self.decay_steps)
        current = self.peak_ent_coef + (self.base_ent_coef - self.peak_ent_coef) * progress
        return current
    
    def get_status(self):
        """Return a status string for logging."""
        if not self.active:
            return "CALM"
        intensity = self.steps_remaining / self.decay_steps
        if intensity > 0.66:
            return "SURGING 🔥"
        elif intensity > 0.33:
            return "ELEVATED ⚡"
        else:
            return "FADING 💨"


# -----------------------------------------
# 2c. Apoptosis Monitor — Synaptic Pruning, Respawning & Senescence
# -----------------------------------------
class ApoptosisMonitor:
    """
    Tracks expert activation frequencies and prunes dead neurons.
    Implements three-phase lifecycle: Apoptosis → Escalation → Senescence.
    
    Biological analogy: In neural development, neurons that fail to form
    useful synaptic connections undergo apoptosis (programmed cell death)
    and are replaced by new neurons. However, if a neuron repeatedly fails
    to integrate, it enters senescence — a permanent state of quiescence
    where the cell stops dividing and is no longer replaced.
    
    Three senescence strategies work together:
      1. Global Respawn Limit: After max_respawns total, apoptosis shuts down.
      2. Per-Expert Death Tracking: Experts that die max_deaths_per_expert
         times are permanently frozen (senescent) and excluded from routing.
      3. Escalating Variance: Each death doubles the respawn variance
         (2x → 4x → 8x...) to search increasingly radical parameter subspaces
         before giving up.
    """
    def __init__(self, check_interval=5, min_activation_rate=0.01, 
                 respawn_variance_multiplier=3.0, max_respawns=500,
                 max_deaths_per_expert=3):
        self.check_interval = check_interval       # PPO updates between checks
        self.min_activation_rate = min_activation_rate  # below this = dead expert
        self.respawn_variance_multiplier = respawn_variance_multiplier
        self.max_respawns = max_respawns            # global respawn budget
        self.max_deaths_per_expert = max_deaths_per_expert  # per-expert limit
        self.active = False
        self.senescent = False  # True once global limit hit
        self.updates_since_neurogenesis = 0
        self.total_respawns = 0
        self.total_senescent = 0  # experts permanently retired
        self.respawn_history = []  # list of {timestep, layer, n_dead, n_respawned, n_senescent}
    
    def start_monitoring(self):
        """Begin monitoring after a neurogenesis event."""
        if self.senescent:
            return  # global budget exhausted, don't restart
        self.active = True
        self.updates_since_neurogenesis = 0
    
    def tick(self):
        """Called once per PPO update. Returns True when it's time to check."""
        if not self.active:
            return False
        self.updates_since_neurogenesis += 1
        return self.updates_since_neurogenesis % self.check_interval == 0
    
    def check_global_budget(self):
        """Check if we've exhausted the global respawn budget."""
        if self.total_respawns >= self.max_respawns:
            self.active = False
            self.senescent = True
            return True
        return False
    
    def get_variance_for_expert(self, death_count):
        """
        Escalating variance: each death doubles the variance multiplier.
        Death 1: base_var * 2^0 = 3.0x
        Death 2: base_var * 2^1 = 6.0x  
        Death 3: base_var * 2^2 = 12.0x (if max_deaths_per_expert > 3)
        """
        return self.respawn_variance_multiplier * (2 ** (death_count - 1))
    
    def get_status(self):
        """Return a status string for logging."""
        if self.senescent:
            return f"SENESCENT 🧓 (budget exhausted: {self.total_respawns}/{self.max_respawns}, {self.total_senescent} permanently retired)"
        if not self.active:
            return ""
        return f"Apoptosis: monitoring (respawns={self.total_respawns}/{self.max_respawns}, senescent={self.total_senescent})"

# -----------------------------------------
# 3. EvoMoRAM Layer — Adaptive Rank Expansion + Apoptosis
# -----------------------------------------
class EvoMoRAMLinear(nn.Module):
    """
    Evolutionary Mixture of Rank-1 Associative Memory (EvoMoRAM) Layer.
    
    Extends MoRAM with stress-triggered neurogenesis and apoptosis:
      1. On a new task, a minimal "probe" adapter (r_probe=2) is added first
      2. If stress is detected, neurogenesis expands the probe to full rank (r_full=16)
      3. The probe's learned weights are PRESERVED and augmented with new capacity
      4. Dead experts (never activated by top-k) are detected and respawned
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
        
        # --- Activation tracking for apoptosis ---
        self._tracking_activations = False
        self._activation_counts = None   # tensor: counts per expert in latest adapter
        self._total_forward_samples = 0  # total samples seen while tracking
        
        # --- Senescence tracking (per-expert death counts) ---
        self._death_counts = {}   # dict: adapter_idx -> tensor of death counts per expert
        self._senescent_mask = {} # dict: adapter_idx -> bool tensor (True = permanently dead)

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
            torch.randn(self.out_features, self.r_probe, device=device) * 0.01
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
        expand_B = torch.randn(self.out_features, r_expand, device=device) * 0.01
        
        # Concatenate: preserve probe + add new capacity
        new_A = nn.Parameter(torch.cat([old_A.data, expand_A], dim=0))  # [r_full, in_features]
        new_B = nn.Parameter(torch.cat([old_B.data, expand_B], dim=1))  # [out_features, r_full]
        
        self.A_list[task_idx] = new_A
        self.B_list[task_idx] = new_B
        self.adapter_states[-1] = 'full'

    def enable_activation_tracking(self):
        """Start tracking expert activation frequencies for the latest adapter."""
        if len(self.A_list) == 0:
            return
        r = self.A_list[-1].shape[0]
        device = self.base_layer.weight.device
        self._tracking_activations = True
        self._activation_counts = torch.zeros(r, device=device)
        self._total_forward_samples = 0

    def disable_activation_tracking(self):
        """Stop tracking."""
        self._tracking_activations = False

    def get_activation_frequencies(self):
        """
        Return per-expert activation frequencies for the latest adapter.
        Returns: tensor of shape [r] with values in [0, 1].
        """
        if self._activation_counts is None or self._total_forward_samples == 0:
            return None
        return self._activation_counts / self._total_forward_samples

    def _ensure_senescence_tracking(self, task_idx, r, device):
        """Lazily initialize per-expert death counts and senescent mask for an adapter."""
        if task_idx not in self._death_counts:
            self._death_counts[task_idx] = torch.zeros(r, dtype=torch.long, device=device)
            self._senescent_mask[task_idx] = torch.zeros(r, dtype=torch.bool, device=device)

    def get_senescent_count(self):
        """Return total number of permanently retired experts across all adapters."""
        return sum(mask.sum().item() for mask in self._senescent_mask.values())

    def respawn_dead_experts(self, dead_indices, apoptosis_monitor):
        """
        Re-initialize dead experts with escalating variance and senescence.
        
        For each dead expert:
          1. Increment its death counter
          2. If deaths >= max_deaths_per_expert → mark as SENESCENT (freeze forever)
          3. Otherwise → respawn with escalating variance (2^(deaths-1) * base_var)
        
        Args:
            dead_indices: list of expert indices that are dead
            apoptosis_monitor: ApoptosisMonitor with senescence config
            
        Returns:
            (n_respawned, n_newly_senescent, per_expert_details)
        """
        if len(self.A_list) == 0:
            return 0, 0, []
        
        task_idx = len(self.A_list) - 1
        A = self.A_list[task_idx]
        B = self.B_list[task_idx]
        device = A.device
        r = A.shape[0]
        
        self._ensure_senescence_tracking(task_idx, r, device)
        
        n_respawned = 0
        n_newly_senescent = 0
        per_expert_details = []  # (idx, action, death_count, variance)
        
        for idx in dead_indices:
            # Skip already senescent experts
            if self._senescent_mask[task_idx][idx]:
                continue
            
            # Increment death counter
            self._death_counts[task_idx][idx] += 1
            death_count = self._death_counts[task_idx][idx].item()
            
            if death_count >= apoptosis_monitor.max_deaths_per_expert:
                # === SENESCENCE: permanently retire this expert ===
                # Zero out weights and freeze
                A.data[idx].zero_()
                B.data[:, idx].zero_()
                self._senescent_mask[task_idx][idx] = True
                n_newly_senescent += 1
                per_expert_details.append((idx, 'senescent', death_count, 0.0))
            else:
                # === ESCALATING RESPAWN: higher variance each death ===
                variance = apoptosis_monitor.get_variance_for_expert(death_count)
                A.data[idx] = torch.randn(self.in_features, device=device) * (
                    variance / np.sqrt(self.in_features)
                )
                B.data[:, idx] = torch.randn(self.out_features, device=device) * 0.01
                n_respawned += 1
                per_expert_details.append((idx, 'respawned', death_count, variance))
        
        # Reset activation tracking after respawn
        if self._activation_counts is not None:
            self._activation_counts.zero_()
            self._total_forward_samples = 0
        
        return n_respawned, n_newly_senescent, per_expert_details

    def zero_senescent_gradients(self):
        """Zero out gradients for permanently retired experts to prevent mutation."""
        if len(self.A_list) == 0:
            return
        task_idx = len(self.A_list) - 1
        if task_idx in self._senescent_mask:
            s_mask = self._senescent_mask[task_idx]
            if s_mask.any():
                A = self.A_list[task_idx]
                B = self.B_list[task_idx]
                if A.grad is not None:
                    A.grad.data[s_mask] = 0.0
                if B.grad is not None:
                    B.grad.data[:, s_mask] = 0.0

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

    def forward(self, x, task_id=None):
        base_out = self.base_layer(x)
        
        active_A_list = list(self.A_list)
        active_B_list = list(self.B_list)

        if len(active_A_list) == 0:
            return base_out

        # Concatenate active memory atoms
        A_cat = torch.cat(active_A_list, dim=0)  # [total_r, in_features]
        B_cat = torch.cat(active_B_list, dim=1)  # [out_features, total_r]

        total_r = A_cat.shape[0]

        # 1. Cosine Similarity Routing
        # L2-normalize experts to remove magnitude bias (prevents gradient starvation)
        A_cat_norm = A_cat / (torch.norm(A_cat, p=2, dim=1, keepdim=True) + 1e-8)
        route_act = F.linear(x, A_cat_norm)  # [batch, total_r]
        
        # Original raw activations for value retrieval
        raw_act = F.linear(x, A_cat)

        # 2. Relevance Scoring
        # Remove the normalization across experts to prevent shrinking scores as total_r grows
        norm_act = route_act 
        
        if self.training:
            # Add noise to routing to encourage exploration of new experts and prevent collapse
            noisy_norm_act = norm_act + torch.randn_like(norm_act) * 0.1
        else:
            noisy_norm_act = norm_act

        # --- Apply Task Isolation Mask ---
        if task_id is not None and task_id < len(active_A_list) - 1:
            # Mask out experts from tasks > task_id
            allowed_r = sum(A.shape[0] for A in active_A_list[:task_id + 1])
            if allowed_r < total_r:
                future_mask = torch.ones(total_r, dtype=torch.bool, device=norm_act.device)
                future_mask[:allowed_r] = False
                noisy_norm_act = noisy_norm_act.masked_fill(future_mask.unsqueeze(0), float('-inf'))

        # 3. Sparsity (Top-k masking)
        current_k = min(self.k, total_r)

        # --- Apply Senescence Mask ---
        if hasattr(self, '_senescent_mask') and len(active_A_list) > 0:
            full_s_mask = torch.zeros(total_r, dtype=torch.bool, device=norm_act.device)
            offset = 0
            for t_idx, A in enumerate(active_A_list):
                if t_idx in self._senescent_mask:
                    s_mask = self._senescent_mask[t_idx]
                    full_s_mask[offset:offset+len(s_mask)] = s_mask
                offset += A.shape[0]
                
            if full_s_mask.any():
                # Mask senescent experts with -inf before top-k selection
                noisy_norm_act = noisy_norm_act.masked_fill(full_s_mask.unsqueeze(0), float('-inf'))

        topk_vals, topk_indices = torch.topk(noisy_norm_act, current_k, dim=-1)
        
        # We still want the ACTUAL `norm_act` values in the mask for the softmax, not the noisy ones!
        mask = torch.full_like(norm_act, float('-inf'))
        selected_true_vals = torch.gather(norm_act, -1, topk_indices)
        mask.scatter_(-1, topk_indices, selected_true_vals)

        # --- Activation tracking for apoptosis ---
        if self._tracking_activations and self.training and self._activation_counts is not None:
            # Compute offset: where the latest adapter's experts start in A_cat
            offset = sum(A.shape[0] for A in active_A_list[:-1]) if len(active_A_list) > 1 else 0
            latest_r = active_A_list[-1].shape[0]
            batch_size = x.shape[0]
            
            # Count how many times each latest-adapter expert appears in top-k
            for b in range(topk_indices.shape[0]):
                for idx in topk_indices[b]:
                    local_idx = idx.item() - offset
                    if 0 <= local_idx < latest_r:
                        self._activation_counts[local_idx] += 1
            self._total_forward_samples += batch_size

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
        # self.actor_out = nn.Linear(64, action_dim)
        
        # # CRITIC NETWORK — Using EvoMoRAM Layers
        # self.critic_fc1 = EvoMoRAMLinear(obs_dim, 128, r_probe=2, r_full=16, k=4)
        # self.critic_fc2 = EvoMoRAMLinear(128, 64, r_probe=2, r_full=16, k=4)
        # self.critic_out = nn.Linear(64, 1)
        
        # self.activation = nn.Tanh()

        print('SMaller Network')
        # ACTOR NETWORK — Using EvoMoRAM Layers
        self.actor_fc1 = EvoMoRAMLinear(obs_dim, 64, r_probe=2, r_full=16, k=4)
        self.actor_fc2 = EvoMoRAMLinear(64, 32, r_probe=2, r_full=16, k=4)
        self.actor_out = nn.ModuleList([nn.Linear(32, action_dim)])
        
        # CRITIC NETWORK — Using EvoMoRAM Layers
        self.critic_fc1 = EvoMoRAMLinear(obs_dim, 64, r_probe=2, r_full=16, k=4)
        self.critic_fc2 = EvoMoRAMLinear(64, 32, r_probe=2, r_full=16, k=4)
        self.critic_out = nn.ModuleList([nn.Linear(32, 1)])
        
        self.activation = nn.Tanh()

    def add_task_head(self):
        # Freeze existing heads
        for head in self.actor_out:
            for param in head.parameters():
                param.requires_grad = False
        for head in self.critic_out:
            for param in head.parameters():
                param.requires_grad = False
                
        # Add new heads
        device = next(self.parameters()).device
        new_actor_head = nn.Linear(32, self.actor_out[0].out_features).to(device)
        new_critic_head = nn.Linear(32, 1).to(device)
        self.actor_out.append(new_actor_head)
        self.critic_out.append(new_critic_head)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, task_id=None):
        # Actor
        x_a = self.activation(self.actor_fc1(state, task_id))
        x_a = self.activation(self.actor_fc2(x_a, task_id))
        
        head_idx = task_id if task_id is not None else len(self.actor_out) - 1
        action_logits = self.actor_out[head_idx](x_a)
        action_probs = Categorical(logits=action_logits)
        action = action_probs.sample()
        
        # Critic
        x_c = self.activation(self.critic_fc1(state, task_id))
        x_c = self.activation(self.critic_fc2(x_c, task_id))
        state_value = self.critic_out[head_idx](x_c)
        
        return action, action_probs.log_prob(action), state_value.squeeze(-1)

    def evaluate(self, state, action, task_id=None):
        # Actor
        x_a = self.activation(self.actor_fc1(state, task_id))
        x_a = self.activation(self.actor_fc2(x_a, task_id))
        
        head_idx = task_id if task_id is not None else len(self.actor_out) - 1
        action_logits = self.actor_out[head_idx](x_a)
        action_probs = Categorical(logits=action_logits)
        
        # Critic
        x_c = self.activation(self.critic_fc1(state, task_id))
        x_c = self.activation(self.critic_fc2(x_c, task_id))
        state_value = self.critic_out[head_idx](x_c)
        
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
        self.base_ent_coef = ent_coef  # remember baseline for adrenaline decay
        self.device = device
        self.lr = lr

        # Adrenaline system — hormonal exploration injection
        self.adrenaline = AdrenalineResponse(spike_multiplier=5.0, decay_steps=15)

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def _rebuild_optimizer(self, lr):
        """Rebuild optimizer with only trainable parameters, preserving momentum."""
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = optim.Adam(trainable_params, lr=lr)
        else:
            self.optimizer.param_groups[0]['params'] = trainable_params
            self.optimizer.param_groups[0]['lr'] = lr
            
            # Initialize zeroed state for new parameters to prevent Adam from failing
            for p in trainable_params:
                if p not in self.optimizer.state:
                    self.optimizer.state[p] = {
                        'step': torch.tensor(0.0),
                        'exp_avg': torch.zeros_like(p, memory_format=torch.preserve_format),
                        'exp_avg_sq': torch.zeros_like(p, memory_format=torch.preserve_format)
                    }
                    
            # Cleanup frozen parameter states to save memory
            trainable_set = set(trainable_params)
            for p in list(self.optimizer.state.keys()):
                if p not in trainable_set:
                    del self.optimizer.state[p]
                    
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
                
        # Add new independent output head for this task
        self.policy.add_task_head()
        
        # Sync old policy
        self.policy_old = copy.deepcopy(self.policy)
        
        # Rebuild optimizer with only trainable params (the probes)
        n_params = self._rebuild_optimizer(new_lr)
        print(f"  --> ATTEMPT PHASE: Added probe adapters (r=2). "
              f"Optimizing {n_params} parameter tensors.")

    def trigger_neurogenesis(self, new_lr):
        """
        NEUROGENESIS + ADRENALINE: Expand probe adapters AND spike exploration.
        
        Called when the StressMonitor detects learning stagnation.
        Probe weights are preserved and augmented with new capacity.
        Simultaneously, the adrenaline response spikes ent_coef to force
        the agent to explore with its newly grown adapters.
        """
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.trigger_neurogenesis()
        
        # Sync old policy
        self.policy_old = copy.deepcopy(self.policy)
        
        # Rebuild optimizer with expanded parameters
        n_params = self._rebuild_optimizer(new_lr)
        
        # 🔥 ADRENALINE: Spike entropy coefficient to force exploration
        spiked_ent_coef = self.adrenaline.trigger(self.base_ent_coef)
        self.ent_coef = spiked_ent_coef
        
        print(f"  --> 🧠 NEUROGENESIS: Expanded to full adapters (r=16). "
              f"Optimizing {n_params} parameter tensors.")
        print(f"  --> 🔥 ADRENALINE: ent_coef spiked {self.base_ent_coef:.4f} → "
              f"{spiked_ent_coef:.4f} (x{self.adrenaline.spike_multiplier:.0f}). "
              f"Decays over {self.adrenaline.decay_steps} updates.")

    def enable_activation_tracking(self):
        """Enable activation tracking on all EvoMoRAM layers for apoptosis."""
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.enable_activation_tracking()

    def disable_activation_tracking(self):
        """Disable activation tracking on all EvoMoRAM layers."""
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.disable_activation_tracking()

    def check_apoptosis(self, apoptosis_monitor, current_lr):
        """
        APOPTOSIS + SENESCENCE: Check for dead experts, respawn or retire them.
        
        Three-phase lifecycle per expert:
          1. First death(s): Respawn with escalating variance (2x, 4x, 8x...)
          2. max_deaths_per_expert reached: Permanently retire (senescence)
          3. Global max_respawns reached: Shut down entire apoptosis system
        
        Returns: (total_dead, total_respawned, total_senescent, layer_reports)
        """
        # Check global budget first
        if apoptosis_monitor.check_global_budget():
            print(f"\n  {'~'*50}")
            print(f"  🧓 SENESCENCE: Global respawn budget exhausted "
                  f"({apoptosis_monitor.total_respawns}/{apoptosis_monitor.max_respawns}). "
                  f"Apoptosis permanently disabled.")
            print(f"  {'~'*50}\n")
            return 0, 0, 0, []
        
        total_respawned = 0
        total_dead = 0
        total_senescent = 0
        layer_reports = []
        
        for name, module in self.policy.named_modules():
            if isinstance(module, EvoMoRAMLinear):
                freqs = module.get_activation_frequencies()
                if freqs is None:
                    continue
                
                # Find dead experts (below activation threshold)
                dead_mask = freqs < apoptosis_monitor.min_activation_rate
                dead_indices = torch.where(dead_mask)[0].tolist()
                
                if len(dead_indices) > 0:
                    n_respawned, n_senescent, details = module.respawn_dead_experts(
                        dead_indices, apoptosis_monitor
                    )
                    total_dead += len(dead_indices)
                    total_respawned += n_respawned
                    total_senescent += n_senescent
                    layer_reports.append((name, len(dead_indices), n_respawned, 
                                         n_senescent, freqs, details))
        
        if total_respawned > 0 or total_senescent > 0:
            # Sync old policy after respawning
            self.policy_old = copy.deepcopy(self.policy)
            # Rebuild optimizer to pick up new parameters
            self._rebuild_optimizer(current_lr)
            apoptosis_monitor.total_respawns += total_respawned
            apoptosis_monitor.total_senescent += total_senescent
            
            # Re-check global budget after this batch
            apoptosis_monitor.check_global_budget()
        
        return total_dead, total_respawned, total_senescent, layer_reports

    def add_new_task(self, new_lr):
        """Legacy: directly add full-rank adapters (same as original MoRAM)."""
        for module in self.policy.modules():
            if isinstance(module, EvoMoRAMLinear):
                module.add_task()
                
        self.policy.add_task_head()
        
        self.policy_old = copy.deepcopy(self.policy)
        n_params = self._rebuild_optimizer(new_lr)
        print(f"  --> Added full MoRAM task. Optimizing {n_params} parameter tensors.")

    def select_action(self, state, task_id=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, task_id)
            
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
            next_value = self.policy_old.critic_out[-1](x_c).squeeze(-1)
            
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
                
                # ZERO OUT GRADIENTS FOR SENESCENT EXPERTS
                for module in self.policy.modules():
                    if isinstance(module, EvoMoRAMLinear):
                        module.zero_senescent_gradients()
                
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

    episode_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            padded_state = pad_state(state, max_obs_dim)
            state_t = torch.FloatTensor(padded_state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                x_a = agent.policy.activation(agent.policy.actor_fc1(state_t, task_id))
                x_a = agent.policy.activation(agent.policy.actor_fc2(x_a, task_id))
                head_idx = task_id if task_id is not None else len(agent.policy.actor_out) - 1
                action_logits = agent.policy.actor_out[head_idx](x_a)
                action_probs = F.softmax(action_logits, dim=-1)
                action = Categorical(probs=action_probs).sample().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        episode_rewards.append(ep_reward)

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
    base_path = "./HorEvoMoRam/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

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

    # Apoptosis hyperparameters
    apoptosis_check_interval = 5   # PPO updates between dead-expert checks
    apoptosis_min_activation = 0.01  # below 1% = dead expert
    apoptosis_variance_mult = 3.0  # respawn with 3x init variance

    update_timestep = 2048 
    gamma = 0.99
    epochs = 10
    eps_clip = 0.2

    agent = None
    max_obs_dim = 0 

    max_action_dim = 0

    # Pre-calculate max observation and action dimensions
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
        f.write("# Continual Learning Results — HorEvoMoRAM (Hormonal Stress-Triggered Neurogenesis)\n\n")

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
        apoptosis_events = []
        apoptosis_monitor = ApoptosisMonitor(
            check_interval=apoptosis_check_interval,
            min_activation_rate=apoptosis_min_activation,
            respawn_variance_multiplier=apoptosis_variance_mult
        )

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
                
                # --- Adrenaline Decay (runs every update cycle) ---
                if agent.adrenaline.active:
                    new_ent_coef = agent.adrenaline.decay()
                    if new_ent_coef is not None:
                        agent.ent_coef = new_ent_coef
                
                # --- Stress Monitoring (Stage 2+ only) ---
                if stress_monitor is not None:
                    current_stress = stress_monitor.update(avg_ep_reward)
                    stress_history.append({
                        "timestep": time_step,
                        "avg_reward": avg_ep_reward,
                        "stress_level": current_stress,
                        "ent_coef": agent.ent_coef,
                        "adrenaline": agent.adrenaline.get_status()
                    })
                    
                    stress_indicator = "🟢" if current_stress == 0 else "🟡" if current_stress < stress_threshold else "🔴"
                    adapter_state = "PROBE" if not neurogenesis_triggered else "FULL"
                    adrenaline_status = agent.adrenaline.get_status()
                    
                    log_line = (f"  Step {time_step}/{stage['timesteps']} | "
                                f"Reward: {avg_ep_reward:.2f} | "
                                f"Stress: {current_stress}/{stress_threshold} {stress_indicator} | "
                                f"Adapters: {adapter_state}")
                    if agent.adrenaline.active:
                        log_line += f" | Adrenaline: {adrenaline_status} (ent={agent.ent_coef:.4f})"
                    print(log_line)
                    
                    # --- Apoptosis + Senescence check (after neurogenesis only) ---
                    if neurogenesis_triggered and apoptosis_monitor.active:
                        if apoptosis_monitor.tick():
                            n_dead, n_respawned, n_senescent, layer_reports = agent.check_apoptosis(
                                apoptosis_monitor, stage["lr"]
                            )
                            if n_respawned > 0 or n_senescent > 0:
                                print(f"\n  {'~'*50}")
                                action_parts = []
                                if n_respawned > 0:
                                    action_parts.append(f"{n_respawned} respawned")
                                if n_senescent > 0:
                                    action_parts.append(f"{n_senescent} retired 🧓")
                                print(f"  💀 APOPTOSIS at step {time_step}: "
                                      f"{n_dead} dead experts → {', '.join(action_parts)}")
                                for lname, n_d, n_r, n_s, freqs, details in layer_reports:
                                    freq_str = ', '.join(f'{f:.3f}' for f in freqs.tolist())
                                    # Show per-expert senescence details
                                    senescent_indices = [d[0] for d in details if d[1] == 'senescent']
                                    escalated = [(d[0], d[2], d[3]) for d in details if d[1] == 'respawned' and d[2] > 1]
                                    line = f"     {lname}: {n_d} dead [{freq_str}]"
                                    if senescent_indices:
                                        line += f" | 🧓 retired: {senescent_indices}"
                                    if escalated:
                                        esc_str = ', '.join(f'idx{i}(death#{d},var={v:.1f}x)' for i, d, v in escalated)
                                        line += f" | ⬆ escalated: {esc_str}"
                                    print(line)
                                budget_str = f"  Budget: {apoptosis_monitor.total_respawns}/{apoptosis_monitor.max_respawns}"
                                if apoptosis_monitor.total_senescent > 0:
                                    budget_str += f" | Senescent: {apoptosis_monitor.total_senescent} experts permanently retired"
                                print(budget_str)
                                print(f"  {'~'*50}\n")
                                apoptosis_events.append({
                                    "timestep": time_step,
                                    "dead": n_dead,
                                    "respawned": n_respawned,
                                    "senescent": n_senescent
                                })
                    
                    # Check for neurogenesis trigger
                    if stress_monitor.is_stressed() and not neurogenesis_triggered:
                        print(f"\n  {'!'*50}")
                        print(f"  🧠 STRESS DETECTED at step {time_step}!")
                        print(f"  Triggering NEUROGENESIS + ADRENALINE...")
                        print(f"  {'!'*50}\n")
                        
                        agent.trigger_neurogenesis(stage["lr"])
                        neurogenesis_triggered = True
                        neurogenesis_timestep = time_step
                        stress_monitor.reset()
                        
                        # Start apoptosis monitoring + activation tracking
                        apoptosis_monitor.start_monitoring()
                        agent.enable_activation_tracking()
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
                f.write(f"- **🔥 Adrenaline Response:** Spiked ent_coef x{agent.adrenaline.spike_multiplier:.0f}, "
                        f"decayed over {agent.adrenaline.decay_steps} updates\n")
                if apoptosis_events:
                    f.write(f"- **💀 Apoptosis Events:** {len(apoptosis_events)} "
                            f"({apoptosis_monitor.total_respawns} experts respawned, "
                            f"{apoptosis_monitor.total_senescent} permanently retired)\n")
                    if apoptosis_monitor.senescent:
                        f.write(f"- **🧓 Senescence:** Global budget exhausted "
                                f"({apoptosis_monitor.total_respawns}/{apoptosis_monitor.max_respawns})\n")
                    for evt in apoptosis_events:
                        parts = [f"{evt['respawned']} respawned"]
                        if evt.get('senescent', 0) > 0:
                            parts.append(f"{evt['senescent']} retired 🧓")
                        f.write(f"  - Step {evt['timestep']}: {evt['dead']} dead → {', '.join(parts)}\n")
                else:
                    f.write(f"- **💀 Apoptosis Events:** None (all experts healthy)\n")
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
        
        mean_reward, std_reward = evaluate_policy(agent, test_env, max_obs_dim, task_id=i, n_eval_episodes=10)
        print(f"Final Reward on {env_id}: {mean_reward:.2f}")
        
        with open(results_md_path, "a") as f:
            f.write(f"| {i+1} | {env_id} | {mean_reward:.2f} | {std_reward:.2f} |\n")
        
        test_env.close()

    # Save final model
    final_model_path = os.path.join(model_path, "ppo_minigrid_horevomoram_final.pt")
    torch.save(agent.policy.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")
    print(f"Results summary saved to {results_md_path}")

    # Summary of neurogenesis events
    print(f"\n{'='*60}")
    print("--- HorEvoMoRAM Neurogenesis + Adrenaline + Apoptosis Summary ---")
    print(f"{'='*60}")
    fc1 = agent.policy.actor_fc1
    for idx, state in enumerate(fc1.adapter_states):
        rank = fc1.A_list[idx].shape[0]
        print(f"  Task {idx+1}: adapter_state={state}, rank={rank}")
    total_ranks = sum(a.shape[0] for a in fc1.A_list)
    print(f"  Total ranks used: {total_ranks}")


if __name__ == "__main__":
    train_all_together()
