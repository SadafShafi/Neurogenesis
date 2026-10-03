# Neurogenesis Project — Experiment-Wise Summary

> **Project Theme:** Bio-inspired continual reinforcement learning in MiniGrid environments. The central question is whether an agent can learn a sequence of increasingly difficult navigation tasks while retaining previously learned skills — inspired by how biological brains grow new neurons (neurogenesis), release stress hormones (adrenaline), and prune unused connections (apoptosis).

---

## Experiment 1 — PPO Baselines (Individual Tasks)

**Directories:** `Stage1-MLP-MiniGrid/`, `Stage-2-MLP-MiniGrid/`, `Stage-3-MLP-MiniGrid/`, `Stage-4-MLP-MiniGrid/`

### Goal
Establish that standard PPO can learn simple MiniGrid navigation tasks *independently*, and find working hyperparameters (architecture, learning rate, entropy) for each grid size.

### Input / Output (Intuition)
- **Input:** A flat vector observation of a MiniGrid grid (agent position, direction, wall/goal locations encoded as numbers). The agent sees the grid world converted into a 1D numerical vector.
- **Output:** A discrete action (turn left, turn right, move forward, etc.). The reward is positive (~0.95–0.98) when the agent reaches the green goal tile quickly, and ~0 if it fails.

### Methodology
- Used **Stable-Baselines3 PPO** with a 2-layer MLP policy (`[128, 64]` for actor and critic).
- Each environment was trained *independently* — no shared weights, no sequential training.
- Environments tested: Empty-5x5, Empty-8x8, Empty-16x16, Empty-Random-6x6.
- Also tested a tiny `[16, 16]` network on the 8x8 grid to probe capacity sensitivity.

### Results

| Environment | Architecture | Timesteps | Mean Reward |
|---|---|---:|---:|
| Empty 5x5 | [128, 64] | 40,000 | 0.00 (early run; later solved) |
| Empty 8x8 | [128, 64] | 80,000 | **0.96** |
| Empty 8x8 | [16, 16] | 80,000 | 0.00 (too small) |
| Empty 16x16 | [128, 128] | 300,000 | **0.98** |
| Empty Random 6x6 | [128, 64] | 60,000 | **0.97** |

> **Takeaway:** The empty-grid tasks are solvable with PPO, but results are sensitive to architecture and hyperparameters. The tiny [16, 16] network completely failed on 8x8.

---

## Experiment 2 — Sequential PPO (`allTogether`)

**Directory:** `allTogether/`

### Goal
Test whether a *single* PPO model (Stable-Baselines3) can learn four empty-grid tasks *one after another* and still remember the earlier tasks — a basic **catastrophic forgetting** check.

### Input / Output (Intuition)
- Same as Experiment 1. Flat grid vector → discrete action → goal-reaching reward.
- The difference is that the same model trains on Task 1, then Task 2, then Task 3, then Task 4 sequentially.

### Methodology
- One PPO model with `[128, 64]` architecture trained through a **4-stage curriculum**: 5x5 → 8x8 → 16x16 → Random-6x6.
- Hyperparameters (learning rate, entropy, batch size) were tuned per stage.
- After all four stages, the final model was evaluated on *all four environments*.

### Results

| Environment | Stage Reward | Final Retention |
|---|---:|---:|
| Empty 5x5 | 0.95 | **0.95** |
| Empty 8x8 | 0.96 | **0.96** |
| Empty 16x16 | 0.98 | **0.98** |
| Empty Random 6x6 | 0.96 | **0.96** |

> **Takeaway:** Full retention on this simple 4-task curriculum. No catastrophic forgetting detected — but these are all variants of the same "walk to goal in empty grid" problem.

---

## Experiment 3 — Custom PPO (`allToPytor`)

**Directory:** `allToPytor/`

### Goal
Replicate the sequential PPO experiment using a **custom PyTorch PPO** implementation (not SB3) to verify results are framework-independent and to have a codebase that can later be extended with MoRAM layers.

### Input / Output (Intuition)
- Identical to Experiment 2. Same environments, same curriculum, same reward scheme.

### Methodology
- Wrote a from-scratch PPO with an explicit `ActorCritic` MLP (`[128, 64]` hidden layers).
- Same 4-stage curriculum: 5x5 → 8x8 → 16x16 → Random-6x6.
- Same per-stage hyperparameter tuning.

### Results

| Environment | Stage Reward | Final Retention |
|---|---:|---:|
| Empty 5x5 | 0.95 | **0.95** |
| Empty 8x8 | 0.96 | **0.96** |
| Empty 16x16 | 0.98 | **0.98** |
| Empty Random 6x6 | 0.97 | **0.97** |

> **Takeaway:** Nearly identical to Experiment 2. The custom PyTorch implementation works correctly and matches SB3 behaviour. This codebase becomes the foundation for all subsequent memory experiments.

---

## Experiment 4 — Fixed-Rank MoRAM

**Directory:** `MeMoRam/`

### Goal
Introduce **Modular low-Rank Associative Memory (MoRAM)** — replace standard linear layers with layers that grow rank-16 adapters for each new task while *freezing* all previous weights. Test whether this structured memory prevents forgetting.

### Input / Output (Intuition)
- Same flat MiniGrid observations and discrete actions.
- Internally, each linear layer now computes: `output = W₀·x + Σ(Bᵢ · softmax(Aᵢ·x / τ))` where `W₀` is the frozen base and each `(Aᵢ, Bᵢ)` pair is a task-specific adapter.

### Methodology
- **Stage 1:** Train base network (no adapters) on 5x5.
- **Stages 2–4:** Freeze all existing weights, add 16 new rank-1 experts per layer, train only the new experts.
- The routing uses a **softmax gate** with temperature `τ = 0.01` and top-k selection (`k = 4`).
- 4-stage curriculum: 5x5 → 8x8 → 16x16 → Random-6x6.
- Final evaluation on all four tasks.

### Results

| Environment | Ranks Added | Stage Reward | Final Retention |
|---|---:|---:|---:|
| Empty 5x5 | 0 (base) | 0.95 | **0.92** (slight drop) |
| Empty 8x8 | 16 | 0.96 | **0.96** |
| Empty 16x16 | 16 | 0.97 | **0.97** |
| Empty Random 6x6 | 16 | 0.96 | **0.96** |

> **Takeaway:** MoRAM retains earlier skills with only a small drop on Task 1 (0.95 → 0.92). The modular adapter approach works for this simple curriculum.

---

## Experiment 5 — EvoMoRAM (Stress-Triggered Neurogenesis)

**Directory:** `evoMoRam/`

### Goal
Extend MoRAM with **adaptive capacity expansion** inspired by neurogenesis. Instead of always adding rank-16, start each new task with a minimal **rank-2 probe**. If the agent struggles (measured by a stress monitor), the probe *grows* to full rank-16. This tests whether the brain-inspired idea of "grow capacity only when stressed" works.

### Input / Output (Intuition)
- Same flat MiniGrid observations and discrete actions.
- Now tackling a **10-task curriculum** that includes hard tasks (LavaGap, Dynamic Obstacles, FourRooms) beyond simple empty grids.
- The stress monitor outputs a "stress level" (0 → 3) based on reward stagnation; level 3 triggers neurogenesis.

### Methodology
- **StressMonitor** tracks per-update reward. Stress accumulates when reward stagnates below 0.97 for 5+ consecutive updates. Stress resets on improvement or mastery.
- **Probe phase:** Each new task gets a rank-2 adapter. If existing knowledge transfers, the probe suffices.
- **Neurogenesis phase:** At stress level 3, the rank-2 probe expands to rank-16. Learned probe weights are *preserved* and augmented with new random columns.
- All previous adapters remain frozen.
- 10-stage curriculum with increasing difficulty and allocated timesteps (40K → 1M).
- Also tested on **LunarLander-v3** (a completely different domain).

### Results

| Environment | Neurogenesis? | Stage Reward | Final Retention |
|---|---|---:|---:|
| Empty 5x5 | Base model | 0.94 | **0.94** |
| Empty Random 6x6 | No (probe OK) | 0.96 | **0.95** |
| Empty 8x8 | Yes (step 32K) | 0.96 | **0.96** |
| Empty 16x16 | No (probe OK) | 0.97 | **0.97** |
| LavaGapS5 | Yes (step 36K) | 0.38 | 0.48 |
| LavaGapS7 | Yes (step 71K) | 0.06 | 0.10 |
| DynObs 5x5 | Yes (step 32K) | 0.30 | −0.30 |
| DynObs 8x8 | Yes (step 65K) | −0.46 | 0.25 |
| DynObs 16x16 | Yes (step 47K) | −0.30 | −0.60 |
| FourRooms | Yes (step 47K) | 0.00 | 0.00 |
| *LunarLander-v3* | Yes (step 49K) | −427.44 | — |

> **Takeaway:** The neurogenesis mechanism works as designed — easy tasks stay at probe rank, hard tasks trigger expansion. However, **adding capacity alone did not solve the harder tasks**. Simple task retention remains excellent.

---

## Experiment 6 — HorEvoMoRAM (Hormonal Exploration + Apoptosis)

**Directory:** `HorEvoMoRam/`

### Goal
Add two more bio-inspired mechanisms on top of EvoMoRAM:
1. **Adrenaline Response** — after neurogenesis, temporarily spike the PPO entropy coefficient (×5) to force exploration, then decay back to baseline.
2. **Apoptosis** — detect and prune "dead" experts (very low activation), respawn them with modified parameters. Experts that repeatedly fail are permanently retired (senescence).

### Input / Output (Intuition)
- Same 10-task MiniGrid curriculum as EvoMoRAM.
- Additional internal signals: adrenaline status (SURGING / ELEVATED / FADING / CALM), expert activation rates, respawn/retire counts.

### Methodology
- **Adrenaline:** After neurogenesis, `ent_coef *= 5` (capped at 0.15), then linearly decayed over 15 PPO updates.
- **Apoptosis Monitor:** Every 5 updates, experts with activation rate < 1% are flagged as dead.
  - Dead experts are **respawned** with increased variance (3× → 9× → 27×...).
  - Experts that die 3+ times are **permanently retired** (senescence).
  - A global budget of 500 total respawns.
- Same 10-stage curriculum as EvoMoRAM.

### Results

| Environment | Neurogenesis? | Apoptosis Events | Stage Reward | Final Retention |
|---|---|---:|---:|---:|
| Empty 5x5 | Base | 0 | 0.95 | **0.95** |
| Empty Random 6x6 | No | 0 | 0.93 | **0.97** |
| Empty 8x8 | No | 0 | 0.96 | **0.96** |
| Empty 16x16 | Yes (step 75K) | 7 | 0.98 | **0.97** |
| LavaGapS5 | Yes (step 69K) | 7 | 0.16 | 0.28 |
| LavaGapS7 | Yes (step 32K) | 11 | 0.09 | 0.00 |
| DynObs 5x5 | Yes (step 67K) | 8 | 0.00 | 0.00 |
| DynObs 8x8 | Yes (step 79K) | 7 | −0.20 | −0.30 |
| DynObs 16x16 | Yes (step 108K) | 15 | −0.10 | −0.30 |
| FourRooms | Yes (step 36K) | 9 | **0.33** | **0.36** |

> **Takeaway:** FourRooms improved from 0.00 (EvoMoRAM) to 0.33/0.36 (HorEvoMoRAM), demonstrating the adrenaline mechanism helped exploration. Apoptosis was actively triggered on harder stages. Overall hard-task performance remains low, but the mechanisms are demonstrably functional. Simple task retention is still excellent.

---

## Experiment 7 — Plain-MLP Ablation

**Directory:** `evoMoRam_ablation/`

### Goal
Establish a **lower bound on task difficulty** by training a fresh, independent MLP for *each* task (no sequential learning, no adapters, no memory). This answers: "Are the hard tasks just fundamentally difficult for this architecture, or is it the continual learning that causes failure?"

### Input / Output (Intuition)
- Same 10 MiniGrid environments.
- Each problem gets its own freshly initialized model — no knowledge transfer, no forgetting possible.
- Architecture matches the small EvoMoRAM base: `obs → 4 → 2 → action` (22,732 parameters).

### Methodology
- Train a plain `nn.Linear` MLP (same tiny architecture as EvoMoRAM base) on each problem from scratch.
- Same per-task hyperparameters and timestep budgets.
- No adapters, no rank expansion, no stress monitor.
- Report final mean reward for each problem independently.

### Results

| Environment | Plain MLP | EvoMoRAM | HorEvoMoRAM |
|---|---:|---:|---:|
| Empty 5x5 | 0.95 | 0.94 | 0.95 |
| Empty Random 6x6 | 0.93 | 0.95 | 0.97 |
| Empty 8x8 | 0.00 | 0.96 | 0.96 |
| Empty 16x16 | 0.97 | 0.97 | 0.97 |
| LavaGapS5 | **0.95** | 0.48 | 0.28 |
| LavaGapS7 | **0.62** | 0.10 | 0.00 |
| DynObs 5x5 | −0.10 | −0.30 | 0.00 |
| DynObs 8x8 | 0.00 | 0.25 | −0.30 |
| DynObs 16x16 | 0.00 | −0.60 | −0.30 |
| FourRooms | 0.36 | 0.00 | **0.36** |

> **Takeaway:** The plain MLP solved LavaGapS5 and LavaGapS7 *much* better than either memory system, showing those failures are not just about task difficulty — the adapters may interfere with task-specific learning. Dynamic Obstacles are hard for all approaches. The comparison is diagnostic, not a controlled benchmark.

---

## Summary Table — All Experiments at a Glance

| # | Experiment | Key Idea | Tasks | Retention | Hard Tasks |
|---|---|---|---:|---|---|
| 1 | PPO Baselines | Vanilla PPO per task | 4 | N/A (no sequence) | N/A |
| 2 | Sequential PPO (SB3) | Same model, 4 tasks in sequence | 4 | ✅ Excellent | N/A |
| 3 | Custom PPO (PyTorch) | Hand-coded PPO, same curriculum | 4 | ✅ Excellent | N/A |
| 4 | MoRAM | Fixed rank-16 adapters per task | 4 | ✅ Good (0.92+) | N/A |
| 5 | EvoMoRAM | Stress-triggered rank expansion | 10 | ✅ Excellent | ❌ Failed |
| 6 | HorEvoMoRAM | + Adrenaline + Apoptosis | 10 | ✅ Excellent | ⚠️ Marginal |
| 7 | Plain-MLP Ablation | Fresh model per task (control) | 10 | N/A (independent) | ⚠️ Mixed |

---

## Key Findings

### ✅ What Worked
- PPO reliably solves empty-grid tasks under proper configurations
- Sequential PPO and MoRAM both prevent catastrophic forgetting on simple curricula
- EvoMoRAM correctly fires neurogenesis only when truly stressed
- HorEvoMoRAM demonstrates functional adrenaline spikes and expert lifecycle management
- Simple task retention remains excellent (0.92–0.97) across all memory approaches

### ❌ What Did Not Work
- Neither EvoMoRAM nor HorEvoMoRAM solved LavaGap, Dynamic Obstacles, or FourRooms reliably
- Adding more adapter capacity did not automatically lead to better exploration or learning
- The plain-MLP ablation beat the memory systems on LavaGap tasks, suggesting adapter routing may interfere
- LunarLander experiment was unsuccessful (−427 reward)
