# Neurogenesis Project Report

## 1. Executive Summary

This project investigates continual reinforcement learning in MiniGrid environments. The central question is whether an agent can learn a sequence of increasingly difficult tasks while retaining skills learned earlier. The project compares ordinary PPO with modular low-rank memory, adaptive memory expansion, hormonal exploration, and plain-MLP controls.

The experiments show three broad results:

1. Ordinary PPO learns the simple empty-grid tasks reliably when trained separately or in a short sequential curriculum.
2. MoRAM and the combined PPO runs retain the four-task empty-grid curriculum well, with final rewards around 0.92-0.98 across previously learned tasks.
3. The adaptive EvoMoRAM and HorEvoMoRAM systems successfully demonstrate their intended structural mechanisms, including probe adapters, stress-triggered neurogenesis, entropy spikes, expert respawning, and expert retirement. However, performance degrades substantially on the harder LavaGap, dynamic-obstacle, FourRooms, and LunarLander tasks. The mechanisms were therefore demonstrated, but the results do not yet establish robust transfer or mastery of the hardest tasks.

This report is based on the experiment summaries stored in the repository. It does not claim that all runs were repeated under one common random seed or one identical evaluation protocol.

## 2. Project Aim

The project is built around a biological analogy:

- Existing model weights represent previously learned knowledge.
- A small adapter, or probe, tests whether existing knowledge transfers to a new task.
- Persistent stress from poor and stagnant reward triggers neurogenesis: the adapter grows from low rank to full rank.
- The hormonal variant adds an adrenaline-like entropy increase to encourage exploration after growth.
- The hormonal variant also applies apoptosis-like expert pruning, respawning inactive experts and permanently retiring experts that repeatedly fail to become useful.

The practical continual-learning objective is to learn tasks in sequence without retraining an independent model for every task and without catastrophically forgetting earlier tasks.

## 3. Common Experimental Setup

The experiments use Gymnasium and MiniGrid environments with `FlatObsWrapper`, which converts MiniGrid observations into flat vectors suitable for MLP policies. Most experiments use PPO with an actor-critic network and deterministic evaluation over ten episodes. Training and evaluation artifacts include saved models, logs, TensorBoard output, and videos where available.

The main sequential curriculum used by the memory experiments was:

| Stage | Environment | Timesteps |
|---|---|---:|
| 1 | MiniGrid-Empty-5x5-v0 | 40,000 |
| 2 | MiniGrid-Empty-Random-6x6-v0 | 60,000 |
| 3 | MiniGrid-Empty-8x8-v0 | 80,000 |
| 4 | MiniGrid-Empty-16x16-v0 | 300,000 |
| 5 | MiniGrid-LavaGapS5-v0 | 150,000 |
| 6 | MiniGrid-LavaGapS7-v0 | 250,000 |
| 7 | MiniGrid-Dynamic-Obstacles-5x5-v0 | 200,000 |
| 8 | MiniGrid-Dynamic-Obstacles-8x8-v0 | 500,000 |
| 9 | MiniGrid-Dynamic-Obstacles-16x16-v0 | 1,000,000 |
| 10 | MiniGrid-FourRooms-v0 | 500,000 |

The learning rate, batch size, and entropy coefficient were adjusted for task difficulty. The larger and more difficult tasks generally used larger batches and/or larger entropy coefficients.

## 4. Initial PPO Baselines

### Stage 1: 5x5 empty grid

The first baseline used Stable-Baselines3 PPO with an MLP policy of `[128, 64]` hidden units for both actor and critic. It trained for 40,000 timesteps on `MiniGrid-Empty-5x5-v0`.

Recorded result: mean reward `0.00 +/- 0.00`.

This run was an early baseline and did not solve the task according to its result summary. Later implementations of the same task reached approximately `0.94-0.95`, so this result should be treated as an unsuccessful early run rather than as a definitive limit of PPO.

### Stage 2: 8x8 empty grid

The Stage 2 standalone PPO run used the same `[128, 64]` architecture, a learning rate of `1e-3`, and 80,000 timesteps on `MiniGrid-Empty-8x8-v0`.

Recorded result: mean reward `0.96 +/- 0.00`.

A second 8x8 run using a smaller `[16, 16]` architecture was also recorded and achieved `0.00 +/- 0.00`. This indicates that network capacity and/or run configuration mattered for this task.

### Stage 3: 16x16 empty grid

The standalone 16x16 experiment trained a PPO MLP for 300,000 timesteps with a `[128, 128]` architecture and entropy coefficient `0.05`.

Recorded result: mean reward `0.98 +/- 0.00`.

### Stage 4: random 6x6 empty grid

The standalone random-position experiment used the `[128, 64]` architecture, learning rate `3e-4`, entropy coefficient `0.01`, and 60,000 timesteps.

Recorded result: mean reward `0.97 +/- 0.01`.

### Baseline conclusion

The initial baselines established that the empty-grid tasks are learnable, but also showed sensitivity to architecture and run configuration. In particular, the successful 8x8 result and the failed small-network 8x8 result should not be pooled together without noting the difference.

## 5. Sequential PPO: `allTogether` and `allToPytor`

These experiments trained one PPO model through four empty-grid tasks, changing hyperparameters between stages and then evaluating the final model on every task. They test whether a conventional model can retain simple skills in a short curriculum.

### `allTogether`

The curriculum was 5x5 empty, 8x8 empty, 16x16 empty, and random 6x6 empty. The final catastrophic-forgetting check recorded:

| Environment | Final mean reward |
|---|---:|
| Empty 5x5 | 0.95 |
| Empty 8x8 | 0.96 |
| Empty 16x16 | 0.98 |
| Empty Random 6x6 | 0.96 |

The result indicates strong retention for this relatively simple four-task curriculum.

### `allToPytor`

This is a custom PPO implementation using an explicitly defined actor-critic MLP with `[128, 64]` hidden layers. It used the same four-task curriculum. Its final evaluation recorded:

| Environment | Final mean reward |
|---|---:|
| Empty 5x5 | 0.95 |
| Empty 8x8 | 0.96 |
| Empty 16x16 | 0.98 |
| Empty Random 6x6 | 0.97 |

The two implementations therefore produced very similar results on the short empty-grid curriculum.

## 6. Fixed-Rank MoRAM

The MoRAM experiment replaced ordinary linear layers with modular low-rank memory. A base network was trained on the first task, and additional rank-16 adapters were added for later tasks. Earlier knowledge and adapters were retained while the new adapter learned the current task.

Adapters added by stage were:

| Stage | Environment | New ranks | Stage reward |
|---|---|---:|---:|
| 1 | Empty 5x5 | 0 | 0.95 +/- 0.01 |
| 2 | Empty 8x8 | 16 | 0.96 +/- 0.00 |
| 3 | Empty 16x16 | 16 | 0.97 +/- 0.00 |
| 4 | Empty Random 6x6 | 16 | 0.96 +/- 0.02 |

The final retention check recorded rewards of `0.92`, `0.96`, `0.97`, and `0.96` for stages 1 through 4. The first task dropped from `0.95` during training to `0.92` at final evaluation, but the earlier tasks remained broadly intact.

This experiment supports the usefulness of fixed modular memory for the simple continual-learning curriculum.

## 7. EvoMoRAM: Stress-Triggered Neurogenesis

EvoMoRAM extends MoRAM with adaptive capacity. Each new task initially receives a rank-2 probe. If the stress monitor observes low reward and repeated stagnation, the probe expands to rank 16. The learned probe weights are preserved during expansion, and previously learned adapters are frozen.

The stress monitor used the following logic:

- Reward at or above `0.97` is treated as mastery and reduces stress.
- Meaningful improvement reduces stress.
- Poor stagnant performance increments stress after a patience period.
- Stress at level `3` triggers neurogenesis and resets the monitor.

### Task-by-task outcome

| Stage | Environment | Mean reward | Neurogenesis |
|---|---|---:|---|
| 1 | Empty 5x5 | 0.94 +/- 0.02 | Base model |
| 2 | Empty Random 6x6 | 0.96 +/- 0.03 | No; rank-2 probe sufficient |
| 3 | Empty 8x8 | 0.96 +/- 0.00 | Yes, step 32,768 |
| 4 | Empty 16x16 | 0.97 +/- 0.00 | No; rank-2 probe sufficient |
| 5 | LavaGapS5 | 0.38 +/- 0.47 | Yes, step 36,864 |
| 6 | LavaGapS7 | 0.06 +/- 0.18 | Yes, step 71,680 |
| 7 | Dynamic Obstacles 5x5 | 0.30 +/- 0.85 | Yes, step 32,768 |
| 8 | Dynamic Obstacles 8x8 | -0.46 +/- 0.82 | Yes, step 65,536 |
| 9 | Dynamic Obstacles 16x16 | -0.30 +/- 0.46 | Yes, step 47,104 |
| 10 | FourRooms | 0.00 +/- 0.00 | Yes, step 47,104 |

The final evaluation showed strong retention of the empty-grid tasks, but weak performance on the hard tasks:

| Environment | Final mean reward |
|---|---:|
| Empty 5x5 | 0.94 |
| Empty Random 6x6 | 0.95 |
| Empty 8x8 | 0.96 |
| Empty 16x16 | 0.97 |
| LavaGapS5 | 0.48 |
| LavaGapS7 | 0.10 |
| Dynamic Obstacles 5x5 | -0.30 |
| Dynamic Obstacles 8x8 | 0.25 |
| Dynamic Obstacles 16x16 | -0.60 |
| FourRooms | 0.00 |

The project also included an EvoMoRAM run on `LunarLander-v3`. It trained for 500,000 timesteps, triggered neurogenesis at step 49,152, and recorded a mean reward of `-427.44 +/- 94.51`. This was unsuccessful as a task-learning result, although it exercised the adaptive mechanism.

### Interpretation

EvoMoRAM behaved as designed on the instrumentation level: hard tasks generated stress and caused capacity expansion, while easy tasks sometimes remained at probe rank. The difficult tasks nevertheless remained difficult after expansion. This suggests that adding capacity alone was not enough; task representation, exploration, optimization stability, and/or the routing rule may also need improvement.

## 8. HorEvoMoRAM: Hormonal Exploration and Apoptosis

HorEvoMoRAM adds two mechanisms to EvoMoRAM:

1. **Adrenaline response:** after neurogenesis, the PPO entropy coefficient is multiplied by five, capped at `0.15`, and decayed back to baseline over 15 updates.
2. **Apoptosis monitor:** experts with very low activation are detected. Some are respawned with modified parameters, while repeatedly unsuccessful experts can be permanently retired through senescence rules.

### Task-by-task outcome

| Stage | Environment | Mean reward | Neurogenesis | Apoptosis events |
|---|---|---:|---|---:|
| 1 | Empty 5x5 | 0.95 +/- 0.00 | Base model | 0 |
| 2 | Empty Random 6x6 | 0.93 +/- 0.07 | No | 0 |
| 3 | Empty 8x8 | 0.96 +/- 0.00 | No | 0 |
| 4 | Empty 16x16 | 0.98 +/- 0.00 | Yes, step 75,776 | 7 |
| 5 | LavaGapS5 | 0.16 +/- 0.33 | Yes, step 69,632 | 7 |
| 6 | LavaGapS7 | 0.09 +/- 0.27 | Yes, step 32,768 | 11 |
| 7 | Dynamic Obstacles 5x5 | 0.00 +/- 0.00 | Yes, step 67,584 | 8 |
| 8 | Dynamic Obstacles 8x8 | -0.20 +/- 0.40 | Yes, step 79,872 | 7 |
| 9 | Dynamic Obstacles 16x16 | -0.10 +/- 0.30 | Yes, step 108,544 | 15 |
| 10 | FourRooms | 0.33 +/- 0.42 | Yes, step 36,864 | 9 |

The final evaluation recorded:

| Environment | Final mean reward |
|---|---:|
| Empty 5x5 | 0.95 |
| Empty Random 6x6 | 0.97 |
| Empty 8x8 | 0.96 |
| Empty 16x16 | 0.97 |
| LavaGapS5 | 0.28 |
| LavaGapS7 | 0.00 |
| Dynamic Obstacles 5x5 | 0.00 |
| Dynamic Obstacles 8x8 | -0.30 |
| Dynamic Obstacles 16x16 | -0.30 |
| FourRooms | 0.36 |

### Interpretation

The hormonal additions improved the recorded FourRooms stage result compared with EvoMoRAM's `0.00`, but the overall hard-task results remained low. The final retention of the empty-grid tasks was excellent, and the logs show that adrenaline and apoptosis were active in the intended stages. These results support the implementation of the mechanisms more strongly than they support a claim of improved general performance.

## 9. Plain-MLP Ablation

The ablation trained a fresh independent plain MLP for each problem. It intentionally removed adapters, rank expansion, and neurogenesis. The network matched the small EvoMoRAM base architecture and had 22,732 total and trainable parameters per task.

| Environment | Mean reward |
|---|---:|
| Empty 5x5 | 0.95 |
| Empty Random 6x6 | 0.93 |
| Empty 8x8 | 0.00 |
| Empty 16x16 | 0.97 |
| LavaGapS5 | 0.95 |
| LavaGapS7 | 0.62 |
| Dynamic Obstacles 5x5 | -0.10 |
| Dynamic Obstacles 8x8 | 0.00 |
| Dynamic Obstacles 16x16 | 0.00 |
| FourRooms | 0.36 |

This ablation shows that some hard-task failures are not caused only by continual learning. Even an independently trained plain MLP failed or struggled on several environments. Conversely, the plain MLP did much better than EvoMoRAM on LavaGapS5 and LavaGapS7, so the adaptive memory system did not consistently improve task-specific learning.

The comparison is not a perfectly controlled benchmark because training schedules and architecture details differ across experiments. It should be used diagnostically rather than as a definitive ranking.

## 10. Overall Findings

### What worked

- PPO solved the simple empty-grid environments under suitable configurations.
- Sequential PPO retained the four-task empty-grid curriculum in both `allTogether` and `allToPytor`.
- Fixed-rank MoRAM retained earlier empty-grid skills with only a modest drop on the first task.
- EvoMoRAM correctly used rank-2 probes for easy transfer and expanded to full rank when stress accumulated.
- HorEvoMoRAM implemented and logged entropy spikes, expert respawning, and expert retirement.
- The final memory-based evaluations retained the simple tasks despite later training stages.

### What did not work reliably

- EvoMoRAM did not solve the harder LavaGap, dynamic-obstacle, or FourRooms tasks consistently.
- HorEvoMoRAM also struggled on most hard tasks, despite additional exploration and pruning mechanisms.
- EvoMoRAM performed poorly on LunarLander in the recorded run.
- More capacity did not automatically produce better exploration or better policies.
- Results vary across runs and implementations, especially for early PPO baselines and the 8x8 task.

## 11. Limitations and Reproducibility Notes

- The result files are summaries, not raw aggregated experiment data.
- Seeds, number of evaluation episodes, and exact checkpoint selection are not uniformly documented for every directory.
- Some results use different architectures, learning rates, entropy coefficients, batch sizes, and curricula.
- The `allTogether` and `allToPytor` experiments cover only four simple tasks, whereas EvoMoRAM and HorEvoMoRAM continue through ten tasks. Direct aggregate comparisons are therefore inappropriate.
- The ablation uses independent models per problem, so it tests task difficulty and model capacity but not continual-learning retention.
- A negative reward on dynamic environments or LunarLander indicates failure under that task's reward scale; rewards should not be compared across unrelated environments as if they shared one scale.
- The reported stress levels and expert lifecycle events demonstrate internal behavior, but they are not themselves evidence that the resulting policy is better.

## 12. Recommended Next Experiments

1. Repeat every main comparison with fixed seeds and multiple independent runs.
2. Use one shared stage order, evaluation protocol, and budget for plain PPO, MoRAM, EvoMoRAM, and HorEvoMoRAM.
3. Separate task-learning performance from retention by reporting both the reward immediately after each stage and the reward after the complete curriculum.
4. Log adapter counts, routing frequencies, entropy, and stress alongside reward curves.
5. Compare probe-only, fixed-rank, neurogenesis-only, adrenaline-only, and apoptosis-only variants.
6. Investigate hard-task representation and exploration before increasing memory capacity further.
7. Validate the method on LunarLander only after confirming the action and observation handling is appropriate for that environment.

## 13. Source Map

The main evidence used for this report is stored in:

- [Stage 1 results](Stage1-MLP-MiniGrid/results.md)
- [Stage 2 results](Stage-2-MLP-MiniGrid/results.md)
- [Stage 2 small-network results](Stage-2-MLP-MiniGrid/results_16_16.md)
- [Stage 3 results](Stage-3-MLP-MiniGrid/results.md)
- [Stage 4 results](Stage-4-MLP-MiniGrid/results.md)
- [Sequential PPO results](allTogether/results.md)
- [Custom PPO results](allToPytor/results.md)
- [MoRAM results](MeMoRam/results.md)
- [EvoMoRAM results](evoMoRam/results.md)
- [HorEvoMoRAM results](HorEvoMoRam/results.md)
- [Plain-MLP ablation results](evoMoRam_ablation/results.md)
- [EvoMoRAM implementation](evoMoRam/evo_moram.py)
- [HorEvoMoRAM implementation](HorEvoMoRam/hor_evo_moram.py)

