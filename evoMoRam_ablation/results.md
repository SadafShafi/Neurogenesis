# Ablation Results — Plain MLP (No Adapters, Separate Model Per Problem)

Each problem is trained with a **fresh, independent model** from scratch.
Same architecture as EvoMoRAM base network (obs→4→2→action) but using plain nn.Linear.
No adapters, no rank expansion, no neurogenesis.

## Problem 1: MiniGrid-Empty-5x5-v0
### Performance
- **Mean Reward:** 0.95 +/- 0.01
- **Timesteps:** 40000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 2: MiniGrid-Empty-Random-6x6-v0
### Performance
- **Mean Reward:** 0.93 +/- 0.06
- **Timesteps:** 60000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 3: MiniGrid-Empty-8x8-v0
### Performance
- **Mean Reward:** 0.00 +/- 0.00
- **Timesteps:** 80000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 4: MiniGrid-Empty-16x16-v0
### Performance
- **Mean Reward:** 0.97 +/- 0.00
- **Timesteps:** 300000

### Configuration
- **Learning Rate:** 0.0005
- **Batch Size:** 128
- **Ent Coef:** 0.05
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 5: MiniGrid-LavaGapS5-v0
### Performance
- **Mean Reward:** 0.95 +/- 0.01
- **Timesteps:** 150000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.02
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 6: MiniGrid-LavaGapS7-v0
### Performance
- **Mean Reward:** 0.62 +/- 0.41
- **Timesteps:** 250000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 128
- **Ent Coef:** 0.02
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 7: MiniGrid-Dynamic-Obstacles-5x5-v0
### Performance
- **Mean Reward:** -0.10 +/- 0.30
- **Timesteps:** 200000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 128
- **Ent Coef:** 0.03
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 8: MiniGrid-Dynamic-Obstacles-8x8-v0
### Performance
- **Mean Reward:** 0.00 +/- 0.00
- **Timesteps:** 500000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 256
- **Ent Coef:** 0.03
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 9: MiniGrid-Dynamic-Obstacles-16x16-v0
### Performance
- **Mean Reward:** 0.00 +/- 0.00
- **Timesteps:** 1000000

### Configuration
- **Learning Rate:** 0.0005
- **Batch Size:** 256
- **Ent Coef:** 0.05
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Problem 10: MiniGrid-FourRooms-v0
### Performance
- **Mean Reward:** 0.36 +/- 0.40
- **Timesteps:** 500000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 256
- **Ent Coef:** 0.05
- **Model:** Fresh plain MLP (no adapters)
- **Total Parameters:** 22732
- **Trainable Parameters:** 22732

## Summary: All Problems
| # | Environment | Mean Reward | Std Reward | Parameters |
|---|-------------|-------------|------------|------------|
| 1 | MiniGrid-Empty-5x5-v0 | 0.95 | 0.01 | 22732 |
| 2 | MiniGrid-Empty-Random-6x6-v0 | 0.93 | 0.06 | 22732 |
| 3 | MiniGrid-Empty-8x8-v0 | 0.00 | 0.00 | 22732 |
| 4 | MiniGrid-Empty-16x16-v0 | 0.97 | 0.00 | 22732 |
| 5 | MiniGrid-LavaGapS5-v0 | 0.95 | 0.01 | 22732 |
| 6 | MiniGrid-LavaGapS7-v0 | 0.62 | 0.41 | 22732 |
| 7 | MiniGrid-Dynamic-Obstacles-5x5-v0 | -0.10 | 0.30 | 22732 |
| 8 | MiniGrid-Dynamic-Obstacles-8x8-v0 | 0.00 | 0.00 | 22732 |
| 9 | MiniGrid-Dynamic-Obstacles-16x16-v0 | 0.00 | 0.00 | 22732 |
| 10 | MiniGrid-FourRooms-v0 | 0.36 | 0.40 | 22732 |
