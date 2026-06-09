# Continual Learning Results - MoRAM Implementation

## Stage 1: MiniGrid-Empty-5x5-v0
### Performance
- **Mean Reward:** 0.95 +/- 0.01
- **Timesteps:** 40000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **MoRAM Ranks Added (This Stage):** 0
- **Total Active MoRAM Ranks:** 0

## Stage 2: MiniGrid-Empty-8x8-v0
### Performance
- **Mean Reward:** 0.96 +/- 0.00
- **Timesteps:** 80000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **MoRAM Ranks Added (This Stage):** 16
- **Total Active MoRAM Ranks:** 16

## Stage 3: MiniGrid-Empty-16x16-v0
### Performance
- **Mean Reward:** 0.97 +/- 0.00
- **Timesteps:** 300000

### Configuration
- **Learning Rate:** 0.0005
- **Batch Size:** 128
- **Ent Coef:** 0.05
- **MoRAM Ranks Added (This Stage):** 16
- **Total Active MoRAM Ranks:** 32

## Stage 4: MiniGrid-Empty-Random-6x6-v0
### Performance
- **Mean Reward:** 0.96 +/- 0.02
- **Timesteps:** 60000

### Configuration
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Ent Coef:** 0.01
- **MoRAM Ranks Added (This Stage):** 16
- **Total Active MoRAM Ranks:** 48

## Final Evaluation (Catastrophic Forgetting Check)
| Stage | Environment | Mean Reward | Std Reward |
|-------|-------------|-------------|------------|
| 1 | MiniGrid-Empty-5x5-v0 | 0.92 | 0.06 |
| 2 | MiniGrid-Empty-8x8-v0 | 0.96 | 0.00 |
| 3 | MiniGrid-Empty-16x16-v0 | 0.97 | 0.00 |
| 4 | MiniGrid-Empty-Random-6x6-v0 | 0.96 | 0.02 |
