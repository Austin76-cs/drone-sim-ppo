# Plan: Autonomous Drone Racing Pipeline for AI Grand Prix

## What I Need

Plan and build an autonomous drone racing system that flies a quadrotor through gates as fast as possible using only camera and IMU input. The competition drone has a single FPV camera, IMU, no GPS/position data.

## Architecture

The system follows the MonoRace architecture (2025 A2RL competition winner). Three stages in a loop:

1. **Perception**: U-Net CNN segments gates from camera images -> detect gate corners -> PnP algorithm estimates 3D gate positions relative to drone
2. **State Estimation**: Extended Kalman Filter fuses gate detections with IMU data to continuously track drone position, velocity, and orientation
3. **Control**: PPO-trained MLP takes the state estimate and outputs motor commands

Camera frame -> find gates -> estimate state -> RL picks actions -> repeat.

## Build Order

Build and validate each stage independently before connecting them:

### Phase 1: PPO on built-in state observations -- COMPLETE
Get PPO working directly on MuJoCo's built-in state observations (skip perception and state estimation entirely). This is the baseline that proves the RL training loop works.

**Status:** Done. PPO trains through a 4-stage curriculum (INTRO -> OFFSET -> SLALOM -> SPRINT). Training runs on CUDA GPU. Checkpoints, VecNormalize stats, and TensorBoard logging all working.

### Phase 2: U-Net gate segmentation -- NOT STARTED
Build the U-Net gate segmentation network. Write a data generator that renders frames from the sim and auto-generates ground truth masks. Train and validate the segmentation.

**U-Net**: Lightweight, ~6 levels, small filter counts. Must be fast since it runs every frame.

### Phase 3: Extended Kalman Filter -- NOT STARTED
Build the EKF. Validate it against sim ground truth to tune noise parameters.

**EKF state**: position(3), quaternion(4), velocity(3), gyro_bias(3), accel_bias(3) = 16D

### Phase 4: Full pipeline integration -- NOT STARTED
Connect all three stages with a Gymnasium wrapper. Camera -> segmentation -> corner detection -> PnP -> EKF -> PPO policy -> motor commands.

### Phase 5: Speed optimization -- NOT STARTED
Iterate on reward function and hyperparameters for speed.

## Tech Stack

- Python 3.12, PyTorch, Stable-Baselines3, OpenCV
- MuJoCo 3.3+ simulator (originally PyFlyt, switched to MuJoCo for more control)
- Gymnasium API
- TensorBoard for logging all training runs
- CUDA GPU for training (3080 Ti on training machine)

## Key Specs

**PPO hyperparameters** (in `configs/default.toml`): lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, net_arch=[256, 256]

**Action space**: 4D continuous [vz, roll, pitch, yaw_rate] — internal PD attitude controller converts to motor commands

**Sim rates**: 500Hz physics, 50Hz policy (10 substeps per policy step)

**Reward function components** (modular, weights in config): gate proximity, gate passage bonus, progress toward next gate, velocity alignment, time penalty, collision penalty, control effort, alive bonus

**Curriculum**: 4 stages with automatic promotion based on rolling success rate:
- INTRO (2 gates, aligned) -> threshold 0.70
- OFFSET (3 gates, lateral offsets) -> threshold 0.64
- SLALOM (6 gates, alternating) -> threshold 0.58
- SPRINT (10 gates, competition-realistic, angled normals) -> threshold 0.50

## Important Constraints

- Build one phase at a time. Don't build everything at once.
- Make the reward function modular and easy to swap components in and out.
- Log everything to TensorBoard.
- Write clean, well-documented code. I'm learning RL and need to understand what each part does.
- Prioritize working over perfect. We iterate later.

## Context

Entering the AI Grand Prix (Anduril's autonomous drone racing competition). Virtual qualifier is May-July 2026. New to RL but using Claude to handle implementation. This build will be ported to the competition sim when it releases.

## References

- MonoRace paper: arxiv.org/abs/2601.15222
- Swift paper: nature.com/articles/s41586-023-06419-4
- Stable-Baselines3: github.com/DLR-RM/stable-baselines3
