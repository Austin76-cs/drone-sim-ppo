# Drone Sim v2

## Project Goal
PPO-based autonomous drone racing agent for the **AI Grand Prix** (Anduril's competition, virtual qualifier May-July 2026). This is v2, a simplified rewrite following the MonoRace paper architecture (2025 A2RL winner).

## Target Architecture (MonoRace pipeline)
1. **U-Net perception** - monocular camera to gate detection (not yet implemented)
2. **EKF state estimation** - fuse perception with IMU (not yet implemented)
3. **PPO control** - SB3 PPO policy outputs `[vz, roll, pitch, yaw_rate]` (Phase 1 - current)

**Current phase:** Phase 1 - PPO training on built-in MuJoCo state observations (no perception/EKF yet).

## Tech Stack
- Python 3.12, MuJoCo 3.3+, Stable-Baselines3, PyTorch, Gymnasium
- TOML config at `configs/default.toml`
- Training uses CUDA GPU (`device="cuda"` in PPO)

## Project Structure
```
configs/default.toml        # All hyperparams, reward weights, curriculum thresholds
assets/mjcf/quadrotor.xml   # MuJoCo quadrotor model
scripts/
  train.py                  # Main training entry point (PPO + SB3)
  visualize.py              # Render trained policy with MuJoCo viewer
  evaluate.py               # Evaluate checkpoints
  visual_test.py            # Quick visual sanity checks
src/dronesim/
  config.py                 # TOML -> dataclass config loading
  types.py                  # DroneState, GateSpec, RewardInfo, EpisodeMetrics
  envs/drone_race_env.py    # Gymnasium env: obs, step, reset, reward
  sim/
    env.py                  # MuJoCo physics wrapper (500Hz sim, 50Hz policy)
    attitude_controller.py  # PD attitude controller (internal, not learned)
    mixer.py                # Motor mixing from thrust/torque to rotor commands
  tasks/
    curriculum.py           # 4-stage curriculum: INTRO -> OFFSET -> SLALOM -> SPRINT
    rewards.py              # Reward components + gate passage detection (ray-plane crossing)
    termination.py          # Crash detection: flip, ground, altitude, spin, off-course
  training/
    callbacks.py            # TensorBoard logging (reward breakdown, curriculum stage)
tests/                      # pytest tests for config, env, rewards
```

## Key Design Decisions
- **Action space:** 4D continuous `[vz, roll, pitch, yaw_rate]` - an internal PD attitude controller converts these to motor commands. The policy does NOT output raw motor thrusts.
- **Sim/policy rate:** 500Hz physics, 50Hz policy (10 physics substeps per policy step)
- **Curriculum:** 4 stages with automatic promotion based on rolling success rate:
  - INTRO (2 gates, aligned) -> OFFSET (3 gates, lateral offsets) -> SLALOM (6 gates, alternating) -> SPRINT (10 gates, competition-realistic with angled normals)
- **Gate passage:** Ray-plane crossing check using previous + current position, not just proximity
- **VecNormalize:** Obs normalization on, reward normalization on during training, off during eval
- **Checkpoints:** Saved to `checkpoints/<run-name>/`, includes VecNormalize stats (`.pkl`)

## Training Commands
```bash
# Standard training
python scripts/train.py --run-name my_run

# Resume from checkpoint
python scripts/train.py --resume checkpoints/my_run/ppo_drone_100000_steps.zip --normalize checkpoints/my_run/vec_normalize.pkl --run-name my_run

# Debug mode (1 env, 50k steps, DummyVecEnv)
python scripts/train.py --debug

# Force a curriculum stage
python scripts/train.py --stage 3  # start at SPRINT

# Visualize trained policy
python scripts/visualize.py checkpoints/my_run/final_model.zip --normalize checkpoints/my_run/vec_normalize.pkl
```

## Predecessor: Drone Sim v1
Located at `../Drone Sim/drone2/`. Had custom SAC, world models, Warp-lang, scripted teacher. v2 simplifies: SB3 replaces custom SAC, numpy replaces torch in env, no Warp. Reuse v1's proven patterns for attitude controller, mixer, MJCF model, and curriculum design when relevant.

## Conventions
- Config changes go in `configs/default.toml`, loaded via `config.py` dataclasses
- Keep env code (numpy) separate from training code (PyTorch/SB3)
- Prefer working over perfect - iterate later
- Write clean, well-documented code (the developers are learning as they build)
