# Drone Racing AI — AI Grand Prix Entry

Autonomous drone racing agent built for the [AI Grand Prix](https://thedroneracingleague.com/ai-grand-prix/) (Anduril / DCL). PPO-trained policy navigates a quadrotor through gate courses at speed using only onboard sensors.

Follows the [MonoRace](https://arxiv.org/abs/2601.15222) architecture (2025 A2RL competition winner).

## Architecture

| Stage | Description | Status |
|-------|-------------|--------|
| **PPO Control** | MLP policy outputs `[thrust, roll_rate, pitch_rate, yaw_rate]`, internal PD attitude controller converts to motor commands | Active — training through curriculum |
| **U-Net Perception** | Monocular camera gate segmentation + PnP pose estimation | Planned |
| **EKF State Estimation** | Fuse gate detections with IMU for full state tracking | Planned |

Currently training Phase 1: PPO on simulator ground-truth state observations (no perception/EKF yet).

## Training

The agent learns through a 4-stage curriculum with automatic promotion:

1. **INTRO** — 2 aligned gates
2. **OFFSET** — 3 gates with lateral offsets
3. **SLALOM** — 6 gates, alternating directions
4. **SPRINT** — 5–15 gates, competition-realistic with angled normals, variable spacing

Gate passage uses ray-plane crossing detection with drone-radius margin to ensure physically valid passes.

### Quick Start

```bash
# Install
pip install -e .

# Train from scratch
python scripts/train.py --run-name my_run

# Resume from checkpoint
python scripts/train.py --resume checkpoints/my_run/best_model.zip \
    --normalize checkpoints/my_run/vec_normalize.pkl --run-name my_run

# Visualize a trained policy
python scripts/visualize.py checkpoints/my_run/final_model.zip \
    --normalize checkpoints/my_run/vec_normalize.pkl

# Run stress test (14 diverse course scenarios)
python scripts/stress_test.py checkpoints/my_run/best_model.zip \
    --normalize checkpoints/my_run/vec_normalize.pkl
```

## Project Structure

```
configs/default.toml           # Hyperparameters, reward weights, curriculum thresholds
assets/mjcf/quadrotor.xml      # MuJoCo quadrotor model
scripts/
    train.py                   # PPO training with SB3
    visualize.py               # Render policy with MuJoCo viewer
    evaluate.py                # Checkpoint evaluation
    stress_test.py             # 14-scenario course stress test
src/dronesim/
    envs/drone_race_env.py     # Gymnasium environment
    sim/
        env.py                 # MuJoCo physics (500Hz sim, 100Hz policy)
        attitude_controller.py # PD attitude controller
        mixer.py               # Motor mixing
    tasks/
        curriculum.py          # 4-stage curriculum system
        rewards.py             # Modular reward components + gate detection
        termination.py         # Crash/boundary detection
    training/
        callbacks.py           # TensorBoard logging
```

## Tech Stack

- Python 3.12, MuJoCo 3.3+, Stable-Baselines3, PyTorch, Gymnasium
- CUDA GPU training (RTX 3080 Ti)
- TensorBoard for experiment tracking

## Competition Timeline

- **Virtual Qualifier 1**: May 2026
- **Virtual Qualifier 2**: June 2026
- **Physical Qualifier**: September 2026, Southern California
- **Finals**: November 2026, Ohio

## References

- [MonoRace: Autonomous Racing with Monocular Vision](https://arxiv.org/abs/2601.15222)
- [Swift: Autonomous Drone Racing (Nature 2023)](https://nature.com/articles/s41586-023-06419-4)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
