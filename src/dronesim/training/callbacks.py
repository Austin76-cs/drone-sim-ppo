from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from dronesim.tasks.curriculum import StageController


class RewardBreakdownCallback(BaseCallback):
    """Log individual reward components to TensorBoard on episode end."""

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            if "reward_gate_proximity" in info:
                self.logger.record("reward/gate_proximity", info["reward_gate_proximity"])
                self.logger.record("reward/gate_passage", info["reward_gate_passage"])
                self.logger.record("reward/progress", info["reward_progress"])
                self.logger.record("reward/velocity_alignment", info["reward_velocity_alignment"])
                if "reward_lateral_velocity_penalty" in info:
                    self.logger.record("reward/lateral_velocity_penalty", info["reward_lateral_velocity_penalty"])
                if "reward_attitude_stability" in info:
                    self.logger.record("reward/attitude_stability", info["reward_attitude_stability"])
                if "reward_angular_rate_stability" in info:
                    self.logger.record("reward/angular_rate_stability", info["reward_angular_rate_stability"])
                self.logger.record("reward/control_effort", info["reward_control_effort"])
            if "gates_cleared" in info:
                self.logger.record("episode/gates_cleared", info["gates_cleared"])
            if "completion" in info:
                self.logger.record("episode/completion", info["completion"])
            if "crash_type" in info:
                crash = info["crash_type"]
                self.logger.record("episode/crash_type", crash)
        return True


class CurriculumCallback(BaseCallback):
    """Synchronize curriculum stage across parallel envs and log to TensorBoard."""

    def __init__(self, stage_controller: StageController, verbose: int = 0):
        super().__init__(verbose)
        self.stage_controller = stage_controller

    def _on_step(self) -> bool:
        self.logger.record("curriculum/stage", int(self.stage_controller.stage))
        self.logger.record("curriculum/mastery", self.stage_controller.mastery)
        return True
