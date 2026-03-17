from __future__ import annotations

import unittest

from dronesim.config import TaskConfig
from dronesim.tasks.curriculum import CurriculumStage, EpisodeSummary, StageController


class TestCurriculum(unittest.TestCase):
    def test_lock_stage_prevents_advancement(self):
        controller = StageController(TaskConfig())
        controller.lock_to_stage(CurriculumStage.INTRO)
        for _ in range(20):
            controller.record_episode(
                EpisodeSummary(
                    stage=CurriculumStage.INTRO,
                    success=True,
                    terminated=False,
                    truncated=False,
                    crash_type="success",
                    completion=1.0,
                    score=1.0,
                    gates_cleared=3,
                    steps=100,
                )
            )
        self.assertEqual(controller.stage, CurriculumStage.INTRO)


if __name__ == "__main__":
    unittest.main()
