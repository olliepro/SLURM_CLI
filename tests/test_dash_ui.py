from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slurm_cli.dash_logic import DashJob  # noqa: E402
from slurm_cli.dash_ui import DashBoard  # noqa: E402


def _job(job_id: str, state: str) -> DashJob:
    return DashJob(
        job_id=job_id,
        state_compact=state,
        name=f"job-{job_id}",
        time_used="0:00",
        time_left="1:00",
        reason="Priority",
        node_list="c0318" if state == "R" else "",
        work_dir="/tmp",
    )


class DashUiStateTests(unittest.TestCase):
    def test_selection_toggle_add_remove(self) -> None:
        board = DashBoard(user_name="test")
        board.update_jobs(jobs=[_job("1", "R"), _job("2", "PD")])
        board.toggle_selected_current()
        self.assertEqual(board.selected_job_ids, {"1"})
        board.toggle_selected_current()
        self.assertEqual(board.selected_job_ids, set())

    def test_focus_retains_job_id_after_refresh(self) -> None:
        board = DashBoard(user_name="test")
        board.update_jobs(jobs=[_job("1", "R"), _job("2", "PD")])
        board.focus_index = 1
        board.update_jobs(jobs=[_job("2", "PD"), _job("1", "R")])
        current = board.current_job()
        assert current is not None
        self.assertEqual(current.job_id, "2")

    def test_join_eligibility_gated_by_state(self) -> None:
        board = DashBoard(user_name="test")
        board.update_jobs(jobs=[_job("1", "PD")])
        self.assertFalse(board.can_join_current())
        board.update_jobs(jobs=[_job("1", "R")])
        self.assertTrue(board.can_join_current())


if __name__ == "__main__":
    unittest.main()
