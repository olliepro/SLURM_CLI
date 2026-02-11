from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from slurm_cli.dash_logic import DashActionResult, DashJob  # noqa: E402
from slurm_cli.dash_ui import DashBoard  # noqa: E402
from slurm_cli.forecast_cli import ForecastPoint, ForecastSnapshot  # noqa: E402
from slurm_cli.forecast_core import ForecastStats  # noqa: E402


def _job(job_id: str, state: str) -> DashJob:
    return DashJob(
        job_id=job_id,
        state_compact=state,
        name=f"job-{job_id}",
        time_used="0:00",
        time_left="1:00",
        reason="Priority",
        node_list="c0318" if state == "R" else "",
        work_dir="workspace",
    )


def _stats() -> ForecastStats:
    return ForecastStats(
        active_gpu_jobs=0,
        running_gpu_jobs=0,
        pending_gpu_jobs=0,
        pending_with_start=0,
        pending_without_start=0,
        forecast_windows=0,
        degenerate_jobs=0,
        degenerate_extra_gpus=0,
        degenerate_nodes=0,
        degenerate_locked_gpus=0,
    )


def _snapshot(available: int, capacity: int, max_colocated: int) -> ForecastSnapshot:
    now = datetime(2026, 2, 11, 9, 0, 0)
    return ForecastSnapshot(
        generated_at=now,
        capacity=capacity,
        max_colocated_available_gpus=max_colocated,
        points=[ForecastPoint(offset_hours=0.0, available_gpus=available)],
        series_times=[now],
        series_available=[available],
        stats=_stats(),
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

    @patch("slurm_cli.dash_ui.join_job_via_remote")
    def test_join_uses_shared_remote_call(self, join_mock) -> None:
        join_mock.return_value = DashActionResult(
            ok=True,
            message="ok",
            affected_job_ids=["1"],
        )
        board = DashBoard(user_name="test")
        board.update_jobs(jobs=[_job("1", "R")])
        exit_code = board._join_from_ui()
        self.assertEqual(exit_code, 0)
        join_mock.assert_called_once_with(job=board.current_job(), editor=None)

    def test_title_includes_availability_and_max_colocated(self) -> None:
        board = DashBoard(user_name="test")
        title = board._title_with_availability(
            title="GPU Availability Forecast (8h, all GPUs)",
            snapshot=_snapshot(available=42, capacity=80, max_colocated=8),
        )
        self.assertEqual(
            title,
            "GPU Availability Forecast (8h, all GPUs) [42/80, max colocated=8]",
        )


if __name__ == "__main__":
    unittest.main()
