from __future__ import annotations

import unittest
from datetime import datetime
from typing import Any, cast
from unittest.mock import patch

from slurm_cli.dash_logic import DashActionResult, DashJob  # noqa: E402
from slurm_cli.dash_ui import (  # noqa: E402
    DashBoard,
    DashboardCommandResult,
    relocation_cluster_options,
)
from slurm_cli.forecast_cli import ForecastPoint, ForecastSnapshot  # noqa: E402
from slurm_cli.forecast_core import ForecastStats  # noqa: E402


def _job(job_id: str, state: str) -> DashJob:
    return DashJob(
        job_id=job_id,
        state_compact=state,
        name=f"job-{job_id}",
        time_used="0:00",
        time_left="1:00",
        start_time=datetime(2026, 2, 11, 10, 0, 0),
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


class FakeScreen:
    def __init__(self) -> None:
        self.erased = False
        self.refreshed = False
        self.rows: list[tuple[int, int, str, int]] = []

    def getmaxyx(self) -> tuple[int, int]:
        return (24, 80)

    def erase(self) -> None:
        self.erased = True

    def refresh(self) -> None:
        self.refreshed = True

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        self.rows.append((y, x, text, attr))


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
        board._join_from_ui()
        self.assertTrue(board.status_message.startswith("OK"))
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

    def test_fallback_launch_command_returns_launch_action(self) -> None:
        board = DashBoard(user_name="test")
        result = board._handle_fallback_command(raw="n")
        self.assertEqual(result, DashboardCommandResult(action="launch"))

    def test_escape_on_main_dashboard_quits(self) -> None:
        board = DashBoard(user_name="test")
        result = board._handle_key(stdscr=cast(Any, FakeScreen()), key=27)
        self.assertEqual(result, DashboardCommandResult(action="quit"))

    def test_relocation_picker_uses_full_screen(self) -> None:
        board = DashBoard(user_name="test")
        screen = FakeScreen()
        board._draw_relocation_picker(
            stdscr=cast(Any, screen),
            options=("ascend", "cardinal"),
            selected=0,
        )
        self.assertTrue(screen.erased)
        self.assertTrue(screen.refreshed)
        self.assertTrue(any("Esc return to dashboard" in row[2] for row in screen.rows))

    def test_relocation_options_exclude_current_cluster(self) -> None:
        self.assertEqual(
            relocation_cluster_options(current_cluster="ascend"),
            ("pitzer", "cardinal"),
        )

    @patch("slurm_cli.dash_ui.open_remote_target")
    @patch("slurm_cli.dash_ui.detect_cluster_name", return_value="pitzer")
    @patch("builtins.input", return_value="1")
    def test_fallback_relocate_uses_shared_remote_call(
        self, _input_mock, _cluster_mock, remote_mock
    ) -> None:
        remote_mock.return_value.ok = True
        remote_mock.return_value.message = "Opened remote editor"
        board = DashBoard(user_name="test", editor_command="cursor")
        result = board._handle_fallback_command(raw="r")
        self.assertIsNone(result)
        self.assertIn("Ascend", board.status_message)
        call_request = remote_mock.call_args.kwargs["request"]
        self.assertEqual(call_request.host, "ascend")
        self.assertEqual(call_request.editor, "cursor")

    @patch("slurm_cli.dash_ui.join_job_via_remote")
    def test_fallback_join_does_not_exit_dashboard(self, join_mock) -> None:
        join_mock.return_value = DashActionResult(
            ok=True,
            message="ok",
            affected_job_ids=["1"],
        )
        board = DashBoard(user_name="test")
        board.update_jobs(jobs=[_job("1", "R")])
        result = board._handle_fallback_command(raw="v")
        self.assertIsNone(result)
        self.assertTrue(board.status_message.startswith("OK"))


if __name__ == "__main__":
    unittest.main()
