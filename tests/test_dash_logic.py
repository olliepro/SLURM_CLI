from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slurm_cli.dash_logic import (  # noqa: E402
    DASH_PENDING,
    DASH_RUNNING,
    DashJob,
    cancel_dash_jobs,
    fetch_dash_jobs,
    join_job_via_remote,
    resolve_primary_host,
)


class DashLogicTests(unittest.TestCase):
    @patch("slurm_cli.dash_logic.subprocess.check_output")
    def test_fetch_dash_jobs_parses_and_sorts(self, check_output_mock) -> None:
        check_output_mock.return_value = (
            "100\tPD\tpending\t0:00\t1:00\tPriority\t\t/work/p\n"
            "101\tR\trunning\t0:10\t0:50\tNone\tc0318\t/work/r\n"
        )
        jobs = fetch_dash_jobs(user_name="testuser")
        self.assertEqual([job.job_id for job in jobs], ["101", "100"])
        self.assertEqual(jobs[0].state_compact, DASH_RUNNING)
        self.assertEqual(jobs[1].state_compact, DASH_PENDING)

    @patch("slurm_cli.dash_logic.subprocess.check_output")
    def test_resolve_primary_host_expands_bracket_nodelist(self, check_output_mock) -> None:
        check_output_mock.return_value = "c0821\nc0822\n"
        host = resolve_primary_host(node_list="c[0821-0822]")
        self.assertEqual(host, "c0821")
        check_output_mock.assert_called_once_with(
            ["scontrol", "show", "hostnames", "c[0821-0822]"],
            text=True,
        )

    @patch("slurm_cli.dash_logic.subprocess.run")
    def test_cancel_dash_jobs_calls_scancel_once(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["scancel"],
            returncode=0,
            stdout="",
            stderr="",
        )
        result = cancel_dash_jobs(job_ids=["12", "34", "12"])
        self.assertTrue(result.ok)
        run_mock.assert_called_once_with(
            ["scancel", "12", "34"],
            capture_output=True,
            text=True,
        )

    @patch("slurm_cli.dash_logic.resolve_primary_host")
    @patch("slurm_cli.dash_logic.subprocess.run")
    def test_join_job_via_remote_calls_zsh_remote(self, run_mock, host_mock) -> None:
        host_mock.return_value = "c0318"
        run_mock.return_value = subprocess.CompletedProcess(
            args=["zsh"],
            returncode=0,
            stdout="",
            stderr="",
        )
        job = DashJob(
            job_id="123",
            state_compact="R",
            name="demo",
            time_used="0:10",
            time_left="0:50",
            reason="None",
            node_list="c0318",
            work_dir="/tmp",
        )
        result = join_job_via_remote(job=job)
        self.assertTrue(result.ok)
        run_mock.assert_called_once_with(
            ["zsh", "-ic", "remote c0318"],
            capture_output=True,
            text=True,
            cwd="/tmp",
        )

    @patch("slurm_cli.dash_logic.subprocess.run")
    def test_join_rejects_pending_job(self, run_mock) -> None:
        job = DashJob(
            job_id="124",
            state_compact="PD",
            name="demo",
            time_used="0:00",
            time_left="1:00",
            reason="Priority",
            node_list="",
            work_dir="",
        )
        result = join_job_via_remote(job=job)
        self.assertFalse(result.ok)
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
