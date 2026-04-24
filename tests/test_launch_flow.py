from __future__ import annotations

import argparse
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from slurm_cli.config_store import Config  # noqa: E402
from slurm_cli.dash_ui import DashboardCommandResult  # noqa: E402
from slurm_cli.interactive_slurm import run_dash_command  # noqa: E402
from slurm_cli.launch_flow import (  # noqa: E402
    LaunchFlowResult,
    ResourceSelection,
    TimeoutSelection,
    build_default_launch_namespace,
    run_dashboard_launch_flow,
    run_launch_flow,
    run_terminal_mode,
    run_vscode_mode,
    resolve_resources,
)


def _resources() -> ResourceSelection:
    return ResourceSelection(
        time_str="00:30:00",
        time_minutes=30,
        gpus=1,
        cpus=4,
        mem_str="50G",
        partition="debug",
    )


def _timeout() -> TimeoutSelection:
    return TimeoutSelection(mode="impatient", limit_seconds=1800, email="")


class LaunchFlowTests(unittest.TestCase):
    @patch("slurm_cli.launch_flow.submit_batch_job", return_value="321")
    @patch("slurm_cli.launch_flow.resolve_resources", return_value=_resources())
    @patch("slurm_cli.launch_flow.resolve_account")
    @patch("slurm_cli.launch_flow.resolve_timeout")
    @patch("slurm_cli.launch_flow.resolve_ui_mode")
    @patch("slurm_cli.launch_flow.Config.load")
    def test_run_dashboard_launch_flow_submits_background_batch(
        self,
        cfg_load_mock,
        ui_mode_mock,
        timeout_mock,
        account_mock,
        _resources_mock,
        submit_mock,
    ) -> None:
        cfg = Config()
        cfg.save = MagicMock()
        cfg_load_mock.return_value = cfg
        account_mock.return_value = MagicMock(account="PTEST")
        result = run_dashboard_launch_flow(args=build_default_launch_namespace())
        self.assertTrue(result.ok)
        self.assertEqual(result.message, "Submitted background allocation job 321.")
        submit_mock.assert_called_once_with(
            gpus=1,
            cpus=4,
            time_str="00:30:00",
            account="PTEST",
            mem="50G",
            email=None,
            job_name="slurmcli-dashboard",
            partition="debug",
        )
        ui_mode_mock.assert_not_called()
        timeout_mock.assert_not_called()
        self.assertEqual(cfg.last_time, "00:30:00")
        self.assertEqual(cfg.last_mem, "50G")
        self.assertEqual(cfg.last_gpus, 1)
        self.assertEqual(cfg.last_cpus, 4)
        self.assertEqual(cfg.last_partition, "debug")

    @patch("slurm_cli.launch_flow.list_partition_names")
    @patch("slurm_cli.launch_flow.ResourcePicker")
    def test_resolve_resources_seeds_picker_from_cached_defaults(
        self,
        picker_mock,
        partitions_mock,
    ) -> None:
        partitions_mock.return_value = ("debug", "gpu")
        picker_mock.return_value.run.return_value = (
            "02:00:00",
            2,
            16,
            "96G",
            "gpu",
        )
        cfg = Config(
            last_time="02:00:00",
            last_gpus=2,
            last_cpus=16,
            last_mem="96G",
            last_partition="gpu",
        )
        resources = resolve_resources(args=build_default_launch_namespace(), cfg=cfg)
        self.assertEqual(resources.partition, "gpu")
        picker_mock.assert_called_once_with(
            time_minutes=120,
            gpus=2,
            cpus=16,
            mem_gb=96,
            initial_partition="gpu",
            available_partitions=("debug", "gpu"),
        )

    @patch("slurm_cli.launch_flow.subprocess.run")
    def test_run_terminal_mode_embedded_returns_to_caller(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["srun"],
            returncode=0,
        )
        result = run_terminal_mode(
            resources=_resources(),
            account="PTEST",
            shell="bash",
            timeout=_timeout(),
            dry_run=False,
            embedded=True,
        )
        self.assertTrue(result.ok)
        self.assertIn("returned to dashboard", result.message)
        run_mock.assert_called_once()

    @patch("slurm_cli.launch_flow.open_vscode_on_host", return_value=0)
    @patch("slurm_cli.launch_flow.wait_for_node", return_value="c0318")
    @patch("slurm_cli.launch_flow.start_allocation_background")
    def test_run_vscode_mode_embedded_returns_after_editor_open(
        self,
        start_mock,
        _wait_mock,
        open_mock,
    ) -> None:
        proc = MagicMock()
        start_mock.return_value = (proc, "123")
        result = run_vscode_mode(
            resources=_resources(),
            account="PTEST",
            timeout=_timeout(),
            dry_run=False,
            embedded=True,
        )
        self.assertTrue(result.ok)
        self.assertIn("Opened remote editor", result.message)
        open_mock.assert_called_once_with(hostname="c0318")

    @patch("slurm_cli.launch_flow.resolve_timeout", return_value=_timeout())
    @patch("slurm_cli.launch_flow.resolve_ui_mode", return_value="terminal")
    @patch("slurm_cli.launch_flow.resolve_resources", return_value=_resources())
    @patch("slurm_cli.launch_flow.resolve_account")
    @patch("slurm_cli.launch_flow.sys.stdin.isatty", return_value=True)
    @patch("slurm_cli.launch_flow._confirm", return_value=False)
    def test_run_launch_flow_returns_canceled_result_for_embedded_cancel(
        self,
        _confirm_mock,
        _isatty_mock,
        account_mock,
        _resources_mock,
        _ui_mode_mock,
        _timeout_mock,
    ) -> None:
        account_mock.return_value = MagicMock(account="PTEST")
        result = run_launch_flow(
            args=build_default_launch_namespace(),
            embedded=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.message, "Canceled.")
        self.assertEqual(result.exit_code, 1)

    @patch("slurm_cli.launch_flow.os.execvp")
    @patch("slurm_cli.launch_flow._confirm", return_value=True)
    def test_run_terminal_mode_cli_uses_execvp(
        self,
        _confirm_mock,
        execvp_mock,
    ) -> None:
        execvp_mock.side_effect = SystemExit(0)
        with self.assertRaises(SystemExit):
            run_terminal_mode(
                resources=_resources(),
                account="PTEST",
                shell="bash",
                timeout=_timeout(),
                dry_run=False,
                embedded=False,
            )
        execvp_mock.assert_called_once()

    @patch("slurm_cli.interactive_slurm.run_dashboard_launch_flow")
    @patch("slurm_cli.interactive_slurm.run_dash_dashboard")
    def test_run_dash_command_uses_dashboard_batch_submit(
        self,
        dashboard_mock,
        launch_mock,
    ) -> None:
        dashboard_mock.side_effect = [
            DashboardCommandResult(action="launch"),
            DashboardCommandResult(action="quit"),
        ]
        launch_mock.return_value = LaunchFlowResult(
            ok=True,
            message="Submitted background allocation job 321.",
        )
        run_dash_command(args=argparse.Namespace(user="alice", editor=None))
        launch_mock.assert_called_once()
        second_dashboard_call = dashboard_mock.call_args_list[1]
        self.assertEqual(
            second_dashboard_call.kwargs["initial_status_message"],
            "Submitted background allocation job 321.",
        )


if __name__ == "__main__":
    unittest.main()
