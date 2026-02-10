from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from slurm_cli.remote_access import (
    RemoteOpenRequest,
    normalize_editor_token,
    normalize_osc_host,
    open_remote_target,
    resolve_editor_command,
)


class RemoteAccessTests(unittest.TestCase):
    def test_normalize_osc_alias(self) -> None:
        self.assertEqual(normalize_osc_host(host="cardinal"), "osc-cardinal-login")

    def test_normalize_editor_alias(self) -> None:
        self.assertEqual(normalize_editor_token(editor="vscode"), "code")
        self.assertEqual(normalize_editor_token(editor="vscodium"), "codium")

    def test_direct_mode_dry_run_uses_explicit_editor(self) -> None:
        result = open_remote_target(
            request=RemoteOpenRequest(
                host="c0318",
                work_dir=Path("."),
                editor="cursor",
                dry_run=True,
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.command[0], "cursor")

    @patch("slurm_cli.remote_access.shutil.which")
    def test_resolve_editor_command_uses_auto_detect(self, which_mock) -> None:
        which_mock.side_effect = lambda token: "/usr/bin/cursor" if token == "cursor" else None
        with patch.dict("os.environ", {}, clear=True):
            resolved = resolve_editor_command(preferred=None)
        assert resolved is not None
        self.assertEqual(resolved.command, "cursor")
        self.assertEqual(resolved.source, "auto")

    @patch("slurm_cli.remote_access.shutil.which", return_value=None)
    def test_open_remote_target_errors_when_no_editor_found(self, _which_mock) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = open_remote_target(
                request=RemoteOpenRequest(
                    host="c0318",
                    work_dir=Path("."),
                    dry_run=True,
                )
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.command, [])
        self.assertIn("No supported editor CLI found", result.message)

    @patch("slurm_cli.remote_access.subprocess.run")
    def test_open_remote_target_executes_direct_mode_once(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["cursor"],
            returncode=0,
            stdout="",
            stderr="",
        )
        result = open_remote_target(
            request=RemoteOpenRequest(
                host="pitzer",
                work_dir=Path("."),
                editor="cursor",
                dry_run=False,
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0][0], "cursor")


if __name__ == "__main__":
    unittest.main()
