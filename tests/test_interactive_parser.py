from __future__ import annotations

import contextlib
import io
import unittest

from slurm_cli.interactive_slurm import (  # noqa: E402
    parse_dash_args,
    parse_launch_args,
    parse_search_args,
)


class SearchParserTests(unittest.TestCase):
    def _assert_parse_fails(self, argv: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_search_args(argv=argv)

    def test_rejects_zero_max_gpus(self) -> None:
        self._assert_parse_fails(argv=["--max-gpus", "0"])

    def test_rejects_malformed_time(self) -> None:
        self._assert_parse_fails(argv=["--max-time", "bad-time"])

    def test_rejects_min_time_greater_than_max_time(self) -> None:
        self._assert_parse_fails(
            argv=[
                "--max-time",
                "01:00:00",
                "--min-time",
                "02:00:00",
                "--max-gpus",
                "4",
                "--min-gpus",
                "1",
            ]
        )

    def test_rejects_min_gpus_greater_than_max_gpus(self) -> None:
        self._assert_parse_fails(argv=["--max-gpus", "2", "--min-gpus", "4"])

    def test_parses_valid_search_flags(self) -> None:
        args = parse_search_args(
            argv=[
                "--max-time",
                "04:00:00",
                "--min-time",
                "00:30:00",
                "--max-gpus",
                "4",
                "--min-gpus",
                "1",
                "--yes",
            ]
        )
        self.assertEqual(args.max_time_minutes, 240)
        self.assertEqual(args.min_time_minutes, 30)
        self.assertEqual(args.max_gpus, 4)
        self.assertEqual(args.min_gpus, 1)
        self.assertTrue(args.yes)

    def test_parses_dash_without_flags(self) -> None:
        args = parse_dash_args(argv=[])
        self.assertEqual(args.user, None)
        self.assertEqual(args.editor, None)
        self.assertFalse(hasattr(args, "remote_mode"))

    def test_parses_dash_with_user_flag(self) -> None:
        args = parse_dash_args(argv=["--user", "alice"])
        self.assertEqual(args.user, "alice")

    def test_parses_dash_with_editor_flag(self) -> None:
        args = parse_dash_args(argv=["--editor", "cursor"])
        self.assertEqual(args.editor, "cursor")

    def test_launch_parser_unchanged(self) -> None:
        args = parse_launch_args(argv=["--shell", "bash"])
        self.assertEqual(args.shell, "bash")


if __name__ == "__main__":
    unittest.main()
