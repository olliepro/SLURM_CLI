from __future__ import annotations

import unittest
from unittest.mock import patch

from slurm_cli.pickers import ResourcePicker  # noqa: E402


class ResourcePickerTests(unittest.TestCase):
    def test_fallback_prompt_returns_selected_partition(self) -> None:
        picker = ResourcePicker(
            time_minutes=30,
            gpus=1,
            cpus=4,
            mem_gb=32,
            initial_partition=None,
            available_partitions=("debug-nextgen", "nextgen", "quad"),
        )
        with patch(
            "builtins.input",
            side_effect=["", "", "", "", "quad"],
        ):
            result = picker._fallback_prompt()
        self.assertEqual(result, ("00:30:00", 1, 4, "32G", "quad"))

    def test_fallback_prompt_keeps_auto_partition_by_default(self) -> None:
        picker = ResourcePicker(
            time_minutes=30,
            gpus=1,
            cpus=4,
            mem_gb=32,
            initial_partition=None,
            available_partitions=("debug-nextgen", "nextgen", "quad"),
        )
        with patch(
            "builtins.input",
            side_effect=["", "", "", "", ""],
        ):
            result = picker._fallback_prompt()
        self.assertEqual(result, ("00:30:00", 1, 4, "32G", None))


if __name__ == "__main__":
    unittest.main()
