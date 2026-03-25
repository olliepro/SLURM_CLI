from __future__ import annotations

import unittest
from unittest.mock import patch

from slurm_cli.slurm_backend import build_sbatch, build_srun  # noqa: E402


class SlurmBackendPartitionTests(unittest.TestCase):
    @patch("slurm_cli.slurm_backend.recommend_partition", return_value="debug")
    def test_build_srun_auto_adds_selected_partition(self, recommend_mock) -> None:
        command = build_srun(
            gpus=1,
            cpus=8,
            time_str="00:30:00",
            account="P123",
            shell="bash",
            mem="32G",
        )
        self.assertIn("--partition=debug", command)
        request = recommend_mock.call_args.kwargs["request"]
        self.assertEqual(request.time_minutes, 30)

    @patch("slurm_cli.slurm_backend.recommend_partition")
    def test_build_sbatch_respects_explicit_partition_override(
        self, recommend_mock
    ) -> None:
        command = build_sbatch(
            gpus=2,
            cpus=16,
            time_str="02:00:00",
            account="P123",
            mem="64G",
            email=None,
            job_name="demo",
            partition="gpu-exp",
        )
        self.assertIn("--partition=gpu-exp", command)
        recommend_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
