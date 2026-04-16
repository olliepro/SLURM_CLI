from __future__ import annotations

import unittest

from slurm_cli.forecast_cli import default_primary_partition


class ForecastCliTests(unittest.TestCase):
    def test_default_primary_partition_uses_gpu_on_cardinal(self) -> None:
        self.assertEqual(default_primary_partition(cluster_name="cardinal"), "gpu")

    def test_default_primary_partition_leaves_other_clusters_unscoped(self) -> None:
        self.assertIsNone(default_primary_partition(cluster_name="ascend"))
        self.assertIsNone(default_primary_partition(cluster_name="pitzer"))
        self.assertIsNone(default_primary_partition(cluster_name=None))


if __name__ == "__main__":
    unittest.main()
