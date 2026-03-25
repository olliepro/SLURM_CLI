from __future__ import annotations

import unittest

from slurm_cli.partition_policy import (  # noqa: E402
    PartitionRequest,
    parse_cluster_name,
    recommend_partition,
)


class PartitionPolicyTests(unittest.TestCase):
    def test_parse_cluster_name_normalizes_case(self) -> None:
        cluster_name = parse_cluster_name(
            config_text="ClusterName = Cardinal\nPriorityType = priority/multifactor\n"
        )
        self.assertEqual(cluster_name, "cardinal")

    def test_ascend_uses_debug_nextgen_for_short_single_gpu(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=1, cpus=8, time_minutes=45, mem_str="32G"),
            cluster_name="ascend",
        )
        self.assertEqual(partition_name, "debug-nextgen")

    def test_ascend_uses_quad_for_four_gpu_request(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=4, cpus=8, time_minutes=90, mem_str="32G"),
            cluster_name="ascend",
        )
        self.assertEqual(partition_name, "quad")

    def test_cardinal_uses_debug_for_short_single_gpu(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=1, cpus=8, time_minutes=60, mem_str="32G"),
            cluster_name="cardinal",
        )
        self.assertEqual(partition_name, "debug")

    def test_cardinal_uses_gpu_for_multi_gpu_request(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=2, cpus=8, time_minutes=30, mem_str="32G"),
            cluster_name="cardinal",
        )
        self.assertEqual(partition_name, "gpu")

    def test_pitzer_uses_standard_debug_partition_when_cpu_fit_is_small(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=2, cpus=32, time_minutes=45, mem_str="32G"),
            cluster_name="pitzer",
        )
        self.assertEqual(partition_name, "gpudebug")

    def test_pitzer_uses_exp_partition_when_cpu_count_requires_it(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=2, cpus=48, time_minutes=120, mem_str="32G"),
            cluster_name="pitzer",
        )
        self.assertEqual(partition_name, "gpu-exp")

    def test_pitzer_uses_quad_partition_for_three_gpu_request(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=3, cpus=32, time_minutes=45, mem_str="32G"),
            cluster_name="pitzer",
        )
        self.assertEqual(partition_name, "gpudebug-quad")

    def test_returns_none_for_cpu_only_request(self) -> None:
        partition_name = recommend_partition(
            request=PartitionRequest(gpus=0, cpus=8, time_minutes=30, mem_str="32G"),
            cluster_name="ascend",
        )
        self.assertIsNone(partition_name)


if __name__ == "__main__":
    unittest.main()
