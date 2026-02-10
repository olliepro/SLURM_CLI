from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slurm_cli.plot_gpu_usage_forecast import (  # noqa: E402
    JobRecord,
    NodeCapacity,
    parse_partition_names,
    partition_node_capacities,
    record_targets_partition,
)


def _record(
    *,
    state: str = "PENDING",
    requested_gpus: int = 1,
    node_expression: str | None = None,
    partitions: tuple[str, ...] = (),
) -> JobRecord:
    return JobRecord(
        job_id=123,
        state=state,
        requested_gpus=requested_gpus,
        allocated_gpus=0,
        start_time=datetime(2026, 2, 10, 12, 0, 0),
        end_time=datetime(2026, 2, 10, 13, 0, 0),
        time_limit_hours=1.0,
        run_time_hours=0.0,
        requested_cpus=1,
        requested_mem_mib=1024,
        requested_nodes=1,
        node_expression=node_expression,
        partition_names=partitions,
    )


def _capacity(node_name: str, partitions: tuple[str, ...]) -> NodeCapacity:
    return NodeCapacity(
        node_name=node_name,
        cpu=64,
        mem_mib=512000,
        gpus=4,
        cpu_alloc=0,
        mem_alloc_mib=0,
        gpu_alloc=0,
        partition_names=partitions,
    )


class ForecastPartitioningTests(unittest.TestCase):
    def test_parse_partition_names_normalizes_star_and_case(self) -> None:
        self.assertEqual(parse_partition_names(value="GPU*,Quad"), ("gpu", "quad"))

    def test_partition_node_capacities_filters_nodes(self) -> None:
        nodes = {
            "node-a": _capacity(node_name="node-a", partitions=("gpu", "quad")),
            "node-b": _capacity(node_name="node-b", partitions=("gpu",)),
        }
        filtered = partition_node_capacities(node_capacities=nodes, partition_name="quad")
        self.assertEqual(set(filtered.keys()), {"node-a"})

    def test_quad_inference_includes_large_gpu_jobs(self) -> None:
        record = _record(requested_gpus=4, partitions=())
        included = record_targets_partition(
            record=record,
            partition_name="quad",
            partition_node_names={"node-a"},
            host_cache={},
            infer_quad_large_gpu=True,
        )
        self.assertTrue(included)

    def test_quad_inference_excludes_small_gpu_jobs(self) -> None:
        record = _record(requested_gpus=3, partitions=())
        included = record_targets_partition(
            record=record,
            partition_name="quad",
            partition_node_names={"node-a"},
            host_cache={},
            infer_quad_large_gpu=True,
        )
        self.assertFalse(included)

    def test_running_node_membership_overrides_missing_partition_field(self) -> None:
        record = _record(state="RUNNING", requested_gpus=4, node_expression="node-[1]", partitions=())
        included = record_targets_partition(
            record=record,
            partition_name="quad",
            partition_node_names={"node001"},
            host_cache={"node-[1]": ["node001"]},
            infer_quad_large_gpu=False,
        )
        self.assertTrue(included)


if __name__ == "__main__":
    unittest.main()
