from __future__ import annotations

import unittest
from datetime import datetime

from slurm_cli.forecast_core import (  # noqa: E402
    JobRecord,
    NodeCapacity,
    collect_job_windows,
    is_unavailable_availability_state,
    max_colocated_available_gpus,
    node_available_gpus,
    parse_array_task_count,
    parse_partition_names,
    partition_node_capacities,
    record_targets_partition,
    total_gpu_capacity,
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
        task_count=1,
    )


def _capacity(
    node_name: str,
    partitions: tuple[str, ...],
    gpus: int = 4,
    gpu_alloc: int = 0,
    state: str = "IDLE",
) -> NodeCapacity:
    return NodeCapacity(
        node_name=node_name,
        cpu=64,
        mem_mib=512000,
        gpus=gpus,
        cpu_alloc=0,
        mem_alloc_mib=0,
        gpu_alloc=gpu_alloc,
        partition_names=partitions,
        state=state,
    )


class ForecastPartitioningTests(unittest.TestCase):
    def test_parse_array_task_count_handles_ranges_steps_and_lists(self) -> None:
        self.assertEqual(parse_array_task_count(array_task_text="27"), 1)
        self.assertEqual(parse_array_task_count(array_task_text="0-1"), 2)
        self.assertEqual(parse_array_task_count(array_task_text="0-15:4%2"), 4)
        self.assertEqual(parse_array_task_count(array_task_text="1,3,7-9"), 5)

    def test_parse_partition_names_normalizes_star_and_case(self) -> None:
        self.assertEqual(parse_partition_names(value="GPU*,Quad"), ("gpu", "quad"))

    def test_partition_node_capacities_filters_nodes(self) -> None:
        nodes = {
            "node-a": _capacity(node_name="node-a", partitions=("gpu", "quad")),
            "node-b": _capacity(node_name="node-b", partitions=("gpu",)),
        }
        filtered = partition_node_capacities(
            node_capacities=nodes, partition_name="quad"
        )
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
        record = _record(
            state="RUNNING", requested_gpus=4, node_expression="node-[1]", partitions=()
        )
        included = record_targets_partition(
            record=record,
            partition_name="quad",
            partition_node_names={"node001"},
            host_cache={"node-[1]": ["node001"]},
            infer_quad_large_gpu=False,
        )
        self.assertTrue(included)

    def test_max_colocated_available_gpus_uses_single_node_peak(self) -> None:
        nodes = {
            "node-a": _capacity(
                node_name="node-a", partitions=("gpu",), gpus=8, gpu_alloc=2
            ),
            "node-b": _capacity(
                node_name="node-b", partitions=("gpu",), gpus=4, gpu_alloc=3
            ),
        }
        self.assertEqual(
            max_colocated_available_gpus(node_capacities=nodes),
            6,
        )

    def test_max_colocated_available_gpus_respects_partition_filter(self) -> None:
        nodes = {
            "node-a": _capacity(
                node_name="node-a", partitions=("gpu",), gpus=8, gpu_alloc=1
            ),
            "node-b": _capacity(
                node_name="node-b", partitions=("quad",), gpus=4, gpu_alloc=1
            ),
        }
        self.assertEqual(
            max_colocated_available_gpus(node_capacities=nodes, partition_name="quad"),
            3,
        )

    def test_node_available_gpus_excludes_maintenance_and_drain(self) -> None:
        maintenance = _capacity(
            node_name="node-a",
            partitions=("gpu",),
            gpus=4,
            gpu_alloc=0,
            state="IDLE+MAINTENANCE+RESERVED",
        )
        drained = _capacity(
            node_name="node-b",
            partitions=("gpu",),
            gpus=4,
            gpu_alloc=0,
            state="IDLE+DRAIN",
        )
        self.assertTrue(is_unavailable_availability_state(state_text=maintenance.state))
        self.assertEqual(node_available_gpus(capacity=maintenance), 0)
        self.assertEqual(node_available_gpus(capacity=drained), 0)

    def test_total_gpu_capacity_excludes_maintenance_and_drain(self) -> None:
        nodes = {
            "node-a": _capacity(node_name="node-a", partitions=("gpu",), gpus=4),
            "node-b": _capacity(
                node_name="node-b",
                partitions=("gpu",),
                gpus=4,
                state="IDLE+MAINTENANCE+RESERVED",
            ),
            "node-c": _capacity(
                node_name="node-c",
                partitions=("gpu",),
                gpus=4,
                state="IDLE+DRAIN",
            ),
        }
        self.assertEqual(total_gpu_capacity(node_capacities=nodes), 4)

    def test_collect_job_windows_counts_pending_array_tasks(self) -> None:
        raw_jobs = (
            "JobId=4959484 ArrayJobId=4959484 ArrayTaskId=0-1 JobState=PENDING "
            "RunTime=00:00:00 TimeLimit=03:00:00 StartTime=2026-04-17T16:00:00 "
            "EndTime=2026-04-17T19:00:00 ReqTRES=cpu=32,mem=343168M,node=1,gres/gpu=1 "
            "AllocTRES=(null) Partition=nextgen NumNodes=1\n"
        )
        windows, stats = collect_job_windows(
            raw_jobs=raw_jobs,
            now=datetime(2026, 4, 17, 15, 0, 0),
            node_capacities={},
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].gpus, 2)
        self.assertEqual(stats.active_gpu_jobs, 2)
        self.assertEqual(stats.pending_gpu_jobs, 2)
        self.assertEqual(stats.pending_with_start, 2)
        self.assertEqual(stats.pending_without_start, 0)


if __name__ == "__main__":
    unittest.main()
