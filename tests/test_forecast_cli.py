from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from slurm_cli.forecast_cli import (
    default_primary_partition,
    debug_marker_from_records,
    matching_debug_partition,
)
from slurm_cli.forecast_core import JobRecord, NodeCapacity


class ForecastCliTests(unittest.TestCase):
    def _record(
        self,
        *,
        state: str,
        node_expression: str | None,
        end_time: datetime | None,
        start_time: datetime | None,
        partitions: tuple[str, ...] = ("debug",),
        requested_gpus: int = 1,
        allocated_gpus: int = 1,
    ) -> JobRecord:
        return JobRecord(
            job_id=123,
            state=state,
            requested_gpus=requested_gpus,
            allocated_gpus=allocated_gpus,
            start_time=start_time,
            end_time=end_time,
            time_limit_hours=1.0,
            run_time_hours=0.0,
            requested_cpus=4,
            requested_mem_mib=1024,
            requested_nodes=1,
            node_expression=node_expression,
            partition_names=partitions,
            task_count=1,
        )

    def _capacity(
        self,
        *,
        node_name: str,
        partitions: tuple[str, ...],
        gpu_alloc: int,
        gpus: int = 4,
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
            state="ALLOCATED" if gpu_alloc else "IDLE",
        )

    def test_default_primary_partition_uses_gpu_on_cardinal(self) -> None:
        self.assertEqual(default_primary_partition(cluster_name="cardinal"), "gpu")

    def test_default_primary_partition_leaves_other_clusters_unscoped(self) -> None:
        self.assertIsNone(default_primary_partition(cluster_name="ascend"))
        self.assertIsNone(default_primary_partition(cluster_name="pitzer"))
        self.assertIsNone(default_primary_partition(cluster_name=None))

    def test_matching_debug_partition_uses_scope_specific_mapping(self) -> None:
        self.assertEqual(
            matching_debug_partition(cluster_name="ascend", target_partition=None),
            "debug",
        )
        self.assertEqual(
            matching_debug_partition(cluster_name="ascend", target_partition="quad"),
            "quad",
        )
        self.assertEqual(
            matching_debug_partition(cluster_name="cardinal", target_partition="gpu"),
            "debug",
        )
        self.assertEqual(
            matching_debug_partition(cluster_name="pitzer", target_partition="quad"),
            "gpudebug-quad",
        )

    def test_debug_marker_from_records_returns_immediate_when_gpu_is_free(self) -> None:
        now = datetime(2026, 2, 11, 9, 0, 0)
        marker = debug_marker_from_records(
            generated_at=now,
            partition_name="debug",
            records=[],
            node_capacities={
                "dbg001": self._capacity(
                    node_name="dbg001",
                    partitions=("debug",),
                    gpu_alloc=3,
                )
            },
            horizon_hours=8.0,
        )
        assert marker is not None
        self.assertEqual(marker.partition_name, "debug")
        self.assertEqual(marker.offset_minutes, 0)
        self.assertEqual(marker.label(), "dbg g1 0h00m")

    def test_debug_marker_from_records_uses_earliest_running_job_end(self) -> None:
        now = datetime(2026, 2, 11, 9, 0, 0)
        marker = debug_marker_from_records(
            generated_at=now,
            partition_name="debug",
            records=[
                self._record(
                    state="RUNNING",
                    node_expression="dbg001",
                    start_time=now - timedelta(minutes=10),
                    end_time=now + timedelta(minutes=45),
                ),
                self._record(
                    state="RUNNING",
                    node_expression="dbg002",
                    start_time=now - timedelta(minutes=5),
                    end_time=now + timedelta(minutes=90),
                ),
                self._record(
                    state="PENDING",
                    node_expression=None,
                    start_time=now + timedelta(minutes=15),
                    end_time=now + timedelta(minutes=75),
                ),
            ],
            node_capacities={
                "dbg001": self._capacity(
                    node_name="dbg001",
                    partitions=("debug",),
                    gpu_alloc=2,
                    gpus=2,
                ),
                "dbg002": self._capacity(
                    node_name="dbg002",
                    partitions=("debug",),
                    gpu_alloc=2,
                    gpus=2,
                ),
            },
            horizon_hours=8.0,
        )
        assert marker is not None
        self.assertEqual(marker.partition_name, "debug")
        self.assertEqual(marker.offset_minutes, 45)
        self.assertEqual(marker.label(), "dbg g1 0h45m")

    def test_debug_marker_from_records_survives_beyond_panel_horizon(self) -> None:
        now = datetime(2026, 2, 11, 9, 0, 0)
        marker = debug_marker_from_records(
            generated_at=now,
            partition_name="debug-quad",
            records=[
                self._record(
                    state="RUNNING",
                    node_expression="dbgq001",
                    start_time=now - timedelta(minutes=30),
                    end_time=now + timedelta(hours=10),
                    partitions=("quad",),
                    requested_gpus=4,
                    allocated_gpus=4,
                )
            ],
            node_capacities={
                "dbgq001": self._capacity(
                    node_name="dbgq001",
                    partitions=("debug-quad",),
                    gpu_alloc=4,
                    gpus=4,
                )
            },
            horizon_hours=8.0,
        )
        assert marker is not None
        self.assertEqual(marker.partition_name, "debug-quad")
        self.assertEqual(marker.offset_minutes, 600)
        self.assertEqual(marker.label(), "dbg g1 10h00m")


if __name__ == "__main__":
    unittest.main()
