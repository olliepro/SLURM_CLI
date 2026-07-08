from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

from slurm_cli.dash_logic import DashJob
from slurm_cli.forecast_core import NodeCapacity
from slurm_cli import agent_query


NOW = datetime(2026, 6, 26, 12, 0, 0)


def _capacity(
    node_name: str,
    partitions: tuple[str, ...],
    gpus: int = 4,
    gpu_alloc: int = 0,
    cpu_alloc: int = 0,
    mem_alloc_mib: int = 0,
    state: str = "IDLE",
) -> NodeCapacity:
    return NodeCapacity(
        node_name=node_name,
        cpu=64,
        mem_mib=512000,
        gpus=gpus,
        cpu_alloc=cpu_alloc,
        mem_alloc_mib=mem_alloc_mib,
        gpu_alloc=gpu_alloc,
        partition_names=partitions,
        state=state,
    )


def _cardinal_capacities() -> dict[str, NodeCapacity]:
    return {
        "d0001": _capacity("d0001", ("debug",), gpus=4),
        "g0001": _capacity("g0001", ("gpu",), gpus=8),
    }


def _fragmented_quad_capacities() -> dict[str, NodeCapacity]:
    return {
        "a0001": _capacity("a0001", ("quad",), gpus=4, gpu_alloc=2),
        "a0002": _capacity("a0002", ("quad",), gpus=4, gpu_alloc=2),
    }


class RecommendTests(unittest.TestCase):
    def test_short_single_gpu_routes_to_debug(self) -> None:
        payload = agent_query.build_recommend(
            gpus=1, cpus=4, time_minutes=30, mem_str="16G", cluster_name="cardinal"
        )
        self.assertEqual(payload["recommended_partition"], "debug")
        self.assertTrue(payload["qualifies_for_debug"])
        self.assertEqual(payload["debug_window_minutes"], 60)

    def test_multi_gpu_does_not_qualify_for_debug(self) -> None:
        payload = agent_query.build_recommend(
            gpus=2, cpus=4, time_minutes=30, mem_str="16G", cluster_name="cardinal"
        )
        self.assertEqual(payload["recommended_partition"], "gpu")
        self.assertFalse(payload["qualifies_for_debug"])
        # Debug is reachable by reshaping to 1 GPU; advice should explain how.
        self.assertEqual(payload["debug_partition"], "debug")
        self.assertIn("GPUs to 1", payload["advice"])

    def test_long_single_gpu_gets_shorten_advice(self) -> None:
        payload = agent_query.build_recommend(
            gpus=1, cpus=4, time_minutes=120, mem_str="16G", cluster_name="cardinal"
        )
        self.assertEqual(payload["recommended_partition"], "gpu")
        self.assertEqual(payload["debug_partition"], "debug")
        self.assertIn("60 min", payload["advice"])


class PlanTests(unittest.TestCase):
    def test_plan_lists_debug_then_standard_with_immediate_start(self) -> None:
        payload = agent_query.build_plan(
            raw_jobs="",
            node_capacities=_cardinal_capacities(),
            now=NOW,
            gpus=1,
            cpus=4,
            time_minutes=30,
            mem_str="16G",
            cluster_name="cardinal",
        )
        self.assertEqual(payload["recommended_partition"], "debug")
        partitions = [option["partition"] for option in payload["options"]]
        self.assertEqual(partitions, ["debug", "gpu"])
        debug_option = payload["options"][0]
        self.assertTrue(debug_option["is_debug"])
        # Empty job text => no occupancy => everything free now.
        self.assertEqual(debug_option["available_now"], 4)
        self.assertTrue(debug_option["start_estimate"]["available_now"])
        self.assertEqual(debug_option["start_estimate"]["in_hours"], 0.0)

    def test_plan_caveats_present(self) -> None:
        payload = agent_query.build_plan(
            raw_jobs="",
            node_capacities=_cardinal_capacities(),
            now=NOW,
            gpus=1,
            cpus=4,
            time_minutes=30,
            mem_str="16G",
            cluster_name="cardinal",
        )
        self.assertTrue(payload["caveats"])
        self.assertTrue(any("priority" in c for c in payload["caveats"]))

    def test_plan_rejects_immediate_start_when_gpus_are_fragmented(self) -> None:
        payload = agent_query.build_plan(
            raw_jobs="",
            node_capacities=_fragmented_quad_capacities(),
            now=NOW,
            gpus=4,
            cpus=48,
            time_minutes=24 * 60,
            mem_str="256G",
            cluster_name="ascend",
        )
        option = payload["options"][0]
        self.assertEqual(option["partition"], "quad")
        self.assertEqual(option["max_colocated_available"], 2)
        self.assertFalse(option["start_estimate"]["available_now"])
        self.assertIsNone(option["start_estimate"]["earliest_start_at"])
        self.assertIn("no single node", option["start_estimate"]["note"])


class ForecastAvailTests(unittest.TestCase):
    def test_forecast_reports_immediate_availability_when_idle(self) -> None:
        payload = agent_query.build_forecast(
            raw_jobs="",
            node_capacities=_cardinal_capacities(),
            now=NOW,
            partition="gpu",
            want_gpus=2,
        )
        self.assertEqual(payload["capacity_gpus"], 8)
        self.assertEqual(payload["available_now"], 8)
        self.assertEqual(payload["earliest_free"]["in_hours"], 0.0)
        self.assertTrue(payload["series"])

    def test_forecast_rejects_immediate_start_when_gpus_are_fragmented(self) -> None:
        payload = agent_query.build_forecast(
            raw_jobs="",
            node_capacities=_fragmented_quad_capacities(),
            now=NOW,
            partition="quad",
            want_gpus=4,
        )
        self.assertEqual(payload["max_colocated_available"], 2)
        self.assertIsNone(payload["earliest_free"]["at"])
        self.assertIn("no single node", payload["earliest_free"]["note"])

    def test_avail_lists_each_gpu_partition(self) -> None:
        payload = agent_query.build_avail(
            node_capacities=_cardinal_capacities(), now=NOW
        )
        names = {row["partition"] for row in payload["partitions"]}
        self.assertEqual(names, {"debug", "gpu"})
        debug_row = next(r for r in payload["partitions"] if r["partition"] == "debug")
        self.assertTrue(debug_row["is_debug"])
        self.assertEqual(debug_row["available_now"], 4)


class JobsTests(unittest.TestCase):
    def test_jobs_serializes_eta_and_state(self) -> None:
        running = DashJob(
            job_id="1",
            state_compact="R",
            name="train",
            time_used="0:10",
            time_left="0:50",
            start_time=None,
            reason="None",
            node_list="g0001",
            work_dir="/home/u/train",
        )
        pending = DashJob(
            job_id="2",
            state_compact="PD",
            name="probe",
            time_used="0:00",
            time_left="1:00",
            start_time=datetime(2026, 6, 26, 13, 0, 0),
            reason="Priority",
            node_list="",
            work_dir="/home/u",
        )
        with mock.patch.object(
            agent_query, "fetch_dash_jobs", return_value=[running, pending]
        ):
            payload = agent_query.build_jobs(user_name="u", now=NOW)
        self.assertEqual(payload["user"], "u")
        first, second = payload["jobs"]
        self.assertEqual(first["state"], "RUNNING")
        self.assertEqual(first["eta"], "0h00m")
        self.assertEqual(second["state"], "PENDING")
        # Pending start is one hour out from NOW.
        self.assertEqual(second["eta"], "1h00m")


if __name__ == "__main__":
    unittest.main()
