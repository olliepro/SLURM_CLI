from __future__ import annotations

import unittest

from slurm_cli.search_logic import (  # noqa: E402
    SearchBounds,
    SearchProbe,
    build_search_probes,
    format_compact_minutes,
)


class SearchLogicTests(unittest.TestCase):
    def test_two_phase_chain_shape(self) -> None:
        bounds = SearchBounds(
            max_time_minutes=240,
            min_time_minutes=30,
            max_gpus=4,
            min_gpus=1,
            switch_minutes=60,
        )
        probes = build_search_probes(bounds=bounds, cpus=8, mem_str="50G")
        pairs = [(probe.time_minutes, probe.gpus) for probe in probes]
        self.assertEqual(pairs, [(240, 4), (120, 4), (60, 4), (30, 4), (30, 2), (30, 1)])

    def test_switch_happens_below_one_hour(self) -> None:
        bounds = SearchBounds(
            max_time_minutes=90,
            min_time_minutes=30,
            max_gpus=4,
            min_gpus=1,
            switch_minutes=60,
        )
        probes = build_search_probes(bounds=bounds, cpus=8, mem_str="50G")
        pairs = [(probe.time_minutes, probe.gpus) for probe in probes]
        self.assertEqual(pairs, [(90, 4), (45, 4), (45, 2), (45, 1)])

    def test_time_halving_clamps_to_minimum(self) -> None:
        bounds = SearchBounds(
            max_time_minutes=70,
            min_time_minutes=50,
            max_gpus=4,
            min_gpus=1,
            switch_minutes=60,
        )
        probes = build_search_probes(bounds=bounds, cpus=4, mem_str="32G")
        self.assertEqual(probes[1].time_minutes, 50)

    def test_gpu_halving_reaches_floor(self) -> None:
        bounds = SearchBounds(
            max_time_minutes=80,
            min_time_minutes=30,
            max_gpus=3,
            min_gpus=1,
            switch_minutes=60,
        )
        probes = build_search_probes(bounds=bounds, cpus=4, mem_str="32G")
        self.assertEqual(probes[-1].gpus, 1)

    def test_duplicate_time_gpu_pairs_are_removed(self) -> None:
        bounds = SearchBounds(
            max_time_minutes=120,
            min_time_minutes=30,
            max_gpus=2,
            min_gpus=1,
            switch_minutes=60,
        )
        probes = build_search_probes(bounds=bounds, cpus=4, mem_str="32G")
        pairs = [(probe.time_minutes, probe.gpus) for probe in probes]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_job_name_and_time_formatting(self) -> None:
        probe = SearchProbe(
            time_minutes=90,
            gpus=2,
            cpus=8,
            mem_str="50G",
            phase="time",
            index=1,
        )
        self.assertEqual(format_compact_minutes(90), "1h30m")
        self.assertEqual(
            probe.job_label(prefix="slurmcli-search"),
            "1h30m-g2-slurmcli-search",
        )


if __name__ == "__main__":
    unittest.main()
