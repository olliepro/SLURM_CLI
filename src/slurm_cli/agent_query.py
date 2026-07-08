"""Non-interactive, machine-readable query surface for agents.

This module exposes the same Slurm parsing/forecast logic that powers the
interactive dashboard, but as plain JSON on stdout with no curses, no prompts,
and no side effects. It is meant to answer the questions an agent (e.g. Claude
Code) actually has when composing an `sbatch`/`srun`:

- "Where should I schedule this job?" -> partition routing, debug-aware.
- "How long until it can start?" -> capacity-forecast start estimate.
- "What is free right now?" -> per-partition availability snapshot.
- "What jobs do I already have queued, and when do they start?"

Design notes:
- Every builder takes already-fetched raw Slurm text and a `now` timestamp so
  it is testable without a live cluster. Only `run_query_command` touches Slurm.
- Output carries `schema_version` so the contract can evolve safely.
- Start estimates are capacity-based, NOT priority-aware; the `caveats` field
  spells this out so an agent does not over-trust them.

Example:
    >>> import json
    >>> payload = build_recommend(  # doctest: +SKIP
    ...     gpus=1, cpus=4, time_minutes=30, mem_str="16G", cluster_name="cardinal"
    ... )
    >>> json.dumps(payload)  # doctest: +SKIP
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from slurm_cli.constants import DEFAULT_CPUS, DEFAULT_GPUS, DEFAULT_MEM_GB
from slurm_cli.dash_logic import fetch_dash_jobs
from slurm_cli.format_utils import parse_time_string
from slurm_cli.forecast_core import (
    NodeCapacity,
    available_series,
    build_event_deltas,
    build_step_series,
    choose_horizon,
    collect_job_windows,
    group_event_deltas,
    max_colocated_available_gpus,
    node_available_gpus,
    parse_node_capacities,
    partition_gpu_capacity,
    partition_node_capacities,
    run_command,
)
from slurm_cli.partition_policy import (
    _DEBUG_MAX_MINUTES,
    PartitionRequest,
    detect_cluster_name,
    recommend_partition,
)

SCHEMA_VERSION = 1
DEFAULT_HORIZON_HOURS = 8.0

# Clusters where debug priority additionally requires a single GPU. On pitzer
# the debug window alone is sufficient (any GPU count maps to a gpudebug* part).
_SINGLE_GPU_DEBUG_CLUSTERS = {"ascend", "cardinal"}

_START_ESTIMATE_CAVEATS = (
    "Capacity-only estimate: it reflects when enough GPUs free up, not Slurm "
    "priority/backfill ordering. Higher-priority pending jobs (or your "
    "fair-share) may take the freed GPUs first, so real start time can be later.",
    "Availability is partition-wide; a job that needs all its GPUs on one node "
    "must also fit within 'max_colocated_available'.",
)


@dataclass(frozen=True)
class PartitionPlan:
    """Forecast-based start plan for one partition and request size."""

    partition: str
    is_debug: bool
    request_gpus: int
    capacity_gpus: int
    available_now: int
    max_colocated_available: int
    colocation_blocked_now: bool
    available_immediately: bool
    earliest_start_at: datetime | None

    def to_dict(self, now: datetime) -> dict[str, Any]:
        """Return the JSON-ready mapping for this partition plan."""

        start: dict[str, Any] = {
            "available_now": self.available_immediately,
            "basis": "capacity_forecast",
        }
        if self.colocation_blocked_now:
            start["earliest_start_at"] = None
            start["in_hours"] = None
            start["note"] = (
                f"Partition has {self.available_now} free GPUs now, but no single "
                f"node has {self.request_gpus} free GPUs "
                f"(max_colocated_available={self.max_colocated_available}). "
                "Use sbatch --test-only for node-level placement timing."
            )
        elif self.earliest_start_at is None:
            start["earliest_start_at"] = None
            start["in_hours"] = None
            start["note"] = (
                "No opening within the forecast horizon; demand stays above "
                "capacity for the whole window."
            )
        else:
            in_hours = max(0.0, (self.earliest_start_at - now).total_seconds() / 3600.0)
            start["earliest_start_at"] = self.earliest_start_at.isoformat()
            start["in_hours"] = round(in_hours, 2)
        return {
            "partition": self.partition,
            "is_debug": self.is_debug,
            "capacity_gpus": self.capacity_gpus,
            "available_now": self.available_now,
            "max_colocated_available": self.max_colocated_available,
            "start_estimate": start,
        }


def _is_debug_partition(partition: Optional[str]) -> bool:
    """Return true when a partition name denotes a high-priority debug queue."""

    return partition is not None and "debug" in partition


def _request(
    gpus: int, cpus: int, time_minutes: int, mem_str: str
) -> PartitionRequest:
    return PartitionRequest(
        gpus=gpus, cpus=cpus, time_minutes=time_minutes, mem_str=mem_str
    )


def _standard_partition(
    gpus: int, cpus: int, mem_str: str, cluster_name: Optional[str]
) -> Optional[str]:
    """Return the partition this shape would use with a non-debug walltime."""

    return recommend_partition(
        request=_request(
            gpus=gpus,
            cpus=cpus,
            time_minutes=_DEBUG_MAX_MINUTES + 1,
            mem_str=mem_str,
        ),
        cluster_name=cluster_name,
    )


def _debug_partition(
    gpus: int, cpus: int, mem_str: str, cluster_name: Optional[str]
) -> Optional[str]:
    """Return the debug partition a qualifying variant of this shape would use."""

    normalized = (cluster_name or "").lower()
    probe_gpus = 1 if normalized in _SINGLE_GPU_DEBUG_CLUSTERS else max(gpus, 1)
    candidate = recommend_partition(
        request=_request(
            gpus=probe_gpus,
            cpus=cpus,
            time_minutes=min(_DEBUG_MAX_MINUTES, _DEBUG_MAX_MINUTES),
            mem_str=mem_str,
        ),
        cluster_name=cluster_name,
    )
    return candidate if _is_debug_partition(candidate) else None


def _debug_advice(
    cluster_name: Optional[str],
    gpus: int,
    time_minutes: int,
    recommended: Optional[str],
    debug_candidate: Optional[str],
) -> str:
    """Return one human/agent-readable line about debug eligibility."""

    normalized = (cluster_name or "").lower()
    if _is_debug_partition(recommended):
        return (
            f"Routed to high-priority debug partition '{recommended}' "
            f"(walltime <= {_DEBUG_MAX_MINUTES} min qualifies)."
        )
    needs_single_gpu = normalized in _SINGLE_GPU_DEBUG_CLUSTERS
    if debug_candidate is None:
        if needs_single_gpu and gpus != 1:
            return (
                f"No debug option for {gpus} GPUs on {normalized or 'this cluster'}: "
                f"debug priority requires a single GPU. Multi-GPU jobs use "
                f"'{recommended}'."
            )
        return (
            f"No high-priority debug partition applies; using '{recommended}'."
        )
    hints = []
    if time_minutes > _DEBUG_MAX_MINUTES:
        hints.append(f"walltime to <= {_DEBUG_MAX_MINUTES} min")
    if needs_single_gpu and gpus != 1:
        hints.append("GPUs to 1")
    joiner = " and ".join(hints) if hints else "the request"
    return (
        f"To use the high-priority '{debug_candidate}' partition, reduce "
        f"{joiner}. As specified, this job uses '{recommended}'."
    )


def _forecast_partition(
    raw_jobs: str,
    now: datetime,
    node_capacities: dict[str, NodeCapacity],
    partition: str,
    horizon_hours: float,
) -> tuple[list[datetime], list[int], int]:
    """Return (times, available_gpus, capacity) for one partition forecast."""

    infer_quad = "quad" in partition
    windows, _ = collect_job_windows(
        raw_jobs=raw_jobs,
        now=now,
        node_capacities=node_capacities,
        target_partition=partition,
        infer_quad_large_gpu=infer_quad,
    )
    capacity = partition_gpu_capacity(
        node_capacities=node_capacities, partition_name=partition
    )
    baseline, events = build_event_deltas(windows=windows, now=now)
    grouped = group_event_deltas(events=events)
    horizon = choose_horizon(
        windows=windows, now=now, horizon_hours=horizon_hours
    )
    times, usage = build_step_series(
        now=now, baseline=baseline, grouped_events=grouped, horizon=horizon
    )
    return times, available_series(usage=usage, capacity=capacity), capacity


def _earliest_at_least(
    times: Sequence[datetime], available: Sequence[int], want_gpus: int
) -> datetime | None:
    """Return the first timestamp where availability meets the GPU demand."""

    for series_time, free in zip(times, available):
        if free >= want_gpus:
            return series_time
    return None


def _colocation_blocked_now(
    earliest: datetime | None, now: datetime, max_colocated: int, want_gpus: int
) -> bool:
    """Return true when aggregate capacity is free now but no one node fits."""

    return earliest is not None and earliest <= now and max_colocated < want_gpus


def _partition_plan(
    raw_jobs: str,
    now: datetime,
    node_capacities: dict[str, NodeCapacity],
    partition: str,
    want_gpus: int,
    horizon_hours: float,
) -> PartitionPlan:
    """Build a forecast-based start plan for one partition."""

    times, available, capacity = _forecast_partition(
        raw_jobs=raw_jobs,
        now=now,
        node_capacities=node_capacities,
        partition=partition,
        horizon_hours=horizon_hours,
    )
    available_now = available[0] if available else 0
    earliest = _earliest_at_least(
        times=times, available=available, want_gpus=want_gpus
    )
    max_colocated = max_colocated_available_gpus(
        node_capacities=node_capacities, partition_name=partition
    )
    colocation_blocked_now = _colocation_blocked_now(
        earliest=earliest,
        now=now,
        max_colocated=max_colocated,
        want_gpus=want_gpus,
    )
    return PartitionPlan(
        partition=partition,
        is_debug=_is_debug_partition(partition),
        request_gpus=want_gpus,
        capacity_gpus=capacity,
        available_now=available_now,
        max_colocated_available=max_colocated,
        colocation_blocked_now=colocation_blocked_now,
        available_immediately=(
            earliest is not None and earliest <= now and not colocation_blocked_now
        ),
        earliest_start_at=earliest,
    )


def gpu_partition_names(node_capacities: dict[str, NodeCapacity]) -> list[str]:
    """Return sorted partition names that host GPU nodes."""

    names: set[str] = set()
    for capacity in node_capacities.values():
        names.update(capacity.partition_names)
    return sorted(names)


def build_recommend(
    gpus: int,
    cpus: int,
    time_minutes: int,
    mem_str: str,
    cluster_name: Optional[str],
) -> dict[str, Any]:
    """Return partition routing for one request, including debug eligibility."""

    recommended = recommend_partition(
        request=_request(
            gpus=gpus, cpus=cpus, time_minutes=time_minutes, mem_str=mem_str
        ),
        cluster_name=cluster_name,
    )
    debug_candidate = _debug_partition(
        gpus=gpus, cpus=cpus, mem_str=mem_str, cluster_name=cluster_name
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster": cluster_name,
        "request": {
            "gpus": gpus,
            "cpus": cpus,
            "time_minutes": time_minutes,
            "mem": mem_str,
        },
        "recommended_partition": recommended,
        "qualifies_for_debug": _is_debug_partition(recommended),
        "debug_window_minutes": _DEBUG_MAX_MINUTES,
        "debug_partition": debug_candidate,
        "advice": _debug_advice(
            cluster_name=cluster_name,
            gpus=gpus,
            time_minutes=time_minutes,
            recommended=recommended,
            debug_candidate=debug_candidate,
        ),
    }


def build_avail(
    node_capacities: dict[str, NodeCapacity],
    now: datetime,
    partition: Optional[str] = None,
) -> dict[str, Any]:
    """Return a current free-GPU snapshot per GPU partition."""

    partitions = (
        [partition] if partition is not None else gpu_partition_names(node_capacities)
    )
    rows = []
    for name in partitions:
        scoped = partition_node_capacities(
            node_capacities=node_capacities, partition_name=name
        )
        if not scoped:
            continue
        rows.append(
            {
                "partition": name,
                "is_debug": _is_debug_partition(name),
                "capacity_gpus": partition_gpu_capacity(
                    node_capacities=node_capacities, partition_name=name
                ),
                "available_now": sum(
                    node_available_gpus(capacity=cap) for cap in scoped.values()
                ),
                "max_colocated_available": max_colocated_available_gpus(
                    node_capacities=node_capacities, partition_name=name
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "partitions": rows,
    }


def build_forecast(
    raw_jobs: str,
    node_capacities: dict[str, NodeCapacity],
    now: datetime,
    partition: str,
    want_gpus: int,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
) -> dict[str, Any]:
    """Return an availability time-series and earliest-free time for a partition."""

    times, available, capacity = _forecast_partition(
        raw_jobs=raw_jobs,
        now=now,
        node_capacities=node_capacities,
        partition=partition,
        horizon_hours=horizon_hours,
    )
    earliest = _earliest_at_least(
        times=times, available=available, want_gpus=want_gpus
    )
    max_colocated = max_colocated_available_gpus(
        node_capacities=node_capacities, partition_name=partition
    )
    colocation_blocked_now = _colocation_blocked_now(
        earliest=earliest,
        now=now,
        max_colocated=max_colocated,
        want_gpus=want_gpus,
    )
    series = [
        {"at": series_time.isoformat(), "free_gpus": free}
        for series_time, free in zip(times, available)
    ]
    earliest_free: dict[str, Any]
    if colocation_blocked_now:
        earliest_free = {
            "gpus": want_gpus,
            "at": None,
            "in_hours": None,
            "note": (
                f"Partition has {available[0] if available else 0} free GPUs now, "
                f"but no single node has {want_gpus} free GPUs "
                f"(max_colocated_available={max_colocated}). Use sbatch --test-only "
                "for node-level placement timing."
            ),
        }
    elif earliest is None:
        earliest_free = {"gpus": want_gpus, "at": None, "in_hours": None}
    else:
        earliest_free = {
            "gpus": want_gpus,
            "at": earliest.isoformat(),
            "in_hours": round(
                max(0.0, (earliest - now).total_seconds() / 3600.0), 2
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "partition": partition,
        "is_debug": _is_debug_partition(partition),
        "capacity_gpus": capacity,
        "available_now": available[0] if available else 0,
        "max_colocated_available": max_colocated,
        "horizon_hours": horizon_hours,
        "earliest_free": earliest_free,
        "series": series,
        "caveats": list(_START_ESTIMATE_CAVEATS),
    }


def build_plan(
    raw_jobs: str,
    node_capacities: dict[str, NodeCapacity],
    now: datetime,
    gpus: int,
    cpus: int,
    time_minutes: int,
    mem_str: str,
    cluster_name: Optional[str],
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
) -> dict[str, Any]:
    """Answer 'where should I schedule this and when can it start?'."""

    recommended = recommend_partition(
        request=_request(
            gpus=gpus, cpus=cpus, time_minutes=time_minutes, mem_str=mem_str
        ),
        cluster_name=cluster_name,
    )
    standard = _standard_partition(
        gpus=gpus, cpus=cpus, mem_str=mem_str, cluster_name=cluster_name
    )
    debug_candidate = _debug_partition(
        gpus=gpus, cpus=cpus, mem_str=mem_str, cluster_name=cluster_name
    )

    candidate_partitions: list[str] = []
    for name in (recommended, standard):
        if name is not None and name not in candidate_partitions:
            candidate_partitions.append(name)

    options = [
        _partition_plan(
            raw_jobs=raw_jobs,
            now=now,
            node_capacities=node_capacities,
            partition=name,
            want_gpus=gpus,
            horizon_hours=horizon_hours,
        ).to_dict(now=now)
        for name in candidate_partitions
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "cluster": cluster_name,
        "request": {
            "gpus": gpus,
            "cpus": cpus,
            "time_minutes": time_minutes,
            "mem": mem_str,
        },
        "recommended_partition": recommended,
        "qualifies_for_debug": _is_debug_partition(recommended),
        "debug_window_minutes": _DEBUG_MAX_MINUTES,
        "debug_partition": debug_candidate,
        "advice": _debug_advice(
            cluster_name=cluster_name,
            gpus=gpus,
            time_minutes=time_minutes,
            recommended=recommended,
            debug_candidate=debug_candidate,
        ),
        "options": options,
        "caveats": list(_START_ESTIMATE_CAVEATS),
    }


def build_jobs(user_name: str, now: datetime) -> dict[str, Any]:
    """Return the caller's pending/running jobs with start ETAs."""

    jobs = fetch_dash_jobs(user_name=user_name)
    rows = []
    for job in jobs:
        rows.append(
            {
                "job_id": job.job_id,
                "state": "RUNNING" if job.is_running() else "PENDING",
                "name": job.name,
                "time_used": job.time_used,
                "time_left": job.time_left,
                "start_time": (
                    job.start_time.isoformat() if job.start_time is not None else None
                ),
                "eta": job.eta_text(as_of=now),
                "reason": job.reason,
                "node_list": job.node_list,
                "work_dir": job.work_dir,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "user": user_name,
        "jobs": rows,
    }


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _fetch_state() -> tuple[str, str]:
    """Fetch raw Slurm job/node text used by the query builders."""

    raw_jobs = run_command(command=["scontrol", "show", "jobs", "-o"])
    raw_nodes = run_command(command=["scontrol", "show", "nodes", "-o"])
    return raw_jobs, raw_nodes


def _time_minutes(value: Optional[str], default_minutes: int) -> int:
    """Parse a `--time` value (HH:MM:SS, D-HH:MM:SS, or whole minutes)."""

    if value is None:
        return default_minutes
    if ":" not in value and "-" not in value and value.isdigit():
        return int(value)
    parsed = parse_time_string(value=value)
    assert parsed is not None, f"Could not parse --time value: {value!r}"
    return parsed


def build_query_parser() -> argparse.ArgumentParser:
    """Build the `gpu query` argument parser."""

    parser = argparse.ArgumentParser(
        prog="gpu query",
        description="Non-interactive JSON queries for agents/scripts.",
    )
    sub = parser.add_subparsers(dest="query_command", required=True)

    def add_request_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--gpus", type=int, default=DEFAULT_GPUS)
        sp.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
        sp.add_argument("--time", default=None, help="HH:MM:SS, D-HH:MM:SS, or minutes")
        sp.add_argument("--mem", default=f"{DEFAULT_MEM_GB}G")
        sp.add_argument("--cluster", default=None, help="Override cluster detection")

    plan = sub.add_parser("plan", help="Where to schedule + when it can start")
    add_request_args(plan)
    plan.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)

    recommend = sub.add_parser("recommend", help="Partition routing only")
    add_request_args(recommend)

    forecast = sub.add_parser("forecast", help="Availability time-series")
    forecast.add_argument("--partition", required=True)
    forecast.add_argument("--gpus", type=int, default=DEFAULT_GPUS)
    forecast.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)

    avail = sub.add_parser("avail", help="Current free GPUs per partition")
    avail.add_argument("--partition", default=None)

    jobs = sub.add_parser("jobs", help="Your pending/running jobs with ETAs")
    jobs.add_argument("--user", default=None)

    return parser


def parse_query_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse `gpu query` arguments."""

    return build_query_parser().parse_args(args=list(argv))


def run_query_command(args: argparse.Namespace, now: datetime | None = None) -> int:
    """Run one `gpu query` subcommand and print JSON to stdout."""

    moment = now if now is not None else datetime.now()
    command = args.query_command

    if command == "recommend":
        payload = build_recommend(
            gpus=args.gpus,
            cpus=args.cpus,
            time_minutes=_time_minutes(value=args.time, default_minutes=30),
            mem_str=args.mem,
            cluster_name=(args.cluster or detect_cluster_name()),
        )
        _emit(payload)
        return 0

    if command == "jobs":
        user_name = args.user or os.environ.get("USER", "")
        assert user_name, "No user: pass --user or set $USER"
        _emit(build_jobs(user_name=user_name, now=moment))
        return 0

    if command == "avail":
        _, raw_nodes = _fetch_state()
        node_capacities = parse_node_capacities(raw_nodes=raw_nodes)
        _emit(
            build_avail(
                node_capacities=node_capacities,
                now=moment,
                partition=args.partition,
            )
        )
        return 0

    if command == "forecast":
        raw_jobs, raw_nodes = _fetch_state()
        node_capacities = parse_node_capacities(raw_nodes=raw_nodes)
        _emit(
            build_forecast(
                raw_jobs=raw_jobs,
                node_capacities=node_capacities,
                now=moment,
                partition=args.partition,
                want_gpus=args.gpus,
                horizon_hours=args.horizon_hours,
            )
        )
        return 0

    if command == "plan":
        raw_jobs, raw_nodes = _fetch_state()
        node_capacities = parse_node_capacities(raw_nodes=raw_nodes)
        _emit(
            build_plan(
                raw_jobs=raw_jobs,
                node_capacities=node_capacities,
                now=moment,
                gpus=args.gpus,
                cpus=args.cpus,
                time_minutes=_time_minutes(value=args.time, default_minutes=30),
                mem_str=args.mem,
                cluster_name=(args.cluster or detect_cluster_name()),
                horizon_hours=args.horizon_hours,
            )
        )
        return 0

    build_query_parser().error(f"unknown query command: {command}")
    return 2


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON payload to stdout."""

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
