"""Core forecast computation for cluster-wide GPU availability.

Inputs:
- Live job metadata from `scontrol show jobs -o`.
- Live node metadata from `scontrol show nodes -o`.

Outputs:
- Parsed records and forecast series for terminal dashboards/CLIs.

Example:
    >>> raw_jobs = run_command(command=["scontrol", "show", "jobs", "-o"])  # doctest: +SKIP
    >>> raw_nodes = run_command(command=["scontrol", "show", "nodes", "-o"])  # doctest: +SKIP
    >>> capacities = parse_node_capacities(raw_nodes=raw_nodes)  # doctest: +SKIP
    >>> windows, stats = collect_job_windows(  # doctest: +SKIP
    ...     raw_jobs=raw_jobs,
    ...     now=datetime.now(),
    ...     node_capacities=capacities,
    ... )
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

ACTIVE_STATES = {"RUNNING", "PENDING"}
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
NONE_VALUES = {"Unknown", "N/A", "None", "(null)", ""}
HALF_HOUR_MINUTES = 30


@dataclass(frozen=True)
class JobRecord:
    """Parsed Slurm job fields needed for usage forecasting.

    Inputs:
    - `job_id`: Slurm job identifier.
    - `state`: Slurm state text (`RUNNING` or `PENDING`).
    - `requested_gpus`: requested GPU count from `ReqTRES`.
    - `allocated_gpus`: allocated GPU count from `AllocTRES`.
    - `start_time`: scheduler start-time estimate, if known.
    - `end_time`: scheduler end-time estimate, if known.
    - `time_limit_hours`: requested walltime in hours.
    - `run_time_hours`: elapsed runtime in hours.
    - `requested_cpus`: total requested CPUs.
    - `requested_mem_mib`: total requested memory in MiB.
    - `requested_nodes`: total requested node count.
    - `node_expression`: node expression used for occupancy placement.
    - `partition_names`: normalized partition names from job metadata.

    Outputs:
    - Immutable job record for conversion into forecast windows.
    """

    job_id: int
    state: str
    requested_gpus: int
    allocated_gpus: int
    start_time: datetime | None
    end_time: datetime | None
    time_limit_hours: float
    run_time_hours: float
    requested_cpus: int
    requested_mem_mib: int
    requested_nodes: int
    node_expression: str | None
    partition_names: tuple[str, ...]

    def projected_gpus(self) -> int:
        """Return GPU count to use in forecast calculations."""

        if self.state == "RUNNING":
            return self.allocated_gpus if self.allocated_gpus > 0 else self.requested_gpus
        return self.requested_gpus


@dataclass(frozen=True)
class JobWindow:
    """Forecastable GPU occupancy interval for one job.

    Inputs:
    - `job_id`: Slurm job identifier.
    - `state`: `RUNNING` or `PENDING`.
    - `gpus`: projected GPU occupancy.
    - `start`: interval start timestamp.
    - `end`: interval end timestamp.

    Outputs:
    - Closed-open occupancy interval `[start, end)`.
    """

    job_id: int
    state: str
    gpus: int
    start: datetime
    end: datetime


@dataclass(frozen=True)
class NodeCapacity:
    """Node-level capacity values needed for degenerate-job detection.

    Inputs:
    - `node_name`: host name.
    - `cpu`: schedulable CPUs from `CfgTRES`.
    - `mem_mib`: schedulable memory from `CfgTRES`, in MiB.
    - `gpus`: schedulable GPUs from `CfgTRES`.
    - `cpu_alloc`: allocated CPUs on node.
    - `mem_alloc_mib`: allocated memory on node, in MiB.
    - `gpu_alloc`: allocated GPUs on node.
    - `partition_names`: normalized partition names hosting this node.

    Outputs:
    - Immutable per-node capacity object.
    """

    node_name: str
    cpu: int
    mem_mib: int
    gpus: int
    cpu_alloc: int
    mem_alloc_mib: int
    gpu_alloc: int
    partition_names: tuple[str, ...]


@dataclass(frozen=True)
class ForecastStats:
    """Summary counters for forecast coverage and exclusions.

    Inputs:
    - Totals of active GPU jobs and known/unknown pending-start jobs.

    Outputs:
    - Structured summary for titles, logs, and plot annotations.
    """

    active_gpu_jobs: int
    running_gpu_jobs: int
    pending_gpu_jobs: int
    pending_with_start: int
    pending_without_start: int
    forecast_windows: int
    degenerate_jobs: int
    degenerate_extra_gpus: int
    degenerate_nodes: int
    degenerate_locked_gpus: int

    def subtitle(self) -> str:
        """Return canonical summary text for plot subtitles."""

        return (
            f"active={self.active_gpu_jobs}, running={self.running_gpu_jobs}, "
            f"pending={self.pending_gpu_jobs}, pending_known_start={self.pending_with_start}, "
            f"pending_unknown_start={self.pending_without_start}, "
            f"degenerate_jobs={self.degenerate_jobs}, extra_gpu_reserved={self.degenerate_extra_gpus}, "
            f"degenerate_nodes={self.degenerate_nodes}, node_locked_gpus={self.degenerate_locked_gpus}"
        )


def run_command(command: list[str]) -> str:
    """Run a command and return stdout.

    Inputs:
    - `command`: argument-vector command.

    Outputs:
    - Command stdout text.
    """

    result = subprocess.run(args=command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr.strip()
    return result.stdout


def parse_fields(line: str) -> dict[str, str]:
    """Convert one Slurm `-o` line into a key-value mapping."""

    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_partition_names(value: str) -> tuple[str, ...]:
    """Parse normalized partition names from Slurm partition text.

    Inputs:
    - `value`: partition field text such as `"gpu,quad"` or `"gpu*"`.

    Outputs:
    - Lowercase unique partition names with no default `*` suffix.
    """

    if value in NONE_VALUES:
        return ()
    names: list[str] = []
    for raw_token in value.split(","):
        cleaned = raw_token.strip().rstrip("*").lower()
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return tuple(names)


def parse_datetime(value: str) -> datetime | None:
    """Parse Slurm datetime text or return `None` when unknown."""

    if value in NONE_VALUES:
        return None
    return datetime.strptime(value, DATETIME_FORMAT)


def parse_duration_hours(value: str) -> float:
    """Parse Slurm duration text into hours.

    Inputs:
    - `value`: `HH:MM:SS`, `D-HH:MM:SS`, or `UNLIMITED`.

    Outputs:
    - Duration in hours.
    """

    if value == "UNLIMITED":
        return 24.0 * 365.0
    days = 0
    clock = value
    if "-" in value:
        day_text, clock = value.split("-", 1)
        days = int(day_text)
    hours_text, minutes_text, seconds_text = clock.split(":")
    hours = int(hours_text)
    minutes = int(minutes_text)
    seconds = int(seconds_text)
    return (days * 24.0) + hours + (minutes / 60.0) + (seconds / 3600.0)


def parse_gpu_count(tres_text: str) -> int:
    """Read total GPU count from a TRES field.

    Inputs:
    - `tres_text`: TRES token string (e.g., `ReqTRES=...` value).

    Outputs:
    - GPU count.
    """

    generic_match = re.search(pattern=r"(?:^|,)gres/gpu=(\d+)", string=tres_text)
    if generic_match is not None:
        return int(generic_match.group(1))
    model_matches = re.findall(pattern=r"(?:^|,)gres/gpu:[^=,]+=(\d+)", string=tres_text)
    return sum(int(match_text) for match_text in model_matches)


def parse_tres_int(tres_text: str, key: str) -> int:
    """Extract integer TRES value for one key, or 0 when missing."""

    match = re.search(pattern=rf"(?:^|,){re.escape(key)}=(\d+)", string=tres_text)
    return int(match.group(1)) if match is not None else 0


def size_to_mib(number_text: str, unit: str) -> int:
    """Convert size token to MiB."""

    factor_by_unit = {
        "": 1 / 1024,
        "K": 1 / (1024 * 1024),
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
        "P": 1024 * 1024 * 1024,
    }
    factor = factor_by_unit.get(unit, 1)
    return int(round(float(number_text) * factor))


def parse_tres_mem_mib(tres_text: str) -> int:
    """Read memory TRES value as MiB."""

    match = re.search(pattern=r"(?:^|,)mem=([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", string=tres_text)
    if match is None:
        return 0
    return size_to_mib(number_text=match.group(1), unit=match.group(2))


def infer_node_expression(state: str, fields: dict[str, str]) -> str | None:
    """Select node expression source by job state."""

    keys = ("NodeList", "SchedNodeList") if state == "RUNNING" else ("SchedNodeList", "NodeList")
    for key in keys:
        value = fields.get(key, "")
        if value not in NONE_VALUES:
            return value
    return None


def infer_requested_nodes(fields: dict[str, str]) -> int:
    """Infer requested node count from TRES or `NumNodes`."""

    req_nodes = parse_tres_int(tres_text=fields.get("ReqTRES", ""), key="node")
    if req_nodes > 0:
        return req_nodes
    match = re.search(pattern=r"\d+", string=fields.get("NumNodes", "1"))
    return int(match.group(0)) if match is not None else 1


def parse_job_record(fields: dict[str, str]) -> JobRecord | None:
    """Build a `JobRecord` when the job is active and requests GPUs."""

    state = fields.get("JobState", "")
    if state not in ACTIVE_STATES:
        return None
    req_tres = fields.get("ReqTRES", "")
    requested_gpus = parse_gpu_count(tres_text=req_tres)
    if requested_gpus <= 0:
        return None
    return JobRecord(
        job_id=int(fields["JobId"]),
        state=state,
        requested_gpus=requested_gpus,
        allocated_gpus=parse_gpu_count(tres_text=fields.get("AllocTRES", "")),
        start_time=parse_datetime(value=fields.get("StartTime", "")),
        end_time=parse_datetime(value=fields.get("EndTime", "")),
        time_limit_hours=parse_duration_hours(value=fields.get("TimeLimit", "00:00:00")),
        run_time_hours=parse_duration_hours(value=fields.get("RunTime", "00:00:00")),
        requested_cpus=parse_tres_int(tres_text=req_tres, key="cpu"),
        requested_mem_mib=parse_tres_mem_mib(tres_text=req_tres),
        requested_nodes=infer_requested_nodes(fields=fields),
        node_expression=infer_node_expression(state=state, fields=fields),
        partition_names=parse_partition_names(value=fields.get("Partition", "")),
    )


def window_from_record(record: JobRecord, now: datetime, projected_gpus: int) -> JobWindow | None:
    """Convert one `JobRecord` into a future-facing occupancy window."""

    if projected_gpus <= 0:
        return None
    if record.state == "RUNNING":
        start = now
        fallback_end = now + timedelta(hours=max(record.time_limit_hours - record.run_time_hours, 0.0))
        end = record.end_time if record.end_time is not None else fallback_end
    else:
        if record.start_time is None:
            return None
        start = max(record.start_time, now)
        fallback_end = start + timedelta(hours=record.time_limit_hours)
        end = record.end_time if record.end_time is not None else fallback_end
    if end <= start:
        return None
    return JobWindow(job_id=record.job_id, state=record.state, gpus=projected_gpus, start=start, end=end)


def parse_node_capacities(raw_nodes: str) -> dict[str, NodeCapacity]:
    """Parse node capacity objects from `scontrol show nodes -o`."""

    capacities: dict[str, NodeCapacity] = {}
    for line in raw_nodes.splitlines():
        fields = parse_fields(line=line)
        cfg_tres = fields.get("CfgTRES", "")
        gpu_count = parse_gpu_count(tres_text=cfg_tres)
        if gpu_count <= 0:
            continue
        node_name = fields.get("NodeName", "")
        cpu_count = parse_tres_int(tres_text=cfg_tres, key="cpu")
        mem_mib = parse_tres_mem_mib(tres_text=cfg_tres)
        alloc_tres = fields.get("AllocTRES", "")
        cpu_alloc = int(fields.get("CPUAlloc", "0"))
        if cpu_alloc <= 0:
            cpu_alloc = parse_tres_int(tres_text=alloc_tres, key="cpu")
        mem_alloc_mib = int(fields.get("AllocMem", "0"))
        if mem_alloc_mib <= 0:
            mem_alloc_mib = parse_tres_mem_mib(tres_text=alloc_tres)
        gpu_alloc = parse_gpu_count(tres_text=alloc_tres)
        if node_name and cpu_count > 0 and mem_mib > 0:
            capacities[node_name] = NodeCapacity(
                node_name=node_name,
                cpu=cpu_count,
                mem_mib=mem_mib,
                gpus=gpu_count,
                cpu_alloc=cpu_alloc,
                mem_alloc_mib=mem_alloc_mib,
                gpu_alloc=gpu_alloc,
                partition_names=parse_partition_names(value=fields.get("Partitions", "")),
            )
    return capacities


def expand_nodelist(node_expression: str | None, cache: dict[str, list[str]]) -> list[str]:
    """Expand one Slurm node expression into hostnames."""

    if node_expression is None or node_expression in NONE_VALUES:
        return []
    if node_expression in cache:
        return cache[node_expression]
    hosts = [
        host.strip()
        for host in run_command(command=["scontrol", "show", "hostname", node_expression]).splitlines()
        if host.strip()
    ]
    cache[node_expression] = hosts
    return hosts


def is_full_node_by_resources(record: JobRecord, capacities: list[NodeCapacity]) -> bool:
    """Return true when CPU or memory request effectively consumes full node(s)."""

    if not capacities:
        return False
    node_count = record.requested_nodes if record.requested_nodes > 0 else len(capacities)
    cpu_per_node = record.requested_cpus / node_count if record.requested_cpus > 0 else 0.0
    mem_per_node = record.requested_mem_mib / node_count if record.requested_mem_mib > 0 else 0.0
    for capacity in capacities:
        if cpu_per_node >= capacity.cpu:
            return True
        if mem_per_node >= capacity.mem_mib * 0.98:
            return True
    return False


def adjusted_projected_gpus(
    record: JobRecord, node_capacities: dict[str, NodeCapacity], host_cache: dict[str, list[str]]
) -> tuple[int, bool, int]:
    """Return degenerate-aware projected GPUs and adjustment flags."""

    base_gpus = record.projected_gpus()
    hosts = expand_nodelist(node_expression=record.node_expression, cache=host_cache)
    capacities = [node_capacities[host] for host in hosts if host in node_capacities]
    if not capacities:
        return base_gpus, False, 0
    total_node_gpus = sum(capacity.gpus for capacity in capacities)
    if base_gpus >= total_node_gpus:
        return base_gpus, False, 0
    if not is_full_node_by_resources(record=record, capacities=capacities):
        return base_gpus, False, 0
    return total_node_gpus, True, total_node_gpus - base_gpus


def running_end_time(record: JobRecord, now: datetime) -> datetime:
    """Return projected running-job end timestamp."""

    if record.end_time is not None:
        return record.end_time
    hours_left = max(record.time_limit_hours - record.run_time_hours, 0.0)
    return now + timedelta(hours=hours_left)


def running_jobs_by_node(
    records: list[JobRecord], node_capacities: dict[str, NodeCapacity], now: datetime
) -> dict[str, list[tuple[int, datetime]]]:
    """Map each node to running jobs and their projected end times."""

    mapping: defaultdict[str, list[tuple[int, datetime]]] = defaultdict(list)
    host_cache: dict[str, list[str]] = {}
    for record in records:
        if record.state != "RUNNING":
            continue
        end_time = running_end_time(record=record, now=now)
        if end_time <= now:
            continue
        hosts = expand_nodelist(node_expression=record.node_expression, cache=host_cache)
        for host in hosts:
            if host in node_capacities:
                mapping[host].append((record.job_id, end_time))
    return dict(mapping)


def degenerate_node_lock_windows(
    records: list[JobRecord], node_capacities: dict[str, NodeCapacity], now: datetime
) -> tuple[list[JobWindow], int, int]:
    """Build lock windows for running nodes blocked by CPU/MEM but not GPUs.

    Inputs:
    - `records`: active GPU job records.
    - `node_capacities`: per-node capacity and allocated resource values.
    - `now`: forecast anchor timestamp.

    Outputs:
    - `(lock_windows, degenerate_nodes, locked_gpus_total)`.
    """

    lock_windows: list[JobWindow] = []
    degenerate_nodes = 0
    locked_gpus_total = 0
    mapping = running_jobs_by_node(records=records, node_capacities=node_capacities, now=now)
    for node_name, jobs in mapping.items():
        capacity = node_capacities[node_name]
        free_gpus = capacity.gpus - capacity.gpu_alloc
        is_cpu_full = capacity.cpu_alloc >= capacity.cpu
        is_mem_full = capacity.mem_alloc_mib >= int(capacity.mem_mib * 0.98)
        if free_gpus <= 0 or not (is_cpu_full or is_mem_full):
            continue
        if len({job_id for job_id, _ in jobs}) < 2:
            continue
        unlock_time = min(end_time for _, end_time in jobs)
        if unlock_time <= now:
            continue
        lock_windows.append(
            JobWindow(job_id=-(degenerate_nodes + 1), state="RUNNING_LOCK", gpus=free_gpus, start=now, end=unlock_time)
        )
        degenerate_nodes += 1
        locked_gpus_total += free_gpus
    return lock_windows, degenerate_nodes, locked_gpus_total


def parse_job_records(raw_jobs: str) -> list[JobRecord]:
    """Parse active GPU `JobRecord` objects from raw Slurm job text."""

    records: list[JobRecord] = []
    for line in raw_jobs.splitlines():
        record = parse_job_record(fields=parse_fields(line=line))
        if record is not None:
            records.append(record)
    return records


def partition_node_capacities(
    node_capacities: dict[str, NodeCapacity], partition_name: str
) -> dict[str, NodeCapacity]:
    """Select node capacities belonging to one partition.

    Inputs:
    - `node_capacities`: full cluster node-capacity mapping.
    - `partition_name`: partition label to keep.

    Outputs:
    - Subset map containing only nodes that host the partition.
    """

    target = partition_name.lower()
    return {
        node_name: capacity
        for node_name, capacity in node_capacities.items()
        if target in capacity.partition_names
    }


def record_targets_partition(
    record: JobRecord,
    partition_name: str,
    partition_node_names: set[str],
    host_cache: dict[str, list[str]],
    infer_quad_large_gpu: bool,
) -> bool:
    """Return true when a job should contribute to one partition forecast."""

    target = partition_name.lower()
    hosts = expand_nodelist(node_expression=record.node_expression, cache=host_cache)
    if hosts:
        if any(host in partition_node_names for host in hosts):
            return True
        if record.state == "RUNNING":
            return False
    if target in record.partition_names:
        return True
    if record.partition_names:
        return False
    if infer_quad_large_gpu and target == "quad" and record.requested_gpus > 3:
        return True
    return False


def filter_records_for_partition(
    records: list[JobRecord],
    partition_name: str,
    partition_node_names: set[str],
    host_cache: dict[str, list[str]],
    infer_quad_large_gpu: bool,
) -> list[JobRecord]:
    """Filter records to jobs expected to occupy one partition."""

    return [
        record
        for record in records
        if record_targets_partition(
            record=record,
            partition_name=partition_name,
            partition_node_names=partition_node_names,
            host_cache=host_cache,
            infer_quad_large_gpu=infer_quad_large_gpu,
        )
    ]


def collect_job_windows(
    raw_jobs: str,
    now: datetime,
    node_capacities: dict[str, NodeCapacity],
    target_partition: str | None = None,
    infer_quad_large_gpu: bool = False,
) -> tuple[list[JobWindow], ForecastStats]:
    """Extract forecast windows and stats from raw Slurm job text.

    Inputs:
    - `raw_jobs`: output from `scontrol show jobs -o`.
    - `now`: forecast anchor timestamp.
    - `node_capacities`: per-node CPU/memory/GPU capacity map.
    - `target_partition`: optional partition filter for partition-specific forecasting.
    - `infer_quad_large_gpu`: when true, treat jobs with `requested_gpus > 3` as quad-eligible.

    Outputs:
    - Tuple of `(windows, stats)`.

    Example:
    - `collect_job_windows(raw_jobs=run_command([...]), now=datetime.now(), node_capacities=nodes)`
    """

    host_cache: dict[str, list[str]] = {}
    records = parse_job_records(raw_jobs=raw_jobs)
    active_capacities = node_capacities
    if target_partition is not None:
        active_capacities = partition_node_capacities(
            node_capacities=node_capacities,
            partition_name=target_partition,
        )
        records = filter_records_for_partition(
            records=records,
            partition_name=target_partition,
            partition_node_names=set(active_capacities.keys()),
            host_cache=host_cache,
            infer_quad_large_gpu=infer_quad_large_gpu,
        )
    windows: list[JobWindow] = []
    degenerate_jobs = 0
    degenerate_extra_gpus = 0
    for record in records:
        projected_gpus, is_degenerate, extra_gpus = adjusted_projected_gpus(
            record=record,
            node_capacities=active_capacities,
            host_cache=host_cache,
        )
        if is_degenerate:
            degenerate_jobs += 1
            degenerate_extra_gpus += extra_gpus
        window = window_from_record(record=record, now=now, projected_gpus=projected_gpus)
        if window is not None:
            windows.append(window)
    lock_windows, degenerate_nodes, degenerate_locked_gpus = degenerate_node_lock_windows(
        records=records, node_capacities=active_capacities, now=now
    )
    windows.extend(lock_windows)
    pending_records = [record for record in records if record.state == "PENDING"]
    pending_with_start = [record for record in pending_records if record.start_time is not None]
    stats = ForecastStats(
        active_gpu_jobs=len(records),
        running_gpu_jobs=sum(1 for record in records if record.state == "RUNNING"),
        pending_gpu_jobs=len(pending_records),
        pending_with_start=len(pending_with_start),
        pending_without_start=len(pending_records) - len(pending_with_start),
        forecast_windows=len(windows),
        degenerate_jobs=degenerate_jobs,
        degenerate_extra_gpus=degenerate_extra_gpus,
        degenerate_nodes=degenerate_nodes,
        degenerate_locked_gpus=degenerate_locked_gpus,
    )
    return windows, stats


def build_event_deltas(windows: list[JobWindow], now: datetime) -> tuple[int, list[tuple[datetime, int]]]:
    """Build baseline usage and future event deltas from forecast windows."""

    baseline = 0
    events: list[tuple[datetime, int]] = []
    for window in windows:
        if window.start <= now < window.end:
            baseline += window.gpus
            events.append((window.end, -window.gpus))
            continue
        if window.start > now:
            events.append((window.start, window.gpus))
            events.append((window.end, -window.gpus))
    return baseline, events


def group_event_deltas(events: list[tuple[datetime, int]]) -> list[tuple[datetime, int]]:
    """Collapse events sharing identical timestamps."""

    grouped: defaultdict[datetime, int] = defaultdict(int)
    for event_time, delta in events:
        grouped[event_time] += delta
    return sorted(grouped.items(), key=lambda item: item[0])


def choose_horizon(windows: list[JobWindow], now: datetime, horizon_hours: float | None) -> datetime:
    """Return plotting horizon based on CLI cap or latest scheduled end."""

    if horizon_hours is not None:
        return now + timedelta(hours=max(horizon_hours, 1.0))
    end_times = [window.end for window in windows if window.end > now]
    if not end_times:
        return now + timedelta(hours=6.0)
    return max(end_times)


def build_step_series(
    now: datetime, baseline: int, grouped_events: list[tuple[datetime, int]], horizon: datetime
) -> tuple[list[datetime], list[int]]:
    """Convert baseline+events into stepwise timeseries points."""

    times = [now]
    usage = [baseline]
    current = baseline
    for event_time, delta in grouped_events:
        if event_time < now or event_time > horizon:
            continue
        times.append(event_time)
        usage.append(current)
        current += delta
        times.append(event_time)
        usage.append(current)
    if times[-1] < horizon:
        times.append(horizon)
        usage.append(current)
    return times, usage


def total_gpu_capacity(node_capacities: dict[str, NodeCapacity]) -> int:
    """Sum cluster GPU capacity from parsed node capacities."""

    return sum(capacity.gpus for capacity in node_capacities.values())


def partition_gpu_capacity(node_capacities: dict[str, NodeCapacity], partition_name: str) -> int:
    """Return total GPU capacity hosted by one partition.

    Inputs:
    - `node_capacities`: full cluster node-capacity mapping.
    - `partition_name`: partition label to aggregate.

    Outputs:
    - Sum of schedulable GPUs on nodes in the partition.
    """

    partition_nodes = partition_node_capacities(
        node_capacities=node_capacities,
        partition_name=partition_name,
    )
    return total_gpu_capacity(node_capacities=partition_nodes)


def format_relative_hours(hours_from_now: float) -> str:
    """Format relative-hour tick label text like `+0.5h` or `+2h`."""

    rounded = round(hours_from_now * 2.0) / 2.0
    if abs(rounded - int(rounded)) < 1e-9:
        return f"+{int(rounded)}h"
    return f"+{rounded:.1f}h"


def build_relative_halfhour_ticks(now: datetime, horizon: datetime) -> tuple[list[datetime], list[str]]:
    """Build half-hour x-ticks from `now` to `horizon` with relative labels."""

    assert horizon > now, "Horizon must be after now."
    elapsed_seconds = (horizon - now).total_seconds()
    step_seconds = HALF_HOUR_MINUTES * 60
    full_steps = int(elapsed_seconds // step_seconds)
    ticks = [now + timedelta(minutes=HALF_HOUR_MINUTES * idx) for idx in range(full_steps + 1)]
    labels = [format_relative_hours(hours_from_now=0.5 * idx) for idx in range(full_steps + 1)]
    if ticks[-1] < horizon:
        ticks.append(horizon)
        labels.append(format_relative_hours(hours_from_now=elapsed_seconds / 3600.0))
    return ticks, labels


def available_series(usage: list[int], capacity: int) -> list[int]:
    """Convert usage values into non-negative available GPU values."""

    return [max(capacity - used_gpus, 0) for used_gpus in usage]
