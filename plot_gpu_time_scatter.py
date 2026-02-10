#!/usr/bin/env python3
"""Create a scatter plot of Slurm GPU requests vs job time signal.

Inputs:
- Live job data from `scontrol show jobs -o`.
- CLI filters for maximum hours, maximum GPUs, and partition selection.

Outputs:
- A PNG scatter plot written to disk.

Example:
    python plot_gpu_time_scatter.py \
      --max-hours 8 \
      --max-gpus 4 \
      --partitions gpu \
      --output time_vs_gpus_scatter_max8h_4gpus.png
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

RUNNING_STATES = {"RUNNING"}
QUEUED_STATES = {"PENDING"}


@dataclass(frozen=True)
class JobPoint:
    """Point-ready Slurm job sample.

    Inputs:
    - `job_id`: numeric Slurm job id.
    - `state`: `RUNNING` or `QUEUED`.
    - `hours`: x-axis time signal in hours.
      - Queued jobs: requested walltime.
      - Running jobs: remaining walltime.
    - `gpus`: requested GPU count.

    Outputs:
    - Immutable data object used for plotting and summaries.
    """

    job_id: int
    state: str
    hours: float
    gpus: int

    def label(self) -> str:
        """Return canonical state text for legends and summaries."""
        return "Running" if self.state == "RUNNING" else "Queued"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting filters and output path.

    Inputs:
    - Command-line flags.

    Outputs:
    - `argparse.Namespace` with `max_hours`, `max_gpus`, `partitions`, `output`.
    """

    parser = argparse.ArgumentParser(
        description="Plot requested time vs GPUs for running and queued Slurm jobs."
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Only include jobs with plotted time signal <= this many hours.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=4,
        help="Only include jobs requesting <= this many GPUs.",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default="gpu",
        help='Comma-separated partitions to include (use "all" for no partition filter).',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("time_vs_gpus_scatter_max8h_4gpus.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--jitter-hours",
        type=float,
        default=0.08,
        help="Uniform jitter radius on x-axis (hours).",
    )
    parser.add_argument(
        "--jitter-gpus",
        type=float,
        default=0.08,
        help="Uniform jitter radius on y-axis (GPU count units).",
    )
    parser.add_argument(
        "--alpha-queued",
        type=float,
        default=0.45,
        help="Point alpha for queued jobs (0..1).",
    )
    parser.add_argument(
        "--alpha-running",
        type=float,
        default=0.55,
        help="Point alpha for running jobs (0..1).",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    """Run a shell command and return stdout as text.

    Inputs:
    - `command`: subprocess argument vector.

    Outputs:
    - Command stdout.
    """

    completed = subprocess.run(args=command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr.strip()
    return completed.stdout


def extract_field(line: str, key: str) -> str:
    """Extract a `key=value` token from one `scontrol show jobs -o` line."""

    match = re.search(pattern=rf"\b{re.escape(key)}=([^\s]+)", string=line)
    assert match is not None, f"Missing field '{key}' in: {line}"
    return match.group(1)


def parse_hours(time_limit: str) -> float:
    """Convert Slurm TimeLimit text to hours.

    Inputs:
    - `time_limit`: values like `HH:MM:SS`, `D-HH:MM:SS`, or `UNLIMITED`.

    Outputs:
    - Duration in hours (`float('inf')` for `UNLIMITED`).
    """

    if time_limit == "UNLIMITED":
        return float("inf")
    days = 0
    clock = time_limit
    if "-" in time_limit:
        day_text, clock = time_limit.split(sep="-", maxsplit=1)
        days = int(day_text)
    pieces = [int(part) for part in clock.split(":")]
    if len(pieces) == 3:
        hours, minutes, seconds = pieces
    elif len(pieces) == 2:
        hours, minutes, seconds = 0, pieces[0], pieces[1]
    elif len(pieces) == 1:
        hours, minutes, seconds = 0, pieces[0], 0
    else:
        assert False, f"Unexpected TimeLimit format: {time_limit}"
    return (days * 24) + hours + (minutes / 60.0) + (seconds / 3600.0)


def parse_gpu_count(req_tres: str) -> int:
    """Read total requested GPUs from a Slurm `ReqTRES` token."""

    generic_match = re.search(pattern=r"(?:^|,)gres/gpu=(\d+)", string=req_tres)
    if generic_match is not None:
        return int(generic_match.group(1))
    model_matches = re.findall(pattern=r"(?:^|,)gres/gpu:[^=,]+=(\d+)", string=req_tres)
    return sum(int(match_text) for match_text in model_matches)


def plotted_hours_for_state(line: str, state_text: str) -> float:
    """Return plotted hours for one job line.

    Inputs:
    - `line`: one `scontrol show jobs -o` record.
    - `state_text`: raw Slurm job state.

    Outputs:
    - Queued jobs: requested time limit in hours.
    - Running jobs: remaining hours (`TimeLimit - RunTime`, floored at 0).
    """

    time_limit_hours = parse_hours(time_limit=extract_field(line=line, key="TimeLimit"))
    if state_text in RUNNING_STATES:
        run_time_hours = parse_hours(time_limit=extract_field(line=line, key="RunTime"))
        return max(time_limit_hours - run_time_hours, 0.0)
    return time_limit_hours


def parse_partition_filter(partition_text: str) -> set[str] | None:
    """Turn CLI partition text into a filter set, or `None` for no filter."""

    lowered = partition_text.strip().lower()
    if lowered in {"", "all", "*"}:
        return None
    selected = {part.strip() for part in partition_text.split(",") if part.strip()}
    assert selected, "Partition filter resolved to an empty set."
    return selected


def build_points(
    scontrol_output: str,
    max_hours: float,
    max_gpus: int,
    partition_filter: set[str] | None,
) -> list[JobPoint]:
    """Build filtered plot points from `scontrol show jobs -o` output.

    Inputs:
    - `scontrol_output`: raw multiline output from `scontrol show jobs -o`.
    - `max_hours`: keep jobs with requested walltime <= this threshold.
    - `max_gpus`: keep jobs requesting <= this GPU count.
    - `partition_filter`: selected partitions, or `None` for all partitions.

    Outputs:
    - `list[JobPoint]` ready for plotting.

    Example:
    - `build_points(output, max_hours=8.0, max_gpus=4, partition_filter={"gpu"})`
    """

    points: list[JobPoint] = []
    allowed_states = RUNNING_STATES | QUEUED_STATES
    for line in scontrol_output.splitlines():
        if not line.strip():
            continue
        state_text = extract_field(line=line, key="JobState")
        if state_text not in allowed_states:
            continue
        partition = extract_field(line=line, key="Partition")
        if partition_filter is not None and partition not in partition_filter:
            continue
        req_tres = extract_field(line=line, key="ReqTRES")
        requested_gpus = parse_gpu_count(req_tres=req_tres)
        if requested_gpus == 0 or requested_gpus > max_gpus:
            continue
        plotted_hours = plotted_hours_for_state(line=line, state_text=state_text)
        if plotted_hours > max_hours:
            continue
        state_label = "RUNNING" if state_text == "RUNNING" else "QUEUED"
        job_id = int(extract_field(line=line, key="JobId"))
        points.append(
            JobPoint(job_id=job_id, state=state_label, hours=plotted_hours, gpus=requested_gpus)
        )
    return points


def split_points_by_state(points: list[JobPoint]) -> tuple[list[JobPoint], list[JobPoint]]:
    """Split points into running and queued lists."""

    running_points = [point for point in points if point.state == "RUNNING"]
    queued_points = [point for point in points if point.state == "QUEUED"]
    return running_points, queued_points


def deterministic_jitter(job_id: int, amplitude: float, salt: int) -> float:
    """Return deterministic jitter in `[-amplitude, amplitude]` for one point."""

    if amplitude <= 0:
        return 0.0
    raw = ((job_id * 1103515245 + salt) % 2147483647) / 2147483647
    return ((2.0 * raw) - 1.0) * amplitude


def jittered_coordinates(
    points: list[JobPoint],
    jitter_hours: float,
    jitter_gpus: float,
    max_hours: float,
    max_gpus: int,
) -> tuple[list[float], list[float]]:
    """Build jittered x/y values with clipping to visible axis bounds."""

    x_values: list[float] = []
    y_values: list[float] = []
    for point in points:
        x_jitter = deterministic_jitter(job_id=point.job_id, amplitude=jitter_hours, salt=17)
        y_jitter = deterministic_jitter(job_id=point.job_id, amplitude=jitter_gpus, salt=53)
        x_values.append(min(max(point.hours + x_jitter, 0.0), max_hours))
        y_values.append(min(max(point.gpus + y_jitter, 0.0), float(max_gpus)))
    return x_values, y_values


def plot_points(
    points: list[JobPoint],
    max_hours: float,
    max_gpus: int,
    output_path: Path,
    jitter_hours: float,
    jitter_gpus: float,
    alpha_queued: float,
    alpha_running: float,
) -> None:
    """Render and save a scatter plot for time vs GPUs.

    Inputs:
    - `points`: filtered job points.
    - `max_hours`: x-axis maximum and filter value.
    - `max_gpus`: y-axis maximum and filter value.
    - `output_path`: PNG file destination.

    Outputs:
    - Writes a PNG file.

    Example:
    - `plot_points(points=points, max_hours=8.0, max_gpus=4, output_path=Path("plot.png"))`
    """

    assert points, "No jobs matched the requested filters."
    running_points, queued_points = split_points_by_state(points=points)
    queued_x, queued_y = jittered_coordinates(
        points=queued_points,
        jitter_hours=jitter_hours,
        jitter_gpus=jitter_gpus,
        max_hours=max_hours,
        max_gpus=max_gpus,
    )
    running_x, running_y = jittered_coordinates(
        points=running_points,
        jitter_hours=jitter_hours,
        jitter_gpus=jitter_gpus,
        max_hours=max_hours,
        max_gpus=max_gpus,
    )
    figure: Figure = plt.figure(figsize=(10, 6))
    axis = cast(Axes, figure.add_subplot(1, 1, 1))
    axis.scatter(
        x=queued_x,
        y=queued_y,
        c="#d62728",
        alpha=alpha_queued,
        s=56,
        label=f"Queued ({len(queued_points)})",
    )
    axis.scatter(
        x=running_x,
        y=running_y,
        c="#1f77b4",
        alpha=alpha_running,
        s=56,
        label=f"Running ({len(running_points)})",
    )
    axis.set_xlim(left=0.0, right=max_hours + 0.1)
    axis.set_ylim(bottom=0.0, top=max_gpus + 0.2)
    axis.set_xlabel("Time (queued=requested, running=time-left) [hours]")
    axis.set_ylabel("Requested GPUs")
    axis.set_title(f"Slurm GPU Jobs (<= {max_hours:g}h, <= {max_gpus} GPUs)")
    axis.grid(visible=True, alpha=0.25, linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    """Generate the plot from live Slurm job metadata."""

    arguments = parse_args()
    partition_filter = parse_partition_filter(partition_text=arguments.partitions)
    slurm_jobs_output = run_command(command=["scontrol", "show", "jobs", "-o"])
    points = build_points(
        scontrol_output=slurm_jobs_output,
        max_hours=arguments.max_hours,
        max_gpus=arguments.max_gpus,
        partition_filter=partition_filter,
    )
    plot_points(
        points=points,
        max_hours=arguments.max_hours,
        max_gpus=arguments.max_gpus,
        output_path=arguments.output,
        jitter_hours=arguments.jitter_hours,
        jitter_gpus=arguments.jitter_gpus,
        alpha_queued=arguments.alpha_queued,
        alpha_running=arguments.alpha_running,
    )
    print(f"Wrote {arguments.output} with {len(points)} points.")


if __name__ == "__main__":
    main()
