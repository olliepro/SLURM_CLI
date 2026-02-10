#!/usr/bin/env python3
"""Terminal CLI for 8-hour GPU availability forecast.

Inputs:
- Live Slurm jobs and nodes via `scontrol`.
- Optional refresh and horizon flags.

Outputs:
- Curses dashboard showing degenerate-corrected GPU availability over time.

Example:
    python forecast_cli.py --refresh-seconds 10
"""

from __future__ import annotations

import argparse
import curses
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from slurm_cli.forecast_core import (
    ForecastStats,
    NodeCapacity,
    available_series,
    build_event_deltas,
    build_step_series,
    collect_job_windows,
    group_event_deltas,
    partition_gpu_capacity,
    parse_node_capacities,
    run_command,
    total_gpu_capacity,
)


HALF_HOUR = 0.5
PAIR_OK = 2
PAIR_WARN = 3
PAIR_ERR = 4


@dataclass(frozen=True)
class ForecastPoint:
    """One sampled forecast point.

    Inputs:
    - `offset_hours`: relative time from now.
    - `available_gpus`: available GPU count at this offset.

    Outputs:
    - Immutable point suitable for chart rendering.
    """

    offset_hours: float
    available_gpus: int

    def label(self) -> str:
        """Return canonical tick text, e.g. `+0.5h`."""

        rounded = round(self.offset_hours * 2.0) / 2.0
        return f"+{int(rounded)}h" if float(rounded).is_integer() else f"+{rounded:.1f}h"


@dataclass(frozen=True)
class ForecastSnapshot:
    """Snapshot payload for one dashboard refresh cycle.

    Inputs:
    - Generation timestamp, capacity, sampled points, full step-series, and forecast stats.

    Outputs:
    - Immutable snapshot object with display helpers.
    """

    generated_at: datetime
    capacity: int
    points: list[ForecastPoint]
    series_times: list[datetime]
    series_available: list[int]
    stats: ForecastStats

    def min_available(self) -> int:
        """Return minimum available GPUs in the current horizon."""

        return min((point.available_gpus for point in self.points), default=0)

    def max_available(self) -> int:
        """Return maximum available GPUs in the current horizon."""

        return max((point.available_gpus for point in self.points), default=0)

    def current_available(self) -> int:
        """Return available GPU count at forecast start (`+0h`)."""

        if not self.points:
            return 0
        return self.points[0].available_gpus

    def availability_fraction(self) -> str:
        """Return canonical current availability text as `avail/total`."""

        return f"{self.current_available()}/{self.capacity}"


@dataclass(frozen=True)
class DashForecastBundle:
    """Dashboard forecast payload with optional partition-specific snapshot.

    Inputs:
    - `all_gpus`: cluster-wide forecast snapshot.
    - `quad_partition`: quad-partition forecast snapshot when available.

    Outputs:
    - Immutable dashboard-ready forecast bundle.
    """

    all_gpus: ForecastSnapshot
    quad_partition: ForecastSnapshot | None


@dataclass(frozen=True)
class ChartGeometry:
    """Rendered chart geometry for aligned tick/label placement.

    Inputs:
    - `left`: left x-origin of the chart area.
    - `width`: chart width in columns.
    - `baseline_y`: y-row of the x-axis baseline.

    Outputs:
    - Immutable geometry used for aligned x-ticks and values.
    """

    left: int
    width: int
    baseline_y: int


def parse_args() -> argparse.Namespace:
    """Parse CLI options.

    Inputs:
    - Command line flags.

    Outputs:
    - Namespace with refresh cadence and horizon in hours.
    """

    parser = argparse.ArgumentParser(description="Curses 8h GPU availability dashboard.")
    parser.add_argument("--refresh-seconds", type=int, default=10, help="Auto refresh period in seconds.")
    parser.add_argument("--horizon-hours", type=float, default=8.0, help="Forecast horizon in hours.")
    parser.add_argument("--once", action="store_true", help="Print one text snapshot and exit.")
    return parser.parse_args()


def half_hour_offsets(horizon_hours: float) -> list[float]:
    """Build relative hour offsets at 30-minute spacing."""

    steps = max(1, int(round(horizon_hours / HALF_HOUR)))
    return [idx * HALF_HOUR for idx in range(steps + 1)]


def step_value_at(query: datetime, times: Sequence[datetime], values: Sequence[int]) -> int:
    """Sample a step-post series at one timestamp."""

    assert len(times) == len(values), "times and values length mismatch"
    value = values[0]
    for series_time, series_value in zip(times, values):
        if series_time > query:
            break
        value = series_value
    return value


def fetch_slurm_state() -> tuple[str, str]:
    """Fetch raw Slurm job and node text used by forecast builders."""

    raw_jobs = run_command(command=["scontrol", "show", "jobs", "-o"])
    raw_nodes = run_command(command=["scontrol", "show", "nodes", "-o"])
    return raw_jobs, raw_nodes


def build_forecast_series(
    now: datetime,
    horizon_hours: float,
    raw_jobs: str,
    node_capacities: dict[str, NodeCapacity],
    target_partition: str | None = None,
    infer_quad_large_gpu: bool = False,
) -> tuple[list[datetime], list[int], int, ForecastStats]:
    """Build one forecast series from parsed capacities and raw jobs."""

    capacities = node_capacities
    windows, stats = collect_job_windows(
        raw_jobs=raw_jobs,
        now=now,
        node_capacities=capacities,
        target_partition=target_partition,
        infer_quad_large_gpu=infer_quad_large_gpu,
    )
    capacity = (
        partition_gpu_capacity(node_capacities=capacities, partition_name=target_partition)
        if target_partition is not None
        else total_gpu_capacity(node_capacities=capacities)
    )
    baseline, events = build_event_deltas(windows=windows, now=now)
    grouped_events = group_event_deltas(events=events)
    horizon = now + timedelta(hours=max(horizon_hours, HALF_HOUR))
    times, usage = build_step_series(now=now, baseline=baseline, grouped_events=grouped_events, horizon=horizon)
    return times, available_series(usage=usage, capacity=capacity), capacity, stats


def snapshot_from_series(
    now: datetime, horizon_hours: float, times: list[datetime], available: list[int], capacity: int, stats: ForecastStats
) -> ForecastSnapshot:
    """Build sampled `ForecastSnapshot` from full step-series arrays."""

    points = [
        ForecastPoint(
            offset_hours=offset,
            available_gpus=step_value_at(query=now + timedelta(hours=offset), times=times, values=available),
        )
        for offset in half_hour_offsets(horizon_hours=horizon_hours)
    ]
    return ForecastSnapshot(
        generated_at=now,
        capacity=capacity,
        points=points,
        series_times=times,
        series_available=available,
        stats=stats,
    )


def build_snapshot(
    now: datetime,
    horizon_hours: float,
    raw_jobs: str,
    node_capacities: dict[str, NodeCapacity],
    target_partition: str | None = None,
    infer_quad_large_gpu: bool = False,
) -> ForecastSnapshot:
    """Build one forecast snapshot from pre-fetched Slurm state."""

    times, available, capacity, stats = build_forecast_series(
        now=now,
        horizon_hours=horizon_hours,
        raw_jobs=raw_jobs,
        node_capacities=node_capacities,
        target_partition=target_partition,
        infer_quad_large_gpu=infer_quad_large_gpu,
    )
    return snapshot_from_series(
        now=now,
        horizon_hours=horizon_hours,
        times=times,
        available=available,
        capacity=capacity,
        stats=stats,
    )


def take_snapshot(horizon_hours: float) -> ForecastSnapshot:
    """Take one live cluster-wide forecast snapshot.

    Inputs:
    - `horizon_hours`: forecast horizon.

    Outputs:
    - `ForecastSnapshot` containing half-hour sampled availability points.

    Example:
    - `snapshot = take_snapshot(horizon_hours=8.0)`
    """

    now = datetime.now()
    raw_jobs, raw_nodes = fetch_slurm_state()
    node_capacities = parse_node_capacities(raw_nodes=raw_nodes)
    return build_snapshot(
        now=now,
        horizon_hours=horizon_hours,
        raw_jobs=raw_jobs,
        node_capacities=node_capacities,
    )


def take_dash_forecast_bundle(horizon_hours: float) -> DashForecastBundle:
    """Take dashboard forecast bundle with cluster and optional quad views.

    Inputs:
    - `horizon_hours`: forecast horizon in hours.

    Outputs:
    - `DashForecastBundle` with all-GPU snapshot and optional quad snapshot.

    Example:
    - `bundle = take_dash_forecast_bundle(horizon_hours=8.0)`
    """

    now = datetime.now()
    raw_jobs, raw_nodes = fetch_slurm_state()
    node_capacities = parse_node_capacities(raw_nodes=raw_nodes)
    all_snapshot = build_snapshot(
        now=now,
        horizon_hours=horizon_hours,
        raw_jobs=raw_jobs,
        node_capacities=node_capacities,
    )
    quad_capacity = partition_gpu_capacity(node_capacities=node_capacities, partition_name="quad")
    if quad_capacity <= 0:
        return DashForecastBundle(all_gpus=all_snapshot, quad_partition=None)
    quad_snapshot = build_snapshot(
        now=now,
        horizon_hours=horizon_hours,
        raw_jobs=raw_jobs,
        node_capacities=node_capacities,
        target_partition="quad",
        infer_quad_large_gpu=True,
    )
    return DashForecastBundle(all_gpus=all_snapshot, quad_partition=quad_snapshot)


def _safe_add(stdscr: "curses.window", y: int, x: int, text: str, attr: int = 0) -> None:
    """Write text with clipping and curses-error safety."""

    height, width = stdscr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    clipped = text[: max(0, width - x - 1)]
    if not clipped:
        return
    try:
        stdscr.addstr(y, x, clipped, attr)
    except curses.error:
        return


def _init_colors() -> None:
    """Initialize color pairs when terminal supports colors."""

    curses.start_color()
    if not curses.has_colors():
        return
    curses.use_default_colors()
    curses.init_pair(PAIR_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(PAIR_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(PAIR_ERR, curses.COLOR_RED, -1)


def _value_attr(available: int) -> int:
    """Map availability value to a display attribute."""

    if available <= 0:
        return curses.color_pair(PAIR_ERR) | curses.A_BOLD
    if available <= 4:
        return curses.color_pair(PAIR_WARN) | curses.A_BOLD
    return curses.color_pair(PAIR_OK) | curses.A_BOLD


def _draw_header(stdscr: "curses.window", snapshot: ForecastSnapshot, refresh_seconds: int) -> int:
    """Draw dashboard header block and return next content row."""

    _safe_add(stdscr, 0, 1, "GPU Availability Forecast (8h, degenerate-corrected)", curses.A_BOLD)
    generated = snapshot.generated_at.strftime("%Y-%m-%d %H:%M:%S")
    summary = f"Generated: {generated} | Capacity: {snapshot.capacity} | Refresh: {refresh_seconds}s"
    _safe_add(stdscr, 1, 1, summary, curses.A_DIM)
    detail = (
        f"min={snapshot.min_available()} max={snapshot.max_available()} "
        f"| degenerate_jobs={snapshot.stats.degenerate_jobs} "
        f"| node_locks={snapshot.stats.degenerate_nodes}"
    )
    _safe_add(stdscr, 2, 1, detail, curses.A_DIM)
    _safe_add(stdscr, 3, 1, "Keys: r refresh | q quit", curses.A_DIM)
    return 5


def _plot_x_for_index(left: int, width: int, index: int, total: int) -> int:
    """Map point index to x-position within chart bounds."""

    if total <= 1:
        return left
    return left + int(round(index * (width - 1) / (total - 1)))


def _plot_y_for_value(top: int, height: int, value: int, vmax: int) -> int:
    """Map availability value to y-position within chart bounds."""

    if vmax <= 0:
        return top + height - 1
    clipped = max(0, min(value, vmax))
    ratio = clipped / vmax
    return top + (height - 1) - int(round(ratio * (height - 1)))


def _y_tick_values(vmax: int, chart_height: int) -> list[int]:
    """Return y-axis tick values spanning `0..vmax` for a chart height.

    Inputs:
    - `vmax`: maximum plotted y-value.
    - `chart_height`: chart height in rows.

    Outputs:
    - Monotonic unique tick values including both endpoints.
    """

    tick_count = max(3, min(10, chart_height))
    if tick_count == 1:
        return [0]
    values = [int(round((idx * vmax) / (tick_count - 1))) for idx in range(tick_count)]
    unique_values = sorted(set(values))
    if not unique_values or unique_values[0] != 0:
        unique_values.insert(0, 0)
    if unique_values[-1] != vmax:
        unique_values.append(vmax)
    return unique_values


def _draw_y_ticks(
    stdscr: "curses.window", top: int, chart_height: int, vmax: int
) -> None:
    """Draw y-axis labels for top, bottom, and intermediate tick rows."""

    used_rows: set[int] = set()
    for value in sorted(_y_tick_values(vmax=vmax, chart_height=chart_height), reverse=True):
        y = _plot_y_for_value(top=top, height=chart_height, value=value, vmax=vmax)
        if y in used_rows:
            continue
        used_rows.add(y)
        _safe_add(stdscr, y, 1, f"{value:>3} |", curses.A_DIM)


def _dense_chart_points(snapshot: ForecastSnapshot, count: int) -> list[ForecastPoint]:
    """Sample dense chart points from full step-series across the horizon."""

    if count <= 1:
        base_value = snapshot.points[0].available_gpus if snapshot.points else 0
        return [ForecastPoint(offset_hours=0.0, available_gpus=base_value)]
    horizon_hours = snapshot.points[-1].offset_hours if snapshot.points else 8.0
    points: list[ForecastPoint] = []
    for idx in range(count):
        offset_hours = horizon_hours * idx / (count - 1)
        query = snapshot.generated_at + timedelta(hours=offset_hours)
        available = step_value_at(query=query, times=snapshot.series_times, values=snapshot.series_available)
        points.append(ForecastPoint(offset_hours=offset_hours, available_gpus=available))
    return points


def _draw_chart(stdscr: "curses.window", top: int, snapshot: ForecastSnapshot) -> ChartGeometry:
    """Draw an ASCII line chart and return chart geometry."""

    height, width = stdscr.getmaxyx()
    chart_height = max(8, min(16, height - top - 8))
    chart_left = 10
    chart_width = max(20, width - chart_left - 2)
    points = _dense_chart_points(snapshot=snapshot, count=chart_width)
    vmax = max(1, max(point.available_gpus for point in points))
    for idx, point in enumerate(points):
        x = chart_left + idx
        y = _plot_y_for_value(top=top, height=chart_height, value=point.available_gpus, vmax=vmax)
        _safe_add(stdscr, y, x, "*", _value_attr(available=point.available_gpus))
    _draw_y_ticks(stdscr=stdscr, top=top, chart_height=chart_height, vmax=vmax)
    baseline_y = top + chart_height
    _safe_add(stdscr, baseline_y, chart_left, "-" * chart_width, curses.A_DIM)
    return ChartGeometry(left=chart_left, width=chart_width, baseline_y=baseline_y)


def _tick_positions(left: int, width: int, count: int) -> list[int]:
    """Return x-positions distributed across the full chart width."""

    if count <= 1:
        return [left]
    return [left + int(round(idx * (width - 1) / (count - 1))) for idx in range(count)]


def _draw_tick_labels(stdscr: "curses.window", snapshot: ForecastSnapshot, geometry: ChartGeometry) -> int:
    """Draw aligned x-ticks and labels across the full chart width."""

    labels = [point.label() for point in snapshot.points]
    values = [point.available_gpus for point in snapshot.points]
    xs = _tick_positions(left=geometry.left, width=geometry.width, count=len(labels))
    for x in xs:
        _safe_add(stdscr, geometry.baseline_y, x, "+", curses.A_DIM)
    last_end = 0
    for x, label in zip(xs, labels):
        start = max(1, x - (len(label) // 2))
        if start <= last_end + 1:
            continue
        _safe_add(stdscr, geometry.baseline_y + 1, start, label, curses.A_DIM)
        last_end = start + len(label) - 1
    for x, value in zip(xs, values):
        text = str(value)
        _safe_add(
            stdscr,
            geometry.baseline_y + 2,
            max(1, x - (len(text) // 2)),
            text,
            _value_attr(available=value),
        )
    return geometry.baseline_y + 2


def _draw(stdscr: "curses.window", snapshot: ForecastSnapshot, refresh_seconds: int) -> None:
    """Render one full dashboard frame."""

    stdscr.erase()
    row = _draw_header(stdscr=stdscr, snapshot=snapshot, refresh_seconds=refresh_seconds)
    geometry = _draw_chart(stdscr=stdscr, top=row, snapshot=snapshot)
    _draw_tick_labels(stdscr=stdscr, snapshot=snapshot, geometry=geometry)
    stdscr.refresh()


def run_curses(refresh_seconds: int, horizon_hours: float) -> int:
    """Run the interactive curses dashboard loop.

    Inputs:
    - `refresh_seconds`: auto-refresh period.
    - `horizon_hours`: forecast horizon.

    Outputs:
    - Process exit status code.
    """

    def _main(stdscr: "curses.window") -> int:
        _init_colors()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        snapshot = take_snapshot(horizon_hours=horizon_hours)
        next_refresh = time.monotonic() + refresh_seconds
        while True:
            _draw(stdscr=stdscr, snapshot=snapshot, refresh_seconds=refresh_seconds)
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("r"), ord("R")):
                snapshot = take_snapshot(horizon_hours=horizon_hours)
                next_refresh = time.monotonic() + refresh_seconds
            if time.monotonic() >= next_refresh:
                snapshot = take_snapshot(horizon_hours=horizon_hours)
                next_refresh = time.monotonic() + refresh_seconds
            time.sleep(0.05)

    try:
        return curses.wrapper(_main)
    except curses.error:
        return 2


def run_once(horizon_hours: float) -> int:
    """Print one snapshot in plain text for non-interactive environments."""

    snapshot = take_snapshot(horizon_hours=horizon_hours)
    print(f"Generated: {snapshot.generated_at:%Y-%m-%d %H:%M:%S}")
    print(f"Capacity: {snapshot.capacity}")
    print(snapshot.stats.subtitle())
    for point in snapshot.points:
        print(f"{point.label():>6}  {point.available_gpus:>3}")
    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    refresh_seconds = max(1, args.refresh_seconds)
    if args.once:
        return run_once(horizon_hours=args.horizon_hours)
    return run_curses(refresh_seconds=refresh_seconds, horizon_hours=args.horizon_hours)


if __name__ == "__main__":
    raise SystemExit(main())
