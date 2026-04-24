from __future__ import annotations

import curses
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Sequence, Set

from slurm_cli.forecast_cli import (
    DashForecastBundle,
    DebugForecastMarker,
    ForecastPoint,
    ForecastSnapshot,
    step_value_at,
)


FORECAST_HORIZON_HOURS = 8.0
FORECAST_REFRESH_SECONDS = 10


@dataclass(frozen=True)
class ForecastRenderState:
    """One render-safe snapshot of forecast worker state.

    Inputs:
    - `bundle`: last successful forecast bundle or `None` if unavailable.
    - `message`: status text for loading or error conditions.
    - `is_loading`: whether a compute is currently in flight.

    Outputs:
    - Immutable forecast render state safe for dashboard drawing.
    """

    bundle: Optional[DashForecastBundle]
    message: str
    is_loading: bool


class DashboardForecastRenderer:
    """Render dashboard forecast panels inside an existing curses layout.

    Inputs:
    - Color-pair identifiers for pending/running/error emphasis.

    Outputs:
    - Rendering helpers for one or two forecast panels.
    """

    def __init__(self, pending_pair: int, running_pair: int, error_pair: int) -> None:
        self.pending_pair = pending_pair
        self.running_pair = running_pair
        self.error_pair = error_pair

    def draw_forecast_area(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        state: ForecastRenderState,
    ) -> None:
        """Draw one or two forecast charts in the lower dashboard region."""

        if bottom < top:
            return
        _, width = stdscr.getmaxyx()
        left = 1
        right = max(left, width - 2)
        if state.bundle is None:
            self._draw_single_forecast_panel(
                stdscr=stdscr,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                snapshot=None,
                debug_marker=None,
                state=state,
            )
            return
        quad_snapshot = state.bundle.quad_partition
        if quad_snapshot is None:
            self._draw_single_forecast_panel(
                stdscr=stdscr,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                snapshot=state.bundle.all_gpus,
                debug_marker=state.bundle.all_debug_marker,
                state=state,
            )
            return
        split = split_panel_columns(left=left, right=right, gap=3)
        if split is None:
            self._draw_single_forecast_panel(
                stdscr=stdscr,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                snapshot=state.bundle.all_gpus,
                debug_marker=state.bundle.all_debug_marker,
                state=state,
            )
            return
        self._draw_dual_forecast_panels(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            split=split,
            all_snapshot=state.bundle.all_gpus,
            all_debug_marker=state.bundle.all_debug_marker,
            quad_snapshot=quad_snapshot,
            quad_debug_marker=state.bundle.quad_debug_marker,
            state=state,
        )

    def _draw_single_forecast_panel(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
        snapshot: ForecastSnapshot | None,
        debug_marker: DebugForecastMarker | None,
        state: ForecastRenderState,
    ) -> None:
        """Draw one full-width primary forecast panel."""

        self._draw_forecast_panel(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            title="GPU Availability Forecast (8h, all GPUs)",
            snapshot=snapshot,
            debug_marker=debug_marker,
            state=state,
        )

    def _draw_dual_forecast_panels(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        split: tuple[int, int, int, int],
        all_snapshot: ForecastSnapshot,
        all_debug_marker: DebugForecastMarker | None,
        quad_snapshot: ForecastSnapshot,
        quad_debug_marker: DebugForecastMarker | None,
        state: ForecastRenderState,
    ) -> None:
        """Draw side-by-side primary and quad forecast panels."""

        left_left, left_right, right_left, right_right = split
        self._draw_forecast_panel(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            left=left_left,
            right=left_right,
            title="GPU Availability Forecast (8h, all GPUs)",
            snapshot=all_snapshot,
            debug_marker=all_debug_marker,
            state=state,
        )
        self._draw_forecast_panel(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            left=right_left,
            right=right_right,
            title="Quad Partition Forecast (8h, <=4GPU nodes)",
            snapshot=quad_snapshot,
            debug_marker=quad_debug_marker,
            state=state,
        )

    def _draw_forecast_panel(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
        title: str,
        snapshot: ForecastSnapshot | None,
        debug_marker: DebugForecastMarker | None,
        state: ForecastRenderState,
    ) -> None:
        """Draw one forecast panel with title, debug label, and chart."""

        if right < left or bottom < top:
            return
        panel_title = self._title_with_availability(title=title, snapshot=snapshot)
        panel_add(
            stdscr=stdscr,
            y=top,
            x=left,
            text=panel_title,
            left=left,
            right=right,
            attr=curses.A_BOLD,
        )
        if snapshot is None:
            self._draw_forecast_placeholder(
                stdscr=stdscr,
                top=top,
                left=left,
                right=right,
                state=state,
            )
            return
        chart_top = top + 1
        if debug_marker is not None:
            panel_add(
                stdscr=stdscr,
                y=top + 1,
                x=left,
                text=f"1-GPU debug ETA: {debug_marker.label()}",
                left=left,
                right=right,
                attr=curses.color_pair(self.pending_pair) | curses.A_BOLD,
            )
            chart_top += 1
        self._draw_forecast_chart(
            stdscr=stdscr,
            top=chart_top,
            bottom=bottom,
            left=left,
            right=right,
            snapshot=snapshot,
            debug_marker=debug_marker,
        )

    def _draw_forecast_placeholder(
        self,
        stdscr: "curses.window",
        top: int,
        left: int,
        right: int,
        state: ForecastRenderState,
    ) -> None:
        """Draw placeholder text when forecast data is unavailable."""

        attr = (
            curses.A_DIM
            if state.is_loading
            else curses.color_pair(self.error_pair) | curses.A_BOLD
        )
        message = state.message or "Loading forecast..."
        panel_add(
            stdscr=stdscr,
            y=top + 1,
            x=left,
            text=message,
            left=left,
            right=right,
            attr=attr,
        )

    def _draw_forecast_chart(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
        snapshot: ForecastSnapshot,
        debug_marker: DebugForecastMarker | None,
    ) -> None:
        """Draw one forecast chart within fixed panel bounds."""

        if bottom - top + 1 < 6 or right - left + 1 < 20:
            panel_add(
                stdscr=stdscr,
                y=top,
                x=left,
                text="Forecast panel too small.",
                left=left,
                right=right,
                attr=curses.A_DIM,
            )
            return
        chart_height = bottom - top - 2
        vmax_seed = snapshot.max_available()
        axis_width = max(6, len(str(max(1, vmax_seed))) + 3)
        chart_left = left + axis_width
        chart_width = right - chart_left + 1
        if chart_width < 12:
            panel_add(
                stdscr=stdscr,
                y=top,
                x=left,
                text="Panel too narrow.",
                left=left,
                right=right,
                attr=curses.A_DIM,
            )
            return
        points = _dense_panel_points(snapshot=snapshot, count=chart_width)
        vmax = max(1, max(point.available_gpus for point in points))
        for idx, point in enumerate(points):
            x = chart_left + idx
            y = _plot_y_for_value(
                top=top,
                height=chart_height,
                value=point.available_gpus,
                vmax=vmax,
            )
            panel_add(
                stdscr=stdscr,
                y=y,
                x=x,
                text="*",
                left=left,
                right=right,
                attr=self._availability_attr(point.available_gpus),
            )
        if debug_marker is not None:
            self._draw_debug_marker(
                stdscr=stdscr,
                top=top,
                chart_height=chart_height,
                chart_left=chart_left,
                chart_width=chart_width,
                snapshot=snapshot,
                debug_marker=debug_marker,
                left=left,
                right=right,
            )
        self._draw_forecast_y_ticks(
            stdscr=stdscr,
            top=top,
            chart_height=chart_height,
            vmax=vmax,
            left=left,
            right=right,
        )
        baseline_y = top + chart_height
        panel_add(
            stdscr=stdscr,
            y=baseline_y,
            x=chart_left,
            text="-" * chart_width,
            left=left,
            right=right,
            attr=curses.A_DIM,
        )
        self._draw_forecast_x_ticks(
            stdscr=stdscr,
            baseline_y=baseline_y,
            snapshot=snapshot,
            chart_left=chart_left,
            chart_width=chart_width,
            left=left,
            right=right,
        )

    def _draw_debug_marker(
        self,
        stdscr: "curses.window",
        top: int,
        chart_height: int,
        chart_left: int,
        chart_width: int,
        snapshot: ForecastSnapshot,
        debug_marker: DebugForecastMarker,
        left: int,
        right: int,
    ) -> None:
        """Draw a vertical marker for the earliest debug allocation ETA.

        Markers beyond the visible horizon clamp to the right edge so the
        panel still shows that a debug ETA exists outside the current window.
        """

        horizon_hours = (
            snapshot.points[-1].offset_hours
            if snapshot.points
            else FORECAST_HORIZON_HOURS
        )
        if horizon_hours <= 0:
            return
        ratio = min(max(debug_marker.offset_hours() / horizon_hours, 0.0), 1.0)
        marker_x = chart_left + int(round(ratio * max(0, chart_width - 1)))
        attr = curses.color_pair(self.pending_pair) | curses.A_BOLD
        for y in range(top, top + chart_height):
            panel_add(
                stdscr=stdscr,
                y=y,
                x=marker_x,
                text="|",
                left=left,
                right=right,
                attr=attr,
            )

    def _draw_forecast_y_ticks(
        self,
        stdscr: "curses.window",
        top: int,
        chart_height: int,
        vmax: int,
        left: int,
        right: int,
    ) -> None:
        """Draw non-overlapping y-axis labels for the chart."""

        used_rows: Set[int] = set()
        values = sorted(
            _forecast_y_tick_values(vmax=vmax, chart_height=chart_height),
            reverse=True,
        )
        for value in values:
            y = _plot_y_for_value(top=top, height=chart_height, value=value, vmax=vmax)
            if y in used_rows:
                continue
            used_rows.add(y)
            panel_add(
                stdscr=stdscr,
                y=y,
                x=left,
                text=f"{value:>3} |",
                left=left,
                right=right,
                attr=curses.A_DIM,
            )

    def _draw_forecast_x_ticks(
        self,
        stdscr: "curses.window",
        baseline_y: int,
        snapshot: ForecastSnapshot,
        chart_left: int,
        chart_width: int,
        left: int,
        right: int,
    ) -> None:
        """Draw x-axis ticks, labels, and sampled values."""

        labels = [point.label() for point in snapshot.points]
        values = [point.available_gpus for point in snapshot.points]
        xs = _tick_positions(left=chart_left, width=chart_width, count=len(labels))
        for x in xs:
            panel_add(
                stdscr=stdscr,
                y=baseline_y,
                x=x,
                text="+",
                left=left,
                right=right,
                attr=curses.A_DIM,
            )
        last_end = left - 1
        for x, label in zip(xs, labels):
            start = max(left, x - (len(label) // 2))
            if start <= last_end + 1:
                continue
            panel_add(
                stdscr=stdscr,
                y=baseline_y + 1,
                x=start,
                text=label,
                left=left,
                right=right,
                attr=curses.A_DIM,
            )
            last_end = start + len(label) - 1
        for x, value in zip(xs, values):
            text = str(value)
            panel_add(
                stdscr=stdscr,
                y=baseline_y + 2,
                x=max(left, x - (len(text) // 2)),
                text=text,
                left=left,
                right=right,
                attr=self._availability_attr(value),
            )

    def _availability_attr(self, available: int) -> int:
        """Return the color emphasis for one forecast availability value."""

        if available <= 0:
            return curses.color_pair(self.error_pair) | curses.A_BOLD
        if available <= 4:
            return curses.color_pair(self.pending_pair) | curses.A_BOLD
        return curses.color_pair(self.running_pair) | curses.A_BOLD

    def _title_with_availability(
        self,
        title: str,
        snapshot: ForecastSnapshot | None,
    ) -> str:
        """Return the chart title decorated with live availability metrics."""

        if snapshot is None:
            return title
        return f"{title} [{snapshot.title_metrics()}]"


def safe_add(
    stdscr: "curses.window",
    y: int,
    x: int,
    text: str,
    attr: int = 0,
) -> None:
    """Write clipped text safely inside the full screen bounds."""

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


def panel_add(
    stdscr: "curses.window",
    y: int,
    x: int,
    text: str,
    left: int,
    right: int,
    attr: int = 0,
) -> None:
    """Write text clipped to one horizontal panel region."""

    if right < left or x > right:
        return
    draw_x = max(left, x)
    offset = draw_x - x
    clipped = text[offset : offset + max(0, right - draw_x + 1)]
    if not clipped:
        return
    safe_add(stdscr=stdscr, y=y, x=draw_x, text=clipped, attr=attr)


def center_text_range(
    stdscr: "curses.window",
    y: int,
    text: str,
    left: int,
    right: int,
    attr: int = 0,
) -> None:
    """Center one text line within a bounded dashboard panel."""

    width = right - left + 1
    if width <= 0:
        return
    x_offset = max(0, (width - len(text)) // 2)
    panel_add(
        stdscr=stdscr,
        y=y,
        x=left + x_offset,
        text=text,
        left=left,
        right=right,
        attr=attr,
    )


def split_panel_columns(
    left: int,
    right: int,
    gap: int,
) -> tuple[int, int, int, int] | None:
    """Split one horizontal region into two equal-width panels."""

    total_width = right - left + 1
    min_width = 36
    if total_width < (2 * min_width) + gap:
        return None
    left_width = (total_width - gap) // 2
    left_right = left + left_width - 1
    right_left = left_right + gap + 1
    return left, left_right, right_left, right


def forecast_panel_top(screen_height: int) -> int:
    """Return the top row used by the lower-half forecast area."""

    minimum_top = 6
    target = max(minimum_top, screen_height // 2)
    return min(target, max(minimum_top, screen_height - 6))


def format_duration_short(minutes: float) -> str:
    """Return a short duration string for blame-panel averages."""

    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def prepare_screen(stdscr: "curses.window") -> None:
    """Initialize the curses screen background and clear prior contents."""

    curses.start_color()
    if curses.has_colors():
        curses.use_default_colors()
        curses.init_pair(1, -1, curses.COLOR_BLACK)
        stdscr.bkgd(" ", curses.color_pair(1))
    else:
        stdscr.bkgd(" ", curses.A_NORMAL)
    stdscr.clear()
    stdscr.refresh()


def _dense_panel_points(snapshot: ForecastSnapshot, count: int) -> List[ForecastPoint]:
    """Sample dense chart points from a step-series snapshot."""

    if count <= 1:
        base = snapshot.points[0].available_gpus if snapshot.points else 0
        return [ForecastPoint(offset_hours=0.0, available_gpus=base)]
    horizon_hours = (
        snapshot.points[-1].offset_hours if snapshot.points else FORECAST_HORIZON_HOURS
    )
    points: List[ForecastPoint] = []
    for idx in range(count):
        offset_hours = horizon_hours * idx / (count - 1)
        query = snapshot.generated_at + timedelta(hours=offset_hours)
        available = step_value_at(
            query=query,
            times=snapshot.series_times,
            values=snapshot.series_available,
        )
        points.append(
            ForecastPoint(offset_hours=offset_hours, available_gpus=available)
        )
    return points


def _plot_y_for_value(top: int, height: int, value: int, vmax: int) -> int:
    """Map one forecast value onto a row inside the chart area."""

    if vmax <= 0:
        return top + height - 1
    clipped = max(0, min(value, vmax))
    ratio = clipped / vmax
    return top + (height - 1) - int(round(ratio * (height - 1)))


def _forecast_y_tick_values(vmax: int, chart_height: int) -> List[int]:
    """Return unique y-axis label values for one chart height."""

    tick_count = max(3, min(10, chart_height))
    if tick_count <= 1:
        return [0]
    values = [int(round((idx * vmax) / (tick_count - 1))) for idx in range(tick_count)]
    unique_values = sorted(set(values))
    if not unique_values or unique_values[0] != 0:
        unique_values.insert(0, 0)
    if unique_values[-1] != vmax:
        unique_values.append(vmax)
    return unique_values


def _tick_positions(left: int, width: int, count: int) -> List[int]:
    """Return evenly spaced x positions across a chart width."""

    if count <= 1:
        return [left]
    return [left + int(round(idx * (width - 1) / (count - 1))) for idx in range(count)]
