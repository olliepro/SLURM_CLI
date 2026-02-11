from __future__ import annotations

import curses
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Sequence, Set

from slurm_cli.dash_logic import (
    DashActionResult,
    DashJob,
    cancel_dash_jobs,
    fetch_dash_jobs,
    join_job_via_remote,
)
from slurm_cli.forecast_cli import (
    DashForecastBundle,
    ForecastPoint,
    ForecastSnapshot,
    step_value_at,
    take_dash_forecast_bundle,
)


PAIR_PENDING = 2
PAIR_RUNNING = 3
PAIR_ERROR = 4
FORECAST_HORIZON_HOURS = 8.0
FORECAST_REFRESH_SECONDS = 10


@dataclass(frozen=True)
class ForecastRenderState:
    """One render-safe snapshot of forecast worker state.

    Inputs:
    - `bundle`: last successful 8-hour forecast bundle or ``None`` if unavailable.
    - `message`: status message for loading or error conditions.
    - `is_loading`: whether a compute is currently in-flight.

    Outputs:
    - Immutable state object safe for UI rendering without in-place mutation.

    Example:
    - `ForecastRenderState(bundle=None, message="Loading forecast...", is_loading=True)`
    """

    bundle: Optional[DashForecastBundle]
    message: str
    is_loading: bool


class DashBoard:
    """Interactive dashboard for pending/running job management.

    Args:
        user_name: Username used for `squeue -u` filtering.
        refresh_seconds: Automatic refresh interval.
        editor_command: Optional editor command/alias for `v` join actions.

    Returns:
        Process exit code: 0 on normal quit/join success, 2 on recoverable failures.
    """

    def __init__(
        self,
        user_name: str,
        refresh_seconds: int = 2,
        editor_command: Optional[str] = None,
    ):
        self.user_name = user_name
        self.refresh_seconds = max(1, refresh_seconds)
        self.editor_command = editor_command
        self.jobs: List[DashJob] = []
        self.focus_index = 0
        self.selected_job_ids: Set[str] = set()
        self.status_message = "Loading jobs..."
        self.forecast_horizon_hours = FORECAST_HORIZON_HOURS
        self.forecast_refresh_seconds = FORECAST_REFRESH_SECONDS
        self._forecast_bundle: Optional[DashForecastBundle] = None
        self._forecast_message = "Loading forecast..."
        self._forecast_loading = True
        self._forecast_lock = threading.Lock()
        self._forecast_refresh_event = threading.Event()
        self._forecast_stop_event = threading.Event()
        self._forecast_thread: Optional[threading.Thread] = None

    def run(self) -> int:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_loop()

    def update_jobs(
        self,
        jobs: Sequence[DashJob],
        focus_job_id: Optional[str] = None,
    ) -> None:
        target_id = focus_job_id or self.current_job_id()
        self.jobs = list(jobs)
        visible_ids = {job.job_id for job in self.jobs}
        self.selected_job_ids = {job_id for job_id in self.selected_job_ids if job_id in visible_ids}
        self.focus_index = self._index_for_job(job_id=target_id)

    def toggle_selected_current(self) -> None:
        job = self.current_job()
        if job is None:
            return
        if job.job_id in self.selected_job_ids:
            self.selected_job_ids.remove(job.job_id)
            return
        self.selected_job_ids.add(job.job_id)

    def toggle_all_jobs(self) -> None:
        if not self.jobs:
            return
        all_ids = {job.job_id for job in self.jobs}
        self.selected_job_ids = set() if self.selected_job_ids == all_ids else all_ids

    def can_join_current(self) -> bool:
        job = self.current_job()
        return job is not None and job.is_running()

    def _curses_main(self, stdscr: "curses.window") -> int:
        _prepare_screen(stdscr=stdscr)
        self._init_colors()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._start_forecast_worker()
        next_refresh = 0.0
        try:
            while True:
                self._draw(stdscr=stdscr)
                result = self._handle_key(stdscr=stdscr, key=stdscr.getch())
                if result is not None:
                    return result
                next_refresh = self._maybe_refresh(next_refresh=next_refresh)
                time.sleep(0.05)
        finally:
            self._stop_forecast_worker()

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.init_pair(PAIR_PENDING, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(PAIR_RUNNING, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(PAIR_ERROR, curses.COLOR_RED, curses.COLOR_BLACK)

    def _refresh_jobs(self) -> None:
        focus_job_id = self.current_job_id()
        try:
            jobs = fetch_dash_jobs(user_name=self.user_name)
            self.status_message = self._refresh_message(job_count=len(jobs))
        except Exception as exc:
            jobs = []
            self.status_message = f"Refresh failed: {exc}"
        self.update_jobs(jobs=jobs, focus_job_id=focus_job_id)

    def _refresh_message(self, job_count: int) -> str:
        if job_count == 0:
            return f"No pending/running jobs for {self.user_name}."
        return f"Loaded {job_count} jobs."

    def _maybe_refresh(self, next_refresh: float) -> float:
        now = time.monotonic()
        if now < next_refresh:
            return next_refresh
        self._refresh_jobs()
        return now + self.refresh_seconds

    def _start_forecast_worker(self) -> None:
        """Launch the background 8-hour forecast refresher thread.

        Inputs:
        - None.

        Outputs:
        - Starts daemon worker and queues immediate forecast computation.
        """

        if self._forecast_thread and self._forecast_thread.is_alive():
            return
        self._forecast_stop_event.clear()
        self._forecast_refresh_event.set()
        self._forecast_thread = threading.Thread(target=self._forecast_loop, name="dash-forecast", daemon=True)
        self._forecast_thread.start()

    def _stop_forecast_worker(self) -> None:
        """Stop the forecast refresher thread before curses exits.

        Inputs:
        - None.

        Outputs:
        - Signals worker shutdown and joins briefly.
        """

        self._forecast_stop_event.set()
        self._forecast_refresh_event.set()
        if self._forecast_thread is not None:
            self._forecast_thread.join(timeout=1.0)
        self._forecast_thread = None

    def _request_forecast_refresh(self) -> None:
        """Trigger an immediate background refresh for forecast state.

        Inputs:
        - None.

        Outputs:
        - Sets worker refresh event.
        """

        self._forecast_refresh_event.set()

    def _set_forecast_loading(self) -> None:
        """Mark forecast state as loading while retaining last good snapshot."""

        with self._forecast_lock:
            self._forecast_loading = True
            if self._forecast_bundle is None:
                self._forecast_message = "Loading forecast..."

    def _set_forecast_ready(self, bundle: DashForecastBundle) -> None:
        """Store a successful forecast bundle and clear transient messages."""

        with self._forecast_lock:
            self._forecast_bundle = bundle
            self._forecast_message = ""
            self._forecast_loading = False

    def _set_forecast_error(self, error_text: str) -> None:
        """Capture background forecast failures for placeholder display."""

        with self._forecast_lock:
            self._forecast_loading = False
            if self._forecast_bundle is None:
                self._forecast_message = f"Forecast failed: {error_text}"

    def _forecast_state(self) -> ForecastRenderState:
        """Return a thread-safe forecast render state snapshot.

        Inputs:
        - None.

        Outputs:
        - Immutable `ForecastRenderState` copied from shared worker state.
        """

        with self._forecast_lock:
            return ForecastRenderState(
                bundle=self._forecast_bundle,
                message=self._forecast_message,
                is_loading=self._forecast_loading,
            )

    def _forecast_loop(self) -> None:
        """Run periodic forecast refresh with on-demand trigger support.

        Inputs:
        - None. Uses configured horizon and refresh period.

        Outputs:
        - Continuously updates shared forecast state until stop event is set.
        """

        next_refresh = 0.0
        while not self._forecast_stop_event.is_set():
            wait_seconds = max(0.0, next_refresh - time.monotonic())
            self._forecast_refresh_event.wait(timeout=wait_seconds)
            force_refresh = self._forecast_refresh_event.is_set()
            self._forecast_refresh_event.clear()
            if self._forecast_stop_event.is_set():
                break
            if not force_refresh and time.monotonic() < next_refresh:
                continue
            self._set_forecast_loading()
            try:
                bundle = take_dash_forecast_bundle(horizon_hours=self.forecast_horizon_hours)
            except Exception as exc:
                self._set_forecast_error(error_text=str(exc))
                next_refresh = time.monotonic() + self.forecast_refresh_seconds
                continue
            self._set_forecast_ready(bundle=bundle)
            next_refresh = time.monotonic() + self.forecast_refresh_seconds

    def _handle_key(self, stdscr: "curses.window", key: int) -> Optional[int]:
        if key in (ord("q"), ord("Q")):
            return 0
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            self._move_focus(delta=-1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            self._move_focus(delta=1)
        elif key == ord(" "):
            self.toggle_selected_current()
        elif key in (ord("a"), ord("A")):
            self.toggle_all_jobs()
        elif key in (ord("r"), ord("R")):
            self._refresh_jobs()
            self._request_forecast_refresh()
        elif key in (ord("c"), ord("C")):
            self._cancel_in_ui(stdscr=stdscr)
        elif key in (ord("v"), ord("V")):
            return self._join_from_ui()
        return None

    def _move_focus(self, delta: int) -> None:
        if not self.jobs:
            self.focus_index = 0
            return
        upper = len(self.jobs) - 1
        self.focus_index = (self.focus_index + delta) % (upper + 1)

    def _cancel_in_ui(self, stdscr: "curses.window") -> None:
        job_ids = self._target_job_ids()
        if not job_ids:
            self.status_message = "No job selected."
            return
        if not self._confirm(stdscr=stdscr, prompt=f"Cancel {len(job_ids)} job(s)? [y/N]"):
            self.status_message = "Cancel aborted."
            return
        result = cancel_dash_jobs(job_ids=job_ids)
        self.status_message = result.summary_line()
        self._refresh_jobs()

    def _join_from_ui(self) -> Optional[int]:
        job = self.current_job()
        if job is None:
            self.status_message = "No focused job."
            return None
        result = join_job_via_remote(job=job, editor=self.editor_command)
        self.status_message = result.summary_line()
        return 0 if result.ok else None

    def _target_job_ids(self) -> List[str]:
        if self.selected_job_ids:
            return sorted(self.selected_job_ids)
        focused = self.current_job_id()
        return [focused] if focused else []

    def _confirm(self, stdscr: "curses.window", prompt: str) -> bool:
        stdscr.nodelay(False)
        try:
            self.status_message = prompt
            self._draw(stdscr=stdscr)
            key = stdscr.getch()
            return key in (ord("y"), ord("Y"))
        finally:
            stdscr.nodelay(True)

    def _draw(self, stdscr: "curses.window") -> None:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        forecast_top = _forecast_panel_top(screen_height=height)
        _center_text(stdscr=stdscr, y=0, text=f"dash: {self.user_name}", attr=curses.A_BOLD)
        _center_text(
            stdscr=stdscr,
            y=1,
            text="R=green PD=yellow • Space=select • c=cancel • v=join • r=refresh • q=quit",
            attr=curses.A_DIM,
        )
        _safe_add(stdscr=stdscr, y=forecast_top - 1, x=1, text="-" * max(1, width - 2), attr=curses.A_DIM)
        self._draw_rows(stdscr=stdscr, top=3, bottom=forecast_top - 2)
        self._draw_forecast_area(stdscr=stdscr, top=forecast_top, bottom=height - 2)
        self._draw_status(stdscr=stdscr)
        stdscr.refresh()

    def _draw_rows(self, stdscr: "curses.window", top: int, bottom: int) -> None:
        if bottom < top:
            return
        if not self.jobs:
            row = top + max(0, (bottom - top) // 2)
            _center_text(stdscr=stdscr, y=row, text="No jobs to display.", attr=curses.A_DIM)
            return
        _, width = stdscr.getmaxyx()
        for row_idx, job in enumerate(self.jobs):
            y = top + row_idx
            if y > bottom:
                break
            marker = "*" if job.job_id in self.selected_job_ids else " "
            prefix = ">" if row_idx == self.focus_index else " "
            text = f"{prefix}[{marker}] {job.display_row()}"
            attr = self._row_attr(job=job, focused=row_idx == self.focus_index)
            _safe_add(stdscr=stdscr, y=y, x=1, text=text[: max(1, width - 2)], attr=attr)

    def _draw_forecast_area(self, stdscr: "curses.window", top: int, bottom: int) -> None:
        """Draw one or two forecast charts in the lower dashboard half."""

        if bottom < top:
            return
        _, width = stdscr.getmaxyx()
        left = 1
        right = max(left, width - 2)
        state = self._forecast_state()
        if state.bundle is None:
            self._draw_single_forecast_panel(
                stdscr=stdscr, top=top, bottom=bottom, left=left, right=right, snapshot=None, state=state
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
                state=state,
            )
            return
        split = _split_panel_columns(left=left, right=right, gap=3)
        if split is None:
            self._draw_single_forecast_panel(
                stdscr=stdscr,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                snapshot=state.bundle.all_gpus,
                state=state,
            )
            return
        self._draw_dual_forecast_panels(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            split=split,
            all_snapshot=state.bundle.all_gpus,
            quad_snapshot=quad_snapshot,
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
        state: ForecastRenderState,
    ) -> None:
        """Draw one full-width all-GPU forecast panel."""

        self._draw_forecast_panel(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            title="GPU Availability Forecast (8h, all GPUs)",
            snapshot=snapshot,
            state=state,
        )

    def _draw_dual_forecast_panels(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        split: tuple[int, int, int, int],
        all_snapshot: ForecastSnapshot,
        quad_snapshot: ForecastSnapshot,
        state: ForecastRenderState,
    ) -> None:
        """Draw side-by-side all-GPU and quad-partition forecast panels."""

        left_left, left_right, right_left, right_right = split
        self._draw_forecast_panel(
            stdscr=stdscr,
            top=top,
            bottom=bottom,
            left=left_left,
            right=left_right,
            title="GPU Availability Forecast (8h, all GPUs)",
            snapshot=all_snapshot,
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
        state: ForecastRenderState,
    ) -> None:
        """Draw a single forecast panel with title and chart/placeholder."""

        if right < left or bottom < top:
            return
        panel_title = self._title_with_availability(title=title, snapshot=snapshot)
        _panel_add(
            stdscr=stdscr,
            y=top,
            x=left,
            text=panel_title,
            left=left,
            right=right,
            attr=curses.A_BOLD,
        )
        if snapshot is None:
            self._draw_forecast_placeholder(stdscr=stdscr, top=top, left=left, right=right, state=state)
            return
        self._draw_forecast_chart(
            stdscr=stdscr,
            top=top + 1,
            bottom=bottom,
            left=left,
            right=right,
            snapshot=snapshot,
        )

    def _title_with_availability(self, title: str, snapshot: ForecastSnapshot | None) -> str:
        """Return title text with current availability fraction when available.

        Inputs:
        - `title`: base panel title.
        - `snapshot`: forecast snapshot for this panel, if computed.

        Outputs:
        - Title string optionally suffixed with snapshot title metrics.
        """

        if snapshot is None:
            return title
        return f"{title} [{snapshot.title_metrics()}]"

    def _draw_forecast_placeholder(
        self, stdscr: "curses.window", top: int, left: int, right: int, state: ForecastRenderState
    ) -> None:
        """Draw immediate placeholder text before forecast data is available.

        Inputs:
        - `stdscr`: active curses screen.
        - `top`: top row of forecast panel.
        - `left`/`right`: horizontal panel bounds.
        - `state`: current thread-safe forecast render state.

        Outputs:
        - One placeholder line indicating loading or forecast error.
        """

        message = state.message or "Loading forecast..."
        attr = curses.A_DIM if state.is_loading else curses.color_pair(PAIR_ERROR) | curses.A_BOLD
        _panel_add(stdscr=stdscr, y=top + 1, x=left, text=message, left=left, right=right, attr=attr)

    def _draw_forecast_chart(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
        snapshot: ForecastSnapshot,
    ) -> None:
        """Draw the forecast chart within the reserved lower panel bounds.

        Inputs:
        - `stdscr`: active curses screen.
        - `top`/`bottom`: inclusive y-bounds available for chart + x-axis labels.
        - `snapshot`: forecast snapshot used for dense step-series plotting.

        Outputs:
        - Drawn chart with y ticks, x ticks, labels, and value row.
        """

        if bottom - top + 1 < 6 or right - left + 1 < 20:
            _panel_add(
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
            _panel_add(stdscr=stdscr, y=top, x=left, text="Panel too narrow.", left=left, right=right, attr=curses.A_DIM)
            return
        points = _dense_panel_points(snapshot=snapshot, count=chart_width)
        vmax = max(1, max(point.available_gpus for point in points))
        for idx, point in enumerate(points):
            x = chart_left + idx
            y = _plot_y_for_value(top=top, height=chart_height, value=point.available_gpus, vmax=vmax)
            _panel_add(
                stdscr=stdscr,
                y=y,
                x=x,
                text="*",
                left=left,
                right=right,
                attr=self._availability_attr(point.available_gpus),
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
        _panel_add(
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

    def _draw_forecast_y_ticks(
        self,
        stdscr: "curses.window",
        top: int,
        chart_height: int,
        vmax: int,
        left: int,
        right: int,
    ) -> None:
        """Draw non-overlapping y-axis labels for forecast availability values."""

        used_rows: Set[int] = set()
        values = sorted(_forecast_y_tick_values(vmax=vmax, chart_height=chart_height), reverse=True)
        for value in values:
            y = _plot_y_for_value(top=top, height=chart_height, value=value, vmax=vmax)
            if y in used_rows:
                continue
            used_rows.add(y)
            _panel_add(stdscr=stdscr, y=y, x=left, text=f"{value:>3} |", left=left, right=right, attr=curses.A_DIM)

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
        """Draw x-axis ticks, labels, and sampled values across chart width."""

        labels = [point.label() for point in snapshot.points]
        values = [point.available_gpus for point in snapshot.points]
        xs = _tick_positions(left=chart_left, width=chart_width, count=len(labels))
        for x in xs:
            _panel_add(stdscr=stdscr, y=baseline_y, x=x, text="+", left=left, right=right, attr=curses.A_DIM)
        last_end = left - 1
        for x, label in zip(xs, labels):
            start = max(left, x - (len(label) // 2))
            if start <= last_end + 1:
                continue
            _panel_add(
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
            _panel_add(
                stdscr=stdscr,
                y=baseline_y + 2,
                x=max(left, x - (len(text) // 2)),
                text=text,
                left=left,
                right=right,
                attr=self._availability_attr(value),
            )

    def _availability_attr(self, available: int) -> int:
        """Map forecast availability values to color-emphasis attributes."""

        if available <= 0:
            return curses.color_pair(PAIR_ERROR) | curses.A_BOLD
        if available <= 4:
            return curses.color_pair(PAIR_PENDING) | curses.A_BOLD
        return curses.color_pair(PAIR_RUNNING) | curses.A_BOLD

    def _row_attr(self, job: DashJob, focused: bool) -> int:
        base = curses.A_BOLD if focused else curses.A_NORMAL
        if focused:
            base |= curses.A_REVERSE
        if job.is_running():
            return base | curses.color_pair(PAIR_RUNNING)
        if job.is_pending():
            return base | curses.color_pair(PAIR_PENDING)
        return base

    def _draw_status(self, stdscr: "curses.window") -> None:
        height, width = stdscr.getmaxyx()
        status = self.status_message or "Ready"
        attr = curses.A_BOLD if status.startswith("OK") else curses.A_DIM
        if status.startswith("ERROR") or status.startswith("Refresh failed"):
            attr = curses.color_pair(PAIR_ERROR) | curses.A_BOLD
        try:
            stdscr.addstr(height - 1, 1, status[: max(1, width - 2)], attr)
        except curses.error:
            return

    def _index_for_job(self, job_id: Optional[str]) -> int:
        if not self.jobs:
            return 0
        if not job_id:
            return min(self.focus_index, len(self.jobs) - 1)
        for idx, job in enumerate(self.jobs):
            if job.job_id == job_id:
                return idx
        return 0

    def current_job(self) -> Optional[DashJob]:
        if not self.jobs:
            return None
        return self.jobs[self.focus_index]

    def current_job_id(self) -> Optional[str]:
        job = self.current_job()
        return None if job is None else job.job_id

    def _fallback_loop(self) -> int:
        self._refresh_jobs()
        while True:
            self._print_plain_table()
            try:
                raw = input("dash> ").strip().lower()
            except EOFError:
                return 0
            result = self._handle_fallback_command(raw=raw)
            if result is not None:
                return result

    def _print_plain_table(self) -> None:
        print("\n=== dash ===")
        if not self.jobs:
            print("No pending/running jobs.")
        for idx, job in enumerate(self.jobs, start=1):
            marker = "*" if job.job_id in self.selected_job_ids else " "
            focus = ">" if (idx - 1) == self.focus_index else " "
            print(f"{focus}{idx:02d}[{marker}] {job.display_row()}")
        print(self.status_message)
        print("Commands: j/k move, s select, a all, c cancel, v join, r refresh, q quit")

    def _handle_fallback_command(self, raw: str) -> Optional[int]:
        if raw in ("q", "quit"):
            return 0
        if raw == "j":
            self._move_focus(delta=1)
        elif raw == "k":
            self._move_focus(delta=-1)
        elif raw in ("s", " "):
            self.toggle_selected_current()
        elif raw == "a":
            self.toggle_all_jobs()
        elif raw == "r":
            self._refresh_jobs()
        elif raw == "c":
            self._cancel_from_fallback()
        elif raw == "v":
            return self._join_from_ui()
        return None

    def _cancel_from_fallback(self) -> None:
        job_ids = self._target_job_ids()
        if not job_ids:
            self.status_message = "No job selected."
            return
        answer = input(f"Cancel {len(job_ids)} job(s)? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            self.status_message = "Cancel aborted."
            return
        result = cancel_dash_jobs(job_ids=job_ids)
        self.status_message = result.summary_line()
        self._refresh_jobs()


def run_dash_dashboard(user_name: str, editor_command: Optional[str] = None) -> int:
    """Run the dash dashboard for one user and return an exit code."""

    return DashBoard(user_name=user_name, refresh_seconds=2, editor_command=editor_command).run()


def _safe_add(stdscr: "curses.window", y: int, x: int, text: str, attr: int = 0) -> None:
    """Write clipped text safely within curses screen bounds."""

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


def _panel_add(
    stdscr: "curses.window", y: int, x: int, text: str, left: int, right: int, attr: int = 0
) -> None:
    """Write text clipped to one horizontal panel range."""

    if right < left or x > right:
        return
    draw_x = max(left, x)
    offset = draw_x - x
    clipped = text[offset : offset + max(0, right - draw_x + 1)]
    if not clipped:
        return
    _safe_add(stdscr=stdscr, y=y, x=draw_x, text=clipped, attr=attr)


def _split_panel_columns(left: int, right: int, gap: int) -> tuple[int, int, int, int] | None:
    """Split a horizontal region into two equal-width panel columns."""

    total_width = right - left + 1
    min_width = 36
    if total_width < (2 * min_width) + gap:
        return None
    left_width = (total_width - gap) // 2
    left_right = left + left_width - 1
    right_left = left_right + gap + 1
    return left, left_right, right_left, right


def _forecast_panel_top(screen_height: int) -> int:
    """Return row index where lower-half forecast panel should start.

    Inputs:
    - `screen_height`: terminal row count.

    Outputs:
    - Y-index anchoring the forecast panel in the lower half of screen.
    """

    minimum_top = 6
    target = max(minimum_top, screen_height // 2)
    return min(target, max(minimum_top, screen_height - 6))


def _plot_y_for_value(top: int, height: int, value: int, vmax: int) -> int:
    """Map a value in `0..vmax` to a chart row in `[top, top+height-1]`.

    Inputs:
    - `top`: top row of chart area.
    - `height`: chart span in rows.
    - `value`: availability value to project.
    - `vmax`: axis maximum value.

    Outputs:
    - Y-row to draw within chart bounds.
    """

    if vmax <= 0:
        return top + height - 1
    clipped = max(0, min(value, vmax))
    ratio = clipped / vmax
    return top + (height - 1) - int(round(ratio * (height - 1)))


def _forecast_y_tick_values(vmax: int, chart_height: int) -> List[int]:
    """Return unique y-axis labels spanning `0..vmax` for chart rows.

    Inputs:
    - `vmax`: plotted y-axis ceiling.
    - `chart_height`: chart span in rows.

    Outputs:
    - Integer y tick values suitable for non-overlapping label rows.
    """

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
    """Return evenly distributed x-positions across `width` columns.

    Inputs:
    - `left`: first chart x-column.
    - `width`: chart width in columns.
    - `count`: number of positions needed.

    Outputs:
    - Integer x-columns spanning the full chart width.
    """

    if count <= 1:
        return [left]
    return [left + int(round(idx * (width - 1) / (count - 1))) for idx in range(count)]


def _dense_panel_points(snapshot: ForecastSnapshot, count: int) -> List[ForecastPoint]:
    """Sample dense points from step-series for full-width chart rendering.

    Inputs:
    - `snapshot`: last computed forecast snapshot.
    - `count`: number of x columns to sample.

    Outputs:
    - Dense `ForecastPoint` sequence for per-column plotting.

    Example:
    - `_dense_panel_points(snapshot=snapshot, count=120)`
    """

    if count <= 1:
        base = snapshot.points[0].available_gpus if snapshot.points else 0
        return [ForecastPoint(offset_hours=0.0, available_gpus=base)]
    horizon_hours = snapshot.points[-1].offset_hours if snapshot.points else FORECAST_HORIZON_HOURS
    points: List[ForecastPoint] = []
    for idx in range(count):
        offset_hours = horizon_hours * idx / (count - 1)
        query = snapshot.generated_at + timedelta(hours=offset_hours)
        available = step_value_at(query=query, times=snapshot.series_times, values=snapshot.series_available)
        points.append(ForecastPoint(offset_hours=offset_hours, available_gpus=available))
    return points


def _prepare_screen(stdscr: "curses.window") -> None:
    curses.start_color()
    if curses.has_colors():
        curses.use_default_colors()
        curses.init_pair(1, -1, curses.COLOR_BLACK)
        stdscr.bkgd(" ", curses.color_pair(1))
    else:
        stdscr.bkgd(" ", curses.A_NORMAL)
    stdscr.clear()
    stdscr.refresh()


def _center_text(stdscr: "curses.window", y: int, text: str, attr: int = 0) -> None:
    _, width = stdscr.getmaxyx()
    x = max(0, (width - len(text)) // 2)
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        return
