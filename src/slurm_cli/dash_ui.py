from __future__ import annotations

import curses
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set

from slurm_cli.dashboard_forecast import (
    FORECAST_HORIZON_HOURS,
    FORECAST_REFRESH_SECONDS,
    DashboardForecastRenderer,
    ForecastRenderState,
    center_text_range,
    forecast_panel_top,
    format_duration_short,
    panel_add,
    prepare_screen,
)
from slurm_cli.dash_logic import (
    BlameRecord,
    DashJob,
    DashTableLayout,
    cancel_dash_jobs,
    fetch_blame_records,
    fetch_dash_jobs,
    join_job_via_remote,
)
from slurm_cli.forecast_cli import DashForecastBundle, take_dash_forecast_bundle
from slurm_cli.partition_policy import detect_cluster_name
from slurm_cli.remote_access import RemoteOpenRequest, open_remote_target


PAIR_PENDING = 2
PAIR_RUNNING = 3
PAIR_ERROR = 4
OSC_CLUSTER_CHOICES = ("pitzer", "ascend", "cardinal")


@dataclass(frozen=True)
class DashboardCommandResult:
    """Structured dashboard exit request used by the outer command runner.

    Inputs:
    - `action`: requested next action (`quit` or `launch`).

    Outputs:
    - Immutable command result consumed by `run_dash_command`.
    """

    action: str


@dataclass(frozen=True)
class DashboardRefreshResult:
    """One completed dashboard refresh snapshot from the background worker.

    Inputs:
    - `jobs`: refreshed dashboard job list, or `None` on refresh failure.
    - `blame_records`: refreshed blame rows, or `None` on refresh failure.
    - `status_message`: status text for the dashboard footer.
    - `refreshed_at`: refresh timestamp for ETA calculations.

    Outputs:
    - Immutable worker payload applied back on the UI thread.
    """

    jobs: List[DashJob] | None
    blame_records: List[BlameRecord] | None
    status_message: str
    refreshed_at: datetime | None


class DashBoard:
    """Interactive dashboard for pending/running job management.

    Args:
        user_name: Username used for `squeue -u` filtering.
        refresh_seconds: Automatic refresh interval.
        editor_command: Optional editor command/alias for `v` join actions.
        initial_status_message: Optional status message shown on first render.

    Returns:
        Structured dashboard command result for the outer CLI runner.
    """

    def __init__(
        self,
        user_name: str,
        refresh_seconds: int = 2,
        editor_command: Optional[str] = None,
        initial_status_message: Optional[str] = None,
    ) -> None:
        self.user_name = user_name
        self.refresh_seconds = max(1, refresh_seconds)
        self.editor_command = editor_command
        self.jobs: List[DashJob] = []
        self.blame_records: List[BlameRecord] = []
        self.focus_index = 0
        self.last_refresh_at: datetime | None = None
        self.selected_job_ids: Set[str] = set()
        self.status_message = initial_status_message or "Loading jobs..."
        self.show_blame_panel = True
        self.forecast_horizon_hours = FORECAST_HORIZON_HOURS
        self.forecast_refresh_seconds = FORECAST_REFRESH_SECONDS
        self._forecast_bundle: DashForecastBundle | None = None
        self._forecast_message = "Loading forecast..."
        self._forecast_loading = True
        self._forecast_lock = threading.Lock()
        self._forecast_refresh_event = threading.Event()
        self._forecast_stop_event = threading.Event()
        self._forecast_thread: threading.Thread | None = None
        self._refresh_lock = threading.Lock()
        self._pending_refresh: DashboardRefreshResult | None = None
        self._refresh_event = threading.Event()
        self._refresh_stop_event = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._forecast_renderer = DashboardForecastRenderer(
            pending_pair=PAIR_PENDING,
            running_pair=PAIR_RUNNING,
            error_pair=PAIR_ERROR,
        )

    def run(self) -> DashboardCommandResult:
        """Run the dashboard in curses mode, with plain-text fallback."""

        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_loop()

    def update_jobs(
        self,
        jobs: Sequence[DashJob],
        focus_job_id: Optional[str] = None,
    ) -> None:
        """Replace the visible jobs while preserving focus/selection when possible."""

        target_id = focus_job_id or self.current_job_id()
        self.jobs = list(jobs)
        visible_ids = {job.job_id for job in self.jobs}
        self.selected_job_ids = {
            job_id for job_id in self.selected_job_ids if job_id in visible_ids
        }
        self.focus_index = self._index_for_job(job_id=target_id)

    def toggle_selected_current(self) -> None:
        """Toggle the selection state of the focused job."""

        job = self.current_job()
        if job is None:
            return
        if job.job_id in self.selected_job_ids:
            self.selected_job_ids.remove(job.job_id)
            return
        self.selected_job_ids.add(job.job_id)

    def toggle_all_jobs(self) -> None:
        """Select or clear every visible job."""

        if not self.jobs:
            return
        all_ids = {job.job_id for job in self.jobs}
        self.selected_job_ids = set() if self.selected_job_ids == all_ids else all_ids

    def can_join_current(self) -> bool:
        """Return `True` when the focused job is eligible for remote join."""

        job = self.current_job()
        return job is not None and job.is_running()

    def _curses_main(self, stdscr: "curses.window") -> DashboardCommandResult:
        prepare_screen(stdscr=stdscr)
        self._init_colors()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._start_refresh_worker()
        self._start_forecast_worker()
        self._request_refresh(message=self.status_message)
        try:
            while True:
                self._apply_pending_refresh()
                self._draw(stdscr=stdscr)
                result = self._handle_key(stdscr=stdscr, key=stdscr.getch())
                if result is not None:
                    return result
                time.sleep(0.05)
        finally:
            self._stop_refresh_worker()
            self._stop_forecast_worker()

    def _init_colors(self) -> None:
        """Initialize color pairs used by the dashboard."""

        if not curses.has_colors():
            return
        curses.init_pair(PAIR_PENDING, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(PAIR_RUNNING, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(PAIR_ERROR, curses.COLOR_RED, curses.COLOR_BLACK)

    def _start_refresh_worker(self) -> None:
        """Launch the background job/blame refresh worker."""

        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._refresh_stop_event.clear()
        self._refresh_event.set()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="dash-refresh",
            daemon=True,
        )
        self._refresh_thread.start()

    def _stop_refresh_worker(self) -> None:
        """Stop the background job/blame refresh worker."""

        self._refresh_stop_event.set()
        self._refresh_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=1.0)
        self._refresh_thread = None

    def _request_refresh(self, message: str) -> None:
        """Request an asynchronous job/blame refresh without clearing old data."""

        if self.jobs:
            self.status_message = message
        self._refresh_event.set()

    def _refresh_loop(self) -> None:
        """Refresh jobs and blame records on a background cadence."""

        next_refresh = 0.0
        while not self._refresh_stop_event.is_set():
            wait_seconds = max(0.0, next_refresh - time.monotonic())
            self._refresh_event.wait(timeout=wait_seconds)
            force_refresh = self._refresh_event.is_set()
            self._refresh_event.clear()
            if self._refresh_stop_event.is_set():
                break
            if not force_refresh and time.monotonic() < next_refresh:
                continue
            update = _fetch_dashboard_refresh(user_name=self.user_name)
            with self._refresh_lock:
                self._pending_refresh = update
            next_refresh = time.monotonic() + self.refresh_seconds

    def _apply_pending_refresh(self) -> None:
        """Apply the newest completed refresh payload on the UI thread."""

        with self._refresh_lock:
            update = self._pending_refresh
            self._pending_refresh = None
        if update is None:
            return
        if update.jobs is not None:
            self.update_jobs(jobs=update.jobs, focus_job_id=self.current_job_id())
            self.last_refresh_at = update.refreshed_at
        if update.blame_records is not None:
            self.blame_records = update.blame_records
        self.status_message = update.status_message

    def _start_forecast_worker(self) -> None:
        """Launch the background forecast refresh worker."""

        if self._forecast_thread and self._forecast_thread.is_alive():
            return
        self._forecast_stop_event.clear()
        self._forecast_refresh_event.set()
        self._forecast_thread = threading.Thread(
            target=self._forecast_loop,
            name="dash-forecast",
            daemon=True,
        )
        self._forecast_thread.start()

    def _stop_forecast_worker(self) -> None:
        """Stop the background forecast refresh worker."""

        self._forecast_stop_event.set()
        self._forecast_refresh_event.set()
        if self._forecast_thread is not None:
            self._forecast_thread.join(timeout=1.0)
        self._forecast_thread = None

    def _request_forecast_refresh(self) -> None:
        """Trigger an immediate forecast refresh."""

        self._forecast_refresh_event.set()

    def _set_forecast_loading(self) -> None:
        """Mark the forecast state as loading."""

        with self._forecast_lock:
            self._forecast_loading = True
            if self._forecast_bundle is None:
                self._forecast_message = "Loading forecast..."

    def _set_forecast_ready(self, bundle: DashForecastBundle) -> None:
        """Store the latest successful forecast bundle."""

        with self._forecast_lock:
            self._forecast_bundle = bundle
            self._forecast_message = ""
            self._forecast_loading = False

    def _set_forecast_error(self, error_text: str) -> None:
        """Capture forecast refresh failures without discarding the last bundle."""

        with self._forecast_lock:
            self._forecast_loading = False
            if self._forecast_bundle is None:
                self._forecast_message = f"Forecast failed: {error_text}"

    def _forecast_state(self) -> ForecastRenderState:
        """Return a render-safe forecast state snapshot."""

        with self._forecast_lock:
            return ForecastRenderState(
                bundle=self._forecast_bundle,
                message=self._forecast_message,
                is_loading=self._forecast_loading,
            )

    def _forecast_loop(self) -> None:
        """Refresh the dashboard forecast in the background."""

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
                bundle = take_dash_forecast_bundle(
                    horizon_hours=self.forecast_horizon_hours
                )
            except Exception as exc:
                self._set_forecast_error(error_text=str(exc))
                next_refresh = time.monotonic() + self.forecast_refresh_seconds
                continue
            self._set_forecast_ready(bundle=bundle)
            next_refresh = time.monotonic() + self.forecast_refresh_seconds

    def _handle_key(
        self,
        stdscr: "curses.window",
        key: int,
    ) -> DashboardCommandResult | None:
        if key in (27, ord("q"), ord("Q")):
            return DashboardCommandResult(action="quit")
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            self._move_focus(delta=-1)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            self._move_focus(delta=1)
        elif key == ord(" "):
            self.toggle_selected_current()
        elif key in (ord("a"), ord("A")):
            self.toggle_all_jobs()
        elif key in (ord("r"), ord("R")):
            self._relocate_from_ui(stdscr=stdscr)
        elif key in (ord("b"), ord("B")):
            self.show_blame_panel = not self.show_blame_panel
        elif key in (ord("c"), ord("C")):
            self._cancel_in_ui(stdscr=stdscr)
        elif key in (ord("v"), ord("V")):
            self._join_from_ui()
        elif key in (ord("n"), ord("N")):
            return DashboardCommandResult(action="launch")
        return None

    def _move_focus(self, delta: int) -> None:
        """Move the focus cursor through the visible job list."""

        if not self.jobs:
            self.focus_index = 0
            return
        upper = len(self.jobs) - 1
        self.focus_index = (self.focus_index + delta) % (upper + 1)

    def _cancel_in_ui(self, stdscr: "curses.window") -> None:
        """Cancel the selected jobs from within the curses dashboard."""

        job_ids = self._target_job_ids()
        if not job_ids:
            self.status_message = "No job selected."
            return
        if not self._confirm(
            stdscr=stdscr,
            prompt=f"Cancel {len(job_ids)} job(s)? [y/N]",
        ):
            self.status_message = "Cancel aborted."
            return
        result = cancel_dash_jobs(job_ids=job_ids)
        self.status_message = result.summary_line()
        self._request_refresh(message="Refreshing jobs...")

    def _join_from_ui(self) -> None:
        """Open the focused running job in the remote editor without exiting."""

        job = self.current_job()
        if job is None:
            self.status_message = "No focused job."
            return
        result = join_job_via_remote(job=job, editor=self.editor_command)
        self.status_message = result.summary_line()

    def _relocate_from_ui(self, stdscr: "curses.window") -> None:
        """Open a remote editor on another OSC cluster without exiting dash."""

        target_cluster = self._choose_relocation_cluster(stdscr=stdscr)
        if target_cluster is None:
            self.status_message = "Relocate canceled."
            return
        self._open_relocation_target(cluster=target_cluster)

    def _choose_relocation_cluster(self, stdscr: "curses.window") -> str | None:
        """Prompt for a destination cluster inside the curses dashboard."""

        options = relocation_cluster_options(current_cluster=detect_cluster_name())
        if not options:
            self.status_message = "No alternate OSC cluster detected."
            return None
        selected = 0
        stdscr.nodelay(False)
        try:
            while True:
                self._draw_relocation_picker(
                    stdscr=stdscr,
                    options=options,
                    selected=selected,
                )
                key = stdscr.getch()
                if key in (27, ord("q"), ord("Q")):
                    return None
                if key in (curses.KEY_UP, ord("k"), ord("K")):
                    selected = (selected - 1) % len(options)
                elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
                    selected = (selected + 1) % len(options)
                elif key in (10, 13, curses.KEY_ENTER):
                    return options[selected]
                elif ord("1") <= key <= ord(str(len(options))):
                    return options[key - ord("1")]
        finally:
            stdscr.nodelay(True)

    def _draw_relocation_picker(
        self,
        stdscr: "curses.window",
        options: Sequence[str],
        selected: int,
    ) -> None:
        """Draw the full-screen relocate destination picker."""

        height, width = stdscr.getmaxyx()
        stdscr.erase()
        center_text_range(
            stdscr=stdscr,
            y=max(1, height // 2 - len(options) - 2),
            text="Relocate Remote Session",
            left=0,
            right=width - 1,
            attr=curses.A_BOLD,
        )
        start_y = max(3, height // 2 - len(options) // 2)
        for idx, cluster in enumerate(options):
            marker = ">" if idx == selected else " "
            label = f"{idx + 1}. {cluster.title()}"
            attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
            center_text_range(
                stdscr=stdscr,
                y=start_y + idx,
                text=f"{marker} {label}",
                left=0,
                right=width - 1,
                attr=attr,
            )
        center_text_range(
            stdscr=stdscr,
            y=min(height - 2, start_y + len(options) + 2),
            text="Enter select | Esc return to dashboard",
            left=0,
            right=width - 1,
            attr=curses.A_DIM,
        )
        stdscr.refresh()

    def _open_relocation_target(self, cluster: str) -> None:
        """Run the shared remote-open path for a selected login cluster."""

        result = open_remote_target(
            request=RemoteOpenRequest(
                host=cluster,
                work_dir=Path.cwd(),
                editor=self.editor_command,
            )
        )
        prefix = "OK" if result.ok else "ERROR"
        self.status_message = (
            f"{prefix}: relocate to {cluster.title()}: {result.message}"
        )

    def _target_job_ids(self) -> List[str]:
        """Return selected job ids, falling back to the focused job."""

        if self.selected_job_ids:
            return sorted(self.selected_job_ids)
        focused = self.current_job_id()
        return [focused] if focused else []

    def _confirm(self, stdscr: "curses.window", prompt: str) -> bool:
        """Prompt for a yes/no confirmation within curses mode."""

        stdscr.nodelay(False)
        try:
            self.status_message = prompt
            self._draw(stdscr=stdscr)
            key = stdscr.getch()
            return key in (ord("y"), ord("Y"))
        finally:
            stdscr.nodelay(True)

    def _draw(self, stdscr: "curses.window") -> None:
        """Render one full curses dashboard frame."""

        stdscr.clear()
        height, width = stdscr.getmaxyx()
        lower_panel_top = forecast_panel_top(screen_height=height)
        blame_width = 65
        can_fit_blame = width >= 145
        show_blame = (
            self.show_blame_panel and can_fit_blame and bool(self.blame_records)
        )
        if show_blame:
            my_jobs_right = width - blame_width - 2
            blame_left = width - blame_width
            for y in range(lower_panel_top):
                panel_add(
                    stdscr=stdscr,
                    y=y,
                    x=my_jobs_right + 1,
                    text="|",
                    left=0,
                    right=width - 1,
                    attr=curses.A_DIM,
                )
        else:
            my_jobs_right = width - 1
            blame_left = width
        center_text_range(
            stdscr=stdscr,
            y=0,
            text=f"dash: {self.user_name}",
            left=0,
            right=my_jobs_right,
            attr=curses.A_BOLD,
        )
        help_text = "Space=select | c=cancel | v=join | n=submit alloc | r=relocate | b=blame | q=quit"
        center_text_range(
            stdscr=stdscr,
            y=1,
            text=help_text,
            left=0,
            right=my_jobs_right,
            attr=curses.A_DIM,
        )
        panel_add(
            stdscr=stdscr,
            y=lower_panel_top - 1,
            x=1,
            text="-" * max(1, my_jobs_right - 1),
            left=0,
            right=my_jobs_right,
            attr=curses.A_DIM,
        )
        self._draw_rows(
            stdscr=stdscr,
            top=3,
            bottom=lower_panel_top - 2,
            left=1,
            right=my_jobs_right,
        )
        if show_blame:
            self._draw_blame_panel(
                stdscr=stdscr,
                top=0,
                bottom=lower_panel_top - 1,
                left=blame_left,
                right=width - 1,
            )
        self._forecast_renderer.draw_forecast_area(
            stdscr=stdscr,
            top=lower_panel_top,
            bottom=height - 2,
            state=self._forecast_state(),
        )
        self._draw_status(stdscr=stdscr)
        stdscr.refresh()

    def _draw_rows(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> None:
        """Draw the main dashboard job table, including the header row."""

        if bottom < top:
            return
        real_right = right if right > 0 else (stdscr.getmaxyx()[1] - 1)
        layout = DashTableLayout.from_width(
            total_width=max(59, real_right - left + 1 - 5),
        )
        header_text = f"     {layout.header_row()}"
        panel_add(
            stdscr=stdscr,
            y=top,
            x=left,
            text=header_text,
            left=left,
            right=real_right,
            attr=curses.A_DIM | curses.A_UNDERLINE,
        )
        if not self.jobs:
            row = top + max(1, (bottom - top) // 2)
            center_text_range(
                stdscr=stdscr,
                y=row,
                text="No jobs to display.",
                left=left,
                right=real_right,
                attr=curses.A_DIM,
            )
            return
        for row_idx, job in enumerate(self.jobs):
            y = top + 1 + row_idx
            if y > bottom:
                break
            marker = "*" if job.job_id in self.selected_job_ids else " "
            prefix = ">" if row_idx == self.focus_index else " "
            row_text = layout.render_job(job=job, as_of=self.last_refresh_at)
            text = f"{prefix}[{marker}] {row_text}"
            panel_add(
                stdscr=stdscr,
                y=y,
                x=left,
                text=text,
                left=left,
                right=real_right,
                attr=self._row_attr(job=job, focused=row_idx == self.focus_index),
            )

    def _draw_blame_panel(
        self,
        stdscr: "curses.window",
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> None:
        """Draw the right-side blame panel when there is enough space."""

        if bottom < top:
            return
        center_text_range(
            stdscr=stdscr,
            y=top,
            text="Top GPU Users (Sorted)",
            left=left,
            right=right,
            attr=curses.A_BOLD,
        )
        header = (
            f"{'Name':<17} {'User':<8} {'PI':<16} {'Run':>3} {'Pnd':>3} {'AvgReq':>7}"
        )
        panel_add(
            stdscr=stdscr,
            y=top + 1,
            x=left,
            text=header,
            left=left,
            right=right,
            attr=curses.A_DIM | curses.A_UNDERLINE,
        )
        for idx, record in enumerate(self.blame_records):
            y = top + 2 + idx
            if y > bottom:
                break
            avg_str = format_duration_short(record.avg_request_minutes)
            display_name = record.full_name if record.full_name else record.username
            name_text = (
                (display_name[:16] + "~") if len(display_name) > 17 else display_name
            )
            user_text = (
                (record.username[:7] + "~")
                if len(record.username) > 8
                else record.username
            )
            coordinator = record.coordinator_name
            coordinator_text = (
                (coordinator[:15] + "~") if len(coordinator) > 16 else coordinator
            )
            line = (
                f"{name_text:<17} {user_text:<8} {coordinator_text:<16} "
                f"{record.running_gpus:>3} {record.pending_gpus:>3} {avg_str:>7}"
            )
            attr = (
                curses.A_BOLD | curses.A_REVERSE
                if record.username == self.user_name
                else curses.A_NORMAL
            )
            if record.running_gpus > 0:
                attr |= curses.color_pair(PAIR_RUNNING)
            panel_add(
                stdscr=stdscr,
                y=y,
                x=left,
                text=line,
                left=left,
                right=right,
                attr=attr,
            )

    def _row_attr(self, job: DashJob, focused: bool) -> int:
        """Return the visual attributes for one job row."""

        base = curses.A_BOLD if focused else curses.A_NORMAL
        if focused:
            base |= curses.A_REVERSE
        if job.is_running():
            return base | curses.color_pair(PAIR_RUNNING)
        if job.is_pending():
            return base | curses.color_pair(PAIR_PENDING)
        return base

    def _draw_status(self, stdscr: "curses.window") -> None:
        """Draw the footer status line."""

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
        """Return the row index for a given job id, falling back sanely."""

        if not self.jobs:
            return 0
        if not job_id:
            return min(self.focus_index, len(self.jobs) - 1)
        for idx, job in enumerate(self.jobs):
            if job.job_id == job_id:
                return idx
        return 0

    def current_job(self) -> Optional[DashJob]:
        """Return the currently focused job, if any."""

        if not self.jobs:
            return None
        return self.jobs[self.focus_index]

    def current_job_id(self) -> Optional[str]:
        """Return the focused job id, if any."""

        job = self.current_job()
        return None if job is None else job.job_id

    def _fallback_loop(self) -> DashboardCommandResult:
        """Run the plain-text fallback dashboard when curses is unavailable."""

        self._apply_refresh_update(
            update=_fetch_dashboard_refresh(user_name=self.user_name)
        )
        while True:
            self._print_plain_table()
            try:
                raw = input("dash> ").strip().lower()
            except EOFError:
                return DashboardCommandResult(action="quit")
            result = self._handle_fallback_command(raw=raw)
            if result is not None:
                return result

    def _print_plain_table(self) -> None:
        """Print the plain-text dashboard table, including the header row."""

        width = shutil.get_terminal_size(fallback=(120, 40)).columns
        layout = DashTableLayout.from_width(total_width=max(59, width - 5))
        print("\n=== dash ===")
        print(f"     {layout.header_row()}")
        if not self.jobs:
            print("No pending/running jobs.")
        for idx, job in enumerate(self.jobs, start=1):
            marker = "*" if job.job_id in self.selected_job_ids else " "
            focus = ">" if (idx - 1) == self.focus_index else " "
            row_text = layout.render_job(job=job, as_of=self.last_refresh_at)
            print(f"{focus}[{marker}] {row_text}")
        print(self.status_message)
        print(
            "Commands: j/k move, s select, a all, c cancel, v join, n submit, r relocate, q quit"
        )

    def _handle_fallback_command(self, raw: str) -> DashboardCommandResult | None:
        """Handle one plain-text dashboard command."""

        if raw in ("q", "quit"):
            return DashboardCommandResult(action="quit")
        if raw == "j":
            self._move_focus(delta=1)
        elif raw == "k":
            self._move_focus(delta=-1)
        elif raw in ("s", " "):
            self.toggle_selected_current()
        elif raw == "a":
            self.toggle_all_jobs()
        elif raw == "r":
            self._relocate_from_fallback()
        elif raw == "c":
            self._cancel_from_fallback()
        elif raw == "v":
            self._join_from_ui()
        elif raw == "n":
            return DashboardCommandResult(action="launch")
        return None

    def _relocate_from_fallback(self) -> None:
        """Prompt for a relocation target in plain-text fallback mode."""

        options = relocation_cluster_options(current_cluster=detect_cluster_name())
        if not options:
            self.status_message = "No alternate OSC cluster detected."
            return
        print("Relocate to:")
        for idx, cluster in enumerate(options, start=1):
            print(f"  {idx}. {cluster.title()}")
        raw = input("cluster> ").strip().lower()
        if raw in ("", "q", "quit", "cancel"):
            self.status_message = "Relocate canceled."
            return
        if raw.isdigit():
            choice_index = int(raw) - 1
            if 0 <= choice_index < len(options):
                self._open_relocation_target(cluster=options[choice_index])
                return
        if raw in options:
            self._open_relocation_target(cluster=raw)
            return
        self.status_message = f"Unknown relocate target: {raw}"

    def _cancel_from_fallback(self) -> None:
        """Cancel jobs from the plain-text fallback dashboard."""

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
        self._apply_refresh_update(
            update=_fetch_dashboard_refresh(user_name=self.user_name)
        )

    def _apply_refresh_update(self, update: DashboardRefreshResult) -> None:
        """Apply a synchronous refresh result for fallback flows."""

        if update.jobs is not None:
            self.update_jobs(jobs=update.jobs, focus_job_id=self.current_job_id())
            self.last_refresh_at = update.refreshed_at
        if update.blame_records is not None:
            self.blame_records = update.blame_records
        self.status_message = update.status_message

    def _title_with_availability(
        self,
        title: str,
        snapshot,
    ) -> str:
        """Expose title formatting for tests and title-only callers."""

        return self._forecast_renderer._title_with_availability(
            title=title,
            snapshot=snapshot,
        )


def run_dash_dashboard(
    user_name: str,
    editor_command: Optional[str] = None,
    initial_status_message: Optional[str] = None,
) -> DashboardCommandResult:
    """Run the dashboard and return the requested follow-up action."""

    return DashBoard(
        user_name=user_name,
        refresh_seconds=2,
        editor_command=editor_command,
        initial_status_message=initial_status_message,
    ).run()


def relocation_cluster_options(current_cluster: Optional[str]) -> tuple[str, ...]:
    """Return selectable OSC clusters excluding the current one when known."""

    current = (current_cluster or "").strip().lower()
    return tuple(cluster for cluster in OSC_CLUSTER_CHOICES if cluster != current)


def _fetch_dashboard_refresh(user_name: str) -> DashboardRefreshResult:
    """Fetch jobs and blame rows for one dashboard refresh cycle."""

    focus_time = datetime.now()
    try:
        jobs = fetch_dash_jobs(user_name=user_name)
        try:
            blame_records = fetch_blame_records()
        except Exception:
            blame_records = []
        return DashboardRefreshResult(
            jobs=jobs,
            blame_records=blame_records,
            status_message=_refresh_message(
                user_name=user_name, job_count=len(jobs), refreshed_at=focus_time
            ),
            refreshed_at=focus_time,
        )
    except Exception as exc:
        return DashboardRefreshResult(
            jobs=None,
            blame_records=None,
            status_message=f"Refresh failed: {exc}",
            refreshed_at=None,
        )


def _refresh_message(user_name: str, job_count: int, refreshed_at: datetime) -> str:
    """Return the canonical dashboard refresh status line."""

    if job_count == 0:
        return f"No pending/running jobs for {user_name}."
    return f"Loaded {job_count} jobs at {refreshed_at:%H:%M:%S}."
