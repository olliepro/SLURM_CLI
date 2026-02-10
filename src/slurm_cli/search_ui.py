from __future__ import annotations

import curses
import time
from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from slurm_cli.constants import (
    SEARCH_DASHBOARD_CLOSE_SECONDS,
    SEARCH_DEFAULT_MIN_GPUS,
    SEARCH_DEFAULT_MIN_TIME_MINUTES,
    SEARCH_JOB_PREFIX,
)
from slurm_cli.format_utils import (
    build_time_options,
    format_minutes_phrase,
    minutes_to_slurm_time,
    nearest_index,
    parse_time_string,
    sanitize_text,
    validate_time,
)

if TYPE_CHECKING:
    from slurm_cli.search_logic import SearchProbe, SearchSubmissionResult


TIME_MINUTE_OPTIONS = build_time_options()


def _center_text(stdscr, y: int, text: str, attr: int = 0) -> None:
    height, width = stdscr.getmaxyx()
    x = max(0, (width - len(text)) // 2)
    stdscr.addstr(y, x, text, attr)


def _prepare_curses_screen(stdscr: "curses.window") -> None:
    curses.start_color()
    if curses.has_colors():
        curses.use_default_colors()
        curses.init_pair(1, -1, curses.COLOR_BLACK)
        stdscr.bkgd(" ", curses.color_pair(1))
    else:
        stdscr.bkgd(" ", curses.A_NORMAL)
    stdscr.clear()
    stdscr.refresh()


class SearchBoundsPicker:
    """Collect search minimum time/GPU bounds with curses and fallback prompts.

    Args:
        max_time_minutes: Maximum probe time selected for search.
        max_gpus: Maximum GPU count selected for search.
        min_time_minutes: Initial minimum time default.
        min_gpus: Initial minimum GPU default.

    Returns:
        Tuple of resolved ``(min_time_minutes, min_gpus)`` or ``None`` on cancel.

    Example:
        >>> SearchBoundsPicker(240, 4, 30, 1).run()  # doctest: +SKIP
        (30, 1)
    """

    def __init__(
        self,
        max_time_minutes: int,
        max_gpus: int,
        min_time_minutes: int,
        min_gpus: int,
    ):
        assert max_time_minutes >= SEARCH_DEFAULT_MIN_TIME_MINUTES
        assert max_gpus >= SEARCH_DEFAULT_MIN_GPUS
        self.max_time_minutes = max_time_minutes
        self.max_gpus = max_gpus
        self.time_options = self._build_time_options()
        bounded_time = max(
            SEARCH_DEFAULT_MIN_TIME_MINUTES,
            min(max_time_minutes, min_time_minutes),
        )
        self.min_time_idx = nearest_index(self.time_options, bounded_time)
        self.min_gpus = max(SEARCH_DEFAULT_MIN_GPUS, min(max_gpus, min_gpus))
        self.focus = 0
        self.canceled = False

    def run(self) -> Optional[Tuple[int, int]]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _build_time_options(self) -> List[int]:
        values = [
            value
            for value in TIME_MINUTE_OPTIONS
            if SEARCH_DEFAULT_MIN_TIME_MINUTES <= value <= self.max_time_minutes
        ]
        if self.max_time_minutes not in values:
            values.append(self.max_time_minutes)
        return sorted(set(values))

    def _fallback_prompt(self) -> Optional[Tuple[int, int]]:
        min_time = self._prompt_time()
        if min_time is None:
            return None
        min_gpus = self._prompt_gpus()
        if min_gpus is None:
            return None
        return min_time, min_gpus

    def _prompt_time(self) -> Optional[int]:
        default = minutes_to_slurm_time(self.time_options[self.min_time_idx])
        while True:
            try:
                raw = input(f"Search min time [{default}]: ").strip()
            except EOFError:
                return None
            value = sanitize_text(raw)
            if not value:
                return self.time_options[self.min_time_idx]
            if not validate_time(value):
                print("Enter time as HH:MM:SS or DD-HH:MM:SS.")
                continue
            minutes = parse_time_string(value)
            if minutes is None or minutes > self.max_time_minutes:
                print(f"Min time must be <= {minutes_to_slurm_time(self.max_time_minutes)}.")
                continue
            if minutes < SEARCH_DEFAULT_MIN_TIME_MINUTES:
                print("Min time must be at least 00:30:00.")
                continue
            self.min_time_idx = nearest_index(self.time_options, minutes)
            return self.time_options[self.min_time_idx]

    def _prompt_gpus(self) -> Optional[int]:
        while True:
            try:
                raw = input(
                    f"Search min GPUs [1-{self.max_gpus}] ({self.min_gpus}): "
                ).strip()
            except EOFError:
                return None
            value = sanitize_text(raw)
            if not value:
                return self.min_gpus
            if value.isdigit():
                parsed = int(value)
                if SEARCH_DEFAULT_MIN_GPUS <= parsed <= self.max_gpus:
                    self.min_gpus = parsed
                    return self.min_gpus
            print(f"Please enter a number between 1 and {self.max_gpus}.")

    def _curses_main(self, stdscr):
        _prepare_curses_screen(stdscr)
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        while True:
            self._draw_bounds_screen(stdscr)
            action = self._handle_bounds_key(stdscr.getch())
            if isinstance(action, tuple):
                return action
            if action is False:
                self.canceled = True
                return None

    def _draw_bounds_screen(self, stdscr) -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Search Bounds", curses.A_BOLD)
        max_text = (
            f"Max: {format_minutes_phrase(self.max_time_minutes)} | GPUs: {self.max_gpus}"
        )
        _center_text(stdscr, 4, max_text, curses.A_DIM)
        rows = [
            ("Min Time", format_minutes_phrase(self.time_options[self.min_time_idx])),
            ("Min GPUs", str(self.min_gpus)),
        ]
        for idx, (label, value) in enumerate(rows):
            attr = curses.A_REVERSE | curses.A_BOLD if idx == self.focus else curses.A_BOLD
            _center_text(stdscr, 7 + idx * 2, f"{label}: {value}", attr)
        _center_text(
            stdscr,
            12,
            "Use arrow keys • Enter to accept • ESC to cancel",
            curses.A_DIM,
        )
        stdscr.refresh()

    def _handle_bounds_key(self, ch: int):
        if ch == curses.KEY_UP:
            self.focus = (self.focus - 1) % 2
        elif ch == curses.KEY_DOWN:
            self.focus = (self.focus + 1) % 2
        elif ch == curses.KEY_LEFT:
            self._nudge(delta=-1)
        elif ch == curses.KEY_RIGHT:
            self._nudge(delta=1)
        elif ch in (10, 13, curses.KEY_ENTER):
            return self.time_options[self.min_time_idx], self.min_gpus
        elif ch == 27:
            return False
        return True

    def _nudge(self, delta: int) -> None:
        if self.focus == 0:
            upper = len(self.time_options) - 1
            self.min_time_idx = max(0, min(upper, self.min_time_idx + delta))
            return
        self.min_gpus = max(
            SEARCH_DEFAULT_MIN_GPUS,
            min(self.max_gpus, self.min_gpus + delta),
        )


class SearchEmailPicker:
    """Collect search notify email via curses with terminal fallback.

    Args:
        initial_email: Pre-filled email value.

    Returns:
        Selected email string or ``None`` if the picker is canceled.
    """

    def __init__(self, initial_email: Optional[str]):
        self.email = sanitize_text(initial_email)
        self.canceled = False

    def run(self) -> Optional[str]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _fallback_prompt(self) -> Optional[str]:
        while True:
            default = f" [{self.email}]" if self.email else ""
            try:
                raw = input(f"Notify email{default}: ").strip()
            except EOFError:
                return None
            value = sanitize_text(raw)
            if value:
                self.email = value
                return self.email
            if self.email:
                return self.email
            print("Email is required for search submissions.")

    def _curses_main(self, stdscr):
        _prepare_curses_screen(stdscr)
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        while True:
            self._draw_email_screen(stdscr=stdscr)
            result = self._handle_email_key(key=stdscr.getch())
            if result is not None:
                return None if self.canceled else result

    def _draw_email_screen(self, stdscr: "curses.window") -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Search Notify Email", curses.A_BOLD)
        value = self.email or "(required)"
        _center_text(stdscr, 5, value, curses.A_REVERSE | curses.A_BOLD)
        _center_text(
            stdscr,
            8,
            "Type to edit • Enter to accept • ESC to cancel",
            curses.A_DIM,
        )
        stdscr.refresh()

    def _handle_email_key(self, key: int) -> Optional[str]:
        if key in (10, 13, curses.KEY_ENTER):
            return self.email or None
        if key == 27:
            self.canceled = True
            return self.email
        if key in (curses.KEY_BACKSPACE, 127, curses.KEY_DC):
            self.email = self.email[:-1]
            return None
        if 32 <= key <= 126:
            char = chr(key)
            if sanitize_text(char):
                self.email += char
        return None


class SearchSubmissionDashboard:
    """Render live search submission statuses in a curses dashboard.

    Args:
        probes: Probe list displayed in submission order.
        job_prefix: Prefix used for job-name rendering in each row.

    Returns:
        Submitted result list when dashboard succeeds, else ``None`` fallback signal.
    """

    def __init__(
        self,
        probes: Sequence["SearchProbe"],
        job_prefix: str = SEARCH_JOB_PREFIX,
        close_after_seconds: int = SEARCH_DASHBOARD_CLOSE_SECONDS,
    ):
        self.probes = list(probes)
        self.job_prefix = job_prefix
        self.rows = [probe.summary_line(prefix=job_prefix) for probe in self.probes]
        self.results: List[Optional["SearchSubmissionResult"]] = [None] * len(self.probes)
        self.close_after_seconds = max(1, int(close_after_seconds))
        self.was_canceled = False

    def run(
        self,
        submitter: Callable[
            [Callable[["SearchSubmissionResult"], None]],
            List["SearchSubmissionResult"],
        ],
        require_confirmation: bool,
    ) -> Optional[List["SearchSubmissionResult"]]:
        try:
            return curses.wrapper(self._curses_main, submitter, require_confirmation)
        except curses.error:
            return None

    def _curses_main(
        self,
        stdscr: "curses.window",
        submitter: Callable[
            [Callable[["SearchSubmissionResult"], None]],
            List["SearchSubmissionResult"],
        ],
        require_confirmation: bool,
    ) -> List["SearchSubmissionResult"]:
        _prepare_curses_screen(stdscr)
        self._init_colors()
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        if require_confirmation and not self._confirm_submission(stdscr=stdscr):
            self.was_canceled = True
            return []
        self._draw(stdscr=stdscr, complete=False)

        def on_result(result: "SearchSubmissionResult") -> None:
            slot = result.probe.index - 1
            if 0 <= slot < len(self.results):
                self.results[slot] = result
            self._draw(stdscr=stdscr, complete=False)

        submitted = submitter(on_result)
        self._wait_for_auto_close(stdscr=stdscr)
        return submitted

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)

    def _draw(
        self,
        stdscr: "curses.window",
        complete: bool,
        remaining_seconds: Optional[int] = None,
    ) -> None:
        stdscr.clear()
        try:
            _center_text(stdscr, 1, "Search Submission Status", curses.A_BOLD)
        except curses.error:
            pass
        height, width = stdscr.getmaxyx()
        for idx, row in enumerate(self.rows):
            y = 3 + idx
            if y >= height - 2:
                break
            status = self._status_for_row(index=idx)
            attr = self._status_attr(status=status)
            text = f"{status.upper():>7} {row}"
            try:
                stdscr.addstr(y, 2, text[: max(1, width - 4)], attr)
            except curses.error:
                continue
        footer = self._footer_text(complete=complete, remaining_seconds=remaining_seconds)
        try:
            _center_text(stdscr, height - 1, footer, curses.A_DIM | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()

    def _footer_text(self, complete: bool, remaining_seconds: Optional[int]) -> str:
        if not complete:
            return "Submitting probes..."
        assert remaining_seconds is not None
        return f"Completed. Auto-closing in {remaining_seconds}s..."

    def _status_for_row(self, index: int) -> str:
        result = self.results[index]
        if result is None:
            return "queued"
        return result.status

    def _status_attr(self, status: str) -> int:
        if status == "pending":
            return curses.color_pair(2) | curses.A_BOLD
        if status == "failed":
            return curses.color_pair(3) | curses.A_BOLD
        return curses.A_DIM

    def _confirm_submission(self, stdscr: "curses.window") -> bool:
        while True:
            self._draw_confirmation(stdscr=stdscr)
            key = stdscr.getch()
            if key in (ord("y"), ord("Y"), 10, 13, curses.KEY_ENTER):
                return True
            if key in (ord("n"), ord("N"), 27):
                return False

    def _draw_confirmation(self, stdscr: "curses.window") -> None:
        stdscr.clear()
        _center_text(stdscr, 1, "Confirm Search Submission", curses.A_BOLD)
        _center_text(
            stdscr,
            2,
            "Enter/y to submit • n/Esc to cancel",
            curses.A_DIM,
        )
        height, width = stdscr.getmaxyx()
        for idx, row in enumerate(self.rows):
            y = 4 + idx
            if y >= height - 2:
                break
            text = f"PLAN    {row}"
            try:
                stdscr.addstr(y, 2, text[: max(1, width - 4)], curses.A_BOLD)
            except curses.error:
                continue
        stdscr.refresh()

    def _wait_for_auto_close(self, stdscr: "curses.window") -> None:
        deadline = time.monotonic() + self.close_after_seconds
        stdscr.nodelay(True)
        while True:
            now = time.monotonic()
            remaining = int(max(0, deadline - now + 0.999))
            self._draw(stdscr=stdscr, complete=True, remaining_seconds=remaining)
            if now >= deadline:
                return
            time.sleep(0.1)
