from __future__ import annotations

import curses
from typing import Any, Dict, List, Optional, Tuple
import time

from slurm_cli.constants import (
    MAX_CPUS,
    TIMEOUT_IMPATIENT,
    TIMEOUT_NOTIFY,
    UI_TERMINAL,
    UI_VSCODE,
)
from slurm_cli.format_utils import (
    build_memory_options,
    build_time_options,
    build_timeout_options,
    format_minutes_phrase,
    format_seconds_phrase,
    humanize_age,
    mem_to_gb,
    minutes_to_slurm_time,
    nearest_index,
    parse_mem,
    parse_time_string,
    sanitize_text,
    validate_time,
)


TIME_MINUTE_OPTIONS = build_time_options()
MEMORY_GB_OPTIONS = build_memory_options()
TIMEOUT_LIMIT_OPTIONS = build_timeout_options()


def _center_text(stdscr, y: int, text: str, attr: int = 0) -> None:
    """Write `text` centered on row `y` using the provided attribute."""

    height, width = stdscr.getmaxyx()
    x = max(0, (width - len(text)) // 2)
    stdscr.addstr(y, x, text, attr)


def _prepare_curses_screen(stdscr: "curses.window") -> None:
    """Ensure the curses screen starts with an explicit black background.

    Example:
        >>> # Executed within curses.wrapper callbacks.
        ... _prepare_curses_screen(stdscr)  # doctest: +SKIP
    """

    curses.start_color()
    if curses.has_colors():
        curses.use_default_colors()
        curses.init_pair(1, -1, curses.COLOR_BLACK)
        stdscr.bkgd(" ", curses.color_pair(1))
    else:
        stdscr.bkgd(" ", curses.A_NORMAL)
    stdscr.clear()
    stdscr.refresh()


class ResourcePicker:
    def __init__(self, time_minutes: int, gpus: int, cpus: int, mem_gb: int):
        self.time_idx = nearest_index(TIME_MINUTE_OPTIONS, max(5, time_minutes))
        self.gpus = max(0, min(4, gpus))
        self.cpus = max(1, min(MAX_CPUS, cpus))
        self.mem_idx = nearest_index(MEMORY_GB_OPTIONS, max(1, mem_gb))
        self.focus = 0
        self.canceled = False
        self._digit_focus = -1
        self._digit_buffer = ""
        self._digit_timestamp = 0.0

    def run(self) -> Optional[Tuple[str, int, int, str]]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _fallback_prompt(self) -> Optional[Tuple[str, int, int, str]]:
        time_minutes = self._prompt_time()
        if time_minutes is None:
            return None
        gpus = self._prompt_gpus()
        if gpus is None:
            return None
        cpus = self._prompt_cpus()
        if cpus is None:
            return None
        mem_gb = self._prompt_memory()
        if mem_gb is None:
            return None
        return minutes_to_slurm_time(time_minutes), gpus, cpus, f"{mem_gb}G"

    def _prompt_time(self) -> Optional[int]:
        default = minutes_to_slurm_time(TIME_MINUTE_OPTIONS[self.time_idx])
        while True:
            try:
                raw = input(f"Compute time [{default}]: ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return TIME_MINUTE_OPTIONS[self.time_idx]
            if validate_time(raw):
                minutes = parse_time_string(raw)
                if minutes is not None:
                    self.time_idx = nearest_index(TIME_MINUTE_OPTIONS, minutes)
                    return TIME_MINUTE_OPTIONS[self.time_idx]
            print("Enter time as HH:MM:SS or DD-HH:MM:SS.")

    def _prompt_gpus(self) -> Optional[int]:
        while True:
            try:
                raw = input(f"GPUs [0-4] ({self.gpus}): ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return self.gpus
            if raw.isdigit() and 0 <= int(raw) <= 4:
                self.gpus = int(raw)
                return self.gpus
            print("Please enter a number between 0 and 4.")

    def _prompt_cpus(self) -> Optional[int]:
        """Prompt for CPU count when the curses UI is unavailable."""

        while True:
            try:
                raw = input(f"CPUs per task [1-{MAX_CPUS}] ({self.cpus}): ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return self.cpus
            if raw.isdigit() and 1 <= int(raw) <= MAX_CPUS:
                self.cpus = int(raw)
                return self.cpus
            print(f"Please enter a number between 1 and {MAX_CPUS}.")

    def _prompt_memory(self) -> Optional[int]:
        default = MEMORY_GB_OPTIONS[self.mem_idx]
        while True:
            try:
                raw = input(f"Memory in GB (1-1024) [{default}]: ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return default
            if raw.isdigit() and 1 <= int(raw) <= 1024:
                self.mem_idx = nearest_index(MEMORY_GB_OPTIONS, int(raw))
                return MEMORY_GB_OPTIONS[self.mem_idx]
            parsed = parse_mem(raw)
            mem_gb = mem_to_gb(parsed) if parsed else None
            if mem_gb:
                self.mem_idx = nearest_index(MEMORY_GB_OPTIONS, mem_gb)
                return MEMORY_GB_OPTIONS[self.mem_idx]
            print("Enter a whole number of GB or a value like 50G/50000M.")

    def _curses_main(self, stdscr):
        """Render the resource picker UI in a curses loop.

        Example:
            >>> # Used through curses.wrapper
            ... ResourcePicker(30, 1, 4, 10)._curses_main(stdscr)  # doctest: +SKIP
        """

        _prepare_curses_screen(stdscr)
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        while True:
            self._draw_resource_screen(stdscr)
            action = self._handle_resource_key(stdscr.getch())
            if isinstance(action, tuple):
                return action
            if action is False:
                self.canceled = True
                return None

    def _draw_resource_screen(self, stdscr) -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Resource Selection", curses.A_BOLD)
        rows = [
            ("Compute Time", format_minutes_phrase(TIME_MINUTE_OPTIONS[self.time_idx])),
            ("GPUs", str(self.gpus)),
            ("CPUs", str(self.cpus)),
            ("Memory", f"{MEMORY_GB_OPTIONS[self.mem_idx]}G"),
        ]
        start_y = 5
        for idx, (label, value) in enumerate(rows):
            attr = curses.A_REVERSE | curses.A_BOLD if idx == self.focus else curses.A_BOLD
            text = f"{label}: {value}"
            _center_text(stdscr, start_y + idx * 2, text, attr)
        _center_text(
            stdscr,
            start_y + len(rows) * 2,
            "Use arrow keys • Enter to accept • ESC to cancel",
            curses.A_DIM,
        )
        stdscr.refresh()

    def _handle_resource_key(self, ch: int):
        row_count = 4
        if ch == curses.KEY_UP:
            self.focus = (self.focus - 1) % row_count
        elif ch == curses.KEY_DOWN:
            self.focus = (self.focus + 1) % row_count
        elif ch == curses.KEY_LEFT:
            self._nudge(-1)
        elif ch == curses.KEY_RIGHT:
            self._nudge(1)
        elif ch in (10, 13, curses.KEY_ENTER):
            return (
                minutes_to_slurm_time(TIME_MINUTE_OPTIONS[self.time_idx]),
                self.gpus,
                self.cpus,
                f"{MEMORY_GB_OPTIONS[self.mem_idx]}G",
            )
        elif ch == 27:
            return False
        elif self.focus in (1, 2) and ord("0") <= ch <= ord("9"):
            self._apply_digit_input(ch)
        return True

    def _nudge(self, delta: int) -> None:
        if self.focus == 0:
            self.time_idx = max(0, min(len(TIME_MINUTE_OPTIONS) - 1, self.time_idx + delta))
        elif self.focus == 1:
            self.gpus = max(0, min(4, self.gpus + delta))
        elif self.focus == 2:
            self.cpus = max(1, min(MAX_CPUS, self.cpus + delta))
        else:
            self.mem_idx = max(0, min(len(MEMORY_GB_OPTIONS) - 1, self.mem_idx + delta))

    def _apply_digit_input(self, key_code: int) -> None:
        """Accumulate typed digits for GPU/CPU selections."""

        digit = chr(key_code)
        now = time.monotonic()
        if self._digit_focus != self.focus or now - self._digit_timestamp > 1.2:
            self._digit_buffer = digit
        else:
            self._digit_buffer += digit
        self._digit_focus = self.focus
        self._digit_timestamp = now
        value = int(self._digit_buffer)
        if self.focus == 1:
            self.gpus = max(0, min(4, value))
        elif self.focus == 2:
            self.cpus = max(1, min(MAX_CPUS, value))


class AccountPicker:
    def __init__(self, accounts: List[Dict[str, Any]], initial_account: Optional[str]):
        self.accounts: List[Dict[str, Any]] = accounts[:]
        self.focus = 0
        if initial_account:
            for idx, acct in enumerate(self.accounts):
                if acct.get("account") == initial_account:
                    self.focus = idx
                    break
        self.canceled = False

    def run(self) -> Optional[Dict[str, Any]]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _fallback_prompt(self) -> Optional[Dict[str, Any]]:
        while True:
            self._print_account_options()
            try:
                raw = input("Select account: ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw and self.accounts:
                entry = dict(self.accounts[0])
                entry["last_used"] = time.time()
                return entry
            if raw.isdigit():
                choice = int(raw) - 1
                if 0 <= choice < len(self.accounts):
                    entry = dict(self.accounts[choice])
                    entry["last_used"] = time.time()
                    return entry
                if choice == len(self.accounts):
                    return self._fallback_create_account()
            print("Choose a listed option.")

    def _print_account_options(self) -> None:
        print("\nAvailable accounts:")
        for idx, acct in enumerate(self.accounts, start=1):
            primary = acct.get("account", "?")
            label = acct.get("label") or ""
            suffix = f" — {label}" if label else ""
            age = humanize_age(_coerce_timestamp(acct.get("last_used")))
            print(f"  {idx}. {primary}{suffix} ({age})")
        print(f"  {len(self.accounts) + 1}. Add a new account…")

    def _fallback_create_account(self) -> Optional[Dict[str, Any]]:
        account_id = self._prompt_text("New account id: ")
        if not account_id:
            return None
        label = self._prompt_text("Description: ")
        if not label:
            return None
        return {"account": account_id, "label": label, "last_used": time.time()}

    def _prompt_text(self, prompt: str) -> Optional[str]:
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if raw:
                return raw
            print("Value is required.")

    def _curses_main(self, stdscr):
        """Render the account picker UI until the user selects an option.

        Example:
            >>> AccountPicker([], None)._curses_main(stdscr)  # doctest: +SKIP
        """

        _prepare_curses_screen(stdscr)
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        total = len(self.accounts) + 1
        while True:
            self._draw_account_screen(stdscr)
            ch = stdscr.getch()
            if ch == curses.KEY_UP:
                self.focus = (self.focus - 1) % total
            elif ch == curses.KEY_DOWN:
                self.focus = (self.focus + 1) % total
            elif ch in (10, 13, curses.KEY_ENTER):
                return self._select_account(stdscr)
            elif ch == 27:
                self.canceled = True
                return None
            elif ch in (ord("n"), ord("N")):
                created = self._create_account(stdscr)
                if created:
                    return created

    def _draw_account_screen(self, stdscr) -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Account Selection", curses.A_BOLD)
        rows: List[str] = []
        for idx, acct in enumerate(self.accounts):
            primary = acct.get("account", "?")
            label = acct.get("label") or ""
            marker = "*" if idx == self.focus else " "
            text = f"[{marker}] {primary}"
            if label:
                text += f" — {label}"
            rows.append(text)
        marker_new = "*" if self.focus == len(self.accounts) else " "
        rows.append(f"[{marker_new}] Add a new account…")
        for idx, text in enumerate(rows):
            attr = curses.A_REVERSE | curses.A_BOLD if idx == self.focus else curses.A_BOLD
            _center_text(stdscr, 5 + idx * 2, text, attr)
            if idx < len(self.accounts):
                age = humanize_age(_coerce_timestamp(self.accounts[idx].get("last_used")))
                attr_age = curses.A_DIM | (curses.A_REVERSE if idx == self.focus else 0)
                _center_text(stdscr, 6 + idx * 2, age, attr_age)
        _center_text(
            stdscr,
            5 + len(rows) * 2,
            "Use arrow keys • Enter to select • n to add • ESC to cancel",
            curses.A_DIM,
        )
        stdscr.refresh()

    def _select_account(self, stdscr) -> Optional[Dict[str, Any]]:
        if self.focus < len(self.accounts):
            entry = dict(self.accounts[self.focus])
            entry["last_used"] = time.time()
            return entry
        return self._create_account(stdscr)

    def _create_account(self, stdscr) -> Optional[Dict[str, Any]]:
        account_id = self._text_input(stdscr, "New account id:")
        if not account_id:
            return None
        label = self._text_input(stdscr, "Description (required):")
        if not label:
            return None
        return {"account": account_id, "label": label, "last_used": time.time()}

    def _text_input(self, stdscr, prompt: str) -> Optional[str]:
        curses.curs_set(1)
        buffer: List[str] = []
        try:
            while True:
                stdscr.clear()
                _center_text(stdscr, 2, prompt, curses.A_BOLD)
                _center_text(stdscr, 4, "".join(buffer))
                _center_text(stdscr, 8, "Press Enter to accept • ESC to cancel", curses.A_DIM)
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (10, 13, curses.KEY_ENTER):
                    return sanitize_text("".join(buffer))
                if ch == 27:
                    return None
                if ch in (curses.KEY_BACKSPACE, 127, curses.KEY_DC):
                    if buffer:
                        buffer.pop()
                elif 32 <= ch <= 126:
                    char = chr(ch)
                    if sanitize_text(char):
                        buffer.append(char)
        finally:
            curses.curs_set(0)
        return None


def _coerce_timestamp(raw: Any) -> Optional[float]:
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


class TimeoutSettingsPicker:
    def __init__(self, mode: str, limit_seconds: int, email: Optional[str]):
        self.mode_idx = 0 if mode == TIMEOUT_IMPATIENT else 1
        self.limit_idx = nearest_index(TIMEOUT_LIMIT_OPTIONS, limit_seconds)
        self.email = sanitize_text(email)
        self.focus = 0
        self.canceled = False

    def run(self) -> Optional[Tuple[str, int, str]]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _fallback_prompt(self) -> Optional[Tuple[str, int, str]]:
        mode = self._prompt_mode()
        if not mode:
            return None
        limit = self._prompt_limit()
        if limit is None:
            return None
        email = self._prompt_email(mode)
        if email is None and mode == TIMEOUT_NOTIFY:
            return None
        return mode, limit, email or ""

    def _prompt_mode(self) -> Optional[str]:
        default = TIMEOUT_IMPATIENT if self.mode_idx == 0 else TIMEOUT_NOTIFY
        while True:
            try:
                raw = input(f"Timeout mode [{default}]: ").strip().lower()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return default
            if raw in (TIMEOUT_IMPATIENT, TIMEOUT_NOTIFY):
                self.mode_idx = 0 if raw == TIMEOUT_IMPATIENT else 1
                return raw
            print("Enter 'impatient' or 'notify'.")

    def _prompt_limit(self) -> Optional[int]:
        default = TIMEOUT_LIMIT_OPTIONS[self.limit_idx]
        while True:
            try:
                raw = input(f"Timeout limit seconds [{default}]: ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                return default
            if raw.isdigit():
                seconds = int(raw)
                if seconds >= 15:
                    self.limit_idx = nearest_index(TIMEOUT_LIMIT_OPTIONS, seconds)
                    return TIMEOUT_LIMIT_OPTIONS[self.limit_idx]
            print("Enter the timeout in seconds (15 or greater).")

    def _prompt_email(self, mode: str) -> Optional[str]:
        if mode != TIMEOUT_NOTIFY:
            return ""
        while True:
            try:
                raw = input(f"Notify email [{self.email}]: ").strip()
            except EOFError:
                return None
            raw = sanitize_text(raw)
            if not raw:
                if self.email:
                    return self.email
                print("Email is required for notify mode.")
                continue
            self.email = raw
            return self.email

    def _curses_main(self, stdscr):
        """Drive the timeout settings picker using a curses loop.

        Example:
            >>> TimeoutSettingsPicker('impatient', 600, '')._curses_main(stdscr)  # doctest: +SKIP
        """

        _prepare_curses_screen(stdscr)
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        while True:
            self._draw_timeout_screen(stdscr)
            result = self._handle_timeout_key(stdscr.getch(), stdscr)
            if isinstance(result, tuple):
                return result
            if result is False:
                self.canceled = True
                return None

    def _draw_timeout_screen(self, stdscr) -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Timeout Settings", curses.A_BOLD)
        modes = [("Impatient", TIMEOUT_IMPATIENT), ("Notify", TIMEOUT_NOTIFY)]
        mode_line = []
        for idx, (label, _) in enumerate(modes):
            marker = "*" if idx == self.mode_idx else " "
            mode_line.append(f"[{marker}] {label}")
        _center_text(stdscr, 5, "   ".join(mode_line), curses.A_REVERSE | curses.A_BOLD if self.focus == 0 else curses.A_BOLD)
        explanation = {
            TIMEOUT_IMPATIENT: "If no allocation is assigned within the limit, exit immediately.",
            TIMEOUT_NOTIFY: "If the limit hits, submit a batch job and email when it starts.",
        }[modes[self.mode_idx][1]]
        _center_text(stdscr, 6, explanation, curses.A_DIM)
        limit_text = f"Timeout limit: {format_seconds_phrase(TIMEOUT_LIMIT_OPTIONS[self.limit_idx])}"
        attr = curses.A_REVERSE | curses.A_BOLD if self.focus == 1 else curses.A_BOLD
        _center_text(stdscr, 8, limit_text, attr)
        email_label = "Notify email: "
        email_value = self.email or "(required)" if modes[self.mode_idx][1] == TIMEOUT_NOTIFY else "N/A"
        email_attr = curses.A_REVERSE | curses.A_BOLD if self.focus == 2 else curses.A_BOLD
        _center_text(stdscr, 10, email_label + email_value, email_attr)
        _center_text(stdscr, 12, "Use arrow keys • Enter to accept • ESC to cancel", curses.A_DIM)
        stdscr.refresh()

    def _handle_timeout_key(self, ch: int, stdscr):
        if ch == curses.KEY_UP:
            self.focus = (self.focus - 1) % 3
        elif ch == curses.KEY_DOWN:
            self.focus = (self.focus + 1) % 3
        elif ch == curses.KEY_LEFT:
            self._adjust(-1)
        elif ch == curses.KEY_RIGHT:
            self._adjust(1)
        elif ch in (10, 13, curses.KEY_ENTER):
            email = self.email if self.mode_idx == 1 else ""
            if self.mode_idx == 1 and not email:
                email = self._inline_email_input(stdscr)
                if not email:
                    return True
                self.email = email
            return (
                TIMEOUT_IMPATIENT if self.mode_idx == 0 else TIMEOUT_NOTIFY,
                TIMEOUT_LIMIT_OPTIONS[self.limit_idx],
                self.email if self.mode_idx == 1 else "",
            )
        elif ch == 27:
            return False
        elif self.focus == 2 and 32 <= ch <= 126 and self.mode_idx == 1:
            char = chr(ch)
            if char in sanitize_text(char):
                self.email += char
        elif self.focus == 2 and ch in (curses.KEY_BACKSPACE, 127, curses.KEY_DC):
            self.email = self.email[:-1]
        return True

    def _adjust(self, delta: int) -> None:
        if self.focus == 0:
            self.mode_idx = (self.mode_idx + delta) % 2
        elif self.focus == 1:
            self.limit_idx = max(0, min(len(TIMEOUT_LIMIT_OPTIONS) - 1, self.limit_idx + delta))

    def _inline_email_input(self, stdscr) -> str:
        curses.curs_set(1)
        buffer = list(self.email)
        try:
            while True:
                stdscr.clear()
                _center_text(stdscr, 2, "Notify email", curses.A_BOLD)
                _center_text(stdscr, 4, "".join(buffer))
                _center_text(stdscr, 8, "Press Enter to accept • ESC to cancel", curses.A_DIM)
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (10, 13, curses.KEY_ENTER):
                    return sanitize_text("".join(buffer))
                if ch == 27:
                    return ""
                if ch in (curses.KEY_BACKSPACE, 127, curses.KEY_DC):
                    if buffer:
                        buffer.pop()
                elif 32 <= ch <= 126:
                    char = chr(ch)
                    if sanitize_text(char):
                        buffer.append(char)
        finally:
            curses.curs_set(0)


class UIModePicker:
    def __init__(self, initial: str):
        self.options = [UI_TERMINAL, UI_VSCODE]
        self.idx = 0 if initial not in self.options else self.options.index(initial)
        self.focus = self.idx
        self.canceled = False

    def run(self) -> Optional[str]:
        try:
            return curses.wrapper(self._curses_main)
        except curses.error:
            return self._fallback_prompt()

    def _fallback_prompt(self) -> Optional[str]:
        default = self.options[self.idx]
        while True:
            raw = input(f"UI mode [terminal/vscode] ({default}): ").strip().lower()
            raw = sanitize_text(raw)
            if not raw:
                return default
            if raw in self.options:
                self.idx = self.options.index(raw)
                return raw
            print("Enter 'terminal' or 'vscode'.")

    def _curses_main(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        while True:
            self._draw_ui_screen(stdscr)
            ch = stdscr.getch()
            if ch == curses.KEY_LEFT:
                self.focus = (self.focus - 1) % 2
                self.idx = self.focus
            elif ch == curses.KEY_RIGHT:
                self.focus = (self.focus + 1) % 2
                self.idx = self.focus
            elif ch in (10, 13, curses.KEY_ENTER):
                return self.options[self.idx]
            elif ch == 27:
                self.canceled = True
                return None

    def _draw_ui_screen(self, stdscr) -> None:
        stdscr.clear()
        _center_text(stdscr, 2, "Attach Where?", curses.A_BOLD)
        labels = [("Terminal", UI_TERMINAL), ("VS Code", UI_VSCODE)]
        parts = []
        options: List[str] = []
        for idx, (label, _) in enumerate(labels):
            marker = "*" if self.idx == idx else " "
            options.append(f"[{marker}] {label}")
        width = stdscr.getmaxyx()[1]
        total = sum(len(item) for item in options) + 3 * (len(options) - 1)
        x = max(0, (width - total) // 2)
        for idx, item in enumerate(options):
            attr = curses.A_BOLD | (curses.A_REVERSE if self.focus == idx else 0)
            stdscr.addstr(5, x, item, attr)
            x += len(item) + 3
        _center_text(stdscr, 7, "Use arrow keys • Enter to accept • ESC to cancel", curses.A_DIM)
        stdscr.refresh()
