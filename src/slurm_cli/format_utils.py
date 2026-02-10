from __future__ import annotations

import re
import string
import time
from typing import Iterable, List, Optional

ALLOWED_TEXT_CHARS = set(string.ascii_letters + string.digits + string.punctuation + " ")
_MEM_RE = re.compile(r"^\s*(\d+)\s*([KMG]?)\s*$", re.IGNORECASE)


def sanitize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    if "\x1b" in text:
        return ""
    return "".join(ch for ch in text if ch in ALLOWED_TEXT_CHARS)


def parse_time_string(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        day_part, time_part = value.split("-", 1) if "-" in value else (None, value)
        days = int(day_part) if day_part is not None else 0
        hours, minutes, seconds = (int(piece) for piece in time_part.split(":"))
    except (ValueError, TypeError):
        return None
    if any(val < 0 for val in (days, hours, minutes, seconds)):
        return None
    if minutes >= 60 or seconds >= 60:
        return None
    if seconds not in (0,):
        return None
    return days * 1440 + hours * 60 + minutes


def minutes_to_slurm_time(minutes: int) -> str:
    minutes = max(0, int(minutes))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{mins:02d}:00"
    return f"{hours:02d}:{mins:02d}:00"


def _format_units(units: Iterable[tuple[int, str]]) -> str:
    parts = [f"{value}{suffix}" for value, suffix in units if value]
    return " ".join(parts) if parts else "0m"


def format_minutes_phrase(minutes: int) -> str:
    minutes = max(0, int(minutes))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    units = ((days, "d"), (hours, "h"), (mins, "m"))
    return _format_units(units)


def format_seconds_phrase(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    units = ((days, "d"), (hrs, "h"), (mins, "m"), (secs, "s"))
    return _format_units(units)


def validate_time(value: str) -> bool:
    return parse_time_string(value) is not None


def parse_mem(mem: Optional[str]) -> Optional[str]:
    if not mem:
        return None
    match = _MEM_RE.match(mem)
    if not match:
        return None
    amount = int(match.group(1))
    if amount <= 0:
        return None
    unit = (match.group(2) or "G").upper()
    return f"{amount}{unit}"


def mem_to_gb(mem: Optional[str]) -> Optional[int]:
    normalized = parse_mem(mem)
    if not normalized:
        return None
    try:
        value = int(normalized[:-1])
    except ValueError:
        return None
    unit = normalized[-1]
    if unit == "G":
        return value
    if unit == "M":
        return max(1, int(round(value / 1024)))
    return None


def humanize_age(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "never"
    delta = max(0.0, time.time() - float(timestamp))
    days = delta / 86400.0
    if days < 1:
        return "today"
    if days < 7:
        return f"{int(days)}d ago"
    weeks = days / 7
    if weeks < 4:
        return f"{int(weeks)}w ago"
    months = days / 30
    if months < 12:
        return f"{int(months)}mo ago"
    years = days / 365
    return f"{int(years)}yr ago"


def build_time_options() -> List[int]:
    options: List[int] = []
    options.extend(range(5, 16, 5))
    options.extend(range(30, 61, 15))
    options.extend(range(90, 181, 30))
    options.extend(range(240, 421, 60))
    options.extend(range(540, 901, 120))
    current = 900
    step_hours = 4
    for _ in range(5):
        for _ in range(4):
            current += step_hours * 60
            options.append(current)
        step_hours *= 2
    options.extend([480])  # ensure 8h appears
    unique = sorted({max(5, value) for value in options})
    return unique


def build_memory_options() -> List[int]:
    values = list(range(1, 33))
    values.extend(range(36, 65, 4))
    values.extend(range(72, 129, 8))
    values.extend(range(144, 257, 16))
    values.extend(range(288, 513, 32))
    values.extend(range(576, 1025, 64))
    nice = {32, 40, 48, 50, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 896, 1024}
    values.extend(nice)
    return sorted({min(1024, max(1, val)) for val in values})


def build_timeout_options() -> List[int]:
    seconds: List[int] = []
    seconds.extend(range(15, 61, 15))
    seconds.extend(range(90, 241, 30))
    seconds.extend(range(300, 901, 60))
    seconds.extend(range(1200, 3601, 300))
    seconds.extend(range(4500, 10801, 900))
    seconds.extend(range(12600, 21601, 1800))
    seconds.extend(range(25200, 43201, 3600))
    return sorted({max(15, value) for value in seconds})


def nearest_index(values: List[int], target: int) -> int:
    if not values:
        return 0
    target = int(target)
    best_idx = 0
    best_delta = abs(values[0] - target)
    for idx, value in enumerate(values[1:], start=1):
        delta = abs(value - target)
        if delta < best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx
