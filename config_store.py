from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from slurm_cli.constants import (
    CONFIG_DIR,
    CONFIG_PATH,
    MAX_RECENT_ACCOUNTS,
)


def _normalize_recent_accounts(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    entries: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        account = item.get("account")
        if not isinstance(account, str) or not account:
            continue
        label = item.get("label") if isinstance(item.get("label"), str) else ""
        last_used = item.get("last_used")
        try:
            last_used_val = float(last_used) if last_used is not None else 0.0
        except (TypeError, ValueError):
            last_used_val = 0.0
        entries.append({"account": account, "label": label, "last_used": last_used_val})
    entries.sort(key=lambda e: e["last_used"], reverse=True)
    return entries[:MAX_RECENT_ACCOUNTS]


def _read_timeout_seconds(data: Dict[str, Any]) -> Optional[int]:
    raw = data.get("last_timeout_limit_seconds")
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    raw = data.get("last_timeout_limit")
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw) * 60
    return None


def _read_positive_int(data: Dict[str, Any], key: str) -> Optional[int]:
    raw = data.get(key)
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    return None


@dataclass
class Config:
    last_account: Optional[str] = None
    recent_accounts: List[Dict[str, Any]] = field(default_factory=list)
    last_time: Optional[str] = None
    last_mem: Optional[str] = None
    last_gpus: Optional[int] = None
    last_cpus: Optional[int] = None
    last_ui: Optional[str] = None
    last_timeout_mode: Optional[str] = None
    last_notify_email: Optional[str] = None
    last_timeout_limit_seconds: Optional[int] = None
    last_search_max_time_minutes: Optional[int] = None
    last_search_min_time_minutes: Optional[int] = None
    last_search_max_gpus: Optional[int] = None
    last_search_min_gpus: Optional[int] = None

    @classmethod
    def load(cls) -> "Config":
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                cfg = cls()
                cfg.last_account = data.get("last_account")
                cfg.recent_accounts = _normalize_recent_accounts(data.get("recent_accounts"))
                cfg.last_time = data.get("last_time")
                cfg.last_mem = data.get("last_mem")
                cfg.last_gpus = data.get("last_gpus")
                cfg.last_cpus = data.get("last_cpus")
                cfg.last_ui = data.get("last_ui")
                cfg.last_timeout_mode = data.get("last_timeout_mode")
                cfg.last_notify_email = data.get("last_notify_email")
                cfg.last_timeout_limit_seconds = _read_timeout_seconds(data)
                cfg.last_search_max_time_minutes = _read_positive_int(
                    data=data, key="last_search_max_time_minutes"
                )
                cfg.last_search_min_time_minutes = _read_positive_int(
                    data=data, key="last_search_min_time_minutes"
                )
                cfg.last_search_max_gpus = _read_positive_int(
                    data=data, key="last_search_max_gpus"
                )
                cfg.last_search_min_gpus = _read_positive_int(
                    data=data, key="last_search_min_gpus"
                )
                return cfg
        except Exception:
            pass
        return cls()

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, indent=2)
        except Exception:
            # Not fatal if saving fails.
            pass


def record_account_use(cfg: Config, entry: Dict[str, Any]) -> None:
    account_id = entry.get("account")
    if not isinstance(account_id, str) or not account_id:
        return
    label = entry.get("label") if isinstance(entry.get("label"), str) else ""
    cfg.last_account = account_id
    snapshot = {
        "account": account_id,
        "label": label,
        "last_used": time.time(),
    }
    filtered = [item for item in cfg.recent_accounts if item.get("account") != account_id]
    cfg.recent_accounts = [snapshot] + filtered[: MAX_RECENT_ACCOUNTS - 1]


def find_account_entry(cfg: Config, account_id: str) -> Optional[Dict[str, Any]]:
    for entry in cfg.recent_accounts:
        if entry.get("account") == account_id:
            return dict(entry)
    return None
