from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional


_CLUSTER_NAME_RE = re.compile(r"^\s*ClusterName\s*=\s*(\S+)\s*$", re.MULTILINE)
_PARTITION_NAME_RE = re.compile(r"^\s*PartitionName=(\S+)\s*$", re.MULTILINE)
_DEBUG_MAX_MINUTES = 60
_PITZER_STANDARD_MAX_CPUS = 40


@dataclass(frozen=True)
class PartitionRequest:
    """Normalized allocation request used for partition selection.

    Args:
        gpus: Requested GPU count for one Slurm allocation.
        cpus: Requested CPU count for one task.
        time_minutes: Requested walltime in whole minutes.
        mem_str: Slurm-compatible memory request string.

    Returns:
        Immutable request descriptor validated with assertions.

    Example:
        >>> PartitionRequest(gpus=1, cpus=4, time_minutes=30, mem_str="16G").gpus
        1
    """

    gpus: int
    cpus: int
    time_minutes: int
    mem_str: str

    def __post_init__(self) -> None:
        assert self.gpus >= 0, "gpus must be non-negative"
        assert self.cpus >= 1, "cpus must be at least 1"
        assert self.time_minutes >= 0, "time_minutes must be non-negative"
        assert bool(self.mem_str), "mem_str must not be empty"

    @property
    def uses_debug_window(self) -> bool:
        """Return true when the request fits the one-hour debug walltime."""

        return self.time_minutes <= _DEBUG_MAX_MINUTES


def parse_cluster_name(config_text: str) -> Optional[str]:
    """Extract the Slurm cluster name from ``scontrol show config`` text.

    Args:
        config_text: Raw config text returned by Slurm.

    Returns:
        Lowercase cluster name, or ``None`` when no cluster line is present.

    Example:
        >>> parse_cluster_name("ClusterName = ascend\\nSchedulerType = sched/backfill")
        'ascend'
    """

    match = _CLUSTER_NAME_RE.search(config_text)
    return match.group(1).lower() if match else None


def detect_cluster_name(
    load_config: Callable[[], str] | None = None,
) -> Optional[str]:
    """Return the current Slurm cluster name when Slurm is available.

    Args:
        load_config: Optional config loader used for tests.

    Returns:
        Lowercase cluster name, or ``None`` when detection fails.
    """

    config_loader = load_config or _load_slurm_config
    try:
        return parse_cluster_name(config_text=config_loader())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def list_partition_names(
    load_partitions: Callable[[], str] | None = None,
) -> tuple[str, ...]:
    """Return normalized partition names visible on the current cluster.

    Args:
        load_partitions: Optional loader returning raw ``scontrol show partition`` text.

    Returns:
        Sorted lowercase partition names. Returns an empty tuple if Slurm is unavailable.

    Example:
        >>> list_partition_names(load_partitions=lambda: "PartitionName=Quad\\nPartitionName=debug\\n")
        ('debug', 'quad')
    """

    partition_loader = load_partitions or _load_partition_listing
    try:
        partition_text = partition_loader()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ()
    names = {
        match.group(1).rstrip("*").lower()
        for match in _PARTITION_NAME_RE.finditer(partition_text)
    }
    return tuple(sorted(names))


def validate_partition_name(
    partition_name: str,
    available_partitions: tuple[str, ...] | None = None,
) -> str:
    """Normalize and validate a user-supplied partition override.

    Args:
        partition_name: Partition name supplied by the user.
        available_partitions: Optional tuple of legal partition names.

    Returns:
        Lowercase validated partition name.

    Example:
        >>> validate_partition_name("Quad", available_partitions=("debug", "quad"))
        'quad'
    """

    normalized_name = partition_name.strip().rstrip("*").lower()
    assert normalized_name, "partition_name must not be empty"
    if available_partitions and normalized_name not in available_partitions:
        raise ValueError(f"Unknown partition: {partition_name}")
    return normalized_name


def recommend_partition(
    request: PartitionRequest,
    cluster_name: Optional[str] = None,
) -> Optional[str]:
    """Choose the highest-priority supported partition for a request.

    Args:
        request: Normalized request shape to route.
        cluster_name: Optional known cluster name. When omitted, detect it live.

    Returns:
        Partition name, or ``None`` when the cluster/request is unsupported.

    Example:
        >>> recommend_partition(
        ...     request=PartitionRequest(gpus=1, cpus=4, time_minutes=30, mem_str="16G"),
        ...     cluster_name="cardinal",
        ... )
        'debug'
    """

    active_cluster = (cluster_name or detect_cluster_name() or "").lower()
    if request.gpus <= 0 or not active_cluster:
        return None
    if active_cluster == "ascend":
        return _recommend_ascend_partition(request=request)
    if active_cluster == "cardinal":
        return _recommend_cardinal_partition(request=request)
    if active_cluster == "pitzer":
        return _recommend_pitzer_partition(request=request)
    return None


def _load_slurm_config() -> str:
    return subprocess.check_output(["scontrol", "show", "config"], text=True)


def _load_partition_listing() -> str:
    return subprocess.check_output(["scontrol", "show", "partition"], text=True)


def _recommend_ascend_partition(request: PartitionRequest) -> str:
    if request.gpus == 1 and request.uses_debug_window:
        return "debug-nextgen"
    if request.gpus <= 3:
        return "nextgen"
    return "quad"


def _recommend_cardinal_partition(request: PartitionRequest) -> str:
    if request.gpus == 1 and request.uses_debug_window:
        return "debug"
    return "gpu"


def _recommend_pitzer_partition(request: PartitionRequest) -> str:
    partition_prefix = "gpudebug" if request.uses_debug_window else "gpu"
    if request.gpus >= 3:
        return f"{partition_prefix}-quad"
    if request.cpus > _PITZER_STANDARD_MAX_CPUS:
        return f"{partition_prefix}-exp"
    return partition_prefix
