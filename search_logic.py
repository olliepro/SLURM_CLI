from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple

from slurm_cli.format_utils import minutes_to_slurm_time
from slurm_cli.slurm_backend import build_sbatch, submit_batch_job

SEARCH_STATUS_PENDING = "pending"
SEARCH_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class SearchBounds:
    """Immutable bounds for two-phase search probe generation.

    Args:
        max_time_minutes: Largest time probe in minutes.
        min_time_minutes: Lowest allowed time probe in minutes.
        max_gpus: Largest GPU probe count.
        min_gpus: Lowest allowed GPU probe count.
        switch_minutes: Time threshold to switch from time to GPU halving.

    Returns:
        None. Validation occurs via assertions in ``__post_init__``.

    Example:
        >>> SearchBounds(240, 30, 4, 1, 60).max_gpus
        4
    """

    max_time_minutes: int
    min_time_minutes: int
    max_gpus: int
    min_gpus: int
    switch_minutes: int

    def __post_init__(self) -> None:
        assert self.max_time_minutes >= self.min_time_minutes >= 30
        assert self.max_gpus >= self.min_gpus >= 1
        assert self.switch_minutes > 0


@dataclass(frozen=True)
class SearchProbe:
    """One probe configuration submitted during search.

    Args:
        time_minutes: Time request in minutes.
        gpus: GPU request count.
        cpus: CPU request count.
        mem_str: Memory request in Slurm format.
        phase: Search phase label, ``time`` or ``gpu``.
        index: 1-based position in submission order.

    Returns:
        Dataclass with canonical formatting helpers for display/job names.
    """

    time_minutes: int
    gpus: int
    cpus: int
    mem_str: str
    phase: str
    index: int

    def human_time_label(self) -> str:
        """Return compact time text (e.g. ``1h30m``) for this probe."""

        return format_compact_minutes(minutes=self.time_minutes)

    def job_label(self, prefix: str) -> str:
        """Build a canonical Slurm job name for this probe."""

        return f"{self.human_time_label()}-g{self.gpus}-{prefix}"

    def to_slurm_time(self) -> str:
        """Return probe time in Slurm walltime format."""

        return minutes_to_slurm_time(minutes=self.time_minutes)

    def summary_line(self, prefix: str) -> str:
        """Return a compact single-line summary used in CLI/UI views."""

        return (
            f"#{self.index:02d} {self.phase.upper()} t={self.human_time_label()} "
            f"g={self.gpus} c={self.cpus} m={self.mem_str} "
            f"name={self.job_label(prefix=prefix)}"
        )


@dataclass(frozen=True)
class SearchSubmissionResult:
    """Submission outcome for one search probe.

    Args:
        probe: Probe definition submitted.
        job_name: Slurm job name used for submission.
        job_id: Slurm job id, when submission succeeds.
        status: ``pending`` or ``failed``.

    Returns:
        Dataclass for plain-text and curses reporting.
    """

    probe: SearchProbe
    job_name: str
    job_id: Optional[str]
    status: str

    def summary_line(self) -> str:
        """Return one formatted line for plain submission reports."""

        job_token = self.job_id or "-"
        return (
            f"{self.status.upper():7} {self.probe.human_time_label():>6} "
            f"g={self.probe.gpus} job={self.job_name} id={job_token}"
        )


def format_compact_minutes(minutes: int) -> str:
    """Convert minutes to compact text (``2d4h``, ``1h30m``, ``30m``).

    Args:
        minutes: Duration in whole minutes.

    Returns:
        Compact duration string for job naming and summaries.

    Example:
        >>> format_compact_minutes(90)
        '1h30m'
    """

    total = max(0, int(minutes))
    days, rem = divmod(total, 1440)
    hours, mins = divmod(rem, 60)
    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins or not parts:
        parts.append(f"{mins}m")
    return "".join(parts)


def build_probe_command(
    probe: SearchProbe,
    account: str,
    email: str,
    job_prefix: str,
) -> List[str]:
    """Build the ``sbatch`` command for a single probe.

    Args:
        probe: Probe request to convert into ``sbatch`` args.
        account: Account charged for the batch job.
        email: Notification address for ``BEGIN`` mail.
        job_prefix: Prefix used in job naming.

    Returns:
        Command list suitable for subprocess execution.
    """

    return build_sbatch(
        gpus=probe.gpus,
        cpus=probe.cpus,
        time_str=probe.to_slurm_time(),
        account=account,
        mem=probe.mem_str,
        email=email,
        job_name=probe.job_label(prefix=job_prefix),
    )


def build_search_probes(bounds: SearchBounds, cpus: int, mem_str: str) -> List[SearchProbe]:
    """Generate ordered probes for the two-phase halving search.

    Args:
        bounds: Search bounds and switch threshold.
        cpus: Fixed CPU request for all probes.
        mem_str: Fixed memory request for all probes.

    Returns:
        Ordered probe list, deduplicated by ``(time_minutes, gpus)``.

    Example:
        >>> bounds = SearchBounds(240, 30, 4, 1, 60)
        >>> [probe.gpus for probe in build_search_probes(bounds, 8, '50G')]
        [4, 4, 4, 2, 1]
    """

    assert cpus >= 1
    time_phase = _build_time_phase(bounds=bounds, cpus=cpus, mem_str=mem_str)
    base_time = time_phase[-1].time_minutes
    gpu_phase = _build_gpu_phase(
        bounds=bounds,
        base_time_minutes=base_time,
        cpus=cpus,
        mem_str=mem_str,
        start_index=len(time_phase),
    )
    merged = _remove_duplicate_probes(probes=time_phase + gpu_phase)
    return _reindex_probes(probes=merged)


def submit_search_probes(
    probes: Sequence[SearchProbe],
    account: str,
    email: str,
    gap_seconds: int,
    job_prefix: str,
    dry_run: bool,
    status_callback: Optional[Callable[[SearchSubmissionResult], None]] = None,
    submit_batch: Callable[..., Optional[str]] = submit_batch_job,
) -> List[SearchSubmissionResult]:
    """Submit search probes sequentially with optional callback updates.

    Args:
        probes: Ordered probe list to submit.
        account: Slurm account identifier.
        email: Notify email required for begin notifications.
        gap_seconds: Delay between consecutive submissions.
        job_prefix: Prefix for probe job names.
        dry_run: If true, do not submit and return synthetic pending results.
        status_callback: Optional callback called after each result.
        submit_batch: Submission function injected for testing.

    Returns:
        Result list preserving probe order.
    """

    assert email
    results: List[SearchSubmissionResult] = []
    for idx, probe in enumerate(probes):
        result = _submit_probe(
            probe=probe,
            account=account,
            email=email,
            job_prefix=job_prefix,
            dry_run=dry_run,
            submit_batch=submit_batch,
        )
        results.append(result)
        if status_callback is not None:
            status_callback(result)
        if _should_pause_between_submissions(
            current_index=idx,
            total=len(probes),
            gap_seconds=gap_seconds,
            dry_run=dry_run,
        ):
            time.sleep(gap_seconds)
    return results


def _build_time_phase(bounds: SearchBounds, cpus: int, mem_str: str) -> List[SearchProbe]:
    probes = [
        SearchProbe(
            time_minutes=bounds.max_time_minutes,
            gpus=bounds.max_gpus,
            cpus=cpus,
            mem_str=mem_str,
            phase="time",
            index=1,
        )
    ]
    current_time = bounds.max_time_minutes
    while True:
        candidate = max(bounds.min_time_minutes, current_time // 2)
        if candidate == current_time:
            return probes
        probes.append(
            SearchProbe(
                time_minutes=candidate,
                gpus=bounds.max_gpus,
                cpus=cpus,
                mem_str=mem_str,
                phase="time",
                index=len(probes) + 1,
            )
        )
        current_time = candidate
        if current_time < bounds.switch_minutes:
            return probes


def _build_gpu_phase(
    bounds: SearchBounds,
    base_time_minutes: int,
    cpus: int,
    mem_str: str,
    start_index: int,
) -> List[SearchProbe]:
    probes: List[SearchProbe] = []
    current_gpu = bounds.max_gpus
    while True:
        candidate = max(bounds.min_gpus, current_gpu // 2)
        if candidate == current_gpu:
            return probes
        probes.append(
            SearchProbe(
                time_minutes=base_time_minutes,
                gpus=candidate,
                cpus=cpus,
                mem_str=mem_str,
                phase="gpu",
                index=start_index + len(probes) + 1,
            )
        )
        current_gpu = candidate


def _remove_duplicate_probes(probes: Sequence[SearchProbe]) -> List[SearchProbe]:
    seen: Set[Tuple[int, int]] = set()
    unique: List[SearchProbe] = []
    for probe in probes:
        key = (probe.time_minutes, probe.gpus)
        if key in seen:
            continue
        seen.add(key)
        unique.append(probe)
    return unique


def _reindex_probes(probes: Sequence[SearchProbe]) -> List[SearchProbe]:
    reindexed: List[SearchProbe] = []
    for idx, probe in enumerate(probes, start=1):
        reindexed.append(
            SearchProbe(
                time_minutes=probe.time_minutes,
                gpus=probe.gpus,
                cpus=probe.cpus,
                mem_str=probe.mem_str,
                phase=probe.phase,
                index=idx,
            )
        )
    return reindexed


def _submit_probe(
    probe: SearchProbe,
    account: str,
    email: str,
    job_prefix: str,
    dry_run: bool,
    submit_batch: Callable[..., Optional[str]],
) -> SearchSubmissionResult:
    if dry_run:
        return SearchSubmissionResult(
            probe=probe,
            job_name=probe.job_label(prefix=job_prefix),
            job_id="DRY-RUN",
            status=SEARCH_STATUS_PENDING,
        )
    job_id = submit_batch(
        gpus=probe.gpus,
        cpus=probe.cpus,
        time_str=probe.to_slurm_time(),
        account=account,
        mem=probe.mem_str,
        email=email,
        job_name=probe.job_label(prefix=job_prefix),
    )
    return SearchSubmissionResult(
        probe=probe,
        job_name=probe.job_label(prefix=job_prefix),
        job_id=job_id,
        status=SEARCH_STATUS_PENDING if job_id else SEARCH_STATUS_FAILED,
    )


def _should_pause_between_submissions(
    current_index: int,
    total: int,
    gap_seconds: int,
    dry_run: bool,
) -> bool:
    if dry_run:
        return False
    if gap_seconds <= 0:
        return False
    return current_index < (total - 1)
