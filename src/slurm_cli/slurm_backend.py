from __future__ import annotations

import re
import select
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote


JOB_ID_RE = re.compile(r"job\s+(\d+)")


def build_srun(
    gpus: int,
    cpus: int,
    time_str: str,
    account: str,
    shell: str,
    mem: str,
) -> List[str]:
    """Construct the interactive `srun` command for terminal sessions.

    Args:
        gpus: Number of GPUs to request via `--gres`.
        cpus: CPUs per task supplied through `--cpus-per-task`.
        time_str: Slurm time string (HH:MM:SS or DD-HH:MM:SS).
        account: Slurm account to charge.
        shell: Shell executed under `srun --pty`.
        mem: Memory request string compatible with Slurm.

    Returns:
        Command arguments suitable for `subprocess` or exec.

    Example:
        >>> build_srun(gpus=1, cpus=4, time_str="00:30:00", account="P123",
        ...           shell="bash", mem="16G")[:4]
        ['srun', '--gres=gpu:1', '--cpus-per-task=4', '--time=00:30:00']
    """

    assert cpus >= 1, "cpus must be at least 1"
    cmd: List[str] = ["srun"]
    if gpus > 0:
        cmd.append(f"--gres=gpu:{gpus}")
    cmd.append(f"--cpus-per-task={cpus}")
    cmd += [
        f"--time={time_str}",
        f"--account={account}",
        f"--mem={mem}",
        "--pty",
        shell,
    ]
    return cmd


def build_sbatch(
    gpus: int,
    cpus: int,
    time_str: str,
    account: str,
    mem: str,
    email: Optional[str],
    job_name: str,
) -> List[str]:
    """Return an `sbatch` command that mirrors the interactive allocation.

    Args:
        gpus: Requested GPU count for the batch job.
        cpus: CPUs per task to reserve.
        time_str: Slurm time specification.
        account: Slurm account identifier.
        mem: Memory request in Slurm format.
        email: Optional address for BEGIN notifications.
        job_name: Identifier shown in Slurm queues.

    Returns:
        Parsed command list beginning with `sbatch`.

    Example:
        >>> build_sbatch(gpus=0, cpus=8, time_str="00:10:00", account="P123",
        ...              mem="8G", email=None, job_name="demo")[0]
        'sbatch'
    """

    assert cpus >= 1, "cpus must be at least 1"
    cmd: List[str] = [
        "sbatch",
        "--parsable",
        f"--cpus-per-task={cpus}",
        f"--time={time_str}",
        f"--account={account}",
        f"--mem={mem}",
        f"--job-name={job_name}",
    ]
    if gpus > 0:
        cmd.insert(1, f"--gres=gpu:{gpus}")
    if email:
        cmd += [f"--mail-user={email}", "--mail-type=BEGIN"]
    cmd += ["--wrap", "sleep infinity"]
    return cmd


def start_allocation_background(
    gpus: int,
    cpus: int,
    time_str: str,
    account: str,
    mem: str,
    job_name: str,
) -> Tuple[subprocess.Popen[str], Optional[str]]:
    """Spawn a blocking `srun` allocation that holds resources in the background.

    Args:
        gpus: Number of GPUs to request.
        cpus: CPUs per task for the held allocation.
        time_str: Requested walltime string.
        account: Account that will be charged.
        mem: Memory request string.
        job_name: Friendly identifier for the allocation.

    Returns:
        Tuple of the running `srun` process and the parsed Slurm job id (if any).

    Example:
        >>> start_allocation_background(0, 4, "00:05:00", "P123", "8G", "demo")  # doctest: +SKIP
        (...)
    """

    assert cpus >= 1, "cpus must be at least 1"
    cmd = ["srun"]
    if gpus > 0:
        cmd.append(f"--gres=gpu:{gpus}")
    cmd += [
        f"--cpus-per-task={cpus}",
        f"--time={time_str}",
        f"--account={account}",
        f"--mem={mem}",
        f"--job-name={job_name}",
        "--pty",
        "sleep",
        "infinity",
    ]
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    job_id = _consume_job_id(proc)
    return proc, job_id


def _consume_job_id(proc: subprocess.Popen[str]) -> Optional[str]:
    stderr = proc.stderr
    if stderr is None:
        proc.wait()
        print("DEBUG: stderr is None")
        return None
    deadline = time.time() + 10
    stderr_lines = []
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"DEBUG: Process exited early with code {proc.returncode}")
            if stderr_lines:
                print("DEBUG: stderr output:")
                print("\n".join(stderr_lines))
            if proc.stdout:
                stdout_content = proc.stdout.read()
                if stdout_content:
                    print("DEBUG: stdout output:")
                    print(stdout_content)
            return None
        ready, _, _ = select.select([stderr], [], [], 0.5)
        if not ready:
            continue
        line = stderr.readline()
        if not line:
            continue
        stderr_lines.append(line.strip())
        print(f"DEBUG: stderr line: {line.strip()}")
        match = JOB_ID_RE.search(line)
        if match:
            return match.group(1)
    print("DEBUG: Timeout waiting for job ID")
    if stderr_lines:
        print("DEBUG: All stderr lines collected:")
        print("\n".join(stderr_lines))
    return None


def get_node_for_job(job_id: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["squeue", "-j", job_id, "-h", "-o", "%N"], text=True
        )
        nodes = out.strip().split()
        return nodes[0] if nodes else None
    except Exception:
        return _fallback_node_lookup(job_id)


def _fallback_node_lookup(job_id: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
        match = re.search(r"NodeList=([\w\-,\[\]]+)", out)
        if not match:
            return None
        nodelist = match.group(1)
        expanded = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        )
        nodes = expanded.strip().splitlines()
        return nodes[0] if nodes else None
    except Exception:
        return None


def submit_batch_job(
    gpus: int,
    cpus: int,
    time_str: str,
    account: str,
    mem: str,
    email: Optional[str],
    job_name: str,
) -> Optional[str]:
    """Submit a batch job mirroring the requested interactive resources.

    Args:
        gpus: Number of GPUs for the allocation.
        cpus: CPUs per task value.
        time_str: Requested walltime string.
        account: Account identifier for the job.
        mem: Memory request string.
        email: Optional notification address.
        job_name: Slurm job name.

    Returns:
        Slurm job id if submission succeeds, otherwise ``None``.

    Example:
        >>> submit_batch_job(0, 4, "00:10:00", "P123", "8G", None, "demo")  # doctest: +SKIP
        '123456'
    """

    cmd = build_sbatch(gpus, cpus, time_str, account, mem, email, job_name)
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return out or None
    except subprocess.CalledProcessError as exc:
        print("ERROR: sbatch failed:", exc)
        return None
    except FileNotFoundError:
        print("ERROR: 'sbatch' not found on PATH.")
        return None


def open_vscode_on_host(hostname: str, path: Optional[Path] = None) -> int:
    cwd = Path.cwd() if path is None else Path(path)
    uri = f"vscode-remote://ssh-remote+{hostname}{quote(str(cwd))}"
    try:
        return subprocess.call(["code", "--new-window", "--folder-uri", uri])
    except FileNotFoundError:
        print(
            "ERROR: 'code' CLI not found. Install VS Code and enable the 'code' shell command."
        )
        return 127


def print_release_instructions(job_id: Optional[str], batch: bool = False) -> None:
    print("\n=== Session info ===")
    if job_id:
        print(f"Slurm job holding your allocation: {job_id}")
        print(f"To release it later: scancel {job_id}")
        if batch:
            print("Batch submission will email when the job begins.")
    else:
        print("No job id available.")
