import os
import platform
import socket
import subprocess
from pathlib import Path

COMMAND_TIMEOUT_S = 5


def _run(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _cpu_model():
    for line in (_read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _mem_total_kb():
    for line in (_read_text("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def _os_name():
    for line in (_read_text("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return platform.system()


def _cgroup():
    for line in (_read_text("/proc/self/cgroup") or "").splitlines():
        if line.startswith("0::"):
            return line[3:]
    return None


def _gpus():
    output = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                   "--format=csv,noheader,nounits"])
    if not output:
        return []
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            index, name, memory_mib, driver = parts
            gpus.append({"index": int(index), "name": name, "memory_total_mib": int(float(memory_mib)),
                         "driver": driver})
    return gpus


def collect_host_info():
    release = platform.release()
    try:
        usable_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable_cpus = os.cpu_count()
    return {
        "hostname": socket.gethostname(),
        "os": _os_name(),
        "kernel": release,
        "arch": platform.machine(),
        "is_wsl": "microsoft" in release.lower(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "usable_cpus": usable_cpus,
        "mem_total_kb": _mem_total_kb(),
        "python": platform.python_version(),
        "cgroup": _cgroup(),
        "gpus": _gpus(),
    }


def collect_git_info(cwd):
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    if commit is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd)
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd),
        "dirty": bool(status),
        "root": _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd),
    }
