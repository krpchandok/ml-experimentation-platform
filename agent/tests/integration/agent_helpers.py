import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

WORKLOAD = Path(__file__).resolve().parent / "workload.py"


def read_records(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_pid_series(records):
    series = defaultdict(list)
    for record in records:
        if record["type"] != "sample":
            continue
        for process in record["procs"]:
            series[process["pid"]].append({**process, "dt": record["dt"]})
    return series


def mean_cpu(entries):
    total_time = sum(entry["dt"] for entry in entries)
    if total_time == 0:
        return 0.0
    return sum(entry["cpu_pct"] * entry["dt"] for entry in entries) / total_time


def total_bytes(entries, key):
    return sum((entry[key] or 0) * entry["dt"] for entry in entries)


def run_workload_with_agent(agent_binary, tmp_path, seconds=4.0, interval_ms=250, extra_idle=0):
    pid_file = tmp_path / "pids.json"
    out = tmp_path / "resources.jsonl"
    workload = subprocess.Popen([
        sys.executable, str(WORKLOAD), "--role", "root", "--seconds", str(seconds), "--dir", str(tmp_path),
        "--pid-file", str(pid_file), "--extra-idle", str(extra_idle),
    ])
    agent = subprocess.Popen([str(agent_binary), "--pid", str(workload.pid), "--interval-ms", str(interval_ms),
                              "--out", str(out)])
    workload.wait(timeout=seconds + 60)
    agent_code = agent.wait(timeout=30)
    roles = json.loads(pid_file.read_text())
    roles["grandchild"] = int((tmp_path / "grandchild.pid").read_text())
    return roles, read_records(out), agent_code
