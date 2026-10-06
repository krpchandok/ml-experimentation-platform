import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_helpers import WORKLOAD, read_records

DEFAULT_AGENT = Path(__file__).resolve().parents[2] / "build" / "mlplat-agent"


def measure(agent, seconds, interval_ms, extra_idle, background):
    sleepers = [subprocess.Popen(["sleep", str(seconds + 30)]) for _ in range(background)]
    try:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "resources.jsonl"
            workload = subprocess.Popen([
                sys.executable, str(WORKLOAD), "--role", "root", "--seconds", str(seconds), "--dir", directory,
                "--pid-file", str(Path(directory) / "pids.json"), "--extra-idle", str(extra_idle),
            ])
            agent_process = subprocess.Popen([str(agent), "--pid", str(workload.pid), "--interval-ms",
                                              str(interval_ms), "--out", str(out)])
            workload.wait()
            _, _, usage = os.wait4(agent_process.pid, 0)
            end = read_records(out)[-1]
    finally:
        for sleeper in sleepers:
            sleeper.kill()
            sleeper.wait()

    external_cpu = usage.ru_utime + usage.ru_stime
    return {
        "interval_ms": interval_ms,
        "tracked_processes": end["max_tracked"],
        "background_processes": background,
        "samples": end["samples"],
        "wall_s": round(end["agent_wall_s"], 2),
        "cpu_s_self": round(end["agent_cpu_s"], 4),
        "cpu_s_wait4": round(external_cpu, 4),
        "cpu_pct_of_core": round(external_cpu / end["agent_wall_s"] * 100, 3),
        "sample_us_mean": round(end["sample_us_mean"], 1),
        "sample_us_max": round(end["sample_us_max"], 1),
        "max_rss_kb": usage.ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=str(DEFAULT_AGENT))
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--intervals", default="1000,100")
    parser.add_argument("--extra-idle", default="0,60")
    parser.add_argument("--background", default="0,1000")
    args = parser.parse_args()

    for interval_ms in [int(value) for value in args.intervals.split(",")]:
        for extra_idle in [int(value) for value in args.extra_idle.split(",")]:
            for background in [int(value) for value in args.background.split(",")]:
                result = measure(args.agent, args.seconds, interval_ms, extra_idle, background)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
