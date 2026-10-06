import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlplat.run_store import RunStore, read_jsonl

COLUMNS = ("run", "verdict", "median step ms", "steps/s", "GPU util %", "data wait %", "main CPU %",
           "worker CPU %", "workers", "CPU-s/step", "agent CPU %", "agent sample ms")


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def row(run):
    summary = run.read_summary() or {}
    totals, roles, features = summary.get("totals", {}), summary.get("roles", {}), summary.get("features", {})
    gpu = summary.get("gpu") or {}
    end = next((record for record in reversed(read_jsonl(run.resources_path)) if record.get("type") == "end"), {})
    wait = features.get("data_wait_fraction")
    return (
        summary.get("name", run.run_id),
        summary.get("verdict", {}).get("primary", "-"),
        fmt((totals.get("step_time_median_s") or 0) * 1000 or None),
        fmt(totals.get("steps_per_sec")),
        fmt(gpu.get("util_mean_pct"), 0),
        fmt(wait * 100 if wait is not None else None, 0),
        fmt((roles.get("main") or {}).get("cpu_mean_pct"), 0),
        fmt((roles.get("workers") or {}).get("cpu_mean_pct"), 0),
        str((roles.get("workers") or {}).get("count_median", "-")),
        fmt(totals.get("cpu_seconds_per_step"), 3),
        fmt(end.get("agent_cpu_pct"), 3),
        fmt((end.get("sample_us_mean") or 0) / 1000 or None, 2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    runs = [run for run in RunStore(args.runs_dir).list() if run.read_meta().get("name", "").startswith(args.prefix)]
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for run in runs:
        print("| " + " | ".join(row(run)) + " |")


if __name__ == "__main__":
    main()
