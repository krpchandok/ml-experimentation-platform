from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from dashboard import wording
from dashboard.scene import SceneNumbers
from mlplat.analyzer import load_run_data
from mlplat.planner import read_plan
from mlplat.predict import PlanError, build_plan, extract_profile, model_step
from mlplat.run_store import RunStore, read_jsonl
from mlplat.targets import load_targets, local_target

MAX_NAMED_PROCESSES = 7
OTHER_PROCESSES = "other processes"
ROLLING_STEPS = 9
MIB = 1024 * 1024
RESERVED_METRIC_KEYS = {"t_mono", "t_wall", "step"}
DEFAULT_TOTAL_STEPS = 20000
OVEN_BOUND_WAIT = 0.10
QUEUE_PREP_FRACTION = 0.85


@dataclass
class RunRow:
    run_id: str
    name: str
    status: str
    started: Optional[datetime]
    duration_s: Optional[float]
    exit_code: Optional[int]
    config_hash: Optional[str]
    final_loss: Optional[float]
    final_accuracy: Optional[float]
    steps_per_sec: Optional[float]
    step_time_median_s: Optional[float]
    verdict: Optional[str]
    severity: Optional[str]


def store(runs_dir=None):
    return RunStore(runs_dir)


def run_rows(runs_dir=None):
    rows = []
    for run in store(runs_dir).list():
        meta = run.read_meta()
        summary = run.read_summary() or {}
        totals = summary.get("totals") or {}
        final = totals.get("final_metrics") or {}
        verdict = summary.get("verdict") or {}
        started = meta.get("start_wall")
        rows.append(RunRow(
            run_id=run.run_id,
            name=meta.get("name", ""),
            status=meta.get("status", ""),
            started=datetime.fromtimestamp(started) if started else None,
            duration_s=meta.get("duration_s"),
            exit_code=meta.get("exit_code"),
            config_hash=meta.get("config_hash"),
            final_loss=final.get("loss"),
            final_accuracy=final.get("accuracy", final.get("acc")),
            steps_per_sec=totals.get("steps_per_sec"),
            step_time_median_s=totals.get("step_time_median_s"),
            verdict=verdict.get("primary"),
            severity=verdict.get("severity"),
        ))
    rows.sort(key=lambda row: row.run_id, reverse=True)
    return rows


def runs_frame(runs_dir=None):
    return pd.DataFrame([row.__dict__ for row in run_rows(runs_dir)])


@dataclass
class RunDetail:
    run_id: str
    meta: dict
    summary: Optional[dict]
    origin: float
    processes: pd.DataFrame
    process_series: pd.DataFrame
    tree: pd.DataFrame
    gpu: pd.DataFrame
    metrics: pd.DataFrame
    window: Optional[tuple]


def process_labels(data):
    infos = sorted(data.processes.values(), key=lambda info: (info.role != "main", info.first_seen, info.pid))
    labels = {}
    named = 0
    for info in infos:
        if info.role == "helper" or named >= MAX_NAMED_PROCESSES:
            labels[info.key] = OTHER_PROCESSES
            continue
        named += 1
        labels[info.key] = "main" if info.role == "main" else f"worker {info.pid}"
    return labels


def load_detail(run):
    data = load_run_data(run)
    origin = data.meta.get("start_mono")
    if origin is None:
        origin = data.samples[0]["t_mono"] - data.samples[0]["dt"] if data.samples else 0.0
    labels = process_labels(data)
    label_order = list(dict.fromkeys(labels[key] for key in sorted(
        labels, key=lambda key: (labels[key] == OTHER_PROCESSES, labels[key] != "main", data.processes[key].first_seen))))

    rows = []
    for sample in data.samples:
        t = sample["t_mono"] - origin
        for entry in sample["procs"]:
            key = entry.get("identity")
            if key is None or key not in labels:
                continue
            rows.append({"t": t, "label": labels[key], "pid": entry["pid"], "cpu_pct": entry["cpu_pct"],
                         "rss_mib": entry["rss_kb"] / 1024.0})
    series = pd.DataFrame(rows, columns=["t", "label", "pid", "cpu_pct", "rss_mib"])
    if not series.empty:
        series = series.groupby(["t", "label"], as_index=False).agg(cpu_pct=("cpu_pct", "sum"),
                                                                    rss_mib=("rss_mib", "sum"))
        series["label"] = pd.Categorical(series["label"], categories=label_order, ordered=True)
        series = series.sort_values(["label", "t"])

    tree = pd.DataFrame([{
        "t": sample["t_mono"] - origin,
        "cpu_cores": sample["tree"]["cpu_pct"] / 100.0,
        "rss_mib": sample["tree"]["rss_kb"] / 1024.0,
        "read_mib_s": (sample["tree"].get("read_bytes_per_s") or 0.0) / MIB,
        "write_mib_s": (sample["tree"].get("write_bytes_per_s") or 0.0) / MIB,
        "system_available_mib": (sample.get("system", {}).get("mem_available_kb") or 0) / 1024.0,
    } for sample in data.samples if sample["procs"]], columns=["t", "cpu_cores", "rss_mib", "read_mib_s", "write_mib_s",
                                             "system_available_mib"])

    gpu_rows = []
    for sample in data.samples:
        gpu = sample.get("gpu")
        if isinstance(gpu, dict):
            for device in gpu.get("devices", []):
                gpu_rows.append({"t": sample["t_mono"] - origin, "device": f"GPU {device['index']}",
                                 "util_pct": device.get("util_pct"), "mem_util_pct": device.get("mem_util_pct"),
                                 "mem_used_mib": device.get("mem_used_mb"), "power_w": device.get("power_w"),
                                 "temp_c": device.get("temp_c")})
    gpu_frame = pd.DataFrame(gpu_rows, columns=["t", "device", "util_pct", "mem_util_pct", "mem_used_mib",
                                                "power_w", "temp_c"])

    summary = run.read_summary()
    window = None
    if summary and summary.get("window", {}).get("start_mono") is not None:
        window = (summary["window"]["start_mono"] - origin, summary["window"]["end_mono"] - origin)

    processes = pd.DataFrame(summary["processes"]) if summary and summary.get("processes") else pd.DataFrame()
    return RunDetail(run.run_id, data.meta, summary, origin, processes, series, tree, gpu_frame,
                     metrics_frame(data.metrics, origin), window)


def has_plan(run):
    return (run.path / "plan.json").exists()


def planned_total_steps(run, fallback=DEFAULT_TOTAL_STEPS):
    plan = read_plan(run)
    if plan:
        return int(plan["total_steps"])
    request = run.read_meta().get("plan_request") or {}
    return int(request.get("total_steps") or fallback)


def plan_in_memory(run, total_steps, prefer="balanced", targets_path=None):
    meta = run.read_meta()
    header = next((record for record in read_jsonl(run.resources_path) if record.get("type") == "header"), None)
    targets = [local_target(meta, header)] + load_targets(targets_path)
    return build_plan(extract_profile(run), targets, total_steps, prefer)


def scene_numbers(run, summary, estimate=False):
    roles = summary.get("roles") or {}
    workers_role = roles.get("workers") or {}
    workers = int(round(workers_role.get("count_median") or 0))
    worker_busy = workers_role.get("cpu_mean_pct")
    worker_busy = worker_busy / 100.0 if worker_busy is not None else None
    totals = summary.get("totals") or {}
    features = summary.get("features") or {}
    gpu = summary.get("gpu") or {}
    step = totals.get("step_time_mean_s") or totals.get("step_time_median_s") or 1.0
    if gpu.get("util_mean_pct") is not None:
        oven_busy = gpu["util_mean_pct"] / 100.0
    elif features.get("data_wait_fraction") is not None:
        oven_busy = 1 - features["data_wait_fraction"]
    else:
        oven_busy = None
    waiting = features.get("data_wait_fraction")
    if waiting is None and oven_busy is not None:
        waiting = 1 - oven_busy
    if waiting is None:
        try:
            profile = extract_profile(run, summary)
            _, prep, _ = model_step(profile, local_target(run.read_meta()), workers)
            compute = profile.compute_s
        except PlanError:
            compute, prep = step, step * QUEUE_PREP_FRACTION
    elif waiting <= OVEN_BOUND_WAIT:
        compute, prep = step, step * QUEUE_PREP_FRACTION
    else:
        compute, prep = step * (1 - waiting), step
    reads = [value for value in (features.get("main_read_bytes_per_s"), features.get("worker_read_bytes_per_s"))
             if value is not None]
    verdict = (summary.get("verdict") or {}).get("primary", "insufficient_data")
    return SceneNumbers(
        verdict=verdict,
        workers=workers,
        worker_busy=worker_busy,
        compute_s=compute,
        prep_s=prep,
        oven_busy=oven_busy,
        steps_per_s=totals.get("steps_per_sec"),
        read_mib_s=sum(reads) / MIB if reads else None,
        slowest_label=wording.slowest_label(verdict, workers, worker_busy),
        slowest_stage=wording.SLOWEST_STAGE.get(verdict),
        estimate=estimate,
    )


def oven_idle_seconds(summary):
    window = summary.get("window") or {}
    span = window.get("seconds")
    gpu = summary.get("gpu") or {}
    wait = (summary.get("features") or {}).get("data_wait_fraction")
    if not span:
        return None, None, None
    if gpu.get("util_mean_pct") is not None:
        busy = gpu["util_mean_pct"] / 100.0
    elif wait is not None:
        busy = 1 - wait
    else:
        return None, span, None
    return span * (1 - busy), span, busy


def metrics_frame(records, origin):
    frame = pd.DataFrame(records)
    if frame.empty or "t_mono" not in frame:
        return pd.DataFrame(columns=["t", "step", "step_time_s", "step_time_rolling_s"])
    frame = frame.sort_values("t_mono").reset_index(drop=True)
    frame["t"] = frame["t_mono"] - origin
    if "step" in frame:
        step_delta = frame["step"].diff()
        time_delta = frame["t_mono"].diff()
        frame["step_time_s"] = (time_delta / step_delta).where(step_delta > 0)
        frame["step_time_rolling_s"] = frame["step_time_s"].rolling(ROLLING_STEPS, min_periods=1, center=True).median()
    return frame


def metric_names(frame):
    return [column for column in frame.columns
            if column not in RESERVED_METRIC_KEYS | {"t", "step_time_s", "step_time_rolling_s"}
            and pd.api.types.is_numeric_dtype(frame[column])]


COMPARISON_FIELDS = (
    ("Median step time (ms)", ("totals", "step_time_median_s"), 1000.0, "lower"),
    ("p90 step time (ms)", ("totals", "step_time_p90_s"), 1000.0, "lower"),
    ("Steps / s", ("totals", "steps_per_sec"), 1.0, "higher"),
    ("CPU-seconds / step", ("totals", "cpu_seconds_per_step"), 1.0, "lower"),
    ("Avg cores (window)", ("totals", "avg_cores_window"), 1.0, None),
    ("Peak cores", ("totals", "peak_cores"), 1.0, None),
    ("Peak RSS (MiB)", ("totals", "peak_rss_mb"), 1.0, None),
    ("Main CPU (% of a core)", ("roles", "main", "cpu_mean_pct"), 1.0, None),
    ("Worker CPU (% of a core)", ("roles", "workers", "cpu_mean_pct"), 1.0, None),
    ("Workers (concurrent)", ("roles", "workers", "count_median"), 1.0, None),
    ("Data wait fraction (%)", ("features", "data_wait_fraction"), 100.0, "lower"),
    ("GPU utilization (%)", ("gpu", "util_mean_pct"), 1.0, "higher"),
    ("Duration (s)", ("totals", "duration_s"), 1.0, "lower"),
)


def dig(summary, path):
    value = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def comparison_frame(summary_a, summary_b):
    rows = []
    for label, path, scale, better in COMPARISON_FIELDS:
        a, b = dig(summary_a, path), dig(summary_b, path)
        if a is None and b is None:
            continue
        a = a * scale if isinstance(a, (int, float)) else None
        b = b * scale if isinstance(b, (int, float)) else None
        change = (b - a) / a * 100.0 if a not in (None, 0) and b is not None else None
        rows.append({"metric": label, "run A": a, "run B": b, "change %": change,
                     "better": better_side(change, better)})
    return pd.DataFrame(rows, columns=["metric", "run A", "run B", "change %", "better"])


def better_side(change, better):
    if change is None or better is None or abs(change) < 1.0:
        return ""
    improved = change < 0 if better == "lower" else change > 0
    return "B" if improved else "A"
