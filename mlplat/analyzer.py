import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from mlplat import verdict as rules
from mlplat.run_store import read_jsonl, write_json_atomic

SUMMARY_SCHEMA = 1
HELPER_CMDLINE_MARKERS = ("multiprocessing.resource_tracker", "multiprocessing.forkserver",
                          "multiprocessing.semaphore_tracker")
HELPER_COMMS = ("nvidia-smi",)
RESERVED_METRIC_KEYS = {"t_mono", "t_wall", "step"}
KIB_PER_MIB = 1024.0


@dataclass
class ProcessInfo:
    key: tuple
    pid: int
    ppid: int
    comm: str
    cmdline: str
    first_seen: float
    role: str = "worker"
    exited_at: Optional[float] = None
    samples: list = field(default_factory=list)


@dataclass
class RunData:
    meta: dict
    header: Optional[dict]
    end: Optional[dict]
    samples: list
    processes: dict
    metrics: list


def load_run_data(run):
    meta = run.read_meta()
    header, end, samples, processes = None, None, [], {}
    current = {}
    for record in read_jsonl(run.resources_path):
        kind = record.get("type")
        if kind == "header":
            header = record
        elif kind == "end":
            end = record
        elif kind == "process":
            key = (record["pid"], record.get("start_ticks", 0))
            processes[key] = ProcessInfo(key, record["pid"], record["ppid"], record["comm"], record["cmdline"],
                                         record["t_mono"])
            current[record["pid"]] = key
        elif kind == "exec" and record["pid"] in current:
            info = processes[current[record["pid"]]]
            info.comm, info.cmdline = record["comm"], record["cmdline"]
        elif kind == "exit" and record["pid"] in current:
            processes[current.pop(record["pid"])].exited_at = record["t_mono"]
        elif kind == "sample":
            samples.append(record)
            for entry in record["procs"]:
                key = current.get(entry["pid"])
                entry["identity"] = key
                if key is not None:
                    processes[key].samples.append({**entry, "t_mono": record["t_mono"], "dt": record["dt"]})
    metrics = sorted(read_jsonl(run.metrics_path), key=lambda record: record.get("t_mono", 0))
    data = RunData(meta, header, end, samples, processes, metrics)
    assign_roles(data)
    return data


def root_pid(data):
    if data.header and data.header.get("root_pid"):
        return data.header["root_pid"]
    return data.meta.get("pid")


def assign_roles(data):
    main_pid = root_pid(data)
    main_assigned = False
    for info in sorted(data.processes.values(), key=lambda item: item.first_seen):
        if info.pid == main_pid and not main_assigned:
            info.role, main_assigned = "main", True
        elif info.comm in HELPER_COMMS or any(marker in info.cmdline for marker in HELPER_CMDLINE_MARKERS):
            info.role = "helper"
        else:
            info.role = "worker"


def weighted_mean(pairs):
    pairs = [(value, weight) for value, weight in pairs if value is not None and weight > 0]
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else None


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


def analysis_window(data):
    steps = [record for record in data.metrics if "t_mono" in record]
    if len(steps) >= 2:
        start, end, source = steps[0]["t_mono"], steps[-1]["t_mono"], "metrics"
        inside = [sample for sample in data.samples if start <= sample["t_mono"] - sample["dt"] / 2 <= end]
        if len(inside) >= rules.MIN_WINDOW_SAMPLES:
            return {"source": source, "start_mono": start, "end_mono": end, "samples": inside}
    samples = data.samples
    if not samples:
        return {"source": "none", "start_mono": None, "end_mono": None, "samples": []}
    return {"source": "all_samples", "start_mono": samples[0]["t_mono"] - samples[0]["dt"],
            "end_mono": samples[-1]["t_mono"], "samples": samples}


def role_of(data, entry):
    info = data.processes.get(entry.get("identity"))
    return info.role if info is not None else "worker"


def per_sample_roles(data, window_samples):
    roles_by_time = []
    for sample in window_samples:
        grouped = {"main": [], "worker": [], "helper": []}
        for entry in sample["procs"]:
            grouped[role_of(data, entry)].append(entry)
        roles_by_time.append((sample, grouped))
    return roles_by_time


def sum_present(entries, key):
    values = [entry.get(key) for entry in entries if entry.get(key) is not None]
    return sum(values) if values else None


def mean_present(entries, key):
    values = [entry.get(key) for entry in entries if entry.get(key) is not None]
    return sum(values) / len(values) if values else None


def gpu_features(window_samples, tree_pids):
    rows = [sample["gpu"] for sample in window_samples if isinstance(sample.get("gpu"), dict)]
    rows = [row for row in rows if row.get("devices")]
    if not rows:
        return None
    used = {proc["device"] for row in rows for proc in row.get("procs", []) if proc.get("pid") in tree_pids}
    utils, mem_utils, mem_used, tree_mem = [], [], [], []
    device_count, memory_total = 0, None
    for row in rows:
        devices = [device for device in row["devices"] if not used or device["index"] in used]
        device_count = max(device_count, len(devices))
        values = [device["util_pct"] for device in devices if device.get("util_pct") is not None]
        if values:
            utils.append(sum(values) / len(values))
        values = [device["mem_util_pct"] for device in devices if device.get("mem_util_pct") is not None]
        if values:
            mem_utils.append(sum(values) / len(values))
        values = [device["mem_used_mb"] for device in devices if device.get("mem_used_mb") is not None]
        if values:
            mem_used.append(sum(values))
        totals = [device["mem_total_mb"] for device in devices if device.get("mem_total_mb") is not None]
        if totals:
            memory_total = sum(totals)
        tree = [proc.get("mem_used_mb") or 0 for proc in row.get("procs", []) if proc.get("pid") in tree_pids]
        if tree:
            tree_mem.append(sum(tree))
    if not utils:
        return None
    return rules.GpuFeatures(
        device_count=device_count,
        util_mean_pct=sum(utils) / len(utils),
        util_p50_pct=statistics.median(utils),
        mem_util_mean_pct=sum(mem_utils) / len(mem_utils) if mem_utils else None,
        memory_used_peak_mb=max(mem_used) if mem_used else None,
        memory_total_mb=memory_total,
        tree_memory_peak_mb=max(tree_mem) if tree_mem else None,
    )


def cgroup_value(samples, key):
    for sample in reversed(samples):
        cgroup = sample.get("cgroup")
        if isinstance(cgroup, dict) and cgroup.get(key) is not None:
            return cgroup[key]
    return None


def usable_cores(data):
    quota = cgroup_value(data.samples, "cpu_quota_cores")
    if quota:
        return float(quota), "cgroup"
    host = data.meta.get("host") or {}
    if host.get("usable_cpus"):
        return float(host["usable_cpus"]), "affinity"
    if data.header and data.header.get("online_cpus"):
        return float(data.header["online_cpus"]), "online_cpus"
    return 1.0, "default"


def memory_limit(data):
    cgroup_max = cgroup_value(data.samples, "memory_max_bytes")
    if cgroup_max:
        return cgroup_max / 1024.0, "cgroup"
    host = data.meta.get("host") or {}
    if host.get("mem_total_kb"):
        return float(host["mem_total_kb"]), "host"
    totals = [sample["system"].get("mem_total_kb") for sample in data.samples if sample.get("system")]
    totals = [total for total in totals if total]
    return (float(totals[-1]), "host") if totals else (None, None)


def data_wait_fraction(metrics):
    timed = [record for record in metrics
             if isinstance(record.get("data_time"), (int, float)) and isinstance(record.get("compute_time"), (int, float))]
    if len(timed) < 2:
        return None
    timed = timed[1:]
    waiting = sum(record["data_time"] for record in timed)
    total = waiting + sum(record["compute_time"] for record in timed)
    return waiting / total if total > 0 else None


def thirds(values):
    if len(values) < 3:
        return None, None
    size = len(values) // 3
    early, late = values[:size], values[-size:]
    return sum(early) / len(early), sum(late) / len(late)


def build_features(data, window):
    samples = window["samples"]
    grouped = per_sample_roles(data, samples)
    cores, limit_source = usable_cores(data)
    limit_kb, memory_source = memory_limit(data)

    main_cpu = weighted_mean((sum_present(groups["main"], "cpu_pct") or 0.0, sample["dt"])
                             for sample, groups in grouped)
    worker_counts = [len(groups["worker"]) for _, groups in grouped]
    worker_cpu = weighted_mean((mean_present(groups["worker"], "cpu_pct"), sample["dt"])
                               for sample, groups in grouped if groups["worker"])
    worker_means = [weighted_mean((entry["cpu_pct"], entry["dt"]) for entry in info.samples
                                  if window["start_mono"] is None or
                                  window["start_mono"] <= entry["t_mono"] - entry["dt"] / 2 <= window["end_mono"])
                    for info in data.processes.values() if info.role == "worker"]
    worker_means = [value for value in worker_means if value is not None]

    tree_majflt = [sum_present(sample["procs"], "majflt_per_s") or 0.0 for sample in samples]
    early_majflt, late_majflt = thirds(tree_majflt)
    available = [sample["system"]["mem_available_kb"] / sample["system"]["mem_total_kb"]
                 for sample in data.samples
                 if sample.get("system", {}).get("mem_available_kb") and sample["system"].get("mem_total_kb")]
    rss_peak = max((sample["tree"]["rss_kb"] for sample in data.samples), default=None)
    tree_pids = {info.pid for info in data.processes.values()}
    oom_kills = cgroup_value(data.samples, "oom_kill")

    features = rules.Features(
        window_samples=len(samples),
        window_seconds=sum(sample["dt"] for sample in samples),
        usable_cores=cores,
        tree_cores_mean=(weighted_mean((sample["tree"]["cpu_pct"], sample["dt"]) for sample in samples) or 0.0) / 100,
        main_cpu_mean_pct=main_cpu or 0.0,
        worker_count=round(statistics.median(worker_counts)) if worker_counts else 0,
        worker_cpu_mean_pct=worker_cpu,
        worker_cpu_min_pct=min(worker_means) if worker_means else None,
        main_read_bytes_per_s=weighted_mean((sum_present(groups["main"], "read_bytes_per_s"), sample["dt"])
                                            for sample, groups in grouped),
        worker_read_bytes_per_s=weighted_mean((sum_present(groups["worker"], "read_bytes_per_s"), sample["dt"])
                                              for sample, groups in grouped if groups["worker"]),
        main_blkio_wait_pct=weighted_mean((mean_present(groups["main"], "blkio_wait_pct"), sample["dt"])
                                          for sample, groups in grouped),
        worker_blkio_wait_pct=weighted_mean((mean_present(groups["worker"], "blkio_wait_pct"), sample["dt"])
                                            for sample, groups in grouped if groups["worker"]),
        system_iowait_pct=weighted_mean((sample.get("system", {}).get("cpu_iowait_pct"), sample["dt"])
                                        for sample in samples),
        rss_peak_kb=rss_peak,
        memory_limit_kb=limit_kb,
        memory_limit_source=memory_source,
        mem_available_min_fraction=min(available) if available else None,
        majflt_early_per_s=early_majflt,
        majflt_late_per_s=late_majflt,
        oom_kills=oom_kills,
        exit_code=data.meta.get("exit_code"),
        data_wait_fraction=data_wait_fraction(data.metrics),
        gpu=gpu_features(samples, tree_pids),
    )
    return features, {"usable_cores_source": limit_source}


def step_stats(metrics):
    stepped = [record for record in metrics if isinstance(record.get("step"), (int, float))]
    durations = []
    for previous, current in zip(stepped, stepped[1:]):
        delta_steps = current["step"] - previous["step"]
        delta_time = current["t_mono"] - previous["t_mono"]
        if delta_steps > 0 and delta_time >= 0:
            durations.append(delta_time / delta_steps)
    if not durations:
        return {"steps": len(stepped), "step_time_mean_s": None, "step_time_median_s": None,
                "step_time_p90_s": None, "steps_per_sec": None, "steps_in_window": 0}
    steps_in_window = stepped[-1]["step"] - stepped[0]["step"]
    span = stepped[-1]["t_mono"] - stepped[0]["t_mono"]
    return {
        "steps": len(stepped),
        "steps_in_window": steps_in_window,
        "step_time_mean_s": span / steps_in_window if steps_in_window else None,
        "step_time_median_s": statistics.median(durations),
        "step_time_p90_s": percentile(durations, 0.9),
        "steps_per_sec": steps_in_window / span if span > 0 else None,
    }


def final_metrics(metrics):
    final = {}
    for record in metrics:
        for key, value in record.items():
            if key not in RESERVED_METRIC_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                final[key] = value
    return final


def last_value(entries, key):
    for entry in reversed(entries):
        if entry.get(key) is not None:
            return entry[key]
    return None


def process_rows(data, window):
    rows = []
    for info in sorted(data.processes.values(), key=lambda item: (item.first_seen, item.pid)):
        in_window = [entry for entry in info.samples if window["start_mono"] is None or
                     window["start_mono"] <= entry["t_mono"] - entry["dt"] / 2 <= window["end_mono"]]
        all_samples = info.samples
        last_seen = all_samples[-1]["t_mono"] if all_samples else info.first_seen
        rows.append({
            "pid": info.pid,
            "ppid": info.ppid,
            "role": info.role,
            "comm": info.comm,
            "cmdline": info.cmdline,
            "first_seen_mono": info.first_seen,
            "last_seen_mono": last_seen,
            "lifetime_s": round((info.exited_at or last_seen) - info.first_seen, 3),
            "exited": info.exited_at is not None,
            "cpu_seconds": last_value(all_samples, "cpu_s") or 0.0,
            "cpu_mean_pct": weighted_mean((entry["cpu_pct"], entry["dt"]) for entry in in_window),
            "cpu_p95_pct": percentile([entry["cpu_pct"] for entry in in_window], 0.95),
            "peak_rss_mb": max((entry["rss_kb"] for entry in all_samples), default=0) / KIB_PER_MIB,
            "read_bytes": last_value(all_samples, "read_bytes"),
            "write_bytes": last_value(all_samples, "write_bytes"),
            "read_bytes_per_s_mean": weighted_mean((entry.get("read_bytes_per_s"), entry["dt"]) for entry in in_window),
            "majflt_total": sum((entry.get("majflt_per_s") or 0) * entry["dt"] for entry in all_samples),
            "blkio_wait_mean_pct": weighted_mean((entry.get("blkio_wait_pct"), entry["dt"]) for entry in in_window),
            "threads_max": max((entry["threads"] for entry in all_samples), default=0),
            "window_samples": len(in_window),
        })
    return rows


def role_summary(rows, features):
    workers = [row for row in rows if row["role"] == "worker" and row["cpu_mean_pct"] is not None]
    main = next((row for row in rows if row["role"] == "main"), None)
    return {
        "main": {"pid": main["pid"], "cpu_mean_pct": main["cpu_mean_pct"], "cpu_seconds": main["cpu_seconds"],
                 "peak_rss_mb": main["peak_rss_mb"], "threads_max": main["threads_max"]} if main else None,
        "workers": {
            "count_median": features.worker_count,
            "count_total": sum(1 for row in rows if row["role"] == "worker"),
            "cpu_mean_pct": features.worker_cpu_mean_pct,
            "cpu_min_pct": min((row["cpu_mean_pct"] for row in workers), default=None),
            "cpu_max_pct": max((row["cpu_mean_pct"] for row in workers), default=None),
            "cpu_seconds": sum(row["cpu_seconds"] for row in rows if row["role"] == "worker"),
        },
        "helpers": {"count": sum(1 for row in rows if row["role"] == "helper"),
                    "cpu_seconds": sum(row["cpu_seconds"] for row in rows if row["role"] == "helper")},
    }


def unavailable_fields(samples):
    names = set()
    for sample in samples:
        for entry in sample["procs"]:
            names.update(entry.get("unavailable", []))
    return sorted(names)


def missing_reason(data):
    agent = data.meta.get("agent") or {}
    if agent.get("available") is False:
        return f"The resource agent was not available for this run ({agent.get('error', 'unknown reason')})."
    if not data.samples:
        return "The resource file contains no samples (the run may have ended before the first sample)."
    return None


def totals(data, window, features, steps, rows):
    samples = data.samples
    span = sum(sample["dt"] for sample in samples)
    cpu_seconds = sum(row["cpu_seconds"] for row in rows)
    window_cpu = sum(sample["tree"]["cpu_pct"] * sample["dt"] / 100 for sample in window["samples"])
    reads = [row["read_bytes"] for row in rows if row["read_bytes"] is not None]
    writes = [row["write_bytes"] for row in rows if row["write_bytes"] is not None]
    return {
        "duration_s": data.meta.get("duration_s"),
        "sampled_s": span,
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_window": window_cpu,
        "avg_cores": cpu_seconds / span if span else None,
        "avg_cores_window": features.tree_cores_mean,
        "peak_cores": max((sample["tree"]["cpu_pct"] for sample in samples), default=0) / 100,
        "peak_rss_mb": (features.rss_peak_kb or 0) / KIB_PER_MIB,
        "read_bytes": sum(reads) if reads else None,
        "write_bytes": sum(writes) if writes else None,
        **steps,
        "cpu_seconds_per_step": window_cpu / steps["steps_in_window"] if steps["steps_in_window"] else None,
        "final_metrics": final_metrics(data.metrics),
    }


def analyze_run(run):
    data = load_run_data(run)
    window = analysis_window(data)
    features, feature_notes = build_features(data, window)
    rows = process_rows(data, window)
    steps = step_stats(data.metrics)
    decision = rules.decide(features, missing_reason(data))
    notes = []
    if window["source"] == "all_samples" and data.metrics:
        notes.append("The training loop was too short to isolate; the whole run, including startup, was analyzed.")
    if features.gpu is None:
        notes.append("No GPU data: verdict rules used CPU-only signals.")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_id": run.run_id,
        "name": data.meta.get("name"),
        "status": data.meta.get("status"),
        "exit_code": data.meta.get("exit_code"),
        "config_hash": data.meta.get("config_hash"),
        "generated_at": time.time(),
        "window": {"source": window["source"], "start_mono": window["start_mono"], "end_mono": window["end_mono"],
                   "seconds": features.window_seconds, "samples": features.window_samples,
                   "warmup_excluded_s": (window["start_mono"] - data.meta["start_mono"])
                   if window["start_mono"] is not None and data.meta.get("start_mono") is not None else None},
        "totals": totals(data, window, features, steps, rows),
        "roles": role_summary(rows, features),
        "processes": rows,
        "gpu": asdict(features.gpu) if features.gpu else None,
        "features": {**asdict(features), **feature_notes},
        "verdict": decision,
        "data_quality": {
            "agent_available": (data.meta.get("agent") or {}).get("available", bool(data.samples)),
            "agent_end_reason": data.end.get("reason") if data.end else None,
            "samples": len(data.samples),
            "delay_accounting": data.header.get("delayacct") if data.header else None,
            "unavailable": unavailable_fields(data.samples),
            "notes": notes,
        },
    }
    write_json_atomic(run.summary_path, summary)
    return summary


def fmt_seconds(value):
    return "-" if value is None else f"{value * 1000:.1f} ms" if value < 1 else f"{value:.2f} s"


def fmt_pct(value):
    return "-" if value is None else f"{value:.0f}%"


def format_report(summary):
    verdict = summary["verdict"]
    totals_ = summary["totals"]
    roles = summary["roles"]
    lines = [
        f"Run {summary['run_id']} ({summary['name']}) - {summary['status']}, exit {summary['exit_code']}",
        "",
        f"VERDICT: {verdict['title']} [{verdict['severity']}]",
    ]
    lines += [f"  - {line}" for line in verdict["evidence"]]
    if verdict["suggestions"]:
        lines.append("  Suggestions:")
        lines += [f"    * {line}" for line in verdict["suggestions"]]
    for finding in verdict["secondary"]:
        lines.append(f"Also: {finding['title']} [{finding['severity']}]")
        lines += [f"  - {line}" for line in finding["evidence"]]
    lines += [
        "",
        f"Window: {summary['window']['source']}, {summary['window']['seconds']:.1f}s, "
        f"{summary['window']['samples']} samples",
        f"CPU: {totals_['cpu_seconds']:.1f} CPU-s total, avg {totals_['avg_cores_window']:.2f} cores in window, "
        f"peak {totals_['peak_cores']:.2f} cores; peak RSS {totals_['peak_rss_mb']:.0f} MiB",
        f"Steps: {totals_['steps']} logged, median step {fmt_seconds(totals_['step_time_median_s'])}, "
        f"p90 {fmt_seconds(totals_['step_time_p90_s'])}, "
        f"{totals_['steps_per_sec'] or 0:.2f} steps/s, "
        f"{totals_['cpu_seconds_per_step'] or 0:.3f} CPU-s/step",
    ]
    if roles["main"]:
        lines.append(f"Main process {roles['main']['pid']}: {fmt_pct(roles['main']['cpu_mean_pct'])} of a core")
    workers = roles["workers"]
    if workers["count_total"]:
        lines.append(f"Workers: ~{workers['count_median']} concurrent ({workers['count_total']} total), "
                     f"{fmt_pct(workers['cpu_mean_pct'])} each on average "
                     f"(min {fmt_pct(workers['cpu_min_pct'])}, max {fmt_pct(workers['cpu_max_pct'])})")
    if summary["gpu"]:
        gpu = summary["gpu"]
        lines.append(f"GPU: {fmt_pct(gpu['util_mean_pct'])} mean utilization over {gpu['device_count']} device(s)")
    for note in summary["data_quality"]["notes"]:
        lines.append(f"Note: {note}")
    return "\n".join(lines)
