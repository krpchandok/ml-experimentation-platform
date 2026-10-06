import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Optional

from mlplat.run_store import read_jsonl
from mlplat.targets import LOCAL_TARGET_ID

PLAN_SCHEMA = 1
MIN_TIMED_STEPS = 3
MIN_COMPUTE_FRACTION = 0.02
MAX_OVERHEAD_FRACTION = 0.25
MEMORY_HEADROOM = 0.90
TIME_TOLERANCE = 1.25
TIME_SLACK_HOURS = 0.5
PREFERENCES = ("balanced", "time", "cost")
DEFAULT_PREFERENCE = "balanced"
FIX_MIN_GAIN = 0.15
SECONDS_PER_HOUR = 3600.0
MIB_PER_GIB = 1024.0

ASSUMPTIONS = (
    "A step's compute part (everything except waiting for data) speeds up in proportion to the target's relative "
    "GPU speed, including Python and kernel-launch overhead.",
    "Data preparation costs a fixed number of CPU-seconds per batch and spreads evenly across data-loader workers, "
    "up to the target's cores minus the cores the main process uses.",
    "With one or more workers, data preparation overlaps with compute, so a step takes the longer of the two plus a "
    "fixed overhead. With zero workers the main process prepares each batch itself, so the two add up.",
    "The fixed overhead is calibrated so the model reproduces the measured step time on the profiled machine.",
    "Every session repeats the measured startup time (imports, data setup, worker start, warm-up steps).",
    "Host memory needed is the main process's peak plus each worker's peak; this overstates memory that forked "
    "workers share.",
)


class PlanError(ValueError):
    pass


@dataclass
class Profile:
    run_id: str
    name: str
    step_time_s: float
    step_time_median_s: Optional[float]
    compute_s: float
    data_wait_s: float
    prep_core_s: Optional[float]
    workers: int
    main_cores: float
    split_method: str
    startup_s: float
    steps_measured: int
    host_memory_main_gb: Optional[float]
    host_memory_per_worker_gb: Optional[float]
    gpu_memory_gb: Optional[float]
    gpu_name: Optional[str]
    gpu_util_pct: Optional[float]
    verdict: Optional[str]
    overhead_s: float = 0.0
    model_self_check_s: Optional[float] = None
    notes: list = field(default_factory=list)


def window_metrics(metrics, summary):
    window = summary.get("window") or {}
    start, end = window.get("start_mono"), window.get("end_mono")
    if start is None or end is None or window.get("source") != "metrics":
        return metrics
    return [record for record in metrics if start <= record.get("t_mono", -1) <= end]


def device_memory_used_gb(resources):
    used = [device["mem_used_mb"] for record in resources if record.get("type") == "sample"
            and isinstance(record.get("gpu"), dict) for device in record["gpu"].get("devices", [])
            if device.get("mem_used_mb") is not None]
    if len(used) < 2:
        return None
    grown = max(used) - min(used)
    return grown / MIB_PER_GIB if grown > 0 else None


def split_step(summary, metrics):
    totals = summary["totals"]
    step = totals.get("step_time_mean_s") or totals.get("step_time_median_s")
    if not step:
        raise PlanError("the profiling run logged fewer than two steps; call mlplat.log(step=...) every step")
    data_times = [record["data_time"] for record in metrics if isinstance(record.get("data_time"), (int, float))]
    gpu = summary.get("gpu")
    if len(data_times) >= MIN_TIMED_STEPS:
        wait = statistics.fmean(data_times[1:])
        method = "logged data_time"
    elif gpu and gpu.get("util_mean_pct") is not None:
        wait = step * (1 - gpu["util_mean_pct"] / 100.0)
        method = "GPU utilization"
    else:
        wait = 0.0
        method = "assumed compute-bound (no data_time and no GPU data)"
    compute = max(step - wait, step * MIN_COMPUTE_FRACTION)
    return step, compute, min(wait, step), method


def prep_core_seconds(summary, workers, wait, method):
    workers_role = (summary.get("roles") or {}).get("workers") or {}
    mean_step = summary["totals"].get("step_time_mean_s") or summary["totals"].get("step_time_median_s")
    if workers > 0 and workers_role.get("cpu_mean_pct") is not None:
        return workers_role["cpu_mean_pct"] / 100.0 * workers * mean_step
    if workers == 0 and not method.startswith("assumed"):
        return wait
    return None


def to_gb(mib):
    return mib / MIB_PER_GIB if mib is not None else None


def memory_profile(summary):
    rows = summary.get("processes") or []
    main = next((row["peak_rss_mb"] for row in rows if row["role"] == "main"), None)
    workers = [row["peak_rss_mb"] for row in rows if row["role"] == "worker" and row.get("peak_rss_mb")]
    return to_gb(main), to_gb(statistics.median(workers)) if workers else None


def extract_profile(run, summary=None):
    summary = summary or run.read_summary()
    if summary is None:
        raise PlanError(f"run {run.run_id} has no summary.json; run mlplat analyze first")
    meta = run.read_meta()
    resources = read_jsonl(run.resources_path)
    metrics = window_metrics(sorted(read_jsonl(run.metrics_path), key=lambda r: r.get("t_mono", 0)), summary)
    step, compute, wait, method = split_step(summary, metrics)
    workers = int(round((summary.get("roles") or {}).get("workers", {}).get("count_median") or 0))
    main_cpu = ((summary.get("roles") or {}).get("main") or {}).get("cpu_mean_pct") or 100.0
    main_memory, worker_memory = memory_profile(summary)
    gpu = summary.get("gpu") or {}
    tree_gpu = gpu.get("tree_memory_peak_mb")
    header = next((record for record in resources if record.get("type") == "header"), {})
    devices = (header.get("gpu") or {}).get("devices") or []
    window = summary.get("window") or {}
    startup = window.get("warmup_excluded_s")
    profile = Profile(
        run_id=run.run_id,
        name=meta.get("name", run.run_id),
        step_time_s=step,
        step_time_median_s=(summary.get("totals") or {}).get("step_time_median_s"),
        compute_s=compute,
        data_wait_s=wait,
        prep_core_s=prep_core_seconds(summary, workers, wait, method),
        workers=workers,
        main_cores=max(1.0, round(main_cpu / 100.0)),
        split_method=method,
        startup_s=float(startup) if startup and startup > 0 else 0.0,
        steps_measured=int((summary.get("totals") or {}).get("steps_in_window") or 0),
        host_memory_main_gb=main_memory,
        host_memory_per_worker_gb=worker_memory,
        gpu_memory_gb=tree_gpu / MIB_PER_GIB if tree_gpu else device_memory_used_gb(resources),
        gpu_name=devices[0]["name"] if devices else None,
        gpu_util_pct=gpu.get("util_mean_pct"),
        verdict=(summary.get("verdict") or {}).get("primary"),
    )
    if profile.prep_core_s is None:
        profile.notes.append("Data-prep cost per batch could not be measured, so worker changes are not modelled.")
    if method.startswith("assumed"):
        profile.notes.append("Log data_time from the training loop for a measured compute/data split.")
    if summary.get("window", {}).get("source") != "metrics":
        profile.notes.append("The profile window could not exclude startup; step times may include warm-up.")
    return profile


def worker_cores(profile, target):
    return max(1.0, target.cpu_cores - profile.main_cores)


def model_step(profile, target, workers):
    compute = profile.compute_s / target.relative_gpu_speed
    if profile.prep_core_s is None:
        return compute, 0.0, compute
    prep_total = profile.prep_core_s / target.relative_cpu_speed
    if workers <= 0:
        return compute, prep_total, compute + prep_total
    prep = prep_total / min(workers, worker_cores(profile, target))
    return compute, prep, max(compute, prep)


def calibrate(profile, local):
    _, _, modelled = model_step(profile, local, profile.workers)
    profile.model_self_check_s = modelled
    overhead = profile.step_time_s - modelled
    profile.overhead_s = min(max(overhead, 0.0), profile.step_time_s * MAX_OVERHEAD_FRACTION)
    if modelled > profile.step_time_s * 1.05:
        profile.notes.append(f"The model over-predicts this machine's step time by "
                             f"{(modelled / profile.step_time_s - 1) * 100:.0f}% before calibration.")
    return profile


@dataclass
class Prediction:
    target_id: str
    workers: int
    step_time_s: float
    compute_s: float
    prep_s: float
    bottleneck: str
    oven_busy: float
    training_hours: float
    total_hours: float
    sessions: int
    fits_one_session: bool
    weeks: Optional[int]
    fits_weekly: Optional[bool]
    cost: float
    host_memory_gb: Optional[float]
    fits_memory: Optional[bool]
    fits_gpu_memory: Optional[bool]
    feasible: bool
    uses_placeholders: bool


def predict(profile, target, workers, total_steps):
    compute, prep, core_step = model_step(profile, target, workers)
    step = core_step + profile.overhead_s
    training_s = step * total_steps
    if target.session_hours:
        usable = target.session_hours * SECONDS_PER_HOUR - profile.startup_s
        sessions = max(1, math.ceil(training_s / usable)) if usable > 0 else 0
    else:
        sessions = 1
    total_s = training_s + max(sessions, 1) * profile.startup_s
    total_h = total_s / SECONDS_PER_HOUR
    weeks = math.ceil(total_h / target.weekly_hours) if target.weekly_hours else None
    memory = None
    if profile.host_memory_main_gb is not None:
        memory = profile.host_memory_main_gb + workers * (profile.host_memory_per_worker_gb or 0.0)
    fits_memory = None if memory is None or target.memory_gb is None else memory <= target.memory_gb * MEMORY_HEADROOM
    fits_gpu = (None if profile.gpu_memory_gb is None or target.gpu_memory_gb is None
                else profile.gpu_memory_gb <= target.gpu_memory_gb * MEMORY_HEADROOM)
    return Prediction(
        target_id=target.id,
        workers=workers,
        step_time_s=step,
        compute_s=compute,
        prep_s=prep,
        bottleneck="bakers" if prep > compute else "oven",
        oven_busy=min(1.0, compute / step) if step > 0 else 0.0,
        training_hours=training_s / SECONDS_PER_HOUR,
        total_hours=total_h,
        sessions=sessions,
        fits_one_session=sessions == 1,
        weeks=weeks,
        fits_weekly=None if not target.weekly_hours else total_h <= target.weekly_hours,
        cost=total_h * target.cost_per_hour,
        host_memory_gb=memory,
        fits_memory=fits_memory,
        fits_gpu_memory=fits_gpu,
        feasible=fits_memory is not False and fits_gpu is not False and sessions > 0,
        uses_placeholders=bool(target.placeholders),
    )


def fixed_workers(profile, target):
    if profile.prep_core_s is None:
        return profile.workers
    capacity = max(1, int(worker_cores(profile, target)))
    compute = profile.compute_s / target.relative_gpu_speed
    prep = profile.prep_core_s / target.relative_cpu_speed
    needed = max(1, math.ceil(prep / compute)) if compute > 0 else capacity
    return max(profile.workers, min(needed, capacity))


def format_hours(hours):
    seconds = hours * SECONDS_PER_HOUR
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    whole = int(minutes // 60)
    return f"{whole} h {int(minutes - whole * 60):02d} min"


def format_cost(cost):
    if cost == 0:
        return "free"
    return "under $0.01" if cost < 0.01 else f"${cost:,.2f}"


def gain(before, after):
    return (before - after) / before if before > 0 else 0.0


def ranking_key(prediction, fastest, prefer):
    blocked = not prediction.feasible
    if prefer == "time":
        return blocked, prediction.total_hours, prediction.cost
    if prefer == "cost":
        return blocked, round(prediction.cost, 2), not prediction.fits_one_session, prediction.total_hours
    close_enough = (prediction.total_hours <= fastest * TIME_TOLERANCE
                    or prediction.total_hours - fastest <= TIME_SLACK_HOURS)
    return (blocked, not close_enough, not prediction.fits_one_session, prediction.fits_weekly is False,
            prediction.cost > 0, prediction.cost, prediction.total_hours)


def pick_best(entries, scenario, prefer=DEFAULT_PREFERENCE):
    if prefer not in PREFERENCES:
        raise PlanError(f"--prefer must be one of {', '.join(PREFERENCES)}")
    feasible = [entry for entry in entries if entry[scenario].feasible]
    if not feasible:
        return None, []
    fastest = min(entry[scenario].total_hours for entry in feasible)
    ranked = sorted(entries, key=lambda entry: ranking_key(entry[scenario], fastest, prefer))
    return ranked[0], ranked


def best_reason(best, ranked, scenario, targets):
    prediction = best[scenario]
    target = targets[best["target_id"]]
    parts = [f"{target.name} finishes in about {format_hours(prediction.total_hours)} "
             f"({format_cost(prediction.cost)})"]
    feasible = [entry for entry in ranked if entry[scenario].feasible]
    fastest = min(feasible, key=lambda entry: entry[scenario].total_hours)
    if fastest is not best:
        faster = fastest[scenario]
        parts.append(f"{targets[fastest['target_id']].name} would take {format_hours(faster.total_hours)} "
                     f"for {format_cost(faster.cost)}, which is not worth it")
    if not prediction.fits_one_session:
        parts.append(f"it needs {prediction.sessions} sessions, so save checkpoints")
    if target.placeholders:
        parts.append("this target still uses placeholder values in targets.yaml")
    return "; ".join(parts) + "."


def fix_first(entries, profile, targets):
    local = next(entry for entry in entries if entry["target_id"] == LOCAL_TARGET_ID)
    now, fixed = local["as_is"], local["fixed"]
    if now.bottleneck != "bakers" or gain(now.total_hours, fixed.total_hours) < FIX_MIN_GAIN:
        return None
    others = [entry for entry in entries if entry["target_id"] != LOCAL_TARGET_ID and entry["as_is"].feasible]
    upgrade = None
    faster_gpus = [entry for entry in others if targets[entry["target_id"]].relative_gpu_speed > 1.0] or others
    if faster_gpus:
        best = min(faster_gpus, key=lambda entry: entry["as_is"].total_hours)
        upgrade = {"target_id": best["target_id"], "name": targets[best["target_id"]].name,
                   "total_hours": best["as_is"].total_hours, "cost": best["as_is"].cost,
                   "gain": gain(now.total_hours, best["as_is"].total_hours)}
    still_bound = fixed.bottleneck == "bakers"
    return {
        "current_workers": profile.workers,
        "suggested_workers": fixed.workers,
        "hours_now": now.total_hours,
        "hours_fixed": fixed.total_hours,
        "gain": gain(now.total_hours, fixed.total_hours),
        "still_data_bound": still_bound,
        "upgrade": upgrade,
    }


def headline(fix, best, scenario, targets):
    if fix:
        message = (f"Fix the data pipeline first: going from {fix['current_workers']} to {fix['suggested_workers']} "
                   f"data-loader workers cuts training on this machine from {format_hours(fix['hours_now'])} to "
                   f"{format_hours(fix['hours_fixed'])}.")
        if fix["upgrade"]:
            message += (f" Moving to {fix['upgrade']['name']} without the fix only gets you to "
                        f"{format_hours(fix['upgrade']['total_hours'])} ({format_cost(fix['upgrade']['cost'])}).")
        if fix["still_data_bound"]:
            message += " Even then the GPU waits for data; cache or move preprocessing to the GPU next."
        return message
    if best is None:
        return "No target has enough memory for this run."
    return f"Train on {targets[best['target_id']].name}."


def build_plan(profile, targets, total_steps, prefer=DEFAULT_PREFERENCE):
    if total_steps <= 0:
        raise PlanError("--total-steps must be positive")
    by_id = {target.id: target for target in targets}
    calibrate(profile, by_id[LOCAL_TARGET_ID])
    entries = []
    for target in targets:
        entries.append({
            "target_id": target.id,
            "as_is": predict(profile, target, profile.workers, total_steps),
            "fixed": predict(profile, target, fixed_workers(profile, target), total_steps),
        })
    fix = fix_first(entries, profile, by_id)
    scenario = "fixed" if fix else "as_is"
    best, ranked = pick_best(entries, scenario, prefer)
    placeholders = sorted({f"{target.id}.{name}" for target in targets for name in target.placeholders})
    return {
        "schema": PLAN_SCHEMA,
        "run_id": profile.run_id,
        "total_steps": total_steps,
        "profile": asdict(profile),
        "targets": [target.to_dict() for target in targets],
        "predictions": [{"target_id": entry["target_id"], "as_is": asdict(entry["as_is"]),
                         "fixed": asdict(entry["fixed"])} for entry in entries],
        "recommendation": {
            "prefer": prefer,
            "scenario": scenario,
            "headline": headline(fix, best, scenario, by_id),
            "best_target_id": best["target_id"] if best else None,
            "best_reason": best_reason(best, ranked, scenario, by_id) if best else None,
            "ranking": [entry["target_id"] for entry in ranked],
            "fix_first": fix,
        },
        "assumptions": list(ASSUMPTIONS),
        "placeholders_used": placeholders,
    }
