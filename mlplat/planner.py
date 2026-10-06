import json
import time
from pathlib import Path

from mlplat.analyzer import analyze_run
from mlplat.launcher import launch_run
from mlplat.predict import DEFAULT_PREFERENCE, PlanError, build_plan, extract_profile, format_cost, format_hours
from mlplat.run_store import read_jsonl, write_json_atomic
from mlplat.targets import load_targets, local_target

PLAN_FILE = "plan.json"
DEFAULT_WARMUP_STEPS = 20
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_WINDOW_STEPS = 200
DEFAULT_STARTUP_TIMEOUT_S = 300.0


class ProfileWindow:
    def __init__(self, metrics_path, warmup_steps=DEFAULT_WARMUP_STEPS, window_steps=DEFAULT_WINDOW_STEPS,
                 window_seconds=DEFAULT_WINDOW_SECONDS, startup_timeout_s=DEFAULT_STARTUP_TIMEOUT_S, clock=time.monotonic):
        self.metrics_path = Path(metrics_path)
        self.warmup_steps = warmup_steps
        self.window_steps = window_steps
        self.window_seconds = window_seconds
        self.startup_timeout_s = startup_timeout_s
        self.clock = clock
        self.started = clock()
        self.offset = 0
        self.first_step = None
        self.warm = None
        self.last_step = None

    def read_new(self):
        if not self.metrics_path.exists():
            return []
        with open(self.metrics_path, encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        complete = chunk[:chunk.rfind("\n") + 1]
        self.offset += len(complete.encode("utf-8"))
        records = []
        for line in complete.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def __call__(self):
        for record in self.read_new():
            step = record.get("step")
            if not isinstance(step, (int, float)):
                continue
            if self.first_step is None:
                self.first_step = step
            self.last_step = step
            if self.warm is None and step >= self.first_step + self.warmup_steps:
                self.warm = (step, record.get("t_mono", self.clock()))
        now = self.clock()
        if self.warm is None:
            if self.first_step is None and now - self.started > self.startup_timeout_s:
                return f"no steps were logged within {self.startup_timeout_s:.0f}s"
            return None
        warm_step, warm_time = self.warm
        if self.last_step - warm_step >= self.window_steps:
            return f"profiled {self.window_steps} steps after warm-up"
        if now - warm_time >= self.window_seconds:
            return f"profiled {self.window_seconds:.0f}s after warm-up"
        return None


def plan_path(run):
    return run.path / PLAN_FILE


def read_plan(run):
    path = plan_path(run)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def plan_for_run(run, total_steps, targets_path=None, summary=None, prefer=DEFAULT_PREFERENCE):
    meta = run.read_meta()
    header = next((record for record in read_jsonl(run.resources_path) if record.get("type") == "header"), None)
    targets = [local_target(meta, header)] + load_targets(targets_path)
    profile = extract_profile(run, summary)
    plan = build_plan(profile, targets, total_steps, prefer)
    plan["created_at"] = time.time()
    plan["targets_file"] = str(targets_path) if targets_path else None
    write_json_atomic(plan_path(run), plan)
    return plan


def profile_and_plan(name, command, total_steps, targets_path=None, warmup_steps=DEFAULT_WARMUP_STEPS,
                     window_steps=DEFAULT_WINDOW_STEPS, window_seconds=DEFAULT_WINDOW_SECONDS,
                     prefer=DEFAULT_PREFERENCE, **launch_options):
    windows = {}

    def stop_when(run):
        if run.run_id not in windows:
            windows[run.run_id] = ProfileWindow(run.metrics_path, warmup_steps, window_steps, window_seconds)
        return windows[run.run_id]()

    extra_meta = {"analysis": {"warmup_steps": warmup_steps},
                  "plan_request": {"total_steps": total_steps, "warmup_steps": warmup_steps,
                                   "window_steps": window_steps, "window_seconds": window_seconds}}
    run, meta = launch_run(name, command, stop_when=stop_when, extra_meta=extra_meta, **launch_options)
    if meta["status"] not in ("profiled", "completed"):
        raise PlanError(f"the profiling run ended with status {meta['status']} (exit {meta['training_exit_code']}); "
                        f"see {run.output_path}")
    summary = run.read_summary() or analyze_run(run)
    return run, plan_for_run(run, total_steps, targets_path, summary, prefer)


def yes_no(value):
    return "unknown" if value is None else ("yes" if value else "NO")


def format_plan(plan):
    targets = {target["id"]: target for target in plan["targets"]}
    recommendation = plan["recommendation"]
    profile = plan["profile"]
    scenario = recommendation["scenario"]
    lines = [
        f"Plan for {plan['total_steps']:,} steps, profiled from run {plan['run_id']} ({profile['name']})",
        "",
        recommendation["headline"],
    ]
    if recommendation["best_target_id"]:
        lines.append(f"Best choice: {recommendation['best_reason']}")
    fix = recommendation["fix_first"]
    if fix:
        lines += ["", f"  Fix it:        {format_hours(fix['hours_now'])} -> {format_hours(fix['hours_fixed'])} "
                      f"({fix['gain'] * 100:.0f}% faster, free)"]
        if fix["upgrade"]:
            lines.append(f"  Faster GPU:    {format_hours(fix['hours_now'])} -> "
                         f"{format_hours(fix['upgrade']['total_hours'])} ({fix['upgrade']['gain'] * 100:.0f}% faster, "
                         f"{format_cost(fix['upgrade']['cost'])}) on {fix['upgrade']['name']}")
    lines += [
        "",
        f"Measured step (mean): {profile['step_time_s'] * 1000:.1f} ms = compute {profile['compute_s'] * 1000:.1f} ms + "
        f"waiting for data {profile['data_wait_s'] * 1000:.1f} ms (split from {profile['split_method']}); "
        f"{profile['workers']} worker(s), data prep {((profile['prep_core_s'] or 0) * 1000):.1f} CPU-ms per batch; "
        f"overhead {profile['overhead_s'] * 1000:.1f} ms; startup {profile['startup_s']:.1f} s",
        "",
        f"{'Target':<34}{'Scenario':<10}{'Workers':>8}{'Step':>10}{'Total':>12}{'Cost':>10}{'Sessions':>10}"
        f"{'Memory':>8}{'Limiting':>10}",
    ]
    order = recommendation["ranking"] or [entry["target_id"] for entry in plan["predictions"]]
    predictions = {entry["target_id"]: entry for entry in plan["predictions"]}
    for target_id in order:
        for name in ("as_is", "fixed"):
            prediction = predictions[target_id][name]
            if name == "fixed" and prediction["workers"] == predictions[target_id]["as_is"]["workers"]:
                continue
            marker = "*" if target_id == recommendation["best_target_id"] and name == scenario else " "
            fits = prediction["fits_memory"] is not False and prediction["fits_gpu_memory"] is not False
            label = targets[target_id]["name"] + (" (placeholder)" if targets[target_id]["placeholders"] else "")
            lines.append(f"{marker}{label[:33]:<33}{name.replace('_', '-'):<10}{prediction['workers']:>8}"
                         f"{prediction['step_time_s'] * 1000:>8.1f}ms{format_hours(prediction['total_hours']):>12}"
                         f"{format_cost(prediction['cost']):>10}{prediction['sessions']:>10}"
                         f"{'ok' if fits else 'NO':>8}{prediction['bottleneck']:>10}")
    for note in profile["notes"]:
        lines.append(f"Note: {note}")
    if plan["placeholders_used"]:
        lines.append(f"Note: {len(plan['placeholders_used'])} catalog values are placeholders; edit targets.yaml with "
                     f"current values before trusting these estimates.")
    lines.append("All times and costs are estimates.")
    return "\n".join(lines)
