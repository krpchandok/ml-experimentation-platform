import json
import math
import subprocess
import sys
import textwrap

import pytest

from mlplat import predict
from mlplat.analyzer import analyze_run
from mlplat.planner import ProfileWindow, plan_for_run, read_plan
from mlplat.predict import PlanError, Profile, build_plan, extract_profile, fixed_workers, model_step
from mlplat.targets import LOCAL_TARGET_ID, Target, TargetError, load_targets, local_target, parse_target
from mlplat.run_store import RunStore
from synthetic import MAIN_PID, SyntheticRun, gpu_row

REPO_TARGETS = predict.__file__.replace("mlplat/predict.py", "targets.yaml")


def make_profile(**overrides):
    values = dict(run_id="r", name="n", step_time_s=0.040, step_time_median_s=0.040, compute_s=0.010, data_wait_s=0.030, prep_core_s=0.040,
                  workers=1, main_cores=1.0, split_method="logged data_time", startup_s=10.0, steps_measured=200,
                  host_memory_main_gb=2.0, host_memory_per_worker_gb=0.5, gpu_memory_gb=3.0, gpu_name="GPU",
                  gpu_util_pct=25.0, verdict="preprocessing_bound")
    values.update(overrides)
    return Profile(**values)


def make_target(target_id, **overrides):
    values = dict(id=target_id, name=target_id.title(), cpu_cores=8, memory_gb=16, gpu_memory_gb=8,
                  relative_gpu_speed=1.0, cost_per_hour=0.0)
    values.update(overrides)
    return Target(**values)


def local(**overrides):
    return make_target(LOCAL_TARGET_ID, measured=True, runnable_here=True, bakery="home_kitchen", **overrides)


def test_repository_catalog_marks_every_value_as_placeholder():
    targets = load_targets(REPO_TARGETS)
    assert {target.id for target in targets} == {"colab-free", "kaggle", "cloud-gpu"}
    for target in targets:
        assert set(target.placeholders) >= {"relative_gpu_speed", "cpu_cores", "cost_per_hour"}
        assert all(target.sources.get(name) for name in target.placeholders)
    assert {target.bakery for target in targets} == {"community_kitchen", "market_stall", "rented_bakery"}


def test_catalog_accepts_plain_values_and_rejects_bad_entries(tmp_path):
    target = parse_target({"id": "x", "name": "X", "relative_gpu_speed": 1.5, "cpu_cores": 4, "cost_per_hour": 0.4,
                           "session_hours": {"value": 6, "placeholder": True, "source": "docs"}})
    assert target.relative_gpu_speed == 1.5 and target.placeholders == ["session_hours"]
    assert target.relative_cpu_speed == 1.0
    bad = [
        ({"name": "no id"}, "needs an 'id'"),
        ({"id": "a", "name": "A", "cpu_cores": 2, "cost_per_hour": 0}, "missing required field relative_gpu_speed"),
        ({"id": "a", "name": "A", "relative_gpu_speed": 0, "cpu_cores": 2, "cost_per_hour": 0}, "must be positive"),
        ({"id": "a", "name": "A", "relative_gpu_speed": "fast", "cpu_cores": 2, "cost_per_hour": 0}, "number"),
        ({"id": "a", "name": "A", "bakery": "castle", "relative_gpu_speed": 1, "cpu_cores": 2, "cost_per_hour": 0},
         "bakery must be"),
    ]
    for entry, message in bad:
        with pytest.raises(TargetError, match=message):
            parse_target(entry)
    path = tmp_path / "targets.yaml"
    path.write_text("targets:\n  - {id: this-machine, name: X, relative_gpu_speed: 1, cpu_cores: 1, cost_per_hour: 0}\n")
    with pytest.raises(TargetError, match="reserved"):
        load_targets(path)
    assert load_targets(tmp_path / "missing.yaml") == []


def test_local_target_comes_from_measured_host_data():
    meta = {"host": {"hostname": "box", "cpu_model": "CPU", "usable_cpus": 28, "mem_total_kb": 16 * 1024 * 1024,
                     "gpus": [{"name": "RTX", "memory_total_mib": 8188}]}}
    header = {"gpu": {"devices": [{"name": "NVIDIA RTX 4060", "mem_total_mb": 8188}]}}
    target = local_target(meta, header)
    assert target.id == LOCAL_TARGET_ID and target.measured and target.runnable_here
    assert target.cpu_cores == 28 and target.memory_gb == pytest.approx(16)
    assert target.gpu_memory_gb == pytest.approx(8188 / 1024)
    assert target.name == "This machine (NVIDIA RTX 4060)" and not target.placeholders


def test_step_model_overlaps_with_workers_and_serializes_without():
    profile = make_profile(compute_s=0.010, prep_core_s=0.040)
    assert model_step(profile, local(), 0) == pytest.approx((0.010, 0.040, 0.050))
    assert model_step(profile, local(), 2) == pytest.approx((0.010, 0.020, 0.020))
    assert model_step(profile, local(), 8) == pytest.approx((0.010, 0.040 / 7, 0.010))
    assert model_step(profile, local(cpu_cores=3), 8) == pytest.approx((0.010, 0.020, 0.020))
    faster = local(relative_gpu_speed=2.0, relative_cpu_speed=2.0)
    assert model_step(profile, faster, 4) == pytest.approx((0.005, 0.005, 0.005))


def test_fixed_workers_are_enough_to_feed_the_gpu_but_capped_by_cores():
    profile = make_profile(compute_s=0.010, prep_core_s=0.045)
    assert fixed_workers(profile, local()) == 5
    assert fixed_workers(profile, local(cpu_cores=2)) == 1
    assert fixed_workers(make_profile(workers=12), local()) == 12
    assert fixed_workers(make_profile(prep_core_s=None, workers=3), local()) == 3


def test_overhead_is_calibrated_to_reproduce_the_measured_step():
    profile = make_profile(step_time_s=0.043, compute_s=0.010, prep_core_s=0.040, workers=1)
    plan = build_plan(profile, [local()], 1000)
    assert plan["profile"]["overhead_s"] == pytest.approx(0.003)
    as_is = plan["predictions"][0]["as_is"]
    assert as_is["step_time_s"] == pytest.approx(0.043)


def test_sessions_weekly_limits_cost_and_memory():
    profile = make_profile(step_time_s=0.040, compute_s=0.040, prep_core_s=0.0, workers=0, startup_s=36.0)
    target = make_target("stall", session_hours=1.0, weekly_hours=2.0, cost_per_hour=2.0, memory_gb=2.0)
    prediction = predict.predict(profile, target, 0, total_steps=225_000)
    assert prediction.training_hours == pytest.approx(2.5)
    assert prediction.sessions == math.ceil(2.5 * 3600 / (3600 - 36))
    assert prediction.total_hours == pytest.approx(2.5 + prediction.sessions * 36 / 3600)
    assert prediction.fits_one_session is False
    assert prediction.fits_weekly is False and prediction.weeks == 2
    assert prediction.cost == pytest.approx(prediction.total_hours * 2.0)
    assert prediction.fits_memory is False and prediction.feasible is False


def test_data_bound_run_gets_fix_first_with_upgrade_comparison():
    profile = make_profile()
    targets = [local(), make_target("cloud", name="Cloud", relative_gpu_speed=2.0, cost_per_hour=1.0)]
    plan = build_plan(profile, targets, 100_000)
    recommendation = plan["recommendation"]
    fix = recommendation["fix_first"]
    assert recommendation["scenario"] == "fixed"
    assert recommendation["headline"].startswith("Fix the data pipeline first")
    assert fix["current_workers"] == 1 and fix["suggested_workers"] == 4
    assert fix["gain"] > 0.7
    assert fix["upgrade"]["name"] == "Cloud" and fix["upgrade"]["gain"] == pytest.approx(0.0, abs=0.01)
    assert recommendation["best_target_id"] == LOCAL_TARGET_ID


def test_compute_bound_run_has_no_fix_and_prefers_free_when_close():
    profile = make_profile(step_time_s=0.050, compute_s=0.050, data_wait_s=0.0, prep_core_s=0.020, workers=4)
    targets = [local(), make_target("cloud", name="Cloud", relative_gpu_speed=1.1, cost_per_hour=1.0)]
    plan = build_plan(profile, targets, 10_000)
    recommendation = plan["recommendation"]
    assert recommendation["fix_first"] is None
    assert recommendation["best_target_id"] == LOCAL_TARGET_ID
    assert recommendation["headline"].startswith("Train on")
    fast = build_plan(make_profile(step_time_s=0.050, compute_s=0.050, data_wait_s=0.0, prep_core_s=0.020, workers=4),
                      targets, 10_000, prefer="time")
    assert fast["recommendation"]["best_target_id"] == "cloud"


def test_big_speedup_on_long_job_beats_free_but_slow():
    profile = make_profile(step_time_s=0.050, compute_s=0.050, data_wait_s=0.0, prep_core_s=0.020, workers=4)
    targets = [local(), make_target("cloud", name="Cloud", relative_gpu_speed=4.0, cost_per_hour=1.0)]
    plan = build_plan(profile, targets, 1_000_000)
    assert plan["recommendation"]["best_target_id"] == "cloud"
    cheap = build_plan(make_profile(step_time_s=0.050, compute_s=0.050, data_wait_s=0.0, prep_core_s=0.020,
                                    workers=4), targets, 1_000_000, prefer="cost")
    assert cheap["recommendation"]["best_target_id"] == LOCAL_TARGET_ID


def test_no_target_fits_memory():
    profile = make_profile(step_time_s=0.050, compute_s=0.050, data_wait_s=0.0, workers=0, prep_core_s=0.0,
                           host_memory_main_gb=64.0)
    plan = build_plan(profile, [local()], 1000)
    assert plan["recommendation"]["best_target_id"] is None
    assert plan["recommendation"]["headline"] == "No target has enough memory for this run."
    with pytest.raises(PlanError):
        build_plan(profile, [local()], 0)
    with pytest.raises(PlanError):
        build_plan(make_profile(), [local()], 10, prefer="vibes")


def logged_run(tmp_path, data_time=0.030, step_time=0.040, workers=(201,), gpu=None, warmup_steps=0):
    run = SyntheticRun(tmp_path)
    run.meta["analysis"] = {"warmup_steps": warmup_steps}
    for pid in workers:
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    procs = {MAIN_PID: {"cpu_pct": 40}, **{pid: {"cpu_pct": 100} for pid in workers}}
    run.steady(0, 30, procs, gpu=gpu)
    run.log_steps(2.0, 29.0, step_time)
    for record in run.metrics:
        record.update(data_time=data_time, compute_time=step_time - data_time)
    written = run.write()
    analyze_run(written)
    return written


def test_profile_split_from_logged_data_time(tmp_path):
    run = logged_run(tmp_path)
    profile = extract_profile(run)
    assert profile.split_method == "logged data_time"
    assert profile.step_time_s == pytest.approx(0.040)
    assert profile.data_wait_s == pytest.approx(0.030)
    assert profile.compute_s == pytest.approx(0.010)
    assert profile.workers == 1
    assert profile.prep_core_s == pytest.approx(0.040, rel=0.02)
    assert profile.startup_s == pytest.approx(2.0)


def test_profile_split_falls_back_to_gpu_utilization(tmp_path):
    run = SyntheticRun(tmp_path)
    run.steady(0, 30, {MAIN_PID: {"cpu_pct": 100}}, gpu=gpu_row(40.0))
    run.log_steps(2.0, 29.0, 0.050)
    written = run.write()
    analyze_run(written)
    profile = extract_profile(written)
    assert profile.split_method == "GPU utilization"
    assert profile.compute_s == pytest.approx(0.020)
    assert profile.workers == 0 and profile.prep_core_s == pytest.approx(0.030)


def test_profile_without_timers_or_gpu_is_flagged(tmp_path):
    run = SyntheticRun(tmp_path)
    run.steady(0, 30, {MAIN_PID: {"cpu_pct": 100}})
    run.log_steps(2.0, 29.0, 0.050)
    written = run.write()
    analyze_run(written)
    profile = extract_profile(written)
    assert profile.split_method.startswith("assumed")
    assert profile.prep_core_s is None
    assert any("data_time" in note for note in profile.notes)


def test_warmup_steps_are_excluded_from_the_profile(tmp_path):
    run = logged_run(tmp_path, warmup_steps=20)
    summary = run.read_summary()
    assert summary["window"]["warmup_steps"] == 20
    assert summary["window"]["start_mono"] == pytest.approx(2.0 + 20 * 0.040)


def test_plan_is_saved_with_the_run(tmp_path):
    run = logged_run(tmp_path)
    plan = plan_for_run(run, 50_000, REPO_TARGETS)
    assert read_plan(run)["recommendation"] == json.loads(json.dumps(plan["recommendation"]))
    assert plan["targets"][0]["id"] == LOCAL_TARGET_ID
    assert len(plan["predictions"]) == 4
    assert plan["placeholders_used"] and plan["assumptions"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def append_steps(path, steps, start_time, step_time):
    with open(path, "a") as handle:
        for index, step in enumerate(steps):
            handle.write(json.dumps({"t_mono": start_time + index * step_time, "step": step}) + "\n")


def test_profile_window_stops_after_warmup_plus_steps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    clock = FakeClock()
    window = ProfileWindow(path, warmup_steps=5, window_steps=10, window_seconds=60, clock=clock)
    assert window() is None
    append_steps(path, range(0, 10), 0.0, 0.1)
    clock.now = 1.0
    assert window() is None
    with open(path, "a") as handle:
        handle.write('{"t_mono": 1.0, "step": 10')
    assert window() is None
    with open(path, "a") as handle:
        handle.write("}\n")
    append_steps(path, range(11, 16), 1.1, 0.1)
    assert window() == "profiled 10 steps after warm-up"


def test_profile_window_stops_after_seconds_and_times_out(tmp_path):
    path = tmp_path / "metrics.jsonl"
    clock = FakeClock()
    window = ProfileWindow(path, warmup_steps=2, window_steps=10_000, window_seconds=5, clock=clock)
    append_steps(path, range(0, 4), 0.0, 1.0)
    clock.now = 4.0
    assert window() is None
    clock.now = 7.5
    assert window() == "profiled 5s after warm-up"
    silent = ProfileWindow(tmp_path / "none.jsonl", startup_timeout_s=10, clock=clock)
    clock.now = 100.0
    assert "no steps were logged" in silent()


ENDLESS_TRAINING = """
    import sys
    import time
    from pathlib import Path
    import mlplat

    marker = Path(sys.argv[1])
    step = 0
    try:
        while True:
            started = time.monotonic()
            time.sleep(0.006)
            data_time = time.monotonic() - started
            busy_until = time.monotonic() + 0.004
            while time.monotonic() < busy_until:
                pass
            mlplat.log(step=step, loss=1.0, data_time=data_time, compute_time=0.004)
            step += 1
    except KeyboardInterrupt:
        marker.write_text(str(step))
        raise
"""


def test_mlplat_plan_profiles_stops_and_saves(mlplat_cli, tmp_path):
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent(ENDLESS_TRAINING))
    marker = tmp_path / "stopped"
    result = mlplat_cli("plan", "--name", "plan-demo", "--total-steps", "100000", "--warmup-steps", "10",
                        "--window-steps", "150", "--window-seconds", "60", "--interval-ms", "200",
                        "--targets", REPO_TARGETS, "--", sys.executable, str(script), str(marker))
    assert result.returncode == 0, result.stderr
    assert int(marker.read_text()) >= 160
    run = RunStore(tmp_path / "runs").list()[0]
    meta = run.read_meta()
    assert meta["status"] == "profiled" and meta["exit_code"] == 0
    assert meta["stop_reason"] == "profiled 150 steps after warm-up"
    plan = read_plan(run)
    assert plan["profile"]["split_method"] == "logged data_time"
    assert plan["profile"]["step_time_s"] == pytest.approx(0.0105, rel=0.3)
    assert "All times and costs are estimates." in result.stdout

    again = mlplat_cli("plan", "--from-run", run.run_id, "--total-steps", "5000", "--targets", REPO_TARGETS, "--json")
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)["total_steps"] == 5000


def test_mlplat_plan_requires_a_command(mlplat_cli):
    result = mlplat_cli("plan", "--total-steps", "10")
    assert result.returncode == 2
    missing = mlplat_cli("plan", "--from-run", "nope", "--total-steps", "10")
    assert missing.returncode == 1 and "no run matches" in missing.stderr
