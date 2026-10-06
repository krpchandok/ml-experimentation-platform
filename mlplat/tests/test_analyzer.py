import json
import subprocess
import sys

import pytest

from mlplat import verdict as rules
from mlplat.analyzer import analyze_run, format_report
from synthetic import MAIN_PID, RESOURCE_TRACKER_CMDLINE, SyntheticRun, gpu_row, scenario

MIB = 1024 * 1024
GIB_KB = 1024 * 1024


def primary(run):
    return analyze_run(run)["verdict"]["primary"]


def all_verdicts(summary):
    return [summary["verdict"]["primary"]] + [finding["verdict"] for finding in summary["verdict"]["secondary"]]


def test_preprocessing_bound_without_gpu(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=30, worker_cpus=[97, 96, 98, 95]))
    verdict = summary["verdict"]
    assert verdict["primary"] == "preprocessing_bound"
    assert verdict["severity"] == "bottleneck"
    assert "4 data-loader worker(s) each averaged 96%" in verdict["evidence"][0]
    assert "waiting for batches" in verdict["evidence"][1]
    assert summary["roles"]["workers"]["count_median"] == 4


def test_preprocessing_bound_with_starved_gpu(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=100, worker_cpus=[98, 99], gpu_util=20))
    assert summary["verdict"]["primary"] == "preprocessing_bound"
    assert "GPU utilization averaged 20%" in summary["verdict"]["evidence"][1]
    assert summary["gpu"]["util_mean_pct"] == pytest.approx(20)


def test_busy_gpu_is_healthy_even_with_saturated_workers(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=100, worker_cpus=[98, 99], gpu_util=92))
    assert summary["verdict"]["primary"] == "healthy"
    assert "preprocessing_bound" not in all_verdicts(summary)


def test_main_process_bound_without_workers(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=99, worker_cpus=[]))
    verdict = summary["verdict"]
    assert verdict["primary"] == "main_process_bound"
    assert any("no data-loader worker processes" in line for line in verdict["evidence"])
    assert "num_workers > 0" in verdict["suggestions"][0]


def test_main_process_bound_with_idle_workers(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=99, worker_cpus=[5, 4]))
    verdict = summary["verdict"]
    assert verdict["primary"] == "main_process_bound"
    assert any("mostly idle" in line for line in verdict["evidence"])
    assert "Python-side overhead" in verdict["suggestions"][0]


def test_multithreaded_main_using_most_cores_is_healthy(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=620, worker_cpus=[20, 20], usable_cpus=8))
    assert summary["verdict"]["primary"] == "healthy"
    assert "of 8 usable cores busy" in summary["verdict"]["evidence"][0]


def test_io_bound_from_reads_and_block_wait(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=20, worker_cpus=[12, 10],
                                   worker_extra={"read_bytes_per_s": 100 * MIB, "blkio_wait_pct": 45}))
    verdict = summary["verdict"]
    assert verdict["primary"] == "io_bound"
    joined = " ".join(verdict["evidence"])
    assert "45% of their time blocked on disk I/O" in joined
    assert "200.0 MiB/s" in joined


def test_io_bound_from_system_iowait_without_delay_accounting(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=15, worker_cpus=[], system={"cpu_iowait_pct": 25}))
    assert summary["verdict"]["primary"] == "io_bound"
    assert "main training process" in summary["verdict"]["evidence"][0]


def test_heavy_reads_with_busy_gpu_are_not_io_bound(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=100, worker_cpus=[20, 20], gpu_util=95,
                                   worker_extra={"read_bytes_per_s": 300 * MIB}))
    assert summary["verdict"]["primary"] == "healthy"


def test_memory_near_host_limit_is_a_warning(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=620, worker_cpus=[], usable_cpus=8, mem_total_kb=4 * GIB_KB,
                                   main_extra={"rss_kb": int(3.8 * GIB_KB)}))
    verdict = summary["verdict"]
    assert verdict["primary"] == "memory_pressure"
    assert verdict["severity"] == "warning"
    assert "95% of the host limit" in verdict["evidence"][0]
    assert "healthy" in all_verdicts(summary)


def test_oom_in_cgroup_is_critical_and_outranks_bottlenecks(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=30, worker_cpus=[97, 97], exit_code=137,
                                   cgroup={"memory_max_bytes": 2 * 1024 ** 3, "oom_kill": 1},
                                   main_extra={"rss_kb": int(1.99 * GIB_KB)}))
    verdict = summary["verdict"]
    assert verdict["primary"] == "memory_pressure"
    assert verdict["severity"] == "critical"
    joined = " ".join(verdict["evidence"])
    assert "cgroup limit" in joined and "OOM killer fired 1 time" in joined and "exit 137" in joined
    assert "preprocessing_bound" in all_verdicts(summary)


def test_rising_major_faults_signal_memory_pressure(tmp_path):
    run = SyntheticRun(tmp_path)
    for second in range(30):
        majflt = 5.0 if second < 10 else 400.0 if second >= 20 else 60.0
        run.sample(second + 1, {MAIN_PID: {"cpu_pct": 40, "majflt_per_s": majflt}})
    run.log_steps(0.5, 29.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["verdict"]["primary"] == "memory_pressure"
    assert "Major page faults rose from 5/s" in summary["verdict"]["evidence"][0]


def test_sigkill_exit_alone_is_not_called_oom(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=20, worker_cpus=[], exit_code=137))
    assert "memory_pressure" not in all_verdicts(summary)


def test_underutilized_when_nothing_is_busy(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=20, worker_cpus=[10, 10]))
    assert summary["verdict"]["primary"] == "underutilized"
    assert summary["verdict"]["severity"] == "info"


def test_insufficient_data_when_agent_missing(tmp_path):
    run = SyntheticRun(tmp_path, agent_available=False)
    run.records = []
    summary = analyze_run(run.write())
    assert summary["verdict"]["primary"] == "insufficient_data"
    assert summary["data_quality"]["agent_available"] is False


def test_insufficient_data_when_too_few_samples(tmp_path):
    summary = analyze_run(scenario(tmp_path, main_cpu=99, worker_cpus=[], seconds=2))
    assert summary["verdict"]["primary"] == "insufficient_data"


def test_warmup_before_first_logged_step_is_excluded(tmp_path):
    run = SyntheticRun(tmp_path)
    for pid in (201, 202):
        run.add_process(pid, MAIN_PID, "python train.py", 10.0)
    run.steady(0, 10, {MAIN_PID: {"cpu_pct": 100}})
    run.steady(10, 20, {MAIN_PID: {"cpu_pct": 25}, 201: {"cpu_pct": 98}, 202: {"cpu_pct": 97}})
    run.log_steps(10.0, 30.0, 0.5)
    summary = analyze_run(run.write())
    assert summary["window"]["source"] == "metrics"
    assert summary["window"]["warmup_excluded_s"] == pytest.approx(10.0)
    assert summary["window"]["samples"] == 20
    assert summary["verdict"]["primary"] == "preprocessing_bound"


def test_step_statistics_handle_sparse_logging(tmp_path):
    run = SyntheticRun(tmp_path)
    run.steady(0, 12, {MAIN_PID: {"cpu_pct": 50}})
    run.log_steps(1.0, 11.0, 0.25, every=4)
    totals = analyze_run(run.write())["totals"]
    assert totals["step_time_median_s"] == pytest.approx(0.25)
    assert totals["steps_per_sec"] == pytest.approx(4.0)
    assert totals["steps_in_window"] == 40
    assert totals["cpu_seconds_per_step"] == pytest.approx(10 * 0.5 / 40, rel=0.15)
    assert totals["final_metrics"]["loss"] == pytest.approx(1 / 41)


def test_helper_processes_do_not_count_as_workers(tmp_path):
    run = SyntheticRun(tmp_path)
    run.add_process(300, MAIN_PID, RESOURCE_TRACKER_CMDLINE, 0.0)
    for pid in (201, 202):
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 30}, 300: {"cpu_pct": 0}, 201: {"cpu_pct": 95}, 202: {"cpu_pct": 95}})
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    roles = {row["pid"]: row["role"] for row in summary["processes"]}
    assert roles == {MAIN_PID: "main", 300: "helper", 201: "worker", 202: "worker"}
    assert summary["roles"]["workers"]["cpu_mean_pct"] == pytest.approx(95)
    assert summary["verdict"]["primary"] == "preprocessing_bound"


def test_logged_data_time_is_used_as_evidence(tmp_path):
    run = SyntheticRun(tmp_path)
    run.add_process(201, MAIN_PID, "python train.py", 0.0)
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 30}, 201: {"cpu_pct": 99}})
    run.log_steps(0.5, 19.5, 0.4)
    for record in run.metrics:
        record.update(data_time=0.3, compute_time=0.1)
    summary = analyze_run(run.write())
    assert summary["features"]["data_wait_fraction"] == pytest.approx(0.75)
    assert any("75% of its time waiting for the next batch" in line for line in summary["verdict"]["evidence"])


def test_recreated_workers_are_pooled_per_sample(tmp_path):
    run = SyntheticRun(tmp_path)
    for pid in (201, 202):
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    run.steady(0, 10, {MAIN_PID: {"cpu_pct": 30}, 201: {"cpu_pct": 96}, 202: {"cpu_pct": 96}})
    run.exit_process(201, 10.5)
    run.exit_process(202, 10.5)
    for pid in (203, 204):
        run.add_process(pid, MAIN_PID, "python train.py", 10.5)
    run.steady(10, 10, {MAIN_PID: {"cpu_pct": 30}, 203: {"cpu_pct": 96}, 204: {"cpu_pct": 96}})
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["roles"]["workers"]["count_median"] == 2
    assert summary["roles"]["workers"]["count_total"] == 4
    assert summary["verdict"]["primary"] == "preprocessing_bound"


def test_gpu_features_use_devices_running_the_tree(tmp_path):
    run = SyntheticRun(tmp_path)
    gpu = gpu_row([95.0, 10.0], devices=2, procs=[{"pid": MAIN_PID, "device": 1, "mem_used_mb": 1500.0},
                                                  {"pid": 9999, "device": 0, "mem_used_mb": 7000.0}])
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 100}}, gpu=gpu)
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["gpu"]["util_mean_pct"] == pytest.approx(10.0)
    assert summary["gpu"]["device_count"] == 1
    assert summary["gpu"]["tree_memory_peak_mb"] == pytest.approx(1500.0)
    assert summary["verdict"]["primary"] == "main_process_bound"


def test_run_missing_from_gpu_process_list_is_gpu_not_used(tmp_path):
    run = SyntheticRun(tmp_path)
    gpu = gpu_row(4.0, procs=[{"pid": 9999, "device": 0, "mem_used_mb": 700.0}])
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 650}}, gpu=gpu)
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["verdict"]["primary"] == "gpu_not_used"
    assert summary["gpu"]["tree_on_gpu"] is False
    assert "model.to('cuda')" in summary["verdict"]["suggestions"][0]


def test_no_process_info_never_claims_gpu_not_used(tmp_path):
    run = SyntheticRun(tmp_path)
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 99}}, gpu=gpu_row(4.0, procs=[], process_info=False))
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["gpu"]["tree_on_gpu"] is None
    assert "gpu_not_used" not in all_verdicts(summary)


def test_unknown_per_process_gpu_memory_stays_unknown(tmp_path):
    run = SyntheticRun(tmp_path)
    gpu = gpu_row(95.0, procs=[{"pid": MAIN_PID, "device": 0, "mem_used_mb": None}])
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 100}}, gpu=gpu)
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["gpu"]["tree_on_gpu"] is True
    assert summary["gpu"]["tree_memory_peak_mb"] is None
    assert summary["verdict"]["primary"] == "healthy"


def test_unavailable_gpu_marker_falls_back_to_cpu_rules(tmp_path):
    run = SyntheticRun(tmp_path)
    run.steady(0, 20, {MAIN_PID: {"cpu_pct": 99}}, gpu="unavailable")
    run.log_steps(0.5, 19.5, 0.5)
    summary = analyze_run(run.write())
    assert summary["gpu"] is None
    assert summary["verdict"]["primary"] == "main_process_bound"
    assert "No GPU data" in summary["data_quality"]["notes"][0]


@pytest.mark.parametrize("worker_cpu, expected", [(rules.WORKER_SATURATED_PCT, "preprocessing_bound"),
                                                   (rules.WORKER_SATURATED_PCT - 0.1, "underutilized")])
def test_worker_saturation_threshold_boundary(worker_cpu, expected):
    features = rules.Features(window_samples=10, window_seconds=10, usable_cores=8, tree_cores_mean=2,
                              main_cpu_mean_pct=30, worker_count=2, worker_cpu_mean_pct=worker_cpu)
    assert rules.decide(features)["primary"] == expected


def cpu_features(**overrides):
    values = dict(window_samples=30, window_seconds=30, usable_cores=28, tree_cores_mean=2.5,
                  main_cpu_mean_pct=220, worker_count=0)
    values.update(overrides)
    return rules.Features(**values)


def test_inline_loading_with_multithreaded_main_is_main_process_bound():
    decision = rules.decide(cpu_features(data_wait_fraction=0.6))
    assert decision["primary"] == "main_process_bound"
    assert "produced every batch itself" in decision["evidence"][0]


def test_multithreaded_main_starved_by_one_worker_is_preprocessing_bound():
    decision = rules.decide(cpu_features(worker_count=1, worker_cpu_mean_pct=98, data_wait_fraction=0.5))
    assert decision["primary"] == "preprocessing_bound"
    assert "50% of its time waiting for the next batch" in decision["evidence"][1]


def test_compute_bound_cpu_run_with_fast_pipeline_is_healthy():
    decision = rules.decide(cpu_features(worker_count=6, worker_cpu_mean_pct=40, main_cpu_mean_pct=390,
                                         tree_cores_mean=6.3, data_wait_fraction=0.03))
    assert decision["primary"] == "healthy"
    assert "compute-bound" in decision["evidence"][0]


def test_multithreaded_main_without_logged_data_time_is_not_misclassified():
    assert rules.decide(cpu_features())["primary"] == "underutilized"


def gpu_features(util):
    return rules.GpuFeatures(device_count=1, util_mean_pct=util, util_p50_pct=util, process_info=True,
                             tree_on_gpu=True)


def test_gpu_in_the_gray_zone_with_logged_waiting_is_starved():
    decision = rules.decide(cpu_features(worker_count=4, worker_cpu_mean_pct=100, main_cpu_mean_pct=74,
                                         data_wait_fraction=0.40, gpu=gpu_features(57)))
    assert decision["primary"] == "preprocessing_bound"
    assert "only 57%" in decision["evidence"][1]


def test_gpu_in_the_gray_zone_without_waiting_is_not_starved():
    decision = rules.decide(cpu_features(worker_count=4, worker_cpu_mean_pct=100, main_cpu_mean_pct=74,
                                         data_wait_fraction=0.05, gpu=gpu_features(57)))
    assert decision["primary"] != "preprocessing_bound"


def test_launch_bound_gpu_loop_is_main_process_bound_even_with_busy_workers():
    decision = rules.decide(cpu_features(worker_count=8, worker_cpu_mean_pct=96, main_cpu_mean_pct=112,
                                         data_wait_fraction=0.05, gpu=gpu_features(77)))
    assert decision["primary"] == "main_process_bound"
    assert "input pipeline kept up" in decision["evidence"][1]
    assert ".item()" in decision["suggestions"][0]


def test_summary_records_thresholds_and_is_written(tmp_path):
    run = scenario(tmp_path, main_cpu=30, worker_cpus=[97, 97])
    summary = analyze_run(run)
    assert json.loads(run.summary_path.read_text())["verdict"]["primary"] == "preprocessing_bound"
    assert summary["verdict"]["rules_version"] == rules.RULES_VERSION
    assert summary["verdict"]["thresholds"]["WORKER_SATURATED_PCT"] == rules.WORKER_SATURATED_PCT
    report = format_report(summary)
    assert "VERDICT: Preprocessing-bound" in report and "Suggestions:" in report


def test_analyze_cli(tmp_path, cli_env):
    scenario(tmp_path / "runs", main_cpu=30, worker_cpus=[97, 97])

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "mlplat", "analyze", *args], env=cli_env, cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)

    latest = cli()
    assert latest.returncode == 0, latest.stderr
    assert "VERDICT: Preprocessing-bound" in latest.stdout
    as_json = cli("latest", "--json")
    assert json.loads(as_json.stdout)["verdict"]["primary"] == "preprocessing_bound"
    missing = cli("nope")
    assert missing.returncode == 1 and "no run matches" in missing.stderr
