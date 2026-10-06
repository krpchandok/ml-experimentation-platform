import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mlplat.run_store import RunStore

TRAIN_WITH_WORKERS = """
    import multiprocessing as mp
    import os
    import mlplat

    def work(n):
        return sum(i * i for i in range(n))

    if __name__ == "__main__":
        print("run id", os.environ["MLPLAT_RUN_ID"], "config", os.environ.get("MLPLAT_CONFIG_PATH"))
        with mp.Pool(2) as pool:
            for step in range(8):
                pool.map(work, [1_500_000] * 4)
                mlplat.log(step=step, loss=1.0 / (step + 1), accuracy=step / 8)
        print("training done")
"""

INTERRUPTIBLE = """
    import subprocess
    import sys
    import time
    from pathlib import Path

    marker = Path(sys.argv[1])
    child = subprocess.Popen([sys.executable, "-c",
        "import sys, time\\ntry:\\n    time.sleep(60)\\nexcept KeyboardInterrupt:\\n    open(sys.argv[1], 'w').write('child')\\n    raise",
        str(marker) + ".child"])
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        marker.write_text("parent")
        child.wait()
        raise
"""


def write_script(tmp_path, source, name="train.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def only_run(tmp_path):
    runs = RunStore(tmp_path / "runs").list()
    assert len(runs) == 1
    return runs[0]


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_successful_run_records_everything(mlplat_cli, tmp_path):
    script = write_script(tmp_path, TRAIN_WITH_WORKERS)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\nhyperparameters:\n  lr: 0.1\n  workers: 2\n")

    result = mlplat_cli("run", "--name", "exp1", "--config", str(config), "--interval-ms", "200", "--",
                        sys.executable, str(script))
    assert result.returncode == 0, result.stderr
    assert "training done" in result.stdout

    run = only_run(tmp_path)
    meta = run.read_meta()
    assert meta["name"] == "exp1"
    assert meta["status"] == "completed"
    assert meta["exit_code"] == 0
    assert meta["config"] == {"model": "tiny", "hyperparameters": {"lr": 0.1, "workers": 2}}
    assert meta["config_hash_source"] == "config" and len(meta["config_hash"]) == 16
    assert meta["command"] == [sys.executable, str(script)]
    assert meta["host"]["logical_cpus"] >= 1 and meta["host"]["mem_total_kb"] > 0
    assert meta["end_wall"] >= meta["start_wall"]
    assert meta["agent"]["available"] and meta["agent"]["exit_code"] == 0
    assert (run.path / "config.yaml").read_text() == config.read_text()

    metrics = read_jsonl(run.metrics_path)
    assert [record["step"] for record in metrics] == list(range(8))
    assert all(meta["start_mono"] <= record["t_mono"] <= meta["end_mono"] for record in metrics)

    resources = read_jsonl(run.resources_path)
    assert resources[0]["root_pid"] == meta["pid"]
    assert resources[-1]["reason"] == "root_exited"
    tracked = {record["pid"] for record in resources if record["type"] == "process"}
    assert meta["pid"] in tracked and len(tracked) >= 3

    output = run.output_path.read_text()
    assert f"run id {run.run_id}" in output and str(config.resolve()) in output and "training done" in output


def test_failing_command_propagates_exit_code(mlplat_cli, tmp_path):
    script = write_script(tmp_path, "import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n")
    result = mlplat_cli("run", "--name", "bad", "--", sys.executable, str(script))
    assert result.returncode == 3
    meta = only_run(tmp_path).read_meta()
    assert meta["status"] == "failed" and meta["exit_code"] == 3
    assert "boom" in only_run(tmp_path).output_path.read_text()


def test_missing_command_is_recorded(mlplat_cli, tmp_path):
    result = mlplat_cli("run", "--name", "nothing", "--", "definitely-not-a-command-xyz")
    assert result.returncode == 127
    meta = only_run(tmp_path).read_meta()
    assert meta["status"] == "failed" and "error" in meta


def test_run_without_agent_still_completes(mlplat_cli, tmp_path):
    script = write_script(tmp_path, "import mlplat\nmlplat.log(step=0, loss=1.0)\n")
    result = mlplat_cli("run", "--name", "noagent", "--agent", str(tmp_path / "no-agent"), "--",
                        sys.executable, str(script))
    assert result.returncode == 0, result.stderr
    run = only_run(tmp_path)
    meta = run.read_meta()
    assert meta["agent"]["available"] is False
    assert meta["status"] == "completed"
    assert "resource agent not found" in result.stderr
    assert run.metrics_path.exists()


def test_sigint_is_forwarded_to_the_whole_training_tree(cli_env, tmp_path):
    script = write_script(tmp_path, INTERRUPTIBLE)
    marker = tmp_path / "marker"
    launcher = subprocess.Popen([sys.executable, "-m", "mlplat", "run", "--name", "ctrlc", "--interval-ms", "100",
                                 "--", sys.executable, str(script), str(marker)],
                                cwd=tmp_path, env=cli_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def ready():
        runs = RunStore(tmp_path / "runs").list()
        if not runs or not runs[0].resources_path.exists():
            return False
        processes = [record for record in read_jsonl(runs[0].resources_path) if record["type"] == "process"]
        return len(processes) >= 2

    assert wait_for(ready)
    launcher.send_signal(signal.SIGINT)
    launcher.wait(timeout=30)

    assert launcher.returncode == 130
    assert marker.read_text() == "parent"
    assert Path(str(marker) + ".child").read_text() == "child"
    meta = only_run(tmp_path).read_meta()
    assert meta["status"] == "interrupted"
    assert meta["signals_received"] == ["SIGINT"]
    assert read_jsonl(only_run(tmp_path).resources_path)[-1]["type"] == "end"


def test_leftover_processes_are_terminated(mlplat_cli, tmp_path):
    script = write_script(tmp_path, """
        import subprocess
        import sys
        child = subprocess.Popen(["sleep", "60"])
        print(child.pid)
    """)
    result = mlplat_cli("run", "--name", "leaky", "--", sys.executable, str(script))
    assert result.returncode == 0
    leaked_pid = int(result.stdout.strip().splitlines()[-1])
    meta = only_run(tmp_path).read_meta()
    assert meta["leftover_processes_terminated"] is True
    assert wait_for(lambda: not Path(f"/proc/{leaked_pid}").exists() or
                    Path(f"/proc/{leaked_pid}/stat").read_text().split(") ")[1].startswith("Z"), timeout=5)


def test_list_shows_runs(mlplat_cli, tmp_path):
    script = write_script(tmp_path, "print('hi')\n")
    mlplat_cli("run", "--name", "listed", "--", sys.executable, str(script))
    result = mlplat_cli("list")
    assert result.returncode == 0
    assert "listed" in result.stdout and "completed" in result.stdout


def test_mlflow_mirror_writes_params_metrics_and_artifacts(mlplat_cli, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    script = write_script(tmp_path, TRAIN_WITH_WORKERS)
    config = tmp_path / "config.yaml"
    config.write_text("model: tiny\nhyperparameters:\n  lr: 0.1\n")
    result = mlplat_cli("run", "--name", "mirrored", "--config", str(config), "--interval-ms", "200",
                        "--mlflow", "on", "--", sys.executable, str(script))
    assert result.returncode == 0, result.stderr

    meta = only_run(tmp_path).read_meta()
    assert "run_id" in meta["mlflow"], meta.get("mlflow")
    client = mlflow.tracking.MlflowClient(tracking_uri=meta["mlflow"]["tracking_uri"])
    mlflow_run = client.get_run(meta["mlflow"]["run_id"])
    assert mlflow_run.info.status == "FINISHED"
    assert mlflow_run.data.params == {"model": "tiny", "hyperparameters.lr": "0.1"}
    assert mlflow_run.data.tags["mlplat.run_id"] == meta["run_id"]
    assert mlflow_run.data.tags["mlplat.config_hash"] == meta["config_hash"]
    assert len(client.get_metric_history(mlflow_run.info.run_id, "loss")) == 8
    assert client.get_metric_history(mlflow_run.info.run_id, "resource.tree_cpu_pct")
    artifacts = {item.path for item in client.list_artifacts(mlflow_run.info.run_id, "mlplat")}
    assert {"mlplat/meta.json", "mlplat/metrics.jsonl", "mlplat/resources.jsonl"} <= artifacts


def test_unreachable_mlflow_does_not_fail_the_run(mlplat_cli, cli_env, tmp_path):
    pytest.importorskip("mlflow")
    cli_env["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:9"
    script = write_script(tmp_path, "import mlplat\nmlplat.log(step=0, loss=1.0)\n")
    started = time.monotonic()
    result = mlplat_cli("run", "--name", "offline", "--", sys.executable, str(script))
    assert result.returncode == 0
    assert time.monotonic() - started < 60
    meta = only_run(tmp_path).read_meta()
    assert meta["status"] == "completed"
    assert meta["mlflow"]["error"] == "mirror failed"
    assert "could not mirror run to MLflow" in result.stderr
