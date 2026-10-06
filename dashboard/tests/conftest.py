import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mlplat" / "tests"))

from mlplat.analyzer import analyze_run
from synthetic import MAIN_PID, SyntheticRun, gpu_row, scenario


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("MLPLAT_RUNS_DIR", str(root))
    return root


@pytest.fixture
def populated(runs_dir):
    slow = scenario(runs_dir, main_cpu=30, worker_cpus=[97])
    analyze_run(slow)
    fast = SyntheticRun(runs_dir, name="fast-gpu")
    for pid in (201, 202, 203):
        fast.add_process(pid, MAIN_PID, "python train.py", 0.0)
    fast.steady(0, 20, {MAIN_PID: {"cpu_pct": 100}, 201: {"cpu_pct": 60}, 202: {"cpu_pct": 55},
                        203: {"cpu_pct": 58}}, gpu=gpu_row(91.0))
    fast.log_steps(0.5, 19.5, 0.1)
    for record in fast.metrics:
        record.update(accuracy=0.9, data_time=0.005, compute_time=0.095)
    fast_run = fast.write()
    analyze_run(fast_run)
    return {"slow": slow, "fast": fast_run}
