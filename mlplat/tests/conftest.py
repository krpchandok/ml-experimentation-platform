import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent"
AGENT_BUILD = AGENT_DIR / "build"


@pytest.fixture(scope="session")
def agent_binary():
    subprocess.run(["cmake", "-S", str(AGENT_DIR), "-B", str(AGENT_BUILD)], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(AGENT_BUILD), "--target", "mlplat-agent", "-j"], check=True,
                   capture_output=True)
    return AGENT_BUILD / "mlplat-agent"


@pytest.fixture
def cli_env(agent_binary, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env["MLPLAT_AGENT"] = str(agent_binary)
    env["MLPLAT_RUNS_DIR"] = str(tmp_path / "runs")
    for key in ("MLFLOW_TRACKING_URI", "MLPLAT_RUN_DIR"):
        env.pop(key, None)
    return env


@pytest.fixture
def mlplat_cli(cli_env, tmp_path):
    def invoke(*args, **kwargs):
        return subprocess.run([sys.executable, "-m", "mlplat", *args], cwd=tmp_path, env=cli_env,
                              capture_output=True, text=True, timeout=120, **kwargs)
    return invoke
