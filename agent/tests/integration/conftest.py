import subprocess
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = AGENT_DIR / "build"


@pytest.fixture(scope="session")
def agent_binary():
    subprocess.run(["cmake", "-S", str(AGENT_DIR), "-B", str(BUILD_DIR)], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--target", "mlplat-agent", "-j"], check=True,
                   capture_output=True)
    return BUILD_DIR / "mlplat-agent"
