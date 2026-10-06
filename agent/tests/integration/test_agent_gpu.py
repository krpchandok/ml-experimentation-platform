import os
import shutil
import subprocess
import sys

import pytest

from agent_helpers import read_records

GPU_WORKLOAD = """
import time
import torch
x = torch.randn(2048, 2048, device="cuda")
deadline = time.time() + 4
while time.time() < deadline:
    y = x @ x
    torch.cuda.synchronize()
"""


def fake_library(agent_binary):
    return agent_binary.parent / "libfake-nvidia-ml.so"


def run_agent(agent_binary, tmp_path, extra_args, env=None, seconds=1.5):
    out = tmp_path / "resources.jsonl"
    sleeper = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
    child_env = {**os.environ, **(env or {})}
    child_env.setdefault("FAKE_NVML_PID", str(sleeper.pid))
    code = subprocess.run([str(agent_binary), "--pid", str(sleeper.pid), "--interval-ms", "200", "--out", str(out),
                           *extra_args], env=child_env, timeout=30).returncode
    sleeper.wait()
    return code, read_records(out)


def samples(records):
    return [record for record in records if record["type"] == "sample"]


def test_fake_nvml_is_loaded_at_runtime(agent_binary, tmp_path):
    code, records = run_agent(agent_binary, tmp_path, ["--nvml-library", str(fake_library(agent_binary))],
                              env={"FAKE_NVML_UTIL": "70"})
    assert code == 0
    header = records[0]["gpu"]
    assert header["available"] is True
    assert header["driver"] == "999.99-fake"
    assert header["devices"][0]["name"] == "Fake GPU 0"
    gpu_samples = [sample["gpu"] for sample in samples(records)]
    assert gpu_samples[0]["devices"][0]["util_source"] == "rates"
    assert gpu_samples[-1]["devices"][0]["util_source"] == "samples"
    assert gpu_samples[-1]["devices"][0]["util_pct"] == pytest.approx(70)
    assert gpu_samples[1]["procs"] == [{"pid": records[0]["root_pid"], "device": 0, "mem_used_mb": 512}]
    assert gpu_samples[-1]["procs"] == []


def test_missing_nvml_records_unavailable_and_keeps_sampling(agent_binary, tmp_path):
    code, records = run_agent(agent_binary, tmp_path, ["--nvml-library", str(tmp_path / "libnvidia-ml.so.1")])
    assert code == 0
    assert records[0]["gpu"]["available"] is False
    assert "cannot load" in records[0]["gpu"]["reason"]
    assert all(sample["gpu"] == "unavailable" for sample in samples(records))
    assert all(sample["procs"] for sample in samples(records)[:-1])


def test_gpu_can_be_disabled(agent_binary, tmp_path):
    code, records = run_agent(agent_binary, tmp_path, ["--gpu", "off"])
    assert code == 0
    assert records[0]["gpu"] == {"available": False, "reason": "disabled with --gpu off"}


def real_gpu_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not real_gpu_available(), reason="needs an NVIDIA GPU with CUDA PyTorch")
def test_real_gpu_utilization_is_attributed_to_the_run(agent_binary, tmp_path):
    out = tmp_path / "resources.jsonl"
    workload = subprocess.Popen([sys.executable, "-c", GPU_WORKLOAD])
    agent = subprocess.Popen([str(agent_binary), "--pid", str(workload.pid), "--interval-ms", "500", "--out", str(out)])
    workload.wait(timeout=120)
    agent.wait(timeout=30)
    records = read_records(out)
    assert records[0]["gpu"]["available"] is True
    utils = [sample["gpu"]["devices"][0]["util_pct"] for sample in samples(records)
             if isinstance(sample["gpu"], dict) and sample["gpu"]["devices"][0]["util_pct"] is not None]
    assert max(utils) >= 80
    listed = {proc["pid"] for sample in samples(records) if isinstance(sample["gpu"], dict)
              for proc in sample["gpu"]["procs"]}
    process_info = any(isinstance(sample["gpu"], dict) and sample["gpu"]["process_info"] for sample in samples(records))
    if process_info:
        assert workload.pid in listed
