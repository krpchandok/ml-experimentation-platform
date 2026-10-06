import signal
import subprocess
import sys
import time

from agent_helpers import mean_cpu, per_pid_series, read_records, run_workload_with_agent, total_bytes

MIB = 1024 * 1024


def test_agent_attributes_load_to_the_right_processes(agent_binary, tmp_path):
    roles, records, agent_code = run_workload_with_agent(agent_binary, tmp_path)
    assert agent_code == 0

    header, end = records[0], records[-1]
    assert header["type"] == "header"
    assert header["root_pid"] == roles["root"]
    assert end["type"] == "end"
    assert end["reason"] == "root_exited"
    assert end["samples"] >= 10

    started = {record["pid"] for record in records if record["type"] == "process"}
    tracked = {roles["root"], roles["cpu"], roles["io"], roles["spawner"], roles["grandchild"]}
    assert tracked <= started

    series = per_pid_series(records)
    cpu = mean_cpu(series[roles["cpu"]])
    grandchild = mean_cpu(series[roles["grandchild"]])
    io = mean_cpu(series[roles["io"]])
    root = mean_cpu(series[roles["root"]])
    spawner = mean_cpu(series[roles["spawner"]])

    assert cpu > 70, cpu
    assert grandchild > 70, grandchild
    assert root < 15, root
    assert spawner < 15, spawner
    assert io < cpu

    io_written = total_bytes(series[roles["io"]], "write_bytes_per_s")
    io_read = total_bytes(series[roles["io"]], "read_bytes_per_s")
    cpu_written = total_bytes(series[roles["cpu"]], "write_bytes_per_s")
    assert io_written > 32 * MIB, io_written
    assert io_read > 8 * MIB, io_read
    assert cpu_written < 1 * MIB

    labels = {record["pid"]: record["cmdline"] for record in records if record["type"] == "process"}
    assert "--role io" in labels[roles["io"]]
    assert "--role cpu" in labels[roles["grandchild"]]


def test_agent_reports_missing_root(agent_binary, tmp_path):
    out = tmp_path / "resources.jsonl"
    code = subprocess.run([str(agent_binary), "--pid", "999999999", "--out", str(out)]).returncode
    records = read_records(out)
    assert code == 3
    assert records[-1]["reason"] == "root_not_found"


def test_agent_rejects_bad_arguments(agent_binary):
    assert subprocess.run([str(agent_binary)], capture_output=True).returncode == 2
    assert subprocess.run([str(agent_binary), "--pid", "1", "--interval-ms", "1"], capture_output=True).returncode == 2


def test_agent_stops_cleanly_on_sigterm(agent_binary, tmp_path):
    out = tmp_path / "resources.jsonl"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    agent = subprocess.Popen([str(agent_binary), "--pid", str(sleeper.pid), "--interval-ms", "100",
                              "--out", str(out)])
    try:
        time.sleep(0.6)
        agent.send_signal(signal.SIGTERM)
        assert agent.wait(timeout=5) == 0
    finally:
        sleeper.kill()
        sleeper.wait()
    records = read_records(out)
    assert records[-1]["type"] == "end"
    assert records[-1]["reason"] == "signal"
    assert records[-1]["samples"] >= 3
