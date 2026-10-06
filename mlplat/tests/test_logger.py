import json
import os
import subprocess
import sys
import textwrap
import time

from mlplat.logger import MetricsLogger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def read_lines(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def run_script(source, env_extra, tmp_path):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(source))
    env = {key: value for key, value in os.environ.items() if key != "MLPLAT_RUN_DIR"}
    env.update(PYTHONPATH=REPO_ROOT, **env_extra)
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=60)


def test_log_writes_timestamped_records(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.jsonl", flush_interval_s=3600)
    before = time.monotonic()
    logger.log(step=1, loss=0.5, accuracy=FakeTensor(0.75), note="warmup")
    logger.log(step=2, loss=float("nan"), weird=object)
    assert not (tmp_path / "metrics.jsonl").exists()
    logger.flush()
    first, second = read_lines(tmp_path / "metrics.jsonl")
    assert first["step"] == 1 and first["loss"] == 0.5 and first["accuracy"] == 0.75 and first["note"] == "warmup"
    assert before <= first["t_mono"] <= time.monotonic()
    assert abs(first["t_wall"] - time.time()) < 5
    assert second["loss"] is None
    assert isinstance(second["weird"], str)
    logger.close()


def test_log_flushes_after_interval(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.jsonl", flush_interval_s=0)
    logger.log(step=0, loss=1.0)
    assert len(read_lines(tmp_path / "metrics.jsonl")) == 1
    logger.close()


def test_buffered_lines_are_written_at_exit(tmp_path):
    result = run_script("""
        import mlplat
        for step in range(3):
            mlplat.log(step=step, loss=step * 0.1)
    """, {"MLPLAT_RUN_DIR": str(tmp_path)}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert [record["step"] for record in read_lines(tmp_path / "metrics.jsonl")] == [0, 1, 2]


def test_log_without_run_dir_is_a_warning_and_noop(tmp_path):
    result = run_script("""
        import mlplat
        mlplat.log(step=0, loss=1.0)
        mlplat.log(step=1, loss=0.5)
        print(mlplat.run_dir())
    """, {}, tmp_path)
    assert result.returncode == 0
    assert result.stderr.count("MLPLAT_RUN_DIR is not set") == 1
    assert result.stdout.strip() == "None"


def test_forked_children_do_not_duplicate_parent_buffer(tmp_path):
    result = run_script("""
        import os
        import mlplat
        mlplat.log(step=0, source="parent")
        pid = os.fork()
        if pid == 0:
            mlplat.log(step=1, source="child")
            mlplat.flush()
            os._exit(0)
        os.waitpid(pid, 0)
        mlplat.flush()
    """, {"MLPLAT_RUN_DIR": str(tmp_path)}, tmp_path)
    assert result.returncode == 0, result.stderr
    sources = sorted(record["source"] for record in read_lines(tmp_path / "metrics.jsonl"))
    assert sources == ["child", "parent"]


def test_log_call_is_cheap(tmp_path):
    calls = 20000
    logger = MetricsLogger(tmp_path / "metrics.jsonl")
    start = time.perf_counter()
    for step in range(calls):
        logger.log(step=step, loss=0.1, accuracy=0.9)
    per_call_us = (time.perf_counter() - start) / calls * 1e6
    logger.close()
    print(f"mlplat.log cost: {per_call_us:.2f} us/call")
    assert per_call_us < 50
    assert len(read_lines(tmp_path / "metrics.jsonl")) == calls
