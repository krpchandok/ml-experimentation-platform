import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from mlplat import mlflow_mirror
from mlplat.analyzer import analyze_run
from mlplat.config import config_hash, load_config
from mlplat.host_info import collect_git_info, collect_host_info
from mlplat.logger import RUN_DIR_ENV
from mlplat.run_store import CONFIG_COPY_FILE, RunStore

AGENT_ENV = "MLPLAT_AGENT"
RUN_ID_ENV = "MLPLAT_RUN_ID"
CONFIG_PATH_ENV = "MLPLAT_CONFIG_PATH"
REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_AGENT = REPO_ROOT / "agent" / "build" / "mlplat-agent"
AGENT_STOP_TIMEOUT_S = 10
LEFTOVER_GRACE_S = 3
META_SCHEMA = 1
FORWARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def warn(message):
    print(f"mlplat: {message}", file=sys.stderr, flush=True)


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def find_agent(explicit=None):
    pinned = explicit or os.environ.get(AGENT_ENV)
    candidates = [pinned] if pinned else [shutil.which("mlplat-agent"), BUNDLED_AGENT]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


def child_environment(run, config_path):
    env = dict(os.environ)
    env[RUN_DIR_ENV] = str(run.path)
    env[RUN_ID_ENV] = run.run_id
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    if config_path:
        env[CONFIG_PATH_ENV] = str(Path(config_path).resolve())
    return env


def own_process_group():
    if sys.version_info >= (3, 11):
        return {"process_group": 0}
    return {"start_new_session": True}


def exit_code_from(returncode):
    return 128 - returncode if returncode < 0 else returncode


class OutputTee(threading.Thread):
    def __init__(self, source, log_path):
        super().__init__(daemon=True)
        self.source = source
        self.log_path = log_path

    def run(self):
        with open(self.log_path, "ab") as log:
            descriptor = self.source.fileno()
            while chunk := os.read(descriptor, 65536):
                log.write(chunk)
                log.flush()
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                except (OSError, ValueError):
                    pass
        self.source.close()


class SignalForwarder:
    def __init__(self):
        self.process_group = None
        self.received = []
        self.previous = {}

    def __enter__(self):
        for signum in FORWARDED_SIGNALS:
            self.previous[signum] = signal.signal(signum, self.handle)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

    def attach(self, process_group):
        self.process_group = process_group
        if self.received:
            self.forward(self.received[0])

    def handle(self, signum, frame):
        self.received.append(signum)
        self.forward(signum if len(self.received) == 1 else signal.SIGKILL)

    def forward(self, signum):
        if self.process_group is None:
            return
        try:
            os.killpg(self.process_group, signum)
        except ProcessLookupError:
            pass

    @property
    def interrupted(self):
        return bool(self.received)


def group_alive(process_group):
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_leftovers(process_group):
    if not group_alive(process_group):
        return False
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + LEFTOVER_GRACE_S
    while time.monotonic() < deadline and group_alive(process_group):
        time.sleep(0.05)
    if group_alive(process_group):
        os.killpg(process_group, signal.SIGKILL)
    return True


def start_agent(agent_path, pid, interval_ms, run):
    if agent_path is None:
        warn("resource agent not found; set MLPLAT_AGENT or build agent/; continuing without resource data")
        return None, {"available": False, "error": "agent binary not found"}
    log = open(run.agent_log_path, "wb")
    try:
        process = subprocess.Popen(
            [str(agent_path), "--pid", str(pid), "--interval-ms", str(interval_ms), "--out", str(run.resources_path)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, **own_process_group(),
        )
    except OSError as error:
        warn(f"could not start resource agent: {error}; continuing without resource data")
        return None, {"available": False, "path": str(agent_path), "error": str(error)}
    finally:
        log.close()
    return process, {"available": True, "path": str(agent_path), "interval_ms": interval_ms, "pid": process.pid}


def stop_agent(process, info):
    if process is None:
        return
    try:
        info["exit_code"] = process.wait(timeout=AGENT_STOP_TIMEOUT_S)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        info["exit_code"] = process.wait(timeout=AGENT_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        info["exit_code"] = process.wait()
        info["error"] = "agent did not stop and was killed"


def status_for(exit_code, interrupted):
    if interrupted:
        return "interrupted"
    return "completed" if exit_code == 0 else "failed"


def launch(name, command, config_path=None, runs_dir=None, agent_path=None, interval_ms=1000, mlflow_mode="auto",
           analyze=True):
    config = load_config(config_path) if config_path else None
    digest, digest_source = config_hash(config, command)
    run = RunStore(runs_dir).create()
    if config_path:
        shutil.copyfile(config_path, run.path / CONFIG_COPY_FILE)

    start_wall = time.time()
    meta = {
        "schema": META_SCHEMA,
        "run_id": run.run_id,
        "name": name,
        "status": "running",
        "command": list(command),
        "command_line": shlex.join(command),
        "cwd": os.getcwd(),
        "config": config,
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "config_hash": digest,
        "config_hash_source": digest_source,
        "git": collect_git_info(os.getcwd()),
        "host": collect_host_info(),
        "start_time": iso(start_wall),
        "start_wall": start_wall,
        "start_mono": time.monotonic(),
        "end_time": None,
        "exit_code": None,
    }
    run.write_meta(meta)
    print(f"mlplat: run {run.run_id} ({name}) -> {run.path}", file=sys.stderr, flush=True)

    with SignalForwarder() as forwarder:
        try:
            process = subprocess.Popen(command, env=child_environment(run, config_path), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, **own_process_group())
        except OSError as error:
            meta.update(status="failed", exit_code=127 if isinstance(error, FileNotFoundError) else 126,
                        error=str(error), agent={"available": False, "error": "training command did not start"})
            return finish(run, meta, mlflow_mode, analyze)

        forwarder.attach(process.pid)
        meta["pid"] = process.pid
        agent_process, meta["agent"] = start_agent(find_agent(agent_path), process.pid, interval_ms, run)
        run.write_meta(meta)

        tee = OutputTee(process.stdout, run.output_path)
        tee.start()
        returncode = process.wait()
        meta["leftover_processes_terminated"] = terminate_leftovers(process.pid)
        tee.join()
        stop_agent(agent_process, meta["agent"])

    exit_code = exit_code_from(returncode)
    meta.update(status=status_for(exit_code, forwarder.interrupted), exit_code=exit_code,
                signals_received=[signal.Signals(signum).name for signum in forwarder.received])
    return finish(run, meta, mlflow_mode, analyze)


def analyze_safely(run):
    try:
        summary = analyze_run(run)
    except Exception as error:
        warn(f"analysis failed: {error}")
        return None
    verdict = summary["verdict"]
    print(f"mlplat: verdict {verdict['primary']}: {verdict['title']}", file=sys.stderr, flush=True)
    return summary


def finish(run, meta, mlflow_mode, analyze):
    end_wall = time.time()
    meta["end_time"] = iso(end_wall)
    meta["end_wall"] = end_wall
    meta["end_mono"] = time.monotonic()
    meta["duration_s"] = round(end_wall - meta["start_wall"], 3)
    run.write_meta(meta)
    summary = analyze_safely(run) if analyze else None
    mlflow_mirror.mirror_if_enabled(run, meta, mlflow_mode, warn, summary)
    print(f"mlplat: run {run.run_id} {meta['status']} (exit {meta['exit_code']}, {meta['duration_s']:.1f}s)",
          file=sys.stderr, flush=True)
    return meta["exit_code"]
