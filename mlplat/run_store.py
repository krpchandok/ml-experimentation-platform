import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR_ENV = "MLPLAT_RUNS_DIR"
META_FILE = "meta.json"
METRICS_FILE = "metrics.jsonl"
RESOURCES_FILE = "resources.jsonl"
OUTPUT_FILE = "output.log"
AGENT_LOG_FILE = "agent.log"
SUMMARY_FILE = "summary.json"
CONFIG_COPY_FILE = "config.yaml"


def default_runs_dir():
    return Path(os.environ.get(RUNS_DIR_ENV) or Path.cwd() / "runs")


def new_run_id(now=None):
    stamp = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return f"{stamp:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_json_atomic(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


class Run:
    def __init__(self, path):
        self.path = Path(path)

    @property
    def run_id(self):
        return self.path.name

    @property
    def meta_path(self):
        return self.path / META_FILE

    @property
    def metrics_path(self):
        return self.path / METRICS_FILE

    @property
    def resources_path(self):
        return self.path / RESOURCES_FILE

    @property
    def output_path(self):
        return self.path / OUTPUT_FILE

    @property
    def agent_log_path(self):
        return self.path / AGENT_LOG_FILE

    @property
    def summary_path(self):
        return self.path / SUMMARY_FILE

    def read_summary(self):
        if not self.summary_path.exists():
            return None
        with open(self.summary_path, encoding="utf-8") as handle:
            return json.load(handle)

    def read_meta(self):
        with open(self.meta_path, encoding="utf-8") as handle:
            return json.load(handle)

    def write_meta(self, meta):
        write_json_atomic(self.meta_path, meta)


class RunStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_runs_dir()

    def create(self):
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            path = self.root / new_run_id()
            try:
                path.mkdir()
                return Run(path)
            except FileExistsError:
                continue

    def get(self, run_id):
        exact = self.root / run_id
        if (exact / META_FILE).exists():
            return Run(exact)
        matches = [run for run in self.list() if run.run_id.startswith(run_id) or run.read_meta().get("name") == run_id]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"no run matches {run_id!r}")
        raise KeyError(f"{run_id!r} is ambiguous: {', '.join(run.run_id for run in matches)}")

    def list(self):
        if not self.root.exists():
            return []
        runs = [Run(path) for path in self.root.iterdir() if (path / META_FILE).exists()]
        return sorted(runs, key=lambda run: run.run_id)
