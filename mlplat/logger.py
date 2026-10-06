import atexit
import json
import math
import os
import threading
import time
import warnings
from pathlib import Path

RUN_DIR_ENV = "MLPLAT_RUN_DIR"
METRICS_FILE = "metrics.jsonl"
FLUSH_INTERVAL_S = 1.0


def _plain(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class MetricsLogger:
    def __init__(self, path, flush_interval_s=FLUSH_INTERVAL_S):
        self.path = Path(path)
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._buffer = []
        self._file = None
        self._last_flush = time.monotonic()
        os.register_at_fork(after_in_child=self._reset_after_fork)
        atexit.register(self.close)

    def log(self, step=None, **metrics):
        now = time.monotonic()
        record = {"t_mono": now, "t_wall": time.time()}
        if step is not None:
            record["step"] = _plain(step)
        for name, value in metrics.items():
            record[name] = _plain(value)
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._buffer.append(line)
            if now - self._last_flush >= self.flush_interval_s:
                self._flush_locked(now)

    def flush(self):
        with self._lock:
            self._flush_locked(time.monotonic())

    def close(self):
        with self._lock:
            self._flush_locked(time.monotonic())
            if self._file is not None:
                self._file.close()
                self._file = None

    def _flush_locked(self, now):
        self._last_flush = now
        if not self._buffer:
            return
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        self._file.write("\n".join(self._buffer) + "\n")
        self._file.flush()
        self._buffer.clear()

    def _reset_after_fork(self):
        self._lock = threading.Lock()
        self._buffer = []
        self._file = None
        self._last_flush = time.monotonic()


_default_logger = None
_default_resolved = False


def _resolve_default():
    global _default_logger, _default_resolved
    if not _default_resolved:
        _default_resolved = True
        run_dir = os.environ.get(RUN_DIR_ENV)
        if run_dir:
            _default_logger = MetricsLogger(Path(run_dir) / METRICS_FILE)
        else:
            warnings.warn(f"{RUN_DIR_ENV} is not set; mlplat.log calls are ignored", stacklevel=3)
    return _default_logger


def log(step=None, **metrics):
    logger = _default_logger if _default_resolved else _resolve_default()
    if logger is not None:
        logger.log(step=step, **metrics)


def flush():
    logger = _default_logger if _default_resolved else _resolve_default()
    if logger is not None:
        logger.flush()


def run_dir():
    value = os.environ.get(RUN_DIR_ENV)
    return Path(value) if value else None
