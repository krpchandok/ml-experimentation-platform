import os

from mlplat.config import flatten
from mlplat.run_store import read_jsonl

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
DEFAULT_EXPERIMENT = "mlplat"
ARTIFACT_PATH = "mlplat"
METRIC_BATCH = 1000
PARAM_BATCH = 100
PARAM_VALUE_LIMIT = 6000
RESERVED_METRIC_KEYS = {"t_mono", "t_wall", "step"}
MLFLOW_STATUS = {"completed": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}
QUICK_FAILURE_ENV = {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "10"}


def resolve_tracking_uri(mode, runs_root):
    if mode == "off":
        return None
    uri = os.environ.get(TRACKING_URI_ENV)
    if uri:
        return uri
    if mode == "on":
        return f"sqlite:///{runs_root / 'mlflow.db'}"
    return None


def training_metrics(records, Metric):
    metrics = []
    for index, record in enumerate(records):
        step = record.get("step")
        step = int(step) if isinstance(step, (int, float)) else index
        timestamp = int(record.get("t_wall", 0) * 1000)
        for key, value in record.items():
            if key in RESERVED_METRIC_KEYS or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics.append(Metric(key, float(value), timestamp, step))
    return metrics


def resource_metrics(records, Metric):
    metrics = []
    for record in records:
        if record.get("type") != "sample":
            continue
        timestamp = int(record["t_wall"] * 1000)
        step = int(record["seq"])
        tree = record.get("tree", {})
        if tree.get("cpu_pct") is not None:
            metrics.append(Metric("resource.tree_cpu_pct", float(tree["cpu_pct"]), timestamp, step))
        if tree.get("rss_kb") is not None:
            metrics.append(Metric("resource.tree_rss_mb", tree["rss_kb"] / 1024.0, timestamp, step))
        gpu = record.get("gpu")
        if isinstance(gpu, dict):
            for device in gpu.get("devices", []):
                if device.get("util_pct") is not None:
                    metrics.append(Metric(f"resource.gpu{device['index']}_util_pct", float(device["util_pct"]),
                                          timestamp, step))
    return metrics


def summary_metrics(summary, Metric, timestamp):
    if not summary:
        return []
    totals = summary["totals"]
    values = {
        "summary.cpu_seconds": totals.get("cpu_seconds"),
        "summary.avg_cores": totals.get("avg_cores_window"),
        "summary.peak_cores": totals.get("peak_cores"),
        "summary.peak_rss_mb": totals.get("peak_rss_mb"),
        "summary.step_time_median_s": totals.get("step_time_median_s"),
        "summary.steps_per_sec": totals.get("steps_per_sec"),
        "summary.cpu_seconds_per_step": totals.get("cpu_seconds_per_step"),
        "summary.gpu_util_mean_pct": (summary.get("gpu") or {}).get("util_mean_pct"),
    }
    return [Metric(key, float(value), timestamp, 0) for key, value in values.items() if value is not None]


def mirror(run, meta, tracking_uri, summary=None):
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)

    git = meta.get("git") or {}
    tags = {
        "mlplat.run_id": meta["run_id"],
        "mlplat.config_hash": meta["config_hash"],
        "mlplat.status": meta["status"],
        "mlplat.command": meta["command_line"][:PARAM_VALUE_LIMIT],
        "mlplat.host": meta["host"]["hostname"],
    }
    if git.get("commit"):
        tags["mlflow.source.git.commit"] = git["commit"]
    if summary:
        tags["mlplat.verdict"] = summary["verdict"]["primary"]
        tags["mlplat.verdict_severity"] = summary["verdict"]["severity"]
    mlflow_run = client.create_run(experiment_id, run_name=meta["name"], tags=tags,
                                   start_time=int(meta["start_wall"] * 1000))
    run_id = mlflow_run.info.run_id

    params = [Param(key, str(value)[:PARAM_VALUE_LIMIT]) for key, value in flatten(meta.get("config") or {}).items()]
    for start in range(0, len(params), PARAM_BATCH):
        client.log_batch(run_id, params=params[start:start + PARAM_BATCH])

    metrics = training_metrics(read_jsonl(run.metrics_path), Metric)
    metrics += resource_metrics(read_jsonl(run.resources_path), Metric)
    metrics += summary_metrics(summary, Metric, int(meta["end_wall"] * 1000))
    for start in range(0, len(metrics), METRIC_BATCH):
        client.log_batch(run_id, metrics=metrics[start:start + METRIC_BATCH])

    meta["mlflow"] = {"run_id": run_id, "experiment_id": experiment_id, "tracking_uri": tracking_uri}
    run.write_meta(meta)
    for path in (run.meta_path, run.metrics_path, run.resources_path, run.output_path, run.summary_path):
        if path.exists():
            client.log_artifact(run_id, str(path), artifact_path=ARTIFACT_PATH)
    client.set_terminated(run_id, status=MLFLOW_STATUS.get(meta["status"], "FAILED"),
                          end_time=int(meta["end_wall"] * 1000))
    return run_id


def mirror_if_enabled(run, meta, mode, warn, summary=None):
    tracking_uri = resolve_tracking_uri(mode, run.path.parent)
    if tracking_uri is None:
        return None
    for key, value in QUICK_FAILURE_ENV.items():
        os.environ.setdefault(key, value)
    try:
        return mirror(run, meta, tracking_uri, summary)
    except ImportError:
        warn("mlflow is not installed; run kept in the local run store only")
    except Exception as error:
        warn(f"could not mirror run to MLflow at {tracking_uri}: {error}")
    meta["mlflow"] = {"error": "mirror failed", "tracking_uri": tracking_uri}
    run.write_meta(meta)
    return None
