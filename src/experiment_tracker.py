from abc import ABC, abstractmethod
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, run_name):
        pass

    @abstractmethod
    def set_tags(self, tags):
        pass

    @abstractmethod
    def log_params(self, params):
        pass

    @abstractmethod
    def log_metrics(self, metrics):
        pass

    @abstractmethod
    def log_model(self, model, name):
        pass

    @abstractmethod
    def end_run(self):
        pass


class MLflowTracker(ExperimentTracker):
    def __init__(self):
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    def start_run(self, run_name):
        return mlflow.start_run(run_name=run_name)

    def set_tags(self, tags):
        mlflow.set_tags(tags)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

    def log_model(self, model, name):
        mlflow.sklearn.log_model(model, name=name, skops_trusted_types=["sklearn.tree._tree.Tree"])

    def end_run(self):
        mlflow.end_run()

    def get_best_model(self, metric="accuracy", ascending=False):
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(order_by=[f"metrics.{metric} {order}"], max_results=1)
        return None if runs.empty else runs.iloc[0]
