from abc import ABC, abstractmethod

import mlflow
import mlflow.sklearn


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
    def end_run(self, status):
        pass


class MLflowTracker(ExperimentTracker):
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

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

    def end_run(self, status="FINISHED"):
        mlflow.end_run(status=status)