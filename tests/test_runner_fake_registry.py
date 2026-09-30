import pandas as pd

from src.experiment import ExperimentRunner, ExperimentConfig
from src.model import Models
from src.datasets.registry import DatasetRegistry


class FakeDatasetRegistry(DatasetRegistry):
    def __init__(self, df, manifest):
        self.df = df
        self.manifest = manifest

    def latest(self, name):
        return self.manifest

    def get(self, name, version):
        return self.manifest if version == self.manifest["delta_version"] else None

    def load(self, name, version):
        return self.df

    def list_datasets(self):
        return [self.manifest["name"]]


class FakeTracker:
    def __init__(self):
        self.tags = {}
        self.params = {}
        self.metrics = {}
        self.model = None

    def start_run(self, run_name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_model(self, model, name):
        self.model = model

    def end_run(self, status="FINISHED"):
        pass


def make_dataframe():
    return pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.3, 5.8, 6.1, 5.5] * 5,
        "sepal_width": [3.5, 3.0, 3.3, 2.7, 2.8, 2.4] * 5,
        "petal_length": [1.4, 1.4, 6.0, 5.1, 4.7, 3.8] * 5,
        "petal_width": [0.2, 0.2, 2.5, 1.9, 1.2, 1.1] * 5,
        "species": ["setosa", "setosa", "virginica", "virginica", "versicolor", "versicolor"] * 5,
    })


def test_experiment_runner_with_fake_registry():
    manifest = {
        "name": "iris",
        "delta_version": 1,
        "rows_in": 30,
        "rows_out": 30,
        "rows_rejected": 0,
        "schema_json": "{}",
        "content_hash": "fake",
    }

    registry = FakeDatasetRegistry(make_dataframe(), manifest)
    tracker = FakeTracker()

    config = ExperimentConfig(
        model=Models.LOGISTIC_REGRESSION,
        dataset="iris",
        hyperparameters={"C": 1.0, "iterations": 100, "random_seed": 42, "split": 0.2},
    )

    result = ExperimentRunner("test-experiment", config, tracker, registry).run()

    assert result.dataset_version == 1
    assert "accuracy" in result.metrics
    assert tracker.tags["dataset_version"] == "1"
    assert tracker.tags["content_hash"] == "fake"
    assert tracker.model is not None
