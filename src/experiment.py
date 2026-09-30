from dataclasses import dataclass
from .model import Models, Datasets, MODEL_BUILDERS
from .trainer import SklearnTrainer
import uuid
from time import perf_counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ExperimentConfig:
    model: Models
    dataset: Datasets
    hyperparameters: dict ## C, iterations, random_seed


class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        experiment_id = str(uuid.uuid4())

        start = perf_counter()

        self.load_dataset()
        self.split_data()
        self.create_model()
        self.train()
        metrics = self.evaluate()

        runtime = perf_counter() - start

        return ExperimentResult(
            experiment_id=experiment_id,
            model=self.config.model.value,
            dataset=self.config.dataset.value,
            metrics=metrics,
            runtime=runtime,
        )

    def load_dataset(self):
        if self.config.dataset == Datasets.IRIS:
            self.data = load_iris()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

    def split_data(self):
        X, y = self.data.data, self.data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.hyperparameters.get("split", 0.2),
            random_state=self.config.hyperparameters.get("random_seed", 42),
        )

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def create_model(self):
        builder = MODEL_BUILDERS.get(self.config.model)
        if builder is None:
            raise ValueError(f"Unsupported model: {self.config.model}")
        self.model = builder(self.config.hyperparameters)

    def train(self):
        self.trainer = SklearnTrainer(self.model, self.X_train, self.y_train, self.X_test, self.y_test)
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()


@dataclass
class ExperimentResult:
    experiment_id: str
    model: str
    dataset: str
    metrics: dict
    runtime: float
