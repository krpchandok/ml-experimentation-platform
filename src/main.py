from src.experiment import ExperimentConfig, ExperimentRunner
from src.model import Models, Datasets


def main():
    hyperparameters = {
        "C": 1.0,
        "iterations": 100,
        "random_seed": 42,
        "split": 0.2,
        "solver": "lbfgs",
    }
    config = ExperimentConfig(Models.LOGISTIC_REGRESSION, Datasets.IRIS, hyperparameters)
    result = ExperimentRunner(config).run()
    print(result)


if __name__ == "__main__":
    main()
