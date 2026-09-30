from pathlib import Path
from dotenv import load_dotenv
import os
import sys
import uuid
from src.experiment import ExperimentRunner, load_config, save_result
from src.experiment_tracker import MLflowTracker
from src.datasets.registry import DatabricksDatasetRegistry

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
RESULTS_DIR = ROOT_DIR / "results"

load_dotenv(ROOT_DIR / ".env")


def main():
    tracker = MLflowTracker(os.environ["MLFLOW_TRACKING_URI"], os.environ["MLFLOW_EXPERIMENT_NAME"])
    dataset_registry = DatabricksDatasetRegistry()

    configLR = load_config(CONFIG_DIR / "logistic_regression.yaml")
    resultLR = ExperimentRunner(str(uuid.uuid4()), configLR, tracker, dataset_registry).run()
    save_result(resultLR, RESULTS_DIR)

    configRF = load_config(CONFIG_DIR / "random_forest.yaml")
    resultRF = ExperimentRunner(str(uuid.uuid4()), configRF, tracker, dataset_registry).run()
    save_result(resultRF, RESULTS_DIR)

    print(f"\nLogistic Regression\n, {resultLR}")
    print(f"\nRandom Forest\n, {resultRF}")

if __name__ == "__main__":
    main()
