from pathlib import Path
import sys
from experiment import ExperimentRunner, load_config, save_result
from experiment_tracker import MLflowTracker

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
RESULTS_DIR = ROOT_DIR / "results"


def main():
    tracker = MLflowTracker()

    configLR = load_config(CONFIG_DIR / "logistic_regression.yaml")
    resultLR = ExperimentRunner(configLR, tracker).run()
    save_result(resultLR, RESULTS_DIR)

    configRF = load_config(CONFIG_DIR / "random_forest.yaml")
    resultRF = ExperimentRunner(configRF, tracker).run()
    save_result(resultRF, RESULTS_DIR)

    print(f"\nLogistic Regression\n, {resultLR}")
    print(f"\nRandom Forest\n, {resultRF}")

if __name__ == "__main__":
    main()
