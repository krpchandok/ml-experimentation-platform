from pathlib import Path
from experiment import ExperimentRunner, load_config, save_result

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    configLR = load_config(CONFIG_DIR / "logistic_regression.yaml")
    resultLR = ExperimentRunner(configLR).run()
    save_result(resultLR, RESULTS_DIR)

    configRF = load_config(CONFIG_DIR / "random_forest.yaml")
    resultRF = ExperimentRunner(configRF).run()
    save_result(resultRF, RESULTS_DIR)

    print(f"\nLogistic Regression\n, {resultLR}")
    print(f"\nRandom Forest\n, {resultRF}")

if __name__ == "__main__":
    main()
