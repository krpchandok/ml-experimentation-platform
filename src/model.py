from enum import Enum
from src.sk_models.logistic_regression import build_logistic_regression


class Models(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"


class Datasets(Enum):
    IRIS = "iris"


MODEL_BUILDERS = {
    Models.LOGISTIC_REGRESSION: build_logistic_regression,
}
