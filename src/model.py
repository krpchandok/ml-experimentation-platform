from enum import Enum
from sk_models.model_builder import build_logistic_regression, build_random_forest


class Models(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


class Datasets(Enum):
    IRIS = "iris"


MODEL_BUILDERS = {
    Models.LOGISTIC_REGRESSION: build_logistic_regression,
    Models.RANDOM_FOREST: build_random_forest
}
