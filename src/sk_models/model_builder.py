from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_logistic_regression(hyperparameters):
    return LogisticRegression(
        C=hyperparameters.get("C", 1.0),
        max_iter=hyperparameters.get("iterations", 100),
        random_state=hyperparameters.get("random_seed", 42),
        solver=hyperparameters.get("solver", "lbfgs"),
    )

def build_random_forest(hyperparameters):
    return RandomForestClassifier(
        n_estimators=hyperparameters.get("n_estimators", 100),
        max_depth=hyperparameters.get("depth", 5),
        random_state=hyperparameters.get("random_seed", 42),
    )