from sklearn.linear_model import LogisticRegression


def build_logistic_regression(hyperparameters):
    return LogisticRegression(
        C=hyperparameters.get("C", 1.0),
        max_iter=hyperparameters.get("iterations", 100),
        random_state=hyperparameters.get("random_seed", 42),
        solver=hyperparameters.get("solver", "lbfgs"),
    )
