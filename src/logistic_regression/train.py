from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from experiment import ExperimentConfig
from src.model import Models, Datasets

hyperparameters = {
    "C": 1.0,
    "max_iter": 100,
    "random_seed": 42,
    "split": 0.2,
    "solver": "lbfgs"
}

config = ExperimentConfig(Models.LOGISTIC_REGRESSION, Datasets.IRIS, hyperparameters)

model = LogisticRegression(
    C=hyperparameters["C"],
    solver=hyperparameters["solver"],
    max_iter=hyperparameters["max_iter"],
)
data = load_iris()
X = data.data[data.target != 2]  # Filter out class 2 for binary classification
y = data.target[data.target != 2]  # Filter out class 2 for binary classification

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=hyperparameters["split"],
    random_state=hyperparameters["random_seed"],
)

scaler = StandardScaler()
## feature scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## train
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print ("\nClassification Report:\n", classification_report(y_test, y_pred))

