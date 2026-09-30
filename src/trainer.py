from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report


class Trainer(ABC):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class SklearnTrainer(Trainer):
    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

        return {"accuracy": accuracy}
