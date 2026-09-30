from dataclasses import dataclass
from .model import Models, Datasets

@dataclass
class ExperimentConfig:
    model: Models
    dataset: Datasets
    hyperparameters: dict ## C, iterations, random_seed

class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        dataset = self.config.model

    def split_data(self):
        split = self.config.hyperparameters["split"]

    def create_model(self):
        pass

    def train(self):
        pass

    def evaluate(self): 
        pass