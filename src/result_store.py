from abc import ABC, abstractmethod
from pathlib import Path
from experiment import save_result


class ResultStore(ABC):
    def __init__(self, result):
        self.result = result
    
    @abstractmethod
    def save(self):
        pass

class JSONResultStore(ResultStore):
    def __init__(self, result, result_dir):
        super().__init__()
        self.result_dir = result_dir

    def save(self):
        save_result(self.result_dir + self.result)
