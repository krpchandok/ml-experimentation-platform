from dataclasses import dataclass
from enum import Enum
import uuid
import yaml
from kafka.producer import JobProducer

class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TrainingJob:
    job_id: str
    experiment_id: str
    config: dict
    status: Status = Status.PENDING

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "config": self.config,
            "status": self.status.value,
        }


def submit_job(config_path, producer=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    job = TrainingJob(
        job_id=str(uuid.uuid4()),
        experiment_id=str(uuid.uuid4()),
        config=config,
    )

    (producer or JobProducer()).send(job)
    return job

