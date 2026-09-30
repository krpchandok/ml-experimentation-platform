from confluent_kafka import Consumer
from pathlib import Path
from dotenv import load_dotenv
import json
import os
import sys

from src.jobs import TrainingJob, Status
from src.experiment import ExperimentRunner, config_from_dict
from src.experiment_tracker import MLflowTracker
from src.result_store import PostgresResultStore
from src.job_store import PostgresJobStore
from src.messaging.producer import TRAINING_JOBS_TOPIC

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


class JobConsumer:
    def __init__(self, bootstrap_servers=None, topic=TRAINING_JOBS_TOPIC, group_id="ml-platform-workers"):
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })
        self.consumer.subscribe([topic])
        self.tracker = MLflowTracker(os.environ["MLFLOW_TRACKING_URI"], os.environ["MLFLOW_EXPERIMENT_NAME"])
        self.job_store = PostgresJobStore()

    def run(self):
        print("Waiting for training jobs...")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue

                self._handle_message(msg)
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()

    def _handle_message(self, msg):
        payload = json.loads(msg.value())

        job = TrainingJob(
            job_id=payload["job_id"],
            experiment_id=payload["experiment_id"],
            config=payload["config"],
            status=Status(payload["status"]),
        )

        try:
            # Worker has actually started the job
            self.job_store.update_status(
                job.job_id,
                Status.RUNNING.value
            )

            config = config_from_dict(job.config)

            result = ExperimentRunner(
                job.experiment_id,
                config,
                self.tracker
            ).run()

            PostgresResultStore(result).save()

            # Everything succeeded
            self.job_store.update_status(
                job.job_id,
                Status.COMPLETED.value
            )

            self.consumer.commit(msg)

            print(f"Job {job.job_id} completed: {result.metrics}")

        except Exception as e:
            self.job_store.update_status(
                job.job_id,
                Status.FAILED.value
            )

            print(f"Job {job.job_id} failed: {e}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    JobConsumer().run()
