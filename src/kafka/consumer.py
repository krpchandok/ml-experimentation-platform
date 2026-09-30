from confluent_kafka import Consumer
from pathlib import Path
from dotenv import load_dotenv
import json
import os
import sys

from jobs import TrainingJob, Status
from experiment import ExperimentRunner, config_from_dict
from experiment_tracker import MLflowTracker
from result_store import PostgresResultStore
from kafka.producer import TRAINING_JOBS_TOPIC

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

        print(f"Running job {job.job_id} ({job.config['model']} / {job.config['dataset']})")

        try:
            config = config_from_dict(job.config)
            result = ExperimentRunner(config, self.tracker).run()
            PostgresResultStore(result).save()
            print(f"Job {job.job_id} completed: {result.metrics}")
            self.consumer.commit(msg)
        except Exception as e:
            print(f"Job {job.job_id} failed, will retry on restart: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    JobConsumer().run()
