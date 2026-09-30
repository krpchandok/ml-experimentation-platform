from confluent_kafka import Producer
from pathlib import Path
from dotenv import load_dotenv
import json
import os

TRAINING_JOBS_TOPIC = "training-jobs"


class JobProducer:
    def __init__(self, bootstrap_servers=None, topic=TRAINING_JOBS_TOPIC):
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
        self.topic = topic
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        })

    def send(self, job):
        self.producer.produce(
            self.topic,
            key=job.job_id,
            value=json.dumps(job.to_dict()),
            callback=self._delivery_report,
        )
        self.producer.flush()

    def _delivery_report(self, err, msg):
        if err is not None:
            print(f"Delivery failed for job {msg.key()}: {err}")
        else:
            print(f"Job {msg.key()} delivered to {msg.topic()} [{msg.partition()}]")
