from abc import ABC, abstractmethod
from pathlib import Path
import json
import psycopg
from src.experiment import save_result


class ResultStore(ABC):
    def __init__(self, result):
        self.result = result

    @abstractmethod
    def save(self):
        pass

class JSONResultStore(ResultStore):
    def __init__(self, result, result_dir):
        super().__init__(result)
        self.result_dir = result_dir

    def save(self):
        return save_result(self.result, self.result_dir)

class PostgresResultStore(ResultStore):
    def __init__(self, result, host="localhost", port=5432, dbname="ml_platform", user="mluser", password="mlpassword"):
        super().__init__(result)
        self.conn = psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    experiment_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metrics JSONB NOT NULL,
                    runtime DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
        self.conn.commit()

    def save(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_results (experiment_id, model, dataset, metrics, runtime)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                (
                    self.result.experiment_id,
                    self.result.model,
                    self.result.dataset,
                    json.dumps(self.result.metrics),
                    self.result.runtime,
                ),
            )
        self.conn.commit()
