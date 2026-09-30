from abc import ABC, abstractmethod
import json
import psycopg


class JobStore(ABC):
    @abstractmethod
    def create_job(self, job):
        pass

    @abstractmethod
    def update_status(self, job_id, status):
        pass

    @abstractmethod
    def get_job(self, job_id):
        pass


class PostgresJobStore(JobStore):
    def __init__(
        self,
        host="localhost",
        port=5432,
        dbname="ml_platform",
        user="mluser",
        password="mlpassword"
    ):
        self.conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id UUID PRIMARY KEY,
                    experiment_id UUID NOT NULL,
                    config JSONB NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
        self.conn.commit()

    def create_job(self, job):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_jobs
                    (job_id, experiment_id, config, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_id) DO NOTHING
                """,
                (
                    job.job_id,
                    job.experiment_id,
                    json.dumps(job.config),
                    job.status.value,
                ),
            )

        self.conn.commit()

    def update_status(self, job_id, status):
        status_value = status.value if hasattr(status, "value") else status

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_jobs
                SET status = %s,
                    updated_at = now()
                WHERE job_id = %s
                """,
                (status_value, job_id),
            )

        self.conn.commit()

    def get_job(self, job_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    job_id,
                    experiment_id,
                    config,
                    status,
                    created_at,
                    updated_at
                FROM training_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            )

            row = cur.fetchone()

        if row is None:
            return None

        job_id, experiment_id, config, status, created_at, updated_at = row

        return {
            "job_id": str(job_id),
            "experiment_id": str(experiment_id),
            "config": config,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
        }