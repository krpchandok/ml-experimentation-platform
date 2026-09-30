from fastapi import FastAPI, HTTPException

from src.jobs import submit_job_from_config
from src.job_store import PostgresJobStore


app = FastAPI(
    title="ML Experimentation Platform",
    version="0.1.0",
)

job_store = PostgresJobStore()


from pydantic import BaseModel
from typing import Any

class ExperimentRequest(BaseModel):
    model: str
    dataset: str
    hyperparameters: dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/experiments")
def create_experiment(request: ExperimentRequest):
    config = {
        "model": request.model,
        "dataset": request.dataset,
        "hyperparameters": request.hyperparameters,
    }

    job = submit_job_from_config(config)

    return {
        "experiment_id": job.experiment_id,
        "job_id": job.job_id,
        "status": job.status.value,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job