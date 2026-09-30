from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from src.jobs import submit_job_from_config
from src.job_store import PostgresJobStore
from src.datasets.registry import DatabricksDatasetRegistry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(
    title="ML Experimentation Platform",
    version="0.1.0",
)

job_store = PostgresJobStore()
dataset_registry = DatabricksDatasetRegistry()


class ExperimentRequest(BaseModel):
    model: str
    dataset: str
    hyperparameters: dict[str, Any]
    dataset_version: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/experiments")
def create_experiment(request: ExperimentRequest):
    if request.dataset_version is not None:
        manifest = dataset_registry.get(request.dataset, request.dataset_version)
    else:
        manifest = dataset_registry.latest(request.dataset)

    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset}",
        )

    config = {
        "model": request.model,
        "dataset": request.dataset,
        "dataset_version": manifest["delta_version"],
        "content_hash": manifest["content_hash"],
        "hyperparameters": request.hyperparameters,
    }

    job = submit_job_from_config(config)

    return {
        "experiment_id": job.experiment_id,
        "job_id": job.job_id,
        "status": job.status.value,
        "dataset_version": manifest["delta_version"],
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


@app.get("/datasets")
def list_datasets():
    return dataset_registry.list_datasets()


@app.get("/datasets/{name}")
def get_dataset(name: str):
    manifest = dataset_registry.latest(name)

    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {name}",
        )

    return manifest
