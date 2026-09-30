# ML Experimentation Platform

An experiment-tracking platform: a FastAPI service accepts experiment requests, pins them to a versioned dataset, and queues them on Kafka; workers train sklearn models against that dataset and log results to MLflow (Databricks) and Postgres.

## Prerequisites

- Python 3.11+ and a virtualenv with `pip install -r requirements.txt`
- Docker (for Postgres + Kafka via `docker compose up -d`)
- Java 17 (required for local PySpark, even in `local[*]` mode for tests) — a JDK install is fine, no admin rights needed if you use a portable zip distribution and set `JAVA_HOME` yourself
- A Databricks workspace (Free Edition serverless works) with the [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) installed and authenticated

## Databricks Free Edition setup

This project expects the following to already exist in your workspace (create them once via the Databricks UI or SDK if they don't):

- A Unity Catalog catalog and schema, e.g. `workspace.ml_platform`
- A managed Volume under that schema for raw data, e.g. `workspace.ml_platform.raw`
- A running SQL warehouse (the Free Edition default "Serverless Starter Warehouse" works)
- A personal access token (PAT)

Fill these into `.env` at the repo root:

```
DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
DATABRICKS_TOKEN="<your PAT>"
DATABRICKS_WAREHOUSE_ID="<sql warehouse id>"
DATABRICKS_CATALOG="workspace"
DATABRICKS_SCHEMA="ml_platform"
MLFLOW_TRACKING_URI="databricks"
MLFLOW_EXPERIMENT_NAME="/Shared/ml-experimentation-platform"
KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
```

Authenticate the Databricks CLI non-interactively using the same host/token:

```
export DATABRICKS_HOST=...
export DATABRICKS_TOKEN=...
databricks bundle validate
```

## End-to-end flow

```
1. Seed raw data into the Volume
2. Deploy + run the Spark ETL job on Databricks serverless -> clean/quarantine/manifest Delta tables
3. Start the API
4. Start one or more workers
5. Submit an experiment -> trains against the versioned dataset -> MLflow + Postgres
```

### 1. Start Postgres + Kafka

```
docker compose up -d
```

### 2. Seed the raw dataset

Uploads a deliberately-dirty iris CSV (nulls, bad types, out-of-range values, an unknown label, duplicates) to the Volume, so the ETL's validation rules are visibly exercised:

```
python scripts/seed_raw_iris.py
```

### 3. Deploy and run the ETL job

```
databricks bundle deploy
databricks bundle run etl_iris
```

This reads the raw CSV from the Volume, validates/cleans it, and writes `workspace.ml_platform.iris_clean`, `iris_quarantine`, and `dataset_manifests` as Delta tables. The ETL only ever runs this way — as a deployed job on serverless compute — never locally; see "Running tests" below for why.

### 4. Start the API

```
python -m uvicorn src.api:app --port 8000
```

- `GET /health`
- `GET /datasets` — list dataset names with a manifest
- `GET /datasets/{name}` — latest manifest for a dataset
- `POST /experiments` — submit an experiment, e.g.:
  ```
  curl -X POST http://127.0.0.1:8000/experiments \
    -H "Content-Type: application/json" \
    -d '{"model": "logistic_regression", "dataset": "iris", "hyperparameters": {"C": 1.0, "split": 0.2}}'
  ```
  Omit `dataset_version` to pin the latest version automatically, or pass it explicitly to reproduce a run against an older dataset snapshot. Returns 404 if the dataset name/version doesn't exist yet.
- `GET /jobs/{job_id}` — job status (`pending` / `running` / `completed` / `failed`)

### 5. Start a worker

```
python -m src.messaging.consumer
```

Pulls jobs off the `training-jobs` Kafka topic, trains the model against the pinned dataset version (loaded from Delta via the SQL warehouse, cached locally as Parquet under `data/cache/`), and logs the run to MLflow and the `experiment_results` Postgres table.

## Running tests

```
pytest tests/etl tests/test_runner_fake_registry.py -v
```

These need Java 17 (`JAVA_HOME` set) for local PySpark, but need **no Databricks connection at all** — they exercise the pure validate/transform functions against small local Spark DataFrames, and `ExperimentRunner` against an in-memory fake dataset registry. The real Delta/Unity Catalog write path in `src/etl/pipeline.py` only runs on Databricks serverless (via `databricks bundle run`), since `databricks-connect` and plain `pyspark` conflict in the same environment.
