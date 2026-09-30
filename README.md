## Architecture

The platform separates data processing, experiment orchestration, model training, and experiment tracking into independent components. This makes it possible to add new models, datasets, and training infrastructure without rewriting the rest of the system.

```text
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │   Experiment API     │
                         └──────────┬───────────┘
                                    │
                           Submit Experiment
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Dataset Registry   │
                         │                      │
                         │ Versioned datasets   │
                         │ + content hashes     │
                         └──────────┬───────────┘
                                    │
                         Pin dataset version
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Postgres       │
                         │                      │
                         │ Job state + results  │
                         └──────────┬───────────┘
                                    │
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │        Kafka         │
                         │    Training Jobs     │
                         └──────────┬───────────┘
                                    │
                         Distribute training work
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │ Worker 1 │   │ Worker 2 │   │ Worker N │
              │          │   │          │   │          │
              │ sklearn  │   │ sklearn  │   │ sklearn  │
              └────┬─────┘   └────┬─────┘   └────┬─────┘
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
                         ┌──────────────────────┐
                         │       MLflow         │
                         │                      │
                         │ Params / Metrics     │
                         │ Models / Runs        │
                         └──────────────────────┘


       Data Pipeline
       ─────────────

  Raw Data
      │
      ▼
 ┌──────────────┐
 │ Spark / ETL  │
 │              │
 │ Validate     │
 │ Transform    │
 │ Quarantine   │
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────────────┐
 │ Databricks / Delta Tables   │
 │                              │
 │ Clean Data                   │
 │ Quarantined Data             │
 │ Dataset Manifests            │
 └──────────────────────────────┘
```

### Components

**FastAPI — Experiment API**

Provides the interface for submitting experiments and querying job status. Experiment requests specify the model, dataset, and hyperparameters.

**Spark + Databricks — Data Pipeline**

Raw datasets are processed through an ETL pipeline that validates records, separates invalid data into quarantine tables, and writes cleaned data to Delta tables. Each dataset version is recorded in a manifest with metadata such as row counts and a content hash.

**Dataset Registry**

Acts as the bridge between the data pipeline and training system. Before training begins, an experiment resolves a dataset name to a specific version, allowing experiments to be reproduced against the same data snapshot.

**Postgres — Job State**

Stores durable training job state, including job configuration, experiment IDs, status, and timestamps. This keeps job state independent from the Kafka queue.

**Kafka — Training Queue**

Training requests are placed onto Kafka so workers can process jobs independently. Multiple workers can consume jobs concurrently, allowing the training layer to scale without changing the API.

**Training Workers**

Workers consume jobs from Kafka, resolve the requested dataset version, and execute model training. The current implementation supports scikit-learn models through an extensible experiment runner.

**MLflow — Experiment Tracking**

Training runs are logged to MLflow with parameters, metrics, models, and dataset metadata. This provides a persistent record of how each experiment performed.

### End-to-End Flow

A typical experiment follows this path:

```text
Raw Dataset
    ↓
Spark ETL
    ↓
Versioned Delta Dataset
    ↓
FastAPI Experiment Request
    ↓
Dataset Version Resolution
    ↓
Postgres Job Record
    ↓
Kafka Training Queue
    ↓
Training Worker
    ↓
Scikit-learn Model
    ↓
MLflow + Postgres Results
```

The system is designed so that individual components can evolve independently. For example, additional model frameworks such as PyTorch can be added to the training layer without changing the API, dataset pipeline, or messaging infrastructure.