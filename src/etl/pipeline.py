from pyspark.sql import SparkSession, functions as F
import json
import os

from src.etl.spec import load_spec
from src.etl.transforms import read_raw, transform
from src.etl.validation import validate


def content_hash(df):
    row = df.select(
        F.sum(F.crc32(F.concat_ws("|", *[F.col(c).cast("string") for c in df.columns]))).alias("checksum")
    ).collect()[0]
    return str(row["checksum"])


def run_pipeline(spec_path, spark=None, catalog=None, schema=None):
    spec = load_spec(spec_path)
    spark = spark or SparkSession.builder.getOrCreate()
    catalog = catalog or os.environ["DATABRICKS_CATALOG"]
    schema = schema or os.environ["DATABRICKS_SCHEMA"]

    raw_df = read_raw(spark, spec)
    clean_df, quarantine_df = validate(raw_df, spec)
    clean_df = transform(clean_df, spec)

    rows_in = raw_df.count()
    rows_rejected = quarantine_df.count()
    rows_out = clean_df.count()
    rejection_rate = rows_rejected / rows_in if rows_in else 0.0

    if rejection_rate > spec.max_rejection_rate:
        raise RuntimeError(
            f"Rejection rate {rejection_rate:.2%} exceeds max {spec.max_rejection_rate:.2%} for dataset {spec.name}"
        )

    clean_table = f"{catalog}.{schema}.{spec.name}_clean"
    quarantine_table = f"{catalog}.{schema}.{spec.name}_quarantine"
    manifest_table = f"{catalog}.{schema}.dataset_manifests"

    clean_df.write.format("delta").mode("overwrite").saveAsTable(clean_table)
    quarantine_df.write.format("delta").mode("overwrite").saveAsTable(quarantine_table)

    history = spark.sql(f"DESCRIBE HISTORY {clean_table} LIMIT 1")
    delta_version = int(history.collect()[0]["version"])

    schema_json = json.dumps({f.name: str(f.dataType) for f in clean_df.schema.fields})

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {manifest_table} (
            name STRING,
            delta_version BIGINT,
            rows_in BIGINT,
            rows_out BIGINT,
            rows_rejected BIGINT,
            schema_json STRING,
            content_hash STRING,
            created_at TIMESTAMP
        ) USING DELTA
    """)

    manifest = {
        "name": spec.name,
        "delta_version": delta_version,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_rejected": rows_rejected,
        "schema_json": schema_json,
        "content_hash": content_hash(clean_df),
    }

    manifest_df = spark.createDataFrame([manifest]).withColumn("created_at", F.current_timestamp())
    manifest_df.write.format("delta").mode("append").saveAsTable(manifest_table)

    return manifest
