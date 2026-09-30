from abc import ABC, abstractmethod
from pathlib import Path
import os
import pandas as pd
from databricks import sql


class DatasetRegistry(ABC):
    @abstractmethod
    def latest(self, name):
        pass

    @abstractmethod
    def get(self, name, version):
        pass

    @abstractmethod
    def load(self, name, version):
        pass

    @abstractmethod
    def list_datasets(self):
        pass


class DatabricksDatasetRegistry(DatasetRegistry):
    def __init__(
        self,
        host=None,
        token=None,
        warehouse_id=None,
        catalog=None,
        schema=None,
        cache_dir="data/cache",
    ):
        self.host = (host or os.environ["DATABRICKS_HOST"]).replace("https://", "")
        self.token = token or os.environ["DATABRICKS_TOKEN"]
        self.warehouse_id = warehouse_id or os.environ["DATABRICKS_WAREHOUSE_ID"]
        self.catalog = catalog or os.environ["DATABRICKS_CATALOG"]
        self.schema = schema or os.environ["DATABRICKS_SCHEMA"]
        self.cache_dir = Path(cache_dir)

    def _connect(self):
        return sql.connect(
            server_hostname=self.host,
            http_path=f"/sql/1.0/warehouses/{self.warehouse_id}",
            access_token=self.token,
        )

    def _manifest_table(self):
        return f"{self.catalog}.{self.schema}.dataset_manifests"

    def latest(self, name):
        query = f"""
            SELECT name, delta_version, rows_in, rows_out, rows_rejected, schema_json, content_hash, created_at
            FROM {self._manifest_table()}
            WHERE name = :name
            ORDER BY delta_version DESC
            LIMIT 1
        """
        return self._fetch_one(query, {"name": name})

    def get(self, name, version):
        query = f"""
            SELECT name, delta_version, rows_in, rows_out, rows_rejected, schema_json, content_hash, created_at
            FROM {self._manifest_table()}
            WHERE name = :name AND delta_version = :version
        """
        return self._fetch_one(query, {"name": name, "version": version})

    def list_datasets(self):
        query = f"SELECT DISTINCT name FROM {self._manifest_table()} ORDER BY name"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [row.name for row in cur.fetchall()]

    def load(self, name, version):
        cache_path = self.cache_dir / name / f"v{version}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        table = f"{self.catalog}.{self.schema}.{name}_clean"
        query = f"SELECT * FROM {table} VERSION AS OF {int(version)}"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                df = cur.fetchall_arrow().to_pandas()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        return df

    def _fetch_one(self, query, params):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()

        if row is None:
            return None

        return {
            "name": row.name,
            "delta_version": row.delta_version,
            "rows_in": row.rows_in,
            "rows_out": row.rows_out,
            "rows_rejected": row.rows_rejected,
            "schema_json": row.schema_json,
            "content_hash": row.content_hash,
            "created_at": row.created_at,
        }
