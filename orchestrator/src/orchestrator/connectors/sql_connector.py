"""
SQLCatalogConnector — reads table schemas and sample rows via SQLAlchemy.

Compatible with any SQLAlchemy dialect:
  - PostgreSQL / MySQL / SQLite (standard)
  - Databricks SQL via databricks-sql-connector (install separately)
  - Snowflake, BigQuery, etc.

Usage:
    from sqlalchemy import create_engine
    engine = create_engine("databricks+connector://token@<host>/<http_path>?catalog=<c>&schema=<s>")
    connector = SQLCatalogConnector(engine, schema="gold")
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, inspect, text


class SQLCatalogConnector:
    def __init__(self, engine: Engine, schema: str | None = None) -> None:
        self._engine = engine
        self._schema = schema

    def list_datasets(self) -> list[str]:
        insp = inspect(self._engine)
        return insp.get_table_names(schema=self._schema)

    def get_schema(self, dataset: str) -> dict[str, str]:
        insp = inspect(self._engine)
        columns = insp.get_columns(dataset, schema=self._schema)
        return {col["name"]: str(col["type"]) for col in columns}

    def sample_rows(self, dataset: str, n: int = 1000) -> pd.DataFrame:
        qualified = f"{self._schema}.{dataset}" if self._schema else dataset
        with self._engine.connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM {qualified} LIMIT {n}"), conn)
