"""
CatalogConnector Protocol — platform-agnostic interface for dataset discovery.

Implementations must provide:
  list_datasets()        → names of available tables/files
  get_schema(dataset)    → {column_name: dtype_string}
  sample_rows(dataset)   → pandas DataFrame with up to n rows

Add new connector implementations (e.g. DatabricksConnector, BigQueryConnector)
by implementing this Protocol.  No changes to the router are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class CatalogConnector(Protocol):
    def list_datasets(self) -> list[str]:
        """Return names of all available datasets in this catalog."""
        ...

    def get_schema(self, dataset: str) -> dict[str, str]:
        """Return {column_name: dtype_string} for the given dataset."""
        ...

    def sample_rows(self, dataset: str, n: int = 1000) -> "pd.DataFrame":
        """Return up to n rows from the dataset as a pandas DataFrame."""
        ...
