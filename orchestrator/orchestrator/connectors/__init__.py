from .base import CatalogConnector
from .file_connector import FileCatalogConnector

__all__ = ["CatalogConnector", "FileCatalogConnector"]

try:
    from .sql_connector import SQLCatalogConnector

    __all__ = ["CatalogConnector", "FileCatalogConnector", "SQLCatalogConnector"]
except ImportError:
    # sqlalchemy is not installed — SQLCatalogConnector is unavailable.
    # Install it with: pip install sqlalchemy
    # For Databricks SQL: pip install databricks-sql-connector sqlalchemy-databricks
    pass
