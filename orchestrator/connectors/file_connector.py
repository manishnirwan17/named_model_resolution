"""
FileCatalogConnector — reads CSV and Parquet files from a local directory.

Dataset names are the filenames without extension.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class FileCatalogConnector:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise ValueError(f"Directory not found: {self._dir}")

    def _path_for(self, dataset: str) -> Path:
        for ext in (".parquet", ".csv"):
            p = self._dir / f"{dataset}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No CSV or Parquet file found for dataset '{dataset}' in {self._dir}")

    def list_datasets(self) -> list[str]:
        names: list[str] = []
        for ext in ("*.parquet", "*.csv"):
            for p in sorted(self._dir.glob(ext)):
                stem = p.stem
                if stem not in names:
                    names.append(stem)
        return names

    def get_schema(self, dataset: str) -> dict[str, str]:
        path = self._path_for(dataset)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=None).head(0)
        else:
            df = pd.read_csv(path, nrows=0)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def sample_rows(self, dataset: str, n: int = 1000) -> pd.DataFrame:
        path = self._path_for(dataset)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return df.head(n)
        else:
            return pd.read_csv(path, nrows=n)
