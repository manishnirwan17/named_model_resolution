"""
Parse gold_layer_datamarts.csv → DatamartCatalog.

The CSV is a multi-section flat file (originally Excel) where three logical
table blocks are stacked vertically with section-header rows between them.
Standard read_csv() cannot parse this layout directly.

Section-header sentinels (first non-empty cell of the row):
  "Datamarts with Granularity Example"
  "Payer/Access Marts"
  "Patient dynamics / market marts"

Within each section the row immediately after the sentinel is the datamart-name
row (up to 4 names across columns A-D).  Subsequent rows are column lists until
the next sentinel or a fully-blank row that separates sections.

The Category/Description table in rows 20-26 (0-indexed) provides per-datamart
category and description text.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .models import DatamartCatalog, DatamartSpec

_SECTION_SENTINELS = frozenset(
    {
        "datamarts with granularity example",
        "payer/access marts",
        "patient dynamics / market marts",
    }
)

# Row range (0-indexed) for the Category/Datamart summary table in the CSV.
_CATEGORY_TABLE_START = 19  # "Category,Datamart,Contents / Description," row
_CATEGORY_TABLE_END = 27    # blank separator row


def _normalize(cell: str) -> str:
    return cell.strip().lower()


def _is_sentinel(row: list[str]) -> bool:
    first = _normalize(row[0]) if row else ""
    return first in _SECTION_SENTINELS


def _is_blank_row(row: list[str]) -> bool:
    return all(c.strip() == "" for c in row)


def parse_catalog(csv_path: str | Path) -> DatamartCatalog:
    """Parse the gold layer datamart spec CSV and return a DatamartCatalog."""
    csv_path = Path(csv_path)
    raw: list[list[str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            raw.append(row)

    # Pad every row to at least 4 columns so index access is safe.
    for i, row in enumerate(raw):
        if len(row) < 4:
            raw[i] = row + [""] * (4 - len(row))

    # ── Step 1: parse Category/Description table ──────────────────────────────
    # Row _CATEGORY_TABLE_START has header "Category,Datamart,Contents..."
    # Rows after that have actual data until _CATEGORY_TABLE_END.
    category_map: dict[str, tuple[str, str]] = {}  # datamart_name → (category, description)
    for row in raw[_CATEGORY_TABLE_START + 1 : _CATEGORY_TABLE_END]:
        if _is_blank_row(row):
            continue
        category = row[0].strip()
        datamart = row[1].strip()
        description = row[2].strip()
        if datamart:
            category_map[datamart] = (category, description)

    # ── Step 2: extract per-section datamart column lists ─────────────────────
    datamarts: dict[str, DatamartSpec] = {}

    i = 0
    while i < len(raw):
        row = raw[i]
        if _is_sentinel(row):
            i += 1
            if i >= len(raw):
                break
            # Name row immediately follows sentinel
            name_row = raw[i]
            # Collect up to 4 datamart names (one per column)
            names = [name_row[col].strip() for col in range(4) if name_row[col].strip()]
            if not names:
                i += 1
                continue

            # Initialise column lists for each discovered datamart
            col_lists: list[list[str]] = [[] for _ in names]

            i += 1
            while i < len(raw):
                data_row = raw[i]
                if _is_sentinel(data_row):
                    break  # next section starts — do NOT advance i (outer loop will)
                if _is_blank_row(data_row):
                    # A blank row signals end of this section's column block
                    i += 1
                    break
                for col_idx, col_list in enumerate(col_lists):
                    if col_idx < len(data_row):
                        cell = data_row[col_idx].strip()
                        if cell:
                            col_list.append(cell)
                i += 1

            for name, col_list in zip(names, col_lists):
                category, description = category_map.get(name, ("", ""))
                datamarts[name] = DatamartSpec(
                    name=name,
                    columns=col_list,
                    category=category,
                    description=description,
                )
        else:
            i += 1

    return DatamartCatalog(datamarts=datamarts)
