"""
dataset_config.py — Schema-driven, auto-detected dataset configuration.

On first run, call DatasetConfig.from_dataframe(df) (invoked automatically by
config.py) to infer column roles from the raw gold table.  The result is saved to
dataset_config.json at the project root and reloaded on every subsequent run.

To enable ground-truth validation checks (V-01/V-02/V-07/V-08), edit
dataset_config.json and populate the "organic_cps" and "true_effects" fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Column-name hints for auto-detection ─────────────────────────────────────
_WEEK_HINTS        = ["week", "date", "period", "week_ending", "week_start"]
_HCP_ID_HINTS      = ["hcp_id", "doctor_id", "physician_id", "npi", "rep_id"]
_PRODUCT_HINTS     = ["product", "brand", "drug", "therapy", "molecule"]
_ANOMALY_HINTS     = ["is_anomaly", "anomaly", "is_flagged", "flag"]
_LIFECYCLE_HINTS   = ["lifecycle_stage", "lifecycle", "phase", "stage", "lc_stage"]
_TARGET_PRIORITY   = ["sales", "revenue", "trx", "units", "scripts", "rx", "nrx"]
_SECONDARY_TARGETS = {"trx", "nrx", "revenue", "units", "scripts", "rx"}
_COMPETITOR_HINTS  = ["competitor", "compet", "rival", "generic"]

_DEFAULT_DECAY = {"field": 0.30, "broadcast": 0.60, "competitor": 0.0}


@dataclass
class ChannelSpec:
    """Specification for a single marketing / engagement channel."""
    name: str
    group: str        # 'field' | 'broadcast' | 'competitor'
    agg: str          # 'sum'  (per-HCP additive) | 'mean' (market-level constant)
    decay: float
    hill_alpha: Optional[float] = None  # None → use DatasetConfig group default


@dataclass
class DatasetConfig:
    """
    Dataset schema and pipeline configuration.

    Serialised to/from dataset_config.json so the auto-detected schema is
    inspectable and editable without touching any pipeline source files.
    """
    target_col: str
    week_col: str = "week"
    hcp_id_col: Optional[str] = "hcp_id"
    product_col: Optional[str] = "product"
    product_value: Optional[str] = None
    is_anomaly_col: Optional[str] = "is_anomaly"
    lifecycle_col: Optional[str] = "lifecycle_stage"
    channels: list = field(default_factory=list)   # list[ChannelSpec]
    fourier_k: int = 2
    fourier_period: int = 52
    hill_alpha_field: float = 0.8
    hill_alpha_broadcast: float = 2.0
    train_weeks: int = 180
    # Ground-truth for validation — cannot be auto-detected; add manually to JSON
    organic_cps: Optional[dict] = None   # {name: "YYYY-MM-DD"}
    true_effects: Optional[dict] = None  # {channel_name: float}

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def target_log_col(self) -> str:
        return f"log_{self.target_col}"

    @property
    def field_channels(self) -> list[str]:
        return [c.name for c in self.channels if c.group == "field"]

    @property
    def broadcast_channels(self) -> list[str]:
        return [c.name for c in self.channels if c.group == "broadcast"]

    @property
    def competitor_channels(self) -> list[str]:
        return [c.name for c in self.channels if c.group == "competitor"]

    @property
    def mmm_channels(self) -> list[str]:
        """Field + broadcast channels (positive-constrained in MMM)."""
        return self.field_channels + self.broadcast_channels

    @property
    def sum_channels(self) -> list[str]:
        """Channels aggregated by SUM across HCPs."""
        return [c.name for c in self.channels if c.agg == "sum"]

    @property
    def mean_channels(self) -> list[str]:
        """Channels aggregated by MEAN (market-level constants)."""
        return [c.name for c in self.channels if c.agg == "mean"]

    @property
    def decay(self) -> dict[str, float]:
        return {c.name: c.decay for c in self.channels}

    def hill_alpha(self, channel_name: str) -> float:
        """Per-channel Hill alpha, falling back to group-level default."""
        for c in self.channels:
            if c.name == channel_name:
                if c.hill_alpha is not None:
                    return c.hill_alpha
                return (self.hill_alpha_field if c.group == "field"
                        else self.hill_alpha_broadcast)
        return self.hill_alpha_field

    @property
    def organic_cp_timestamps(self) -> Optional[dict]:
        """Return organic_cps values as pd.Timestamp, or None if not configured."""
        if not self.organic_cps:
            return None
        return {k: pd.Timestamp(v) for k, v in self.organic_cps.items()}

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write to JSON.  Human-readable so users can edit ground-truth fields."""
        d = {
            "target_col":           self.target_col,
            "week_col":             self.week_col,
            "hcp_id_col":           self.hcp_id_col,
            "product_col":          self.product_col,
            "product_value":        self.product_value,
            "is_anomaly_col":       self.is_anomaly_col,
            "lifecycle_col":        self.lifecycle_col,
            "fourier_k":            self.fourier_k,
            "fourier_period":       self.fourier_period,
            "hill_alpha_field":     self.hill_alpha_field,
            "hill_alpha_broadcast": self.hill_alpha_broadcast,
            "train_weeks":          self.train_weeks,
            "organic_cps":          self.organic_cps,
            "true_effects":         self.true_effects,
            "channels": [
                {
                    "name":       c.name,
                    "group":      c.group,
                    "agg":        c.agg,
                    "decay":      c.decay,
                    "hill_alpha": c.hill_alpha,
                }
                for c in self.channels
            ],
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  [DatasetConfig] saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "DatasetConfig":
        """Load from JSON."""
        with open(path) as f:
            d = json.load(f)
        channels = [
            ChannelSpec(
                name=c["name"], group=c["group"], agg=c["agg"],
                decay=c["decay"], hill_alpha=c.get("hill_alpha"),
            )
            for c in d.get("channels", [])
        ]
        return cls(
            target_col=d["target_col"],
            week_col=d.get("week_col", "week"),
            hcp_id_col=d.get("hcp_id_col"),
            product_col=d.get("product_col"),
            product_value=d.get("product_value"),
            is_anomaly_col=d.get("is_anomaly_col"),
            lifecycle_col=d.get("lifecycle_col"),
            channels=channels,
            fourier_k=d.get("fourier_k", 2),
            fourier_period=d.get("fourier_period", 52),
            hill_alpha_field=d.get("hill_alpha_field", 0.8),
            hill_alpha_broadcast=d.get("hill_alpha_broadcast", 2.0),
            train_weeks=d.get("train_weeks", 180),
            organic_cps=d.get("organic_cps"),
            true_effects=d.get("true_effects"),
        )

    # ── Auto-detection ────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                        target_col: Optional[str] = None) -> "DatasetConfig":
        """
        Auto-detect dataset schema from a pandas DataFrame.

        Detection logic
        ---------------
        * Structural columns (week, hcp_id, product, …) — matched by name pattern.
        * Target column — priority list; first numeric column as fallback.
        * Broadcast vs field — coefficient of variation (CV = std/mean) of each
          channel across HCPs within the same week.  CV < 0.01 means every HCP in
          a given week has the same value (market-level constant) → broadcast/mean.
          CV ≥ 0.01 means per-HCP activity → field/sum.
        * Competitor channels — name substring matching.
        * Default decays: field=0.30, broadcast=0.60, competitor=0.0.

        organic_cps and true_effects are left as None (not auto-detectable).
        Edit dataset_config.json to add them and enable V-01/V-02/V-07/V-08.
        """
        df = df.head(1000).copy()

        def _find(hints: list[str]) -> Optional[str]:
            for h in hints:
                for col in df.columns:
                    if col.lower() == h.lower():
                        return col
            return None

        week_col       = _find(_WEEK_HINTS) or "week"
        hcp_id_col     = _find(_HCP_ID_HINTS)
        is_anomaly_col = _find(_ANOMALY_HINTS)
        lifecycle_col  = _find(_LIFECYCLE_HINTS)

        # Product column: string/categorical with low cardinality
        product_col = _find(_PRODUCT_HINTS)
        if product_col is None:
            for col in df.select_dtypes(include=["object", "category"]).columns:
                if col != week_col and df[col].nunique() < 20:
                    product_col = col
                    break

        # Target column
        if target_col is None:
            target_col = _find(_TARGET_PRIORITY)
        if target_col is None:
            meta = {week_col, hcp_id_col, is_anomaly_col, product_col, lifecycle_col}
            for col in df.select_dtypes(include="number").columns:
                if col not in meta:
                    target_col = col
                    break
        if target_col is None:
            raise ValueError(
                "Could not auto-detect target column. "
                "Pass target_col= explicitly to from_dataframe()."
            )

        # Most-frequent product value
        product_value = None
        if product_col and product_col in df.columns:
            product_value = str(df[product_col].mode().iloc[0])

        # Columns to exclude from channel detection
        meta_cols = {
            c for c in [week_col, hcp_id_col, product_col, is_anomaly_col,
                         lifecycle_col, target_col] if c is not None
        } | _SECONDARY_TARGETS

        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in meta_cols
        ]

        channels: list[ChannelSpec] = []
        for col in numeric_cols:
            group, agg = _classify_channel(df, col, hcp_id_col, week_col)
            if any(h in col.lower() for h in _COMPETITOR_HINTS):
                group, agg = "competitor", "mean"
            channels.append(ChannelSpec(
                name=col, group=group, agg=agg,
                decay=_DEFAULT_DECAY[group],
            ))

        n_field = sum(1 for c in channels if c.group == "field")
        n_bcast = sum(1 for c in channels if c.group == "broadcast")
        n_comp  = sum(1 for c in channels if c.group == "competitor")
        print(f"  [DatasetConfig] auto-detected: target='{target_col}', "
              f"{n_field} field, {n_bcast} broadcast, {n_comp} competitor channels")
        if lifecycle_col is None:
            print("  [DatasetConfig] no lifecycle column detected — "
                  "lc_num covariate will be skipped")
        print("  [DatasetConfig] organic_cps / true_effects left as null — "
              "edit dataset_config.json to enable V-01/V-02/V-07/V-08")

        return cls(
            target_col=target_col,
            week_col=week_col,
            hcp_id_col=hcp_id_col,
            product_col=product_col,
            product_value=product_value,
            is_anomaly_col=is_anomaly_col,
            lifecycle_col=lifecycle_col,
            channels=channels,
        )

    @classmethod
    def from_parquet(cls, path: str,
                      target_col: Optional[str] = None) -> "DatasetConfig":
        """Convenience wrapper: load a local parquet file and auto-detect schema."""
        df = pd.read_parquet(path)
        return cls.from_dataframe(df, target_col=target_col)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_channel(df: pd.DataFrame, col: str,
                       hcp_id_col: Optional[str],
                       week_col: str) -> tuple[str, str]:
    """
    Classify a column as 'broadcast' (mean) or 'field' (sum).

    A broadcast metric (TV GRPs, impressions) is the same for every HCP in a
    given week — coefficient of variation across HCPs is near zero.
    A field metric (visits, calls) varies per HCP — CV is non-trivial.
    """
    if hcp_id_col is None or hcp_id_col not in df.columns or week_col not in df.columns:
        return "field", "sum"
    try:
        grp = df.groupby(week_col)[col]
        cv  = (grp.std() / (grp.mean().abs() + 1e-9)).mean()
        if cv < 0.01:
            return "broadcast", "mean"
    except Exception:
        pass
    return "field", "sum"
