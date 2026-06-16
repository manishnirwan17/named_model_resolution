"""
03_mmm_data_prep.py — Adstock, saturation, scaling, final model matrix for MMM.

Reads:  MARKET_SERIES  (output of 01_data_prep.py)
Writes: MODEL_MATRIX   (X matrix + y vector, ready for PyMC in 04_mmm_fit.py)

Channel groups, decay rates, and Hill alphas are driven entirely by DATASET
(DatasetConfig) — no hardcoded column lists in this file.
"""

from __future__ import annotations

try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except Exception:
    pass
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import (
    MARKET_SERIES, MODEL_MATRIX, MODEL_OUT,
    read_parquet, write_parquet, PARAMS, DATASET,
)


# ── Transformations ───────────────────────────────────────────────────────────

def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric carry-over adstock.
    adstock[t] = x[t] + decay * adstock[t-1]
    """
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + decay * (out[t - 1] if t > 0 else 0.0)
    return out


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """
    Hill (diminishing returns) saturation.
    sat(x) = x^alpha / (x^alpha + K^alpha)
    alpha < 1: fully concave;  alpha > 1: S-curve
    K: half-saturation point (set to median of adstocked series).
    """
    xa = np.power(np.maximum(x, 0), alpha)
    Ka = K ** alpha
    return xa / (xa + Ka + 1e-12)


def transform_channels(mkt: pd.DataFrame, dataset_config=None) -> tuple[pd.DataFrame, dict]:
    """
    Apply adstock + Hill saturation to all MMM channels (field + broadcast).
    Channel groups, decay rates, and Hill alphas come from dataset_config (or
    the global DATASET when dataset_config is None).
    Returns the enriched DataFrame and a metadata dict for back-transforms.
    """
    ds = dataset_config if dataset_config is not None else DATASET
    meta: dict = {}

    for col in ds.field_channels + ds.broadcast_channels:
        if col not in mkt.columns:
            continue
        decay_val = ds.decay.get(col, 0.3)
        alpha_val = ds.hill_alpha(col)
        group_tag = "A" if col in ds.field_channels else "B"

        adstocked = geometric_adstock(mkt[col].values, decay_val)
        K = float(np.median(adstocked[adstocked > 0])) if (adstocked > 0).any() else 1.0
        saturated = hill_saturation(adstocked, alpha_val, K)

        mkt[f"{col}_ads"] = adstocked
        mkt[f"{col}_sat"] = saturated
        meta[col] = {
            "type":       group_tag,
            "decay":      decay_val,
            "hill_alpha": alpha_val,
            "hill_K":     K,
        }

    return mkt, meta


def build_model_matrix(mkt: pd.DataFrame, dataset_config=None) -> tuple[np.ndarray, np.ndarray,
                                                    list[str], dict]:
    """
    Build final X (feature matrix) and y (log target) for MMM.

    Feature columns:
      - saturated versions of field + broadcast channels
      - competitor channels (no adstock/saturation)
      - lifecycle numeric (if available)
      - week_idx (trend)
      - Fourier sine/cosine terms

    Returns: X_scaled, y, feature_names, scaler_params
    """
    ds = dataset_config if dataset_config is not None else DATASET

    # Saturated channel features
    sat_cols = [f"{c}_sat" for c in ds.mmm_channels if f"{c}_sat" in mkt.columns]

    # Competitor channels (enter model directly, no transform)
    comp_cols = [c for c in ds.competitor_channels if c in mkt.columns]

    # Covariates
    cov_cols: list[str] = []
    if ds.lifecycle_col and "lc_num" in mkt.columns:
        cov_cols.append("lc_num")
    cov_cols.append("week_idx")

    # Fourier terms
    fourier_cols = [f"{fn}_{k}"
                    for k in range(1, ds.fourier_k + 1)
                    for fn in ("sin", "cos")
                    if f"{fn}_{k}" in mkt.columns]

    feature_cols = [c for c in sat_cols + comp_cols + cov_cols + fourier_cols
                    if c in mkt.columns]

    X_raw = mkt[feature_cols].fillna(0).values.astype(float)
    y     = mkt[ds.target_log_col].values.astype(float)

    # Standardise on TRAINING rows only; apply same scaler to full series
    train_mask = (mkt["split"] == "train").values
    X_mean = X_raw[train_mask].mean(axis=0)
    X_std  = X_raw[train_mask].std(axis=0)
    X_std  = np.where(X_std == 0, 1.0, X_std)

    X_scaled = (X_raw - X_mean) / X_std

    scaler_params = {
        "feature_cols": feature_cols,
        "X_mean": X_mean.tolist(),
        "X_std":  X_std.tolist(),
        "y_mean": float(y[train_mask].mean()),
        "y_std":  float(y[train_mask].std()),
    }
    return X_scaled, y, feature_cols, scaler_params


def mmm_data_prep() -> None:
    print("=" * 60)
    print("  03  MMM DATA PREP")
    print("=" * 60)

    mkt = read_parquet(MARKET_SERIES)

    mkt, channel_meta = transform_channels(mkt)
    print(f"  Transformed {len(channel_meta)} channels (adstock + Hill saturation)")

    X, y, feature_cols, scaler_params = build_model_matrix(mkt)

    print(f"  X shape: {X.shape}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]  ({DATASET.target_log_col})")
    print(f"  Features: {feature_cols}")

    # Collinearity check
    if X.shape[1] > 1:
        corr = np.corrcoef(X.T)
        n    = len(feature_cols)
        hi_corr = [
            (feature_cols[i], feature_cols[j], float(corr[i, j]))
            for i in range(n) for j in range(i + 1, n)
            if abs(corr[i, j]) > 0.85
        ]
        if hi_corr:
            print(f"\n  [warn] High collinearity pairs (|r|>0.85) — consider grouping:")
            for a, b, r in hi_corr[:10]:
                print(f"    {a} <-> {b}  r={r:.3f}")
        else:
            print("  No high collinearity pairs (|r|>0.85).")

    x_df = pd.DataFrame(X, columns=[f"X_{c}" for c in feature_cols])
    x_df["y"]     = y
    x_df["week"]  = mkt[DATASET.week_col].values
    x_df["split"] = mkt["split"].values
    write_parquet(x_df, MODEL_MATRIX)

    meta_path = Path(MODEL_OUT) / "mmm_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"scaler": scaler_params, "channel_meta": channel_meta}, f, indent=2)

    print(f"\n  Model matrix written -> {MODEL_MATRIX}")
    print(f"  Scaler + channel meta -> {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    mmm_data_prep()
