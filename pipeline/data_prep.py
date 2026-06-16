"""
01_data_prep.py — Load gold layer, aggregate to market level, build feature set.

Reads:  GOLD_LABELLED  (UC table on Databricks; parquet file locally)
Writes: MARKET_SERIES  (~N rows x ~M cols; one row per week, market-level; UC table)

Column roles (which columns are the target, channels, etc.) are driven entirely by
DATASET (DatasetConfig), auto-detected on first run and cached in dataset_config.json.

HYBRID EXECUTION
----------------
The expensive step is the HCP x week -> market aggregation over a potentially huge
gold table. That step runs in SPARK when a SparkSession is available: filtering and
the groupby happen in Spark, and ONLY the ~N-row aggregate is pulled into the driver
via .toPandas(). All downstream feature engineering (~N rows) stays in pandas because
it is trivially small and Spark would add overhead for no gain.

If Spark is NOT available, it falls back to a pure-pandas path. WARNING: the pandas
fallback must read the ENTIRE gold table into driver/local memory before filtering and
aggregating. On a large multi-million-row gold table this can exhaust memory and crash.
The pandas path is intended for local development on a small/sampled extract only.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.config import (
    ON_DATABRICKS, GOLD_LABELLED, MARKET_SERIES,
    read_parquet, write_parquet, PARAMS, DATASET,
)


# ── Spark availability guard ──────────────────────────────────────────────────
def get_spark():
    """Return an active SparkSession if one exists, else None."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        return spark
    except Exception:
        return None


# ── Aggregation: Spark path (large table) ─────────────────────────────────────
def aggregate_to_market_spark(spark, source_ref: str) -> pd.DataFrame:
    """
    Filter + collapse HCP x product x week -> week market-level aggregate in Spark,
    then pull only the small result into pandas.
    """
    from pyspark.sql import functions as F

    print("  [spark] reading + filtering + aggregating in Spark (driver stays light)")

    sdf = spark.table(source_ref) if ON_DATABRICKS else spark.read.parquet(source_ref)

    n_total = sdf.count()

    # Anomaly filter
    filter_cond = F.lit(True)
    if DATASET.is_anomaly_col and DATASET.is_anomaly_col in sdf.columns:
        n_anom = sdf.filter(F.col(DATASET.is_anomaly_col) == 1).count()
        print(f"  {n_total:,} rows  |  {n_anom:,} labelled anomaly rows")
        filter_cond = filter_cond & (F.col(DATASET.is_anomaly_col) == 0)
    else:
        print(f"  {n_total:,} rows")

    # Product filter
    if DATASET.product_col and DATASET.product_value:
        filter_cond = filter_cond & (F.col(DATASET.product_col) == DATASET.product_value)

    sdf = sdf.filter(filter_cond)
    n_clean = sdf.count()
    if DATASET.hcp_id_col and DATASET.hcp_id_col in sdf.columns:
        n_hcp = sdf.select(DATASET.hcp_id_col).distinct().count()
        label = DATASET.product_value or "all products"
        print(f"  Using clean {label} rows: {n_clean:,} rows across {n_hcp:,} HCPs")

    # Identify extra target-adjacent columns present in the table
    sdf_cols = sdf.columns
    extra_targets = [c for c in ["trx", "nrx"]
                     if c in sdf_cols and c != DATASET.target_col]

    sum_cols  = [c for c in DATASET.sum_channels + [DATASET.target_col] + extra_targets
                 if c in sdf_cols]
    mean_cols = [c for c in DATASET.mean_channels if c in sdf_cols]

    agg_exprs = (
        [F.sum(c).alias(c) for c in sum_cols]
        + [F.mean(c).alias(c) for c in mean_cols]
    )
    if DATASET.lifecycle_col and DATASET.lifecycle_col in sdf_cols:
        agg_exprs.append(F.first(DATASET.lifecycle_col).alias(DATASET.lifecycle_col))

    group_cols = [DATASET.week_col]
    if DATASET.product_col and DATASET.product_col in sdf_cols:
        group_cols.append(DATASET.product_col)

    mkt = sdf.groupBy(*group_cols).agg(*agg_exprs).toPandas()
    print(f"  [spark] collected {len(mkt):,} market-level rows to pandas")
    return mkt


# ── Aggregation: pandas fallback (small extract only) ─────────────────────────
def aggregate_to_market_pandas(source_ref: str) -> pd.DataFrame:
    """
    Pure-pandas aggregation. WARNING: reads the ENTIRE source into memory first.
    Safe only for a small/sampled local extract.
    """
    print("  [pandas] WARNING: loading the full table into memory before aggregating.")
    print("           If the gold table is large this may exhaust memory and crash.")

    df = read_parquet(source_ref)
    print(f"  {len(df):,} rows loaded")

    if DATASET.is_anomaly_col and DATASET.is_anomaly_col in df.columns:
        print(f"  {df[DATASET.is_anomaly_col].sum():,} labelled anomaly rows")
        df = df[df[DATASET.is_anomaly_col] == 0].copy()

    if DATASET.product_col and DATASET.product_value and DATASET.product_col in df.columns:
        df = df[df[DATASET.product_col] == DATASET.product_value].copy()
        if DATASET.hcp_id_col and DATASET.hcp_id_col in df.columns:
            label = DATASET.product_value
            print(f"  Using clean {label} rows: {len(df):,} rows "
                  f"across {df[DATASET.hcp_id_col].nunique():,} HCPs")

    extra_targets = [c for c in ["trx", "nrx"]
                     if c in df.columns and c != DATASET.target_col]

    agg_dict: dict = {}
    for c in DATASET.sum_channels + [DATASET.target_col] + extra_targets:
        if c in df.columns:
            agg_dict[c] = "sum"
    for c in DATASET.mean_channels:
        if c in df.columns:
            agg_dict[c] = "mean"
    if DATASET.lifecycle_col and DATASET.lifecycle_col in df.columns:
        agg_dict[DATASET.lifecycle_col] = "first"

    group_cols = [DATASET.week_col]
    if DATASET.product_col and DATASET.product_col in df.columns:
        group_cols.append(DATASET.product_col)

    mkt = df.groupby(group_cols).agg(agg_dict).reset_index()
    return mkt


def aggregate_to_market(source_ref: str) -> pd.DataFrame:
    """Dispatch to Spark if available, else pandas with a memory warning."""
    spark = get_spark()
    if spark is not None:
        return aggregate_to_market_spark(spark, source_ref)
    return aggregate_to_market_pandas(source_ref)


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(mkt: pd.DataFrame) -> pd.DataFrame:
    """
    Add log(target), lifecycle numeric, week index, organic CP flags, Fourier
    seasonality, and log-transformed broadcast columns.
    All decisions are driven by DATASET — no hardcoded column names.
    """
    mkt = mkt.sort_values(DATASET.week_col).reset_index(drop=True)

    # Log-transform target
    mkt[DATASET.target_log_col] = np.log(mkt[DATASET.target_col].clip(lower=1e-6))

    # Lifecycle numeric (optional — skipped if lifecycle_col not in dataset)
    if DATASET.lifecycle_col and DATASET.lifecycle_col in mkt.columns:
        stage_map = {"pre_launch": 0, "launch": 1, "growth": 2, "maturity": 3, "decline": 4}
        mkt["lc_num"] = mkt[DATASET.lifecycle_col].map(stage_map).fillna(2).astype(int)

    # Week index (always present)
    mkt["week_idx"] = np.arange(len(mkt))

    # Organic CP step-function flags (optional — skipped if organic_cps not configured)
    if DATASET.organic_cp_timestamps:
        for name, ts in DATASET.organic_cp_timestamps.items():
            mkt[f"flag_{name}"] = (mkt[DATASET.week_col] >= ts).astype(int)

    # Fourier seasonality terms
    K = DATASET.fourier_k
    t = mkt["week_idx"].values
    for k in range(1, K + 1):
        mkt[f"sin_{k}"] = np.sin(2 * np.pi * k * t / DATASET.fourier_period)
        mkt[f"cos_{k}"] = np.cos(2 * np.pi * k * t / DATASET.fourier_period)

    # Log-transform broadcast channels (large-scale impression counts)
    for col in DATASET.broadcast_channels:
        if col in mkt.columns:
            mkt[f"{col}_log"] = np.log1p(mkt[col])

    return mkt


def split(mkt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = DATASET.train_weeks
    return mkt.iloc[:n_train].copy(), mkt.iloc[n_train:].copy()


def data_prep() -> None:
    print("=" * 60)
    print("  01  DATA PREP")
    print("=" * 60)

    print(f"\nLoading {GOLD_LABELLED} ...")

    mkt = aggregate_to_market(GOLD_LABELLED)
    print(f"  After aggregation: {len(mkt):,} market-level rows")

    mkt[DATASET.week_col] = pd.to_datetime(mkt[DATASET.week_col])

    # Completeness check — warn and forward-fill if weeks are missing
    actual_weeks = mkt[DATASET.week_col].nunique()
    expected_weeks = DATASET.train_weeks + (len(mkt) - DATASET.train_weeks)
    if actual_weeks < len(mkt):
        print(f"  [warn] {actual_weeks} unique weeks in {len(mkt)} rows — "
              "forward-filling gaps ...")
        full_spine = pd.date_range(
            mkt[DATASET.week_col].min(), mkt[DATASET.week_col].max(), freq="W-MON")
        mkt = (mkt.set_index(DATASET.week_col)
                  .reindex(full_spine)
                  .ffill()
                  .reset_index()
                  .rename(columns={"index": DATASET.week_col}))
    else:
        print(f"  Week spine complete: {actual_weeks} weeks "
              f"({mkt[DATASET.week_col].min().date()} -> "
              f"{mkt[DATASET.week_col].max().date()})")

    mkt = add_features(mkt)
    train, test = split(mkt)
    mkt["split"] = "train"
    mkt.loc[mkt.index >= DATASET.train_weeks, "split"] = "test"

    log_col = DATASET.target_log_col
    print(f"\n  Train: {len(train)} weeks | Test: {len(test)} weeks")
    print(f"  {log_col} range: [{mkt[log_col].min():.2f}, {mkt[log_col].max():.2f}]")
    print(f"  Columns: {list(mkt.columns)}")

    write_parquet(mkt, MARKET_SERIES)
    print(f"\n  Written -> {MARKET_SERIES}")
    print("=" * 60)


if __name__ == "__main__":
    data_prep()
