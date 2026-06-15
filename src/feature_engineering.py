"""
src/feature_engineering.py
==========================
All feature construction logic lives here.

Key design decisions:
- Lag and rolling features are always computed on the SORTED DataFrame, using only
  information from the past (shift > 0, so no leakage into the current row).
- If lags of the TARGET variable are included, we document that explicitly and
  ensure they use past-only data (shift ≥ 1).
- Temporal split is done here so that scalers are fit ONLY on the training set.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import CFG
except ImportError:
    from config import CFG


# ---------------------------------------------------------------------------
# 1. Temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the DatetimeIndex.

    These capture seasonality and shift patterns in the plant.
    """
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["day"] = idx.day
    df["month"] = idx.month
    df["dayofweek"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    cols: list[str],
    lag_periods: list[int],
) -> pd.DataFrame:
    """Add lagged values for a list of columns.

    Using shift(n) with n >= 1 guarantees that at any row t, we only see
    values from time t-n — no leakage.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of column names to lag
    lag_periods : list of positive integers (number of rows to lag)

    Returns
    -------
    pd.DataFrame with new {col}_lag_{n} columns appended.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lag_periods:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# 3. Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add rolling mean, std, and diff features.

    We use shift(1) before computing rolling stats to ensure each window
    contains only past data — the current observation is excluded.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of column names
    windows : list of window sizes in rows
    min_periods : minimum observations required in window

    Returns
    -------
    pd.DataFrame with rolling feature columns appended.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        shifted = df[col].shift(1)  # shift first to avoid current-row leakage
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = (
                shifted.rolling(window=w, min_periods=min_periods).mean()
            )
            df[f"{col}_roll_std_{w}"] = (
                shifted.rolling(window=w, min_periods=min_periods).std()
            )
        df[f"{col}_diff_1"] = df[col].diff(1)  # diff does not look forward
    return df


# ---------------------------------------------------------------------------
# 4. Temporal split (no randomness)
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train / validation / test using temporal order.

    Why NOT random split?
    ---------------------
    This is a time-series problem. A random split would allow the model to use
    future information (e.g. rows from March) to predict past observations
    (e.g. rows from January). This constitutes data leakage and would produce
    over-optimistic metrics that do not reflect real operational performance.

    Parameters
    ----------
    df : pd.DataFrame (must be sorted chronologically)
    train_ratio : float, e.g. 0.70
    val_ratio : float, e.g. 0.15

    Returns
    -------
    train, val, test DataFrames (non-overlapping, in order)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(
        f"[feature_engineering] Split sizes — "
        f"train: {len(train)} ({train_ratio:.0%}), "
        f"val: {len(val)} ({val_ratio:.0%}), "
        f"test: {len(test)} ({1 - train_ratio - val_ratio:.0%})"
    )
    print(
        f"  Train: {train.index[0]} → {train.index[-1]}\n"
        f"  Val:   {val.index[0]} → {val.index[-1]}\n"
        f"  Test:  {test.index[0]} → {test.index[-1]}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# 5. Build feature matrix X and target vector y
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cfg: dict | None = None,
    include_target_lags: bool = True,
    include_feed_features: bool = True,
) -> pd.DataFrame:
    """Orchestrate all feature engineering steps on a given split.

    IMPORTANT — leakage guard:
    --------------------------
    Lab measurements (`% Iron Concentrate`, `% Silica Concentrate`) arrive
    with a delay in the real process. They MUST NOT be used as input features
    because at inference time they would not yet be available.
    Only lag versions of the target (shift >= 1) are allowed, and only when
    `include_target_lags=True`.

    Parameters
    ----------
    df : pd.DataFrame (one of train / val / test — already split)
    cfg : dict, optional
    include_target_lags : bool
        Whether to add lagged values of the target. Set to False for strict
        no-leakage baseline comparisons.
    include_feed_features : bool
        Whether to include `% Iron Feed` and `% Silica Feed` as predictors.
        Useful for sensitivity analysis when their real-time availability is
        uncertain operationally.

    Returns
    -------
    pd.DataFrame with all engineered features.
    """
    cfg = cfg or CFG
    target = cfg["target"]
    lab_cols = cfg["feature_groups"]["lab_measurements"]
    lag_periods = cfg["feature_engineering"]["lag_periods"]
    rolling_windows = cfg["feature_engineering"]["rolling_windows"]

    fg = cfg["feature_groups"]
    feed_cols = fg["feed"] if include_feed_features else []
    operational_cols = (
        feed_cols + fg["flow"] + fg["pulp"] + fg["flotation_columns"]
    )

    df = add_temporal_features(df)
    df = add_lag_features(df, cols=operational_cols, lag_periods=lag_periods)
    df = add_rolling_features(df, cols=operational_cols, windows=rolling_windows)

    # Target lags — only past values, lag >= 1
    if include_target_lags and target in df.columns:
        df = add_lag_features(df, cols=[target], lag_periods=lag_periods)

    # Drop lab measurement columns (available only with delay)
    cols_to_drop = [c for c in lab_cols if c in df.columns and c != target]
    df = df.drop(columns=cols_to_drop)

    return df


# ---------------------------------------------------------------------------
# 6. Prepare X, y matrices
# ---------------------------------------------------------------------------

def prepare_Xy(
    df_engineered: pd.DataFrame,
    cfg: dict | None = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) from an engineered DataFrame.

    Drops rows with NaN values introduced by lags/rolling (initial rows).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    cfg = cfg or CFG
    target = cfg["target"]

    if drop_na:
        df_engineered = df_engineered.dropna()

    y = df_engineered[target]
    X = df_engineered.drop(columns=[target])
    return X, y


# ---------------------------------------------------------------------------
# 7. Scale features (fit on train, transform val/test)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame]:
    """Fit a StandardScaler on training features only."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return scaler, X_scaled


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a pre-fitted scaler to val or test features."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


# ---------------------------------------------------------------------------
# 8. Full feature engineering pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering_pipeline(
    df_clean: pd.DataFrame,
    cfg: dict | None = None,
    save: bool = True,
    include_target_lags: bool = True,
    include_feed_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Full pipeline: build features → split → prepare X/y → save.

    Returns
    -------
    X_train, X_val, X_test, scaler, y_train, y_val, y_test
    (note: scaler is a StandardScaler fit only on X_train)
    """
    cfg = cfg or CFG
    split_cfg = cfg["split"]
    processed_path = Path(cfg["paths"]["data_processed"])

    # 1. Engineer features on the full dataset (preserving temporal order)
    df_feat = build_features(
        df_clean,
        cfg=cfg,
        include_target_lags=include_target_lags,
        include_feed_features=include_feed_features,
    )

    # 2. Temporal split
    train_df, val_df, test_df = temporal_split(
        df_feat,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
    )

    # 3. Prepare X, y — NaN rows dropped within each split independently
    X_train, y_train = prepare_Xy(train_df, cfg=cfg)
    X_val, y_val = prepare_Xy(val_df, cfg=cfg)
    X_test, y_test = prepare_Xy(test_df, cfg=cfg)

    # Align columns: val/test must have same columns as train after dropna
    common_cols = X_train.columns
    X_val = X_val.reindex(columns=common_cols)
    X_test = X_test.reindex(columns=common_cols)

    if save:
        X_train.to_parquet(processed_path / "X_train.parquet")
        X_val.to_parquet(processed_path / "X_val.parquet")
        X_test.to_parquet(processed_path / "X_test.parquet")
        y_train.to_frame().to_parquet(processed_path / "y_train.parquet")
        y_val.to_frame().to_parquet(processed_path / "y_val.parquet")
        y_test.to_frame().to_parquet(processed_path / "y_test.parquet")
        print(f"[feature_engineering] Processed splits saved to: {processed_path}")

    return X_train, X_val, X_test, y_train, y_val, y_test
