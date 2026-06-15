"""
src/data_preprocessing.py
=========================
Functions for loading, cleaning, and validating the raw dataset.

Design principles:
- All operations are pure functions with no side effects.
- No global state; callers pass cfg = load_config().
- Temporal integrity is enforced from the first load step.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from .config import CFG
except ImportError:
    from config import CFG


def _download_raw_from_kagglehub(raw_path: Path) -> Path:
    """Download dataset via kagglehub and copy the CSV to data/raw/.

    Parameters
    ----------
    raw_path : Path
        Expected destination CSV path under data/raw/.

    Returns
    -------
    Path
        Path to the downloaded (or copied) CSV in data/raw/.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    dataset_dir = Path(
        kagglehub.dataset_download("edumagalhaes/quality-prediction-in-a-mining-process")
    )

    csv_candidates = sorted(dataset_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"KaggleHub download succeeded but no CSV file was found in: {dataset_dir}"
        )

    source_csv = None
    for candidate in csv_candidates:
        if "mining" in candidate.name.lower() or "flotation" in candidate.name.lower():
            source_csv = candidate
            break
    if source_csv is None:
        source_csv = csv_candidates[0]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, raw_path)
    print(f"[data_preprocessing] Downloaded dataset from KaggleHub: {source_csv}")
    print(f"[data_preprocessing] Copied CSV to: {raw_path}")
    return raw_path


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_raw_data(cfg: dict | None = None) -> pd.DataFrame:
    """Load the raw CSV file and return a DataFrame without any transformation.

    The dataset uses a comma as the decimal separator (Brazilian locale).
    We handle that transparently here so callers receive numeric columns.

    Parameters
    ----------
    cfg : dict, optional
        Project config. Defaults to the global CFG singleton.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all columns as-is (strings for numerics with comma decimal
        are converted to float).
    """
    cfg = cfg or CFG
    raw_path = Path(cfg["paths"]["data_raw"]) / cfg["data"]["raw_filename"]

    if not raw_path.exists():
        print(
            "[data_preprocessing] Raw CSV not found in data/raw/. "
            "Attempting auto-download via kagglehub..."
        )
        raw_path = _download_raw_from_kagglehub(raw_path)

    df = pd.read_csv(raw_path, sep=",", decimal=",", dayfirst=False)
    return df


# ---------------------------------------------------------------------------
# 2. Datetime parsing and sorting
# ---------------------------------------------------------------------------

def parse_and_sort_datetime(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Parse the datetime column and sort the DataFrame chronologically.

    Temporal ordering is mandatory for this problem:
    - We must avoid any information from the future leaking into training.
    - Rolling features, lags, and splits all depend on strict chronological order.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : dict, optional

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by datetime with a proper DatetimeIndex.
    """
    cfg = cfg or CFG
    dt_col = cfg["data"]["datetime_column"]

    # The date column may contain a decimal comma too; cast to string first
    df[dt_col] = df[dt_col].astype(str).str.strip()
    df[dt_col] = pd.to_datetime(df[dt_col], format=cfg["data"]["datetime_format"])

    df = df.sort_values(dt_col).reset_index(drop=True)
    df = df.set_index(dt_col)
    return df


# ---------------------------------------------------------------------------
# 3. Column name normalization
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names.

    Whitespace in column names causes subtle bugs (column lookups fail silently).
    """
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# 4. Data quality report
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a summary of data quality indicators per column.

    Returns
    -------
    pd.DataFrame
        One row per column with: dtype, n_missing, pct_missing, n_duplicated_rows,
        n_unique, min, max, mean, std.
    """
    report = pd.DataFrame(index=df.columns)
    report["dtype"] = df.dtypes
    report["n_missing"] = df.isnull().sum()
    report["pct_missing"] = (df.isnull().mean() * 100).round(2)
    report["n_unique"] = df.nunique()

    num_cols = df.select_dtypes(include=[np.number]).columns
    report.loc[num_cols, "min"] = df[num_cols].min()
    report.loc[num_cols, "max"] = df[num_cols].max()
    report.loc[num_cols, "mean"] = df[num_cols].mean().round(4)
    report.loc[num_cols, "std"] = df[num_cols].std().round(4)

    # Duplicate rows (overall dataset)
    n_dup = df.duplicated().sum()
    report["n_dup_rows_total"] = n_dup  # same value for all rows; informational

    return report


# ---------------------------------------------------------------------------
# 5. Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Flag outliers using the IQR method.

    We use a wider factor (3× instead of 1.5×) because sensor data in industrial
    processes has legitimate extreme values during process disturbances.

    Returns
    -------
    pd.DataFrame
        Boolean mask DataFrame (True = outlier) for numeric columns only.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[num_cols] < (Q1 - factor * IQR)) | (df[num_cols] > (Q3 + factor * IQR))
    return mask


def outlier_summary(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Return per-column outlier counts and percentages."""
    mask = detect_outliers_iqr(df, factor=factor)
    summary = pd.DataFrame({
        "n_outliers": mask.sum(),
        "pct_outliers": (mask.mean() * 100).round(2),
    })
    return summary.sort_values("n_outliers", ascending=False)


# ---------------------------------------------------------------------------
# 6. Remove duplicates
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (all columns identical).

    In a time-indexed DataFrame duplicate timestamps may represent sensor
    glitches where the same reading was recorded twice.
    """
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[data_preprocessing] Removed {n_removed} duplicate rows.")
    return df


# ---------------------------------------------------------------------------
# 7. Sampling frequency analysis
# ---------------------------------------------------------------------------

def sampling_frequency_report(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse the inter-sample time deltas to detect mixed sampling frequencies.

    Different variables in this dataset have different update frequencies:
    - Sensor variables: updated every ~20 minutes (operational data).
    - Lab measurements (% Iron Concentrate, % Silica Concentrate): updated ~once per
      hour or less — they arrive with a significant delay.

    This mismatch must be acknowledged and handled during feature engineering.

    Returns
    -------
    pd.DataFrame
        Frequency counts of inter-row time deltas in minutes.
    """
    deltas = df.index.to_series().diff().dt.total_seconds().div(60).dropna()
    freq_counts = deltas.value_counts().rename_axis("delta_minutes").reset_index()
    freq_counts.columns = ["delta_minutes", "count"]
    return freq_counts.sort_values("delta_minutes")


# ---------------------------------------------------------------------------
# 8. Full preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(cfg: dict | None = None) -> pd.DataFrame:
    """End-to-end preprocessing: load → normalize → parse → deduplicate → save.

    The cleaned DataFrame is saved to data/interim/.
    """
    cfg = cfg or CFG
    interim_path = Path(cfg["paths"]["data_interim"]) / "data_cleaned.parquet"

    df = load_raw_data(cfg)
    df = normalize_column_names(df)
    df = parse_and_sort_datetime(df, cfg)
    df = remove_duplicates(df)

    # Ensure all non-index columns are numeric (coerce errors to NaN)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    df.to_parquet(interim_path)
    print(f"[data_preprocessing] Cleaned data saved to: {interim_path}")
    print(f"[data_preprocessing] Shape: {df.shape}")
    return df
