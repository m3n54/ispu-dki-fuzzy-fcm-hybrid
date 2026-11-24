"""Data acquisition, cleaning, leak-free next-day pairing, and chronological splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = ["pm10", "pm25", "so2", "co", "o3", "no2"]
FEATURE_LABELS: dict[str, str] = {
    "pm10": "PM10",
    "pm25": "PM2.5",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "no2": "NO2",
}
CANONICAL_SPKU_COLS: list[str] = [
    "tanggal",
    "stasiun",
    "pm10",
    "pm25",
    "so2",
    "co",
    "o3",
    "no2",
    "max",
    "critical",
    "categori",
]

COLUMN_ALIASES: dict[str, str] = {
    "lokasi_spku": "stasiun",
    "pm_sepuluh": "pm10",
    "pm_duakomalima": "pm25",
    "kategori": "categori",
}


class StandardScalerNumpy:
    """Pure NumPy Z-score feature scaler with JSON serialization."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        self.mean_ = None if mean is None else np.asarray(mean, dtype=float)
        self.std_ = None if std is None else np.asarray(std, dtype=float)

    def fit(self, X: np.ndarray) -> "StandardScalerNumpy":
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return np.asarray(X_scaled, dtype=float) * self.std_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before serialization.")
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardScalerNumpy":
        return cls(mean=np.asarray(data["mean"], dtype=float), std=np.asarray(data["std"], dtype=float))


def _repair_shifted_spku_header(df: pd.DataFrame) -> pd.DataFrame:
    """Fix Jakarta Open Data CSVs (June/July 2021) where headers were shifted right by 1 column."""
    cols = [str(c).strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols
    if "location" in cols and "stasiun" not in cols and len(cols) == 11:
        # Check if 'pm10' column actually contains station strings like 'DKI1 ...'
        sample_vals = df["pm10"].dropna().astype(str).head(10).tolist()
        if any("DKI" in v for v in sample_vals):
            df.columns = CANONICAL_SPKU_COLS
    return df


def load_raw_ispu(path_or_dir: str | Path) -> pd.DataFrame:
    """Load raw ISPU CSV file or directory of monthly SPKU CSV files."""
    target = Path(path_or_dir)
    if target.is_file():
        df = pd.read_csv(target)
        return _repair_shifted_spku_header(df)

    spku_files = sorted(target.glob("*di-spku-bulan-*.csv"))
    if not spku_files:
        spku_files = sorted(target.glob("*.csv"))
    if not spku_files:
        raise FileNotFoundError(f"No CSV files found in {target}")

    frames = []
    for f in spku_files:
        part = pd.read_csv(f)
        part = _repair_shifted_spku_header(part)
        part["source_file"] = f.name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def clean_ispu_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize columns, coerce pollutant values, remove missing/invalid/duplicate rows."""
    raw_rows = int(len(df))
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    work = work.rename(columns=COLUMN_ALIASES)

    if "stasiun" not in work.columns or "tanggal" not in work.columns:
        raise ValueError("Dataset must contain 'tanggal' and 'stasiun' columns.")

    work["tanggal"] = pd.to_datetime(work["tanggal"], errors="coerce")
    work["stasiun"] = work["stasiun"].astype(str).str.strip()

    numeric_cols = [*FEATURE_COLS, "max"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "categori" not in work.columns:
        work["categori"] = ""
    work["categori"] = work["categori"].astype(str).str.strip().str.upper()

    valid_mask = work["tanggal"].notna() & (work["stasiun"] != "") & (work["stasiun"] != "nan")
    for col in numeric_cols:
        valid_mask &= work[col].notna() & (work[col] >= 0.0) & (work[col] <= 500.0)

    filtered = work.loc[valid_mask].copy()
    dropped_invalid = raw_rows - int(len(filtered))

    before_dedup = len(filtered)
    filtered = filtered.sort_values(["stasiun", "tanggal"]).drop_duplicates(
        subset=["stasiun", "tanggal"], keep="first"
    )
    dropped_duplicates = before_dedup - int(len(filtered))
    filtered = filtered.reset_index(drop=True)

    stats = {
        "raw_rows": raw_rows,
        "dropped_invalid_or_missing": int(dropped_invalid),
        "dropped_duplicates": int(dropped_duplicates),
        "clean_rows": int(len(filtered)),
        "stations": sorted(filtered["stasiun"].unique().tolist()),
        "date_min": str(filtered["tanggal"].min().date()) if len(filtered) else None,
        "date_max": str(filtered["tanggal"].max().date()) if len(filtered) else None,
    }
    return filtered, stats


def build_next_day_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Pair day t pollutants with day t+1 ISPU max strictly within the same station (delta == 1 day)."""
    ordered = df.sort_values(["stasiun", "tanggal"]).copy()
    ordered["next_tanggal"] = ordered.groupby("stasiun")["tanggal"].shift(-1)
    ordered["target_ispu_next"] = ordered.groupby("stasiun")["max"].shift(-1)
    ordered["target_category_next"] = ordered.groupby("stasiun")["categori"].shift(-1)

    delta_days = (ordered["next_tanggal"] - ordered["tanggal"]).dt.days
    paired = ordered.loc[delta_days == 1].copy()
    paired = paired.sort_values(["tanggal", "stasiun"]).reset_index(drop=True)
    return paired


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> dict[str, Any]:
    """Chronologically split by unique dates so train < val < test with zero temporal overlap."""
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios.")

    unique_dates = np.array(sorted(df["tanggal"].unique()))
    n_dates = len(unique_dates)
    if n_dates < 5:
        raise ValueError("Need at least 5 distinct dates to perform chronological 3-way split.")

    tr_end = max(1, int(np.floor(n_dates * train_ratio)))
    val_end = max(tr_end + 1, int(np.floor(n_dates * (train_ratio + val_ratio))))
    if val_end >= n_dates:
        val_end = n_dates - 1

    train_dates = set(unique_dates[:tr_end])
    val_dates = set(unique_dates[tr_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_df = df.loc[df["tanggal"].isin(train_dates)].sort_values(["tanggal", "stasiun"]).reset_index(drop=True)
    val_df = df.loc[df["tanggal"].isin(val_dates)].sort_values(["tanggal", "stasiun"]).reset_index(drop=True)
    test_df = df.loc[df["tanggal"].isin(test_dates)].sort_values(["tanggal", "stasiun"]).reset_index(drop=True)

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["target_ispu_next"].to_numpy(dtype=float)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=float)
    y_val = val_df["target_ispu_next"].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df["target_ispu_next"].to_numpy(dtype=float)

    scaler = StandardScalerNumpy().fit(X_train)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": scaler.transform(X_train),
        "X_val_scaled": scaler.transform(X_val),
        "X_test_scaled": scaler.transform(X_test),
        "scaler": scaler,
        "boundaries": {
            "train_start": str(train_df["tanggal"].min().date()),
            "train_end": str(train_df["tanggal"].max().date()),
            "val_start": str(val_df["tanggal"].min().date()),
            "val_end": str(val_df["tanggal"].max().date()),
            "test_start": str(test_df["tanggal"].min().date()),
            "test_end": str(test_df["tanggal"].max().date()),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
    }
