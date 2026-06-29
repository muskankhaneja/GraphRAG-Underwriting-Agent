"""
data_engineering.py
--------------------
Ingests the Kaggle S&P 500 Corporate Credit Rating with Financial Ratios dataset,
maps raw S&P rating labels to a binary risk target, scales features, performs a
stratified train/test split, and exports processed parquet files.

The `Ticker` column is preserved in a separate metadata parquet so that Phase 2
(GraphRAG / EDGAR ingestion) can use it for entity linking without leaking it
into the model feature matrix.
"""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rating → binary target mapping
# ---------------------------------------------------------------------------
INVESTMENT_GRADE = {
    "AAA", "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
}

SPECULATIVE_GRADE = {
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC+", "CCC", "CCC-",
    "CC", "C", "D",
}

ALL_KNOWN_RATINGS = INVESTMENT_GRADE | SPECULATIVE_GRADE

# 25 numeric feature columns expected in the dataset
FEATURE_COLUMNS = [
    "currentRatio",
    "quickRatio",
    "cashRatio",
    "daysOfSalesOutstanding",
    "netProfitMargin",
    "pretaxProfitMargin",
    "grossProfitMargin",
    "operatingProfitMargin",
    "returnOnAssets",
    "returnOnEquity",
    "returnOnCapitalEmployed",
    "netIncomePerEBT",
    "ebtPerEbit",
    "ebitPerRevenue",
    "debtRatio",
    "debtEquityRatio",
    "longTermDebtToCapitalization",
    "totalDebtToCapitalization",
    "interestCoverage",
    "cashFlowToDebtRatio",
    "companyEquityMultiplier",
    "receivablesTurnover",
    "payablesTurnover",
    "inventoryTurnover",
    "fixedAssetTurnover",
    "assetTurnover",
]

METADATA_COLUMNS = ["Ticker", "Name", "Sector", "Rating"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_dataset() -> Path:
    """Download the S&P 500 credit rating dataset via the Kaggle CLI."""
    dataset_slug = "kirtandelwadia/sp-500-with-financial-ratios"
    zip_path = RAW_DIR / "sp500-credit-ratings.zip"

    if not zip_path.exists():
        print(f"[data_engineering] Downloading dataset: {dataset_slug}")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_slug,
                "-p", str(RAW_DIR),
            ],
            check=True,
        )
        # Kaggle CLI names the zip after the dataset slug tail
        downloaded = list(RAW_DIR.glob("*.zip"))
        if not downloaded:
            raise FileNotFoundError(
                "Kaggle download produced no zip file. Check your kaggle.json credentials."
            )
        downloaded[0].rename(zip_path)

    csv_path = RAW_DIR / "ratings.csv"
    if not csv_path.exists():
        print("[data_engineering] Extracting zip …")
        with zipfile.ZipFile(zip_path, "r") as z:
            # Extract the first CSV found in the archive
            csv_members = [m for m in z.namelist() if m.endswith(".csv")]
            if not csv_members:
                raise ValueError("No CSV found inside the downloaded zip.")
            z.extract(csv_members[0], RAW_DIR)
            extracted = RAW_DIR / csv_members[0]
            if extracted != csv_path:
                extracted.rename(csv_path)

    return csv_path


def _map_rating_to_binary(rating: str) -> int | None:
    """Return 0 (investment grade) or 1 (speculative grade), or None if unknown."""
    if rating in INVESTMENT_GRADE:
        return 0
    if rating in SPECULATIVE_GRADE:
        return 1
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full data-engineering pipeline.

    Returns
    -------
    X_train, X_test : DataFrames of scaled feature columns
    y_train, y_test : Series of binary targets (0 / 1)
    """
    if csv_path is None:
        csv_path = _download_dataset()

    print(f"[data_engineering] Loading {csv_path} …")
    df = pd.read_csv(csv_path)

    # Normalise column names (dataset variants differ in casing)
    df.columns = [c.strip() for c in df.columns]

    # ---- Validate Rating column ----
    if "Rating" not in df.columns:
        raise ValueError("Expected column 'Rating' not found. Check dataset version.")

    df["Rating"] = df["Rating"].str.strip()
    df["binary_target"] = df["Rating"].apply(_map_rating_to_binary)

    unknown_ratings = df.loc[df["binary_target"].isna(), "Rating"].unique()
    if len(unknown_ratings) > 0:
        print(f"[data_engineering] Dropping {df['binary_target'].isna().sum()} rows with "
              f"unrecognised ratings: {unknown_ratings}")
    df = df.dropna(subset=["binary_target"]).copy()
    df["binary_target"] = df["binary_target"].astype(int)

    # ---- Feature matrix ----
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = set(FEATURE_COLUMNS) - set(available_features)
    if missing_features:
        print(f"[data_engineering] WARNING: {len(missing_features)} expected feature columns "
              f"not found in dataset and will be skipped: {missing_features}")

    # Drop rows missing more than 30 % of features
    feature_df = df[available_features].copy()
    missing_frac = feature_df.isna().mean(axis=1)
    excessive_missing = missing_frac > 0.30
    if excessive_missing.sum() > 0:
        print(f"[data_engineering] Dropping {excessive_missing.sum()} rows with >30 % missing features.")
    df = df.loc[~excessive_missing].copy()
    feature_df = df[available_features].copy()

    # ---- Metadata (preserved for Phase 2 entity linking) ----
    available_meta = [c for c in METADATA_COLUMNS if c in df.columns]
    metadata_df = df[available_meta].copy()

    y = df["binary_target"]

    # ---- Stratified split (before any fitting) ----
    (X_raw_train, X_raw_test,
     y_train, y_test,
     meta_train, meta_test) = train_test_split(
        feature_df, y, metadata_df,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ---- Median imputation (fit on train only) ----
    train_medians = X_raw_train.median()
    X_raw_train = X_raw_train.fillna(train_medians)
    X_raw_test = X_raw_test.fillna(train_medians)

    # ---- Scaling (fit on train only) ----
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw_train),
        columns=available_features,
        index=X_raw_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_raw_test),
        columns=available_features,
        index=X_raw_test.index,
    )

    # ---- Persist artifacts ----
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_columns": available_features, "train_medians": train_medians.to_dict()}, f)
    print(f"[data_engineering] Scaler saved → {scaler_path}")

    X_train_scaled.to_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test_scaled.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train.to_frame("binary_target").to_parquet(PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame("binary_target").to_parquet(PROCESSED_DIR / "y_test.parquet")
    meta_test.to_parquet(PROCESSED_DIR / "metadata_test.parquet")

    class_dist = y_train.value_counts(normalize=True).round(3).to_dict()
    print(f"[data_engineering] Train class distribution: {class_dist}")
    print(f"[data_engineering] Train rows: {len(X_train_scaled)} | Test rows: {len(X_test_scaled)}")
    print(f"[data_engineering] Features used: {len(available_features)}")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    load_and_prepare()
