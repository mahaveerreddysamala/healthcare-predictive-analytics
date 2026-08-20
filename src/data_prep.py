"""Data validation and feature preparation utilities."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "readmitted"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, remove duplicate rows, and validate the target."""
    data = df.copy()
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    data = data.drop_duplicates().reset_index(drop=True)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing")
    data = data.dropna(subset=[TARGET])
    return data


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Create reproducible stratified train/test splits."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
