from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required: list[str], target_col: str) -> None:
    missing = [c for c in required + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    val_ratio_on_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_on_train_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_s": X_train_s,
        "X_val_s": X_val_s,
        "X_test_s": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": le,
        "scaler": scaler,
    }
