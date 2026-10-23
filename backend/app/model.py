from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "data" / "predictive_maintenance.csv",
    Path(__file__).resolve().parents[2] / "data" / "predictive_maintenance.csv",
)
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = ["Type", *NUMERIC_FEATURES]
TARGET = "Failure Type"
REQUIRED_COLUMNS = [*FEATURES, "Target", TARGET]
PRODUCT_TYPES = {"L", "M", "H"}


@dataclass
class ModelBundle:
    pipeline: ImbPipeline
    accuracy: float
    report: dict[str, Any]
    class_counts: dict[str, int]
    row_count: int
    feature_ranges: dict[str, dict[str, float]]
    non_failure_ranges: dict[str, dict[str, float]]
    feature_importances: list[dict[str, float | str]]


def prepare_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw_df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    df = raw_df.loc[:, REQUIRED_COLUMNS].copy()
    df = df.loc[df[TARGET] != "Random Failures"]
    df = df.loc[~((df["Target"] == 1) & (df[TARGET] == "No Failure"))]
    df = df.dropna(subset=REQUIRED_COLUMNS)
    for column in NUMERIC_FEATURES + ["Target"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES + ["Target"])
    df["Type"] = df["Type"].astype(str)
    df[TARGET] = df[TARGET].astype(str)

    if df.empty or df[TARGET].nunique() < 2:
        raise ValueError("The cleaned dataset must contain at least two failure classes.")
    if df[TARGET].value_counts().min() < 2:
        raise ValueError("Each failure class needs at least two rows for stratified validation.")
    if not set(df["Type"]).issubset(PRODUCT_TYPES):
        raise ValueError("Type values must be L, M, or H.")
    return df


def train_model(csv_bytes: bytes) -> ModelBundle:
    df = prepare_dataset(pd.read_csv(BytesIO(csv_bytes)))
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ]
    )
    pipeline = ImbPipeline(
        steps=[
            ("balance", SMOTENC(categorical_features=[0], random_state=42)),
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    ranges = {
        feature: {
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "median": float(df[feature].median()),
        }
        for feature in NUMERIC_FEATURES
    }
    non_failure = df.loc[df[TARGET] == "No Failure"]
    non_failure_ranges = {
        feature: {
            "lower": float(non_failure[feature].quantile(0.05)),
            "upper": float(non_failure[feature].quantile(0.95)),
        }
        for feature in NUMERIC_FEATURES
    }
    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importances = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(
            transformed_feature_names,
            pipeline.named_steps["classifier"].feature_importances_,
        )
    ]
    feature_importances.sort(key=lambda item: float(item["importance"]), reverse=True)
    return ModelBundle(
        pipeline=pipeline,
        accuracy=float(accuracy_score(y_test, predictions)),
        report=report,
        class_counts={str(key): int(value) for key, value in df[TARGET].value_counts().items()},
        row_count=len(df),
        feature_ranges=ranges,
        non_failure_ranges=non_failure_ranges,
        feature_importances=feature_importances,
    )


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    data_path = next((path for path in DATA_PATH_CANDIDATES if path.is_file()), None)
    if data_path is None:
        checked_paths = ", ".join(str(path) for path in DATA_PATH_CANDIDATES)
        raise FileNotFoundError(f"Demo dataset not found. Checked: {checked_paths}")
    return train_model(data_path.read_bytes())
