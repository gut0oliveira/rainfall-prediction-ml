"""Train and compare models that predict next-day rainfall in Sydney."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/"
    "Weather_Data.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Weather_Data.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics.csv"
TARGET = "RainTomorrow"


def download_dataset(destination: Path) -> None:
    """Download the source dataset when it is not available locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Downloading dataset to {path}...")
        download_dataset(path)

    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")
    return data


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=[TARGET, "Date"], errors="ignore")
    target = data[TARGET].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Target contains values other than 'Yes' and 'No'.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2_000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Support Vector Machine": SVC(kernel="linear"),
    }

    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(features)), ("model", estimator)]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "jaccard": jaccard_score(y_test, predictions, zero_division=0),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    results = evaluate_models(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, float_format="%.4f")

    print(f"Dataset: {len(data):,} observations")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
