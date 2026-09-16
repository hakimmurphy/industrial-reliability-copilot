import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text


DEFAULT_PROCESSED = Path("data/processed/ai4i2020_clean.parquet")
DEFAULT_MODEL_OUT = Path("app/models/baseline_failure_model.joblib")
DEFAULT_METRICS_OUT = Path("app/evaluation/baseline_metrics.json")

FEATURES = [
    "machine_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "delta_temp_k",
    "power_proxy",
]
TARGET = "machine_failure"
CATEGORICAL_FEATURES = ["machine_type"]
NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "delta_temp_k",
    "power_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline machine-failure model")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL URL (optional)")
    parser.add_argument("--source-dataset", default="AI4I", help="Dataset name filter for DB load")
    parser.add_argument("--processed-path", default=str(DEFAULT_PROCESSED), help="Fallback parquet path")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT), help="Path to save model artifact")
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS_OUT), help="Path to save metrics JSON")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def load_from_database(database_url: str, source_dataset: str) -> pd.DataFrame:
    query = text(
        """
        SELECT
            o.machine_type,
            o.air_temperature_k,
            o.process_temperature_k,
            o.rotational_speed_rpm,
            o.torque_nm,
            o.tool_wear_min,
            o.delta_temp_k,
            o.power_proxy,
            l.machine_failure
        FROM sensor_observations_ai4i o
        JOIN failure_labels_ai4i l ON l.observation_id = o.observation_id
        WHERE o.source_dataset = :source_dataset
        """
    )

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"source_dataset": source_dataset})
    finally:
        engine.dispose()


def load_from_parquet(processed_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(processed_path)
    required = {
        "machine_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in fallback parquet: {sorted(missing)}")

    if "delta_temp_k" not in df.columns:
        df["delta_temp_k"] = df["process_temperature_k"] - df["air_temperature_k"]
    if "power_proxy" not in df.columns:
        df["power_proxy"] = df["torque_nm"] * df["rotational_speed_rpm"]

    return df


def load_training_data(database_url: str | None, source_dataset: str, processed_path: Path) -> tuple[pd.DataFrame, str]:
    if database_url:
        try:
            df = load_from_database(database_url=database_url, source_dataset=source_dataset)
            if not df.empty:
                return df, "database"
        except Exception:
            pass

    if not processed_path.exists():
        raise FileNotFoundError(
            f"No training data found. DB unavailable and parquet missing at {processed_path}"
        )
    return load_from_parquet(processed_path), "parquet"


def build_models(random_state: int) -> dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "precision_at_0_5": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix_at_0_5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main() -> None:
    args = parse_args()
    processed_path = Path(args.processed_path)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)

    df, data_source = load_training_data(
        database_url=args.database_url,
        source_dataset=args.source_dataset,
        processed_path=processed_path,
    )

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    x = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_models(random_state=args.random_state)
    results: dict[str, dict] = {}
    best_name = ""
    best_pr_auc = -1.0
    best_model: Pipeline | None = None

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[name] = metrics

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained.")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_source": data_source,
        "rows_total": int(len(df)),
        "positive_rate": float(y.mean()),
        "features": FEATURES,
        "target": TARGET,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "models": results,
        "selected_model": best_name,
        "selected_model_pr_auc": best_pr_auc,
        "selected_threshold": 0.5,
        "model_artifact": str(model_out),
    }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
