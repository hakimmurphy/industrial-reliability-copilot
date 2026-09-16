import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DEFAULT_SOURCE = Path("data/ai4i2020.csv")
FALLBACK_SOURCE = Path("data/raw/ai4i2020.csv")
DEFAULT_PROCESSED = Path("data/processed/ai4i2020_clean.parquet")
DEFAULT_DATASET_NAME = "AI4I"

RENAME_MAP = {
    "UDI": "source_row_id",
    "Product ID": "product_id",
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

REQUIRED_RAW_COLUMNS = set(RENAME_MAP.keys())
REQUIRED_RENAMED_COLUMNS = {
    "source_row_id",
    "product_id",
    "machine_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
}


@dataclass
class IngestionStats:
    source_rows: int
    observation_rows_loaded: int
    label_rows_loaded: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def resolve_source_path(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if DEFAULT_SOURCE.exists():
        return DEFAULT_SOURCE
    return FALLBACK_SOURCE


def extract_csv(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError("AI4I source file not found. Put ai4i2020.csv in data/ or data/raw/.")

    logging.info("Extracting CSV from %s", source_path)
    return pd.read_csv(source_path)


def validate_raw_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["UDI"].isna().any():
        raise ValueError("Found null UDI values; cannot build source_row_id.")

    if df["UDI"].duplicated().any():
        dup_count = int(df["UDI"].duplicated().sum())
        raise ValueError(f"Found duplicated UDI values: {dup_count}")


def transform_ai4i(df: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    logging.info("Transforming AI4I rows")
    out = df.rename(columns=RENAME_MAP).copy()

    if missing := (REQUIRED_RENAMED_COLUMNS - set(out.columns)):
        raise ValueError(f"Missing transformed columns: {sorted(missing)}")

    out["source_dataset"] = source_dataset
    out["asset_id"] = pd.NA

    out["source_row_id"] = out["source_row_id"].astype(int)
    out["product_id"] = out["product_id"].astype(str)
    out["machine_type"] = out["machine_type"].astype(str)

    valid_machine_types = {"L", "M", "H"}
    invalid = set(out["machine_type"].unique()) - valid_machine_types
    if invalid:
        raise ValueError(f"Invalid machine_type values found: {sorted(invalid)}")

    out["air_temperature_k"] = out["air_temperature_k"].astype(float)
    out["process_temperature_k"] = out["process_temperature_k"].astype(float)
    out["rotational_speed_rpm"] = out["rotational_speed_rpm"].astype(int)
    out["torque_nm"] = out["torque_nm"].astype(float)
    out["tool_wear_min"] = out["tool_wear_min"].astype(int)

    for col in ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]:
        out[col] = out[col].astype(bool)

    return out


def split_observations_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_cols = [
        "source_dataset",
        "source_row_id",
        "asset_id",
        "product_id",
        "machine_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    label_cols = ["source_dataset", "source_row_id", "machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]

    return df[obs_cols].copy(), df[label_cols].copy()


def write_processed(df: pd.DataFrame, output_path: Path = DEFAULT_PROCESSED) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info("Wrote processed parquet to %s", output_path)


def _iter_batched_records(df: pd.DataFrame, batch_size: int = 2000):
    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def load_to_postgres(
    observations_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    database_url: str,
    *,
    batch_size: int,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
) -> None:
    logging.info("Loading observations and labels into PostgreSQL")
    engine = create_engine(database_url, pool_pre_ping=True)

    insert_observations_sql = text(
        """
        INSERT INTO sensor_observations_ai4i (
            source_dataset,
            source_row_id,
            asset_id,
            product_id,
            machine_type,
            air_temperature_k,
            process_temperature_k,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_min
        )
        VALUES (
            :source_dataset,
            :source_row_id,
            :asset_id,
            :product_id,
            :machine_type,
            :air_temperature_k,
            :process_temperature_k,
            :rotational_speed_rpm,
            :torque_nm,
            :tool_wear_min
        )
        ON CONFLICT (source_dataset, source_row_id) DO UPDATE
        SET
            asset_id = EXCLUDED.asset_id,
            product_id = EXCLUDED.product_id,
            machine_type = EXCLUDED.machine_type,
            air_temperature_k = EXCLUDED.air_temperature_k,
            process_temperature_k = EXCLUDED.process_temperature_k,
            rotational_speed_rpm = EXCLUDED.rotational_speed_rpm,
            torque_nm = EXCLUDED.torque_nm,
            tool_wear_min = EXCLUDED.tool_wear_min
        """
    )

    insert_labels_sql = text(
        """
        INSERT INTO failure_labels_ai4i (
            observation_id,
            machine_failure,
            twf,
            hdf,
            pwf,
            osf,
            rnf,
            label_source
        )
        SELECT
            o.observation_id,
            :machine_failure,
            :twf,
            :hdf,
            :pwf,
            :osf,
            :rnf,
            :source_dataset
        FROM sensor_observations_ai4i o
        WHERE o.source_dataset = :source_dataset
          AND o.source_row_id = :source_row_id
        ON CONFLICT (observation_id) DO UPDATE
        SET
            machine_failure = EXCLUDED.machine_failure,
            twf = EXCLUDED.twf,
            hdf = EXCLUDED.hdf,
            pwf = EXCLUDED.pwf,
            osf = EXCLUDED.osf,
            rnf = EXCLUDED.rnf,
            label_source = EXCLUDED.label_source,
            labeled_at = NOW()
        """
    )

    try:
        with engine.begin() as conn:
            if statement_timeout_ms > 0:
                conn.execute(text(f"SET LOCAL statement_timeout = {statement_timeout_ms}"))
            if lock_timeout_ms > 0:
                conn.execute(text(f"SET LOCAL lock_timeout = {lock_timeout_ms}"))

            total_obs = len(observations_df)
            total_lbl = len(labels_df)
            obs_batches = max((total_obs + batch_size - 1) // batch_size, 1)
            lbl_batches = max((total_lbl + batch_size - 1) // batch_size, 1)

            obs_loaded = 0
            lbl_loaded = 0
            phase_start = time.perf_counter()

            for idx, batch in enumerate(_iter_batched_records(observations_df, batch_size=batch_size), start=1):
                batch_start = time.perf_counter()
                conn.execute(insert_observations_sql, batch)
                obs_loaded += len(batch)
                elapsed = time.perf_counter() - batch_start
                logging.info(
                    "Observation batch %s/%s loaded (%s/%s rows, %.2fs)",
                    idx,
                    obs_batches,
                    obs_loaded,
                    total_obs,
                    elapsed,
                )

            for idx, batch in enumerate(_iter_batched_records(labels_df, batch_size=batch_size), start=1):
                batch_start = time.perf_counter()
                conn.execute(insert_labels_sql, batch)
                lbl_loaded += len(batch)
                elapsed = time.perf_counter() - batch_start
                logging.info(
                    "Label batch %s/%s loaded (%s/%s rows, %.2fs)",
                    idx,
                    lbl_batches,
                    lbl_loaded,
                    total_lbl,
                    elapsed,
                )

            total_elapsed = time.perf_counter() - phase_start
            logging.info(
                "PostgreSQL load phase complete in %.2fs (observations=%s, labels=%s)",
                total_elapsed,
                obs_loaded,
                lbl_loaded,
            )
    finally:
        engine.dispose()


def verify_load(database_url: str, source_dataset: str, expected_rows: int) -> IngestionStats:
    engine = create_engine(database_url, pool_pre_ping=True)
    obs_count_sql = text(
        """
        SELECT COUNT(*)
        FROM sensor_observations_ai4i
        WHERE source_dataset = :source_dataset
        """
    )
    label_count_sql = text(
        """
        SELECT COUNT(*)
        FROM failure_labels_ai4i l
        JOIN sensor_observations_ai4i o ON o.observation_id = l.observation_id
        WHERE o.source_dataset = :source_dataset
        """
    )

    try:
        with engine.connect() as conn:
            observation_rows_loaded = int(conn.execute(obs_count_sql, {"source_dataset": source_dataset}).scalar_one())
            label_rows_loaded = int(conn.execute(label_count_sql, {"source_dataset": source_dataset}).scalar_one())
    finally:
        engine.dispose()

    if observation_rows_loaded < expected_rows or label_rows_loaded < expected_rows:
        raise RuntimeError(
            "Load verification failed: database row counts are smaller than expected source rows. "
            f"expected={expected_rows}, observations={observation_rows_loaded}, labels={label_rows_loaded}"
        )

    return IngestionStats(
        source_rows=expected_rows,
        observation_rows_loaded=observation_rows_loaded,
        label_rows_loaded=label_rows_loaded,
    )


def run_ingestion(
    source_path: Path,
    source_dataset: str,
    processed_output: Path,
    database_url: str | None,
    *,
    batch_size: int,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
) -> IngestionStats:
    raw_df = extract_csv(source_path)
    validate_raw_schema(raw_df)

    transformed_df = transform_ai4i(raw_df, source_dataset=source_dataset)
    write_processed(transformed_df, processed_output)

    observations_df, labels_df = split_observations_and_labels(transformed_df)

    if not database_url:
        logging.info("DATABASE_URL not provided. Skipping database load and verification.")
        return IngestionStats(
            source_rows=len(transformed_df),
            observation_rows_loaded=0,
            label_rows_loaded=0,
        )

    load_to_postgres(
        observations_df=observations_df,
        labels_df=labels_df,
        database_url=database_url,
        batch_size=batch_size,
        statement_timeout_ms=statement_timeout_ms,
        lock_timeout_ms=lock_timeout_ms,
    )
    return verify_load(database_url=database_url, source_dataset=source_dataset, expected_rows=len(transformed_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI4I production-style ingestion pipeline")
    parser.add_argument("--source-path", default=None, help="Optional explicit path to ai4i2020.csv")
    parser.add_argument("--source-dataset", default=DEFAULT_DATASET_NAME, help="Source dataset name stored in DB")
    parser.add_argument(
        "--processed-output",
        default=str(DEFAULT_PROCESSED),
        help="Path to write transformed parquet",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL SQLAlchemy URL; if omitted, DB load is skipped",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows per batch for PostgreSQL writes (default: 2000)",
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=180000,
        help="PostgreSQL statement timeout in milliseconds; use 0 to disable (default: 180000)",
    )
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=30000,
        help="PostgreSQL lock timeout in milliseconds; use 0 to disable (default: 30000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(verbose=args.verbose)

    source_path = resolve_source_path(args.source_path)
    stats = run_ingestion(
        source_path=source_path,
        source_dataset=args.source_dataset,
        processed_output=Path(args.processed_output),
        database_url=args.database_url,
        batch_size=args.batch_size,
        statement_timeout_ms=args.statement_timeout_ms,
        lock_timeout_ms=args.lock_timeout_ms,
    )

    logging.info(
        "Ingestion complete: source_rows=%s, observations_in_db=%s, labels_in_db=%s",
        stats.source_rows,
        stats.observation_rows_loaded,
        stats.label_rows_loaded,
    )
