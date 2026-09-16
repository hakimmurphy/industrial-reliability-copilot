import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DEFAULT_SOURCE = Path("data/MetroPT3(AirCompressor).csv")
FALLBACK_SOURCE = Path("data/raw/MetroPT3(AirCompressor).csv")
DEFAULT_PROCESSED = Path("data/processed/metropt3_clean.parquet")
DEFAULT_DATASET_NAME = "METROPT3"

RENAME_MAP = {
    "Unnamed: 0": "source_row_id",
    "timestamp": "observed_at",
    "TP2": "tp2",
    "TP3": "tp3",
    "H1": "h1",
    "DV_pressure": "dv_pressure",
    "Reservoirs": "reservoirs",
    "Oil_temperature": "oil_temperature",
    "Motor_current": "motor_current",
    "COMP": "comp",
    "DV_eletric": "dv_electric",
    "Towers": "towers",
    "MPG": "mpg",
    "LPS": "lps",
    "Pressure_switch": "pressure_switch",
    "Oil_level": "oil_level",
    "Caudal_impulses": "caudal_impulses",
}

REQUIRED_RAW_COLUMNS = set(RENAME_MAP.keys())


@dataclass
class IngestionStats:
    source_rows: int
    observation_rows_loaded: int


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
        raise FileNotFoundError("MetroPT3 source file not found. Put MetroPT3(AirCompressor).csv in data/ or data/raw/.")

    logging.info("Extracting CSV from %s", source_path)
    return pd.read_csv(source_path)


def validate_raw_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def transform_metropt3(df: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    logging.info("Transforming MetroPT3 rows")
    out = df.rename(columns=RENAME_MAP).copy()

    out["source_dataset"] = source_dataset
    out["source_row_id"] = pd.to_numeric(out["source_row_id"], errors="coerce")
    if out["source_row_id"].isna().any():
        raise ValueError("source_row_id contains non-numeric values in MetroPT3 input")
    out["source_row_id"] = out["source_row_id"].astype(int)

    out["observed_at"] = pd.to_datetime(out["observed_at"], errors="coerce")

    numeric_cols = [
        "tp2",
        "tp3",
        "h1",
        "dv_pressure",
        "reservoirs",
        "oil_temperature",
        "motor_current",
        "comp",
        "dv_electric",
        "towers",
        "mpg",
        "lps",
        "oil_level",
        "caudal_impulses",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["pressure_switch"] = out["pressure_switch"].astype(str).str.strip().str.lower().map(
        {
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "on": True,
            "off": False,
        }
    )

    ordered_cols = [
        "source_dataset",
        "source_row_id",
        "observed_at",
        "tp2",
        "tp3",
        "h1",
        "dv_pressure",
        "reservoirs",
        "oil_temperature",
        "motor_current",
        "comp",
        "dv_electric",
        "towers",
        "mpg",
        "lps",
        "pressure_switch",
        "oil_level",
        "caudal_impulses",
    ]
    return out[ordered_cols].copy()


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
    database_url: str,
    *,
    batch_size: int,
    statement_timeout_ms: int,
    lock_timeout_ms: int,
) -> None:
    logging.info("Loading MetroPT3 observations into PostgreSQL")
    engine = create_engine(database_url, pool_pre_ping=True)

    insert_sql = text(
        """
        INSERT INTO sensor_observations_metropt3 (
            source_dataset,
            source_row_id,
            observed_at,
            tp2,
            tp3,
            h1,
            dv_pressure,
            reservoirs,
            oil_temperature,
            motor_current,
            comp,
            dv_electric,
            towers,
            mpg,
            lps,
            pressure_switch,
            oil_level,
            caudal_impulses
        )
        VALUES (
            :source_dataset,
            :source_row_id,
            :observed_at,
            :tp2,
            :tp3,
            :h1,
            :dv_pressure,
            :reservoirs,
            :oil_temperature,
            :motor_current,
            :comp,
            :dv_electric,
            :towers,
            :mpg,
            :lps,
            :pressure_switch,
            :oil_level,
            :caudal_impulses
        )
        ON CONFLICT (source_dataset, source_row_id) DO UPDATE
        SET
            observed_at = EXCLUDED.observed_at,
            tp2 = EXCLUDED.tp2,
            tp3 = EXCLUDED.tp3,
            h1 = EXCLUDED.h1,
            dv_pressure = EXCLUDED.dv_pressure,
            reservoirs = EXCLUDED.reservoirs,
            oil_temperature = EXCLUDED.oil_temperature,
            motor_current = EXCLUDED.motor_current,
            comp = EXCLUDED.comp,
            dv_electric = EXCLUDED.dv_electric,
            towers = EXCLUDED.towers,
            mpg = EXCLUDED.mpg,
            lps = EXCLUDED.lps,
            pressure_switch = EXCLUDED.pressure_switch,
            oil_level = EXCLUDED.oil_level,
            caudal_impulses = EXCLUDED.caudal_impulses,
            ingested_at = NOW()
        """
    )

    try:
        with engine.begin() as conn:
            if statement_timeout_ms > 0:
                conn.execute(text(f"SET LOCAL statement_timeout = {statement_timeout_ms}"))
            if lock_timeout_ms > 0:
                conn.execute(text(f"SET LOCAL lock_timeout = {lock_timeout_ms}"))

            total = len(observations_df)
            batches = max((total + batch_size - 1) // batch_size, 1)
            loaded = 0
            phase_start = time.perf_counter()

            for idx, batch in enumerate(_iter_batched_records(observations_df, batch_size=batch_size), start=1):
                batch_start = time.perf_counter()
                conn.execute(insert_sql, batch)
                loaded += len(batch)
                elapsed = time.perf_counter() - batch_start
                logging.info(
                    "MetroPT3 batch %s/%s loaded (%s/%s rows, %.2fs)",
                    idx,
                    batches,
                    loaded,
                    total,
                    elapsed,
                )

            total_elapsed = time.perf_counter() - phase_start
            logging.info("MetroPT3 PostgreSQL load phase complete in %.2fs", total_elapsed)
    finally:
        engine.dispose()


def verify_load(database_url: str, source_dataset: str, expected_rows: int) -> IngestionStats:
    engine = create_engine(database_url, pool_pre_ping=True)
    count_sql = text(
        """
        SELECT COUNT(*)
        FROM sensor_observations_metropt3
        WHERE source_dataset = :source_dataset
        """
    )

    try:
        with engine.connect() as conn:
            observation_rows_loaded = int(conn.execute(count_sql, {"source_dataset": source_dataset}).scalar_one())
    finally:
        engine.dispose()

    if observation_rows_loaded < expected_rows:
        raise RuntimeError(
            "Load verification failed: database row counts are smaller than expected source rows. "
            f"expected={expected_rows}, observations={observation_rows_loaded}"
        )

    return IngestionStats(source_rows=expected_rows, observation_rows_loaded=observation_rows_loaded)


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

    transformed_df = transform_metropt3(raw_df, source_dataset=source_dataset)
    write_processed(transformed_df, processed_output)

    if not database_url:
        logging.info("DATABASE_URL not provided. Skipping database load and verification.")
        return IngestionStats(source_rows=len(transformed_df), observation_rows_loaded=0)

    load_to_postgres(
        observations_df=transformed_df,
        database_url=database_url,
        batch_size=batch_size,
        statement_timeout_ms=statement_timeout_ms,
        lock_timeout_ms=lock_timeout_ms,
    )
    return verify_load(database_url=database_url, source_dataset=source_dataset, expected_rows=len(transformed_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetroPT3 production-style ingestion pipeline")
    parser.add_argument("--source-path", default=None, help="Optional explicit path to MetroPT3 CSV")
    parser.add_argument("--source-dataset", default=DEFAULT_DATASET_NAME, help="Source dataset name stored in DB")
    parser.add_argument("--processed-output", default=str(DEFAULT_PROCESSED), help="Path to write transformed parquet")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL SQLAlchemy URL; if omitted, DB load is skipped",
    )
    parser.add_argument("--batch-size", type=int, default=2000, help="Rows per batch for PostgreSQL writes")
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=180000,
        help="PostgreSQL statement timeout in milliseconds; use 0 to disable",
    )
    parser.add_argument(
        "--lock-timeout-ms",
        type=int,
        default=30000,
        help="PostgreSQL lock timeout in milliseconds; use 0 to disable",
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
        "MetroPT3 ingestion complete: source_rows=%s, observations_in_db=%s",
        stats.source_rows,
        stats.observation_rows_loaded,
    )