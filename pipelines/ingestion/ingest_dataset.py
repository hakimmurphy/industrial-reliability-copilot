import argparse
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from pipelines.ingestion.ingest_ai4i import run_ingestion as run_ai4i_ingestion
from pipelines.ingestion.ingest_metropt3 import run_ingestion as run_metropt3_ingestion


AI4I_HEADER_MARKERS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}

METROPT3_HEADER_MARKERS = {
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
}


@dataclass
class RoutingResult:
    dataset_key: str
    source_rows: int
    loaded_rows: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def detect_dataset_from_headers(source_path: Path) -> str:
    headers = set(pd.read_csv(source_path, nrows=0).columns)
    if AI4I_HEADER_MARKERS.issubset(headers):
        return "ai4i"
    if METROPT3_HEADER_MARKERS.issubset(headers):
        return "metropt3"
    raise ValueError("Could not identify dataset type from CSV headers")


def ensure_audit_table(database_url: str) -> None:
    ddl = text(
        """
        CREATE TABLE IF NOT EXISTS ingestion_audit_log (
            audit_id BIGSERIAL PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_checksum_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('started', 'succeeded', 'failed')),
            source_rows INTEGER,
            loaded_rows INTEGER,
            error_message TEXT,
            started_at TIMESTAMP NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMP,
            CONSTRAINT uq_ingestion_audit_file_checksum UNIQUE (dataset_name, file_checksum_sha256)
        )
        """
    )
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(ddl)
    finally:
        engine.dispose()


def find_successful_audit(database_url: str, dataset_name: str, checksum: str) -> bool:
    engine = create_engine(database_url, pool_pre_ping=True)
    sql = text(
        """
        SELECT 1
        FROM ingestion_audit_log
        WHERE dataset_name = :dataset_name
          AND file_checksum_sha256 = :checksum
          AND status = 'succeeded'
        LIMIT 1
        """
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"dataset_name": dataset_name, "checksum": checksum}).first()
            return row is not None
    finally:
        engine.dispose()


def create_audit_started(database_url: str, dataset_name: str, source_path: Path, checksum: str) -> int:
    engine = create_engine(database_url, pool_pre_ping=True)
    sql = text(
        """
        INSERT INTO ingestion_audit_log (dataset_name, file_path, file_checksum_sha256, status)
        VALUES (:dataset_name, :file_path, :checksum, 'started')
        ON CONFLICT (dataset_name, file_checksum_sha256)
        DO UPDATE SET
            file_path = EXCLUDED.file_path,
            status = 'started',
            source_rows = NULL,
            loaded_rows = NULL,
            error_message = NULL,
            started_at = NOW(),
            finished_at = NULL
        RETURNING audit_id
        """
    )
    try:
        with engine.begin() as conn:
            return int(
                conn.execute(
                    sql,
                    {
                        "dataset_name": dataset_name,
                        "file_path": str(source_path),
                        "checksum": checksum,
                    },
                ).scalar_one()
            )
    finally:
        engine.dispose()


def update_audit_succeeded(database_url: str, audit_id: int, source_rows: int, loaded_rows: int) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    sql = text(
        """
        UPDATE ingestion_audit_log
        SET status = 'succeeded',
            source_rows = :source_rows,
            loaded_rows = :loaded_rows,
            finished_at = NOW(),
            error_message = NULL
        WHERE audit_id = :audit_id
        """
    )
    try:
        with engine.begin() as conn:
            conn.execute(sql, {"audit_id": audit_id, "source_rows": source_rows, "loaded_rows": loaded_rows})
    finally:
        engine.dispose()


def update_audit_failed(database_url: str, audit_id: int, error_message: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    sql = text(
        """
        UPDATE ingestion_audit_log
        SET status = 'failed',
            finished_at = NOW(),
            error_message = :error_message
        WHERE audit_id = :audit_id
        """
    )
    try:
        with engine.begin() as conn:
            conn.execute(sql, {"audit_id": audit_id, "error_message": error_message[:2000]})
    finally:
        engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset router for AI4I and MetroPT3 CSV ingestion")
    parser.add_argument("--source-path", required=True, help="Path to incoming CSV file")
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "ai4i", "metropt3"],
        help="Dataset selector (auto detects from headers)",
    )
    parser.add_argument("--source-dataset", default=None, help="Optional source dataset label stored in DB")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL SQLAlchemy URL")
    parser.add_argument("--batch-size", type=int, default=2000, help="Rows per write batch")
    parser.add_argument("--statement-timeout-ms", type=int, default=180000)
    parser.add_argument("--lock-timeout-ms", type=int, default=30000)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    source_path = Path(args.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    dataset_key = args.dataset
    if dataset_key == "auto":
        dataset_key = detect_dataset_from_headers(source_path)

    source_dataset = args.source_dataset or ("AI4I" if dataset_key == "ai4i" else "METROPT3")

    checksum = sha256_of_file(source_path)
    audit_id = None

    if args.database_url:
        ensure_audit_table(args.database_url)
        if find_successful_audit(args.database_url, source_dataset, checksum):
            logging.info("Skipping ingest: checksum already loaded successfully for %s", source_dataset)
            return
        audit_id = create_audit_started(args.database_url, source_dataset, source_path, checksum)
        logging.info("Created ingestion audit row %s", audit_id)

    try:
        if dataset_key == "ai4i":
            stats = run_ai4i_ingestion(
                source_path=source_path,
                source_dataset=source_dataset,
                processed_output=Path("data/processed/ai4i_clean.parquet"),
                database_url=args.database_url,
                batch_size=args.batch_size,
                statement_timeout_ms=args.statement_timeout_ms,
                lock_timeout_ms=args.lock_timeout_ms,
            )
            result = RoutingResult(
                dataset_key=dataset_key,
                source_rows=stats.source_rows,
                loaded_rows=stats.observation_rows_loaded,
            )
        elif dataset_key == "metropt3":
            stats = run_metropt3_ingestion(
                source_path=source_path,
                source_dataset=source_dataset,
                processed_output=Path("data/processed/metropt3_clean.parquet"),
                database_url=args.database_url,
                batch_size=args.batch_size,
                statement_timeout_ms=args.statement_timeout_ms,
                lock_timeout_ms=args.lock_timeout_ms,
            )
            result = RoutingResult(
                dataset_key=dataset_key,
                source_rows=stats.source_rows,
                loaded_rows=stats.observation_rows_loaded,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_key}")

        logging.info(
            "Ingestion route complete: dataset=%s source_rows=%s loaded_rows=%s",
            result.dataset_key,
            result.source_rows,
            result.loaded_rows,
        )

        if args.database_url and audit_id is not None:
            update_audit_succeeded(args.database_url, audit_id, result.source_rows, result.loaded_rows)
    except Exception as exc:
        if args.database_url and audit_id is not None:
            update_audit_failed(args.database_url, audit_id, str(exc))
        raise


if __name__ == "__main__":
    main()