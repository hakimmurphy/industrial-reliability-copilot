# Industrial Reliability Copilot

An enterprise AI starter system that combines sensor and maintenance data, SQL, RAG hooks, an LLM SQL agent, and evaluation checks to help engineers diagnose equipment issues.

## Repository layout

```text
industrial-reliability-copilot/
├── app/
│   ├── api/
│   ├── agents/
│   ├── rag/
│   ├── tools/
│   ├── models/
│   └── evaluation/
├── pipelines/
│   ├── ingestion/
│   └── preprocessing/
├── sql/
│   ├── schema.sql
│   └── queries/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── notebooks/
├── docker/
├── .github/
│   └── workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Progression roadmap

- ✅ 1. Understand AI4I
- ✅ 2. Quick EDA
- ✅ 3. Basic feature engineering
- ✅ 4. Design PostgreSQL schema
- ✅ 5. Build ingestion pipeline
- ✅ 6. FastAPI
- ✅ 7. ML model (baseline)
- ✅ 8. RAG
- ✅ 9. Agent/tool calling
- ✅ 10. Evaluation
- ✅ 11. Docker/CI/CD
- ✅ 12. Neon

Step 4 artifacts:

- Schema: `sql/schema.sql`
- Diagnostics: `sql/queries/common_diagnostics.sql`

Step 5 artifacts:

- Ingestion pipeline: `pipelines/ingestion/ingest_ai4i.py`
- Ingestion pipeline (MetroPT3): `pipelines/ingestion/ingest_metropt3.py`
- Dataset router + audit logging: `pipelines/ingestion/ingest_dataset.py`

Step 6 artifacts:

- API app: `app/api/main.py`
- SQL agent: `app/agents/sql_agent.py`
- SQL execution tool: `app/tools/sql_tool.py`

Step 7 artifacts:

- Baseline training pipeline: `pipelines/training/train_baseline.py`
- Model output: `app/models/baseline_failure_model.joblib`
- Metrics output: `app/evaluation/baseline_metrics.json`

Step 8 artifacts:

- RAG retriever: `app/rag/retriever.py`
- Knowledge docs: `knowledge/`
- RAG endpoints: `POST /rag/retrieve`, `POST /rag/answer`

Step 9 artifacts:

- Investigation agent: `app/agents/investigation_agent.py`
- Orchestration endpoint: `POST /investigate`

Step 10 artifacts:

- Benchmark cases: `app/evaluation/benchmark_cases.json`
- Evaluation runner: `app/evaluation/run_evaluation.py`
- Evaluation report: `app/evaluation/step10_eval_report.json`

Step 11 artifacts:

- Container image config: `Dockerfile`
- Compose orchestration with healthchecks: `docker-compose.yml`
- CI pipeline (tests + integration + training + evaluation): `.github/workflows/ci.yml`
- Scheduled ingestion workflow: `.github/workflows/scheduled-ingestion.yml`
- Scheduled training + evaluation workflow: `.github/workflows/scheduled-training-eval.yml`
- Build context hardening: `.dockerignore`

### ML leakage rule

For `Machine failure` prediction, do not use failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) as model predictors.

Recommended predictor set:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `delta_temp_k`
- `power_proxy`

## Neon setup (recommended for portfolio)

This project is confirmed working with Neon Postgres and FastAPI.

### Neon verification snapshot (2026-09-11)

- PASS: `.env` exists and `DATABASE_URL` is set.
- PASS: `psql` is available and `sql/schema.sql` applies successfully on Neon.
- BLOCKED: dependency install can fail with `OSError: [Errno 28] No space left on device` when local disk is nearly full.
- INCOMPLETE: ingestion/API/pytest verification was interrupted in a low-disk environment.

Recommended preflight before running the full sequence:

```bash
df -h .
```

If available space is very low, free disk space first (target at least 2 GB free) and then continue.

Important: do not `source .env` directly for `DATABASE_URL`. Connection strings can contain shell-significant characters (for example `&`) that cause shell parse errors. Use `awk` extraction as shown below.

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

1. Create your local env file from the template:

```bash
cp .env.example .env
```

1. Edit `.env` and set real values:

- `DATABASE_URL` (your Neon connection string)
- `OPENAI_API_KEY` (optional; app works without it using fallback behavior)

1. Apply schema to Neon:

```bash
DBURL="$(awk -F= '/^[[:space:]]*DATABASE_URL[[:space:]]*=/{sub(/^[^=]*=/,""); print; exit}' .env)"
PSQL_URL="${DBURL/postgresql+psycopg2/postgresql}"
psql "$PSQL_URL" -f sql/schema.sql
```

1. Run ingestion with DB load:

```bash
DATABASE_URL="$(awk -F= '/^[[:space:]]*DATABASE_URL[[:space:]]*=/{sub(/^[^=]*=/,""); print; exit}' .env)" \
python pipelines/ingestion/ingest_ai4i.py --verbose
```

1. Verify row counts:

```bash
DBURL="$(awk -F= '/^[[:space:]]*DATABASE_URL[[:space:]]*=/{sub(/^[^=]*=/,""); print; exit}' .env)"
PSQL_URL="${DBURL/postgresql+psycopg2/postgresql}"
psql "$PSQL_URL" -c "select 'sensor_observations_ai4i' as table, count(*) as rows from sensor_observations_ai4i union all select 'failure_labels_ai4i' as table, count(*) as rows from failure_labels_ai4i;"
```

1. Start FastAPI:

```bash
uvicorn app.api.main:app --reload
```

1. Test health:

```bash
curl http://127.0.0.1:8000/health
```

1. Test SQL generation:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Show failure rate by machine type","execute_sql":false}'
```

1. Test SQL execution:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Show failure rate by machine type","execute_sql":true}'
```

### Where secrets are stored

- `.env.example` is a committed template only (no real passwords or keys).
- `.env` is your local, real secret file and is ignored by git.
- In production, store `DATABASE_URL` and `OPENAI_API_KEY` in your host's secret manager/environment settings.

## Local PostgreSQL setup (alternative)

Use this only if you prefer running your own Postgres locally instead of Neon.

1. Ensure PostgreSQL 16 is running.

If `5432` is occupied, run PostgreSQL 16 on `5433`.

1. Create app user and database:

```sql
CREATE ROLE irc_user WITH LOGIN PASSWORD 'your_strong_password';
CREATE DATABASE reliability OWNER irc_user;
GRANT ALL PRIVILEGES ON DATABASE reliability TO irc_user;
```

1. Apply schema:

```bash
psql -h localhost -p 5433 -U irc_user -d reliability -f sql/schema.sql
```

## Optional Docker flow

Use Docker after the Neon/local setup works end-to-end. Docker is for repeatable runtime, not required for first connection.

Environment defaults in compose:

- `POSTGRES_DB=reliability`
- `POSTGRES_USER=irc_user`
- `POSTGRES_PASSWORD=your_strong_password`
- Host port mapping defaults to `5433:5432`

1. Copy env file:

```bash
cp .env.example .env
```

1. Start API container against Neon (recommended):

```bash
docker compose up --build api
```

1. Or start a full local stack (Postgres + API) with the local profile:

```bash
docker compose --profile localdb up --build
```

In Neon mode, the API reads `DATABASE_URL` from `.env`.
In local profile mode, if `DATABASE_URL` is not set, compose falls back to the local `postgres` service URL.

1. Verify container health:

```bash
docker compose ps
```

1. Test API:

```bash
curl http://localhost:8000/health
```

1. Ask a question (generate SQL only):

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Show failure rate by machine type","execute_sql":false}'
```

1. Ask and execute SQL against PostgreSQL:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Show failure rate by machine type","execute_sql":true}'
```

## Ingestion flow

Run ingestion without a database (parquet-only mode):

```bash
python pipelines/ingestion/ingest_ai4i.py
```

Run ingestion with database load and verification:

```bash
DATABASE_URL=postgresql+psycopg2://irc_user:your_strong_password@localhost:5433/reliability \
python pipelines/ingestion/ingest_ai4i.py --verbose
```

Run MetroPT3 ingestion with database load and verification:

```bash
DATABASE_URL="$(awk -F= '/^[[:space:]]*DATABASE_URL[[:space:]]*=/{sub(/^[^=]*=/,""); print; exit}' .env)" \
python pipelines/ingestion/ingest_metropt3.py --verbose --source-path "data/MetroPT3(AirCompressor).csv"
```

Run routed ingestion (auto-detect AI4I vs MetroPT3 from CSV headers):

```bash
DATABASE_URL="$(awk -F= '/^[[:space:]]*DATABASE_URL[[:space:]]*=/{sub(/^[^=]*=/,""); print; exit}' .env)" \
python pipelines/ingestion/ingest_dataset.py \
  --source-path "data/MetroPT3(AirCompressor).csv" \
  --dataset auto \
  --verbose
```

The router writes ingestion metadata to `ingestion_audit_log` (dataset, checksum, status, row counts, error message).

## Automated ingestion

Scheduled workflow: `.github/workflows/scheduled-ingestion.yml`

- Runs every 24 hours (cron) and on manual trigger.
- Uses `DATABASE_URL` from GitHub Actions secrets.
- Applies schema, selects a CSV source, routes dataset type, and ingests.
- Prevents duplicate successful loads by dataset + file checksum.

Manual dispatch inputs:

- `source_path` (example: `data/MetroPT3(AirCompressor).csv`)
- `dataset` (`auto`, `ai4i`, or `metropt3`)

## Testing

```bash
pytest -q
```

## Baseline ML training

Train from PostgreSQL (recommended):

```bash
DATABASE_URL=postgresql+psycopg2://irc_user:your_strong_password@localhost:5433/reliability \
python pipelines/training/train_baseline.py
```

Train from parquet fallback:

```bash
python pipelines/training/train_baseline.py
```

Outputs:

- Trained model artifact at `app/models/baseline_failure_model.joblib`
- Evaluation summary at `app/evaluation/baseline_metrics.json`

## RAG usage

Retrieve relevant knowledge snippets:

```bash
curl -X POST http://127.0.0.1:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"what should we inspect when torque is high?","top_k":3}'
```

Generate a context-grounded answer:

```bash
curl -X POST http://127.0.0.1:8000/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What inspections should we prioritize for high wear and torque?","top_k":3}'
```

## Agent/tool calling usage

Run one end-to-end investigation (SQL + ML scoring + RAG guidance):

```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"question":"Find high-risk operating conditions and what to inspect next","execute_sql":true,"max_rows":50,"risk_threshold":0.5,"rag_top_k":3}'
```

## Evaluation usage

Run benchmark evaluation suite:

```bash
python app/evaluation/run_evaluation.py
```

Run with explicit files:

```bash
python app/evaluation/run_evaluation.py \
  --cases app/evaluation/benchmark_cases.json \
  --output app/evaluation/step10_eval_report.json
```

Run with explicit database URL for SQL execution cases:

```bash
python app/evaluation/run_evaluation.py \
  --database-url postgresql+psycopg2://irc_user:your_strong_password@localhost:5433/reliability
```

The report includes section pass rates for:

- SQL generation safety and shape
- RAG retrieval quality
- End-to-end investigate orchestration

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` performs:

- dependency install
- schema apply against ephemeral PostgreSQL service
- ingestion run
- unit tests
- baseline training
- evaluation benchmark run

## Automated retraining and evaluation

Scheduled workflow: `.github/workflows/scheduled-training-eval.yml`

- Triggers when `.github/workflows/scheduled-ingestion.yml` completes successfully.
- Also runs every 6 hours and via manual dispatch.
- Trains the baseline model from PostgreSQL.
- Runs the benchmark evaluation suite.
- Enforces quality gates and fails the run if thresholds are not met.
- Uploads model + evaluation artifacts for inspection.

Required GitHub secret:

- `DATABASE_URL`

Manual dispatch inputs:

- `pr_auc_threshold` (default `0.55`)
- `overall_pass_rate_threshold` (default `0.80`)

Uploaded artifacts:

- `app/models/baseline_failure_model.joblib`
- `app/evaluation/baseline_metrics.json`
- `app/evaluation/step10_eval_report.json`

## Operations playbook

Use this checklist when `.github/workflows/scheduled-training-eval.yml` fails.

1. Confirm trigger and failure step

- Open the workflow run summary and identify whether failure occurred in training, evaluation, or quality gates.

1. Diagnose quality gate failures

- Download artifacts from the run.
- Check `app/evaluation/baseline_metrics.json` for `selected_model_pr_auc`.
- Check `app/evaluation/step10_eval_report.json` for `summary.overall.pass_rate`.
- Compare values to workflow thresholds (`pr_auc_threshold`, `overall_pass_rate_threshold`).

1. Rerun manually with explicit thresholds

- Re-run using workflow dispatch and set:
  - `pr_auc_threshold`
  - `overall_pass_rate_threshold`

1. Verify data freshness when performance regresses

- Confirm `.github/workflows/scheduled-ingestion.yml` completed successfully before the train/eval run.
- Review ingestion logs for row count anomalies or source-file changes.

1. Escalation criteria

- If PR-AUC or pass rate drops for 3 consecutive runs, open an incident issue and attach the two JSON artifacts from each failed run.
- If training fails before metrics are produced, treat as pipeline reliability issue (not model quality) and prioritize infrastructure/debugging.

## Notes

- The SQL tool enforces read-only query prefixes (`SELECT`, `WITH`).
- If `OPENAI_API_KEY` is not set, the agent falls back to heuristic SQL generation.
- This starter is intentionally modular for extension with richer RAG and agent orchestration.
- The SQL agent is aligned with normalized tables (`sensor_observations_ai4i`, `failure_labels_ai4i`) and the `ml_training_dataset_ai4i` view.
