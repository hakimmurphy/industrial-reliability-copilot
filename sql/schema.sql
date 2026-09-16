-- Step 4 schema (leakage-safe): predictors are stored separately from failure labels.

CREATE TABLE IF NOT EXISTS assets (
    asset_id BIGSERIAL PRIMARY KEY,
    asset_tag TEXT UNIQUE NOT NULL,
    asset_type TEXT NOT NULL,
    site_name TEXT,
    commissioned_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor_observations_ai4i (
    observation_id BIGSERIAL PRIMARY KEY,
    source_dataset TEXT NOT NULL DEFAULT 'AI4I',
    source_row_id INTEGER NOT NULL,
    asset_id BIGINT REFERENCES assets(asset_id),
    product_id TEXT NOT NULL,
    machine_type TEXT NOT NULL CHECK (machine_type IN ('L', 'M', 'H')),
    air_temperature_k NUMERIC(8, 3) NOT NULL,
    process_temperature_k NUMERIC(8, 3) NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm NUMERIC(10, 3) NOT NULL,
    tool_wear_min INTEGER NOT NULL,
    delta_temp_k NUMERIC(8, 3) GENERATED ALWAYS AS (process_temperature_k - air_temperature_k) STORED,
    power_proxy NUMERIC(16, 3) GENERATED ALWAYS AS (torque_nm * rotational_speed_rpm) STORED,
    observed_at TIMESTAMP,
    ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sensor_observations_source UNIQUE (source_dataset, source_row_id)
);

CREATE TABLE IF NOT EXISTS failure_labels_ai4i (
    label_id BIGSERIAL PRIMARY KEY,
    observation_id BIGINT UNIQUE NOT NULL REFERENCES sensor_observations_ai4i(observation_id) ON DELETE CASCADE,
    machine_failure BOOLEAN NOT NULL,
    twf BOOLEAN NOT NULL,
    hdf BOOLEAN NOT NULL,
    pwf BOOLEAN NOT NULL,
    osf BOOLEAN NOT NULL,
    rnf BOOLEAN NOT NULL,
    label_source TEXT NOT NULL DEFAULT 'AI4I',
    labeled_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    observation_id BIGINT NOT NULL REFERENCES sensor_observations_ai4i(observation_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_failure_probability NUMERIC(6, 5) NOT NULL CHECK (predicted_failure_probability >= 0 AND predicted_failure_probability <= 1),
    threshold_used NUMERIC(6, 5),
    predicted_failure BOOLEAN,
    predicted_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sensor_observations_metropt3 (
    observation_id BIGSERIAL PRIMARY KEY,
    source_dataset TEXT NOT NULL DEFAULT 'METROPT3',
    source_row_id INTEGER NOT NULL,
    observed_at TIMESTAMP,
    tp2 NUMERIC(12, 5),
    tp3 NUMERIC(12, 5),
    h1 NUMERIC(12, 5),
    dv_pressure NUMERIC(12, 5),
    reservoirs NUMERIC(12, 5),
    oil_temperature NUMERIC(12, 5),
    motor_current NUMERIC(12, 5),
    comp NUMERIC(12, 5),
    dv_electric NUMERIC(12, 5),
    towers NUMERIC(12, 5),
    mpg NUMERIC(12, 5),
    lps NUMERIC(12, 5),
    pressure_switch BOOLEAN,
    oil_level NUMERIC(12, 5),
    caudal_impulses NUMERIC(12, 5),
    ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sensor_observations_metro_source UNIQUE (source_dataset, source_row_id)
);

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
);

CREATE INDEX IF NOT EXISTS idx_obs_machine_type ON sensor_observations_ai4i(machine_type);
CREATE INDEX IF NOT EXISTS idx_obs_torque_nm ON sensor_observations_ai4i(torque_nm);
CREATE INDEX IF NOT EXISTS idx_obs_tool_wear_min ON sensor_observations_ai4i(tool_wear_min);
CREATE INDEX IF NOT EXISTS idx_label_machine_failure ON failure_labels_ai4i(machine_failure);
CREATE INDEX IF NOT EXISTS idx_pred_model_time ON model_predictions(model_name, model_version, predicted_at);
CREATE INDEX IF NOT EXISTS idx_metro_observed_at ON sensor_observations_metropt3(observed_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_audit_dataset_started ON ingestion_audit_log(dataset_name, started_at DESC);

-- Training view intentionally excludes failure-mode columns (TWF/HDF/PWF/OSF/RNF)
-- from predictor features to prevent target leakage.
CREATE OR REPLACE VIEW ml_training_dataset_ai4i AS
SELECT
    o.observation_id,
    o.source_row_id,
    o.product_id,
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
JOIN failure_labels_ai4i l ON l.observation_id = o.observation_id;
