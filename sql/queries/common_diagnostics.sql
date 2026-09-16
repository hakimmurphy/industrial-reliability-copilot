-- Failure rate by machine type
SELECT
    o.machine_type,
    ROUND(100.0 * AVG(l.machine_failure::int), 2) AS failure_rate_pct,
    COUNT(*) AS observations
FROM sensor_observations_ai4i o
JOIN failure_labels_ai4i l ON l.observation_id = o.observation_id
GROUP BY o.machine_type
ORDER BY failure_rate_pct DESC;

-- Top failure modes
SELECT
    SUM(l.twf::int) AS tool_wear_failures,
    SUM(l.hdf::int) AS heat_dissipation_failures,
    SUM(l.pwf::int) AS power_failures,
    SUM(l.osf::int) AS overstrain_failures,
    SUM(l.rnf::int) AS random_failures
FROM failure_labels_ai4i l;

-- Operating envelope where failures occur
SELECT
    o.machine_type,
    ROUND(AVG(o.air_temperature_k), 2) AS avg_air_temp_k,
    ROUND(AVG(o.process_temperature_k), 2) AS avg_process_temp_k,
    ROUND(AVG(o.rotational_speed_rpm), 2) AS avg_rpm,
    ROUND(AVG(o.torque_nm), 2) AS avg_torque,
    ROUND(AVG(o.tool_wear_min), 2) AS avg_tool_wear,
    ROUND(AVG(o.delta_temp_k), 2) AS avg_delta_temp_k,
    ROUND(AVG(o.power_proxy), 2) AS avg_power_proxy
FROM sensor_observations_ai4i o
JOIN failure_labels_ai4i l ON l.observation_id = o.observation_id
WHERE l.machine_failure = TRUE
GROUP BY o.machine_type
ORDER BY o.machine_type;
