import pandas as pd


def add_delta_temperature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_temperature_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    return out


def add_power_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["power_proxy"] = out["torque_nm"] * out["rotational_speed_rpm"]
    return out
