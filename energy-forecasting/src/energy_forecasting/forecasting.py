"""Recursive multi-hour forecasts and a simple business-cost simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from energy_forecasting.features import FEATURE_COLUMNS


def _calendar_features(timestamp: pd.Timestamp) -> dict[str, float]:
    hour_angle = 2 * np.pi * timestamp.hour / 24
    weekday_angle = 2 * np.pi * timestamp.dayofweek / 7
    year_day_angle = 2 * np.pi * (timestamp.dayofyear - 1) / 365.25
    return {
        "hour_sin": float(np.sin(hour_angle)),
        "hour_cos": float(np.cos(hour_angle)),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
        "year_day_sin": float(np.sin(year_day_angle)),
        "year_day_cos": float(np.cos(year_day_angle)),
        "is_weekend": int(timestamp.dayofweek >= 5),
    }


def _estimate_future_temperature(
    history: pd.DataFrame, future_timestamp: pd.Timestamp
) -> float:
    """Use recent same-hour readings as a modest future-weather proxy."""
    recent = history.tail(14 * 24)
    same_hour = recent.loc[
        pd.to_datetime(recent["timestamp"]).dt.hour == future_timestamp.hour,
        "outdoor_temp_c",
    ].dropna()
    if len(same_hour):
        return float(same_hour.mean())
    return float(recent["outdoor_temp_c"].dropna().mean())


def _feature_row(
    timestamp: pd.Timestamp,
    consumption_history: list[float],
    temperature: float,
) -> pd.DataFrame:
    """Make exactly the columns used during model training from prior readings."""
    values = {
        "consumption_lag_1h": consumption_history[-1],
        "consumption_lag_24h": consumption_history[-24],
        "consumption_lag_168h": consumption_history[-168],
        "rolling_mean_24h": float(np.mean(consumption_history[-24:])),
        "rolling_mean_168h": float(np.mean(consumption_history[-168:])),
        "outdoor_temp_c": temperature,
        **_calendar_features(timestamp),
    }
    return pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def forecast_next_hours(
    history: pd.DataFrame,
    model: Any | None,
    model_name: str,
    feature_columns: list[str],
    horizon: int = 48,
    tariff_per_kwh: float = 8.0,
) -> pd.DataFrame:
    """
    Recursively forecast demand. The seasonal baseline repeats the previous
    day's same-hour reading; ML models use their own prior predictions as lags.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one hour.")
    if len(history) < 168:
        raise ValueError("At least 168 hourly readings are required to forecast.")
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError("Saved model feature list does not match the current project features.")

    history = history.sort_values("timestamp").reset_index(drop=True)
    known_values = history["consumption_kwh"].astype(float).tolist()
    last_timestamp = pd.Timestamp(history["timestamp"].iloc[-1])
    future_timestamps = pd.date_range(
        last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    peak_threshold = float(history["consumption_kwh"].quantile(0.90))
    rows: list[dict[str, Any]] = []

    for timestamp in future_timestamps:
        if model_name == "Seasonal naive (24-hour)":
            prediction = known_values[-24]
        else:
            feature_row = _feature_row(
                timestamp,
                known_values,
                _estimate_future_temperature(history, timestamp),
            )
            prediction = float(model.predict(feature_row)[0])
        prediction = max(float(prediction), 0.0)
        known_values.append(prediction)
        rows.append(
            {
                "timestamp": timestamp,
                "forecast_kwh": prediction,
                "estimated_cost": prediction * tariff_per_kwh,
                "above_historical_p90": prediction >= peak_threshold,
                "model": model_name,
            }
        )

    return pd.DataFrame(rows)
