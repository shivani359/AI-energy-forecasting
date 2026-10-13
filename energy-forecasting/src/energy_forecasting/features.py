"""Leakage-aware calendar, weather, and historical-demand features."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "consumption_lag_1h",
    "consumption_lag_24h",
    "consumption_lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "outdoor_temp_c",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_day_sin",
    "year_day_cos",
    "is_weekend",
]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar variables without imposing false numeric order."""
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    hour_angle = 2 * np.pi * timestamp.dt.hour / 24
    weekday_angle = 2 * np.pi * timestamp.dt.dayofweek / 7
    year_day_angle = 2 * np.pi * (timestamp.dt.dayofyear - 1) / 365.25
    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    result["weekday_sin"] = np.sin(weekday_angle)
    result["weekday_cos"] = np.cos(weekday_angle)
    result["year_day_sin"] = np.sin(year_day_angle)
    result["year_day_cos"] = np.cos(year_day_angle)
    result["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    return result


def build_supervised_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-hour-ahead examples using only information available before each
    target timestamp. Rolling statistics are shifted to prevent target leakage.
    """
    result = frame.sort_values("timestamp").copy()
    result["consumption_lag_1h"] = result["consumption_kwh"].shift(1)
    result["consumption_lag_24h"] = result["consumption_kwh"].shift(24)
    result["consumption_lag_168h"] = result["consumption_kwh"].shift(168)
    prior_consumption = result["consumption_kwh"].shift(1)
    result["rolling_mean_24h"] = prior_consumption.rolling(24).mean()
    result["rolling_mean_168h"] = prior_consumption.rolling(168).mean()
    result = add_calendar_features(result)
    return result.dropna(subset=FEATURE_COLUMNS + ["consumption_kwh"]).reset_index(drop=True)
