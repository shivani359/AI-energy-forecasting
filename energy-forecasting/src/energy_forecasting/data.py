"""Synthetic data generation and input CSV validation/cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _temperature_profile(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return a smooth, plausible outdoor-temperature proxy in degrees Celsius."""
    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()
    annual_cycle = 9.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
    daily_cycle = 3.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    return 22.0 + annual_cycle + daily_cycle


def generate_synthetic_energy_data(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """
    Make a repeatable hourly facility-demand dataset for practice and demos.

    This is simulated data, not a measurement from a real customer or utility.
    """
    if days < 30:
        raise ValueError("days must be at least 30.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    temperature = _temperature_profile(timestamps)

    # Simulated production shifts: weekday daytime operation and a smaller
    # overnight baseload. The smooth transitions make the data less artificial.
    day_shift = np.exp(-0.5 * ((hour - 13) / 4.2) ** 2)
    evening_shift = np.exp(-0.5 * ((hour - 21) / 3.0) ** 2)
    weekend_factor = np.where(weekday < 5, 1.0, 0.58)
    production_load = weekend_factor * (33.0 * day_shift + 11.0 * evening_shift)
    heating_cooling = 0.9 * np.maximum(temperature - 25, 0) + 0.65 * np.maximum(
        17 - temperature, 0
    )
    daily_cycle = 7.0 * np.sin(2 * np.pi * (hour - 6) / 24)
    slow_growth = np.linspace(0, 4.0, len(timestamps))
    noise = rng.normal(0, 4.0, len(timestamps))
    consumption = (
        78.0
        + production_load
        + 2.3 * heating_cooling
        + daily_cycle
        + slow_growth
        + noise
    )
    consumption = np.maximum(consumption, 15.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "consumption_kwh": np.round(consumption, 3),
            "outdoor_temp_c": np.round(temperature, 2),
            "data_source": "synthetic_training_simulation",
        }
    )


def save_synthetic_energy_data(frame: pd.DataFrame, destination: Path) -> None:
    """Save the generated dataset, creating its parent directory if necessary."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def clean_energy_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Validate, sort, de-duplicate, and fill only short interior hourly gaps.

    Input columns are `timestamp`, `consumption_kwh`, and optionally
    `outdoor_temp_c` (or `temperature_c`). Longer gaps are rejected instead of
    being silently interpolated.
    """
    required = {"timestamp", "consumption_kwh"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    cleaned = frame.copy()
    input_rows = len(cleaned)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    cleaned["consumption_kwh"] = pd.to_numeric(
        cleaned["consumption_kwh"], errors="coerce"
    )
    temp_column = (
        "outdoor_temp_c"
        if "outdoor_temp_c" in cleaned.columns
        else "temperature_c"
        if "temperature_c" in cleaned.columns
        else None
    )
    if temp_column and temp_column != "outdoor_temp_c":
        cleaned = cleaned.rename(columns={temp_column: "outdoor_temp_c"})
    elif not temp_column:
        cleaned["outdoor_temp_c"] = np.nan
    else:
        cleaned["outdoor_temp_c"] = pd.to_numeric(
            cleaned["outdoor_temp_c"], errors="coerce"
        )

    invalid_mask = (
        cleaned["timestamp"].isna()
        | cleaned["consumption_kwh"].isna()
        | (cleaned["consumption_kwh"] < 0)
    )
    invalid_rows = int(invalid_mask.sum())
    cleaned = cleaned.loc[~invalid_mask].copy()
    if len(cleaned) < 24 * 30:
        raise ValueError(
            "At least 30 days of valid hourly readings are required; "
            "use the 180-day demo dataset for a full forecast."
        )

    duplicate_count = int(cleaned["timestamp"].duplicated(keep=False).sum())
    cleaned = (
        cleaned.groupby("timestamp", as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "mean"),
            outdoor_temp_c=("outdoor_temp_c", "mean"),
        )
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    median_interval = cleaned.index.to_series().diff().dropna().median()
    if median_interval != pd.Timedelta(hours=1):
        raise ValueError(
            "This starter project expects hourly interval-energy data. "
            "Aggregate or resample your source CSV to one row per hour first."
        )

    expected_index = pd.date_range(cleaned.index.min(), cleaned.index.max(), freq="h")
    cleaned = cleaned.reindex(expected_index)
    missing_energy_before_fill = int(cleaned["consumption_kwh"].isna().sum())
    cleaned["consumption_kwh"] = cleaned["consumption_kwh"].interpolate(
        method="time", limit=3, limit_area="inside"
    )
    remaining_energy_gaps = int(cleaned["consumption_kwh"].isna().sum())
    if remaining_energy_gaps:
        raise ValueError(
            f"Found {remaining_energy_gaps} unfilled energy gap(s). Only interior "
            "gaps up to 3 hours are interpolated; repair longer gaps before training."
        )
    cleaned["outdoor_temp_c"] = cleaned["outdoor_temp_c"].interpolate(
        method="time", limit=3, limit_area="inside"
    )
    cleaned["outdoor_temp_c"] = cleaned["outdoor_temp_c"].fillna(
        pd.Series(_temperature_profile(cleaned.index), index=cleaned.index)
    )
    cleaned = cleaned.reset_index(names="timestamp")
    cleaned["data_source"] = "user_csv"

    report = {
        "input_rows": input_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_rows_aggregated": duplicate_count,
        "short_energy_gaps_interpolated": missing_energy_before_fill,
        "output_rows": len(cleaned),
    }
    return cleaned, report


def load_energy_csv(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read a CSV from disk and apply the project's explicit cleaning rules."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    return clean_energy_data(pd.read_csv(path))
