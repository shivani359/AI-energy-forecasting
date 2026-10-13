"""Small, dependency-light checks for the energy forecasting pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_forecasting.data import clean_energy_data, generate_synthetic_energy_data
from energy_forecasting.features import build_supervised_features
from energy_forecasting.forecasting import forecast_next_hours
from energy_forecasting.modeling import evaluate_models


class EnergyForecastingTests(unittest.TestCase):
    def test_synthetic_data_is_hourly_and_reproducible(self) -> None:
        first = generate_synthetic_energy_data(days=30, seed=7)
        second = generate_synthetic_energy_data(days=30, seed=7)
        self.assertEqual(len(first), 30 * 24)
        self.assertTrue(first["consumption_kwh"].equals(second["consumption_kwh"]))
        self.assertTrue((first["consumption_kwh"] > 0).all())

    def test_cleaner_sorts_deduplicates_and_fills_a_short_gap(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=24 * 30, freq="h")
        source = pd.DataFrame(
            {
                "timestamp": timestamps.delete(100).append(pd.DatetimeIndex([timestamps[5]])),
                "consumption_kwh": np.arange(24 * 30 - 1, dtype=float).tolist() + [5.0],
                "outdoor_temp_c": 20.0,
            }
        )
        cleaned, report = clean_energy_data(source)
        self.assertEqual(len(cleaned), 24 * 30)
        self.assertEqual(report["duplicate_rows_aggregated"], 2)
        self.assertEqual(report["short_energy_gaps_interpolated"], 1)
        self.assertTrue(cleaned["timestamp"].is_monotonic_increasing)

    def test_features_use_previous_values_and_drop_warmup_rows(self) -> None:
        frame = generate_synthetic_energy_data(days=30, seed=3)
        features = build_supervised_features(frame)
        self.assertEqual(features.iloc[0]["consumption_lag_1h"], frame.iloc[167]["consumption_kwh"])
        self.assertEqual(features.iloc[0]["consumption_lag_168h"], frame.iloc[0]["consumption_kwh"])
        self.assertEqual(len(features), len(frame) - 168)

    def test_model_evaluation_is_chronological_and_returns_required_metrics(self) -> None:
        frame = generate_synthetic_energy_data(days=45, seed=11)
        result = evaluate_models(frame)
        self.assertGreater(result["test_rows"], 0)
        self.assertTrue(result["test_start"] > "2024-01-01")
        self.assertIn(result["selected_model"], {row["model"] for row in result["metrics"]})
        for row in result["metrics"]:
            self.assertGreaterEqual(row["rmse_kwh"], 0)
            self.assertTrue(np.isfinite(row["r2"]))

    def test_forecast_returns_requested_rows_and_costs(self) -> None:
        frame = generate_synthetic_energy_data(days=30, seed=9)
        forecast = forecast_next_hours(
            history=frame,
            model=None,
            model_name="Seasonal naive (24-hour)",
            feature_columns=[
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
            ],
            horizon=12,
            tariff_per_kwh=5,
        )
        self.assertEqual(len(forecast), 12)
        self.assertTrue((forecast["forecast_kwh"] >= 0).all())
        self.assertTrue(np.allclose(forecast["estimated_cost"], forecast["forecast_kwh"] * 5))


if __name__ == "__main__":
    unittest.main()
