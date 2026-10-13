"""Chronological model comparison, metric calculation, and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecasting.features import FEATURE_COLUMNS, build_supervised_features

SEASONAL_BASELINE = "Seasonal naive (24-hour)"
MODEL_FACTORIES = {
    "Ridge regression": lambda: Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
    ),
    "Random forest": lambda: RandomForestRegressor(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}


def _calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "rmse_kwh": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mae_kwh": float(mean_absolute_error(actual_values, predicted_values)),
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def evaluate_models(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Compare a seasonal baseline and two ML regressors on the last 20% of time.

    Test rows use the prior observed readings as lag inputs, matching a rolling
    one-hour-ahead operational evaluation. No random shuffling is used.
    """
    supervised = build_supervised_features(frame)
    if len(supervised) < 400:
        raise ValueError(
            "Not enough usable hourly observations after creating 168-hour lags. "
            "Use at least 30 days of data."
        )

    split_index = int(len(supervised) * 0.8)
    train_rows = supervised.iloc[:split_index]
    test_rows = supervised.iloc[split_index:]
    if len(test_rows) < 24:
        raise ValueError("The holdout period must contain at least 24 hourly observations.")

    x_train = train_rows[FEATURE_COLUMNS]
    y_train = train_rows["consumption_kwh"]
    x_test = test_rows[FEATURE_COLUMNS]
    y_test = test_rows["consumption_kwh"]

    metrics: list[dict[str, Any]] = []
    baseline = frame.set_index("timestamp")["consumption_kwh"].reindex(
        pd.DatetimeIndex(test_rows["timestamp"]) - pd.Timedelta(hours=24)
    )
    if baseline.isna().any():
        raise ValueError("Could not align 24-hour seasonal baseline with holdout timestamps.")
    baseline_predicted = baseline.to_numpy()
    metrics.append(
        {"model": SEASONAL_BASELINE, **_calculate_metrics(y_test, baseline_predicted)}
    )

    fitted_models: dict[str, Any] = {}
    for model_name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(x_train, y_train)
        prediction = np.maximum(model.predict(x_test), 0.0)
        metrics.append({"model": model_name, **_calculate_metrics(y_test, prediction)})
        fitted_models[model_name] = model

    selected_name = min(metrics, key=lambda row: row["rmse_kwh"])["model"]
    return {
        "selected_model": selected_name,
        "split": "chronological 80% train / 20% test",
        "target": "next-hour consumption_kwh",
        "test_start": pd.Timestamp(test_rows["timestamp"].iloc[0]).isoformat(),
        "test_end": pd.Timestamp(test_rows["timestamp"].iloc[-1]).isoformat(),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "actual": y_test.to_numpy(),
        "predictions": {
            SEASONAL_BASELINE: baseline_predicted,
            **{
                name: np.maximum(model.predict(x_test), 0.0)
                for name, model in fitted_models.items()
            },
        },
        "test_timestamps": test_rows["timestamp"].astype(str).tolist(),
    }


def refit_selected_model(frame: pd.DataFrame, selected_name: str) -> Any | None:
    """Refit the selected ML candidate on all known data before future use."""
    if selected_name == SEASONAL_BASELINE:
        return None
    if selected_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown selected model: {selected_name}")
    supervised = build_supervised_features(frame)
    model = MODEL_FACTORIES[selected_name]()
    model.fit(supervised[FEATURE_COLUMNS], supervised["consumption_kwh"])
    return model


def save_training_results(
    results: dict[str, Any],
    fitted_model: Any | None,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Persist evaluation evidence and the selected model's inference metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_results = {
        key: value
        for key, value in results.items()
        if key not in {"actual", "predictions", "test_timestamps"}
    }
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")

    prediction_rows = pd.DataFrame(
        {
            "timestamp": results["test_timestamps"],
            "actual_kwh": results["actual"],
        }
    )
    for model_name, prediction in results["predictions"].items():
        prediction_rows[f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_predicted_kwh"] = prediction
    prediction_rows.to_csv(output_dir / "holdout_predictions.csv", index=False)

    metadata = {
        "selected_model": results["selected_model"],
        "feature_columns": results["feature_columns"],
        "target": results["target"],
        "training_scope": "refitted on all available history after holdout evaluation",
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if fitted_model is not None:
        joblib.dump(fitted_model, model_dir / "best_model.joblib")

