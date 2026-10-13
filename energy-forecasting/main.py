"""Command-line entry point for the AI Energy Consumption Forecasting project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_forecasting.data import (
    generate_synthetic_energy_data,
    load_energy_csv,
    save_synthetic_energy_data,
)
from energy_forecasting.features import FEATURE_COLUMNS
from energy_forecasting.forecasting import forecast_next_hours
from energy_forecasting.modeling import (
    evaluate_models,
    refit_selected_model,
    save_training_results,
)
from energy_forecasting.visualization import (
    plot_evaluation,
    plot_hourly_profile,
    plot_history_and_forecast,
)


def parse_args() -> argparse.Namespace:
    """Parse options while keeping the default run simple for first-time users."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate energy models, then forecast future demand."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "train", "forecast"),
        default="all",
        help="Run the full workflow (default), only train, or only forecast with a saved model.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV with timestamp, consumption_kwh, and optionally outdoor_temp_c.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days in the reproducible synthetic dataset (minimum 30).",
    )
    parser.add_argument("--horizon", type=int, default=48, help="Forecast horizon in hours.")
    parser.add_argument(
        "--tariff",
        type=float,
        default=8.0,
        help="Illustrative cost per kWh, in any currency unit you choose.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation data.")
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Input/generated data folder."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="CSV and report output folder."
    )
    parser.add_argument(
        "--image-dir", type=Path, default=PROJECT_ROOT / "images", help="Chart output folder."
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models", help="Saved model folder."
    )
    return parser.parse_args()


def get_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load a user's CSV or create the project's deterministic demo dataset."""
    if args.input:
        frame, cleaning_report = load_energy_csv(args.input)
        print(f"Loaded {len(frame):,} hourly rows from {args.input}")
        print(f"Cleaning summary: {json.dumps(cleaning_report, sort_keys=True)}")
        return frame

    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    demo_path = raw_dir / "synthetic_energy.csv"
    if not demo_path.exists() or args.mode in ("all", "train"):
        frame = generate_synthetic_energy_data(days=args.days, seed=args.seed)
        save_synthetic_energy_data(frame, demo_path)
        print(f"Created reproducible simulation data: {demo_path}")
        return frame

    frame, _ = load_energy_csv(demo_path)
    return frame


def train(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, object | None]:
    """Score a seasonal baseline and two ML models, then fit the winner on all history."""
    results = evaluate_models(frame)
    selected_name = results["selected_model"]
    fitted_model = refit_selected_model(frame, selected_name)
    save_training_results(
        results=results,
        fitted_model=fitted_model,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
    print("\nChronological holdout results (one-hour-ahead):")
    for row in results["metrics"]:
        print(
            f"  {row['model']:<28} RMSE {row['rmse_kwh']:>7.2f} kWh | "
            f"MAE {row['mae_kwh']:>7.2f} kWh | R² {row['r2']:>6.3f}"
        )
    print(f"Selected by holdout RMSE: {selected_name}")
    return results, fitted_model


def main() -> None:
    args = parse_args()
    if args.days < 30:
        raise SystemExit("--days must be at least 30 so weekly lag features have enough history.")
    if args.horizon < 1 or args.horizon > 720:
        raise SystemExit("--horizon must be between 1 and 720 hours.")
    if args.tariff < 0:
        raise SystemExit("--tariff cannot be negative.")

    frame = get_data(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    fitted_model = None
    results = None
    metadata_path = args.model_dir / "model_metadata.json"
    if args.mode in ("all", "train"):
        results, fitted_model = train(frame, args)
        plot_evaluation(results, args.image_dir / "model_evaluation.png")
        plot_hourly_profile(frame, args.image_dir / "hourly_demand_profile.png")
    elif args.mode == "forecast":
        if not metadata_path.exists():
            raise SystemExit(
                f"No saved model found at {metadata_path}. Run `python main.py --mode train` first."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["selected_model"] != "Seasonal naive (24-hour)":
            fitted_model = joblib.load(args.model_dir / "best_model.joblib")

    if args.mode in ("all", "forecast"):
        model_name = (
            results["selected_model"]
            if results
            else json.loads(metadata_path.read_text(encoding="utf-8"))["selected_model"]
        )
        forecast = forecast_next_hours(
            history=frame,
            model=fitted_model,
            model_name=model_name,
            feature_columns=FEATURE_COLUMNS,
            horizon=args.horizon,
            tariff_per_kwh=args.tariff,
        )
        forecast_path = args.output_dir / "next_48_hours_forecast.csv"
        if args.horizon != 48:
            forecast_path = args.output_dir / f"next_{args.horizon}_hours_forecast.csv"
        forecast.to_csv(forecast_path, index=False)
        plot_history_and_forecast(
            history=frame,
            forecast=forecast,
            destination=args.image_dir / "energy_forecast.png",
            model_name=model_name,
        )

        total_kwh = float(forecast["forecast_kwh"].sum())
        total_cost = float(forecast["estimated_cost"].sum())
        peak_hour = forecast.loc[forecast["forecast_kwh"].idxmax()]
        report = {
            "model": model_name,
            "horizon_hours": args.horizon,
            "forecast_energy_kwh": round(total_kwh, 2),
            "illustrative_tariff_per_kwh": args.tariff,
            "illustrative_cost": round(total_cost, 2),
            "historical_90th_percentile_kwh": round(
                float(frame["consumption_kwh"].quantile(0.90)), 2
            ),
            "predicted_peak_hour": pd.Timestamp(peak_hour["timestamp"]).isoformat(),
            "predicted_peak_kwh": round(float(peak_hour["forecast_kwh"]), 2),
            "forecast_csv": str(forecast_path.relative_to(PROJECT_ROOT)),
            "cost_is_illustrative": True,
        }
        (args.output_dir / "business_simulation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("\nForecast simulation:")
        print(json.dumps(report, indent=2))

    print(f"\nResults: {args.output_dir}")
    print(f"Charts:  {args.image_dir}")


if __name__ == "__main__":
    main()
