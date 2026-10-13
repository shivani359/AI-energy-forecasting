"""Save clear, presentation-ready charts for the README and project evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_history_and_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    destination: Path,
    model_name: str,
) -> None:
    """Plot recent actual demand and the recursively generated forecast."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    recent = history.tail(24 * 7)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        pd.to_datetime(recent["timestamp"]),
        recent["consumption_kwh"],
        color="#257A70",
        linewidth=1.8,
        label="Observed demand",
    )
    ax.plot(
        pd.to_datetime(forecast["timestamp"]),
        forecast["forecast_kwh"],
        color="#E27A35",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Forecast",
    )
    ax.axvline(pd.Timestamp(forecast["timestamp"].iloc[0]), color="#89939C", linestyle="--")
    ax.set_title(f"Hourly energy demand forecast · {model_name}")
    ax.set_ylabel("Energy consumption (kWh)")
    ax.set_xlabel("Timestamp")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(results: dict[str, Any], destination: Path) -> None:
    """Show holdout predictions against observations for the winning model."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.to_datetime(results["test_timestamps"])
    selected = results["selected_model"]
    actual = np.asarray(results["actual"])
    predicted = np.asarray(results["predictions"][selected])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(timestamps, actual, color="#3A4550", linewidth=1.2, label="Actual")
    axes[0].plot(timestamps, predicted, color="#E27A35", linewidth=1.2, label="Predicted")
    axes[0].set_title(f"Chronological test period · selected model: {selected}")
    axes[0].set_ylabel("kWh")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    residuals = actual - predicted
    axes[1].plot(timestamps, residuals, color="#257A70", linewidth=0.9)
    axes[1].axhline(0, color="#89939C", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Actual − predicted (kWh)")
    axes[1].set_xlabel("Timestamp")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_profile(frame: pd.DataFrame, destination: Path) -> None:
    """Plot median demand by hour and compare weekdays with weekends."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = frame.copy()
    profile["timestamp"] = pd.to_datetime(profile["timestamp"])
    profile["hour"] = profile["timestamp"].dt.hour
    profile["day_type"] = np.where(profile["timestamp"].dt.dayofweek < 5, "Weekday", "Weekend")
    summary = profile.groupby(["day_type", "hour"])["consumption_kwh"].median().unstack(0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for day_type, color in (("Weekday", "#257A70"), ("Weekend", "#E27A35")):
        if day_type in summary.columns:
            ax.plot(summary.index, summary[day_type], marker="o", color=color, label=day_type)
    ax.set_title("Typical hourly demand profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Median consumption (kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)
