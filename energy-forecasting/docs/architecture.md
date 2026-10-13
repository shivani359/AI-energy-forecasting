# Project explanation and architecture

## What is energy-consumption forecasting?

### In plain language

Energy forecasting estimates how much electricity a building, factory, data
center, or utility area will use later. This project estimates the next hour
from past hourly meter readings, calendar patterns, and a temperature signal.
The estimate is not a meter reading or a guarantee; it is a planning aid.

### In technical language

This is a supervised regression problem on an hourly time series. The target is
`consumption_kwh` at time *t*. Predictors include demand at *t−1*, *t−24*, and
*t−168*, lagged rolling averages, calendar cycles, and outdoor temperature.
Models are compared on a chronological holdout set. The selected candidate is
refit on all available history and used in a recursive multi-step forecast.

## Why organizations forecast

- **Smart cities and utilities:** estimate near-term area demand, plan reserves,
  schedule generation, and identify unusual peaks before they occur.
- **Electricity boards:** improve load scheduling and procurement decisions.
  Forecasts support operations; they do not replace grid safety or dispatch
  systems.
- **Manufacturing:** align energy-intensive production with expected demand,
  plan equipment schedules, and investigate shifts from a normal load profile.
- **Data centers:** plan cooling and power capacity, review expected peaks, and
  support workload scheduling within reliability constraints.
- **Renewable-energy operators:** compare expected site demand or generation
  with weather-dependent supply to support storage and balancing plans.

When a company combines a forecast with its actual tariff, operating
constraints, and a verified action plan, it can estimate costs and decide
whether a flexible load can move to a lower-cost time. A forecast by itself
does not prove savings. This demo uses a user-chosen flat cost per kWh only to
show the arithmetic.

## Beginner-friendly stack choices

| Option | Tools and model types | Difficulty | GPU? | Expected result |
| --- | --- | --- | --- | --- |
| **A — easiest** | Python, Pandas, NumPy, Matplotlib; seasonal-naive baseline and simple Linear Regression | Beginner | No | Clean hourly data, one forecast line, and basic error metrics |
| **B — recommended** | Python, Pandas, NumPy, Matplotlib, scikit-learn; seasonal-naive baseline, Ridge regression, Random Forest | Beginner to intermediate | No | Reproducible time-aware model comparison, holdout metrics, saved charts, and a multi-hour forecast |
| **C — advanced** | Python, PyTorch or TensorFlow, windowed sequences, LSTM/GRU; optional experiment tracking | Advanced | Not required for a small experiment; a GPU can shorten training on larger datasets | A sequence model and a more complex training workflow; still requires baseline comparison and time-aware validation |

**Selected for this project: Option B.** It runs quickly on a normal laptop,
uses libraries employers commonly recognize, compares ML against a simple
baseline, and is straightforward to explain. The LSTM route is not automatically
more accurate; it also needs more data and careful validation.

## End-to-end workflow

1. **Collect:** read historical, hourly interval-energy measurements and
   optional outdoor temperature.
2. **Clean:** parse timestamps and numbers, remove invalid readings, average
   duplicate timestamps, sort chronologically, and interpolate only short
   interior gaps. Longer gaps fail explicitly.
3. **Engineer features:** create lagged demand, shifted rolling averages, and
   cyclical hour, weekday, and annual calendar features.
4. **Train and compare:** compare a prior-day seasonal baseline, scaled Ridge
   regression, and Random Forest regression.
5. **Evaluate:** hold out the latest 20% of the timeline; report RMSE, MAE, and
   R². The split is chronological rather than random.
6. **Forecast:** refit the lowest-holdout-RMSE candidate on all history and
   recursively predict each future hour.
7. **Visualize and simulate:** save charts, forecast CSV, metrics, and an
   illustrative tariff-based cost summary.

## Text architecture diagram

```text
CSV meter data ──────────────┐
                             ├─> validate & clean ─> hourly time series
Synthetic demo generator ────┘                            │
                                                          v
                             calendar + temperature + lag features
                                                          │
                                    chronological train/test split
                                              ┌───────────┼───────────┐
                                              v           v           v
                                         24h baseline    Ridge    Random Forest
                                              └───────────┼───────────┘
                                                          v
                                      RMSE / MAE / R² comparison
                                                          │
                                    refit winner on full history
                                                          │
                             recursive hourly forecast + cost scenario
                                                          │
                       CSV reports + JSON metrics + PNG charts + model
```

## Modules and data flow

| Module | Responsibility |
| --- | --- |
| `main.py` | Command-line options and full run orchestration |
| `src/energy_forecasting/data.py` | Repeatable simulation, CSV input, cleaning checks |
| `src/energy_forecasting/features.py` | Leakage-aware supervised feature table |
| `src/energy_forecasting/modeling.py` | Holdout evaluation, model selection, persistence |
| `src/energy_forecasting/forecasting.py` | Recursive future predictions and cost fields |
| `src/energy_forecasting/visualization.py` | PNG evidence charts |
| `tests/test_pipeline.py` | Core data, feature, evaluation, and forecast checks |

Input rows become a sorted hourly series. Feature creation drops the first week
because weekly lags and rolling averages need seven days of prior observations.
The evaluation result determines which algorithm is used for a future forecast;
the algorithm is refit on all history after evaluation.

## Data contract and limitations

Required CSV columns:

```csv
timestamp,consumption_kwh
2025-01-01 00:00:00,82.4
2025-01-01 01:00:00,77.9
```

Optional `outdoor_temp_c` (or `temperature_c`) adds an observed temperature
feature. The starter expects hourly **interval energy** in kWh per row, not
instantaneous kW readings or cumulative meter counters. Convert the source data
to this format before training. If temperature is absent, the demo creates a
simple seasonal proxy; future temperature is estimated from recent same-hour
observations. Both are assumptions, not a weather forecast.

The generated data imitates shift schedules, weekday/weekend differences,
temperature-related HVAC load, a mild trend, and measurement noise. It is
synthetic and does not represent a particular utility, facility, or real
customer. A production system would additionally need source-system
integration, access controls, time-zone and daylight-saving policies, drift
monitoring, uncertainty intervals, domain review, and operational safeguards.
