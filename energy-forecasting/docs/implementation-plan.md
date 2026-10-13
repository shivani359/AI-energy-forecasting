# Beginner implementation plan

This plan follows the ten phases requested in the project brief. Every phase
has a check you can perform before moving on.

| Phase | What to do | Why | Expected output | Common mistake | How to verify |
| --- | --- | --- | --- | --- | --- |
| 1. Setup | Install Python 3.10+, create a virtual environment, install `requirements.txt`. | Isolates this project's libraries. | `python --version` and imports work. | Installing into a different Python than the one used to run the project. | Run `python -c "import pandas, sklearn, matplotlib"` in the activated environment. |
| 2. Dataset loading | Start with the built-in generator, then try a CSV with `timestamp` and `consumption_kwh`. | No corporate account or credentials are needed. | Hourly table with time and energy values. | Confusing kW with kWh or cumulative meter values with interval values. | Inspect the first rows and time differences. |
| 3. Data cleaning | Parse time and numeric fields, handle duplicates, sort, check gaps. | Models need valid, consistently spaced observations. | Clean, chronological hourly series. | Silently filling a long outage or treating duplicate readings as separate hours. | Review the printed cleaning summary and fail-fast error messages. |
| 4. Feature engineering | Build prior-hour/day/week lags, shifted rolling averages, calendar cycles, and temperature. | Past demand and recurring schedules are useful predictors. | Supervised table with one target per hour. | Calculating rolling means that include the value being predicted. | Confirm the first feature row starts after 168 history hours. |
| 5. Model building | Compare a 24-hour seasonal baseline, Ridge, and Random Forest. | Baselines show whether ML adds value. | Three candidate models. | Assuming a complex model must perform better. | Check the model comparison in `outputs/model_metrics.json`. |
| 6. Evaluation | Train on the first 80%, test on the last 20%, and calculate RMSE, MAE, R². | Time order resembles deployment and avoids future-to-past leakage. | Holdout metrics and actual-versus-predicted CSV. | Random splitting a time series or reporting only R². | Verify test timestamps occur after training timestamps in the report. |
| 7. Forecasting | Refit the winning candidate on all history and forecast 48 hours. | Uses all known history after the honest model comparison. | Forecast CSV with time, kWh, cost scenario, and peak flag. | Reusing a model without its feature ordering or feeding future actual demand into recursive lags. | Confirm timestamps start one hour after the last historical row. |
| 8. Visualization | Review demand forecast, model holdout, residual, and daily profile charts. | Charts reveal behavior that a single score can hide. | Three PNG files under `images/`. | Hiding poor fit with a truncated axis or showing only training data. | Compare the holdout actual and prediction lines and inspect residuals around zero. |
| 9. GitHub publishing | Add a clear README, keep data provenance and limitations explicit, commit a clean repo. | Hiring reviewers need to reproduce the result and understand its scope. | Public repository and a concise proof-of-work story. | Uploading secrets, customer data, or unexplained generated binaries. | Clone the repository into a fresh folder and run the documented command. |
| 10. Final output | Summarize the best model, test metrics, forecast horizon, assumptions, and next steps. | Converts code into a business-readable result. | JSON report, CSV forecasts, charts, and concise interview explanation. | Claiming simulated cost exposure is realized savings. | Use the report's `cost_is_illustrative` flag and state the synthetic-data caveat. |

## Recommended order to learn

1. Run the default workflow unchanged.
2. Read the charts and `model_metrics.json` before editing the model.
3. Change the synthetic-data seed and explain which results change.
4. Replace the synthetic input with a public dataset that you are permitted to
   use, after converting it to hourly interval kWh.
5. Make one justified change at a time (for example, add a holiday flag) and
   compare it using the same chronological holdout.
6. Record what improved and what did not in the README; do not report only the
   best run.
