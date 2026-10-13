# GitHub, portfolio, and interview strategy

## Suggested repository identity

- **Repository name:** `ai-energy-consumption-forecasting`
- **Description:** `Reproducible hourly energy-demand forecasting demo with synthetic facility data, time-aware ML evaluation, and business-facing outputs.`
- **Topics:** `energy-forecasting`, `time-series`, `machine-learning`, `scikit-learn`, `python`, `energy-analytics`, `synthetic-data`

## Publish from a clean local checkout

Create an empty GitHub repository with the name above. Do not initialize it
with a second README if you already have one locally. This project lives inside
a larger Replit workspace, so first download/export or copy only the
`energy-forecasting/` folder into a clean standalone directory on your computer.
Then open a terminal in that standalone folder:

```bash
git init
git add README.md requirements.txt .gitignore main.py src tests docs notebooks data images outputs models
git status
git commit -m "Build reproducible energy forecasting pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-energy-consumption-forecasting.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Inspect `git status` before
committing. The generated source CSV and fitted model binary are ignored;
`model_metadata.json` and the small result charts/reports may be committed as
proof. The generator recreates the demo CSV after cloning. Never commit tokens,
private meter readings, customer information, or organization-specific
operational data.

## A useful commit sequence

Use small commits that tell the project story:

1. `docs: define use case and synthetic-data assumptions`
2. `data: add repeatable hourly facility simulation`
3. `features: add leakage-aware calendar and lag features`
4. `models: compare seasonal baseline ridge and random forest`
5. `reports: save holdout metrics forecast and charts`
6. `tests: cover cleaning features and forecast behavior`

The exact commit order may differ; commit tested milestones and write messages
that explain the intent. Avoid committing every generated output on every run.

## Make the repository easy to review

- Put the use case, one-command run, chart preview, metrics, and limitations near
  the top of `README.md`.
- Add a short caption to each chart explaining what the reader should notice.
- Keep the synthetic-data label visible; do not imply that synthetic readings
  came from a factory or utility.
- Include a source/license note before adding any downloaded public dataset.
- State the evaluation method and compare against the simple baseline.
- Keep `requirements.txt` and `tests/` current.
- Add a short `docs/` note before extending the project into a dashboard, API,
  or live data pipeline.
- If you change the model, feature set, or forecast horizon, rerun the full
  pipeline and update the evidence consistently.

## Proof-building checklist

Capture these items for a portfolio or internship application:

1. The GitHub repository page with the README and charts visible.
2. The terminal showing the documented run completing successfully.
3. The model comparison report showing baseline and ML metrics.
4. The holdout chart and future forecast chart.
5. A short note on one error pattern you found and a change you tested.

Do not fabricate metrics or annotate a chart as if it were real facility data.
For extra data realism, later substitute an openly licensed public energy
dataset and document its source, time zone, units, and any processing.

## Two-minute interview explanation

> I built a reproducible hourly demand-forecasting project with a clearly
> labeled synthetic facility dataset because I did not have access to corporate
> meter data. I compare a prior-day baseline against Ridge regression and
> Random Forest using a chronological holdout, then refit the best candidate on
> all history and generate a recursive 48-hour forecast. The outputs include
> RMSE, MAE, R², residual charts, and an illustrative flat-tariff cost scenario.
> The key limitation is that synthetic data and a temperature proxy demonstrate
> the workflow but do not establish real-world savings or deployment accuracy.

Be ready to explain why time-series data must not be randomly shuffled, what
RMSE means in kWh, how a negative R² should be interpreted, why the seasonal
baseline matters, and what you would validate before a real operational
deployment.
