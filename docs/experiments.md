# Experiments: can the models be improved?

The main results use two untuned models. This page tests five candidate
improvements and three harder benchmarks in the **same** rolling-origin
backtest: 8 origins (1994–2001), 12 months ahead, every forecaster refit at
every origin on data before it. None of the candidates replace the main
results. Run `python -m src.experiments` (about a minute) to rebuild
`results/experiments.csv`.

## Candidates

| Candidate | Idea |
|---|---|
| Holt-Winters on log CO₂ | Fit on the log scale, map back with exp. Intervals come out wider above than below, the side every miss fell on. |
| Damped Holt-Winters | Let the trend flatten over the horizon. |
| AIC-tuned SARIMA | Try 24 order combinations at each origin, keep the lowest AIC. The choice sees only the training window. |
| Ensemble | Average SARIMA and Holt-Winters. |
| Recent drift (3, 5, 10 years) | Seasonal naive + drift, averaging the yearly rise over recent years only, since growth speeds up. |

## Results

| Forecaster | MAE (ppm) | Relative MAE vs drift | Beats drift | Coverage | Coverage in 1998 | Fit time (s) |
|---|---|---|---|---|---|---|
| Holt-Winters on log CO₂ | **0.451** | 0.832 | 6 of 8 | **96.9%** | **9 of 12** | 1.0 |
| AIC-tuned SARIMA | 0.455 | 0.761 | 6 of 8 | 92.7% | 5 of 12 | 56.6 |
| Ensemble | 0.456 | 0.731 | 7 of 8 | 92.7% | 5 of 12 | 6.4 |
| SARIMA (main) | 0.457 | 0.759 | 6 of 8 | 92.7% | 5 of 12 | 5.2 |
| Holt-Winters (main) | 0.461 | **0.718** | **8 of 8** | 91.7% | 5 of 12 | 1.1 |
| Damped Holt-Winters | 0.552 | 0.868 | 7 of 8 | 90.6% | 5 of 12 | 1.0 |
| Drift, last 10 years | 0.572 | 0.920 | 6 of 8 | – | – | 0.0 |
| Seasonal naive + drift (main benchmark) | 0.658 | 1.000 | – | – | – | 0.0 |
| Drift, last 5 years | 0.704 | 1.175 | 2 of 8 | – | – | 0.0 |
| Drift, last 3 years | 0.760 | 1.350 | 2 of 8 | – | – | 0.0 |

Fit time is the total over all 8 origins on one machine, so treat it as a
rough comparison only.

## What it shows

- **No candidate meaningfully improves accuracy.** The best MAE moves from
  0.457 to 0.451, about 1%. On 96 forecasts that is noise, not an
  improvement.
- **Log-scale Holt-Winters fixes calibration, not accuracy.** Coverage rises
  from 91.7% to 96.9%, and in 1998 it covers 9 of 12 months instead of 5.
  The price is intervals about 13% wider. This is the one candidate worth
  keeping in mind.
- **Tuning bought nothing.** The AIC search picked (1,1,1)(0,1,1,12) at every
  origin, the default without its seasonal AR term, and it lands almost
  exactly where the default does at 11× the cost.
- **The ensemble is steadier but not better.** It beats the drift benchmark
  in 7 of 8 years, but Holt-Winters alone already does 8 of 8.
- **Damping hurts.** CO₂ did not level off, so a trend that flattens loses.
- **A harder benchmark exists.** Averaging the drift over the last 10 years
  cuts the benchmark's MAE from 0.658 to 0.572. Against it, the main models'
  edge shrinks from about 25% to about 16–17%. But I picked 10 years after
  seeing all three windows, and 3 or 5 years do *worse* than the full
  history, so this benchmark is tuned on the test years and flatters itself.
  An honest version would choose the window inside each training window.

## Takeaway

On this series, model choice moves the error by a percent or two. The
benchmark you compare against moves the headline by ten. Getting the
comparison right mattered more than getting the model right.
