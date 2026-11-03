# forecast

SARIMA and Holt-Winters forecasts of monthly CO₂, tested against naive
baselines with a rolling-origin backtest, interval coverage and horizon decay.

## The question

Can a statistical model beat the hardest naive baseline on this series, and
when it says it is 95% confident, is it right 95% of the time?

## Data

Mauna Loa atmospheric CO₂ (ppm), from `statsmodels.datasets.co2`. 2,284 weekly
readings are averaged into **526 monthly observations, Mar 1958 – Dec 2001**.
The 5 months with no readings are filled by time interpolation.

The series has a strong, accelerating trend (0.83 ppm/year in the first decade,
1.52 in the last) and a stable annual cycle of about 6 ppm.

![Monthly CO₂ series](results/figures/series.png)

The full step-by-step write-up, with every table, is in
[docs/walkthrough.md](docs/walkthrough.md).

## How to run it

Needs Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_all            # every step, about 15 seconds
python -m pytest                 # all tests
```

Each step also runs on its own, e.g. `python -m src.run_backtest`. Steps 5–7
read the backtest output, so run step 4 first.

## Tests and reproducibility

- **No leakage:** tests check that every forecaster only ever trains on data
  before its origin. Wrecking later data must not change earlier forecasts.
- **Metrics** are checked against hand-worked examples.
- **Data loader** tests cover missing weeks, empty months, unsorted and
  duplicate readings.
- **Reproducible:** dependencies are pinned and nothing in the pipeline is
  random. A test reruns the backtest and checks it matches the committed
  results. `scripts/check_clean_clone.sh` clones the repo, installs from
  scratch, runs everything, and fails if any result changes.

`python -m pytest -m "not slow"` skips the model-fitting tests (about 2
seconds instead of 20).
