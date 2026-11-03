# forecast

Monthly time-series forecasting with honest validation: can a statistical model
beat a naive baseline, and can its stated confidence be trusted?

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
