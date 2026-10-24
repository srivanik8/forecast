# forecast

Monthly time-series forecasting with honest validation: can a statistical model
beat a naive baseline, and can its stated confidence be trusted?

## Data

Mauna Loa atmospheric CO₂ (ppm) from `statsmodels.datasets.co2`: weekly readings
resampled to **526 monthly observations** (Mar 1958 – Dec 2001). 5 months with
no readings are filled by time interpolation.

![Monthly series](results/figures/series.png)

## Exploration

![Seasonal decomposition](results/figures/decomposition.png)

- **Trend:** strong and accelerating, from 0.83 ppm/year in the first decade
  to 1.52 ppm/year in the last.
- **Seasonality:** stable annual cycle, peaking in May and bottoming out in
  October (about 6.1 ppm peak to trough).
- **Residuals:** small (std 0.28 ppm).

## How to run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.explore            # summary + figures into results/
python -m pytest                 # tests
```
