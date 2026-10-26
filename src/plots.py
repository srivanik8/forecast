"""figures. every function takes data in and writes a png out, nothing else."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 130
TITLES = {"sarima": "SARIMA(1,1,1)(1,1,1,12)", "holt_winters": "Holt-Winters (additive)"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series(series, path, ylabel="CO₂ (ppm)"):
    """the raw monthly series - look at it before modelling anything."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, lw=1, color="#1f4e79")
    ax.set_title(f"Monthly series, {series.index[0]:%b %Y} – {series.index[-1]:%b %Y}")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_decomposition(decomp, path):
    """observed, trend, seasonal and residual on a shared time axis."""
    parts = [
        ("Observed", decomp.observed),
        ("Trend", decomp.trend),
        ("Seasonal", decomp.seasonal),
        ("Residual", decomp.resid),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, (name, comp) in zip(axes, parts):
        if name == "Residual":
            ax.scatter(comp.index, comp.values, s=4, color="#1f4e79")
            ax.axhline(0, color="grey", lw=0.8)
        else:
            ax.plot(comp.index, comp.values, lw=1, color="#1f4e79")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_title("Seasonal decomposition (additive, period 12)")
    return _save(fig, path)


def plot_seasonal_profile(seasonal, path):
    """one year of the seasonal component, so the shape is readable."""
    profile = seasonal.groupby(seasonal.index.month).mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(months, profile.values, color="#1f4e79")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Seasonal effect (ppm)")
    ax.set_title("Average seasonal effect by month")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_baseline_forecasts(history, actual, forecasts, path, ylabel="CO₂ (ppm)"):
    """each baseline's forecast against what actually happened."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.index, history.values, lw=1, color="grey", label="training data")
    ax.plot(actual.index, actual.values, lw=1.5, color="black", label="actual")
    for name, f in forecasts.items():
        ax.plot(f.index, f.values, lw=1.2, ls="--", label=name.replace("_", " "))
    ax.set_title(f"Baseline forecasts from {actual.index[0]:%b %Y}")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_model_forecasts(history, actual, forecasts, path, ylabel="CO₂ (ppm)"):
    """one panel per model: forecast, 95% interval, and what actually happened."""
    fig, axes = plt.subplots(1, len(forecasts), figsize=(5 * len(forecasts), 3.8),
                             sharey=True, squeeze=False)
    for ax, (name, f) in zip(axes[0], forecasts.items()):
        ax.plot(history.index, history.values, lw=1, color="grey", label="training data")
        ax.fill_between(f.index, f["lower"], f["upper"], color="#1f4e79", alpha=0.2,
                        label="95% interval")
        ax.plot(f.index, f["mean"], lw=1.2, ls="--", color="#1f4e79", label="forecast")
        ax.plot(actual.index, actual.values, lw=1.5, color="black", label="actual")
        ax.set_title(TITLES.get(name, name))
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=8, loc="upper left")
    return _save(fig, path)
