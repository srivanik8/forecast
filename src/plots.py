"""
three figures for the readme.

reads the csvs that forecast.py wrote, so run that first. nothing here
recomputes a model - plots should never disagree with the numbers.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

#no display in ci or on a server, so pick the file backend before pyplot loads
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forecast import (  # noqa: E402
    HORIZON,
    SEASON,
    load_series,
    origins,
    sarima,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

#one fixed hue order, used the same way in every figure
BLUE = "#2563eb"     #whatever actually happened
AMBER = "#d97706"    #whatever we predicted
TEAL = "#0d9488"     #the baseline we are trying to beat

SURFACE = "#fcfcfb"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"

NICE_NAMES = {
    "sarima": "sarima (1,1,1)(1,1,1,12)",
    "holt_winters": "holt-winters (additive)",
    "seasonal_naive_drift": "seasonal naive + drift",
    "seasonal_naive": "seasonal naive",
    "naive": "naive (last value)",
}


def style_axes(ax):
    """recessive grid and axes so the data is the loudest thing on the page"""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def new_figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# figure 1 - the forecast against what really happened
# ---------------------------------------------------------------------------

def plot_forecast(y):
    #refit on the very last origin, the one that leaves a full unseen year
    cut = origins(len(y))[-1]
    train = y.iloc[:cut]
    actual = y.iloc[cut:cut + HORIZON]

    point, lo, hi = sarima(train, HORIZON, with_interval=True)
    forecast = pd.Series(point, index=actual.index)

    #five years of context is enough to read the season without squashing it
    history = train.iloc[-SEASON * 5:]

    fig, ax = new_figure(9, 4.6)

    ax.plot(history.index, history.values, color=BLUE, linewidth=2,
            label="observed")
    #join the last training point to the test window so the line has no gap
    joined = pd.concat([train.iloc[-1:], actual])
    ax.plot(joined.index, joined.values, color=BLUE, linewidth=2)

    ax.fill_between(forecast.index, lo, hi, color=AMBER, alpha=0.18,
                    linewidth=0, label="95% interval")
    ax.plot(forecast.index, forecast.values, color=AMBER, linewidth=2,
            linestyle="--", marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="forecast")

    #a thin marker for where the model stopped being allowed to see the data
    ax.axvline(train.index[-1], color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("forecast origin", xy=(train.index[-1], ax.get_ylim()[0]),
                xytext=(-6, 12), textcoords="offset points",
                ha="right", fontsize=8.5, color=MUTED)

    ax.set_title("twelve months ahead, against the actual outcome",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_ylabel("co2 (ppm)", fontsize=9.5, color=MUTED)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK)

    save(fig, "forecast_vs_actual.png")


# ---------------------------------------------------------------------------
# figure 2 - every model against the baseline
# ---------------------------------------------------------------------------

def plot_model_comparison():
    summary = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    summary = summary.sort_values("mae", ascending=False)

    labels = [NICE_NAMES.get(m, m) for m in summary["model"]]
    #colour follows the entity - the baseline stays teal wherever it ranks
    colors = [TEAL if m == "seasonal_naive" else BLUE for m in summary["model"]]

    fig, ax = new_figure(8.4, 4.2)
    ax.grid(axis="y", visible=False)

    bars = ax.barh(labels, summary["mae"], height=0.62, color=colors)
    for bar in bars:
        bar.set_linewidth(2)
        bar.set_edgecolor(SURFACE)

    #direct labels beat making anyone read values off an axis
    for bar, value in zip(bars, summary["mae"]):
        ax.annotate(f"{value:.2f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)

    reference = float(summary.loc[summary["model"] == "seasonal_naive", "mae"].iloc[0])
    ax.axvline(reference, color=TEAL, linewidth=1, linestyle="--", alpha=0.7)
    #pin the caption to the top of the line, data x but axes y
    ax.text(reference, 1.0, " the baseline to beat",
            transform=ax.get_xaxis_transform(), va="top", ha="left",
            fontsize=8.5, color=TEAL)

    ax.set_title("mean absolute error over 8 rolling origins",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_xlabel("mae (ppm) - lower is better", fontsize=9.5, color=MUTED)
    ax.set_xlim(0, summary["mae"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_color(INK)

    save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# figure 3 - accuracy decaying with distance
# ---------------------------------------------------------------------------

def plot_error_by_horizon():
    by_step = pd.read_csv(os.path.join(RESULTS, "error_by_horizon.csv"))

    fig, ax = new_figure(8.4, 4.0)

    ax.plot(by_step["months_ahead"], by_step["mae"], color=BLUE, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)

    #label the two ends only, the shape carries everything in between
    for idx in (0, len(by_step) - 1):
        row = by_step.iloc[idx]
        ax.annotate(f"{row['mae']:.2f}", xy=(row["months_ahead"], row["mae"]),
                    xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK)

    growth = by_step["mae"].iloc[-1] / by_step["mae"].iloc[0]
    ax.set_title(f"error roughly {growth:.1f}x worse at twelve months than at one",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_xlabel("months ahead", fontsize=9.5, color=MUTED)
    ax.set_ylabel("mae (ppm)", fontsize=9.5, color=MUTED)
    ax.set_xticks(np.arange(1, HORIZON + 1))
    ax.set_ylim(0, by_step["mae"].max() * 1.32)

    save(fig, "error_by_horizon.png")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    y = load_series()

    plot_forecast(y)
    plot_model_comparison()
    plot_error_by_horizon()


if __name__ == "__main__":
    main()
