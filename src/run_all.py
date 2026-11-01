"""
run the whole pipeline, steps 1 to 7, in order.

run from the repo root:  python -m src.run_all

every result in results/ is rebuilt from scratch. nothing in the pipeline is
random, so a rerun reproduces the committed results exactly.
"""

import time

from src import (compare_baselines, coverage, evaluate, explore, fit_models, horizon,
                 run_backtest)

STEPS = [
    ("1 explore", explore),
    ("2 baselines", compare_baselines),
    ("3 models", fit_models),
    ("4 backtest", run_backtest),
    ("5 metrics", evaluate),
    ("6 coverage", coverage),
    ("7 horizon", horizon),
]


def main():
    start = time.time()
    for name, step in STEPS:
        print(f"\n=== step {name} ===\n")
        step.main()
    print(f"\nall steps done in {time.time() - start:.0f}s, results in results/")


if __name__ == "__main__":
    main()
