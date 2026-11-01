#!/usr/bin/env bash
# check that a clean clone reproduces every committed result.
#
# usage: scripts/check_clean_clone.sh [repo url or path]   (default: this repo)
#
# clones into a temporary folder, installs the pinned requirements into a new
# virtual environment, runs the whole pipeline and the tests, then fails if
# any file under results/ differs from what is committed.
set -euo pipefail

src="${1:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --quiet "$src" "$work/forecast"
cd "$work/forecast"

python3 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

.venv/bin/python -m src.run_all > /dev/null
.venv/bin/python -m pytest -q

if [ -n "$(git status --porcelain results/)" ]; then
    echo "results changed on a clean run:"
    git status --short results/
    exit 1
fi
echo "clean clone reproduces every committed result"
