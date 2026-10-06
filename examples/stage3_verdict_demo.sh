#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-150}"
THREADS="${THREADS:-4}"
PREPROCESS_COST="${PREPROCESS_COST:-6}"
WORKER_COUNTS="${WORKER_COUNTS:-0 1 6}"
export MLPLAT_RUNS_DIR="${MLPLAT_RUNS_DIR:-$ROOT/runs}"

cmake -S "$ROOT/agent" -B "$ROOT/agent/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ROOT/agent/build" --target mlplat-agent -j > /dev/null

cd "$ROOT"
for workers in $WORKER_COUNTS; do
  echo "=== scenario: num_workers=$workers device=$DEVICE ==="
  "$PYTHON" -m mlplat run --name "demo-$DEVICE-workers-$workers" -- \
    "$PYTHON" examples/train_image_classifier.py --device "$DEVICE" --threads "$THREADS" --steps "$STEPS" \
    --preprocess-cost "$PREPROCESS_COST" --workers "$workers" > /dev/null
  "$PYTHON" -m mlplat analyze latest
  echo
done
"$PYTHON" -m mlplat list
