#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
STEPS="${STEPS:-1500}"
PREPROCESS_COST="${PREPROCESS_COST:-6}"
WORKER_COUNTS="${WORKER_COUNTS:-1 12}"
PREFIX="${PREFIX:-gpu}"
export MLPLAT_RUNS_DIR="${MLPLAT_RUNS_DIR:-$ROOT/runs}"

cmake -S "$ROOT/agent" -B "$ROOT/agent/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ROOT/agent/build" --target mlplat-agent -j > /dev/null

cd "$ROOT"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0))"
for workers in $WORKER_COUNTS; do
  echo "=== $PREFIX: num_workers=$workers preprocess_cost=$PREPROCESS_COST steps=$STEPS ==="
  "$PYTHON" -m mlplat run --name "$PREFIX-workers-$workers" -- \
    "$PYTHON" examples/train_image_classifier.py --device cuda --steps "$STEPS" \
    --preprocess-cost "$PREPROCESS_COST" --workers "$workers" > /dev/null
  "$PYTHON" -m mlplat analyze latest
  echo
done
"$PYTHON" examples/summarize_runs.py --prefix "$PREFIX"
