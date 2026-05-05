#!/usr/bin/env bash

# Predictive Maintenance RUL Prediction - Complete Pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then
	PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
	PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
	PYTHON_BIN="python"
fi

cd "$SCRIPT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

echo "================================"
echo "RUL Prediction Pipeline"
echo "================================"

cd "$SCRIPT_DIR"

echo ""
echo "1. Skipping data download as data already exists..."
# "$PYTHON_BIN" src/download_data.py

echo ""
echo "2. Training model on FD001, FD002, FD003, FD004..."
"$PYTHON_BIN" src/train.py --train-datasets FD001 FD002 FD003 FD004 --test-datasets FD001 FD002 FD003 FD004 --mode multi-source

echo ""
echo "3. Evaluating saved checkpoint..."
"$PYTHON_BIN" src/evaluate.py --dataset FD001

echo ""
echo "================================"
echo "Pipeline completed successfully!"
echo "================================"
