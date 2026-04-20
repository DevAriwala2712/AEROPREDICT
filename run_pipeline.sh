#!/bin/bash

# Predictive Maintenance RUL Prediction - Complete Pipeline

set -e  # Exit on error

echo "================================"
echo "RUL Prediction Pipeline"
echo "================================"

cd /Users/devariwala/development/no10

echo ""
echo "1. Downloading official NASA C-MAPSS data..."
/Users/devariwala/development/no10/.venv/bin/python src/download_data.py

echo ""
echo "2. Training model on FD001..."
/Users/devariwala/development/no10/.venv/bin/python src/train.py --dataset FD001

echo ""
echo "3. Evaluating saved checkpoint..."
/Users/devariwala/development/no10/.venv/bin/python src/evaluate.py --dataset FD001

echo ""
echo "================================"
echo "Pipeline completed successfully!"
echo "================================"
