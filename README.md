# Predictive Maintenance of Aircraft Engines - NASA C-MAPSS RUL

Deep learning baseline for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan benchmark. The project trains on all four NASA subsets (`FD001`, `FD002`, `FD003`, `FD004`) simultaneously for robust multi-source learning across multiple operating conditions and fault modes.

## What The Pipeline Does
- Loads training and testing subsets for all four C-MAPSS datasets (`FD001` to `FD004`)
- Parses engine-wise time-series records and handles cross-dataset normalization
- Computes capped training RUL targets (default 125 cycles)
- Builds fixed-length sequences per engine (default 50 cycles)
- Drops channels that are globally constant across the training sets
- Normalizes features using train-only statistics to avoid data leakage
- Trains an advanced Bidirectional PyTorch LSTM (128 hidden units) with a self-attention mechanism, optimized via AdamW and a learning rate scheduler (ReduceLROnPlateau) alongside early stopping
- Evaluates RMSE, MAE, NASA score, and Monte Carlo dropout uncertainty

## Quick Start

### 1. Refresh the official NASA data
```bash
./.venv/bin/python src/download_data.py
```

### 2. Train the Multi-Source Model
```bash
./.venv/bin/python src/train.py --train-datasets FD001 FD002 FD003 FD004 --test-datasets FD001 FD002 FD003 FD004 --mode multi-source
```

### 3. Evaluate the saved checkpoint
```bash
./.venv/bin/python src/evaluate.py --dataset FD002
```

### 4. Run the complete pipeline
```bash
bash run_pipeline.sh
```

### 5. Launch Interactive Dashboard (Backend & Frontend)
```bash
./.venv/bin/python src/api_server.py
```
Access the dashboard at `http://127.0.0.1:8000`

## Advanced Model Architecture (MHA-LSTM)
The system uses a state-of-the-art **Multi-Head Self-Attention LSTM**:
- **Feature Engineering**: Rolling window statistics (mean/std) for temporal dynamics.
- **Backbone**: 3-layer Bidirectional LSTM (hidden size 128).
- **Attention**: Multi-Head Self-Attention (4 heads) for cycle weighting.
- **Head**: 3-layer dense prediction head with dropout.
- **Training**: Universal multi-source training on FD001-FD004.
- **Optimizer**: AdamW + ReduceLROnPlateau.

## Data Notes
- C-MAPSS rows contain `unit id`, `cycle`, `3 operational settings`, and `21 sensor measurements`
- FD001/FD003 contain one operating condition, while FD002/FD004 contain six operating conditions
- The model robustly learns across 1 or 2 fault modes by training on all datasets
- Feature count after filtering may be lower than 24 because globally constant channels are removed

## Saved Artifacts
- `models/lstm_rul.pth`: best validation checkpoint
- `models/scaler.pkl`: train-fit feature scaler
- `models/training_history.json`: epoch-by-epoch metrics

## Diagnostics
```bash
bash test_environment.sh
```

## Notes
- This is a regression problem, so the primary metrics are RMSE and MAE rather than classification accuracy
- Monte Carlo dropout is used for uncertainty estimation after deterministic prediction
- `train_advanced.py` is kept as a thin alias to the main training entrypoint

## References
- NASA C-MAPSS: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
