# AeroPredict

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

- 6 dashboard pages
- 4 supported C-MAPSS datasets: FD001, FD002, FD003, FD004
- 21 sensor channels per engine cycle, plus operational settings
- FD001 evaluation: RMSE 16.1 cycles, MAE 11.8 cycles, best epoch 28
- Data Explorer sample: 1,402 telemetry records
- Model architecture: 2-layer LSTM with Monte Carlo dropout for uncertainty estimates

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

1. Load NASA C-MAPSS train, test, and RUL files.
2. Build capped RUL targets for each engine sequence.
3. Scale features using training-only statistics.
4. Train the LSTM with validation and early stopping.
5. Save the best checkpoint and training history.
6. Serve the model through the Flask API and dashboard pages.

## Setup

### 1. Install dependencies

```bash
bash test_environment.sh
```

### 2. Start the API and dashboard

On Windows:

```powershell
python src/api_server.py
```

On macOS or Linux:

```bash
python3 src/api_server.py
```

Then open:

- http://127.0.0.1:8000/Main_Dashboard.html
- http://127.0.0.1:8000/Live_Prediction_Panel.html
- http://127.0.0.1:8000/Model_Insights.html
- http://127.0.0.1:8000/Model_Details_Architecture.html
- http://127.0.0.1:8000/Data_Explorer.html
- http://127.0.0.1:8000/API_Deployment_Demo.html

### 3. Run the training pipeline

```bash
python src/train.py
```

## API Endpoints

- `/api/summary` returns dataset, configuration, metrics, and artifact metadata
- `/api/history` returns the training curve history
- `/api/sample-prediction` returns one engine prediction with uncertainty
- `/api/all-predictions` returns actual vs predicted RUL scatter data
- `/api/engine-ids` returns valid engine IDs for the selected dataset
- `/api/explorer` returns the telemetry table data
- `/api/notifications` returns dashboard notification items

## Saved Artifacts

- `models/lstm_rul.pth`: trained checkpoint
- `models/scaler.pkl`: fitted scaler used for inference
- `models/training_history.json`: epoch-level training metrics
- `models/training_loss.png`: training curve figure
- `models/predictions_analysis.png`: prediction quality figure

## Data Notes

- FD001 is the main in-distribution evaluation dataset.
- FD002, FD003, and FD004 are exposed as fresh cross-dataset test views in the dashboard.
- Some constant or near-constant channels are dropped during preprocessing.
- The explorer page and dashboard cards are driven by live API responses, not hardcoded mock data.

## Troubleshooting

- If the dashboard says the server is offline, confirm `src/api_server.py` is running.
- If a page appears stale, refresh the browser after restarting the API.
- If training artifacts are missing, rerun the training step so the model files and history JSON are regenerated.

## References

- NASA C-MAPSS data: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Saxena, Goebel, Simon, and Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
