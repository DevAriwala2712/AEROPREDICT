# Troubleshooting Guide

## Common Issues

### Dataset files contain HTML instead of numeric rows
Cause: an old download URL saved webpage content instead of the real NASA files.

Fix:
```bash
./.venv/bin/python src/download_data.py
```

### Missing dataset file
Fix:
```bash
./.venv/bin/python src/download_data.py
```

### Torch import or dependency errors
Fix:
```bash
./.venv/bin/python -m pip install --upgrade -r requirements.txt
```

### Training is slow
Tips:
```bash
./.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
./.venv/bin/python src/train.py --dataset FD001 --epochs 10 --batch-size 64
```

### Out-of-memory during training
Try:
```bash
./.venv/bin/python src/train.py --dataset FD001 --batch-size 32 --seq-length 30
```

## Diagnostic Commands

### Full environment check
```bash
bash test_environment.sh
```

### Data loading check
```bash
./.venv/bin/python -c "from src.data_loader import load_data; train_df, test_df, rul_df = load_data('FD001'); print(train_df.shape, test_df.shape, rul_df.shape)"
```

### Evaluation check
```bash
./.venv/bin/python src/evaluate.py --dataset FD001 --mc-samples 10
```

## Reset Steps
```bash
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete
./.venv/bin/python -m pip install --upgrade -r requirements.txt
./.venv/bin/python src/download_data.py
bash test_environment.sh
```
