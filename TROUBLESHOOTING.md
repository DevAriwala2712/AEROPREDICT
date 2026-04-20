# Troubleshooting Guide

## Common Issues

### Dataset files contain HTML instead of numeric rows
Cause: an old download URL saved webpage content instead of the real NASA files.

Fix:
```bash
/Users/devariwala/development/no10/.venv/bin/python /Users/devariwala/development/no10/src/download_data.py
```

### Missing dataset file
Fix:
```bash
/Users/devariwala/development/no10/.venv/bin/python /Users/devariwala/development/no10/src/download_data.py
```

### Torch import or dependency errors
Fix:
```bash
/Users/devariwala/development/no10/.venv/bin/python -m pip install --upgrade -r /Users/devariwala/development/no10/requirements.txt
```

### Training is slow
Tips:
```bash
/Users/devariwala/development/no10/.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
/Users/devariwala/development/no10/.venv/bin/python /Users/devariwala/development/no10/src/train.py --dataset FD001 --epochs 10 --batch-size 64
```

### Out-of-memory during training
Try:
```bash
/Users/devariwala/development/no10/.venv/bin/python /Users/devariwala/development/no10/src/train.py --dataset FD001 --batch-size 32 --seq-length 30
```

## Diagnostic Commands

### Full environment check
```bash
bash /Users/devariwala/development/no10/test_environment.sh
```

### Data loading check
```bash
/Users/devariwala/development/no10/.venv/bin/python -c "from src.data_loader import load_data; train_df, test_df, rul_df = load_data('FD001'); print(train_df.shape, test_df.shape, rul_df.shape)"
```

### Evaluation check
```bash
/Users/devariwala/development/no10/.venv/bin/python /Users/devariwala/development/no10/src/evaluate.py --dataset FD001 --mc-samples 10
```

## Reset Steps
```bash
cd /Users/devariwala/development/no10
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete
/Users/devariwala/development/no10/.venv/bin/python -m pip install --upgrade -r requirements.txt
/Users/devariwala/development/no10/.venv/bin/python src/download_data.py
bash test_environment.sh
```
