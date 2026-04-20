"""RUL prediction package for NASA C-MAPSS turbofan degradation modeling."""

__version__ = "1.0.0"
__author__ = "Rolls-Royce Aerospace"
__description__ = "Deep Learning for Remaining Useful Life Prediction"

from .model import LSTMRULPredictor, predict_with_uncertainty
from .data_loader import (
    create_sequences_per_engine,
    load_data,
    nasa_score,
    prepare_test_data,
    prepare_train_data,
)

__all__ = [
    'LSTMRULPredictor',
    'predict_with_uncertainty',
    'load_data',
    'prepare_train_data',
    'prepare_test_data',
    'create_sequences_per_engine',
    'nasa_score',
]
