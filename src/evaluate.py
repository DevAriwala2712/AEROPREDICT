from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import DEFAULT_DATASET, nasa_score, load_data, prepare_test_data, prepare_train_data
from model import LSTMRULPredictor, predict_with_uncertainty

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LSTM RUL model on NASA C-MAPSS data.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="C-MAPSS subset to evaluate, default: FD001")
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "lstm_rul.pth"))
    parser.add_argument("--scaler", default=str(MODELS_DIR / "scaler.pkl"))
    parser.add_argument("--mc-samples", type=int, default=100)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {
        "state_dict": checkpoint,
        "config": {
            "dataset": DEFAULT_DATASET,
            "seq_length": 50,
            "max_rul": 125,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "feature_columns": None,
        },
        "metrics": {},
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    config = checkpoint["config"]
    dataset = args.dataset.upper()
    seq_length = int(config.get("seq_length", 50))
    max_rul = config.get("max_rul", 125)
    feature_columns = config.get("feature_columns")

    train_df, test_df, rul_df = load_data(dataset)
    if feature_columns is None:
        _, feature_columns, _ = prepare_train_data(train_df, max_rul=max_rul)
    X_test, y_test = prepare_test_data(test_df, rul_df, feature_columns, seq_length=seq_length, max_rul=max_rul)

    with Path(args.scaler).open("rb") as handle:
        scaler = pickle.load(handle)
    try:
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    except ValueError as exc:
        raise ValueError(
            "Saved scaler does not match the current feature set. "
            "Re-run src/train.py after refreshing the dataset."
        ) from exc

    device = get_device()
    model = LSTMRULPredictor(
        input_size=len(feature_columns),
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    score = nasa_score(y_test, predictions)

    mean_pred, std_pred = predict_with_uncertainty(model, X_test_tensor, n_samples=args.mc_samples)
    mean_pred = mean_pred.flatten()
    std_pred = std_pred.flatten()

    print(f"Dataset: {dataset}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"NASA Score: {score:.4f}")
    print(f"Mean prediction sample: {mean_pred[:5]}")
    print(f"Uncertainty (std) sample: {std_pred[:5]}")
    print(f"Average uncertainty (std): {std_pred.mean():.4f}")


if __name__ == "__main__":
    main()
