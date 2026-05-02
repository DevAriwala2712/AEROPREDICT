from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from data_loader import (
    DEFAULT_DATASET,
    create_sequences_per_engine,
    load_data,
    nasa_score,
    prepare_test_samples,
    prepare_train_data,
)
from model import LSTMRULPredictor, predict_with_uncertainty

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LSTM RUL model on NASA C-MAPSS data.")
    parser.add_argument(
        "--dataset",
        default="ALL",
        help="Dataset to evaluate (FD001/FD002/FD003/FD004) or ALL (default).",
    )
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
            "train_datasets": [DEFAULT_DATASET],
            "test_datasets": [DEFAULT_DATASET],
            "seq_length": 50,
            "max_rul": 125,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "feature_columns": None,
        },
        "metrics": {},
    }


def resolve_eval_datasets(requested: str, config: dict) -> list[str]:
    requested = requested.upper().strip()
    if requested == "ALL":
        configured = config.get("test_datasets") or config.get("train_datasets") or [config.get("dataset", DEFAULT_DATASET)]
        datasets = [str(d).upper() for d in configured]
        datasets = [d for d in datasets if d in ALL_DATASETS]
        if datasets:
            return datasets
        return ALL_DATASETS

    if requested not in ALL_DATASETS:
        raise ValueError(f"Unsupported dataset: {requested}. Allowed: {ALL_DATASETS + ['ALL']}")
    return [requested]


def make_scaler_for_dataset(
    dataset: str,
    train_datasets: list[str],
    saved_scaler: StandardScaler,
    trained_features: list[str],
    seq_length: int,
    max_rul: int,
) -> tuple[StandardScaler, list[str], list[str]]:
    train_df, _, _ = load_data(dataset)
    prepared_train_df, feature_columns, target = prepare_train_data(train_df, max_rul=max_rul)

    available = [f for f in trained_features if f in feature_columns]
    missing = [f for f in trained_features if f not in feature_columns]

    if dataset in train_datasets:
        return saved_scaler, available, missing

    X_train_ds, _ = create_sequences_per_engine(prepared_train_df, available, target, seq_length=seq_length)
    if missing:
        zeros_tr = np.zeros((X_train_ds.shape[0], X_train_ds.shape[1], len(missing)), dtype=np.float32)
        X_train_ds = np.concatenate([X_train_ds, zeros_tr], axis=-1)

    fresh_scaler = StandardScaler()
    fresh_scaler.fit(X_train_ds.reshape(-1, X_train_ds.shape[-1]))
    return fresh_scaler, available, missing


def evaluate_dataset(
    dataset: str,
    model: LSTMRULPredictor,
    config: dict,
    saved_scaler: StandardScaler,
    train_datasets: list[str],
    mc_samples: int,
    device: torch.device,
) -> dict:
    seq_length = int(config.get("seq_length", 50))
    max_rul = int(config.get("max_rul", 125))
    trained_features = list(config.get("feature_columns") or [])

    _, test_df, rul_df = load_data(dataset)
    scaler, available, missing = make_scaler_for_dataset(
        dataset,
        train_datasets,
        saved_scaler,
        trained_features,
        seq_length,
        max_rul,
    )

    X_test, y_test, _ = prepare_test_samples(
        test_df,
        rul_df,
        available,
        seq_length=seq_length,
        max_rul=max_rul,
    )
    if missing:
        zeros = np.zeros((X_test.shape[0], X_test.shape[1], len(missing)), dtype=np.float32)
        X_test = np.concatenate([X_test, zeros], axis=-1)

    X_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    score = float(nasa_score(y_test, predictions))

    mean_pred, std_pred = predict_with_uncertainty(model, X_test_tensor, n_samples=mc_samples)
    mean_pred = mean_pred.flatten()
    std_pred = std_pred.flatten()

    return {
        "dataset": dataset,
        "rmse": rmse,
        "mae": mae,
        "nasa_score": score,
        "avg_uncertainty_std": float(std_pred.mean()),
        "min_uncertainty_std": float(std_pred.min()),
        "max_uncertainty_std": float(std_pred.max()),
        "samples": int(len(y_test)),
        "missing_features": missing,
        "mean_prediction_sample": mean_pred[:5].tolist(),
        "uncertainty_sample": std_pred[:5].tolist(),
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    config = checkpoint["config"]

    with Path(args.scaler).open("rb") as handle:
        saved_scaler = pickle.load(handle)

    train_datasets = [str(d).upper() for d in (config.get("train_datasets") or [config.get("dataset", DEFAULT_DATASET)])]
    eval_datasets = resolve_eval_datasets(args.dataset, config)

    device = get_device()
    feature_columns = list(config.get("feature_columns") or [])
    model = LSTMRULPredictor(
        input_size=int(config.get("input_size", len(feature_columns))),
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results = []
    for dataset in eval_datasets:
        results.append(
            evaluate_dataset(
                dataset,
                model,
                config,
                saved_scaler,
                train_datasets,
                args.mc_samples,
                device,
            )
        )

    print("Model Evaluation Summary")
    print("=" * 80)
    print(f"Trained on: {train_datasets}")
    print(f"Evaluated on: {eval_datasets}")
    print(f"Device: {device}")
    print("-" * 80)
    print(f"{'Dataset':<10} | {'Samples':<8} | {'RMSE':<8} | {'MAE':<8} | {'NASA':<10} | {'AvgStd':<8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['dataset']:<10} | {r['samples']:<8} | {r['rmse']:<8.2f} | {r['mae']:<8.2f} | "
            f"{r['nasa_score']:<10.2f} | {r['avg_uncertainty_std']:<8.3f}"
        )

    print("\nDetailed Stats")
    print("=" * 80)
    for r in results:
        print(f"Dataset: {r['dataset']}")
        print(f"  RMSE: {r['rmse']:.4f}")
        print(f"  MAE: {r['mae']:.4f}")
        print(f"  NASA Score: {r['nasa_score']:.4f}")
        print(
            f"  Uncertainty std (avg/min/max): "
            f"{r['avg_uncertainty_std']:.4f} / {r['min_uncertainty_std']:.4f} / {r['max_uncertainty_std']:.4f}"
        )
        print(f"  Missing features padded: {r['missing_features']}")
        print(f"  Mean prediction sample: {r['mean_prediction_sample']}")
        print(f"  Uncertainty sample: {r['uncertainty_sample']}")
        print("-" * 80)

    print("\nCheckpoint Metrics (saved at training time)")
    print("=" * 80)
    print(json.dumps(checkpoint.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
