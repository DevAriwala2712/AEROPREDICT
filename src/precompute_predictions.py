from __future__ import annotations

import numpy as np

from api_server import (
    SUPPORTED_DATASETS,
    _get_dataset_cache,
    _write_prediction_cache,
    service,
)


def main() -> None:
    if not service.available or service.model is None or service.scaler is None:
        raise RuntimeError("Cannot precompute predictions without a loaded model and scaler")

    caches = {}
    datasets = list(dict.fromkeys(service.train_datasets + service.test_datasets))
    for dataset in datasets:
        print(f"Precomputing predictions for {dataset}...")
        cache = _get_dataset_cache(dataset, force_recompute=True)
        if "error" in cache:
            raise RuntimeError(f"Failed to precompute {dataset}: {cache['error']}")
        _write_prediction_cache(dataset, cache)
        caches[dataset] = cache

    if set(SUPPORTED_DATASETS).issubset(service.train_datasets):
        unit_ids = []
        targets = []
        predictions = []
        sequences = []
        offset = 0

        for dataset in SUPPORTED_DATASETS:
            cache = caches.get(dataset)
            if cache is None:
                raise RuntimeError(f"Missing prediction cache for {dataset}")
            dataset_ids = cache["unit_ids"]
            unit_ids.append(dataset_ids + offset)
            targets.append(cache["y_test"])
            predictions.append(cache["predictions"])
            sequences.append(cache["X_test"])
            offset += int(dataset_ids.max()) if len(dataset_ids) else 0

        multi_cache = {
            "dataset": "MULTI",
            "X_test": np.concatenate(sequences, axis=0),
            "y_test": np.concatenate(targets, axis=0),
            "unit_ids": np.concatenate(unit_ids, axis=0),
            "predictions": np.concatenate(predictions, axis=0),
            "feature_columns": list(service.config.get("feature_columns", [])),
            "missing_features": [],
            "scaler": service.scaler,
            "use_original_scaler": True,
        }
        _write_prediction_cache("MULTI", multi_cache)
        print("Precomputed combined MULTI cache.")

    service._ensure_artifacts()


if __name__ == "__main__":
    main()
