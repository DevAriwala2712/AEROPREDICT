import torch
import torch.nn as nn
import numpy as np


class LSTMRULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def predict_with_uncertainty(model, x, n_samples=100, batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.train()
    means = []
    standard_deviations = []
    try:
        with torch.inference_mode():
            for start in range(0, x.size(0), batch_size):
                batch = x[start:start + batch_size]
                predictions = [
                    model(batch).cpu().numpy()
                    for _ in range(n_samples)
                ]
                predictions = np.stack(predictions)
                means.append(predictions.mean(axis=0))
                standard_deviations.append(predictions.std(axis=0))
    finally:
        model.train(was_training)

    return np.concatenate(means, axis=0), np.concatenate(standard_deviations, axis=0)
