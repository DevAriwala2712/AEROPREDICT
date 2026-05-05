import torch
import torch.nn as nn
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        # Pooling over sequence length: use the attention weights to weight the context
        # Or just take the mean/last. Let's use a simple global average pooling of the context
        pooled = torch.mean(context, dim=1)
        return self.out_linear(pooled), attn_weights

class LSTMRULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=self.bidirectional
        )
        self.attention = MultiHeadSelfAttention(hidden_size * self.num_directions, num_heads=4)
        self.dropout = nn.Dropout(dropout)
        
        # Deeper prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out, attn_weights = self.attention(out)
        
        out = self.head(out)
        return out


def predict_with_uncertainty(model, x, n_samples=100):
    was_training = model.training
    model.train()
    predictions = []
    for _ in range(n_samples):
        pred = model(x)
        predictions.append(pred.detach().cpu().numpy())
    if not was_training:
        model.eval()
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    return mean, std
