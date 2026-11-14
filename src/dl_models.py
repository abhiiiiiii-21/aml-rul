"""PyTorch models for Phase 2. Input shape: (batch, window, n_features).

Every model returns RUL divided by `RUL_SCALE` (shape (batch,)). `train.py`
handles the scaling. Add new architectures here (Transformer, TCN, hybrid
encoders) and register them in `build_model`.
"""
import torch
from torch import nn

N_FEATURES = 14


class RNNRegressor(nn.Module):
    """LSTM or GRU encoder, last hidden state, linear head."""

    def __init__(
        self,
        kind: str = "lstm",
        n_features: int = N_FEATURES,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x))).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """1D convolutions over time, global average pooling, linear head."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        channels: int = 64,
        kernel: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x))).squeeze(-1)


def build_model(name: str, **kwargs) -> nn.Module:
    if name in ("lstm", "gru"):
        return RNNRegressor(kind=name, **kwargs)
    if name == "cnn1d":
        return CNN1DRegressor(**kwargs)
    raise ValueError(f"unknown model: {name}")
