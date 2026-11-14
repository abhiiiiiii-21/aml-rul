"""PyTorch models for Phase 2. Input shape: (batch, window, n_features).

Every model returns RUL divided by `RUL_SCALE` (shape (batch,)). `train.py`
handles the scaling. Add new architectures here (Transformer, TCN, hybrid
encoders) and register them in `build_model`.
"""
import math
from typing import Optional, Union

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
    """1D convolutions over time, global pooling over time, linear head.

    The defaults reproduce the original two-layer model (no normalisation, no
    residual, average pooling). Options, each changed one at a time in notebook 04:
    `layers`, `kernel` (odd), `dilation` (an int, or "exp" for 1, 2, 4, ...),
    `norm` ("none", "batch" or "layer"), `residual` (skip connection around
    every layer after the first) and `pool` ("avg", "max" or "last").

    Convolutions use symmetric zero padding. This is safe: a sample is a window
    that already ends at the current cycle, so every position in it is in the
    past, and the head pools over the whole window.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        channels: int = 64,
        kernel: int = 5,
        dropout: float = 0.2,
        layers: int = 2,
        dilation: Union[int, str] = 1,
        norm: str = "none",
        residual: bool = False,
        pool: str = "avg",
    ):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so that 'same' padding is symmetric")
        if norm not in ("none", "batch", "layer") or pool not in ("avg", "max", "last"):
            raise ValueError("norm must be none/batch/layer and pool avg/max/last")
        self.pool, self.residual = pool, residual
        self.dilations = [2**i if dilation == "exp" else int(dilation) for i in range(layers)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i, d in enumerate(self.dilations):
            self.convs.append(
                nn.Conv1d(
                    n_features if i == 0 else channels, channels, kernel,
                    padding=d * (kernel - 1) // 2, dilation=d,
                )
            )
            self.norms.append(
                nn.BatchNorm1d(channels) if norm == "batch"
                else nn.GroupNorm(1, channels) if norm == "layer"  # one group = layer norm
                else nn.Identity()
            )
        self.receptive_field = 1 + sum(d * (kernel - 1) for d in self.dilations)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector, shape (batch, channels)."""
        h = x.transpose(1, 2)  # (batch, features, time)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            out = torch.relu(norm(conv(h)))
            h = h + out if (self.residual and i > 0) else out
        if self.pool == "avg":
            return h.mean(dim=-1)
        if self.pool == "max":
            return h.amax(dim=-1)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x))).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Small Transformer encoder over the window, pooling over time, linear head.

    Linear projection of the 14 sensors to `d_model`, positional encoding
    (`pos`: "sinusoidal" or "learned"), `layers` pre-norm encoder blocks
    (multi-head self-attention + feed-forward, residual connections), then
    `pool` ("mean" or "last") over time. Attention cost is O(window^2).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        dim_ff: Optional[int] = None,
        dropout: float = 0.1,
        pos: str = "sinusoidal",
        pool: str = "mean",
        max_len: int = 128,
    ):
        super().__init__()
        if pos not in ("sinusoidal", "learned") or pool not in ("mean", "last"):
            raise ValueError("pos must be sinusoidal/learned and pool mean/last")
        self.pool = pool
        self.proj = nn.Linear(n_features, d_model)
        if pos == "sinusoidal":
            t = torch.arange(max_len).unsqueeze(1)
            freq = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(t * freq)
            pe[:, 1::2] = torch.cos(t * freq[: d_model // 2])
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_ff or 2 * d_model, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.pos_embed[:, : x.shape[1]]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector, shape (batch, d_model)."""
        h = self.encoder(self._embed(x))
        return h.mean(dim=1) if self.pool == "mean" else h[:, -1]

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Head-averaged attention of the first layer, shape (batch, time, time)."""
        layer = self.encoder.layers[0]
        h = layer.norm1(self._embed(x))  # pre-norm: attention sees the normalised input
        _, weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encode(x))).squeeze(-1)


def build_model(name: str, **kwargs) -> nn.Module:
    if name in ("lstm", "gru"):
        return RNNRegressor(kind=name, **kwargs)
    if name == "cnn1d":
        return CNN1DRegressor(**kwargs)
    if name == "transformer":
        return TransformerRegressor(**kwargs)
    raise ValueError(f"unknown model: {name}")
