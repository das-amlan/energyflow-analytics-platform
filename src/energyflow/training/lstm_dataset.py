import numpy as np
import torch
from torch.utils.data import Dataset


class DemandScaler:
    """Standard scaler fitted on training observations only."""

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)

        self.mean = float(values.mean())
        self.std = float(values.std())

        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise ValueError("Invalid scaling statistics.")

        if self.std == 0:
            raise ValueError("Training demand has zero variance.")

        return self

    def transform(self, values):
        return (np.asarray(values) - self.mean) / self.std

    def inverse_transform(self, values):
        return np.asarray(values) * self.std + self.mean


class ForecastDataset(Dataset):
    """Create 168-hour history and 24-hour target tensors."""

    def __init__(
        self,
        demand,
        windows,
        scaler,
        lookback=168,
        horizon=24,
    ):
        self.demand = np.asarray(demand, dtype=np.float32)
        self.indices = windows["start_idx"].to_numpy(
            dtype=np.int64
        )
        self.scaler = scaler
        self.lookback = lookback
        self.horizon = horizon

        if len(self.indices) == 0:
            raise ValueError("No forecast windows supplied.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]

        history = self.demand[
            start - self.lookback:start
        ]

        future = self.demand[
            start:start + self.horizon
        ]

        x = self.scaler.transform(history).astype(np.float32)
        y = self.scaler.transform(future).astype(np.float32)

        return (
            torch.from_numpy(x[:, None]),
            torch.from_numpy(y),
        )
