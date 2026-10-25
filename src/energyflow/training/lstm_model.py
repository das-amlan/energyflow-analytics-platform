import torch
from torch import nn


class EnergyLSTM(nn.Module):
    """Forecast multiple future hours from historical demand."""

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=24,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x: [batch, 168, 1]
        output, _ = self.lstm(x)

        # Last hidden output
        last_output = output[:, -1, :]

        # Result: [batch, 24]
        return self.head(last_output)
