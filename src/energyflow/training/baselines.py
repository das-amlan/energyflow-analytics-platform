import numpy as np


def predict_daily_naive(
    demand: np.ndarray,
    start_indices: np.ndarray,
    horizon: int = 24,
) -> np.ndarray:
    """Repeat the previous 24 hours."""

    offsets = np.arange(horizon)

    return demand[
        start_indices[:, None] - horizon + offsets[None, :]
    ]


def predict_weekly_naive(
    demand: np.ndarray,
    start_indices: np.ndarray,
    horizon: int = 24,
    seasonal_period: int = 168,
) -> np.ndarray:
    """Repeat the same hourly profile from one week earlier."""

    offsets = np.arange(horizon)

    return demand[
        start_indices[:, None] - seasonal_period + offsets[None, :]
    ]


def get_actual_values(
    demand: np.ndarray,
    start_indices: np.ndarray,
    horizon: int = 24,
) -> np.ndarray:
    """Return observed values for each forecast horizon."""

    offsets = np.arange(horizon)

    return demand[start_indices[:, None] + offsets[None, :]]
