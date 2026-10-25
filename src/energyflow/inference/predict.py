import numpy as np
import torch


def predict_demand(
    model,
    history,
    scaler,
    lookback=168,
):
    """
    Generate a 24-hour demand forecast.

    history: 1D historical demand array
    return: predicted demand in MW
    """

    values = np.asarray(history, dtype=np.float32)

    if values.ndim != 1:
        raise ValueError("History must be a 1D array.")

    if len(values) != lookback:
        raise ValueError(
            f"Expected {lookback} observations, "
            f"received {len(values)}."
        )

    if not np.isfinite(values).all():
        raise ValueError("History contains invalid values.")

    mean = float(scaler["mean"])
    std = float(scaler["std"])

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid scaler configuration.")

    normalized = (values - mean) / std

    tensor = torch.from_numpy(
        normalized.reshape(1, lookback, 1)
    )

    model.eval()

    with torch.inference_mode():
        prediction = model(tensor).cpu().numpy()[0]

    forecast = prediction * std + mean

    return forecast.astype(np.float64)


def predict_demand_batch(
    model,
    histories,
    scaler,
    lookback=168,
    batch_size=128,
):
    """Forecast multiple historical windows efficiently."""

    values = np.asarray(histories, dtype=np.float32)

    if values.ndim != 2 or values.shape[1] != lookback:
        raise ValueError(
            f"Expected shape (n, {lookback})."
        )

    if not np.isfinite(values).all():
        raise ValueError("Input contains invalid values.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    mean = float(scaler["mean"])
    std = float(scaler["std"])

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid scaler configuration.")

    predictions = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            normalized = (batch - mean) / std

            tensor = torch.from_numpy(
                normalized[:, :, None]
            )

            output = model(tensor).cpu().numpy()
            predictions.append(output * std + mean)

    if not predictions:
        raise ValueError("No histories supplied.")

    return np.vstack(predictions).astype(np.float64)
