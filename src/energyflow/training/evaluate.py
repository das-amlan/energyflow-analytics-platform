import numpy as np


def evaluate_forecast(y_true, y_pred) -> dict:
    """Calculate common forecasting metrics."""

    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError("Prediction and actual shapes do not match.")

    if actual.size == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    error = actual - predicted

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )

    smape = 100 * np.mean(smape_terms)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "smape": float(smape),
    }
