import numpy as np
import pytest
import torch

from energyflow.training.lstm_model import EnergyLSTM
from energyflow.inference.predict import (
    predict_demand,
    predict_demand_batch,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return EnergyLSTM(
        input_size=1,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        forecast_horizon=24,
    ).eval()


@pytest.fixture
def scaler():
    return {"mean": 30000.0, "std": 6000.0}


def test_single_forecast_shape(model, scaler):
    history = np.full(168, 30000.0)

    forecast = predict_demand(model, history, scaler)

    assert forecast.shape == (24,)
    assert np.isfinite(forecast).all()


def test_batch_forecast_shape(model, scaler):
    histories = np.full((5, 168), 30000.0)

    forecasts = predict_demand_batch(
        model, histories, scaler, batch_size=2
    )

    assert forecasts.shape == (5, 24)


def test_single_matches_batch(model, scaler):
    history = np.linspace(25000, 35000, 168)

    single = predict_demand(model, history, scaler)
    batch = predict_demand_batch(
        model, history.reshape(1, -1), scaler
    )[0]

    np.testing.assert_allclose(
        single, batch, rtol=1e-5, atol=1e-2
    )


def test_invalid_history(model, scaler):
    with pytest.raises(ValueError):
        predict_demand(model, np.ones(100), scaler)
