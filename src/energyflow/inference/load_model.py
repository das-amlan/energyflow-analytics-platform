import json
from pathlib import Path

import mlflow
import torch

from energyflow.training.lstm_model import EnergyLSTM


def load_lstm_model(run_id: str):
    """Return the trained model, scaler parameters, and configuration."""

    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="lstm_artifacts",
    )
    artifact_dir = Path(artifact_path)

    model_config = json.loads(
        (artifact_dir / "model_config.json").read_text()
    )
    scaler_config = json.loads(
        (artifact_dir / "scaler.json").read_text()
    )

    model = EnergyLSTM(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        forecast_horizon=model_config["forecast_horizon"],
    )

    state_dict = torch.load(
        artifact_dir / "lstm_weights.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    if scaler_config["std"] <= 0:
        raise ValueError("Invalid scaler standard deviation.")

    return model, scaler_config, model_config
