import copy
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from energyflow.config import get_table_name
from energyflow.training.splits import (
    prepare_series,
    create_windows,
    split_windows,
)
from energyflow.training.evaluate import evaluate_forecast
from energyflow.training.lstm_model import EnergyLSTM
from energyflow.training.lstm_dataset import (
    DemandScaler,
    ForecastDataset,
)


def predict_dataset(model, loader, scaler, device):
    """Return actual and predicted demand in MW."""

    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x).cpu().numpy()

            predictions.append(
                scaler.inverse_transform(output)
            )
            actuals.append(
                scaler.inverse_transform(y.numpy())
            )

    return (
        np.concatenate(actuals),
        np.concatenate(predictions),
    )


def train_lstm(spark, config):
    """Train LSTM and log evaluation results to MLflow."""

    seed = config["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Keep CPU usage reasonable in Databricks.
    torch.set_num_threads(2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    # ---------------------------------------
    # 1. Load demand and forecasting windows
    # ---------------------------------------

    table = get_table_name(config, "features")
    df = prepare_series(spark, table)

    lookback = config["model"]["input_sequence_length"]
    horizon = config["model"]["forecast_horizon"]

    windows = create_windows(
        df,
        lookback=lookback,
        horizon=horizon,
        stride=24,
    )

    ratios = config["training"]

    splits = split_windows(
        windows,
        df,
        train_ratio=ratios["train_ratio"],
        validation_ratio=ratios["validation_ratio"],
    )

    for name, split in splits.items():
        print(f"{name}: {len(split)} windows")

    demand = df["demand_mw"].to_numpy(
        dtype=np.float32
    )

    # ---------------------------------------
    # 2. Fit scaler using training period only
    # ---------------------------------------

    first_validation_time = splits[
        "validation"
    ]["forecast_start"].min()

    # Include only observations before the
    # first validation forecast origin.
    train_mask = (
        df["datetime_local"] < first_validation_time
    ).to_numpy()

    scaler = DemandScaler().fit(demand[train_mask])

    # ---------------------------------------
    # 3. Create PyTorch datasets
    # ---------------------------------------

    cfg = config["model"]["lstm"]

    loaders = {}

    for split_name, split_df in splits.items():
        dataset = ForecastDataset(
            demand=demand,
            windows=split_df,
            scaler=scaler,
            lookback=lookback,
            horizon=horizon,
        )

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=(split_name == "train"),
            num_workers=0,
        )

    # ---------------------------------------
    # 4. Initialize model
    # ---------------------------------------

    model = EnergyLSTM(
        input_size=1,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        forecast_horizon=horizon,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )

    criterion = nn.HuberLoss()

    max_epochs = cfg["epochs"]
    patience = 6

    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    # ---------------------------------------
    # 5. Train and track with MLflow
    # ---------------------------------------

    with mlflow.start_run(run_name="lstm_24h_v1") as run:

        mlflow.log_params({
            "model_type": "LSTM",
            "lookback": lookback,
            "horizon": horizon,
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "train_windows": len(splits["train"]),
            "validation_windows": len(splits["validation"]),
            "test_windows": len(splits["test"]),
        })

        for epoch in range(1, max_epochs + 1):

            model.train()
            total_loss = 0.0
            total_samples = 0

            for x, y in loaders["train"]:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                output = model(x)
                loss = criterion(output, y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            train_loss = total_loss / total_samples

            # Validation uses original MW units.
            val_actual, val_pred = predict_dataset(
                model,
                loaders["validation"],
                scaler,
                device,
            )

            val_metrics = evaluate_forecast(
                val_actual,
                val_pred,
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "validation_mae",
                val_metrics["mae"],
                step=epoch,
            )

            mlflow.log_metric(
                "validation_rmse",
                val_metrics["rmse"],
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{max_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val MAE: {val_metrics['mae']:.2f} MW"
            )

            # Early stopping on validation MAE
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        # -----------------------------------
        # 6. Restore best model
        # -----------------------------------

        model.load_state_dict(best_state)
        model.eval()

        results = {}

        for split_name in ["validation", "test"]:

            actual, predicted = predict_dataset(
                model,
                loaders[split_name],
                scaler,
                device,
            )

            metrics = evaluate_forecast(
                actual,
                predicted,
            )

            results[split_name] = metrics

            mlflow.log_metrics({
                f"best_{split_name}_{key}": value
                for key, value in metrics.items()
            })

            print(f"{split_name}: {metrics}")

        mlflow.log_metric("best_epoch", best_epoch)

        # -----------------------------------
        # 7. Save reproducible model artifacts
        # -----------------------------------

        model_config = {
            "input_size": 1,
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "forecast_horizon": horizon,
            "lookback": lookback,
        }

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)

            torch.save(
                model.cpu().state_dict(),
                artifact_dir / "lstm_weights.pth",
            )

            (artifact_dir / "scaler.json").write_text(
                json.dumps({
                    "mean": scaler.mean,
                    "std": scaler.std,
                }, indent=2)
            )

            (artifact_dir / "model_config.json").write_text(
                json.dumps(model_config, indent=2)
            )

            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path="lstm_artifacts",
            )

        print("MLflow run ID:", run.info.run_id)
        print("Best epoch:", best_epoch)

    return model, results
