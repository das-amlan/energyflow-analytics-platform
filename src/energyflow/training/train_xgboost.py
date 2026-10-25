import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from energyflow.config import load_config, get_table_name
from energyflow.training.splits import (
    prepare_series,
    create_windows,
    split_windows,
)
from energyflow.training.evaluate import evaluate_forecast


def build_training_matrix(df, windows, horizon=24, lookback=168):
    """
    Build one training row per future forecast hour.

    All demand features refer to values available before
    the forecast origin. Future calendar features are known.
    """
    demand = df["demand_mw"].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(df["datetime_local"])

    records = []
    targets = []

    for start in windows["start_idx"].to_numpy(dtype=int):
        history = demand[start - lookback:start]

        for h in range(horizon):
            forecast_time = timestamps.iloc[start + h]

            records.append({
                "horizon": h + 1,
                "hour": forecast_time.hour,
                "day_of_week": forecast_time.dayofweek,
                "month": forecast_time.month,
                "is_weekend": int(forecast_time.dayofweek >= 5),
                "lag_24": demand[start - 24 + h],
                "lag_48": demand[start - 48 + h],
                "lag_168": demand[start - 168 + h],
                "history_mean_24": history[-24:].mean(),
                "history_mean_168": history.mean(),
                "history_std_24": history[-24:].std(),
            })

            targets.append(demand[start + h])

    X = pd.DataFrame(records).astype("float64")
    y = np.asarray(targets, dtype="float64")

    return X, y


def train_xgboost(spark, config):
    table = get_table_name(config, "features")
    df = prepare_series(spark, table)

    lookback = config["model"]["input_sequence_length"]
    horizon = config["model"]["forecast_horizon"]

    if lookback < 168 or horizon > 24:
        raise ValueError(
            "This feature design requires lookback >= 168 "
            "and forecast horizon <= 24."
        )

    windows = create_windows(
        df,
        lookback=lookback,
        horizon=horizon,
        stride=24,
    )

    splits = split_windows(
        windows,
        df,
        train_ratio=config["training"]["train_ratio"],
        validation_ratio=config["training"]["validation_ratio"],
    )

    X_train, y_train = build_training_matrix(
        df, splits["train"], horizon, lookback=lookback
    )
    X_val, y_val = build_training_matrix(
        df, splits["validation"], horizon, lookback=lookback
    )
    X_test, y_test = build_training_matrix(
        df, splits["test"], horizon, lookback=lookback
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=42,
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="xgboost_24h_v1"):
        mlflow.log_params({
            "model": "xgboost",
            "lookback": lookback,
            "forecast_horizon": horizon,
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "train_windows": len(splits["train"]),
        })

        model.fit(X_train, y_train)

        results = {}

        for name, X, y in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            predictions = model.predict(X)
            metrics = evaluate_forecast(y, predictions)

            mlflow.log_metrics({
                f"{name}_{key}": value
                for key, value in metrics.items()
            })

            results[name] = metrics
            print(f"{name}: {metrics}")

        mlflow.xgboost.log_model(
            model,
            name="xgboost_forecaster",
            input_example=X_train.head(5),
        )

    return model, results
