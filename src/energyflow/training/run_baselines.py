import argparse

import mlflow
import numpy as np
import pandas as pd

from energyflow.config import load_config, get_table_name
from energyflow.training.splits import (
    prepare_series,
    create_windows,
    split_windows,
)
from energyflow.training.baselines import (
    predict_daily_naive,
    predict_weekly_naive,
    get_actual_values,
)
from energyflow.training.evaluate import evaluate_forecast


def run_baselines(spark, config: dict) -> pd.DataFrame:
    """Evaluate and track daily and weekly naive forecasts."""

    feature_table = get_table_name(config, "features")

    lookback = config["model"]["input_sequence_length"]
    horizon = config["model"]["forecast_horizon"]

    print("Loading feature data...")
    df = prepare_series(spark, feature_table)

    print("Generating forecast windows...")
    windows = create_windows(
        df,
        lookback=lookback,
        horizon=horizon,
        stride=24,
    )

    training_cfg = config["training"]

    splits = split_windows(
        windows,
        df,
        train_ratio=training_cfg["train_ratio"],
        validation_ratio=training_cfg["validation_ratio"],
    )

    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df):,} windows")

    demand = df["demand_mw"].to_numpy()

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    models = {
        "daily_naive": predict_daily_naive,
        "weekly_naive": predict_weekly_naive,
    }

    results = []

    for model_name, prediction_function in models.items():
        print(f"\nEvaluating {model_name}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                "model_type": model_name,
                "dataset": "PJME",
                "lookback": lookback,
                "horizon": horizon,
                "stride": 24,
                "train_ratio": training_cfg["train_ratio"],
                "validation_ratio": training_cfg["validation_ratio"],
            })

            for split_name in ["validation", "test"]:
                split_df = splits[split_name]

                indices = split_df["start_idx"].to_numpy(
                    dtype=np.int64
                )

                actual = get_actual_values(
                    demand, indices, horizon
                )

                predicted = prediction_function(
                    demand, indices, horizon
                )

                metrics = evaluate_forecast(actual, predicted)

                for metric, value in metrics.items():
                    mlflow.log_metric(
                        f"{split_name}_{metric}",
                        value,
                    )

                results.append({
                    "model": model_name,
                    "split": split_name,
                    "windows": len(split_df),
                    **metrics,
                })

                print(
                    f"{split_name}: "
                    f"MAE={metrics['mae']:.2f}, "
                    f"RMSE={metrics['rmse']:.2f}, "
                    f"sMAPE={metrics['smape']:.2f}%"
                )

    return pd.DataFrame(results)


def main():
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    spark = SparkSession.builder.getOrCreate()

    print(run_baselines(spark, config).to_string(index=False))


if __name__ == "__main__":
    main()
