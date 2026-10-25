import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from energyflow.config import get_table_name
from energyflow.training.splits import (
    prepare_series,
    create_windows,
    split_windows,
)
from energyflow.inference.load_model import load_lstm_model
from energyflow.inference.predict import predict_demand_batch


def run_batch_inference(
    spark,
    config: dict,
    run_id: str,
    split_name: str = "test",
):
    """Replay valid historical forecasts and save predictions."""

    if split_name not in {"validation", "test"}:
        raise ValueError("Use validation or test split.")

    lookback = int(config["model"]["input_sequence_length"])
    horizon = int(config["model"]["forecast_horizon"])

    # Load the same source used in training.
    feature_table = get_table_name(config, "features")
    series = prepare_series(spark, feature_table)

    windows = create_windows(
        series,
        lookback=lookback,
        horizon=horizon,
        stride=24,
    )

    ratios = config["training"]

    splits = split_windows(
        windows,
        series,
        train_ratio=ratios["train_ratio"],
        validation_ratio=ratios["validation_ratio"],
    )

    selected = splits[split_name]
    starts = selected["start_idx"].to_numpy(dtype=np.int64)

    # All windows have already passed continuity checks.
    demand = series["demand_mw"].to_numpy(dtype=np.float32)

    model, scaler, model_config = load_lstm_model(run_id)

    if (
        model_config["lookback"] != lookback
        or model_config["forecast_horizon"] != horizon
    ):
        raise ValueError(
            "Saved model does not match project configuration."
        )

    print(f"Running {len(starts):,} forecast windows...")

    # Gather historical inputs before accessing targets.
    input_offsets = np.arange(-lookback, 0)
    input_indices = starts[:, None] + input_offsets[None, :]

    histories = demand[input_indices]

    forecasts = predict_demand_batch(
        model=model,
        histories=histories,
        scaler=scaler,
        lookback=lookback,
    )

    if forecasts.shape != (len(starts), horizon):
        raise ValueError("Unexpected prediction shape.")

    # Targets are retrieved after inference.
    target_offsets = np.arange(horizon)
    target_indices = starts[:, None] + target_offsets[None, :]

    actuals = demand[target_indices].astype(np.float64)

    # Flatten to one row per forecasted hour.
    n = len(starts)

    result = pd.DataFrame({
        "forecast_origin": np.repeat(
            series["datetime_local"].iloc[starts].to_numpy(),
            horizon,
        ),
        "forecast_timestamp": (
            series["datetime_local"]
            .iloc[target_indices.reshape(-1)]
            .to_numpy()
        ),
        "horizon_hour": np.tile(
            np.arange(1, horizon + 1), n
        ).astype("int32"),
        "segment_id": np.repeat(
            selected["segment_id"].to_numpy(dtype="int64"),
            horizon,
        ),
        "predicted_mw": forecasts.reshape(-1),
        "actual_mw": actuals.reshape(-1),
        "model_run_id": np.repeat(run_id, n * horizon),
        "data_split": np.repeat(split_name, n * horizon),
    })

    result["error_mw"] = (
        result["actual_mw"] - result["predicted_mw"]
    )

    result["absolute_error_mw"] = result["error_mw"].abs()

    # A replay is an offline backtest, not a live prediction.
    result["forecast_type"] = "historical_replay"

    prediction_table = get_table_name(config, "predictions")

    # Enforce explicit types for reliable Spark conversion.
    result["forecast_origin"] = pd.to_datetime(
        result["forecast_origin"]
    )
    result["forecast_timestamp"] = pd.to_datetime(
        result["forecast_timestamp"]
    )

    # Define a schema to avoid pandas/Spark type inference issues.
    from pyspark.sql.types import (
        StructType, StructField,
        TimestampNTZType, IntegerType,
        LongType, DoubleType, StringType,
    )

    schema = StructType([
        StructField("forecast_origin", TimestampNTZType(), False),
        StructField("forecast_timestamp", TimestampNTZType(), False),
        StructField("horizon_hour", IntegerType(), False),
        StructField("segment_id", LongType(), False),
        StructField("predicted_mw", DoubleType(), False),
        StructField("actual_mw", DoubleType(), False),
        StructField("model_run_id", StringType(), False),
        StructField("data_split", StringType(), False),
        StructField("error_mw", DoubleType(), False),
        StructField("absolute_error_mw", DoubleType(), False),
        StructField("forecast_type", StringType(), False),
    ])

    rows = [
        (
            row.forecast_origin.to_pydatetime(),
            row.forecast_timestamp.to_pydatetime(),
            int(row.horizon_hour),
            int(row.segment_id),
            float(row.predicted_mw),
            float(row.actual_mw),
            str(row.model_run_id),
            str(row.data_split),
            float(row.error_mw),
            float(row.absolute_error_mw),
            str(row.forecast_type),
        )
        for row in result.itertuples(index=False)
    ]

    spark_df = spark.createDataFrame(rows, schema=schema)

    # Re-running the same replay replaces the snapshot.
    # Later, a production predictions table will use an
    # incremental write strategy with unique forecast keys.
    (
        spark_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(prediction_table)
    )

    print(f"Saved {len(result):,} prediction rows.")
    print(f"Delta table: {prediction_table}")

    return result
