import pandas as pd
import numpy as np

from pyspark.sql import functions as F

from energyflow.config import get_table_name
from energyflow.training.evaluate import evaluate_forecast
from energyflow.monitoring.drift import create_drift_report


def load_predictions(spark, config):
    """Load saved predictions in chronological order."""

    table = get_table_name(config, "predictions")

    df = (
        spark.table(table)
        .select(
            "forecast_origin",
            "forecast_timestamp",
            "horizon_hour",
            "predicted_mw",
            "actual_mw",
            "model_run_id",
        )
        .orderBy("forecast_origin", "horizon_hour")
        .toPandas()
    )

    df["forecast_origin"] = pd.to_datetime(
        df["forecast_origin"]
    )

    if df.empty:
        raise ValueError("Prediction table is empty.")

    return df


def split_monitoring_periods(
    df: pd.DataFrame,
    reference_fraction=0.30,
    current_fraction=0.30,
):
    """
    Compare non-overlapping early and late replay periods.

    Middle observations are excluded to separate
    reference and current periods chronologically.
    """

    origins = (
        pd.Series(df["forecast_origin"].unique())
        .sort_values()
        .reset_index(drop=True)
    )

    n = len(origins)
    n_reference = int(n * reference_fraction)
    n_current = int(n * current_fraction)

    if n_reference < 2 or n_current < 2:
        raise ValueError("Not enough forecast windows.")

    if n_reference + n_current > n:
        raise ValueError("Monitoring periods overlap.")

    reference_origins = set(
        origins.iloc[:n_reference]
    )

    current_origins = set(
        origins.iloc[-n_current:]
    )

    reference = df[
        df["forecast_origin"].isin(reference_origins)
    ].copy()

    current = df[
        df["forecast_origin"].isin(current_origins)
    ].copy()

    return reference, current


def calculate_daily_metrics(df: pd.DataFrame):
    """Calculate forecast performance per forecast-origin day."""

    data = df.copy()
    data["forecast_date"] = (
        data["forecast_origin"].dt.date
    )

    rows = []

    for (run_id, date), group in data.groupby(
        ["model_run_id", "forecast_date"]
    ):
        metrics = evaluate_forecast(
            group["actual_mw"].to_numpy(),
            group["predicted_mw"].to_numpy(),
        )

        rows.append({
            "model_run_id": str(run_id),
            "forecast_date": date,
            "forecast_count": int(len(group)),
            "mae_mw": metrics["mae"],
            "rmse_mw": metrics["rmse"],
            "smape_pct": metrics["smape"],
        })

    return pd.DataFrame(rows)


def run_monitoring(spark, config):
    """Run drift and performance monitoring."""

    predictions = load_predictions(spark, config)

    reference, current = split_monitoring_periods(
        predictions
    )

    print("Reference rows:", len(reference))
    print("Current rows:", len(current))

    reference_metrics = evaluate_forecast(
        reference["actual_mw"],
        reference["predicted_mw"],
    )

    current_metrics = evaluate_forecast(
        current["actual_mw"],
        current["predicted_mw"],
    )

    print("Reference metrics:", reference_metrics)
    print("Current metrics:", current_metrics)

    # Evidently report
    drift_snapshot = create_drift_report(
        reference,
        current,
    )

    # Performance summaries for Streamlit
    daily_metrics = calculate_daily_metrics(
        predictions
    )

    output_table = get_table_name(
        config, "monitoring"
    )

    # Convert pandas dates to plain Python dates.
    daily_metrics["forecast_date"] = pd.to_datetime(
        daily_metrics["forecast_date"]
    ).dt.date

    spark_df = spark.createDataFrame(
        daily_metrics.to_dict("records")
    )

    (
        spark_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(output_table)
    )

    print("Monitoring table:", output_table)
    print("Monitoring rows:", len(daily_metrics))

    return {
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "drift_snapshot": drift_snapshot,
        "daily_metrics": daily_metrics,
    }
