
"""Continuity-aware feature engineering for PJM demand."""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from energyflow.config import get_table_name


def create_features(df: DataFrame, config: dict) -> DataFrame:
    """
    Generate forecasting features without crossing
    ambiguous timestamps or missing-hour boundaries.

    No future demand values are used.
    """

    # 1. Exclude ambiguous clock hours from modeling.
    # Keep them unchanged in the clean Delta table.
    unique_hours = df.filter(
        F.col("datetime_local").isNotNull()
        & F.col("demand_mw").isNotNull()
        & (F.col("demand_mw") > 0)
        & (~F.col("is_duplicate_hour"))
    ).select("datetime_local", "demand_mw")

    # 2. Detect discontinuities between unique clock times.
    # An unpartitioned ordering is intentional here.
    time_order = Window.orderBy("datetime_local")

    ordered = unique_hours.withColumn(
        "previous_datetime",
        F.lag("datetime_local").over(time_order)
    )

    ordered = ordered.withColumn(
        "gap_hours",
        F.expr(
            "timestampdiff("
            "SECOND, previous_datetime, datetime_local"
            ") / 3600.0"
        )
    )

    # Break the sequence whenever the observed clock
    # moves by something other than exactly one hour.
    ordered = ordered.withColumn(
        "new_segment",
        F.when(
            F.col("previous_datetime").isNull()
            | (F.col("gap_hours") != 1),
            1
        ).otherwise(0)
    )

    # 3. Assign an ID to every continuous hourly segment.
    cumulative_window = time_order.rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
    )

    segmented = ordered.withColumn(
        "segment_id",
        F.sum("new_segment").over(cumulative_window)
    )

    # 4. Partition all historical calculations by segment.
    history_window = Window.partitionBy(
        "segment_id"
    ).orderBy("datetime_local")

    features = segmented

    # Calendar features
    features = (
        features
        .withColumn("hour", F.hour("datetime_local"))
        .withColumn(
            "day_of_week",
            F.weekday("datetime_local")
        )
        .withColumn("month", F.month("datetime_local"))
        .withColumn(
            "is_weekend",
            F.when(
                F.col("day_of_week") >= 5, 1
            ).otherwise(0)
        )
    )

    # Lag features
    for lag in config["features"]["lags"]:
        features = features.withColumn(
            f"lag_{lag}",
            F.lag("demand_mw", lag).over(history_window)
        )

    # Rolling statistics: prior observations only.
    # The current demand value is excluded.
    for window_size in config["features"]["rolling_windows"]:
        rolling_window = history_window.rowsBetween(
            -window_size, -1
        )

        features = features.withColumn(
            f"rolling_mean_{window_size}",
            F.avg("demand_mw").over(rolling_window)
        )

    # Number of earlier observations in this segment.
    features = features.withColumn(
        "history_count",
        F.row_number().over(history_window) - 1
    )

    required_history = config["model"][
        "input_sequence_length"
    ]

    features = features.withColumn(
        "has_full_history",
        F.col("history_count") >= required_history
    )

    return features.drop(
        "previous_datetime",
        "new_segment"
    )


def save_features(
    spark: SparkSession,
    config: dict
) -> int:
    """Build and overwrite the feature Delta table."""

    clean_table = get_table_name(config, "clean")
    feature_table = get_table_name(config, "features")

    clean_df = spark.table(clean_table)
    feature_df = create_features(clean_df, config)

    (
        feature_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(feature_table)
    )

    return spark.table(feature_table).count()
