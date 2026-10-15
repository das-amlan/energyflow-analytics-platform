"""Clean PJM demand observations while preserving timestamp anomalies."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from energyflow.config import get_table_name


def clean_pjm(df: DataFrame) -> DataFrame:
    """Standardize demand data without dropping ambiguous hours."""

    timestamp = F.expr(
        "try_cast(datetime_raw AS TIMESTAMP_NTZ)"
    )

    cleaned = df.select(
        F.col("datetime_raw"),
        timestamp.alias("datetime_local"),
        F.col("demand_mw").cast("double").alias("demand_mw"),
    )

    # Identify repeated local-clock timestamps.
    counts = cleaned.groupBy("datetime_local").agg(
        F.count("*").alias("timestamp_count")
    )

    cleaned = cleaned.join(
        counts,
        on="datetime_local",
        how="left",
    )

    cleaned = cleaned.withColumn(
        "is_duplicate_hour",
        F.col("timestamp_count") > 1,
    )

    # Retain only valid numeric records.
    # No deduplication or interpolation is performed.
    cleaned = cleaned.filter(
        F.col("datetime_local").isNotNull()
        & F.col("demand_mw").isNotNull()
        & (F.col("demand_mw") > 0)
    )

    return cleaned.select(
        "datetime_raw",
        "datetime_local",
        "demand_mw",
        "is_duplicate_hour",
    )


def save_clean_data(
    spark: SparkSession,
    config: dict,
) -> int:
    """Create a cleaned Delta table from the raw table."""

    raw_table = get_table_name(config, "raw")
    clean_table = get_table_name(config, "clean")

    raw_df = spark.table(raw_table)
    clean_df = clean_pjm(raw_df)

    (
        clean_df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(clean_table)
    )

    return spark.table(clean_table).count()
