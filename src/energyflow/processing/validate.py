"""Data quality validation for PJM energy demand."""

from pyspark.sql import DataFrame, functions as F


def validate_pjm(df: DataFrame) -> dict:
    """Return data quality metrics without modifying the dataset."""

    required_columns = {"datetime_raw", "demand_mw"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    parsed = df.withColumn(
        "_parsed_time",
        F.expr("try_cast(datetime_raw AS TIMESTAMP_NTZ)")
    )

    stats = parsed.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.col("datetime_raw").isNull().cast("int"))
            .alias("missing_timestamps"),
        F.sum(F.col("demand_mw").isNull().cast("int"))
            .alias("missing_demand"),
        F.sum(F.col("_parsed_time").isNull().cast("int"))
            .alias("invalid_timestamps"),
        F.sum((F.col("demand_mw") <= 0).cast("int"))
            .alias("non_positive_demand"),
        F.min("_parsed_time").cast("string")
            .alias("earliest_timestamp"),
        F.max("_parsed_time").cast("string")
            .alias("latest_timestamp"),
    ).first().asDict()

    duplicates = (
        parsed.groupBy("datetime_raw")
        .count()
        .filter(F.col("count") > 1)
    )

    stats["duplicate_timestamp_groups"] = duplicates.count()

    duplicate_excess = duplicates.select(
        F.sum(F.col("count") - 1).alias("extra")
    ).first()["extra"]

    stats["duplicate_excess_rows"] = int(
        duplicate_excess or 0
    )

    # Distinct-clock continuity diagnostic.
    # A complete hourly local-clock range is only a diagnostic,
    # not proof of physically consecutive hours at DST changes.
    valid_times = (
        parsed.filter(F.col("_parsed_time").isNotNull())
        .select("_parsed_time")
        .distinct()
    )

    ordered = valid_times.orderBy("_parsed_time")

    from pyspark.sql.window import Window

    window = Window.orderBy("_parsed_time")

    gaps = ordered.withColumn(
        "_previous",
        F.lag("_parsed_time").over(window)
    ).filter(F.col("_previous").isNotNull())

    gap_stats = gaps.select(
        (
            F.unix_timestamp("_parsed_time")
            - F.unix_timestamp("_previous")
        ).alias("gap_seconds")
    ).agg(
        F.sum(
            (F.col("gap_seconds") > 3600).cast("int")
        ).alias("gaps_over_one_hour")
    ).first()

    stats["gaps_over_one_hour"] = int(
        gap_stats["gaps_over_one_hour"] or 0
    )

    stats["passed_basic_checks"] = all([
        stats["missing_timestamps"] == 0,
        stats["missing_demand"] == 0,
        stats["invalid_timestamps"] == 0,
        stats["non_positive_demand"] == 0,
    ])

    return stats
