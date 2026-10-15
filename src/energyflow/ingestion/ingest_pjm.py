import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from energyflow.config import load_config, get_table_name


def ingest_pjm(spark: SparkSession, config: dict) -> int:
    """Ingest the PJM CSV into a raw Delta table."""

    catalog = config["databricks"]["catalog"]
    schema = config["databricks"]["schema"]

    source = config["data"].get("source_file")

    if not source:
        raise ValueError(
            "Missing 'data.source_file' in config/project.yaml. "
            "Please specify the Unity Catalog volume and filename."
        )
    volume = source["volume"]
    filename = source["filename"]

    source_path = (
        f"/Volumes/{catalog}/{schema}/{volume}/{filename}"
    )

    target_table = get_table_name(config, "raw")

    df = spark.read.option("header", True).csv(source_path)

    expected_columns = {
        config["data"]["timestamp_column"],
        config["data"]["target_column"],
    }

    missing = expected_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    timestamp_col = config["data"]["timestamp_column"]
    target_col = config["data"]["target_column"]

    df = df.select(
        F.col(timestamp_col).cast("string").alias("datetime_raw"),
        F.col(target_col).cast("double").alias("demand_mw"),
    )

    df = df.withColumn(
        "ingested_at",
        F.current_timestamp()
    )

    # Full historical snapshot ingestion.
    # Overwrite makes repeated runs idempotent.
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(target_table)
    )

    return df.count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to project.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    spark = SparkSession.builder.getOrCreate()

    count = ingest_pjm(spark, config)

    print(f"Successfully ingested {count:,} rows.")


if __name__ == "__main__":
    main()
