import argparse

from pyspark.sql import SparkSession

from energyflow.config import load_config, get_table_name
from energyflow.ingestion.ingest_pjm import ingest_pjm
from energyflow.processing.validate import validate_pjm
from energyflow.processing.clean import save_clean_data
from energyflow.processing.features import save_features


def run_pipeline(spark: SparkSession, config: dict) -> None:
    """Execute the complete data preparation workflow."""

    print("Step 1: Ingesting PJM data")
    ingested = ingest_pjm(spark, config)
    print(f"Ingested {ingested:,} rows")

    print("Step 2: Validating raw data")
    raw_df = spark.table(get_table_name(config, "raw"))
    report = validate_pjm(raw_df)

    if not report["passed_basic_checks"]:
        raise ValueError(f"Data validation failed: {report}")

    print(f"Validation passed: {report}")

    print("Step 3: Cleaning data")
    cleaned = save_clean_data(spark, config)
    print(f"Cleaned {cleaned:,} rows")

    print("Step 4: Engineering features")
    features = save_features(spark, config)
    print(f"Generated {features:,} feature rows")

    print("EnergyFlow pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the project YAML configuration",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spark = SparkSession.builder.getOrCreate()

    run_pipeline(spark, config)


if __name__ == "__main__":
    main()
