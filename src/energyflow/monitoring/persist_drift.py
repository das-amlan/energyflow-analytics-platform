from datetime import datetime, timezone

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    TimestampType,
)

from energyflow.config import get_table_name


def save_drift_report(spark, config, snapshot):
    """Save an Evidently snapshot for the Streamlit app."""

    table_name = get_table_name(config, "drift_reports")

    report_html = snapshot.get_html_str(as_iframe=False)
    report_json = snapshot.json()

    schema = StructType([
        StructField("generated_at", TimestampType(), False),
        StructField("report_html", StringType(), False),
        StructField("report_json", StringType(), False),
    ])

    rows = [(
        datetime.now(timezone.utc),
        report_html,
        report_json,
    )]

    df = spark.createDataFrame(rows, schema)

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )

    print(f"Evidently report saved to {table_name}")
    