import os

import pandas as pd
from databricks import sql
from databricks.sdk.core import Config


CATALOG = "workspace"
SCHEMA = "energyflow"

warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
http_path = f"/sql/1.0/warehouses/{warehouse_id}"

TABLES = {
    "history": f"{CATALOG}.{SCHEMA}.pjm_clean",
    "predictions": f"{CATALOG}.{SCHEMA}.pjm_predictions",
    "monitoring": f"{CATALOG}.{SCHEMA}.pjm_monitoring",
}


def get_connection():
    """Connect using the Databricks App's managed identity."""

    http_path = os.environ.get("DATABRICKS_SQL_HTTP_PATH")

    if not http_path:
        raise RuntimeError(
            "Missing DATABRICKS_SQL_HTTP_PATH. "
            "Configure it in Databricks Apps."
        )

    cfg = Config()

    return sql.connect(
        server_hostname=cfg.host.removeprefix("https://")
                                .removeprefix("http://"),
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


def query_dataframe(query: str, parameters=None) -> pd.DataFrame:
    """Execute an internal read-only query."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters=parameters)
            return cursor.fetchall_arrow().to_pandas()


def get_historical_demand(limit=2000):
    """Return recent historical observations."""

    return query_dataframe(
        f"""
        SELECT datetime_local, demand_mw
        FROM {TABLES['history']}
        WHERE is_duplicate_hour = false
        ORDER BY datetime_local DESC
        LIMIT {int(limit)}
        """
    ).sort_values("datetime_local")


def get_forecast_origins():
    """List available historical forecast origins."""

    return query_dataframe(
        f"""
        SELECT DISTINCT forecast_origin
        FROM {TABLES['predictions']}
        ORDER BY forecast_origin DESC
        """
    )


def get_forecast(origin):
    """Return the 24-hour forecast at the selected origin."""

    return query_dataframe(
        f"""
        SELECT
            forecast_timestamp,
            horizon_hour,
            predicted_mw,
            actual_mw,
            absolute_error_mw
        FROM {TABLES['predictions']}
        WHERE forecast_origin = ?
        ORDER BY horizon_hour
        """,
        parameters=[origin],
    )


def get_monitoring():
    """Return saved model-performance metrics."""

    return query_dataframe(
        f"""
        SELECT
            forecast_date,
            forecast_count,
            mae_mw,
            rmse_mw,
            smape_pct
        FROM {TABLES['monitoring']}
        ORDER BY forecast_date
        """
    )
