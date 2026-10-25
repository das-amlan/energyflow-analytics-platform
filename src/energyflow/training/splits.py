import numpy as np
import pandas as pd


def prepare_series(spark, table_name: str) -> pd.DataFrame:
    """Read the feature table into a small, ordered pandas DataFrame."""

    df = (
        spark.table(table_name)
        .select("datetime_local", "demand_mw", "segment_id")
        .orderBy("datetime_local")
        .toPandas()
    )

    df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    df["demand_mw"] = df["demand_mw"].astype("float64")

    if df[["datetime_local", "demand_mw", "segment_id"]].isna().any().any():
        raise ValueError("Missing values in the forecasting source.")

    return df.reset_index(drop=True)


def create_windows(
    df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 24,
    stride: int = 24,
):
    """
    Create valid forecast-window indices.

    A window contains:
      - lookback historical demand observations
      - horizon future demand observations

    Input and target must belong to the same segment.
    """

    records = []

    for segment_id, group in df.groupby("segment_id", sort=False):
        indices = group.index.to_numpy()

        if len(indices) < lookback + horizon:
            continue

        for offset in range(
            lookback,
            len(indices) - horizon + 1,
            stride,
        ):
            start = int(indices[offset])

            records.append({
                "segment_id": int(segment_id),
                "start_idx": start,
                "forecast_start": df.loc[start, "datetime_local"],
                "forecast_end": df.loc[
                    start + horizon - 1, "datetime_local"
                ],
            })

    if not records:
        raise ValueError("No valid forecasting windows were generated.")

    return pd.DataFrame(records).sort_values(
        "forecast_start"
    ).reset_index(drop=True)


def split_windows(
    windows: pd.DataFrame,
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split chronologically.

    Targets may not cross split boundaries.
    Historical inputs may precede a boundary because
    they would have been known at forecast time.
    """

    if not (0 < train_ratio < 1):
        raise ValueError("Invalid train_ratio.")

    if not (0 < validation_ratio < 1):
        raise ValueError("Invalid validation_ratio.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to < 1.")

    n = len(df)

    train_boundary = df.iloc[int(n * train_ratio)]["datetime_local"]
    val_boundary = df.iloc[
        int(n * (train_ratio + validation_ratio))
    ]["datetime_local"]

    train = windows[
        windows["forecast_end"] < train_boundary
    ].copy()

    validation = windows[
        (windows["forecast_start"] >= train_boundary)
        & (windows["forecast_end"] < val_boundary)
    ].copy()

    test = windows[
        windows["forecast_start"] >= val_boundary
    ].copy()

    if any(len(split) == 0 for split in [train, validation, test]):
        raise ValueError("At least one chronological split is empty.")

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
