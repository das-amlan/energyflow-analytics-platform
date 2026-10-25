import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


DRIFT_COLUMNS = ["predicted_mw", "actual_mw"]


def create_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Compare distributions in two historical periods.

    predicted_mw: model-output distribution
    actual_mw: observed-demand distribution
    """

    if reference.empty or current.empty:
        raise ValueError("Monitoring datasets cannot be empty.")

    reference_df = reference[DRIFT_COLUMNS].copy()
    current_df = current[DRIFT_COLUMNS].copy()

    for frame in (reference_df, current_df):
        if frame.isna().any().any():
            raise ValueError("Drift inputs contain null values.")

    definition = DataDefinition(
        numerical_columns=DRIFT_COLUMNS
    )

    ref_dataset = Dataset.from_pandas(
        reference_df,
        data_definition=definition,
    )

    current_dataset = Dataset.from_pandas(
        current_df,
        data_definition=definition,
    )

    report = Report([
        DataDriftPreset(columns=DRIFT_COLUMNS)
    ])

    snapshot = report.run(
        current_dataset,
        ref_dataset,
    )

    return snapshot
