"""Unit tests for project configuration."""

import pytest

from energyflow.config import (
    load_config,
    get_table_name,
)


def test_config_loads():
    config = load_config()

    assert config["project"]["name"] == (
        "energyflow-analytics-platform"
    )

    assert config["data"]["region"] == "PJME"
    assert config["data"]["target_column"] == "PJME_MW"


def test_forecasting_parameters():
    config = load_config()

    assert config["model"]["input_sequence_length"] == 168
    assert config["model"]["forecast_horizon"] == 24


def test_table_name():
    config = load_config()

    table = get_table_name(config, "raw")

    assert table == "workspace.energyflow.pjm_raw"


def test_missing_config():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")
