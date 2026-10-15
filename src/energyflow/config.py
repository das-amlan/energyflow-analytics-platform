"""Configuration utilities for EnergyFlow Analytics Platform."""

import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "project.yaml"
)


def load_config(config_path=None):
    """Load and validate the project YAML configuration."""

    path = Path(
        config_path
        or os.environ.get("ENERGYFLOW_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    required_sections = [
        "project",
        "databricks",
        "data",
        "features",
        "model",
        "training",
        "mlflow",
        "monitoring",
        "app",
    ]

    missing = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing:
        raise ValueError(
            f"Missing configuration sections: {missing}"
        )

    return config


def get_table_name(config, table_key):
    """Return a fully qualified Unity Catalog table name."""

    catalog = config["databricks"]["catalog"]
    schema = config["databricks"]["schema"]
    table = config["data"]["tables"][table_key]

    return f"{catalog}.{schema}.{table}"
