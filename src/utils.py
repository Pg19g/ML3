"""Utility functions for the ML3 pipeline."""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(json_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], json_path: str) -> None:
    """Save data to JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_data_path(subdir: str = "") -> Path:
    """Get data directory path."""
    root = get_project_root()
    data_path = root / "data" / subdir
    ensure_dir(data_path)
    return data_path


def get_config_path(config_name: str) -> Path:
    """Get configuration file path."""
    root = get_project_root()
    return root / "config" / config_name


def get_model_registry_path() -> Path:
    """Get model registry directory path."""
    root = get_project_root()
    registry_path = root / "models" / "registry"
    ensure_dir(registry_path)
    return registry_path


def get_reports_path() -> Path:
    """Get reports directory path."""
    root = get_project_root()
    reports_path = root / "reports"
    ensure_dir(reports_path)
    return reports_path


def load_parquet(path: str, **kwargs) -> pd.DataFrame:
    """Load parquet file with error handling."""
    try:
        return pd.read_parquet(path, **kwargs)
    except FileNotFoundError:
        logging.warning(f"File not found: {path}")
        return pd.DataFrame()


def save_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save DataFrame to parquet with error handling."""
    ensure_dir(Path(path).parent)
    df.to_parquet(path, **kwargs)


def generate_model_id() -> str:
    """Generate unique model ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"model_{timestamp}"


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    name: str = "DataFrame"
) -> None:
    """Validate DataFrame has required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def check_leakage(
    df: pd.DataFrame,
    date_col: str = "date",
    feature_cols: list = None,
    max_future_days: int = 0
) -> Dict[str, Any]:
    """
    Check for data leakage in features.
    
    Returns dict with leakage statistics.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [date_col, 'symbol']]
    
    results = {
        'has_leakage': False,
        'leakage_columns': [],
        'max_future_leak_days': 0
    }
    
    # This is a placeholder - actual implementation would check
    # if any feature values reference future data
    
    return results


def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """Winsorize series at specified percentiles."""
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower, upper)


def cross_sectional_standardize(
    df: pd.DataFrame,
    feature_cols: list,
    group_col: str = None
) -> pd.DataFrame:
    """
    Standardize features cross-sectionally (per time period).
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to standardize
        group_col: Optional grouping column (e.g., 'sector')
    """
    result = df.copy()
    
    # Filter feature_cols to only those that exist in the DataFrame
    existing_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Skipping {len(missing_cols)} missing columns in standardization: {missing_cols[:5]}...")
    
    if not existing_cols:
        logger.warning("No feature columns to standardize")
        return result
    
    if group_col and group_col in df.columns:
        # Standardize within groups
        for col in existing_cols:
            result[col] = df.groupby(['date', group_col])[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    else:
        # Standardize across entire cross-section
        for col in existing_cols:
            result[col] = df.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    return result


def get_env_variable(name: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} not set")
    return value
