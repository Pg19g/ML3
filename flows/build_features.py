"""Flow for building features."""

import pandas as pd
import logging

from prefect import flow, task

from src.features import FeatureEngineer
from src.utils import (
    setup_logging,
    get_data_path,
    load_parquet,
    save_parquet
)

logger = setup_logging(__name__)


@task(name="load_pit_panel")
def load_pit_panel() -> pd.DataFrame:
    """Load PIT panel."""
    file_path = str(get_data_path("pit") / "daily_panel.parquet")
    pit_panel = load_parquet(file_path)
    
    logger.info(f"Loaded PIT panel with {len(pit_panel)} rows")
    return pit_panel


@task(name="build_features")
def build_features(pit_panel: pd.DataFrame) -> pd.DataFrame:
    """Build features from PIT panel."""
    engineer = FeatureEngineer()
    
    features_df = engineer.build_features(pit_panel)
    
    feature_names = engineer.get_feature_names()
    logger.info(f"Built {len(feature_names)} features")
    
    return features_df


@task(name="save_features")
def save_features(features_df: pd.DataFrame, file_path: str) -> None:
    """Save features to parquet."""
    save_parquet(features_df, file_path, index=False)
    logger.info(f"Saved features with {len(features_df)} rows to {file_path}")


@flow(name="build-features")
def build_features_flow() -> None:
    """Build features from PIT panel."""
    logger.info("Starting feature build flow")
    
    # Load PIT panel
    pit_panel = load_pit_panel()
    
    if pit_panel.empty:
        logger.error("No PIT panel data available")
        return
    
    # Build features
    features_df = build_features(pit_panel)
    
    # Save
    file_path = str(get_data_path("pit") / "features.parquet")
    save_features(features_df, file_path)
    
    logger.info("Feature build flow complete")


if __name__ == "__main__":
    build_features_flow()
