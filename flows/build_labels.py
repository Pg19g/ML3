"""Flow for building labels."""

import pandas as pd
import logging

from prefect import flow, task

from src.labels import LabelGenerator
from src.utils import (
    setup_logging,
    get_data_path,
    load_parquet,
    save_parquet
)

logger = setup_logging(__name__)


@task(name="load_features")
def load_features() -> pd.DataFrame:
    """Load features data."""
    file_path = str(get_data_path("pit") / "features.parquet")
    features_df = load_parquet(file_path)
    
    logger.info(f"Loaded features with {len(features_df)} rows")
    return features_df


@task(name="build_labels")
def build_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build labels from features data."""
    generator = LabelGenerator()
    
    labels_df = generator.build_labels(features_df)
    
    label_names = generator.get_label_names()
    logger.info(f"Built {len(label_names)} labels")
    
    return labels_df


@task(name="save_labels")
def save_labels(labels_df: pd.DataFrame, file_path: str) -> None:
    """Save labels to parquet."""
    save_parquet(labels_df, file_path, index=False)
    logger.info(f"Saved labels with {len(labels_df)} rows to {file_path}")


@flow(name="build-labels")
def build_labels_flow() -> None:
    """Build labels from features data."""
    logger.info("Starting label build flow")
    
    # Load features
    features_df = load_features()
    
    if features_df.empty:
        logger.error("No features data available")
        return
    
    # Build labels
    labels_df = build_labels(features_df)
    
    # Save
    file_path = str(get_data_path("pit") / "labels.parquet")
    save_labels(labels_df, file_path)
    
    logger.info("Label build flow complete")


if __name__ == "__main__":
    build_labels_flow()
