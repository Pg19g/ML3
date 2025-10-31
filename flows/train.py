"""Flow for training models."""

import pandas as pd
import logging
from typing import Optional

from prefect import flow, task

from src.train import ModelTrainer
from src.features import FeatureEngineer
from src.labels import LabelGenerator
from src.utils import (
    setup_logging,
    get_data_path,
    load_parquet,
    get_reports_path,
    save_json
)

logger = setup_logging(__name__)


@task(name="load_training_data")
def load_training_data() -> pd.DataFrame:
    """Load data for training."""
    file_path = str(get_data_path("pit") / "labels.parquet")
    data = load_parquet(file_path)
    
    logger.info(f"Loaded training data with {len(data)} rows")
    return data


@task(name="train_model")
def train_model(
    data: pd.DataFrame,
    model_type: str = 'lightgbm'
) -> dict:
    """Train model with cross-validation."""
    trainer = ModelTrainer()
    engineer = FeatureEngineer()
    generator = LabelGenerator()
    
    # Get feature and label columns
    feature_cols = engineer.get_feature_names()
    label_col = generator.get_primary_label()
    
    logger.info(f"Training {model_type} with {len(feature_cols)} features")
    logger.info(f"Target label: {label_col}")
    
    # Cross-validate
    cv_results = trainer.cross_validate(
        data,
        feature_cols,
        label_col,
        model_type
    )
    
    return cv_results


@task(name="save_model")
def save_model(cv_results: dict) -> str:
    """Save best model to registry."""
    trainer = ModelTrainer()
    engineer = FeatureEngineer()
    generator = LabelGenerator()
    
    # Use the first model (or could select best)
    model = cv_results['models'][0]
    model_type = cv_results['model_type']
    
    feature_cols = engineer.get_feature_names()
    label_col = generator.get_primary_label()
    
    model_id = trainer.save_model(
        model,
        model_type,
        feature_cols,
        label_col,
        cv_results['avg_metrics']
    )
    
    logger.info(f"Saved model: {model_id}")
    return model_id


@task(name="save_results")
def save_results(cv_results: dict, model_id: str) -> None:
    """Save training results to reports."""
    reports_path = get_reports_path()
    
    # Save metrics
    metrics_file = reports_path / f"{model_id}_metrics.json"
    save_json(cv_results['avg_metrics'], str(metrics_file))
    
    # Save fold results
    fold_results_file = reports_path / f"{model_id}_folds.json"
    fold_data = {
        'model_id': model_id,
        'model_type': cv_results['model_type'],
        'n_folds': cv_results['n_folds'],
        'fold_results': cv_results['fold_results']
    }
    save_json(fold_data, str(fold_results_file))
    
    logger.info(f"Saved training results to {reports_path}")


@flow(name="train")
def train_flow(model_type: str = 'lightgbm') -> str:
    """
    Train model with cross-validation.
    
    Args:
        model_type: 'lightgbm' or 'xgboost'
        
    Returns:
        Model ID
    """
    logger.info(f"Starting training flow with {model_type}")
    
    # Load data
    data = load_training_data()
    
    if data.empty:
        logger.error("No training data available")
        return None
    
    # Train model
    cv_results = train_model(data, model_type)
    
    # Save model
    model_id = save_model(cv_results)
    
    # Save results
    save_results(cv_results, model_id)
    
    logger.info(f"Training flow complete. Model ID: {model_id}")
    return model_id


if __name__ == "__main__":
    train_flow()
