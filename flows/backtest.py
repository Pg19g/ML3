"""Flow for backtesting models."""

import pandas as pd
import logging
from typing import Optional

from prefect import flow, task

from src.backtest import Backtester
from src.registry import ModelRegistry
from src.utils import (
    setup_logging,
    get_data_path,
    load_parquet,
    get_reports_path,
    save_json,
    save_parquet
)

logger = setup_logging(__name__)


@task(name="load_backtest_data")
def load_backtest_data() -> pd.DataFrame:
    """Load data for backtesting."""
    file_path = str(get_data_path("pit") / "labels.parquet")
    data = load_parquet(file_path)
    
    logger.info(f"Loaded backtest data with {len(data)} rows")
    return data


@task(name="generate_predictions")
def generate_predictions(
    data: pd.DataFrame,
    model_id: str
) -> pd.DataFrame:
    """Generate predictions using model."""
    registry = ModelRegistry()
    
    model_info = registry.get_model(model_id)
    if model_info is None:
        raise ValueError(f"Model {model_id} not found")
    
    metadata = model_info['metadata']
    feature_cols = metadata['feature_cols']
    
    # Filter feature_cols to only include columns that exist in data
    available_features = [col for col in feature_cols if col in data.columns]
    missing_features = [col for col in feature_cols if col not in data.columns]
    
    if missing_features:
        logger.warning(f"Skipping {len(missing_features)} missing feature columns: {missing_features[:5]}...")
    
    if not available_features:
        raise ValueError("No feature columns available in the dataset")
    
    logger.info(f"Using {len(available_features)} available features (out of {len(feature_cols)} requested)")
    
    # Filter to rows with valid features
    valid_mask = data[available_features].notna().all(axis=1)
    data_valid = data[valid_mask].copy()
    
    logger.info(f"Generating predictions for {len(data_valid)} rows")
    
    # Score
    predictions = registry.score_model(model_id, data_valid[available_features])
    data_valid['prediction'] = predictions
    
    return data_valid


@task(name="run_backtest")
def run_backtest(data: pd.DataFrame) -> dict:
    """Run backtest on predictions."""
    backtester = Backtester()
    
    results = backtester.run_backtest(data)
    
    logger.info(f"Backtest complete")
    return results


@task(name="save_backtest_results")
def save_backtest_results(results: dict, model_id: str) -> None:
    """Save backtest results."""
    reports_path = get_reports_path()
    
    # Save metrics
    metrics_file = reports_path / f"{model_id}_backtest_metrics.json"
    save_json(results['metrics'], str(metrics_file))
    
    # Save portfolio returns
    returns_file = reports_path / f"{model_id}_portfolio_returns.parquet"
    save_parquet(results['portfolio_returns'], str(returns_file), index=False)
    
    logger.info(f"Saved backtest results to {reports_path}")


@flow(name="backtest")
def backtest_flow(model_id: str) -> dict:
    """
    Run backtest for a trained model.
    
    Args:
        model_id: Model ID to backtest
        
    Returns:
        Backtest results dictionary
    """
    logger.info(f"Starting backtest flow for model {model_id}")
    
    # Load data
    data = load_backtest_data()
    
    if data.empty:
        logger.error("No backtest data available")
        return None
    
    # Generate predictions
    data_with_predictions = generate_predictions(data, model_id)
    
    # Run backtest
    results = run_backtest(data_with_predictions)
    
    # Save results
    save_backtest_results(results, model_id)
    
    logger.info(f"Backtest flow complete for model {model_id}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        backtest_flow(model_id)
    else:
        logger.error("Please provide model_id as argument")
