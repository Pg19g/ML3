"""Label generation for ML models."""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from src.utils import setup_logging, load_config, get_config_path

logger = setup_logging(__name__)


class LabelGenerator:
    """Generate labels for ML models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize label generator.
        
        Args:
            config_path: Path to labels.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("labels.yaml"))
        
        self.config = load_config(config_path)
        self.labels = self.config.get('labels', [])
        self.classification = self.config.get('classification', [])
    
    def compute_forward_returns(
        self,
        df: pd.DataFrame,
        horizon: int,
        price_col: str = 'AdjClose'
    ) -> pd.Series:
        """
        Compute forward returns for given horizon.
        
        Forward return = (Price[t+horizon] / Price[t]) - 1
        
        Args:
            df: DataFrame with price data (must have symbol, date, price_col)
            horizon: Number of periods forward
            price_col: Column name for price
            
        Returns:
            Series with forward returns
        """
        # Shift prices backward (negative shift = future values)
        future_price = df.groupby('symbol')[price_col].shift(-horizon)
        current_price = df[price_col]
        
        forward_return = (future_price / current_price) - 1
        return forward_return
    
    def compute_binary_direction(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: float = 0.0,
        price_col: str = 'AdjClose'
    ) -> pd.Series:
        """
        Compute binary direction label (up/down).
        
        Args:
            df: DataFrame with price data
            horizon: Number of periods forward
            threshold: Threshold for classification (default 0.0)
            price_col: Column name for price
            
        Returns:
            Series with binary labels (1 = up, 0 = down)
        """
        forward_return = self.compute_forward_returns(df, horizon, price_col)
        binary_label = (forward_return > threshold).astype(int)
        return binary_label
    
    def build_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build all labels from data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with labels added
        """
        result = df.copy()
        result = result.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info("Building labels...")
        
        # Regression labels (forward returns)
        for label_config in self.labels:
            if not label_config.get('enabled', True):
                continue
            
            label_name = label_config['name']
            label_type = label_config['type']
            horizon = label_config['horizon']
            
            if label_type == 'forward_return':
                result[label_name] = self.compute_forward_returns(result, horizon)
                logger.info(f"Computed label: {label_name} (horizon={horizon})")
        
        # Classification labels
        for label_config in self.classification:
            if not label_config.get('enabled', True):
                continue
            
            label_name = label_config['name']
            label_type = label_config['type']
            horizon = label_config['horizon']
            threshold = label_config.get('threshold', 0.0)
            
            if label_type == 'binary_direction':
                result[label_name] = self.compute_binary_direction(
                    result, horizon, threshold
                )
                logger.info(f"Computed label: {label_name} (horizon={horizon})")
        
        # Validate labels
        self._validate_labels(result)
        
        return result
    
    def _validate_labels(self, df: pd.DataFrame) -> None:
        """Validate labels for quality checks."""
        validation = self.config.get('validation', {})
        min_obs = validation.get('min_observations', 100)
        max_missing = validation.get('max_missing_pct', 0.5)
        
        label_cols = self.get_label_names()
        
        for col in label_cols:
            if col not in df.columns:
                continue
            
            # Check number of observations
            n_valid = df[col].notna().sum()
            if n_valid < min_obs:
                logger.warning(
                    f"Label {col} has only {n_valid} valid observations "
                    f"(minimum: {min_obs})"
                )
            
            # Check missing percentage
            missing_pct = df[col].isna().mean()
            if missing_pct > max_missing:
                logger.warning(
                    f"Label {col} has {missing_pct:.1%} missing values "
                    f"(maximum: {max_missing:.1%})"
                )
    
    def get_label_names(self) -> List[str]:
        """Get list of all enabled label names."""
        regression_labels = [
            l['name'] for l in self.labels if l.get('enabled', True)
        ]
        classification_labels = [
            l['name'] for l in self.classification if l.get('enabled', True)
        ]
        return regression_labels + classification_labels
    
    def get_primary_label(self) -> str:
        """Get primary label name (first enabled label)."""
        all_labels = self.get_label_names()
        if not all_labels:
            raise ValueError("No labels enabled in configuration")
        return all_labels[0]
