"""Feature engineering for technical and fundamental features."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from src.utils import (
    setup_logging,
    load_config,
    get_config_path,
    winsorize,
    cross_sectional_standardize
)

logger = setup_logging(__name__)


class FeatureEngineer:
    """Feature engineering for ML models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature engineer.
        
        Args:
            config_path: Path to features.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("features.yaml"))
        
        self.config = load_config(config_path)
        self.technical_features = self.config.get('technical', [])
        self.fundamental_features = self.config.get('fundamental', [])
        self.preprocessing = self.config.get('preprocessing', {})
        self.shift_days = self.config.get('shift_technical_features', 1)
    
    def compute_technical_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute technical features from price data.
        
        All technical features are shifted by shift_days to prevent leakage.
        
        Args:
            df: DataFrame with OHLCV data (must have symbol, date, AdjClose, Volume, etc.)
            
        Returns:
            DataFrame with technical features added
        """
        result = df.copy()
        result = result.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        for feature_config in self.technical_features:
            if not feature_config.get('enabled', True):
                continue
            
            feature_name = feature_config['name']
            feature_type = feature_config['type']
            
            logger.info(f"Computing technical feature: {feature_name}")
            
            if feature_type == 'return':
                window = feature_config['window']
                result[feature_name] = result.groupby('symbol')['AdjClose'].pct_change(window)
                
            elif feature_type == 'momentum':
                window = feature_config['window']
                result[feature_name] = result.groupby('symbol')['AdjClose'].pct_change(window)
                
            elif feature_type == 'volatility':
                window = feature_config['window']
                returns = result.groupby('symbol')['AdjClose'].pct_change()
                result[feature_name] = returns.groupby(result['symbol']).rolling(window).std().reset_index(0, drop=True)
                
            elif feature_type == 'rsi':
                window = feature_config['window']
                result[feature_name] = result.groupby('symbol', group_keys=False).apply(
                    lambda x: self._compute_rsi(x['AdjClose'], window)
                ).reset_index(drop=True)
                
            elif feature_type == 'atr':
                window = feature_config['window']
                result[feature_name] = result.groupby('symbol', group_keys=False).apply(
                    lambda x: self._compute_atr(x, window)
                ).reset_index(drop=True)
                
            elif feature_type == 'beta':
                window = feature_config['window']
                benchmark = feature_config.get('benchmark', 'SPY.US')
                # Simplified - would need benchmark data
                result[feature_name] = 1.0  # Placeholder
                
            elif feature_type == 'turnover':
                window = feature_config['window']
                result[feature_name] = result.groupby('symbol')['Volume'].rolling(window).mean().reset_index(0, drop=True)
                
            elif feature_type == 'skewness':
                window = feature_config['window']
                returns = result.groupby('symbol')['AdjClose'].pct_change()
                result[feature_name] = returns.groupby(result['symbol']).rolling(window).skew().reset_index(0, drop=True)
                
            elif feature_type == 'kurtosis':
                window = feature_config['window']
                returns = result.groupby('symbol')['AdjClose'].pct_change()
                result[feature_name] = returns.groupby(result['symbol']).rolling(window).kurt().reset_index(0, drop=True)
        
        # Shift all technical features to prevent leakage
        feature_cols = [f['name'] for f in self.technical_features if f.get('enabled', True)]
        for col in feature_cols:
            if col in result.columns:
                result[col] = result.groupby('symbol')[col].shift(self.shift_days)
        
        logger.info(f"Computed {len(feature_cols)} technical features")
        return result
    
    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def compute_fundamental_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute fundamental features (ratios) from raw fundamental data.
        
        Args:
            df: DataFrame with raw fundamental fields
            
        Returns:
            DataFrame with fundamental features added
        """
        result = df.copy()
        
        for feature_config in self.fundamental_features:
            if not feature_config.get('enabled', True):
                continue
            
            feature_name = feature_config['name']
            feature_type = feature_config['type']
            
            if feature_type == 'ratio':
                numerator = feature_config['numerator']
                denominator = feature_config['denominator']
                
                if numerator in result.columns and denominator in result.columns:
                    result[feature_name] = result[numerator] / (result[denominator] + 1e-10)
                    logger.debug(f"Computed fundamental feature: {feature_name}")
        
        logger.info(f"Computed fundamental features")
        return result
    
    def preprocess_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Preprocess features: winsorize, standardize, handle missing values.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        result = df.copy()
        
        # Winsorize outliers
        if self.preprocessing.get('winsorize', False):
            limits = tuple(self.preprocessing.get('winsorize_limits', [0.01, 0.99]))
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result.groupby('date')[col].transform(
                        lambda x: winsorize(x, limits)
                    )
        
        # Cross-sectional standardization
        if self.preprocessing.get('standardize_cross_sectional', False):
            group_col = 'sector' if self.preprocessing.get('standardize_by_sector', False) else None
            result = cross_sectional_standardize(result, feature_cols, group_col)
        
        # Handle missing values
        fill_method = self.preprocessing.get('fill_method', 'forward')
        max_fill = self.preprocessing.get('max_fill_periods', 5)
        
        if fill_method == 'forward':
            for col in feature_cols:
                if col in result.columns:
                    result[col] = result.groupby('symbol')[col].fillna(method='ffill', limit=max_fill)
        
        logger.info(f"Preprocessed {len(feature_cols)} features")
        return result
    
    def build_features(
        self,
        pit_panel: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build all features from PIT panel.
        
        Args:
            pit_panel: Point-in-time panel with prices and fundamentals
            
        Returns:
            DataFrame with all features
        """
        logger.info("Building features...")
        
        # Compute technical features
        df_with_tech = self.compute_technical_features(pit_panel)
        
        # Compute fundamental features
        df_with_fund = self.compute_fundamental_features(df_with_tech)
        
        # Get list of all feature columns
        tech_feature_cols = [
            f['name'] for f in self.technical_features if f.get('enabled', True)
        ]
        fund_feature_cols = [
            f['name'] for f in self.fundamental_features if f.get('enabled', True)
        ]
        all_feature_cols = tech_feature_cols + fund_feature_cols
        
        # Preprocess features
        df_final = self.preprocess_features(df_with_fund, all_feature_cols)
        
        logger.info(f"Built {len(all_feature_cols)} features")
        return df_final
    
    def get_feature_names(self) -> List[str]:
        """Get list of all enabled feature names."""
        tech_features = [
            f['name'] for f in self.technical_features if f.get('enabled', True)
        ]
        fund_features = [
            f['name'] for f in self.fundamental_features if f.get('enabled', True)
        ]
        return tech_features + fund_features
