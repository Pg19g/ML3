"""Model training with time-series cross-validation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
import xgboost as xgb

from src.utils import (
    setup_logging,
    load_config,
    get_config_path,
    get_model_registry_path,
    generate_model_id,
    save_json
)
from src.calendars import get_calendar

logger = setup_logging(__name__)


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.
    
    Implements embargo period to prevent leakage between train and test sets.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 21,
        calendar_name: str = "NYSE"
    ):
        """
        Initialize Purged K-Fold.
        
        Args:
            n_splits: Number of folds
            embargo_days: Number of trading days to embargo between train/test
            calendar_name: Trading calendar name
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.calendar = get_calendar(calendar_name)
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = df[date_col].unique()
        dates = sorted(dates)
        
        n_dates = len(dates)
        fold_size = n_dates // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Test period
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else n_dates
            
            test_dates = dates[test_start_idx:test_end_idx]
            
            # Train period (all data before test, with embargo)
            if test_start_idx > 0:
                # Apply embargo
                embargo_date = test_dates[0]
                embargo_start = self.calendar.add_trading_days(
                    pd.Timestamp(embargo_date),
                    -self.embargo_days
                )
                
                train_dates = [d for d in dates if d < embargo_start]
            else:
                train_dates = []
            
            # Get indices
            train_mask = df[date_col].isin(train_dates)
            test_mask = df[date_col].isin(test_dates)
            
            train_indices = df[train_mask].index.values
            test_indices = df[test_mask].index.values
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        logger.info(f"Created {len(splits)} purged K-fold splits")
        return splits


class ModelTrainer:
    """Train and evaluate ML models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to train.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("train.yaml"))
        
        self.config = load_config(config_path)
        self.models_config = self.config.get('models', {})
        self.cv_config = self.config.get('cv', {})
        self.metrics_config = self.config.get('metrics', {})
        self.random_seed = self.config.get('random_seed', 42)
        
        np.random.seed(self.random_seed)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature columns
            label_col: Label column name
            
        Returns:
            Tuple of (X, y)
        """
        # Remove rows with missing labels
        valid_mask = df[label_col].notna()
        df_valid = df[valid_mask].copy()
        
        # Filter feature_cols to only include columns that exist in df
        available_features = [col for col in feature_cols if col in df_valid.columns]
        missing_features = [col for col in feature_cols if col not in df_valid.columns]
        
        if missing_features:
            logger.warning(f"Skipping {len(missing_features)} missing feature columns: {missing_features[:5]}...")
        
        if not available_features:
            raise ValueError("No feature columns available in the dataset")
        
        logger.info(f"Using {len(available_features)} available features (out of {len(feature_cols)} requested)")
        
        # Remove rows with too many missing features
        feature_missing = df_valid[available_features].isna().sum(axis=1)
        max_missing = len(available_features) * 0.5
        df_valid = df_valid[feature_missing < max_missing]
        
        X = df_valid[available_features].copy()
        y = df_valid[label_col].copy()
        
        # Fill remaining missing values with median
        X = X.fillna(X.median())
        
        logger.info(f"Prepared data: {len(X)} samples, {len(available_features)} features")
        return X, y
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.Booster:
        """Train LightGBM model."""
        params = self.models_config['lightgbm']['params'].copy()
        params['random_state'] = self.random_seed
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> xgb.Booster:
        """Train XGBoost model."""
        params = self.models_config['xgboost']['params'].copy()
        params['random_state'] = self.random_seed
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            model_type: 'lightgbm' or 'xgboost'
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        if model_type == 'lightgbm':
            y_pred = model.predict(X)
        elif model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            y_pred = model.predict(dtest)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compute metrics
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y, y_pred)
        metrics['r2'] = r2_score(y, y_pred)
        
        # Information Coefficient (IC)
        ic, ic_pval = pearsonr(y, y_pred)
        metrics['ic'] = ic
        metrics['ic_pval'] = ic_pval
        
        # Rank IC
        rank_ic, rank_ic_pval = spearmanr(y, y_pred)
        metrics['rank_ic'] = rank_ic
        metrics['rank_ic_pval'] = rank_ic_pval
        
        return metrics
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        model_type: str = 'lightgbm'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature columns
            label_col: Label column name
            model_type: 'lightgbm' or 'xgboost'
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting cross-validation with {model_type}")
        
        # Prepare data
        X, y = self.prepare_data(df, feature_cols, label_col)
        df_valid = df.loc[X.index].copy()
        
        # Create CV splitter
        cv_method = self.cv_config.get('method', 'purged_kfold')
        n_splits = self.cv_config.get('n_splits', 5)
        embargo_days = self.cv_config.get('embargo_days', 21)
        
        if cv_method == 'purged_kfold':
            cv = PurgedKFold(n_splits=n_splits, embargo_days=embargo_days)
            splits = cv.split(df_valid, date_col='date')
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        # Train and evaluate on each fold
        fold_results = []
        models = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Train model
            if model_type == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train, X_test, y_test)
            elif model_type == 'xgboost':
                model = self.train_xgboost(X_train, y_train, X_test, y_test)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, model_type)
            test_metrics = self.evaluate_model(model, X_test, y_test, model_type)
            
            fold_results.append({
                'fold': fold_idx,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            models.append(model)
        
        # Aggregate results
        cv_results = {
            'model_type': model_type,
            'n_folds': len(splits),
            'fold_results': fold_results,
            'avg_metrics': self._aggregate_metrics(fold_results),
            'models': models
        }
        
        logger.info(f"Cross-validation complete. Avg IC: {cv_results['avg_metrics']['test_ic']:.4f}")
        return cv_results
    
    def _aggregate_metrics(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        metrics = {}
        
        # Get metric names from first fold
        metric_names = fold_results[0]['test_metrics'].keys()
        
        for metric_name in metric_names:
            values = [f['test_metrics'][metric_name] for f in fold_results]
            metrics[f'test_{metric_name}'] = np.mean(values)
            metrics[f'test_{metric_name}_std'] = np.std(values)
        
        return metrics
    
    def save_model(
        self,
        model: Any,
        model_type: str,
        feature_cols: List[str],
        label_col: str,
        metrics: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> str:
        """
        Save model to registry.
        
        Args:
            model: Trained model
            model_type: Model type
            feature_cols: Feature columns
            label_col: Label column
            metrics: Model metrics
            model_id: Optional model ID
            
        Returns:
            Model ID
        """
        if model_id is None:
            model_id = generate_model_id()
        
        registry_path = get_model_registry_path()
        model_dir = registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        if model_type == 'lightgbm':
            model.save_model(str(model_dir / 'model.txt'))
        elif model_type == 'xgboost':
            model.save_model(str(model_dir / 'model.json'))
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'feature_cols': feature_cols,
            'label_col': label_col,
            'metrics': metrics,
            'created_at': pd.Timestamp.now().isoformat()
        }
        save_json(metadata, str(model_dir / 'metadata.json'))
        
        logger.info(f"Saved model to {model_dir}")
        return model_id
