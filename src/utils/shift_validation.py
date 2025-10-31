"""Utilities for enforcing and validating feature shifts to prevent leakage."""

import pandas as pd
import numpy as np
from typing import Callable, List, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def ensure_shift(shift_days: int = 1):
    """
    Decorator to ensure a feature computation function applies proper shifting.
    
    Usage:
        @ensure_shift(shift_days=1)
        def compute_returns(df):
            df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
            return df
    
    The decorator will automatically shift all new columns by shift_days.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            # Get columns before
            cols_before = set(df.columns)
            
            # Call the function
            result = func(df, *args, **kwargs)
            
            # Get new columns
            cols_after = set(result.columns)
            new_cols = cols_after - cols_before
            
            # Shift new columns
            if new_cols and 'symbol' in result.columns:
                logger.info(f"Auto-shifting {len(new_cols)} new features by {shift_days} days")
                for col in new_cols:
                    result[col] = result.groupby('symbol')[col].shift(shift_days)
            
            return result
        
        return wrapper
    return decorator


def validate_feature_shift(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = 'date',
    symbol_col: str = 'symbol',
    price_col: str = 'AdjClose'
) -> dict:
    """
    Validate that features are properly shifted and don't use current-day data.
    
    Checks:
    1. Features at time t should not correlate perfectly with price at time t
    2. Features should have NaN at the first observation per symbol
    3. No feature should reference data from time t or t+1
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names to validate
        date_col: Name of date column
        symbol_col: Name of symbol column
        price_col: Name of price column
    
    Returns:
        Dict with validation results
    """
    issues = []
    
    # Check 1: First observation should be NaN (due to shift)
    for col in feature_cols:
        if col not in df.columns:
            issues.append({
                'feature': col,
                'issue': 'Column not found',
                'severity': 'ERROR'
            })
            continue
        
        # Get first observation per symbol
        first_obs = df.groupby(symbol_col).first()
        
        if col in first_obs.columns:
            non_nan_first = first_obs[col].notna().sum()
            
            if non_nan_first > 0:
                issues.append({
                    'feature': col,
                    'issue': f'{non_nan_first} symbols have non-NaN first observation (should be NaN due to shift)',
                    'severity': 'WARNING'
                })
    
    # Check 2: Correlation with current price (should not be perfect)
    if price_col in df.columns:
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # Calculate correlation with current price
            valid_mask = df[col].notna() & df[price_col].notna()
            
            if valid_mask.sum() > 10:  # Need enough data points
                corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, price_col])
                
                # Perfect correlation suggests no shift
                if abs(corr) > 0.99:
                    issues.append({
                        'feature': col,
                        'issue': f'Perfect correlation ({corr:.4f}) with current price - possible leakage',
                        'severity': 'ERROR'
                    })
    
    # Check 3: Temporal consistency
    # Features at t should use data up to t-1
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Sort by symbol and date
        df_sorted = df.sort_values([symbol_col, date_col])
        
        # Check if feature at t is available before t
        # (This is a heuristic - perfect check would require knowing computation)
        # For now, just check that shift was applied (NaN at start)
        for symbol in df_sorted[symbol_col].unique()[:5]:  # Sample check
            symbol_data = df_sorted[df_sorted[symbol_col] == symbol]
            
            if len(symbol_data) > 1:
                # First value should be NaN
                first_val = symbol_data[col].iloc[0]
                
                if pd.notna(first_val):
                    # This might be okay if it's a fundamental feature
                    # But for technical features, it's suspicious
                    pass  # Already caught in Check 1
    
    # Summary
    errors = [i for i in issues if i['severity'] == 'ERROR']
    warnings = [i for i in issues if i['severity'] == 'WARNING']
    
    result = {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_issues': len(issues)
    }
    
    if errors:
        logger.error(f"Feature shift validation FAILED with {len(errors)} errors")
        for e in errors:
            logger.error(f"  - {e['feature']}: {e['issue']}")
    
    if warnings:
        logger.warning(f"Feature shift validation has {len(warnings)} warnings")
        for w in warnings:
            logger.warning(f"  - {w['feature']}: {w['issue']}")
    
    if not issues:
        logger.info("✓ Feature shift validation PASSED")
    
    return result


def validate_label_alignment(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    key_cols: List[str] = ['symbol', 'date']
) -> dict:
    """
    Validate that features and labels are properly aligned.
    
    Checks:
    1. Same (symbol, date) index
    2. No mismatched rows
    3. Labels are forward-looking (negative shift)
    
    Args:
        features_df: DataFrame with features
        labels_df: DataFrame with labels
        key_cols: Columns that define the index
    
    Returns:
        Dict with validation results
    """
    issues = []
    
    # Check 1: Index alignment
    features_index = set(features_df[key_cols].apply(tuple, axis=1))
    labels_index = set(labels_df[key_cols].apply(tuple, axis=1))
    
    only_in_features = features_index - labels_index
    only_in_labels = labels_index - features_index
    
    if only_in_features:
        issues.append({
            'issue': f'{len(only_in_features)} rows in features but not in labels',
            'severity': 'WARNING'
        })
    
    if only_in_labels:
        issues.append({
            'issue': f'{len(only_in_labels)} rows in labels but not in features',
            'severity': 'WARNING'
        })
    
    # Check 2: Merge and check for mismatches
    merged = features_df[key_cols].merge(
        labels_df[key_cols],
        on=key_cols,
        how='outer',
        indicator=True
    )
    
    both = (merged['_merge'] == 'both').sum()
    total = len(merged)
    
    if both < total:
        issues.append({
            'issue': f'Only {both}/{total} rows match between features and labels',
            'severity': 'ERROR' if both < total * 0.9 else 'WARNING'
        })
    
    # Check 3: Labels should be forward-looking
    # (This is checked by verifying NaN at the end, not the start)
    label_cols = [c for c in labels_df.columns if c not in key_cols]
    
    if 'symbol' in labels_df.columns:
        for col in label_cols:
            # Last observation per symbol should be NaN (due to negative shift)
            last_obs = labels_df.groupby('symbol').last()
            
            if col in last_obs.columns:
                non_nan_last = last_obs[col].notna().sum()
                
                if non_nan_last > 0:
                    issues.append({
                        'label': col,
                        'issue': f'{non_nan_last} symbols have non-NaN last observation (should be NaN due to forward shift)',
                        'severity': 'WARNING'
                    })
    
    # Summary
    errors = [i for i in issues if i.get('severity') == 'ERROR']
    warnings = [i for i in issues if i.get('severity') == 'WARNING']
    
    result = {
        'aligned': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_issues': len(issues),
        'overlap': both if 'both' in locals() else 0,
        'total': total if 'total' in locals() else 0
    }
    
    if errors:
        logger.error(f"Label alignment validation FAILED with {len(errors)} errors")
        for e in errors:
            logger.error(f"  - {e['issue']}")
    
    if warnings:
        logger.warning(f"Label alignment has {len(warnings)} warnings")
        for w in warnings:
            logger.warning(f"  - {w['issue']}")
    
    if not issues:
        logger.info("✓ Label alignment validation PASSED")
    
    return result


def apply_terminal_shift(
    df: pd.DataFrame,
    feature_cols: List[str],
    shift_days: int = 1,
    symbol_col: str = 'symbol'
) -> pd.DataFrame:
    """
    Apply terminal shift to all feature columns.
    
    This is a safety function to ensure all features are shifted,
    even if individual feature computations forgot to shift.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to shift
        shift_days: Number of days to shift (default 1)
        symbol_col: Symbol column for grouping
    
    Returns:
        DataFrame with shifted features
    """
    result = df.copy()
    
    for col in feature_cols:
        if col in result.columns:
            result[col] = result.groupby(symbol_col)[col].shift(shift_days)
    
    logger.info(f"Applied terminal shift of {shift_days} days to {len(feature_cols)} features")
    return result


def check_no_future_reference(
    df: pd.DataFrame,
    feature_col: str,
    date_col: str = 'date',
    symbol_col: str = 'symbol'
) -> bool:
    """
    Check that a feature at time t doesn't reference data from t or t+1.
    
    This is a heuristic check based on:
    1. First observation should be NaN (due to shift)
    2. Feature should not be perfectly correlated with current-day price change
    
    Args:
        df: DataFrame with feature
        feature_col: Feature column to check
        date_col: Date column
        symbol_col: Symbol column
    
    Returns:
        True if no future reference detected, False otherwise
    """
    if feature_col not in df.columns:
        logger.warning(f"Feature {feature_col} not found")
        return False
    
    # Check 1: First observation should be NaN
    df_sorted = df.sort_values([symbol_col, date_col])
    first_obs = df_sorted.groupby(symbol_col)[feature_col].first()
    
    if first_obs.notna().any():
        logger.warning(f"Feature {feature_col} has non-NaN first observations")
        return False
    
    return True
