"""SCD-2 validity interval construction for fundamentals."""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, Dict, Any
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class IntervalBuilder:
    """
    Builds SCD-2 (Slowly Changing Dimension Type 2) validity intervals.
    
    For each symbol & statement_type:
    - valid_from = effective_from
    - valid_to = min(next_effective_from - 1d, next_report_as_of - 1d)
    
    This ensures proper temporal semantics where each version is valid
    until either a new version or a new report becomes available.
    """
    
    def __init__(self):
        """Initialize interval builder."""
        pass
    
    def build_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build validity intervals for versioned fundamentals.
        
        Logic:
        - For each symbol & statement_type, sort by effective_from
        - valid_from = effective_from
        - valid_to = min(next_effective_from - 1d, next_report_as_of - 1d)
          where next_report_as_of is the following period's as_of_date
        
        Args:
            df: Versioned fundamentals with effective_from and as_of_date
        
        Returns:
            DataFrame with valid_from and valid_to columns
        """
        result = df.copy()
        
        # Ensure datetime types
        result['effective_from'] = pd.to_datetime(result['effective_from'])
        result['as_of_date'] = pd.to_datetime(result['as_of_date'])
        
        # Sort by symbol, statement_type, effective_from
        result = result.sort_values([
            'symbol', 'statement_type', 'effective_from'
        ]).reset_index(drop=True)
        
        # valid_from = effective_from
        result['valid_from'] = result['effective_from']
        
        # Compute next_effective_from (next version of same report)
        result['next_effective_from'] = result.groupby([
            'symbol', 'statement_type', 'period_end'
        ])['effective_from'].shift(-1)
        
        # Compute next_report_as_of (next period's as_of_date for same statement_type)
        result['next_report_as_of'] = result.groupby([
            'symbol', 'statement_type'
        ])['as_of_date'].shift(-1)
        
        # valid_to = min(next_effective_from - 1d, next_report_as_of - 1d)
        # Use the earlier of the two to ensure no overlap
        result['valid_to_version'] = result['next_effective_from'] - timedelta(days=1)
        result['valid_to_report'] = result['next_report_as_of'] - timedelta(days=1)
        
        # Take minimum (earlier date)
        result['valid_to'] = result[['valid_to_version', 'valid_to_report']].min(axis=1)
        
        # For the last version of the last report per (symbol, statement_type),
        # set valid_to to far future
        result['valid_to'] = result['valid_to'].fillna(pd.Timestamp('2099-12-31'))
        
        # Drop temporary columns
        result = result.drop(columns=[
            'next_effective_from', 'next_report_as_of',
            'valid_to_version', 'valid_to_report'
        ])
        
        # Validate intervals
        self._validate_intervals(result)
        
        logger.info(f"Built validity intervals for {len(result)} rows")
        return result
    
    def _validate_intervals(self, df: pd.DataFrame) -> None:
        """
        Validate that intervals are well-formed.
        
        Checks:
        - valid_from <= valid_to
        - No overlapping intervals for same (symbol, statement_type)
        """
        # Check valid_from <= valid_to
        invalid = df['valid_from'] > df['valid_to']
        if invalid.any():
            logger.error(f"Found {invalid.sum()} rows with valid_from > valid_to")
            sample = df[invalid].head(5)[[
                'symbol', 'statement_type', 'period_end',
                'valid_from', 'valid_to', 'version_id'
            ]]
            logger.error(f"Sample invalid intervals:\n{sample}")
        
        # Check for overlaps within same (symbol, statement_type)
        overlaps = 0
        for (symbol, stmt_type), group in df.groupby(['symbol', 'statement_type']):
            group = group.sort_values('valid_from')
            
            for i in range(len(group) - 1):
                current_to = group.iloc[i]['valid_to']
                next_from = group.iloc[i + 1]['valid_from']
                
                # Should have current_to < next_from (no overlap)
                if current_to >= next_from:
                    overlaps += 1
                    if overlaps <= 5:  # Log first 5
                        logger.warning(
                            f"Overlap detected for {symbol} {stmt_type}: "
                            f"interval {i} ends {current_to}, "
                            f"interval {i+1} starts {next_from}"
                        )
        
        if overlaps > 0:
            logger.error(f"Found {overlaps} overlapping intervals")
        else:
            logger.info("✓ No overlapping intervals detected")
    
    def get_interval_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about validity intervals.
        
        Args:
            df: DataFrame with validity intervals
        
        Returns:
            Dict with interval statistics
        """
        stats = {}
        
        # Interval durations
        df['interval_days'] = (df['valid_to'] - df['valid_from']).dt.days
        
        # Filter out far-future valid_to
        finite_intervals = df[df['valid_to'] < pd.Timestamp('2099-01-01')]
        
        if len(finite_intervals) > 0:
            stats['avg_interval_days'] = finite_intervals['interval_days'].mean()
            stats['median_interval_days'] = finite_intervals['interval_days'].median()
            stats['min_interval_days'] = finite_intervals['interval_days'].min()
            stats['max_interval_days'] = finite_intervals['interval_days'].max()
        else:
            stats['avg_interval_days'] = None
            stats['median_interval_days'] = None
            stats['min_interval_days'] = None
            stats['max_interval_days'] = None
        
        # Count of intervals per symbol
        intervals_per_symbol = df.groupby('symbol').size()
        stats['avg_intervals_per_symbol'] = intervals_per_symbol.mean()
        stats['max_intervals_per_symbol'] = intervals_per_symbol.max()
        
        # Count by statement type
        stats['quarterly_intervals'] = (df['statement_type'] == 'quarterly').sum()
        stats['annual_intervals'] = (df['statement_type'] == 'annual').sum()
        
        return stats
