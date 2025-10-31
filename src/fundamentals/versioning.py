"""Versioning and restatement handling for fundamentals data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from src.calendars import get_calendar
from src.utils import setup_logging

logger = setup_logging(__name__)


class FundamentalsVersioning:
    """
    Handles versioning and restatements for fundamentals data.
    
    Key concepts:
    - as_of_date: When the original report became available
    - updated_at: When a restatement/correction was made
    - effective_from: When this version of the data became available
    - version_id: Sequential version number per (symbol, statement_type, period_end)
    """
    
    def __init__(
        self,
        q_lag_days: int = 60,
        y_lag_days: int = 90,
        extra_trading_lag: int = 2,
        calendar_name: str = 'NYSE'
    ):
        """
        Initialize versioning handler.
        
        Args:
            q_lag_days: Lag days for quarterly reports
            y_lag_days: Lag days for annual reports
            extra_trading_lag: Extra trading days after rounding
            calendar_name: Trading calendar to use
        """
        self.q_lag_days = q_lag_days
        self.y_lag_days = y_lag_days
        self.extra_trading_lag = extra_trading_lag
        self.calendar = get_calendar(calendar_name)
    
    def compute_as_of_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute as_of_date for each fundamental row.
        
        Logic:
        - as_of_base = max(filing_date, period_end + lag_days)
          where lag_days = 60 (Q) / 90 (Y) if filing_date missing
        - as_of_date = add_trading_days(next_trading_day(as_of_base), extra_trading_lag)
        
        Args:
            df: DataFrame with fundamentals
        
        Returns:
            DataFrame with as_of_date column
        """
        result = df.copy()
        
        # Ensure datetime types
        result['period_end'] = pd.to_datetime(result['period_end'])
        if 'filing_date' in result.columns:
            result['filing_date'] = pd.to_datetime(result['filing_date'], errors='coerce')
        
        # Determine lag based on statement type
        result['lag_days'] = result['statement_type'].map({
            'quarterly': self.q_lag_days,
            'annual': self.y_lag_days
        })
        
        # Compute as_of_base
        if 'filing_date' in result.columns:
            # When filing_date exists: max(filing_date, period_end + lag_days)
            # When missing: period_end + lag_days
            result['as_of_base'] = result.apply(
                lambda row: (
                    max(row['filing_date'], row['period_end'] + timedelta(days=row['lag_days']))
                    if pd.notna(row['filing_date'])
                    else row['period_end'] + timedelta(days=row['lag_days'])
                ),
                axis=1
            )
        else:
            result['as_of_base'] = result.apply(
                lambda row: row['period_end'] + timedelta(days=row['lag_days']),
                axis=1
            )
        
        # Round to next trading day
        result['as_of_date'] = self.calendar.align_dates_to_trading_days(
            result['as_of_base'],
            direction='forward'
        )
        
        # Add extra trading lag
        result['as_of_date'] = result['as_of_date'].apply(
            lambda d: self.calendar.add_trading_days(d, self.extra_trading_lag)
        )
        
        # Drop temporary columns
        result = result.drop(columns=['lag_days', 'as_of_base'])
        
        logger.info(f"Computed as_of_date for {len(result)} rows")
        return result
    
    def compute_effective_from(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute effective_from for each version.
        
        Logic:
        - If updated_at present:
            effective_from_version = add_trading_days(next_trading_day(updated_at), extra_trading_lag)
            effective_from = max(as_of_date, effective_from_version)
        - Else:
            effective_from = as_of_date
        
        Args:
            df: DataFrame with as_of_date
        
        Returns:
            DataFrame with effective_from column
        """
        result = df.copy()
        
        if 'updated_at' in result.columns:
            # Convert updated_at to datetime
            result['updated_at'] = pd.to_datetime(result['updated_at'], errors='coerce')
            
            # Compute effective_from_version for rows with updated_at
            has_update = result['updated_at'].notna()
            
            if has_update.any():
                # Round updated_at to next trading day
                result.loc[has_update, 'updated_at_trading'] = self.calendar.align_dates_to_trading_days(
                    result.loc[has_update, 'updated_at'],
                    direction='forward'
                )
                
                # Add extra trading lag
                result.loc[has_update, 'effective_from_version'] = result.loc[has_update, 'updated_at_trading'].apply(
                    lambda d: self.calendar.add_trading_days(d, self.extra_trading_lag)
                )
                
                # effective_from = max(as_of_date, effective_from_version)
                result.loc[has_update, 'effective_from'] = result.loc[has_update, [
                    'as_of_date', 'effective_from_version'
                ]].max(axis=1)
                
                # For rows without update, effective_from = as_of_date
                result.loc[~has_update, 'effective_from'] = result.loc[~has_update, 'as_of_date']
                
                # Drop temporary columns
                result = result.drop(columns=['updated_at_trading', 'effective_from_version'], errors='ignore')
            else:
                # No updates, effective_from = as_of_date
                result['effective_from'] = result['as_of_date']
        else:
            # No updated_at column, effective_from = as_of_date
            result['effective_from'] = result['as_of_date']
        
        logger.info(f"Computed effective_from for {len(result)} rows")
        return result
    
    def assign_version_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign version_id by ascending effective_from per (symbol, statement_type, period_end).
        
        Args:
            df: DataFrame with effective_from
        
        Returns:
            DataFrame with version_id column
        """
        result = df.copy()
        
        # Sort by key and effective_from
        result = result.sort_values([
            'symbol', 'statement_type', 'period_end', 'effective_from'
        ]).reset_index(drop=True)
        
        # Assign version_id
        result['version_id'] = result.groupby([
            'symbol', 'statement_type', 'period_end'
        ]).cumcount() + 1
        
        # Log versioning stats
        total_versions = len(result)
        unique_reports = result.groupby([
            'symbol', 'statement_type', 'period_end'
        ]).size()
        
        restated = (unique_reports > 1).sum()
        max_versions = unique_reports.max()
        
        logger.info(
            f"Assigned version IDs: {total_versions} total versions, "
            f"{restated} reports with restatements, "
            f"max versions per report: {max_versions}"
        )
        
        return result
    
    def build_versioned_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete versioned fundamentals with as_of_date, effective_from, version_id.
        
        Args:
            df: Raw fundamentals DataFrame
        
        Returns:
            Versioned fundamentals DataFrame
        """
        logger.info("Building versioned fundamentals...")
        
        # Step 1: Compute as_of_date
        result = self.compute_as_of_date(df)
        
        # Step 2: Compute effective_from
        result = self.compute_effective_from(result)
        
        # Step 3: Assign version_ids
        result = self.assign_version_ids(result)
        
        logger.info(f"Built versioned fundamentals: {len(result)} rows")
        return result
    
    def get_version_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about versioning and restatements.
        
        Args:
            df: Versioned fundamentals DataFrame
        
        Returns:
            Dict with versioning statistics
        """
        stats = {}
        
        # Total versions
        stats['total_versions'] = len(df)
        
        # Unique reports
        unique_reports = df.groupby([
            'symbol', 'statement_type', 'period_end'
        ]).size()
        stats['unique_reports'] = len(unique_reports)
        
        # Restatements
        stats['reports_with_restatements'] = (unique_reports > 1).sum()
        stats['restatement_rate'] = stats['reports_with_restatements'] / stats['unique_reports']
        
        # Version distribution
        stats['max_versions_per_report'] = unique_reports.max()
        stats['avg_versions_per_report'] = unique_reports.mean()
        
        # Symbols with most restatements
        symbol_restatements = df[df['version_id'] > 1].groupby('symbol').size()
        if len(symbol_restatements) > 0:
            stats['top_restating_symbols'] = symbol_restatements.nlargest(10).to_dict()
        else:
            stats['top_restating_symbols'] = {}
        
        return stats
