"""Point-in-time (PIT) data processing to prevent look-ahead bias."""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, Dict, Any
import logging

from src.calendars import get_calendar
from src.utils import setup_logging, load_config, get_config_path

logger = setup_logging(__name__)


class PITProcessor:
    """Process fundamentals data with point-in-time logic."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PIT processor.
        
        Args:
            config_path: Path to pit.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("pit.yaml"))
        
        self.config = load_config(config_path)
        self.q_lag_days = self.config['q_lag_days']
        self.y_lag_days = self.config['y_lag_days']
        self.extra_trading_lag = self.config['extra_trading_lag']
        self.stale_max_days = self.config['stale_max_days']
        
        calendar_name = self.config.get('calendar', 'NYSE')
        self.calendar = get_calendar(calendar_name)
    
    def compute_as_of_date(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute as_of_date for fundamentals availability.
        
        The as_of_date is when the fundamental data becomes available:
        - If filing_date exists: max(filing_date, period_end + lag_days)
        - Otherwise: period_end + lag_days
        
        Then round up to next trading day and add extra_trading_lag.
        
        Args:
            df: DataFrame with fundamentals (must have period_end, statement_type)
            
        Returns:
            DataFrame with as_of_date column
        """
        result = df.copy()
        
        # Handle empty DataFrame
        if result.empty:
            return result
        
        # Determine lag based on statement type
        result['lag_days'] = result['statement_type'].map({
            'quarterly': self.q_lag_days,
            'yearly': self.y_lag_days
        })
        
        # Calculate initial as_of_date
        result['period_end'] = pd.to_datetime(result['period_end'])
        
        if 'filing_date' in result.columns and result['filing_date'].notna().any():
            result['filing_date'] = pd.to_datetime(result['filing_date'], errors='coerce')
            result['as_of_date'] = result.apply(
                lambda row: max(
                    row['filing_date'] if pd.notna(row['filing_date']) else row['period_end'],
                    row['period_end'] + timedelta(days=row['lag_days'])
                ),
                axis=1
            )
        else:
            result['as_of_date'] = result.apply(
                lambda row: row['period_end'] + timedelta(days=row['lag_days']),
                axis=1
            )
        
        # Round to next trading day
        result['as_of_date'] = self.calendar.align_dates_to_trading_days(
            result['as_of_date'],
            direction='forward'
        )
        
        # Add extra trading lag
        result['as_of_date'] = result['as_of_date'].apply(
            lambda d: self.calendar.add_trading_days(d, self.extra_trading_lag)
        )
        
        logger.info(f"Computed as_of_date for {len(result)} fundamental rows")
        return result
    
    def compute_validity_intervals(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute validity intervals for fundamentals.
        
        Each fundamental row is valid from as_of_date until the next
        fundamental becomes available (next as_of_date - 1 day).
        
        Args:
            df: DataFrame with as_of_date
            
        Returns:
            DataFrame with valid_from and valid_to columns
        """
        result = df.copy()
        result = result.sort_values(['symbol', 'as_of_date']).reset_index(drop=True)
        
        # valid_from is the as_of_date
        result['valid_from'] = result['as_of_date']
        
        # valid_to is the day before next as_of_date
        result['valid_to'] = result.groupby('symbol')['as_of_date'].shift(-1)
        result['valid_to'] = result['valid_to'] - timedelta(days=1)
        
        # For the last row per symbol, set valid_to to far future
        result['valid_to'] = result['valid_to'].fillna(pd.Timestamp('2099-12-31'))
        
        logger.info(f"Computed validity intervals for {len(result)} rows")
        return result
    
    def as_of_join(
        self,
        daily_panel: pd.DataFrame,
        fundamentals: pd.DataFrame,
        on: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Perform as-of join between daily panel and fundamentals.
        
        For each (symbol, date) in daily_panel, join the fundamental row
        where valid_from <= date <= valid_to.
        
        Args:
            daily_panel: DataFrame with (date, symbol) rows
            fundamentals: DataFrame with validity intervals
            on: Column to join on (default 'symbol')
            
        Returns:
            Joined DataFrame
        """
        # Ensure date columns are datetime
        daily_panel['date'] = pd.to_datetime(daily_panel['date'])
        fundamentals['valid_from'] = pd.to_datetime(fundamentals['valid_from'])
        fundamentals['valid_to'] = pd.to_datetime(fundamentals['valid_to'])
        
        # Merge on symbol
        merged = daily_panel.merge(
            fundamentals,
            on=on,
            how='left',
            suffixes=('', '_fund')
        )
        
        # Filter to valid rows
        valid_mask = (
            (merged['date'] >= merged['valid_from']) &
            (merged['date'] <= merged['valid_to'])
        )
        
        # Keep only valid rows
        result = merged[valid_mask].copy()
        
        # Check for duplicates (should not happen with proper validity intervals)
        duplicates = result.duplicated(subset=['date', on], keep='first')
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate rows in as-of join")
            result = result[~duplicates]
        
        logger.info(f"As-of join: {len(daily_panel)} daily rows -> {len(result)} with fundamentals")
        return result
    
    def add_staleness_flags(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Add staleness flags for fundamentals.
        
        Mark fundamentals as stale if they are older than stale_max_days.
        
        Args:
            df: DataFrame with date and valid_from columns
            date_col: Name of date column
            
        Returns:
            DataFrame with is_stale_fund and days_since_fund columns
        """
        result = df.copy()
        
        if 'valid_from' not in result.columns:
            logger.warning("No valid_from column, cannot compute staleness")
            return result
        
        # Calculate days since fundamental
        result['days_since_fund'] = (
            result[date_col] - result['valid_from']
        ).dt.days
        
        # Mark as stale
        result['is_stale_fund'] = result['days_since_fund'] > self.stale_max_days
        
        stale_pct = result['is_stale_fund'].mean() * 100
        logger.info(f"Staleness: {stale_pct:.1f}% of rows are stale")
        
        return result
    
    def build_pit_panel(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build complete PIT panel from prices and fundamentals.
        
        Args:
            prices: DataFrame with daily prices (date, symbol, OHLCV)
            fundamentals: DataFrame with raw fundamentals
            
        Returns:
            PIT panel with prices and fundamentals properly aligned
        """
        logger.info("Building PIT panel...")
        
        # Compute as_of_date for fundamentals
        fund_with_asof = self.compute_as_of_date(fundamentals)
        
        # Compute validity intervals
        fund_with_validity = self.compute_validity_intervals(fund_with_asof)
        
        # Create daily panel from prices
        daily_panel = prices[['date', 'symbol']].copy()
        daily_panel = daily_panel.drop_duplicates()
        
        # Perform as-of join
        pit_panel = self.as_of_join(daily_panel, fund_with_validity, on='symbol')
        
        # Add staleness flags
        pit_panel = self.add_staleness_flags(pit_panel)
        
        # Merge back with prices
        pit_panel = pit_panel.merge(
            prices,
            on=['date', 'symbol'],
            how='left'
        )
        
        logger.info(f"Built PIT panel with {len(pit_panel)} rows")
        return pit_panel
    
    def check_pit_integrity(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Check PIT integrity - ensure no future data leakage.
        
        Args:
            df: DataFrame to check
            date_col: Name of date column
            
        Returns:
            Dictionary with integrity check results
        """
        results = {
            'passed': True,
            'violations': 0,
            'checks': []
        }
        
        # Check 1: valid_from should not be after date
        if 'valid_from' in df.columns:
            violations = (df['valid_from'] > df[date_col]).sum()
            results['checks'].append({
                'name': 'valid_from <= date',
                'passed': violations == 0,
                'violations': int(violations)
            })
            if violations > 0:
                results['passed'] = False
                results['violations'] += violations
        
        # Check 2: period_end should not be after date
        if 'period_end' in df.columns:
            violations = (df['period_end'] > df[date_col]).sum()
            results['checks'].append({
                'name': 'period_end <= date',
                'passed': violations == 0,
                'violations': int(violations)
            })
            if violations > 0:
                results['passed'] = False
                results['violations'] += violations
        
        logger.info(f"PIT integrity check: {'PASSED' if results['passed'] else 'FAILED'}")
        return results
