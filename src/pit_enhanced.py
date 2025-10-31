"""Enhanced Point-in-Time processing with interval joins and source timestamps."""

import pandas as pd
import polars as pl
import numpy as np
from datetime import timedelta, datetime
from typing import Optional, Dict, Any, List
import logging

from src.calendars import get_calendar
from src.utils import setup_logging, load_config, get_config_path

logger = setup_logging(__name__)


class PITProcessorEnhanced:
    """
    Enhanced PIT processor with:
    - Interval-based joins (using Polars)
    - Source timestamp tracking
    - Staleness handling (nulling out stale data)
    - Comprehensive validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize enhanced PIT processor."""
        if config_path is None:
            config_path = str(get_config_path("pit.yaml"))
        
        self.config = load_config(config_path)
        self.q_lag_days = self.config['q_lag_days']
        self.y_lag_days = self.config['y_lag_days']
        self.extra_trading_lag = self.config['extra_trading_lag']
        self.stale_max_days = self.config['stale_max_days']
        
        calendar_name = self.config.get('calendar', 'NYSE')
        self.calendar = get_calendar(calendar_name)
    
    def compute_as_of_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute as_of_date for fundamentals availability.
        
        Rule: as_of_date = max(filing_date, period_end + lag_days)
        where lag_days = 60 (Q), 90 (Y) when filing_date is missing.
        
        Then round up to next trading day and add extra_trading_lag.
        """
        result = df.copy()
        
        # Ensure datetime types
        result['period_end'] = pd.to_datetime(result['period_end'])
        
        # Determine lag based on statement type
        result['lag_days'] = result['statement_type'].map({
            'quarterly': self.q_lag_days,
            'annual': self.y_lag_days,
            'yearly': self.y_lag_days  # Support both 'annual' and 'yearly'
        })
        
        # Calculate base as_of_date
        if 'filing_date' in result.columns:
            result['filing_date'] = pd.to_datetime(result['filing_date'], errors='coerce')
            
            # When filing_date exists: max(filing_date, period_end + lag_days)
            # When missing: period_end + lag_days
            result['as_of_date'] = result.apply(
                lambda row: (
                    max(row['filing_date'], row['period_end'] + timedelta(days=row['lag_days']))
                    if pd.notna(row['filing_date'])
                    else row['period_end'] + timedelta(days=row['lag_days'])
                ),
                axis=1
            )
        else:
            # No filing_date column
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
    
    def build_validity_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build validity intervals per symbol.
        
        For each symbol:
        - valid_from = as_of_date
        - valid_to = next_as_of_date - 1 day
        
        This ensures no overlap and proper point-in-time semantics.
        """
        result = df.copy()
        result = result.sort_values(['symbol', 'as_of_date']).reset_index(drop=True)
        
        # valid_from is the as_of_date
        result['valid_from'] = result['as_of_date']
        
        # valid_to is the day before next as_of_date
        result['next_as_of'] = result.groupby('symbol')['as_of_date'].shift(-1)
        result['valid_to'] = result['next_as_of'] - timedelta(days=1)
        
        # For the last row per symbol, set valid_to to far future
        result['valid_to'] = result['valid_to'].fillna(pd.Timestamp('2099-12-31'))
        
        # Drop temporary column
        result = result.drop(columns=['next_as_of'])
        
        logger.info(f"Built validity intervals for {len(result)} rows")
        
        # Validate intervals
        invalid = result['valid_from'] > result['valid_to']
        if invalid.any():
            logger.warning(f"Found {invalid.sum()} invalid intervals (valid_from > valid_to)")
        
        return result
    
    def interval_join_polars(
        self,
        daily_panel: pd.DataFrame,
        fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform interval join using Polars for efficiency.
        
        For each (symbol, date), join the fundamental row where:
        - valid_from <= date <= valid_to
        
        Uses Polars' join_asof for efficient temporal joins.
        """
        # Convert to Polars
        daily_pl = pl.from_pandas(daily_panel)
        fund_pl = pl.from_pandas(fundamentals)
        
        # Ensure date columns are datetime
        daily_pl = daily_pl.with_columns([
            pl.col('date').cast(pl.Date)
        ])
        
        fund_pl = fund_pl.with_columns([
            pl.col('valid_from').cast(pl.Date),
            pl.col('valid_to').cast(pl.Date)
        ])
        
        # Sort by symbol and date for join_asof
        daily_pl = daily_pl.sort(['symbol', 'date'])
        fund_pl = fund_pl.sort(['symbol', 'valid_from'])
        
        # Perform as-of join on valid_from
        joined = daily_pl.join_asof(
            fund_pl,
            left_on='date',
            right_on='valid_from',
            by='symbol',
            strategy='backward'  # Get the most recent valid_from <= date
        )
        
        # Filter to ensure date <= valid_to
        joined = joined.filter(
            pl.col('date') <= pl.col('valid_to')
        )
        
        # Convert back to pandas
        result = joined.to_pandas()
        
        logger.info(
            f"Interval join: {len(daily_panel)} daily rows -> "
            f"{len(result)} with fundamentals"
        )
        
        return result
    
    def add_source_timestamps(
        self,
        df: pd.DataFrame,
        price_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Add source timestamps to track data lineage.
        
        - source_ts_price: timestamp of price data (usually date at market close)
        - source_ts_fund: as_of_date of the fundamental report
        """
        result = df.copy()
        
        # Price timestamp: use date at 16:00 (market close) if not provided
        if price_timestamp is None:
            if 'date' in result.columns:
                result['source_ts_price'] = pd.to_datetime(result['date']) + timedelta(hours=16)
            else:
                result['source_ts_price'] = pd.Timestamp.now()
        else:
            result['source_ts_price'] = price_timestamp
        
        # Fundamental timestamp: use as_of_date (when data became available)
        if 'as_of_date' in result.columns:
            result['source_ts_fund'] = pd.to_datetime(result['as_of_date'])
        elif 'valid_from' in result.columns:
            result['source_ts_fund'] = pd.to_datetime(result['valid_from'])
        else:
            result['source_ts_fund'] = pd.NaT
        
        return result
    
    def handle_staleness(
        self,
        df: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle stale fundamentals by nulling them out.
        
        If date - valid_from > stale_max_days:
        - Set is_stale_fund = True
        - Null out all fundamental features
        """
        result = df.copy()
        
        if 'valid_from' not in result.columns or 'date' not in result.columns:
            logger.warning("Missing valid_from or date columns, cannot handle staleness")
            return result
        
        # Calculate days since fundamental
        result['days_since_fund'] = (
            pd.to_datetime(result['date']) - pd.to_datetime(result['valid_from'])
        ).dt.days
        
        # Mark as stale
        result['is_stale_fund'] = result['days_since_fund'] > self.stale_max_days
        
        # Null out stale fundamentals
        if fundamental_cols is not None:
            stale_mask = result['is_stale_fund'] == True
            if stale_mask.any():
                result.loc[stale_mask, fundamental_cols] = np.nan
                logger.info(
                    f"Nulled out {stale_mask.sum()} stale fundamental rows "
                    f"({stale_mask.mean()*100:.1f}%)"
                )
        
        return result
    
    def build_pit_panel(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Build complete PIT panel with interval joins and source timestamps.
        
        Steps:
        1. Compute as_of_date for fundamentals
        2. Build validity intervals
        3. Perform interval join with daily prices
        4. Add source timestamps
        5. Handle staleness
        6. Validate PIT constraints
        """
        logger.info("Building enhanced PIT panel...")
        
        # Step 1: Compute as_of_date
        fund_with_asof = self.compute_as_of_date(fundamentals)
        
        # Step 2: Build validity intervals
        fund_with_validity = self.build_validity_intervals(fund_with_asof)
        
        # Step 3: Create daily panel from prices
        daily_panel = prices[['date', 'symbol']].drop_duplicates()
        
        # Step 4: Interval join
        pit_panel = self.interval_join_polars(daily_panel, fund_with_validity)
        
        # Step 5: Add source timestamps
        pit_panel = self.add_source_timestamps(pit_panel)
        
        # Step 6: Handle staleness
        if fundamental_cols is None:
            # Auto-detect fundamental columns
            fundamental_cols = [
                col for col in pit_panel.columns
                if col not in ['date', 'symbol', 'valid_from', 'valid_to', 
                              'source_ts_price', 'source_ts_fund', 'as_of_date',
                              'period_end', 'statement_type', 'filing_date',
                              'days_since_fund', 'is_stale_fund', 'lag_days']
            ]
        
        pit_panel = self.handle_staleness(pit_panel, fundamental_cols)
        
        # Step 7: Merge back with full price data
        pit_panel = pit_panel.merge(
            prices,
            on=['date', 'symbol'],
            how='left'
        )
        
        # Step 8: Validate PIT constraints
        self.validate_pit_integrity(pit_panel)
        
        logger.info(f"Built PIT panel with {len(pit_panel)} rows")
        return pit_panel
    
    def validate_pit_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate PIT integrity constraints.
        
        Checks:
        1. source_ts_price <= date
        2. source_ts_fund <= date (if not null)
        3. valid_from <= date (if not null)
        4. max(source_ts_price, source_ts_fund) <= date
        
        Returns dict with validation results.
        """
        violations = []
        
        # Check 1: source_ts_price <= date
        if 'source_ts_price' in df.columns and 'date' in df.columns:
            df_check = df.copy()
            df_check['ts_price_date'] = pd.to_datetime(df_check['source_ts_price']).dt.date
            df_check['date_only'] = pd.to_datetime(df_check['date']).dt.date
            
            price_leakage = df_check['ts_price_date'] > df_check['date_only']
            if price_leakage.any():
                violations.append({
                    'check': 'source_ts_price <= date',
                    'violations': price_leakage.sum(),
                    'severity': 'CRITICAL'
                })
        
        # Check 2: source_ts_fund <= date
        if 'source_ts_fund' in df.columns and 'date' in df.columns:
            df_check = df.copy()
            mask = df_check['source_ts_fund'].notna()
            
            if mask.any():
                df_check['ts_fund_date'] = pd.to_datetime(df_check['source_ts_fund']).dt.date
                df_check['date_only'] = pd.to_datetime(df_check['date']).dt.date
                
                fund_leakage = mask & (df_check['ts_fund_date'] > df_check['date_only'])
                if fund_leakage.any():
                    violations.append({
                        'check': 'source_ts_fund <= date',
                        'violations': fund_leakage.sum(),
                        'severity': 'CRITICAL'
                    })
        
        # Check 3: valid_from <= date
        if 'valid_from' in df.columns and 'date' in df.columns:
            mask = df['valid_from'].notna()
            invalid = mask & (pd.to_datetime(df['valid_from']) > pd.to_datetime(df['date']))
            
            if invalid.any():
                violations.append({
                    'check': 'valid_from <= date',
                    'violations': invalid.sum(),
                    'severity': 'CRITICAL'
                })
        
        # Log results
        if violations:
            logger.error(f"PIT integrity check FAILED with {len(violations)} violations:")
            for v in violations:
                logger.error(f"  - {v['check']}: {v['violations']} violations ({v['severity']})")
        else:
            logger.info("✓ PIT integrity check PASSED")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'total_violations': sum(v['violations'] for v in violations)
        }
