"""Complete PIT processor with interval joins, staleness, and leakage checks."""

import pandas as pd
import polars as pl
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from src.fundamentals.versioning import FundamentalsVersioning
from src.fundamentals.intervals import IntervalBuilder
from src.fundamentals.precedence import PrecedencePolicy
from src.utils import setup_logging, load_config

logger = setup_logging(__name__)


class PITFundamentalsProcessor:
    """
    Complete point-in-time fundamentals processor.
    
    Handles:
    - Versioning and restatements
    - SCD-2 validity intervals
    - Interval joins into daily panel
    - Staleness handling
    - Quarterly vs annual precedence
    - Source timestamp tracking
    - Leakage validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PIT processor.
        
        Args:
            config_path: Path to pit_enhanced.yaml config
        """
        if config_path is None:
            from src.utils import get_config_path
            config_path = str(get_config_path("pit_enhanced.yaml"))
        
        self.config = load_config(config_path)
        
        # Initialize components
        self.versioning = FundamentalsVersioning(
            q_lag_days=self.config['q_lag_days'],
            y_lag_days=self.config['y_lag_days'],
            extra_trading_lag=self.config['extra_trading_lag'],
            calendar_name=self.config['calendar']
        )
        
        self.interval_builder = IntervalBuilder()
        
        self.precedence = PrecedencePolicy(
            policy=self.config['precedence']
        )
        
        self.stale_max_days = self.config['stale_max_days']
        self.join_method = self.config.get('join_method', 'polars')
    
    def build_fundamentals_intervals(
        self,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build fundamentals intervals from raw fundamentals.
        
        Steps:
        1. Compute versioning (as_of_date, effective_from, version_id)
        2. Build validity intervals (valid_from, valid_to)
        3. Compute ratios at filing level
        
        Args:
            fundamentals_df: Raw fundamentals DataFrame
        
        Returns:
            Fundamentals intervals DataFrame
        """
        logger.info("Building fundamentals intervals...")
        
        # Step 1: Versioning
        versioned = self.versioning.build_versioned_fundamentals(fundamentals_df)
        
        # Step 2: Intervals
        intervals = self.interval_builder.build_intervals(versioned)
        
        logger.info(f"Built fundamentals intervals: {len(intervals)} rows")
        return intervals
    
    def join_to_daily_panel(
        self,
        daily_df: pd.DataFrame,
        intervals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join fundamentals intervals to daily panel using interval join.
        
        Args:
            daily_df: Daily panel (symbol, date)
            intervals_df: Fundamentals intervals
        
        Returns:
            Daily panel with fundamentals joined
        """
        logger.info(f"Joining fundamentals to daily panel ({len(daily_df)} rows)...")
        
        if self.join_method == 'polars':
            result = self._join_polars(daily_df, intervals_df)
        elif self.join_method == 'duckdb':
            result = self._join_duckdb(daily_df, intervals_df)
        else:
            raise ValueError(f"Invalid join_method: {self.join_method}")
        
        # Add source timestamps
        result = self._add_source_timestamps(result)
        
        # Handle staleness
        result = self._handle_staleness(result)
        
        logger.info(f"Joined fundamentals: {len(result)} rows")
        return result
    
    def _join_polars(
        self,
        daily_df: pd.DataFrame,
        intervals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join using Polars join_asof.
        
        Args:
            daily_df: Daily panel
            intervals_df: Fundamentals intervals
        
        Returns:
            Joined DataFrame
        """
        # Convert to Polars
        daily_pl = pl.from_pandas(daily_df)
        intervals_pl = pl.from_pandas(intervals_df)
        
        # Ensure datetime types
        daily_pl = daily_pl.with_columns([
            pl.col('date').cast(pl.Date)
        ])
        
        intervals_pl = intervals_pl.with_columns([
            pl.col('valid_from').cast(pl.Date),
            pl.col('valid_to').cast(pl.Date)
        ])
        
        # Join asof on valid_from
        joined = daily_pl.join_asof(
            intervals_pl,
            left_on='date',
            right_on='valid_from',
            by='symbol',
            strategy='backward'
        )
        
        # Filter to ensure date <= valid_to
        joined = joined.filter(
            pl.col('date') <= pl.col('valid_to')
        )
        
        # Convert back to pandas
        result = joined.to_pandas()
        
        return result
    
    def _join_duckdb(
        self,
        daily_df: pd.DataFrame,
        intervals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join using DuckDB interval join.
        
        Args:
            daily_df: Daily panel
            intervals_df: Fundamentals intervals
        
        Returns:
            Joined DataFrame
        """
        import duckdb
        
        # Create DuckDB connection
        conn = duckdb.connect(':memory:')
        
        # Register DataFrames
        conn.register('daily', daily_df)
        conn.register('fundamentals', intervals_df)
        
        # Interval join query
        query = """
        SELECT d.*, f.*, 
               f.effective_from AS source_ts_fund
        FROM daily d
        LEFT JOIN fundamentals f
          ON d.symbol = f.symbol
         AND d.date BETWEEN f.valid_from AND f.valid_to
        """
        
        result = conn.execute(query).df()
        
        conn.close()
        
        return result
    
    def _add_source_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add source timestamps.
        
        Args:
            df: Joined DataFrame
        
        Returns:
            DataFrame with source timestamps
        """
        result = df.copy()
        
        # source_ts_price = date (or upstream price timestamp if tracked)
        if 'source_ts_price' not in result.columns:
            result['source_ts_price'] = pd.to_datetime(result['date'])
        
        # source_ts_fund = effective_from (already added in join)
        if 'source_ts_fund' not in result.columns and 'effective_from' in result.columns:
            result['source_ts_fund'] = result['effective_from']
        
        return result
    
    def _handle_staleness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle stale fundamentals.
        
        If date - valid_from > stale_max_days:
        - Set fundamental features to NaN
        - Set is_stale_fund = 1
        
        Args:
            df: DataFrame with fundamentals
        
        Returns:
            DataFrame with staleness handled
        """
        result = df.copy()
        
        # Compute days since fundamental became valid
        result['days_since_fund'] = (
            pd.to_datetime(result['date']) - pd.to_datetime(result['valid_from'])
        ).dt.days
        
        # Identify stale rows
        is_stale = result['days_since_fund'] > self.stale_max_days
        
        # Set is_stale_fund flag
        result['is_stale_fund'] = is_stale.astype(int)
        
        # Null out fundamental columns for stale rows
        if is_stale.any():
            fundamental_cols = self._get_fundamental_cols(result)
            
            result.loc[is_stale, fundamental_cols] = np.nan
            
            logger.info(
                f"Nulled {is_stale.sum()} stale rows "
                f"(>{self.stale_max_days} days old)"
            )
        
        return result
    
    def _get_fundamental_cols(self, df: pd.DataFrame) -> List[str]:
        """Get list of fundamental columns to null out when stale."""
        exclude_cols = {
            'symbol', 'date', 'statement_type', 'period_end', 'filing_date',
            'updated_at', 'as_of_date', 'effective_from', 'valid_from', 'valid_to',
            'version_id', 'source_ts_price', 'source_ts_fund', 'is_stale_fund',
            'days_since_fund', 'report_currency', 'audited'
        }
        
        return [col for col in df.columns if col not in exclude_cols]
    
    def validate_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that no future data leakage exists.
        
        Checks:
        - max(source_ts_price, source_ts_fund) <= date for all rows
        
        Args:
            df: Daily panel with fundamentals
        
        Returns:
            Dict with validation results
        """
        violations = []
        
        # Check source_ts_price <= date
        if 'source_ts_price' in df.columns:
            price_violations = df['source_ts_price'] > df['date']
            if price_violations.any():
                violations.append({
                    'type': 'source_ts_price > date',
                    'count': price_violations.sum(),
                    'sample': df[price_violations].head(5)[[
                        'symbol', 'date', 'source_ts_price'
                    ]].to_dict('records')
                })
        
        # Check source_ts_fund <= date
        if 'source_ts_fund' in df.columns:
            fund_violations = df['source_ts_fund'] > df['date']
            if fund_violations.any():
                violations.append({
                    'type': 'source_ts_fund > date',
                    'count': fund_violations.sum(),
                    'sample': df[fund_violations].head(5)[[
                        'symbol', 'date', 'source_ts_fund'
                    ]].to_dict('records')
                })
        
        is_valid = len(violations) == 0
        
        if is_valid:
            logger.info("✓ No leakage violations detected")
        else:
            total_violations = sum(v['count'] for v in violations)
            logger.error(f"✗ Found {total_violations} leakage violations")
        
        return {
            'valid': is_valid,
            'violations': violations,
            'total_violations': sum(v['count'] for v in violations) if violations else 0
        }
    
    def process(
        self,
        fundamentals_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete PIT processing pipeline.
        
        Args:
            fundamentals_df: Raw fundamentals
            daily_df: Daily panel (symbol, date)
            output_dir: Directory to save outputs
        
        Returns:
            Dict with 'intervals' and 'daily_panel' DataFrames
        """
        logger.info("Starting PIT fundamentals processing...")
        
        # Step 1: Build intervals
        intervals = self.build_fundamentals_intervals(fundamentals_df)
        
        # Step 2: Join to daily panel
        daily_panel = self.join_to_daily_panel(daily_df, intervals)
        
        # Step 3: Apply precedence policy
        if self.config['precedence'] in ['quarter_over_annual', 'both_suffixes']:
            daily_panel = self.precedence.apply_precedence(daily_panel)
        
        # Step 4: Validate leakage
        leakage_result = self.validate_leakage(daily_panel)
        
        if not leakage_result['valid'] and self.config.get('validation', {}).get('fail_on_violations', True):
            raise ValueError(
                f"Leakage validation failed: {leakage_result['total_violations']} violations"
            )
        
        # Step 5: Save outputs
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            intervals_path = output_dir / 'fundamentals_intervals.parquet'
            daily_panel_path = output_dir / 'daily_panel.parquet'
            
            intervals.to_parquet(intervals_path, index=False)
            daily_panel.to_parquet(daily_panel_path, index=False)
            
            logger.info(f"Saved intervals to {intervals_path}")
            logger.info(f"Saved daily panel to {daily_panel_path}")
        
        logger.info("PIT fundamentals processing complete!")
        
        return {
            'intervals': intervals,
            'daily_panel': daily_panel
        }
