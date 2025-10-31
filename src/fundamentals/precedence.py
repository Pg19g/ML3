"""Quarterly vs Annual precedence policy handling."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class PrecedencePolicy:
    """
    Handles quarterly vs annual precedence policies.
    
    Two modes:
    1. quarter_over_annual: Quarterly overrides annual when both exist;
       fallback to annual when quarterly missing/stale
    2. both_suffixes: Expose both with _q and _y suffixes; no auto-override
    """
    
    def __init__(self, policy: str = 'quarter_over_annual'):
        """
        Initialize precedence policy.
        
        Args:
            policy: 'quarter_over_annual' or 'both_suffixes'
        """
        if policy not in ['quarter_over_annual', 'both_suffixes']:
            raise ValueError(
                f"Invalid policy: {policy}. "
                f"Must be 'quarter_over_annual' or 'both_suffixes'"
            )
        
        self.policy = policy
        logger.info(f"Precedence policy: {policy}")
    
    def apply_precedence(
        self,
        df: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply precedence policy to fundamentals data.
        
        Args:
            df: DataFrame with fundamentals (must have statement_type column)
            fundamental_cols: List of fundamental columns to apply policy to
                             If None, auto-detect
        
        Returns:
            DataFrame with precedence policy applied
        """
        if self.policy == 'quarter_over_annual':
            return self._apply_quarter_over_annual(df, fundamental_cols)
        else:  # both_suffixes
            return self._apply_both_suffixes(df, fundamental_cols)
    
    def _apply_quarter_over_annual(
        self,
        df: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply quarter_over_annual policy.
        
        For each (symbol, date):
        - If quarterly data exists and is not stale, use it
        - Otherwise, use annual data if available
        
        Args:
            df: DataFrame with fundamentals
            fundamental_cols: Columns to apply policy to
        
        Returns:
            Resolved DataFrame with single set of fundamental columns
        """
        result = df.copy()
        
        # Auto-detect fundamental columns if not provided
        if fundamental_cols is None:
            fundamental_cols = self._detect_fundamental_cols(result)
        
        # Separate quarterly and annual data
        quarterly = result[result['statement_type'] == 'quarterly'].copy()
        annual = result[result['statement_type'] == 'annual'].copy()
        
        if len(quarterly) == 0 and len(annual) == 0:
            logger.warning("No quarterly or annual data found")
            return result
        
        # Create resolved view
        # Start with quarterly data
        if len(quarterly) > 0:
            resolved = quarterly.copy()
            
            # For dates without quarterly data, add annual data
            if len(annual) > 0:
                # Find (symbol, date) pairs in annual but not in quarterly
                quarterly_keys = set(quarterly[['symbol', 'date']].apply(tuple, axis=1))
                annual_only = annual[
                    ~annual[['symbol', 'date']].apply(tuple, axis=1).isin(quarterly_keys)
                ]
                
                if len(annual_only) > 0:
                    resolved = pd.concat([resolved, annual_only], ignore_index=True)
                    logger.info(
                        f"Added {len(annual_only)} annual-only rows as fallback"
                    )
        else:
            # No quarterly data, use annual only
            resolved = annual.copy()
        
        # Sort by symbol and date
        resolved = resolved.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info(
            f"Resolved {len(resolved)} rows using quarter_over_annual policy "
            f"(quarterly: {len(quarterly)}, annual: {len(annual)})"
        )
        
        return resolved
    
    def _apply_both_suffixes(
        self,
        df: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply both_suffixes policy.
        
        Exposes both quarterly and annual data with _q and _y suffixes.
        
        Args:
            df: DataFrame with fundamentals
            fundamental_cols: Columns to apply policy to
        
        Returns:
            DataFrame with both sets of columns (_q and _y suffixes)
        """
        result = df.copy()
        
        # Auto-detect fundamental columns if not provided
        if fundamental_cols is None:
            fundamental_cols = self._detect_fundamental_cols(result)
        
        # Separate quarterly and annual data
        quarterly = result[result['statement_type'] == 'quarterly'].copy()
        annual = result[result['statement_type'] == 'annual'].copy()
        
        # Rename fundamental columns with suffixes
        quarterly_renamed = quarterly.rename(
            columns={col: f"{col}_q" for col in fundamental_cols}
        )
        annual_renamed = annual.rename(
            columns={col: f"{col}_y" for col in fundamental_cols}
        )
        
        # Merge on (symbol, date)
        key_cols = ['symbol', 'date']
        
        # Keep only key columns and renamed fundamental columns from each
        quarterly_subset = quarterly_renamed[
            key_cols + [f"{col}_q" for col in fundamental_cols]
        ]
        annual_subset = annual_renamed[
            key_cols + [f"{col}_y" for col in fundamental_cols]
        ]
        
        # Outer join to get both
        resolved = quarterly_subset.merge(
            annual_subset,
            on=key_cols,
            how='outer'
        )
        
        logger.info(
            f"Created both_suffixes view with {len(resolved)} rows "
            f"({len(fundamental_cols)} cols × 2 suffixes)"
        )
        
        return resolved
    
    def _detect_fundamental_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect fundamental columns.
        
        Excludes metadata columns like symbol, date, statement_type, etc.
        
        Args:
            df: DataFrame
        
        Returns:
            List of fundamental column names
        """
        exclude_cols = {
            'symbol', 'date', 'statement_type', 'period_end', 'filing_date',
            'updated_at', 'as_of_date', 'effective_from', 'valid_from', 'valid_to',
            'version_id', 'source_ts_price', 'source_ts_fund', 'is_stale_fund',
            'days_since_fund', 'report_currency', 'audited'
        }
        
        fundamental_cols = [
            col for col in df.columns
            if col not in exclude_cols and not col.endswith('_q') and not col.endswith('_y')
        ]
        
        logger.debug(f"Detected {len(fundamental_cols)} fundamental columns")
        return fundamental_cols
    
    def validate_precedence(
        self,
        df: pd.DataFrame,
        fundamental_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate that precedence policy is correctly applied.
        
        Args:
            df: DataFrame with precedence applied
            fundamental_cols: Columns to validate
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        if self.policy == 'quarter_over_annual':
            # Check that we don't have both quarterly and annual for same (symbol, date)
            if 'statement_type' in df.columns:
                duplicates = df.duplicated(subset=['symbol', 'date'], keep=False)
                
                if duplicates.any():
                    issues.append({
                        'issue': f'Found {duplicates.sum()} duplicate (symbol, date) pairs',
                        'severity': 'ERROR'
                    })
        
        elif self.policy == 'both_suffixes':
            # Check that _q and _y columns exist
            if fundamental_cols is None:
                fundamental_cols = self._detect_fundamental_cols(df)
            
            for col in fundamental_cols:
                if f"{col}_q" not in df.columns:
                    issues.append({
                        'issue': f'Missing quarterly column: {col}_q',
                        'severity': 'WARNING'
                    })
                
                if f"{col}_y" not in df.columns:
                    issues.append({
                        'issue': f'Missing annual column: {col}_y',
                        'severity': 'WARNING'
                    })
        
        return {
            'valid': len([i for i in issues if i['severity'] == 'ERROR']) == 0,
            'issues': issues
        }
