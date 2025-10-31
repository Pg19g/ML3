"""Compute financial ratios at filing level with TTM support."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class RatioComputer:
    """
    Compute financial ratios at filing level.
    
    Key principles:
    - Ratios computed when filing becomes available, not daily
    - TTM (Trailing Twelve Months) built only from filings <= that time
    - Handles missing data gracefully
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ratio computer.
        
        Args:
            config: Ratio configuration from pit_enhanced.yaml
        """
        self.config = config or {}
        self.ttm_enabled = self.config.get('ttm', {}).get('enabled', True)
        self.ttm_periods_required = self.config.get('ttm', {}).get('periods_required', 4)
    
    def compute_all_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all configured ratios.
        
        Args:
            df: Fundamentals DataFrame with raw fields
        
        Returns:
            DataFrame with ratios added
        """
        result = df.copy()
        
        logger.info("Computing ratios at filing level...")
        
        # Margin ratios
        if 'margins' in self.config:
            result = self._compute_margins(result)
        
        # Leverage ratios
        if 'leverage' in self.config:
            result = self._compute_leverage(result)
        
        # Coverage ratios
        if 'coverage' in self.config:
            result = self._compute_coverage(result)
        
        # Return ratios
        if 'returns' in self.config:
            result = self._compute_returns(result)
        
        # Accruals
        if 'accruals' in self.config:
            result = self._compute_accruals(result)
        
        # TTM ratios (if enabled)
        if self.ttm_enabled:
            result = self._compute_ttm_ratios(result)
        
        logger.info(f"Computed ratios for {len(result)} filings")
        return result
    
    def _compute_margins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute margin ratios."""
        result = df.copy()
        
        for ratio_config in self.config.get('margins', []):
            name = ratio_config['name']
            numerator = ratio_config['numerator']
            denominator = ratio_config['denominator']
            
            if numerator in result.columns and denominator in result.columns:
                result[name] = self._safe_divide(
                    result[numerator],
                    result[denominator]
                )
                logger.debug(f"Computed {name}")
        
        return result
    
    def _compute_leverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute leverage ratios."""
        result = df.copy()
        
        for ratio_config in self.config.get('leverage', []):
            name = ratio_config['name']
            numerator = ratio_config['numerator']
            denominator = ratio_config['denominator']
            
            if numerator in result.columns and denominator in result.columns:
                result[name] = self._safe_divide(
                    result[numerator],
                    result[denominator]
                )
                logger.debug(f"Computed {name}")
        
        return result
    
    def _compute_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute coverage ratios."""
        result = df.copy()
        
        for ratio_config in self.config.get('coverage', []):
            name = ratio_config['name']
            numerator = ratio_config['numerator']
            denominator = ratio_config['denominator']
            
            if numerator in result.columns and denominator in result.columns:
                result[name] = self._safe_divide(
                    result[numerator],
                    result[denominator]
                )
                logger.debug(f"Computed {name}")
        
        return result
    
    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute return ratios."""
        result = df.copy()
        
        for ratio_config in self.config.get('returns', []):
            name = ratio_config['name']
            numerator = ratio_config['numerator']
            denominator = ratio_config['denominator']
            
            # Special handling for ROIC (needs InvestedCapital)
            if name == 'roic' and denominator == 'InvestedCapital':
                if 'TotalAssets' in result.columns and 'CurrentLiabilities' in result.columns:
                    result['InvestedCapital'] = (
                        result['TotalAssets'] - result['CurrentLiabilities']
                    )
            
            if numerator in result.columns and denominator in result.columns:
                result[name] = self._safe_divide(
                    result[numerator],
                    result[denominator]
                )
                logger.debug(f"Computed {name}")
        
        return result
    
    def _compute_accruals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute accrual ratios."""
        result = df.copy()
        
        for ratio_config in self.config.get('accruals', []):
            name = ratio_config['name']
            numerator = ratio_config['numerator']
            denominator = ratio_config['denominator']
            
            # Special handling for accruals (NetIncome - OperatingCashFlow)
            if numerator == 'NetIncome_minus_OperatingCashFlow':
                if 'NetIncome' in result.columns and 'OperatingCashFlow' in result.columns:
                    result['NetIncome_minus_OperatingCashFlow'] = (
                        result['NetIncome'] - result['OperatingCashFlow']
                    )
            
            if numerator in result.columns and denominator in result.columns:
                result[name] = self._safe_divide(
                    result[numerator],
                    result[denominator]
                )
                logger.debug(f"Computed {name}")
        
        return result
    
    def _compute_ttm_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute TTM (Trailing Twelve Months) ratios.
        
        For quarterly data, sum last 4 quarters for income statement items,
        use most recent for balance sheet items.
        
        Args:
            df: DataFrame with fundamentals
        
        Returns:
            DataFrame with TTM ratios added
        """
        result = df.copy()
        
        # Only compute TTM for quarterly data
        quarterly = result[result['statement_type'] == 'quarterly'].copy()
        
        if len(quarterly) == 0:
            logger.info("No quarterly data for TTM computation")
            return result
        
        # Sort by symbol and period_end
        quarterly = quarterly.sort_values(['symbol', 'period_end'])
        
        # Income statement items to sum (TTM)
        income_items = [
            'TotalRevenue', 'GrossProfit', 'OperatingIncome', 'NetIncome',
            'OperatingCashFlow', 'InterestExpense'
        ]
        
        # Compute rolling sum for income items
        for item in income_items:
            if item in quarterly.columns:
                quarterly[f'{item}_ttm'] = quarterly.groupby('symbol')[item].transform(
                    lambda x: x.rolling(window=self.ttm_periods_required, min_periods=self.ttm_periods_required).sum()
                )
        
        # Balance sheet items use most recent value (no summing)
        balance_items = [
            'TotalAssets', 'TotalStockholdersEquity', 'TotalDebt',
            'CurrentLiabilities'
        ]
        
        for item in balance_items:
            if item in quarterly.columns:
                quarterly[f'{item}_ttm'] = quarterly[item]
        
        # Compute TTM ratios
        # Gross margin TTM
        if 'GrossProfit_ttm' in quarterly.columns and 'TotalRevenue_ttm' in quarterly.columns:
            quarterly['gross_margin_ttm'] = self._safe_divide(
                quarterly['GrossProfit_ttm'],
                quarterly['TotalRevenue_ttm']
            )
        
        # Operating margin TTM
        if 'OperatingIncome_ttm' in quarterly.columns and 'TotalRevenue_ttm' in quarterly.columns:
            quarterly['operating_margin_ttm'] = self._safe_divide(
                quarterly['OperatingIncome_ttm'],
                quarterly['TotalRevenue_ttm']
            )
        
        # Net margin TTM
        if 'NetIncome_ttm' in quarterly.columns and 'TotalRevenue_ttm' in quarterly.columns:
            quarterly['net_margin_ttm'] = self._safe_divide(
                quarterly['NetIncome_ttm'],
                quarterly['TotalRevenue_ttm']
            )
        
        # ROE TTM
        if 'NetIncome_ttm' in quarterly.columns and 'TotalStockholdersEquity_ttm' in quarterly.columns:
            quarterly['roe_ttm'] = self._safe_divide(
                quarterly['NetIncome_ttm'],
                quarterly['TotalStockholdersEquity_ttm']
            )
        
        # ROA TTM
        if 'NetIncome_ttm' in quarterly.columns and 'TotalAssets_ttm' in quarterly.columns:
            quarterly['roa_ttm'] = self._safe_divide(
                quarterly['NetIncome_ttm'],
                quarterly['TotalAssets_ttm']
            )
        
        # Merge TTM columns back to result
        ttm_cols = [col for col in quarterly.columns if col.endswith('_ttm')]
        
        if ttm_cols:
            # Update quarterly rows in result
            result.loc[result['statement_type'] == 'quarterly', ttm_cols] = quarterly[ttm_cols].values
            
            logger.info(f"Computed {len(ttm_cols)} TTM metrics")
        
        return result
    
    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        Safe division handling zeros and NaNs.
        
        Args:
            numerator: Numerator series
            denominator: Denominator series
        
        Returns:
            Ratio series with NaN for invalid divisions
        """
        # Replace zeros in denominator with NaN to avoid division by zero
        denom_safe = denominator.replace(0, np.nan)
        
        # Perform division
        ratio = numerator / denom_safe
        
        return ratio
    
    def get_ratio_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about ratio coverage.
        
        Args:
            df: DataFrame with ratios
        
        Returns:
            Dict with coverage statistics
        """
        stats = {}
        
        # Find ratio columns
        ratio_cols = []
        for category in ['margins', 'leverage', 'coverage', 'returns', 'accruals']:
            if category in self.config:
                ratio_cols.extend([r['name'] for r in self.config[category]])
        
        # Add TTM ratios
        ttm_ratio_cols = [col for col in df.columns if col.endswith('_ttm') and 'ratio' in col or 'margin' in col or 'roe' in col or 'roa' in col]
        ratio_cols.extend(ttm_ratio_cols)
        
        # Compute coverage for each ratio
        for col in ratio_cols:
            if col in df.columns:
                total = len(df)
                non_null = df[col].notna().sum()
                coverage = non_null / total if total > 0 else 0
                
                stats[col] = {
                    'coverage': coverage,
                    'non_null': non_null,
                    'total': total
                }
        
        return stats
