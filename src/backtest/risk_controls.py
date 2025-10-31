"""Risk controls for backtesting: liquidity filters, position limits, risk metrics."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class RiskControls:
    """
    Risk controls for backtesting.
    
    Features:
    - Liquidity filters (min volume, min price)
    - Position limits (max position size, max concentration)
    - Risk metrics (volatility, drawdown, VaR)
    - Exposure limits (gross, net, leverage)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk controls.
        
        Args:
            config: Risk configuration dict
        """
        self.config = config or {}
        
        # Liquidity filters
        self.min_volume = self.config.get('min_volume', 100000)
        self.min_price = self.config.get('min_price', 5.0)
        self.min_market_cap = self.config.get('min_market_cap', None)
        
        # Position limits
        self.max_position_size = self.config.get('max_position_size', 0.05)  # 5% of portfolio
        self.max_concentration = self.config.get('max_concentration', 0.20)  # 20% in single stock
        self.max_sector_exposure = self.config.get('max_sector_exposure', 0.30)  # 30% per sector
        
        # Exposure limits
        self.max_gross_exposure = self.config.get('max_gross_exposure', 2.0)  # 200%
        self.max_net_exposure = self.config.get('max_net_exposure', 1.0)  # 100%
        self.max_leverage = self.config.get('max_leverage', 2.0)
        
        # Risk metrics
        self.var_confidence = self.config.get('var_confidence', 0.95)
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.20)  # 20%
        
        logger.info(f"RiskControls initialized with config: {self.config}")
    
    def apply_liquidity_filters(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume',
        price_col: str = 'close',
        market_cap_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply liquidity filters to remove illiquid stocks.
        
        Args:
            df: DataFrame with stock data
            volume_col: Column name for volume
            price_col: Column name for price
            market_cap_col: Column name for market cap (optional)
        
        Returns:
            Filtered DataFrame
        """
        result = df.copy()
        initial_count = len(result)
        
        # Filter by volume
        if volume_col in result.columns:
            result = result[result[volume_col] >= self.min_volume]
            logger.debug(f"Volume filter: {initial_count} -> {len(result)} rows")
        
        # Filter by price
        if price_col in result.columns:
            result = result[result[price_col] >= self.min_price]
            logger.debug(f"Price filter: {initial_count} -> {len(result)} rows")
        
        # Filter by market cap
        if self.min_market_cap and market_cap_col and market_cap_col in result.columns:
            result = result[result[market_cap_col] >= self.min_market_cap]
            logger.debug(f"Market cap filter: {initial_count} -> {len(result)} rows")
        
        removed = initial_count - len(result)
        if removed > 0:
            logger.info(f"Liquidity filters removed {removed} rows ({removed/initial_count*100:.1f}%)")
        
        return result
    
    def apply_position_limits(
        self,
        df: pd.DataFrame,
        weight_col: str = 'weight',
        symbol_col: str = 'symbol',
        date_col: str = 'date',
        sector_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply position limits to cap individual positions.
        
        Args:
            df: DataFrame with positions
            weight_col: Column name for weights
            symbol_col: Column name for symbols
            date_col: Column name for dates
            sector_col: Column name for sectors (optional)
        
        Returns:
            DataFrame with capped positions
        """
        result = df.copy()
        
        # Cap individual position sizes
        result[weight_col] = result[weight_col].clip(
            -self.max_position_size,
            self.max_position_size
        )
        
        # Check concentration per symbol
        for date in result[date_col].unique():
            date_mask = result[date_col] == date
            date_df = result[date_mask]
            
            # Cap concentration
            for symbol in date_df[symbol_col].unique():
                symbol_mask = date_mask & (result[symbol_col] == symbol)
                symbol_weight = result.loc[symbol_mask, weight_col].abs().sum()
                
                if symbol_weight > self.max_concentration:
                    # Scale down
                    scale_factor = self.max_concentration / symbol_weight
                    result.loc[symbol_mask, weight_col] *= scale_factor
        
        # Cap sector exposure (if sector column provided)
        if sector_col and sector_col in result.columns:
            for date in result[date_col].unique():
                date_mask = result[date_col] == date
                
                for sector in result[sector_col].unique():
                    sector_mask = date_mask & (result[sector_col] == sector)
                    sector_exposure = result.loc[sector_mask, weight_col].abs().sum()
                    
                    if sector_exposure > self.max_sector_exposure:
                        # Scale down
                        scale_factor = self.max_sector_exposure / sector_exposure
                        result.loc[sector_mask, weight_col] *= scale_factor
        
        return result
    
    def check_exposure_limits(
        self,
        df: pd.DataFrame,
        weight_col: str = 'weight',
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Check exposure limits.
        
        Args:
            df: DataFrame with positions
            weight_col: Column name for weights
            date_col: Column name for dates
        
        Returns:
            Dict with exposure metrics and violations
        """
        violations = []
        
        # Compute exposures by date
        exposures = df.groupby(date_col)[weight_col].agg([
            ('gross', lambda x: x.abs().sum()),
            ('net', lambda x: x.sum()),
            ('long', lambda x: x[x > 0].sum()),
            ('short', lambda x: x[x < 0].sum())
        ]).reset_index()
        
        # Check gross exposure
        gross_violations = exposures[exposures['gross'] > self.max_gross_exposure]
        if len(gross_violations) > 0:
            violations.append({
                'type': 'gross_exposure',
                'count': len(gross_violations),
                'max_value': gross_violations['gross'].max(),
                'limit': self.max_gross_exposure
            })
        
        # Check net exposure
        net_violations = exposures[exposures['net'].abs() > self.max_net_exposure]
        if len(net_violations) > 0:
            violations.append({
                'type': 'net_exposure',
                'count': len(net_violations),
                'max_value': net_violations['net'].abs().max(),
                'limit': self.max_net_exposure
            })
        
        return {
            'exposures': exposures,
            'violations': violations,
            'has_violations': len(violations) > 0
        }
    
    def compute_risk_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> Dict[str, float]:
        """
        Compute risk metrics.
        
        Args:
            returns: Series of portfolio returns
            window: Rolling window for metrics (default 252 trading days)
        
        Returns:
            Dict with risk metrics
        """
        metrics = {}
        
        # Volatility (annualized)
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        metrics['sharpe'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Value at Risk (VaR)
        metrics['var_95'] = returns.quantile(1 - self.var_confidence)
        
        # Conditional VaR (CVaR / Expected Shortfall)
        var_threshold = returns.quantile(1 - self.var_confidence)
        metrics['cvar_95'] = returns[returns <= var_threshold].mean()
        
        # Calmar ratio (return / max drawdown)
        if metrics['max_drawdown'] < 0:
            metrics['calmar'] = (returns.mean() * 252) / abs(metrics['max_drawdown'])
        else:
            metrics['calmar'] = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            metrics['sortino'] = (returns.mean() * 252) / downside_std
        else:
            metrics['sortino'] = 0
        
        # Win rate
        metrics['win_rate'] = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return metrics
    
    def check_drawdown_limit(
        self,
        returns: pd.Series,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if drawdown exceeds threshold.
        
        Args:
            returns: Series of portfolio returns
            threshold: Drawdown threshold (default from config)
        
        Returns:
            (is_violated, current_drawdown)
        """
        if threshold is None:
            threshold = self.max_drawdown_threshold
        
        # Compute current drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        current_drawdown = drawdown.iloc[-1]
        
        is_violated = current_drawdown < -threshold
        
        return is_violated, current_drawdown
    
    def generate_risk_report(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
        weight_col: str = 'weight',
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            df: DataFrame with positions
            returns: Series of portfolio returns
            weight_col: Column name for weights
            date_col: Column name for dates
        
        Returns:
            Dict with complete risk report
        """
        report = {}
        
        # Risk metrics
        report['risk_metrics'] = self.compute_risk_metrics(returns)
        
        # Exposure check
        report['exposure_check'] = self.check_exposure_limits(df, weight_col, date_col)
        
        # Drawdown check
        is_violated, current_dd = self.check_drawdown_limit(returns)
        report['drawdown_check'] = {
            'is_violated': is_violated,
            'current_drawdown': current_dd,
            'threshold': self.max_drawdown_threshold
        }
        
        # Position statistics
        report['position_stats'] = {
            'avg_num_positions': df.groupby(date_col)[weight_col].apply(lambda x: (x != 0).sum()).mean(),
            'max_num_positions': df.groupby(date_col)[weight_col].apply(lambda x: (x != 0).sum()).max(),
            'avg_position_size': df[df[weight_col] != 0][weight_col].abs().mean(),
            'max_position_size': df[weight_col].abs().max()
        }
        
        # Overall assessment
        has_violations = (
            report['exposure_check']['has_violations'] or
            report['drawdown_check']['is_violated']
        )
        
        report['overall'] = {
            'has_violations': has_violations,
            'status': 'FAIL' if has_violations else 'PASS'
        }
        
        return report
