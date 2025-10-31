"""Backtesting framework for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from src.utils import setup_logging, load_config, get_config_path

logger = setup_logging(__name__)


class Backtester:
    """Backtest trading strategies based on model predictions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize backtester.
        
        Args:
            config_path: Path to backtest.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("backtest.yaml"))
        
        self.config = load_config(config_path)
        self.strategy_config = self.config.get('strategy', {})
        self.costs_config = self.config.get('costs', {})
        self.risk_config = self.config.get('risk', {})
    
    def rank_based_strategy(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'prediction',
        long_top_pct: float = 0.1,
        short_bottom_pct: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate positions based on cross-sectional ranking.
        
        Args:
            df: DataFrame with predictions (must have date, symbol, prediction)
            prediction_col: Column name for predictions
            long_top_pct: Percentage of top-ranked stocks to long
            short_bottom_pct: Percentage of bottom-ranked stocks to short
            
        Returns:
            DataFrame with positions
        """
        result = df.copy()
        
        # Rank predictions within each date
        result['rank'] = result.groupby('date')[prediction_col].rank(
            ascending=False,
            pct=True
        )
        
        # Generate positions
        result['position'] = 0.0
        
        # Long positions (top percentile)
        long_mask = result['rank'] <= long_top_pct
        n_long = result[long_mask].groupby('date').size()
        result.loc[long_mask, 'position'] = 1.0
        
        # Short positions (bottom percentile)
        if short_bottom_pct > 0:
            short_mask = result['rank'] >= (1 - short_bottom_pct)
            n_short = result[short_mask].groupby('date').size()
            result.loc[short_mask, 'position'] = -1.0
        
        # Normalize positions (equal weight)
        weighting = self.strategy_config.get('weighting', 'equal')
        if weighting == 'equal':
            result['weight'] = result.groupby('date')['position'].transform(
                lambda x: x / x.abs().sum() if x.abs().sum() > 0 else 0
            )
        else:
            result['weight'] = result['position']
        
        return result
    
    def compute_returns(
        self,
        df: pd.DataFrame,
        return_col: str = 'ret_1d_fwd'
    ) -> pd.DataFrame:
        """
        Compute portfolio returns.
        
        Args:
            df: DataFrame with positions and forward returns
            return_col: Column name for forward returns
            
        Returns:
            DataFrame with portfolio returns
        """
        result = df.copy()
        
        # Strategy return = weight * forward_return
        result['strategy_return'] = result['weight'] * result[return_col]
        
        # Aggregate to portfolio level
        portfolio_returns = result.groupby('date').agg({
            'strategy_return': 'sum',
            'weight': lambda x: x.abs().sum()  # Gross exposure
        }).reset_index()
        
        portfolio_returns = portfolio_returns.rename(columns={
            'weight': 'gross_exposure'
        })
        
        return portfolio_returns
    
    def apply_transaction_costs(
        self,
        df: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply transaction costs based on turnover.
        
        Args:
            df: DataFrame with positions
            prev_weights: Previous period weights
            
        Returns:
            DataFrame with transaction costs
        """
        result = df.copy()
        
        if prev_weights is None:
            # First period - full turnover
            result['turnover'] = result['weight'].abs()
        else:
            # Compute change in weights
            prev_weights_aligned = prev_weights.reindex(result.index, fill_value=0)
            result['turnover'] = (result['weight'] - prev_weights_aligned).abs()
        
        # Apply costs
        commission_pct = self.costs_config.get('commission_pct', 0.001)
        slippage_pct = self.costs_config.get('slippage_pct', 0.0005)
        
        total_cost_pct = commission_pct + slippage_pct
        result['transaction_cost'] = result['turnover'] * total_cost_pct
        
        return result
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        prediction_col: str = 'prediction',
        return_col: str = 'ret_1d_fwd'
    ) -> Dict[str, Any]:
        """
        Run complete backtest.
        
        Args:
            df: DataFrame with predictions and returns
            prediction_col: Column name for predictions
            return_col: Column name for forward returns
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest...")
        
        # Generate positions
        long_top_pct = self.strategy_config.get('long_top_pct', 0.1)
        short_bottom_pct = self.strategy_config.get('short_bottom_pct', 0.0)
        
        df_with_positions = self.rank_based_strategy(
            df,
            prediction_col,
            long_top_pct,
            short_bottom_pct
        )
        
        # Compute returns by date
        all_portfolio_returns = []
        dates = sorted(df_with_positions['date'].unique())
        prev_weights = None
        
        for date in dates:
            date_data = df_with_positions[df_with_positions['date'] == date].copy()
            
            # Apply transaction costs
            date_data = self.apply_transaction_costs(date_data, prev_weights)
            
            # Compute returns
            portfolio_return = (date_data['weight'] * date_data[return_col]).sum()
            transaction_cost = date_data['transaction_cost'].sum()
            net_return = portfolio_return - transaction_cost
            
            all_portfolio_returns.append({
                'date': date,
                'gross_return': portfolio_return,
                'transaction_cost': transaction_cost,
                'net_return': net_return,
                'gross_exposure': date_data['weight'].abs().sum(),
                'n_positions': (date_data['weight'] != 0).sum()
            })
            
            # Update prev_weights for next iteration
            prev_weights = date_data.set_index('symbol')['weight']
        
        portfolio_df = pd.DataFrame(all_portfolio_returns)
        portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
        
        # Compute equity curve
        portfolio_df['cumulative_return'] = (1 + portfolio_df['net_return']).cumprod()
        
        # Compute metrics
        metrics = self.compute_metrics(portfolio_df)
        
        results = {
            'portfolio_returns': portfolio_df,
            'metrics': metrics,
            'positions': df_with_positions
        }
        
        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2%}")
        return results
    
    def compute_metrics(
        self,
        portfolio_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            portfolio_df: DataFrame with portfolio returns
            
        Returns:
            Dictionary of metrics
        """
        returns = portfolio_df['net_return']
        
        metrics = {}
        
        # Total return
        metrics['total_return'] = portfolio_df['cumulative_return'].iloc[-1] - 1
        
        # CAGR
        n_days = len(portfolio_df)
        n_years = n_days / 252
        metrics['cagr'] = (1 + metrics['total_return']) ** (1 / n_years) - 1
        
        # Volatility (annualized)
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        metrics['sharpe_ratio'] = (
            metrics['cagr'] / metrics['volatility']
            if metrics['volatility'] > 0 else 0
        )
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = (
            metrics['cagr'] / downside_std
            if downside_std > 0 else 0
        )
        
        # Max drawdown
        cumulative = portfolio_df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Calmar ratio
        metrics['calmar_ratio'] = (
            metrics['cagr'] / abs(metrics['max_drawdown'])
            if metrics['max_drawdown'] != 0 else 0
        )
        
        # Win rate
        metrics['win_rate'] = (returns > 0).mean()
        
        # Average turnover
        metrics['avg_turnover'] = portfolio_df['gross_exposure'].mean()
        
        # Average transaction cost
        metrics['avg_transaction_cost'] = portfolio_df['transaction_cost'].mean()
        
        return metrics
    
    def compare_to_benchmark(
        self,
        portfolio_df: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compare strategy to benchmark.
        
        Args:
            portfolio_df: DataFrame with portfolio returns
            benchmark_returns: Series with benchmark returns
            
        Returns:
            Dictionary of comparison metrics
        """
        strategy_returns = portfolio_df.set_index('date')['net_return']
        
        # Align dates
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        metrics = {}
        metrics['excess_return'] = excess_returns.sum()
        metrics['information_ratio'] = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0 else 0
        )
        
        # Beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Alpha
        rf_rate = 0.0  # Risk-free rate
        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = benchmark_returns.mean() * 252
        metrics['alpha'] = strategy_mean - (rf_rate + metrics['beta'] * (benchmark_mean - rf_rate))
        
        return metrics
