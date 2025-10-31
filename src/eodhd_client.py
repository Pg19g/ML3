"""EODHD API client with rate limiting and retry logic."""

import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import logging

from src.utils import setup_logging, load_config, get_env_variable, get_config_path

logger = setup_logging(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class EODHDClient:
    """Client for EODHD API with rate limiting and retry logic."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize EODHD client.
        
        Args:
            config_path: Path to eodhd.yaml config file
        """
        if config_path is None:
            config_path = str(get_config_path("eodhd.yaml"))
        
        self.config = load_config(config_path)
        self.base_url = self.config['base_url']
        self.api_key = get_env_variable(self.config['api_key_env'])
        
        # Rate limiting
        rate_limit = self.config['rate_limit']
        self.rate_limiter = RateLimiter(rate_limit['calls_per_second'])
        self.retry_attempts = rate_limit['retry_attempts']
        self.retry_backoff = rate_limit['retry_backoff_factor']
        
        self.session = requests.Session()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        self.rate_limiter.wait()
        
        if params is None:
            params = {}
        
        params['api_token'] = self.api_key
        params['fmt'] = self.config['request_params']['fmt']
        
        url = f"{self.base_url}{endpoint}"
        
        logger.debug(f"Making request to {url}")
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def get_eod_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get end-of-day prices for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = self.config['endpoints']['eod_prices'].format(symbol=symbol)
        
        params = {
            'order': 'd',
            'fmt': 'json'
        }
        
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        try:
            data = self._make_request(endpoint, params)
            
            if not data:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Standardize column names
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjusted_close': 'AdjClose',
                'volume': 'Volume'
            })
            
            df['Date'] = pd.to_datetime(df['Date'])
            df['Symbol'] = symbol
            
            # Select and order columns
            cols = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
            df = df[cols]
            
            logger.info(f"Retrieved {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamentals(
        self,
        symbol: str,
        statement_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')
            statement_type: Type of statement ('all', 'income', 'balance', 'cash')
            
        Returns:
            Dictionary with fundamental data
        """
        endpoint = self.config['endpoints']['fundamentals'].format(symbol=symbol)
        
        params = {}
        if statement_type != "all":
            params['filter'] = statement_type
        
        try:
            data = self._make_request(endpoint, params)
            logger.info(f"Retrieved fundamentals for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def get_bulk_prices(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get prices for multiple symbols.
        
        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Combined DataFrame with all symbols
        """
        all_data = []
        
        for symbol in symbols:
            logger.info(f"Fetching prices for {symbol}")
            df = self.get_eod_prices(symbol, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        logger.info(f"Retrieved total {len(combined)} rows for {len(symbols)} symbols")
        return combined
    
    def parse_fundamentals_to_dataframe(
        self,
        fundamentals_data: Dict[str, Any],
        symbol: str
    ) -> pd.DataFrame:
        """
        Parse fundamentals JSON to DataFrame format.
        
        Args:
            fundamentals_data: Raw fundamentals data from API
            symbol: Stock symbol
            
        Returns:
            DataFrame with fundamental data rows
        """
        rows = []
        
        # Extract quarterly and annual financials
        for statement_type in ['Financials']:
            if statement_type not in fundamentals_data:
                continue
            
            financials = fundamentals_data[statement_type]
            
            for period_type in ['quarterly', 'yearly']:
                if period_type not in financials:
                    continue
                
                statements = financials[period_type]
                
                for date_key, values in statements.items():
                    if not isinstance(values, dict):
                        continue
                    
                    row = {
                        'symbol': symbol,
                        'period_end': date_key,
                        'statement_type': period_type,
                        'filing_date': values.get('filing_date', None),
                    }
                    
                    # Add all financial fields
                    for key, value in values.items():
                        if key != 'filing_date':
                            row[key] = value
                    
                    rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df['period_end'] = pd.to_datetime(df['period_end'])
        
        if 'filing_date' in df.columns:
            df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        
        return df
