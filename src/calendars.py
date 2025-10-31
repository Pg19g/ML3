"""Trading calendar utilities using pandas_market_calendars."""

import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class TradingCalendar:
    """Trading calendar for handling business days and date operations."""
    
    def __init__(self, calendar_name: str = "NYSE"):
        """
        Initialize trading calendar.
        
        Args:
            calendar_name: Name of the market calendar (e.g., 'NYSE', 'NASDAQ')
        """
        self.calendar_name = calendar_name
        self.calendar = mcal.get_calendar(calendar_name)
        self._trading_days_cache = {}
        
    def get_trading_days(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DatetimeIndex:
        """
        Get trading days between start and end dates.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DatetimeIndex of trading days
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_key = (start_date, end_date)
        if cache_key in self._trading_days_cache:
            return self._trading_days_cache[cache_key]
        
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index
        
        self._trading_days_cache[cache_key] = trading_days
        return trading_days
    
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """Check if date is a trading day."""
        date_str = date.strftime("%Y-%m-%d")
        trading_days = self.get_trading_days(date_str, date_str)
        return len(trading_days) > 0
    
    def next_trading_day(
        self,
        date: pd.Timestamp,
        n: int = 1
    ) -> pd.Timestamp:
        """
        Get the next trading day(s) after given date.
        
        Args:
            date: Reference date
            n: Number of trading days forward
            
        Returns:
            Next trading day
        """
        start = date.strftime("%Y-%m-%d")
        end = (date + timedelta(days=n * 5)).strftime("%Y-%m-%d")  # Buffer
        
        trading_days = self.get_trading_days(start, end)
        
        # Find first trading day after date
        future_days = trading_days[trading_days > date]
        
        if len(future_days) < n:
            logger.warning(f"Not enough trading days after {date}")
            return future_days[-1] if len(future_days) > 0 else date
        
        return future_days[n - 1]
    
    def previous_trading_day(
        self,
        date: pd.Timestamp,
        n: int = 1
    ) -> pd.Timestamp:
        """
        Get the previous trading day(s) before given date.
        
        Args:
            date: Reference date
            n: Number of trading days backward
            
        Returns:
            Previous trading day
        """
        end = date.strftime("%Y-%m-%d")
        start = (date - timedelta(days=n * 5)).strftime("%Y-%m-%d")  # Buffer
        
        trading_days = self.get_trading_days(start, end)
        
        # Find last trading day before date
        past_days = trading_days[trading_days < date]
        
        if len(past_days) < n:
            logger.warning(f"Not enough trading days before {date}")
            return past_days[0] if len(past_days) > 0 else date
        
        return past_days[-n]
    
    def round_to_trading_day(
        self,
        date: pd.Timestamp,
        direction: str = "forward"
    ) -> pd.Timestamp:
        """
        Round date to nearest trading day.
        
        Args:
            date: Date to round
            direction: 'forward' or 'backward'
            
        Returns:
            Rounded trading day
        """
        if self.is_trading_day(date):
            return date
        
        if direction == "forward":
            return self.next_trading_day(date, n=1)
        else:
            return self.previous_trading_day(date, n=1)
    
    def trading_days_between(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> int:
        """Count trading days between two dates."""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        trading_days = self.get_trading_days(start_str, end_str)
        return len(trading_days)
    
    def add_trading_days(
        self,
        date: pd.Timestamp,
        n_days: int
    ) -> pd.Timestamp:
        """
        Add n trading days to date.
        
        Args:
            date: Starting date
            n_days: Number of trading days to add (can be negative)
            
        Returns:
            Resulting date
        """
        if n_days > 0:
            return self.next_trading_day(date, n=n_days)
        elif n_days < 0:
            return self.previous_trading_day(date, n=abs(n_days))
        else:
            return date
    
    def align_dates_to_trading_days(
        self,
        dates: pd.Series,
        direction: str = "forward"
    ) -> pd.Series:
        """
        Align a series of dates to trading days.
        
        Args:
            dates: Series of dates
            direction: 'forward' or 'backward'
            
        Returns:
            Series of aligned dates
        """
        return dates.apply(lambda d: self.round_to_trading_day(d, direction))


def get_calendar(calendar_name: str = "NYSE") -> TradingCalendar:
    """Get trading calendar instance."""
    return TradingCalendar(calendar_name)
