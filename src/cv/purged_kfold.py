"""PurgedKFold cross-validation with embargo for time-series data.

Based on Advances in Financial Machine Learning by Marcos Lopez de Prado.
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional
from sklearn.model_selection import KFold
import logging

from src.calendars import get_calendar
from src.utils import setup_logging

logger = setup_logging(__name__)


class PurgedKFold:
    """
    Purged K-Fold cross-validation with embargo for time-series data.
    
    Key features:
    - Purging: Remove training samples whose labels overlap with test set
    - Embargo: Add buffer period after test set to prevent leakage
    - Time-aware: Respects temporal ordering
    
    Based on Lopez de Prado (2018), Chapter 7.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 21,
        purge_days: int = 0,
        calendar_name: str = 'NYSE'
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            embargo_days: Trading days to embargo after test set
            purge_days: Trading days to purge before test set
            calendar_name: Trading calendar to use
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.calendar = get_calendar(calendar_name)
        
        logger.info(
            f"PurgedKFold: {n_splits} splits, "
            f"{embargo_days} day embargo, "
            f"{purge_days} day purge"
        )
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo.
        
        Args:
            X: Features DataFrame with DatetimeIndex
            y: Labels (optional, not used for splitting)
            groups: Group labels (optional, not used)
        
        Yields:
            (train_indices, test_indices) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        # Get unique dates
        dates = X.index.unique().sort_values()
        n_dates = len(dates)
        
        # Standard K-Fold on dates
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        for fold_idx, (train_date_idx, test_date_idx) in enumerate(kf.split(dates)):
            # Get train and test dates
            train_dates = dates[train_date_idx]
            test_dates = dates[test_date_idx]
            
            # Apply purging: remove train dates close to test dates
            if self.purge_days > 0:
                train_dates = self._purge_train_dates(
                    train_dates, test_dates, self.purge_days
                )
            
            # Apply embargo: remove train dates after test dates
            if self.embargo_days > 0:
                train_dates = self._embargo_train_dates(
                    train_dates, test_dates, self.embargo_days
                )
            
            # Convert dates to indices
            train_idx = X.index.isin(train_dates)
            test_idx = X.index.isin(test_dates)
            
            # Get integer indices
            train_indices = np.where(train_idx)[0]
            test_indices = np.where(test_idx)[0]
            
            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )
            
            yield train_indices, test_indices
    
    def _purge_train_dates(
        self,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        purge_days: int
    ) -> pd.DatetimeIndex:
        """
        Purge training dates that are too close to test dates.
        
        Args:
            train_dates: Training dates
            test_dates: Test dates
            purge_days: Number of trading days to purge
        
        Returns:
            Purged training dates
        """
        if purge_days == 0:
            return train_dates
        
        # Find test date boundaries
        test_start = test_dates.min()
        test_end = test_dates.max()
        
        # Compute purge boundaries
        purge_start = self.calendar.subtract_trading_days(test_start, purge_days)
        purge_end = self.calendar.add_trading_days(test_end, purge_days)
        
        # Remove train dates in purge window
        purged = train_dates[
            (train_dates < purge_start) | (train_dates > purge_end)
        ]
        
        removed = len(train_dates) - len(purged)
        if removed > 0:
            logger.debug(f"Purged {removed} training dates")
        
        return purged
    
    def _embargo_train_dates(
        self,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        embargo_days: int
    ) -> pd.DatetimeIndex:
        """
        Embargo training dates after test set.
        
        Args:
            train_dates: Training dates
            test_dates: Test dates
            embargo_days: Number of trading days to embargo
        
        Returns:
            Embargoed training dates
        """
        if embargo_days == 0:
            return train_dates
        
        # Find test end date
        test_end = test_dates.max()
        
        # Compute embargo end
        embargo_end = self.calendar.add_trading_days(test_end, embargo_days)
        
        # Remove train dates in embargo period
        embargoed = train_dates[
            (train_dates < test_dates.min()) | (train_dates > embargo_end)
        ]
        
        removed = len(train_dates) - len(embargoed)
        if removed > 0:
            logger.debug(f"Embargoed {removed} training dates")
        
        return embargoed
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class TimeSeriesCV:
    """
    Time-series cross-validation with expanding or rolling window.
    
    Simpler alternative to PurgedKFold for basic time-series CV.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        mode: str = 'expanding'
    ):
        """
        Initialize TimeSeriesCV.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (in days). If None, computed automatically
            gap: Gap between train and test (in days)
            mode: 'expanding' or 'rolling'
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.mode = mode
        
        if mode not in ['expanding', 'rolling']:
            raise ValueError("mode must be 'expanding' or 'rolling'")
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Args:
            X: Features DataFrame with DatetimeIndex
            y: Labels (optional)
            groups: Group labels (optional)
        
        Yields:
            (train_indices, test_indices) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        # Get unique dates
        dates = X.index.unique().sort_values()
        n_dates = len(dates)
        
        # Compute test size if not provided
        if self.test_size is None:
            test_size = n_dates // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Test set
            test_end = n_dates - i * test_size
            test_start = test_end - test_size
            
            if test_start < 0:
                break
            
            # Train set
            if self.mode == 'expanding':
                # Expanding window: use all data before test
                train_start = 0
                train_end = test_start - self.gap
            else:  # rolling
                # Rolling window: use fixed window before test
                train_end = test_start - self.gap
                train_start = max(0, train_end - test_size)
            
            if train_end <= train_start:
                continue
            
            # Get dates
            train_dates = dates[train_start:train_end]
            test_dates = dates[test_start:test_end]
            
            # Convert to indices
            train_idx = X.index.isin(train_dates)
            test_idx = X.index.isin(test_dates)
            
            train_indices = np.where(train_idx)[0]
            test_indices = np.where(test_idx)[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits
