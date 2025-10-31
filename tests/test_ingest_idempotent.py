"""Tests for idempotent ingestion."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.ingest_incremental import IncrementalIngester


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    
    return pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates),
        'date': dates,
        'open': 100.0 + np.random.randn(len(dates)),
        'high': 105.0 + np.random.randn(len(dates)),
        'low': 99.0 + np.random.randn(len(dates)),
        'close': 103.0 + np.random.randn(len(dates)),
        'adj_close': 103.0 + np.random.randn(len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })


def test_upsert_prices_creates_new_file(temp_data_dir, sample_prices):
    """Test that upsert creates new file when none exists."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Upsert
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    
    # Check file exists
    year = sample_prices['date'].iloc[0].year
    file_path = temp_data_dir / 'raw' / 'prices' / f'year={year}' / f'prices_{year}.parquet'
    
    assert file_path.exists()
    
    # Read and verify
    df = pd.read_parquet(file_path)
    assert len(df) == len(sample_prices)


def test_upsert_prices_idempotent(temp_data_dir, sample_prices):
    """Test that re-running upsert doesn't increase row count."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # First upsert
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    
    # Get row count
    year = sample_prices['date'].iloc[0].year
    file_path = temp_data_dir / 'raw' / 'prices' / f'year={year}' / f'prices_{year}.parquet'
    df1 = pd.read_parquet(file_path)
    count1 = len(df1)
    
    # Second upsert with same data
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    
    # Get row count again
    df2 = pd.read_parquet(file_path)
    count2 = len(df2)
    
    # Should be the same (idempotent)
    assert count1 == count2, \
        f"Row count changed from {count1} to {count2}, not idempotent"


def test_upsert_prices_updates_existing(temp_data_dir, sample_prices):
    """Test that upsert updates existing rows with new data."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # First upsert
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    
    # Modify data (update adj_close)
    modified_prices = sample_prices.copy()
    modified_prices['adj_close'] = modified_prices['adj_close'] * 1.1
    
    # Second upsert with modified data
    ingester.upsert_prices(modified_prices, partition_by_year=True)
    
    # Read and verify
    year = sample_prices['date'].iloc[0].year
    file_path = temp_data_dir / 'raw' / 'prices' / f'year={year}' / f'prices_{year}.parquet'
    df = pd.read_parquet(file_path)
    
    # Row count should be the same
    assert len(df) == len(sample_prices)
    
    # Values should be updated
    assert not df['adj_close'].equals(sample_prices['adj_close'])


def test_upsert_prices_adds_new_rows(temp_data_dir):
    """Test that upsert adds new rows."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # First batch
    dates1 = pd.date_range('2023-01-01', '2023-01-05', freq='D')
    prices1 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates1),
        'date': dates1,
        'adj_close': [100.0] * len(dates1),
        'volume': [1000000] * len(dates1)
    })
    
    ingester.upsert_prices(prices1, partition_by_year=True)
    
    # Second batch (new dates)
    dates2 = pd.date_range('2023-01-06', '2023-01-10', freq='D')
    prices2 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates2),
        'date': dates2,
        'adj_close': [101.0] * len(dates2),
        'volume': [2000000] * len(dates2)
    })
    
    ingester.upsert_prices(prices2, partition_by_year=True)
    
    # Read and verify
    file_path = temp_data_dir / 'raw' / 'prices' / 'year=2023' / 'prices_2023.parquet'
    df = pd.read_parquet(file_path)
    
    # Should have all rows
    assert len(df) == len(prices1) + len(prices2)


def test_year_partitioning(temp_data_dir):
    """Test that data is correctly partitioned by year."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Data spanning two years
    dates = pd.date_range('2022-12-25', '2023-01-05', freq='D')
    prices = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates),
        'date': dates,
        'adj_close': [100.0] * len(dates),
        'volume': [1000000] * len(dates)
    })
    
    ingester.upsert_prices(prices, partition_by_year=True)
    
    # Check both year partitions exist
    file_2022 = temp_data_dir / 'raw' / 'prices' / 'year=2022' / 'prices_2022.parquet'
    file_2023 = temp_data_dir / 'raw' / 'prices' / 'year=2023' / 'prices_2023.parquet'
    
    assert file_2022.exists()
    assert file_2023.exists()
    
    # Verify data in each partition
    df_2022 = pd.read_parquet(file_2022)
    df_2023 = pd.read_parquet(file_2023)
    
    assert all(pd.to_datetime(df_2022['date']).dt.year == 2022)
    assert all(pd.to_datetime(df_2023['date']).dt.year == 2023)


def test_no_duplicate_keys(temp_data_dir, sample_prices):
    """Test that no duplicate (symbol, date) pairs exist after upsert."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Upsert twice
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    ingester.upsert_prices(sample_prices, partition_by_year=True)
    
    # Read and check for duplicates
    year = sample_prices['date'].iloc[0].year
    file_path = temp_data_dir / 'raw' / 'prices' / f'year={year}' / f'prices_{year}.parquet'
    df = pd.read_parquet(file_path)
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['symbol', 'date'], keep=False)
    
    assert not duplicates.any(), \
        f"Found {duplicates.sum()} duplicate (symbol, date) pairs"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
