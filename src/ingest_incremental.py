"""Incremental and idempotent data ingestion for EODHD."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from src.eodhd_client import EODHDClient
from src.utils import setup_logging
from src.utils.schemas import enforce_schema

logger = setup_logging(__name__)


class IncrementalIngester:
    """
    Handles incremental and idempotent data ingestion.
    
    Features:
    - Incremental updates based on last available date
    - Idempotent upserts (no duplicate keys)
    - Year-based partitioning
    - Full refresh mode
    """
    
    def __init__(
        self,
        data_dir: Path,
        client: Optional[EODHDClient] = None
    ):
        """
        Initialize incremental ingester.
        
        Args:
            data_dir: Base directory for data storage
            client: EODHD client (creates new one if None)
        """
        self.data_dir = Path(data_dir)
        self.client = client or EODHDClient()
    
    def get_last_available_date(
        self,
        symbol: str,
        data_type: str = 'prices'
    ) -> Optional[datetime]:
        """
        Get the last available date for a symbol.
        
        Args:
            symbol: Ticker symbol
            data_type: 'prices' or 'fundamentals'
        
        Returns:
            Last available date, or None if no data exists
        """
        if data_type == 'prices':
            pattern = self.data_dir / 'raw' / 'prices' / 'year=*' / '*.parquet'
        else:
            pattern = self.data_dir / 'raw' / 'fundamentals' / '*.parquet'
        
        # Find all parquet files
        files = list(self.data_dir.glob(str(pattern).replace(str(self.data_dir) + '/', '')))
        
        if not files:
            return None
        
        # Read and find max date for this symbol
        max_date = None
        
        for file in files:
            try:
                df = pd.read_parquet(file)
                
                if 'symbol' in df.columns:
                    symbol_data = df[df['symbol'] == symbol]
                    
                    if len(symbol_data) > 0:
                        date_col = 'date' if 'date' in symbol_data.columns else 'period_end'
                        symbol_max = pd.to_datetime(symbol_data[date_col]).max()
                        
                        if max_date is None or symbol_max > max_date:
                            max_date = symbol_max
            except Exception as e:
                logger.warning(f"Error reading {file}: {e}")
        
        return max_date
    
    def ingest_prices_incremental(
        self,
        symbols: List[str],
        since: Optional[datetime] = None,
        full_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Ingest prices incrementally.
        
        Args:
            symbols: List of symbols to ingest
            since: Start date (if None, uses last available date per symbol)
            full_refresh: If True, ignore existing data and fetch all
        
        Returns:
            DataFrame with ingested prices
        """
        all_prices = []
        
        for symbol in symbols:
            logger.info(f"Ingesting prices for {symbol}...")
            
            # Determine start date
            if full_refresh:
                start_date = None  # Fetch all available
            elif since is not None:
                start_date = since
            else:
                # Incremental: start from last available date + 1 day
                last_date = self.get_last_available_date(symbol, 'prices')
                
                if last_date is not None:
                    start_date = last_date + timedelta(days=1)
                    logger.info(f"Last available date for {symbol}: {last_date}, fetching from {start_date}")
                else:
                    start_date = None  # No existing data, fetch all
            
            # Fetch data
            try:
                prices = self.client.get_eod_prices(
                    symbol=symbol,
                    from_date=start_date.strftime('%Y-%m-%d') if start_date else None,
                    to_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if prices and len(prices) > 0:
                    all_prices.append(prices)
                    logger.info(f"Fetched {len(prices)} price records for {symbol}")
                else:
                    logger.info(f"No new prices for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching prices for {symbol}: {e}")
        
        # Combine all prices
        if not all_prices:
            logger.warning("No prices fetched")
            return pd.DataFrame()
        
        df = pd.concat(all_prices, ignore_index=True)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def ingest_fundamentals_incremental(
        self,
        symbols: List[str],
        since: Optional[datetime] = None,
        full_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Ingest fundamentals incrementally.
        
        Args:
            symbols: List of symbols to ingest
            since: Start date (if None, uses last available date per symbol)
            full_refresh: If True, ignore existing data and fetch all
        
        Returns:
            DataFrame with ingested fundamentals
        """
        all_fundamentals = []
        
        for symbol in symbols:
            logger.info(f"Ingesting fundamentals for {symbol}...")
            
            # For fundamentals, we typically fetch all and upsert
            # (since EODHD doesn't support incremental fundamental queries)
            
            try:
                fundamentals = self.client.get_fundamentals(symbol=symbol)
                
                if fundamentals and len(fundamentals) > 0:
                    all_fundamentals.append(fundamentals)
                    logger.info(f"Fetched {len(fundamentals)} fundamental records for {symbol}")
                else:
                    logger.info(f"No fundamentals for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        
        # Combine all fundamentals
        if not all_fundamentals:
            logger.warning("No fundamentals fetched")
            return pd.DataFrame()
        
        df = pd.concat(all_fundamentals, ignore_index=True)
        
        # Ensure date columns are datetime
        if 'period_end' in df.columns:
            df['period_end'] = pd.to_datetime(df['period_end'])
        if 'filing_date' in df.columns:
            df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        
        # Add updated_at timestamp
        df['updated_at'] = datetime.now()
        
        return df
    
    def upsert_prices(
        self,
        new_data: pd.DataFrame,
        partition_by_year: bool = True
    ) -> None:
        """
        Upsert prices data (idempotent).
        
        Merges new data with existing data, removing duplicates based on
        primary key (symbol, date).
        
        Args:
            new_data: New price data to upsert
            partition_by_year: If True, partition by year
        """
        if len(new_data) == 0:
            logger.info("No data to upsert")
            return
        
        # Validate schema
        enforce_schema(new_data, 'prices_daily', raise_on_error=True)
        
        # Ensure date is datetime
        new_data['date'] = pd.to_datetime(new_data['date'])
        
        # Add year column for partitioning
        new_data['year'] = new_data['date'].dt.year
        
        if partition_by_year:
            # Upsert per year
            for year in new_data['year'].unique():
                year_data = new_data[new_data['year'] == year].copy()
                self._upsert_year_partition(year_data, year, 'prices')
        else:
            # Single file upsert
            self._upsert_single_file(new_data, 'prices')
    
    def _upsert_year_partition(
        self,
        new_data: pd.DataFrame,
        year: int,
        data_type: str
    ) -> None:
        """Upsert data for a specific year partition."""
        # Create partition directory
        partition_dir = self.data_dir / 'raw' / data_type / f'year={year}'
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # File path
        file_path = partition_dir / f'{data_type}_{year}.parquet'
        
        # Read existing data if exists
        if file_path.exists():
            existing_data = pd.read_parquet(file_path)
            
            # Combine and remove duplicates
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates based on primary key
            pk_cols = ['symbol', 'date']
            combined = combined.drop_duplicates(subset=pk_cols, keep='last')
            
            logger.info(
                f"Upserted {len(new_data)} rows into year={year} "
                f"(existing: {len(existing_data)}, final: {len(combined)})"
            )
        else:
            combined = new_data
            logger.info(f"Created new partition for year={year} with {len(combined)} rows")
        
        # Sort by date for better compression
        combined = combined.sort_values(['symbol', 'date'])
        
        # Drop year column before saving
        if 'year' in combined.columns:
            combined = combined.drop(columns=['year'])
        
        # Write to parquet
        combined.to_parquet(file_path, index=False, compression='snappy')
        
        logger.info(f"Wrote {len(combined)} rows to {file_path}")
    
    def upsert_fundamentals(
        self,
        new_data: pd.DataFrame
    ) -> None:
        """
        Upsert fundamentals data (idempotent).
        
        Merges new data with existing data, removing duplicates based on
        primary key (symbol, statement_type, period_end).
        
        Args:
            new_data: New fundamental data to upsert
        """
        if len(new_data) == 0:
            logger.info("No data to upsert")
            return
        
        # Validate schema
        enforce_schema(new_data, 'fundamentals', raise_on_error=True)
        
        # File path
        file_path = self.data_dir / 'raw' / 'fundamentals' / 'fundamentals.parquet'
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing data if exists
        if file_path.exists():
            existing_data = pd.read_parquet(file_path)
            
            # Combine and remove duplicates
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates based on primary key
            pk_cols = ['symbol', 'statement_type', 'period_end']
            combined = combined.drop_duplicates(subset=pk_cols, keep='last')
            
            logger.info(
                f"Upserted {len(new_data)} rows "
                f"(existing: {len(existing_data)}, final: {len(combined)})"
            )
        else:
            combined = new_data
            logger.info(f"Created new fundamentals file with {len(combined)} rows")
        
        # Sort for better compression
        combined = combined.sort_values(['symbol', 'period_end'])
        
        # Write to parquet
        combined.to_parquet(file_path, index=False, compression='snappy')
        
        logger.info(f"Wrote {len(combined)} rows to {file_path}")
    
    def _upsert_single_file(
        self,
        new_data: pd.DataFrame,
        data_type: str
    ) -> None:
        """Upsert data to a single file (no partitioning)."""
        file_path = self.data_dir / 'raw' / data_type / f'{data_type}.parquet'
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing data if exists
        if file_path.exists():
            existing_data = pd.read_parquet(file_path)
            
            # Combine and remove duplicates
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates based on primary key
            pk_cols = ['symbol', 'date']
            combined = combined.drop_duplicates(subset=pk_cols, keep='last')
            
            logger.info(
                f"Upserted {len(new_data)} rows "
                f"(existing: {len(existing_data)}, final: {len(combined)})"
            )
        else:
            combined = new_data
            logger.info(f"Created new file with {len(combined)} rows")
        
        # Sort for better compression
        combined = combined.sort_values(['symbol', 'date'])
        
        # Write to parquet
        combined.to_parquet(file_path, index=False, compression='snappy')
        
        logger.info(f"Wrote {len(combined)} rows to {file_path}")
