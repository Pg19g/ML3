"""Flow for ingesting price data from EODHD."""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from prefect import flow, task

from src.eodhd_client import EODHDClient
from src.utils import (
    setup_logging,
    load_config,
    get_config_path,
    get_data_path,
    load_parquet,
    save_parquet
)

logger = setup_logging(__name__)


@task(name="load_universe")
def load_universe() -> list:
    """Load universe of symbols from config."""
    config = load_config(str(get_config_path("universe.yaml")))
    symbols = config.get('symbols', [])
    logger.info(f"Loaded {len(symbols)} symbols from universe")
    return symbols


@task(name="fetch_prices")
def fetch_prices(
    symbols: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch prices from EODHD API."""
    client = EODHDClient()
    
    # Load date range from config if not provided
    if start_date is None or end_date is None:
        universe_config = load_config(str(get_config_path("universe.yaml")))
        date_range = universe_config.get('date_range', {})
        start_date = start_date or date_range.get('start', '2020-01-01')
        end_date = end_date or date_range.get('end')
    
    logger.info(f"Fetching prices for {len(symbols)} symbols from {start_date} to {end_date}")
    
    df = client.get_bulk_prices(symbols, start_date, end_date)
    
    logger.info(f"Fetched {len(df)} price rows")
    return df


@task(name="merge_with_existing")
def merge_with_existing(new_data: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Merge new data with existing parquet file (idempotent)."""
    existing_df = load_parquet(file_path)
    
    if existing_df.empty:
        logger.info("No existing data, using new data only")
        return new_data
    
    # Combine and deduplicate
    combined = pd.concat([existing_df, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
    combined = combined.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    
    logger.info(f"Merged data: {len(existing_df)} existing + {len(new_data)} new = {len(combined)} total")
    return combined


@task(name="save_prices")
def save_prices(df: pd.DataFrame, file_path: str) -> None:
    """Save prices to parquet file."""
    # Rename columns to standard format
    df = df.rename(columns={'Date': 'date', 'Symbol': 'symbol'})
    
    save_parquet(df, file_path, index=False)
    logger.info(f"Saved {len(df)} rows to {file_path}")


@flow(name="ingest-prices")
def ingest_prices_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    incremental: bool = True
) -> None:
    """
    Ingest price data from EODHD.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        incremental: If True, merge with existing data
    """
    logger.info("Starting price ingestion flow")
    
    # Load universe
    symbols = load_universe()
    
    # Fetch prices
    new_prices = fetch_prices(symbols, start_date, end_date)
    
    if new_prices.empty:
        logger.warning("No prices fetched")
        return
    
    # File path
    file_path = str(get_data_path("raw") / "prices_daily.parquet")
    
    # Merge with existing if incremental
    if incremental:
        final_prices = merge_with_existing(new_prices, file_path)
    else:
        final_prices = new_prices
    
    # Save
    save_prices(final_prices, file_path)
    
    logger.info("Price ingestion flow complete")


if __name__ == "__main__":
    ingest_prices_flow()
