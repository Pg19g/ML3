"""Flow for ingesting fundamental data from EODHD."""

import pandas as pd
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


@task(name="fetch_fundamentals")
def fetch_fundamentals(symbols: list) -> pd.DataFrame:
    """Fetch fundamentals from EODHD API."""
    client = EODHDClient()
    
    all_fundamentals = []
    
    for symbol in symbols:
        logger.info(f"Fetching fundamentals for {symbol}")
        
        fund_data = client.get_fundamentals(symbol)
        
        if not fund_data:
            logger.warning(f"No fundamentals for {symbol}")
            continue
        
        # Parse to DataFrame
        df = client.parse_fundamentals_to_dataframe(fund_data, symbol)
        
        if not df.empty:
            all_fundamentals.append(df)
    
    if not all_fundamentals:
        logger.warning("No fundamentals fetched")
        return pd.DataFrame()
    
    combined = pd.concat(all_fundamentals, ignore_index=True)
    logger.info(f"Fetched {len(combined)} fundamental rows")
    
    return combined


@task(name="merge_fundamentals")
def merge_fundamentals(new_data: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Merge new fundamentals with existing data."""
    existing_df = load_parquet(file_path)
    
    if existing_df.empty:
        logger.info("No existing fundamentals, using new data only")
        return new_data
    
    # Combine and deduplicate
    combined = pd.concat([existing_df, new_data], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=['symbol', 'period_end', 'statement_type'],
        keep='last'
    )
    combined = combined.sort_values(['symbol', 'period_end']).reset_index(drop=True)
    
    logger.info(f"Merged fundamentals: {len(existing_df)} existing + {len(new_data)} new = {len(combined)} total")
    return combined


@task(name="save_fundamentals")
def save_fundamentals(df: pd.DataFrame, file_path: str) -> None:
    """Save fundamentals to parquet file."""
    save_parquet(df, file_path, index=False)
    logger.info(f"Saved {len(df)} fundamental rows to {file_path}")


@flow(name="ingest-fundamentals")
def ingest_fundamentals_flow(incremental: bool = True) -> None:
    """
    Ingest fundamental data from EODHD.
    
    Args:
        incremental: If True, merge with existing data
    """
    logger.info("Starting fundamentals ingestion flow")
    
    # Load universe
    universe_config = load_config(str(get_config_path("universe.yaml")))
    symbols = universe_config.get('symbols', [])
    logger.info(f"Loaded {len(symbols)} symbols from universe")
    
    # Fetch fundamentals
    new_fundamentals = fetch_fundamentals(symbols)
    
    if new_fundamentals.empty:
        logger.warning("No fundamentals fetched")
        return
    
    # File path
    file_path = str(get_data_path("raw") / "fundamentals.parquet")
    
    # Merge with existing if incremental
    if incremental:
        final_fundamentals = merge_fundamentals(new_fundamentals, file_path)
    else:
        final_fundamentals = new_fundamentals
    
    # Save
    save_fundamentals(final_fundamentals, file_path)
    
    logger.info("Fundamentals ingestion flow complete")


if __name__ == "__main__":
    ingest_fundamentals_flow()
