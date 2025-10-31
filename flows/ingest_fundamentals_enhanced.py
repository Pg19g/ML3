"""Enhanced flow for incremental fundamentals ingestion from EODHD."""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging
import argparse

from prefect import flow, task

from src.ingest_incremental import IncrementalIngester
from src.utils import (
    setup_logging,
    load_config,
    get_config_path,
    get_data_path
)

logger = setup_logging(__name__)


@task(name="load_universe")
def load_universe() -> list:
    """Load universe of symbols from config."""
    config = load_config(str(get_config_path("universe.yaml")))
    symbols = config.get('symbols', [])
    logger.info(f"Loaded {len(symbols)} symbols from universe")
    return symbols


@task(name="ingest_fundamentals_incremental")
def ingest_fundamentals_incremental(
    symbols: list,
    since: Optional[datetime] = None,
    full_refresh: bool = False
) -> pd.DataFrame:
    """
    Ingest fundamentals incrementally.
    
    Args:
        symbols: List of symbols to ingest
        since: Start date (if None, auto-detect from last available)
        full_refresh: If True, fetch all data from scratch
    
    Returns:
        DataFrame with ingested fundamentals
    """
    data_dir = get_data_path("")
    ingester = IncrementalIngester(data_dir)
    
    logger.info(
        f"Ingesting fundamentals: full_refresh={full_refresh}, "
        f"since={since.strftime('%Y-%m-%d') if since else 'auto'}"
    )
    
    df = ingester.ingest_fundamentals_incremental(
        symbols=symbols,
        since=since,
        full_refresh=full_refresh
    )
    
    return df


@task(name="upsert_fundamentals")
def upsert_fundamentals(df: pd.DataFrame) -> None:
    """
    Upsert fundamentals to storage (idempotent).
    
    Args:
        df: DataFrame with fundamentals to upsert
    """
    if len(df) == 0:
        logger.warning("No fundamentals to upsert")
        return
    
    data_dir = get_data_path("")
    ingester = IncrementalIngester(data_dir)
    
    ingester.upsert_fundamentals(df)
    
    logger.info(f"Upserted {len(df)} fundamental rows")


@flow(name="ingest-fundamentals-enhanced")
def ingest_fundamentals_flow(
    since: Optional[str] = None,
    full_refresh: bool = False
) -> None:
    """
    Enhanced fundamentals ingestion flow with incremental support.
    
    Args:
        since: Start date (YYYY-MM-DD), if None uses last available date
        full_refresh: If True, fetch all data from scratch
    """
    logger.info("Starting enhanced fundamentals ingestion flow")
    
    # Load universe
    symbols = load_universe()
    
    # Parse since date
    since_dt = None
    if since:
        try:
            since_dt = datetime.strptime(since, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {since}, expected YYYY-MM-DD")
            return
    
    # Ingest fundamentals
    new_fundamentals = ingest_fundamentals_incremental(
        symbols=symbols,
        since=since_dt,
        full_refresh=full_refresh
    )
    
    if new_fundamentals.empty:
        logger.warning("No fundamentals fetched")
        return
    
    # Upsert to storage
    upsert_fundamentals(new_fundamentals)
    
    logger.info("Enhanced fundamentals ingestion flow complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest fundamentals from EODHD")
    parser.add_argument(
        '--since',
        type=str,
        help='Start date (YYYY-MM-DD), if not provided uses last available date'
    )
    parser.add_argument(
        '--full-refresh',
        action='store_true',
        help='Fetch all data from scratch, ignoring existing data'
    )
    
    args = parser.parse_args()
    
    ingest_fundamentals_flow(
        since=args.since,
        full_refresh=args.full_refresh
    )
