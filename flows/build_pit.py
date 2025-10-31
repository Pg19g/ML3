"""Flow for building point-in-time panel."""

import pandas as pd
import logging

from prefect import flow, task

from src.pit import PITProcessor
from src.utils import (
    setup_logging,
    get_data_path,
    load_parquet,
    save_parquet
)

logger = setup_logging(__name__)


@task(name="load_raw_data")
def load_raw_data() -> tuple:
    """Load raw prices and fundamentals."""
    prices_path = str(get_data_path("raw") / "prices_daily.parquet")
    fundamentals_path = str(get_data_path("raw") / "fundamentals.parquet")
    
    prices = load_parquet(prices_path)
    fundamentals = load_parquet(fundamentals_path)
    
    logger.info(f"Loaded {len(prices)} price rows and {len(fundamentals)} fundamental rows")
    
    return prices, fundamentals


@task(name="build_pit_panel")
def build_pit_panel(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time panel."""
    processor = PITProcessor()
    
    pit_panel = processor.build_pit_panel(prices, fundamentals)
    
    logger.info(f"Built PIT panel with {len(pit_panel)} rows")
    return pit_panel


@task(name="check_integrity")
def check_integrity(pit_panel: pd.DataFrame) -> dict:
    """Check PIT integrity."""
    processor = PITProcessor()
    
    results = processor.check_pit_integrity(pit_panel)
    
    if results['passed']:
        logger.info("PIT integrity check PASSED")
    else:
        logger.error(f"PIT integrity check FAILED with {results['violations']} violations")
    
    return results


@task(name="save_pit_panel")
def save_pit_panel(pit_panel: pd.DataFrame, file_path: str) -> None:
    """Save PIT panel to parquet."""
    save_parquet(pit_panel, file_path, index=False)
    logger.info(f"Saved PIT panel with {len(pit_panel)} rows to {file_path}")


@flow(name="build-pit")
def build_pit_flow() -> None:
    """Build point-in-time panel from raw data."""
    logger.info("Starting PIT build flow")
    
    # Load raw data
    prices, fundamentals = load_raw_data()
    
    if prices.empty:
        logger.error("No price data available")
        return
    
    if fundamentals.empty:
        logger.warning("No fundamental data available, building PIT panel with prices only")
        fundamentals = pd.DataFrame()
    
    # Build PIT panel
    pit_panel = build_pit_panel(prices, fundamentals)
    
    # Check integrity
    integrity_results = check_integrity(pit_panel)
    
    # Save
    file_path = str(get_data_path("pit") / "daily_panel.parquet")
    save_pit_panel(pit_panel, file_path)
    
    logger.info("PIT build flow complete")


if __name__ == "__main__":
    build_pit_flow()
