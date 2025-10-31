"""Command-line interface for ML3."""

import click
import logging
from pathlib import Path

from src.utils import setup_logging

logger = setup_logging(__name__)


@click.group()
def cli():
    """ML3 - Point-in-Time ML Market Data Pipeline."""
    pass


@cli.group()
def data():
    """Data ingestion and processing commands."""
    pass


@data.command('ingest-prices')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--incremental/--full', default=True, help='Incremental or full refresh')
def ingest_prices(start_date, end_date, incremental):
    """Ingest price data from EODHD."""
    from flows.ingest_prices import ingest_prices_flow
    ingest_prices_flow(start_date, end_date, incremental)


@data.command('ingest-fundamentals')
@click.option('--incremental/--full', default=True, help='Incremental or full refresh')
def ingest_fundamentals(incremental):
    """Ingest fundamental data from EODHD."""
    from flows.ingest_fundamentals import ingest_fundamentals_flow
    ingest_fundamentals_flow(incremental)


@data.command('build-pit')
def build_pit():
    """Build point-in-time panel."""
    from flows.build_pit import build_pit_flow
    build_pit_flow()


@cli.group()
def features():
    """Feature engineering commands."""
    pass


@features.command('build')
def build_features():
    """Build features from PIT panel."""
    from flows.build_features import build_features_flow
    build_features_flow()


@cli.group()
def labels():
    """Label generation commands."""
    pass


@labels.command('build')
def build_labels():
    """Build labels from features."""
    from flows.build_labels import build_labels_flow
    build_labels_flow()


@cli.group()
def train():
    """Model training commands."""
    pass


@train.command('run')
@click.option('--model-type', default='lightgbm', type=click.Choice(['lightgbm', 'xgboost']))
def train_run(model_type):
    """Train model with cross-validation."""
    from flows.train import train_flow
    model_id = train_flow(model_type)
    click.echo(f"Model trained: {model_id}")


@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command('run')
@click.argument('model_id')
def backtest_run(model_id):
    """Run backtest for a model."""
    from flows.backtest import backtest_flow
    backtest_flow(model_id)


@cli.group()
def models():
    """Model registry commands."""
    pass


@models.command('list')
def list_models():
    """List all models in registry."""
    from src.registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        click.echo("No models in registry")
        return
    
    click.echo(f"Found {len(models)} models:")
    for model in models:
        click.echo(f"  - {model['model_id']} ({model['model_type']}) - {model.get('created_at', 'N/A')}")


@models.command('info')
@click.argument('model_id')
def model_info(model_id):
    """Show model information."""
    from src.registry import ModelRegistry
    registry = ModelRegistry()
    summary = registry.get_model_summary(model_id)
    
    if 'error' in summary:
        click.echo(f"Error: {summary['error']}")
        return
    
    click.echo(f"Model ID: {summary['model_id']}")
    click.echo(f"Type: {summary['model_type']}")
    click.echo(f"Created: {summary.get('created_at', 'N/A')}")
    click.echo(f"Features: {summary['n_features']}")
    click.echo(f"Label: {summary['label']}")
    
    if summary.get('metrics'):
        click.echo("\nMetrics:")
        for key, value in summary['metrics'].items():
            click.echo(f"  {key}: {value}")


@models.command('delete')
@click.argument('model_id')
@click.confirmation_option(prompt='Are you sure you want to delete this model?')
def delete_model(model_id):
    """Delete a model from registry."""
    from src.registry import ModelRegistry
    registry = ModelRegistry()
    success = registry.delete_model(model_id)
    
    if success:
        click.echo(f"Model {model_id} deleted")
    else:
        click.echo(f"Failed to delete model {model_id}")


@cli.command('dashboard')
@click.option('--port', default=8501, help='Port to run dashboard on')
def dashboard(port):
    """Launch Streamlit dashboard."""
    import subprocess
    import sys
    
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    
    click.echo(f"Starting dashboard on port {port}...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port)
    ])


@cli.command('api')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to run API on')
def api(host, port):
    """Launch FastAPI service."""
    import uvicorn
    
    click.echo(f"Starting API on {host}:{port}...")
    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=True
    )


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
