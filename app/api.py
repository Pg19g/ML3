"""FastAPI service for ML3."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

from src.registry import ModelRegistry
from src.utils import setup_logging, get_data_path, load_parquet

logger = setup_logging(__name__)

app = FastAPI(
    title="ML3 API",
    description="Point-in-Time ML Market Data Pipeline API",
    version="0.1.0"
)


class FlowTrigger(BaseModel):
    """Request model for triggering flows."""
    flow_name: str
    parameters: Optional[Dict[str, Any]] = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_id: str
    symbols: List[str]
    date: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "ML3 API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    """List all models in registry."""
    registry = ModelRegistry()
    models = registry.list_models()
    return {"models": models}


@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    """Get model information."""
    registry = ModelRegistry()
    summary = registry.get_model_summary(model_id)
    
    if 'error' in summary:
        raise HTTPException(status_code=404, detail=summary['error'])
    
    return summary


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Delete a model."""
    registry = ModelRegistry()
    success = registry.delete_model(model_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": f"Model {model_id} deleted"}


@app.post("/models/{model_id}/predict")
def predict(model_id: str, request: PredictionRequest):
    """Generate predictions using a model."""
    registry = ModelRegistry()
    
    # Load latest data
    data_path = str(get_data_path("pit") / "labels.parquet")
    data = load_parquet(data_path)
    
    if data.empty:
        raise HTTPException(status_code=400, detail="No data available")
    
    # Filter by symbols
    data = data[data['symbol'].isin(request.symbols)]
    
    # Filter by date if provided
    if request.date:
        data = data[data['date'] == request.date]
    
    if data.empty:
        raise HTTPException(status_code=400, detail="No data for specified symbols/date")
    
    # Get model info
    model_info = registry.get_model(model_id)
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metadata = model_info['metadata']
    feature_cols = metadata['feature_cols']
    
    # Score
    try:
        predictions = registry.score_model(model_id, data[feature_cols])
        
        results = []
        for idx, (_, row) in enumerate(data.iterrows()):
            results.append({
                'symbol': row['symbol'],
                'date': str(row['date']),
                'prediction': float(predictions[idx])
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/flows/trigger")
def trigger_flow(request: FlowTrigger):
    """Trigger a Prefect flow."""
    flow_name = request.flow_name
    params = request.parameters
    
    try:
        if flow_name == "ingest-prices":
            from flows.ingest_prices import ingest_prices_flow
            ingest_prices_flow(**params)
        elif flow_name == "ingest-fundamentals":
            from flows.ingest_fundamentals import ingest_fundamentals_flow
            ingest_fundamentals_flow(**params)
        elif flow_name == "build-pit":
            from flows.build_pit import build_pit_flow
            build_pit_flow()
        elif flow_name == "build-features":
            from flows.build_features import build_features_flow
            build_features_flow()
        elif flow_name == "build-labels":
            from flows.build_labels import build_labels_flow
            build_labels_flow()
        elif flow_name == "train":
            from flows.train import train_flow
            model_id = train_flow(**params)
            return {"message": f"Training complete", "model_id": model_id}
        elif flow_name == "backtest":
            from flows.backtest import backtest_flow
            backtest_flow(**params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown flow: {flow_name}")
        
        return {"message": f"Flow {flow_name} triggered successfully"}
        
    except Exception as e:
        logger.error(f"Flow trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/stats")
def get_data_stats():
    """Get data statistics."""
    stats = {}
    
    # Prices
    prices_path = str(get_data_path("raw") / "prices_daily.parquet")
    try:
        prices = load_parquet(prices_path)
        if not prices.empty:
            stats['prices'] = {
                'n_rows': len(prices),
                'n_symbols': prices['symbol'].nunique() if 'symbol' in prices.columns else 0,
                'date_range': {
                    'start': str(prices['date'].min()) if 'date' in prices.columns else None,
                    'end': str(prices['date'].max()) if 'date' in prices.columns else None
                }
            }
    except:
        stats['prices'] = {'n_rows': 0}
    
    # Fundamentals
    fund_path = str(get_data_path("raw") / "fundamentals.parquet")
    try:
        fund = load_parquet(fund_path)
        if not fund.empty:
            stats['fundamentals'] = {
                'n_rows': len(fund),
                'n_symbols': fund['symbol'].nunique() if 'symbol' in fund.columns else 0
            }
    except:
        stats['fundamentals'] = {'n_rows': 0}
    
    # Features
    features_path = str(get_data_path("pit") / "features.parquet")
    try:
        features = load_parquet(features_path)
        if not features.empty:
            stats['features'] = {
                'n_rows': len(features),
                'n_columns': len(features.columns)
            }
    except:
        stats['features'] = {'n_rows': 0}
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
