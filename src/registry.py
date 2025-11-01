"""Model registry for managing trained models."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import joblib
import json

import lightgbm as lgb
import xgboost as xgb

from src.utils import (
    setup_logging,
    get_model_registry_path,
    load_json,
    save_json
)

logger = setup_logging(__name__)


class ModelRegistry:
    """Manage model registry."""
    
    def __init__(self):
        """Initialize model registry."""
        self.registry_path = get_model_registry_path()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in registry.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for model_dir in self.registry_path.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / 'metadata.json'
                if metadata_path.exists():
                    metadata = load_json(str(metadata_path))
                    models.append(metadata)
        
        # Sort by creation date
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with model and metadata
        """
        model_dir = self.registry_path / model_id
        
        if not model_dir.exists():
            logger.error(f"Model {model_id} not found")
            return None
        
        metadata_path = model_dir / 'metadata.json'
        if not metadata_path.exists():
            logger.error(f"Metadata not found for model {model_id}")
            return None
        
        metadata = load_json(str(metadata_path))
        model_type = metadata['model_type']
        
        # Load model
        if model_type == 'lightgbm':
            model_path = model_dir / 'model.txt'
            model = lgb.Booster(model_file=str(model_path))
        elif model_type == 'xgboost':
            model_path = model_dir / 'model.json'
            model = xgb.Booster()
            model.load_model(str(model_path))
        elif model_type == 'external':
            # External model (ONNX or joblib)
            if (model_dir / 'model.onnx').exists():
                import onnxruntime as ort
                model = ort.InferenceSession(str(model_dir / 'model.onnx'))
            elif (model_dir / 'model.pkl').exists():
                model = joblib.load(str(model_dir / 'model.pkl'))
            else:
                logger.error(f"No model file found for {model_id}")
                return None
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        return {
            'model': model,
            'metadata': metadata
        }
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful
        """
        model_dir = self.registry_path / model_id
        
        if not model_dir.exists():
            logger.error(f"Model {model_id} not found")
            return False
        
        # Delete all files in directory
        for file in model_dir.iterdir():
            file.unlink()
        
        # Delete directory
        model_dir.rmdir()
        
        logger.info(f"Deleted model {model_id}")
        return True
    
    def import_external_model(
        self,
        model_path: str,
        model_id: str,
        feature_cols: List[str],
        label_col: str,
        model_format: str = 'auto'
    ) -> str:
        """
        Import external model into registry.
        
        Args:
            model_path: Path to model file
            model_id: Model ID
            feature_cols: Required feature columns
            label_col: Label column
            model_format: 'onnx', 'joblib', or 'auto'
            
        Returns:
            Model ID
        """
        logger.info(f"Importing external model: {model_path}")
        
        # Detect format
        if model_format == 'auto':
            if model_path.endswith('.onnx'):
                model_format = 'onnx'
            elif model_path.endswith(('.pkl', '.joblib')):
                model_format = 'joblib'
            else:
                raise ValueError(f"Cannot detect model format from {model_path}")
        
        # Create model directory
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Copy model file
        import shutil
        if model_format == 'onnx':
            shutil.copy(model_path, str(model_dir / 'model.onnx'))
        elif model_format == 'joblib':
            shutil.copy(model_path, str(model_dir / 'model.pkl'))
        
        # Create metadata
        metadata = {
            'model_id': model_id,
            'model_type': 'external',
            'model_format': model_format,
            'feature_cols': feature_cols,
            'label_col': label_col,
            'created_at': pd.Timestamp.now().isoformat(),
            'source': 'external_import'
        }
        
        save_json(metadata, str(model_dir / 'metadata.json'))
        
        logger.info(f"Imported model as {model_id}")
        return model_id
    
    def validate_model_schema(
        self,
        model_id: str,
        required_features: List[str]
    ) -> Dict[str, Any]:
        """
        Validate model schema against required features.
        
        Args:
            model_id: Model ID
            required_features: List of required feature names
            
        Returns:
            Dictionary with validation results
        """
        model_info = self.get_model(model_id)
        
        if model_info is None:
            return {
                'valid': False,
                'error': 'Model not found'
            }
        
        metadata = model_info['metadata']
        model_features = metadata.get('feature_cols', [])
        
        # Check if all model features are available
        missing_features = set(model_features) - set(required_features)
        extra_features = set(required_features) - set(model_features)
        
        valid = len(missing_features) == 0
        
        return {
            'valid': valid,
            'model_features': model_features,
            'required_features': required_features,
            'missing_features': list(missing_features),
            'extra_features': list(extra_features)
        }
    
    def score_model(
        self,
        model_id: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Score data with model.
        
        Args:
            model_id: Model ID
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        model_info = self.get_model(model_id)
        
        if model_info is None:
            raise ValueError(f"Model {model_id} not found")
        
        model = model_info['model']
        metadata = model_info['metadata']
        model_type = metadata['model_type']
        
        # Ensure features are in correct order
        # Filter to only use features that exist in X
        feature_cols = metadata['feature_cols']
        available_features = [col for col in feature_cols if col in X.columns]
        
        if len(available_features) < len(feature_cols):
            missing = [col for col in feature_cols if col not in X.columns]
            logger.warning(f"Model expects {len(feature_cols)} features but only {len(available_features)} available. Missing: {missing[:5]}...")
        
        X_ordered = X[available_features]
        
        # Score
        if model_type == 'lightgbm':
            predictions = model.predict(X_ordered)
        elif model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X_ordered)
            predictions = model.predict(dmatrix)
        elif model_type == 'external':
            model_format = metadata.get('model_format', 'joblib')
            if model_format == 'onnx':
                import onnxruntime as ort
                input_name = model.get_inputs()[0].name
                predictions = model.run(None, {input_name: X_ordered.values.astype(np.float32)})[0]
                predictions = predictions.flatten()
            elif model_format == 'joblib':
                predictions = model.predict(X_ordered)
            else:
                raise ValueError(f"Unknown model format: {model_format}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions
    
    def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get model summary.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with model summary
        """
        model_info = self.get_model(model_id)
        
        if model_info is None:
            return {'error': 'Model not found'}
        
        metadata = model_info['metadata']
        
        summary = {
            'model_id': model_id,
            'model_type': metadata['model_type'],
            'created_at': metadata.get('created_at'),
            'n_features': len(metadata.get('feature_cols', [])),
            'label': metadata.get('label_col'),
            'metrics': metadata.get('metrics', {})
        }
        
        return summary
