"""Schema validation utilities for ML3 data artifacts."""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates DataFrames against schemas defined in config/schemas.yaml."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize validator with schema definitions."""
        if schema_path is None:
            schema_path = Path(__file__).parent.parent.parent / "config" / "schemas.yaml"
        
        with open(schema_path, 'r') as f:
            self.schemas = yaml.safe_load(f)
    
    def validate(self, df: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """
        Validate a DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema_name: Name of schema in schemas.yaml
        
        Returns:
            Dict with 'valid' (bool) and 'errors' (list of error messages)
        """
        if schema_name not in self.schemas:
            return {
                'valid': False,
                'errors': [f"Schema '{schema_name}' not found"]
            }
        
        schema = self.schemas[schema_name]
        errors = []
        
        # Check columns exist
        errors.extend(self._validate_columns(df, schema))
        
        # Check data types
        errors.extend(self._validate_dtypes(df, schema))
        
        # Check primary key uniqueness
        errors.extend(self._validate_primary_key(df, schema))
        
        # Check validation rules
        if 'validation' in self.schemas and schema_name in self.schemas['validation']:
            errors.extend(self._validate_rules(df, schema_name))
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': []
        }
    
    def _validate_columns(self, df: pd.DataFrame, schema: Dict) -> List[str]:
        """Check that required columns exist."""
        errors = []
        
        if 'columns' not in schema:
            return errors
        
        required_cols = [
            col for col, spec in schema['columns'].items()
            if not spec.get('nullable', True) or col in schema.get('primary_key', [])
        ]
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        return errors
    
    def _validate_dtypes(self, df: pd.DataFrame, schema: Dict) -> List[str]:
        """Check that column data types match schema."""
        errors = []
        
        if 'columns' not in schema:
            return errors
        
        type_mapping = {
            'string': ['object', 'string'],
            'int64': ['int64', 'Int64'],
            'float64': ['float64', 'Float64'],
            'date': ['datetime64[ns]', 'object'],  # date can be datetime or object
            'datetime': ['datetime64[ns]'],
            'boolean': ['bool', 'boolean']
        }
        
        for col, spec in schema['columns'].items():
            if col not in df.columns:
                continue
            
            expected_type = spec.get('type')
            if expected_type and expected_type in type_mapping:
                actual_type = str(df[col].dtype)
                valid_types = type_mapping[expected_type]
                
                if not any(actual_type.startswith(vt) for vt in valid_types):
                    errors.append(
                        f"Column '{col}': expected {expected_type}, got {actual_type}"
                    )
        
        return errors
    
    def _validate_primary_key(self, df: pd.DataFrame, schema: Dict) -> List[str]:
        """Check primary key uniqueness."""
        errors = []
        
        pk_cols = schema.get('primary_key', [])
        if not pk_cols:
            return errors
        
        # Check all PK columns exist
        missing_pk = set(pk_cols) - set(df.columns)
        if missing_pk:
            errors.append(f"Primary key columns missing: {missing_pk}")
            return errors
        
        # Check for duplicates
        duplicates = df.duplicated(subset=pk_cols, keep=False)
        if duplicates.any():
            n_dupes = duplicates.sum()
            errors.append(
                f"Primary key violation: {n_dupes} duplicate rows on {pk_cols}"
            )
            
            # Log sample of duplicates
            dupe_sample = df[duplicates].head(5)[pk_cols].to_dict('records')
            logger.warning(f"Sample duplicates: {dupe_sample}")
        
        return errors
    
    def _validate_rules(self, df: pd.DataFrame, schema_name: str) -> List[str]:
        """Validate custom rules from validation section."""
        errors = []
        rules = self.schemas['validation'].get(schema_name, [])
        
        for rule in rules:
            try:
                if "No duplicate" in rule:
                    # Already checked in _validate_primary_key
                    continue
                
                elif "must be a trading day" in rule and 'date' in df.columns:
                    # Would need trading calendar - skip for now or implement
                    pass
                
                elif "adj_close > 0" in rule and 'adj_close' in df.columns:
                    invalid = (df['adj_close'] <= 0) & df['adj_close'].notna()
                    if invalid.any():
                        errors.append(f"adj_close <= 0 for {invalid.sum()} rows")
                
                elif "volume >= 0" in rule and 'volume' in df.columns:
                    invalid = (df['volume'] < 0) & df['volume'].notna()
                    if invalid.any():
                        errors.append(f"volume < 0 for {invalid.sum()} rows")
                
                elif "high >= low" in rule:
                    if 'high' in df.columns and 'low' in df.columns:
                        invalid = (df['high'] < df['low']) & df['high'].notna() & df['low'].notna()
                        if invalid.any():
                            errors.append(f"high < low for {invalid.sum()} rows")
                
                elif "statement_type in" in rule and 'statement_type' in df.columns:
                    valid_types = ['quarterly', 'annual']
                    invalid = ~df['statement_type'].isin(valid_types)
                    if invalid.any():
                        errors.append(
                            f"Invalid statement_type for {invalid.sum()} rows"
                        )
                
                elif "filing_date >= period_end" in rule:
                    if 'filing_date' in df.columns and 'period_end' in df.columns:
                        mask = df['filing_date'].notna() & df['period_end'].notna()
                        invalid = mask & (df['filing_date'] < df['period_end'])
                        if invalid.any():
                            errors.append(
                                f"filing_date < period_end for {invalid.sum()} rows"
                            )
                
                elif "source_ts_price <= date" in rule:
                    if 'source_ts_price' in df.columns and 'date' in df.columns:
                        # Convert to comparable format
                        df_check = df.copy()
                        if df_check['source_ts_price'].dtype == 'datetime64[ns]':
                            df_check['source_ts_price_date'] = df_check['source_ts_price'].dt.date
                        else:
                            df_check['source_ts_price_date'] = pd.to_datetime(
                                df_check['source_ts_price']
                            ).dt.date
                        
                        if df_check['date'].dtype == 'datetime64[ns]':
                            df_check['date_only'] = df_check['date'].dt.date
                        else:
                            df_check['date_only'] = pd.to_datetime(df_check['date']).dt.date
                        
                        invalid = df_check['source_ts_price_date'] > df_check['date_only']
                        if invalid.any():
                            errors.append(
                                f"LEAKAGE: source_ts_price > date for {invalid.sum()} rows"
                            )
                
                elif "source_ts_fund <= date" in rule:
                    if 'source_ts_fund' in df.columns and 'date' in df.columns:
                        mask = df['source_ts_fund'].notna()
                        if mask.any():
                            df_check = df[mask].copy()
                            
                            if df_check['source_ts_fund'].dtype == 'datetime64[ns]':
                                df_check['source_ts_fund_date'] = df_check['source_ts_fund'].dt.date
                            else:
                                df_check['source_ts_fund_date'] = pd.to_datetime(
                                    df_check['source_ts_fund']
                                ).dt.date
                            
                            if df_check['date'].dtype == 'datetime64[ns]':
                                df_check['date_only'] = df_check['date'].dt.date
                            else:
                                df_check['date_only'] = pd.to_datetime(df_check['date']).dt.date
                            
                            invalid = df_check['source_ts_fund_date'] > df_check['date_only']
                            if invalid.any():
                                errors.append(
                                    f"LEAKAGE: source_ts_fund > date for {invalid.sum()} rows"
                                )
                
                elif "valid_from <= date" in rule:
                    if 'valid_from' in df.columns and 'date' in df.columns:
                        mask = df['valid_from'].notna()
                        invalid = mask & (df['valid_from'] > df['date'])
                        if invalid.any():
                            errors.append(
                                f"PIT violation: valid_from > date for {invalid.sum()} rows"
                            )
                
                elif "max(source_ts_price, source_ts_fund) <= date" in rule:
                    if all(c in df.columns for c in ['source_ts_price', 'source_ts_fund', 'date']):
                        df_check = df.copy()
                        
                        # Convert to dates
                        df_check['ts_price_date'] = pd.to_datetime(
                            df_check['source_ts_price']
                        ).dt.date
                        df_check['ts_fund_date'] = pd.to_datetime(
                            df_check['source_ts_fund']
                        ).dt.date
                        df_check['date_only'] = pd.to_datetime(df_check['date']).dt.date
                        
                        # Get max of source timestamps
                        df_check['max_source_ts'] = df_check[['ts_price_date', 'ts_fund_date']].max(axis=1)
                        
                        invalid = df_check['max_source_ts'] > df_check['date_only']
                        if invalid.any():
                            errors.append(
                                f"LEAKAGE: max(source_ts) > date for {invalid.sum()} rows"
                            )
            
            except Exception as e:
                logger.warning(f"Error validating rule '{rule}': {e}")
        
        return errors
    
    def get_schema_info(self, schema_name: str) -> Dict[str, Any]:
        """Get schema information."""
        if schema_name not in self.schemas:
            return {}
        
        schema = self.schemas[schema_name]
        return {
            'description': schema.get('description', ''),
            'primary_key': schema.get('primary_key', []),
            'columns': list(schema.get('columns', {}).keys()),
            'partition_by': schema.get('partition_by')
        }


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
    """
    Convenience function to validate a DataFrame.
    
    Args:
        df: DataFrame to validate
        schema_name: Name of schema in schemas.yaml
    
    Returns:
        Validation result dict
    """
    validator = SchemaValidator()
    return validator.validate(df, schema_name)


def enforce_schema(df: pd.DataFrame, schema_name: str, raise_on_error: bool = True) -> pd.DataFrame:
    """
    Validate DataFrame and optionally raise on errors.
    
    Args:
        df: DataFrame to validate
        schema_name: Schema name
        raise_on_error: If True, raise ValueError on validation errors
    
    Returns:
        Original DataFrame if valid
    
    Raises:
        ValueError: If validation fails and raise_on_error=True
    """
    result = validate_dataframe(df, schema_name)
    
    if not result['valid']:
        error_msg = f"Schema validation failed for '{schema_name}':\n"
        error_msg += "\n".join(f"  - {e}" for e in result['errors'])
        
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)
    
    return df
