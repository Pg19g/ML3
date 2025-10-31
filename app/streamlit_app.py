"""Streamlit dashboard for ML3."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import ModelRegistry
from src.utils import get_data_path, load_parquet, get_reports_path, load_json
from src.features import FeatureEngineer
from src.labels import LabelGenerator

# Page config
st.set_page_config(
    page_title="ML3 Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("ML3 - Point-in-Time ML Market Data Pipeline")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Data", "Features & Labels", "Training", "Backtest", "Model Import", "Diagnostics"]
)


# Helper functions
def run_flow(flow_name, **kwargs):
    """Run a Prefect flow."""
    try:
        if flow_name == "ingest-prices":
            from flows.ingest_prices import ingest_prices_flow
            ingest_prices_flow(**kwargs)
        elif flow_name == "ingest-fundamentals":
            from flows.ingest_fundamentals import ingest_fundamentals_flow
            ingest_fundamentals_flow(**kwargs)
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
            return train_flow(**kwargs)
        elif flow_name == "backtest":
            from flows.backtest import backtest_flow
            return backtest_flow(**kwargs)
        return True
    except Exception as e:
        st.error(f"Error running flow: {e}")
        return False


# Page: Data
if page == "Data":
    st.header("Data Management")
    
    # Data statistics
    st.subheader("Data Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    # Prices
    prices_path = str(get_data_path("raw") / "prices_daily.parquet")
    try:
        prices = load_parquet(prices_path)
        if not prices.empty:
            with col1:
                st.metric("Price Rows", f"{len(prices):,}")
                st.metric("Symbols", prices['symbol'].nunique())
                if 'date' in prices.columns:
                    st.metric("Date Range", f"{prices['date'].min()} to {prices['date'].max()}")
        else:
            with col1:
                st.info("No price data available")
    except:
        with col1:
            st.info("No price data available")
    
    # Fundamentals
    fund_path = str(get_data_path("raw") / "fundamentals.parquet")
    try:
        fund = load_parquet(fund_path)
        if not fund.empty:
            with col2:
                st.metric("Fundamental Rows", f"{len(fund):,}")
                st.metric("Symbols", fund['symbol'].nunique())
        else:
            with col2:
                st.info("No fundamental data available")
    except:
        with col2:
            st.info("No fundamental data available")
    
    # Features
    features_path = str(get_data_path("pit") / "features.parquet")
    try:
        features = load_parquet(features_path)
        if not features.empty:
            with col3:
                st.metric("Feature Rows", f"{len(features):,}")
                st.metric("Features", len(features.columns))
        else:
            with col3:
                st.info("No features available")
    except:
        with col3:
            st.info("No features available")
    
    st.divider()
    
    # Data operations
    st.subheader("Data Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ingestion**")
        if st.button("Refresh Prices"):
            with st.spinner("Ingesting prices..."):
                success = run_flow("ingest-prices", incremental=True)
                if success:
                    st.success("Prices refreshed!")
                    st.rerun()
        
        if st.button("Refresh Fundamentals"):
            with st.spinner("Ingesting fundamentals..."):
                success = run_flow("ingest-fundamentals", incremental=True)
                if success:
                    st.success("Fundamentals refreshed!")
                    st.rerun()
    
    with col2:
        st.write("**Processing**")
        if st.button("Build PIT Panel"):
            with st.spinner("Building PIT panel..."):
                success = run_flow("build-pit")
                if success:
                    st.success("PIT panel built!")
                    st.rerun()
        
        if st.button("Build Features"):
            with st.spinner("Building features..."):
                success = run_flow("build-features")
                if success:
                    st.success("Features built!")
                    st.rerun()
        
        if st.button("Build Labels"):
            with st.spinner("Building labels..."):
                success = run_flow("build-labels")
                if success:
                    st.success("Labels built!")
                    st.rerun()
    
    st.divider()
    
    # PIT Integrity Checks
    st.subheader("PIT Integrity Checks")
    
    try:
        pit_path = str(get_data_path("pit") / "daily_panel.parquet")
        pit_panel = load_parquet(pit_path)
        
        if not pit_panel.empty:
            from src.pit import PITProcessor
            processor = PITProcessor()
            integrity = processor.check_pit_integrity(pit_panel)
            
            if integrity['passed']:
                st.success("✅ PIT integrity checks PASSED")
            else:
                st.error(f"❌ PIT integrity checks FAILED: {integrity['violations']} violations")
            
            # Show check details
            for check in integrity.get('checks', []):
                status = "✅" if check['passed'] else "❌"
                st.write(f"{status} {check['name']}: {check['violations']} violations")
        else:
            st.info("No PIT panel available")
    except Exception as e:
        st.warning(f"Could not check PIT integrity: {e}")


# Page: Features & Labels
elif page == "Features & Labels":
    st.header("Features & Labels")
    
    # Load feature config
    engineer = FeatureEngineer()
    generator = LabelGenerator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Features")
        
        # Technical features
        st.write("**Technical Features**")
        tech_features = engineer.technical_features
        for feat in tech_features[:10]:  # Show first 10
            enabled = feat.get('enabled', True)
            st.checkbox(feat['name'], value=enabled, key=f"tech_{feat['name']}")
        
        if len(tech_features) > 10:
            st.write(f"... and {len(tech_features) - 10} more")
        
        # Fundamental features
        st.write("**Fundamental Features**")
        fund_features = engineer.fundamental_features
        for feat in fund_features[:10]:  # Show first 10
            enabled = feat.get('enabled', True)
            st.checkbox(feat['name'], value=enabled, key=f"fund_{feat['name']}")
        
        if len(fund_features) > 10:
            st.write(f"... and {len(fund_features) - 10} more")
    
    with col2:
        st.subheader("Labels")
        
        labels = generator.labels
        for label in labels:
            enabled = label.get('enabled', True)
            st.checkbox(
                f"{label['name']} (horizon={label['horizon']})",
                value=enabled,
                key=f"label_{label['name']}"
            )
    
    st.divider()
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    try:
        features_path = str(get_data_path("pit") / "features.parquet")
        features = load_parquet(features_path)
        
        if not features.empty:
            feature_cols = engineer.get_feature_names()
            available_features = [f for f in feature_cols if f in features.columns]
            
            if available_features:
                selected_feature = st.selectbox("Select feature", available_features)
                
                # Plot distribution
                fig = px.histogram(
                    features,
                    x=selected_feature,
                    nbins=50,
                    title=f"Distribution of {selected_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("**Summary Statistics**")
                st.write(features[selected_feature].describe())
        else:
            st.info("No features available. Build features first.")
    except Exception as e:
        st.error(f"Error loading features: {e}")


# Page: Training
elif page == "Training":
    st.header("Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        model_type = st.selectbox("Model Type", ["lightgbm", "xgboost"])
        
        st.write("**Cross-Validation**")
        st.write("- Method: Purged K-Fold")
        st.write("- Splits: 5")
        st.write("- Embargo: 21 trading days")
        
        if st.button("Start Training", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                model_id = run_flow("train", model_type=model_type)
                if model_id:
                    st.success(f"Training complete! Model ID: {model_id}")
                    st.session_state['last_trained_model'] = model_id
    
    with col2:
        st.subheader("Recent Models")
        
        registry = ModelRegistry()
        models = registry.list_models()
        
        if models:
            for model in models[:5]:  # Show last 5
                with st.expander(f"{model['model_id']} ({model['model_type']})"):
                    st.write(f"Created: {model.get('created_at', 'N/A')}")
                    st.write(f"Features: {len(model.get('feature_cols', []))}")
                    
                    metrics = model.get('metrics', {})
                    if metrics:
                        st.write("**Metrics:**")
                        for key, value in list(metrics.items())[:5]:
                            st.write(f"- {key}: {value:.4f}")
        else:
            st.info("No models trained yet")
    
    st.divider()
    
    # Model results
    if 'last_trained_model' in st.session_state:
        st.subheader("Training Results")
        
        model_id = st.session_state['last_trained_model']
        
        # Load metrics
        reports_path = get_reports_path()
        metrics_file = reports_path / f"{model_id}_metrics.json"
        
        if metrics_file.exists():
            metrics = load_json(str(metrics_file))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("IC", f"{metrics.get('test_ic', 0):.4f}")
            with col2:
                st.metric("Rank IC", f"{metrics.get('test_rank_ic', 0):.4f}")
            with col3:
                st.metric("RMSE", f"{metrics.get('test_rmse', 0):.4f}")
            with col4:
                st.metric("R²", f"{metrics.get('test_r2', 0):.4f}")


# Page: Backtest
elif page == "Backtest":
    st.header("Backtesting")
    
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        st.info("No models available. Train a model first.")
    else:
        model_ids = [m['model_id'] for m in models]
        selected_model = st.selectbox("Select Model", model_ids)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Configuration")
            st.write("- Type: Rank-based")
            st.write("- Long: Top 10%")
            st.write("- Short: None (long-only)")
            st.write("- Rebalance: Weekly")
        
        with col2:
            st.subheader("Costs")
            st.write("- Commission: 0.1%")
            st.write("- Slippage: 0.05%")
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                results = run_flow("backtest", model_id=selected_model)
                if results:
                    st.success("Backtest complete!")
                    st.session_state['backtest_results'] = results
                    st.session_state['backtest_model'] = selected_model
        
        st.divider()
        
        # Show results
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            metrics = results['metrics']
            
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:.2%}")
                st.metric("CAGR", f"{metrics['cagr']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
            with col4:
                st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                st.metric("Avg Turnover", f"{metrics['avg_turnover']:.2f}")
            
            st.divider()
            
            # Equity curve
            st.subheader("Equity Curve")
            
            portfolio_returns = results['portfolio_returns']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_returns['date'],
                y=portfolio_returns['cumulative_return'],
                mode='lines',
                name='Strategy'
            ))
            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)


# Page: Model Import
elif page == "Model Import":
    st.header("Import External Model")
    
    st.write("""
    Upload an external model (ONNX or joblib/pickle format) to deploy in the pipeline.
    The model must match the feature schema defined in the configuration.
    """)
    
    uploaded_file = st.file_uploader("Upload Model File", type=['onnx', 'pkl', 'joblib'])
    
    if uploaded_file:
        model_id = st.text_input("Model ID", value=f"external_{uploaded_file.name.split('.')[0]}")
        
        # Feature validation
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_names()
        
        st.write(f"**Required Features ({len(feature_cols)}):**")
        st.write(", ".join(feature_cols[:10]) + "...")
        
        generator = LabelGenerator()
        label_col = generator.get_primary_label()
        st.write(f"**Target Label:** {label_col}")
        
        if st.button("Import Model"):
            # Save uploaded file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Import model
            registry = ModelRegistry()
            try:
                imported_id = registry.import_external_model(
                    tmp_path,
                    model_id,
                    feature_cols,
                    label_col
                )
                st.success(f"Model imported successfully: {imported_id}")
                
                # Validate schema
                validation = registry.validate_model_schema(imported_id, feature_cols)
                
                if validation['valid']:
                    st.success("✅ Model schema validated")
                else:
                    st.warning("⚠️ Schema validation issues:")
                    if validation['missing_features']:
                        st.write(f"Missing features: {validation['missing_features']}")
                
            except Exception as e:
                st.error(f"Error importing model: {e}")


# Page: Diagnostics
elif page == "Diagnostics":
    st.header("Diagnostics")
    
    # Data staleness
    st.subheader("Data Staleness")
    
    try:
        features_path = str(get_data_path("pit") / "features.parquet")
        features = load_parquet(features_path)
        
        if not features.empty and 'is_stale_fund' in features.columns:
            stale_pct = features['is_stale_fund'].mean() * 100
            
            if stale_pct > 30:
                st.warning(f"⚠️ {stale_pct:.1f}% of fundamental data is stale")
            else:
                st.success(f"✅ {stale_pct:.1f}% of fundamental data is stale")
            
            # Staleness by symbol
            stale_by_symbol = features.groupby('symbol')['is_stale_fund'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=stale_by_symbol.head(20).index,
                y=stale_by_symbol.head(20).values * 100,
                title="Top 20 Symbols by Staleness %",
                labels={'x': 'Symbol', 'y': 'Staleness %'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No staleness data available")
    except Exception as e:
        st.error(f"Error loading staleness data: {e}")
    
    st.divider()
    
    # Coverage heatmap
    st.subheader("Data Coverage")
    
    try:
        features_path = str(get_data_path("pit") / "features.parquet")
        features = load_parquet(features_path)
        
        if not features.empty:
            # Missing data by feature
            engineer = FeatureEngineer()
            feature_cols = engineer.get_feature_names()
            available_features = [f for f in feature_cols if f in features.columns]
            
            missing_pct = features[available_features].isna().mean() * 100
            missing_pct = missing_pct.sort_values(ascending=False)
            
            fig = px.bar(
                x=missing_pct.head(20).index,
                y=missing_pct.head(20).values,
                title="Top 20 Features by Missing Data %",
                labels={'x': 'Feature', 'y': 'Missing %'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No coverage data available")
    except Exception as e:
        st.error(f"Error loading coverage data: {e}")
