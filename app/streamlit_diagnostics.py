"""Enhanced diagnostics page for Streamlit dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_data_path, load_parquet
from app.diagnostics import FundamentalsDiagnostics
from src.backtest.risk_controls import RiskControls


def render_diagnostics_page():
    """Render complete diagnostics page with all checks."""
    st.title("🔍 System Diagnostics")
    
    st.markdown("""
    Comprehensive diagnostics for data quality, PIT integrity, and risk controls.
    """)
    
    # Tabs for different diagnostic categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Fundamentals PIT",
        "📈 Data Quality",
        "⚠️ Risk Controls",
        "🔧 System Health"
    ])
    
    # Tab 1: Fundamentals PIT Diagnostics
    with tab1:
        render_fundamentals_diagnostics()
    
    # Tab 2: Data Quality
    with tab2:
        render_data_quality_diagnostics()
    
    # Tab 3: Risk Controls
    with tab3:
        render_risk_diagnostics()
    
    # Tab 4: System Health
    with tab4:
        render_system_health()


def render_fundamentals_diagnostics():
    """Render fundamentals PIT diagnostics."""
    st.header("Fundamentals Point-in-Time Diagnostics")
    
    # Load data
    data_dir = get_data_path("")
    
    try:
        daily_panel_path = data_dir / 'pit' / 'daily_panel.parquet'
        intervals_path = data_dir / 'pit' / 'fundamentals_intervals.parquet'
        
        if daily_panel_path.exists():
            daily_panel = load_parquet(str(daily_panel_path))
        else:
            st.warning("Daily panel not found. Run `build-pit` flow first.")
            return
        
        if intervals_path.exists():
            intervals = load_parquet(str(intervals_path))
        else:
            intervals = None
        
        # Initialize diagnostics
        diagnostics = FundamentalsDiagnostics(stale_max_days=540)
        
        # Render diagnostics
        has_violations = diagnostics.render_diagnostics_page(
            daily_panel=daily_panel,
            intervals=intervals,
            data_dir=data_dir
        )
        
        # Warning if violations
        if has_violations:
            st.error("⚠️ **CRITICAL**: Leakage violations detected! Training is blocked until resolved.")
        
    except Exception as e:
        st.error(f"Error loading fundamentals data: {e}")


def render_data_quality_diagnostics():
    """Render data quality diagnostics."""
    st.header("Data Quality Checks")
    
    data_dir = get_data_path("")
    
    # Check prices
    st.subheader("📊 Price Data Quality")
    
    try:
        prices_path = data_dir / 'raw' / 'prices_daily.parquet'
        if prices_path.exists():
            prices = load_parquet(str(prices_path))
            
            # Missing values
            st.write("**Missing Values**")
            missing = prices.isnull().sum()
            missing_pct = (missing / len(prices)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values in price data")
            
            # Duplicates
            st.write("**Duplicate Rows**")
            duplicates = prices.duplicated(subset=['symbol', 'date']).sum()
            
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate (symbol, date) pairs")
            else:
                st.success("✅ No duplicate rows")
            
            # Outliers
            st.write("**Price Outliers**")
            
            if 'close' in prices.columns:
                # Compute daily returns
                prices_sorted = prices.sort_values(['symbol', 'date'])
                prices_sorted['return'] = prices_sorted.groupby('symbol')['close'].pct_change()
                
                # Find extreme returns (>20% single day)
                extreme_returns = prices_sorted[prices_sorted['return'].abs() > 0.20]
                
                if len(extreme_returns) > 0:
                    st.warning(f"⚠️ Found {len(extreme_returns)} extreme returns (>20% single day)")
                    st.dataframe(
                        extreme_returns[['symbol', 'date', 'close', 'return']].head(20),
                        use_container_width=True
                    )
                else:
                    st.success("✅ No extreme price movements detected")
        else:
            st.info("No price data available")
    
    except Exception as e:
        st.error(f"Error checking price data: {e}")
    
    st.divider()
    
    # Check features
    st.subheader("🔧 Feature Data Quality")
    
    try:
        features_path = data_dir / 'features' / 'features.parquet'
        if features_path.exists():
            features = load_parquet(str(features_path))
            
            # Missing values heatmap
            st.write("**Feature Completeness**")
            
            # Get feature columns (exclude symbol, date)
            feature_cols = [col for col in features.columns if col not in ['symbol', 'date']]
            
            # Compute completeness
            completeness = {}
            for col in feature_cols:
                completeness[col] = (features[col].notna().sum() / len(features)) * 100
            
            completeness_df = pd.DataFrame({
                'Feature': list(completeness.keys()),
                'Completeness %': list(completeness.values())
            }).sort_values('Completeness %')
            
            # Plot
            fig = px.bar(
                completeness_df,
                x='Completeness %',
                y='Feature',
                orientation='h',
                title='Feature Completeness'
            )
            fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="80% threshold")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Features below threshold
            low_completeness = completeness_df[completeness_df['Completeness %'] < 80]
            if len(low_completeness) > 0:
                st.warning(f"⚠️ {len(low_completeness)} features have <80% completeness")
                st.dataframe(low_completeness, use_container_width=True)
        else:
            st.info("No feature data available")
    
    except Exception as e:
        st.error(f"Error checking feature data: {e}")


def render_risk_diagnostics():
    """Render risk control diagnostics."""
    st.header("Risk Control Diagnostics")
    
    # Load backtest results
    reports_dir = Path(get_data_path("")) / 'reports'
    
    try:
        # Find latest backtest report
        backtest_files = list(reports_dir.glob('backtest_*.json'))
        
        if not backtest_files:
            st.info("No backtest results available. Run a backtest first.")
            return
        
        # Load latest
        latest_backtest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        
        import json
        with open(latest_backtest, 'r') as f:
            backtest_results = json.load(f)
        
        st.write(f"**Latest Backtest**: {latest_backtest.name}")
        
        # Load positions and returns
        positions_path = reports_dir / 'backtest_positions.parquet'
        returns_path = reports_dir / 'backtest_returns.parquet'
        
        if positions_path.exists() and returns_path.exists():
            positions = load_parquet(str(positions_path))
            returns_df = load_parquet(str(returns_path))
            
            # Convert to Series
            returns = returns_df.set_index('date')['portfolio_return']
            
            # Initialize risk controls
            rc = RiskControls()
            
            # Generate risk report
            risk_report = rc.generate_risk_report(
                positions,
                returns,
                weight_col='weight',
                date_col='date'
            )
            
            # Display overall status
            status = risk_report['overall']['status']
            if status == 'PASS':
                st.success(f"✅ **Risk Status**: {status}")
            else:
                st.error(f"⚠️ **Risk Status**: {status}")
            
            st.divider()
            
            # Risk metrics
            st.subheader("📊 Risk Metrics")
            
            metrics = risk_report['risk_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Volatility (Ann.)", f"{metrics['volatility']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            
            with col2:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                st.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
            
            with col3:
                st.metric("VaR 95%", f"{metrics['var_95']*100:.2f}%")
                st.metric("CVaR 95%", f"{metrics['cvar_95']*100:.2f}%")
            
            with col4:
                st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            
            st.divider()
            
            # Exposure check
            st.subheader("📈 Exposure Analysis")
            
            exposure_check = risk_report['exposure_check']
            
            if exposure_check['has_violations']:
                st.error("⚠️ Exposure limit violations detected!")
                
                for violation in exposure_check['violations']:
                    st.write(f"**{violation['type']}**: {violation['count']} violations")
                    st.write(f"Max value: {violation['max_value']:.2f}, Limit: {violation['limit']:.2f}")
            else:
                st.success("✅ No exposure violations")
            
            # Plot exposures over time
            exposures = exposure_check['exposures']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=exposures['date'],
                y=exposures['gross'],
                name='Gross Exposure',
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=exposures['date'],
                y=exposures['net'],
                name='Net Exposure',
                mode='lines'
            ))
            
            fig.update_layout(
                title='Exposure Over Time',
                xaxis_title='Date',
                yaxis_title='Exposure',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Position statistics
            st.subheader("📊 Position Statistics")
            
            pos_stats = risk_report['position_stats']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg # Positions", f"{pos_stats['avg_num_positions']:.0f}")
                st.metric("Max # Positions", f"{pos_stats['max_num_positions']:.0f}")
            
            with col2:
                st.metric("Avg Position Size", f"{pos_stats['avg_position_size']*100:.2f}%")
                st.metric("Max Position Size", f"{pos_stats['max_position_size']*100:.2f}%")
        
        else:
            st.info("Detailed position/return data not available")
    
    except Exception as e:
        st.error(f"Error loading risk diagnostics: {e}")


def render_system_health():
    """Render system health diagnostics."""
    st.header("System Health")
    
    data_dir = get_data_path("")
    
    # Check all artifacts
    artifacts = {
        'Raw Prices': data_dir / 'raw' / 'prices_daily.parquet',
        'Raw Fundamentals': data_dir / 'raw' / 'fundamentals.parquet',
        'PIT Panel': data_dir / 'pit' / 'daily_panel.parquet',
        'Features': data_dir / 'features' / 'features.parquet',
        'Labels': data_dir / 'labels' / 'labels.parquet',
        'Trained Model': data_dir.parent / 'models' / 'registry' / 'latest_model.joblib'
    }
    
    health_data = []
    
    for name, path in artifacts.items():
        if path.exists():
            # Get file info
            mtime = path.stat().st_mtime
            size_mb = path.stat().st_size / (1024 * 1024)
            last_modified = pd.Timestamp.fromtimestamp(mtime)
            
            # Try to get row count
            try:
                if path.suffix == '.parquet':
                    df = load_parquet(str(path))
                    row_count = len(df)
                else:
                    row_count = "N/A"
            except:
                row_count = "Error"
            
            health_data.append({
                'Artifact': name,
                'Status': '✅',
                'Last Modified': last_modified.strftime('%Y-%m-%d %H:%M'),
                'Size (MB)': f"{size_mb:.2f}",
                'Rows': row_count
            })
        else:
            health_data.append({
                'Artifact': name,
                'Status': '❌',
                'Last Modified': 'Never',
                'Size (MB)': '0',
                'Rows': 0
            })
    
    health_df = pd.DataFrame(health_data)
    
    st.dataframe(health_df, use_container_width=True, hide_index=True)
    
    # Overall health score
    available = (health_df['Status'] == '✅').sum()
    total = len(health_df)
    health_score = (available / total) * 100
    
    st.metric("System Health Score", f"{health_score:.0f}%")
    
    if health_score == 100:
        st.success("✅ All systems operational")
    elif health_score >= 80:
        st.warning("⚠️ Some artifacts missing")
    else:
        st.error("❌ Critical artifacts missing")


if __name__ == "__main__":
    render_diagnostics_page()
