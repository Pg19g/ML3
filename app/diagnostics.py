"""Dashboard diagnostics for fundamentals layer."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src.utils import setup_logging

logger = setup_logging(__name__)


class FundamentalsDiagnostics:
    """
    Diagnostics for fundamentals layer.
    
    Provides:
    - Leakage table (rows where max(source_ts_*) > date)
    - Coverage heatmap (symbol × date presence)
    - Fundamentals age histogram
    - Version counts per symbol
    """
    
    def __init__(self, stale_max_days: int = 540):
        """
        Initialize diagnostics.
        
        Args:
            stale_max_days: Threshold for staleness
        """
        self.stale_max_days = stale_max_days
    
    def show_leakage_table(self, daily_panel: pd.DataFrame) -> bool:
        """
        Show leakage violations table.
        
        Args:
            daily_panel: Daily panel with source timestamps
        
        Returns:
            True if violations found, False otherwise
        """
        st.subheader("📊 Leakage Detection")
        
        violations = []
        
        # Check source_ts_price > date
        if 'source_ts_price' in daily_panel.columns:
            price_violations = daily_panel[
                daily_panel['source_ts_price'] > daily_panel['date']
            ]
            
            if len(price_violations) > 0:
                violations.append({
                    'type': 'Price Timestamp',
                    'count': len(price_violations),
                    'data': price_violations
                })
        
        # Check source_ts_fund > date
        if 'source_ts_fund' in daily_panel.columns:
            fund_violations = daily_panel[
                daily_panel['source_ts_fund'] > daily_panel['date']
            ]
            
            if len(fund_violations) > 0:
                violations.append({
                    'type': 'Fundamental Timestamp',
                    'count': len(fund_violations),
                    'data': fund_violations
                })
        
        if violations:
            st.error(f"⚠️ Found {sum(v['count'] for v in violations)} leakage violations!")
            
            for v in violations:
                st.write(f"**{v['type']}**: {v['count']} violations")
                
                # Show sample
                st.dataframe(
                    v['data'].head(20)[[
                        'symbol', 'date', 'source_ts_price', 'source_ts_fund'
                    ]],
                    use_container_width=True
                )
            
            return True
        else:
            st.success("✅ No leakage violations detected")
            return False
    
    def show_coverage_heatmap(self, daily_panel: pd.DataFrame):
        """
        Show coverage heatmap (symbol × date).
        
        Args:
            daily_panel: Daily panel
        """
        st.subheader("🗓️ Data Coverage Heatmap")
        
        # Create pivot table (symbol × date)
        # Value = 1 if data exists, 0 otherwise
        pivot = daily_panel.pivot_table(
            index='symbol',
            columns='date',
            values='close',  # or any column
            aggfunc='count',
            fill_value=0
        )
        
        # Convert to binary (1 if exists, 0 if not)
        pivot = (pivot > 0).astype(int)
        
        # Create heatmap
        fig = px.imshow(
            pivot,
            labels=dict(x="Date", y="Symbol", color="Coverage"),
            color_continuous_scale=['white', 'green'],
            aspect='auto'
        )
        
        fig.update_layout(
            title="Data Coverage by Symbol and Date",
            xaxis_title="Date",
            yaxis_title="Symbol",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage stats
        total_cells = pivot.size
        filled_cells = pivot.sum().sum()
        coverage_pct = (filled_cells / total_cells) * 100
        
        st.metric("Overall Coverage", f"{coverage_pct:.1f}%")
    
    def show_fundamentals_age_histogram(self, daily_panel: pd.DataFrame):
        """
        Show histogram of fundamentals age (days_since_fund).
        
        Args:
            daily_panel: Daily panel with days_since_fund
        """
        st.subheader("📈 Fundamentals Age Distribution")
        
        if 'days_since_fund' not in daily_panel.columns:
            st.warning("No days_since_fund column found")
            return
        
        # Filter out NaN values
        ages = daily_panel['days_since_fund'].dropna()
        
        if len(ages) == 0:
            st.warning("No fundamentals age data available")
            return
        
        # Create histogram
        fig = px.histogram(
            ages,
            nbins=50,
            labels={'value': 'Days Since Fundamental', 'count': 'Frequency'},
            title="Distribution of Fundamental Age"
        )
        
        # Add vertical line for stale_max_days
        fig.add_vline(
            x=self.stale_max_days,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Stale Threshold ({self.stale_max_days} days)"
        )
        
        fig.update_layout(
            xaxis_title="Days Since Fundamental Became Valid",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Age", f"{ages.mean():.0f} days")
        
        with col2:
            st.metric("Median Age", f"{ages.median():.0f} days")
        
        with col3:
            stale_count = (ages > self.stale_max_days).sum()
            stale_pct = (stale_count / len(ages)) * 100
            st.metric("Stale Rows", f"{stale_pct:.1f}%")
    
    def show_version_counts(self, intervals_df: pd.DataFrame):
        """
        Show version counts per symbol (to spot frequent restatements).
        
        Args:
            intervals_df: Fundamentals intervals with version_id
        """
        st.subheader("🔄 Restatement Analysis")
        
        if 'version_id' not in intervals_df.columns:
            st.warning("No version_id column found")
            return
        
        # Count versions per (symbol, statement_type, period_end)
        version_counts = intervals_df.groupby([
            'symbol', 'statement_type', 'period_end'
        ])['version_id'].max().reset_index()
        
        version_counts.columns = ['symbol', 'statement_type', 'period_end', 'num_versions']
        
        # Reports with restatements (num_versions > 1)
        restated = version_counts[version_counts['num_versions'] > 1]
        
        if len(restated) == 0:
            st.info("No restatements detected")
            return
        
        # Show summary
        total_reports = len(version_counts)
        restated_reports = len(restated)
        restatement_rate = (restated_reports / total_reports) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Reports", total_reports)
        
        with col2:
            st.metric("Restatement Rate", f"{restatement_rate:.1f}%")
        
        # Top restating symbols
        st.write("**Top Restating Symbols**")
        
        symbol_restatements = restated.groupby('symbol')['num_versions'].agg(['count', 'sum', 'max'])
        symbol_restatements.columns = ['Reports Restated', 'Total Versions', 'Max Versions']
        symbol_restatements = symbol_restatements.sort_values('Reports Restated', ascending=False)
        
        st.dataframe(
            symbol_restatements.head(10),
            use_container_width=True
        )
        
        # Distribution of version counts
        fig = px.histogram(
            version_counts,
            x='num_versions',
            nbins=10,
            labels={'num_versions': 'Number of Versions', 'count': 'Frequency'},
            title="Distribution of Version Counts"
        )
        
        fig.update_layout(
            xaxis_title="Number of Versions per Report",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_build_timestamps(self, data_dir: Path):
        """
        Show last build timestamps and row counts.
        
        Args:
            data_dir: Data directory path
        """
        st.subheader("⏰ Build Status")
        
        files = {
            'Fundamentals Intervals': data_dir / 'pit' / 'fundamentals_intervals.parquet',
            'Daily Panel': data_dir / 'pit' / 'daily_panel.parquet',
            'Features': data_dir / 'features' / 'features.parquet',
            'Labels': data_dir / 'labels' / 'labels.parquet'
        }
        
        build_info = []
        
        for name, path in files.items():
            if path.exists():
                # Get modification time
                mtime = path.stat().st_mtime
                last_modified = pd.Timestamp.fromtimestamp(mtime)
                
                # Get row count
                try:
                    df = pd.read_parquet(path)
                    row_count = len(df)
                except Exception as e:
                    row_count = "Error"
                
                build_info.append({
                    'Artifact': name,
                    'Last Built': last_modified.strftime('%Y-%m-%d %H:%M:%S'),
                    'Row Count': row_count,
                    'Status': '✅'
                })
            else:
                build_info.append({
                    'Artifact': name,
                    'Last Built': 'Never',
                    'Row Count': 0,
                    'Status': '❌'
                })
        
        st.dataframe(
            pd.DataFrame(build_info),
            use_container_width=True,
            hide_index=True
        )
    
    def render_diagnostics_page(
        self,
        daily_panel: Optional[pd.DataFrame] = None,
        intervals: Optional[pd.DataFrame] = None,
        data_dir: Optional[Path] = None
    ) -> bool:
        """
        Render complete diagnostics page.
        
        Args:
            daily_panel: Daily panel DataFrame
            intervals: Fundamentals intervals DataFrame
            data_dir: Data directory path
        
        Returns:
            True if violations found (should block training), False otherwise
        """
        st.title("🔍 Fundamentals Diagnostics")
        
        has_violations = False
        
        # Build timestamps
        if data_dir:
            self.show_build_timestamps(data_dir)
        
        st.divider()
        
        # Leakage detection
        if daily_panel is not None:
            has_violations = self.show_leakage_table(daily_panel)
            st.divider()
        
        # Coverage heatmap
        if daily_panel is not None:
            self.show_coverage_heatmap(daily_panel)
            st.divider()
        
        # Fundamentals age
        if daily_panel is not None:
            self.show_fundamentals_age_histogram(daily_panel)
            st.divider()
        
        # Version counts
        if intervals is not None:
            self.show_version_counts(intervals)
        
        return has_violations
