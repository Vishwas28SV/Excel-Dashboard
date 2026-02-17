"""
Interactive Excel Dashboard Generator
A production-ready Streamlit application that converts Excel files into interactive dashboards
with automatic column detection, dynamic filtering, and rich visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Excel Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
""", unsafe_allow_html=True)


class ExcelAnalyzer:
    """Handles Excel file analysis and column type inference."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.date_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self._analyze_columns()
    
    def _analyze_columns(self) -> None:
        """Automatically infer column types from the DataFrame."""
        for col in self.df.columns:
            # Skip completely null columns
            if self.df[col].isna().all():
                continue
            
            # Check for date columns
            if self._is_date_column(col):
                self.date_columns.append(col)
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_columns.append(col)
            # Everything else is categorical
            else:
                # Only consider as categorical if unique values are reasonable
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.5 or self.df[col].nunique() < 50:
                    self.categorical_columns.append(col)
    
    def _is_date_column(self, col: str) -> bool:
        """Determine if a column contains date values."""
        # Check if already datetime type
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return True
        
        # Try to parse as datetime
        try:
            # Sample first non-null value
            sample = self.df[col].dropna().head(10)
            if len(sample) == 0:
                return False
            
            # Try parsing
            pd.to_datetime(sample, errors='raise')
            # If successful, convert the column
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            return True
        except (ValueError, TypeError):
            return False
    
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics for the dataset."""
        stats = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'date_columns': len(self.date_columns),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns)
        }
        return stats


class DashboardGenerator:
    """Generates interactive dashboard components."""
    
    def __init__(self, df: pd.DataFrame, analyzer: ExcelAnalyzer):
        self.df = df
        self.analyzer = analyzer
    
    def create_kpi_cards(self) -> None:
        """Generate KPI metric cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📋 Total Records",
                value=f"{len(self.df):,}"
            )
        
        with col2:
            # Find a rating-like column (numeric, 0-5 or 0-10 range)
            rating_col = self._find_rating_column()
            if rating_col:
                avg_rating = self.df[rating_col].mean()
                st.metric(
                    label=f"⭐ Avg {rating_col}",
                    value=f"{avg_rating:.2f}"
                )
            else:
                # Show average of first numeric column
                if self.analyzer.numeric_columns:
                    first_numeric = self.analyzer.numeric_columns[0]
                    avg_val = self.df[first_numeric].mean()
                    st.metric(
                        label=f"📊 Avg {first_numeric}",
                        value=f"{avg_val:,.2f}"
                    )
        
        with col3:
            # Find customer/user/ID column
            customer_col = self._find_customer_column()
            if customer_col:
                unique_customers = self.df[customer_col].nunique()
                st.metric(
                    label=f"👥 Unique {customer_col}",
                    value=f"{unique_customers:,}"
                )
            else:
                # Show unique count of first categorical
                if self.analyzer.categorical_columns:
                    first_cat = self.analyzer.categorical_columns[0]
                    unique_count = self.df[first_cat].nunique()
                    st.metric(
                        label=f"🏷️ Unique {first_cat}",
                        value=f"{unique_count:,}"
                    )
        
        with col4:
            # Show date range if date columns exist
            if self.analyzer.date_columns:
                date_col = self.analyzer.date_columns[0]
                date_range = (self.df[date_col].max() - self.df[date_col].min()).days
                st.metric(
                    label="📅 Date Range",
                    value=f"{date_range} days"
                )
            else:
                # Show completeness metric
                completeness = (1 - self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
                st.metric(
                    label="✅ Data Completeness",
                    value=f"{completeness:.1f}%"
                )
    
    def _find_rating_column(self) -> Optional[str]:
        """Find a column that looks like a rating."""
        for col in self.analyzer.numeric_columns:
            if any(keyword in col.lower() for keyword in ['rating', 'score', 'review']):
                # Check if values are in typical rating range
                min_val, max_val = self.df[col].min(), self.df[col].max()
                if (0 <= min_val <= 1 and 0 <= max_val <= 5) or (0 <= min_val <= 1 and 0 <= max_val <= 10):
                    return col
        return None
    
    def _find_customer_column(self) -> Optional[str]:
        """Find a column that looks like customer/user identifier."""
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['customer', 'user', 'client', 'name', 'id']):
                # Make sure it has reasonable cardinality
                if self.df[col].nunique() > 1:
                    return col
        return None
    
    def create_categorical_charts(self) -> None:
        """Generate bar charts for categorical distributions."""
        if not self.analyzer.categorical_columns:
            st.info("No categorical columns found for visualization.")
            return
        
        # Limit to top 3 categorical columns for cleaner dashboard
        cats_to_plot = self.analyzer.categorical_columns[:3]
        
        for i, cat_col in enumerate(cats_to_plot):
            # Count values and get top 10
            value_counts = self.df[cat_col].value_counts().head(10)
            
            # Create a DataFrame for Plotly
            chart_data = pd.DataFrame({
                cat_col: value_counts.index,
                'Count': value_counts.values
            })
            
            fig = px.bar(
                chart_data,
                x=cat_col,
                y='Count',
                title=f"Distribution of {cat_col} (Top 10)",
                color='Count',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_time_series_chart(self) -> None:
        """Generate line chart for date-based trends."""
        if not self.analyzer.date_columns:
            st.info("No date columns found for time series visualization.")
            return
        
        date_col = self.analyzer.date_columns[0]
        
        # Aggregate by date
        daily_counts = self.df.groupby(self.df[date_col].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Count']
        
        fig = px.line(
            daily_counts,
            x='Date',
            y='Count',
            title=f"Trend Over Time ({date_col})",
            markers=True
        )
        
        fig.update_layout(
            hovermode='x unified',
            height=400
        )
        
        fig.update_traces(
            line_color='#1f77b4',
            line_width=2,
            marker=dict(size=6)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # If we have numeric columns, show trends for them too
        if self.analyzer.numeric_columns:
            numeric_col = st.selectbox(
                "Select metric to view trend:",
                self.analyzer.numeric_columns,
                key="numeric_trend_selector"
            )
            
            # Aggregate numeric column by date
            daily_avg = self.df.groupby(self.df[date_col].dt.date)[numeric_col].mean().reset_index()
            daily_avg.columns = ['Date', numeric_col]
            
            fig2 = px.line(
                daily_avg,
                x='Date',
                y=numeric_col,
                title=f"Average {numeric_col} Over Time",
                markers=True
            )
            
            fig2.update_layout(
                hovermode='x unified',
                height=400
            )
            
            fig2.update_traces(
                line_color='#ff7f0e',
                line_width=2,
                marker=dict(size=6)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def create_data_preview(self, max_rows: int = 100) -> None:
        """Display data table preview."""
        st.subheader("📄 Data Preview")
        
        # Display info about the dataset
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Showing first {min(max_rows, len(self.df))} of {len(self.df)} rows**")
        with col2:
            st.write(f"**{len(self.df.columns)} columns**")
        
        # Show the data
        st.dataframe(
            self.df.head(max_rows),
            use_container_width=True,
            height=400
        )


class FilterManager:
    """Manages dynamic filtering of the DataFrame."""
    
    def __init__(self, df: pd.DataFrame, analyzer: ExcelAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.filtered_df = df.copy()
    
    def apply_filters(self) -> pd.DataFrame:
        """Apply all selected filters and return filtered DataFrame."""
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        if self.analyzer.date_columns:
            self._apply_date_filter()
        
        # Categorical filters
        if self.analyzer.categorical_columns:
            self._apply_categorical_filters()
        
        # Numeric range filters
        if self.analyzer.numeric_columns:
            self._apply_numeric_filters()
        
        # Show filter summary
        if len(self.filtered_df) < len(self.df):
            st.sidebar.success(f"✅ {len(self.filtered_df):,} / {len(self.df):,} records match filters")
        
        return self.filtered_df
    
    def _apply_date_filter(self) -> None:
        """Apply date range filter."""
        st.sidebar.subheader("📅 Date Range")
        
        date_col = st.sidebar.selectbox(
            "Select date column:",
            self.analyzer.date_columns
        )
        
        min_date = self.df[date_col].min()
        max_date = self.df[date_col].max()
        
        date_range = st.sidebar.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_filter"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (self.filtered_df[date_col] >= pd.to_datetime(start_date)) & \
                   (self.filtered_df[date_col] <= pd.to_datetime(end_date))
            self.filtered_df = self.filtered_df[mask]
    
    def _apply_categorical_filters(self) -> None:
        """Apply categorical filters."""
        st.sidebar.subheader("🏷️ Categories")
        
        # Limit to first 3 categorical columns to avoid clutter
        cats_to_filter = self.analyzer.categorical_columns[:3]
        
        for cat_col in cats_to_filter:
            unique_values = sorted(self.df[cat_col].dropna().unique())
            
            # Only show filter if there are reasonable number of unique values
            if len(unique_values) <= 50:
                selected = st.sidebar.multiselect(
                    f"Filter by {cat_col}:",
                    options=unique_values,
                    key=f"filter_{cat_col}"
                )
                
                if selected:
                    self.filtered_df = self.filtered_df[self.filtered_df[cat_col].isin(selected)]
    
    def _apply_numeric_filters(self) -> None:
        """Apply numeric range filters."""
        st.sidebar.subheader("📊 Numeric Ranges")
        
        # Show expandable section for numeric filters
        with st.sidebar.expander("Advanced Numeric Filters", expanded=False):
            # Limit to first 2 numeric columns
            nums_to_filter = self.analyzer.numeric_columns[:2]
            
            for num_col in nums_to_filter:
                min_val = float(self.df[num_col].min())
                max_val = float(self.df[num_col].max())
                
                selected_range = st.slider(
                    f"{num_col}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"numeric_filter_{num_col}"
                )
                
                self.filtered_df = self.filtered_df[
                    (self.filtered_df[num_col] >= selected_range[0]) &
                    (self.filtered_df[num_col] <= selected_range[1])
                ]


def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and validate Excel file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        df = pd.read_excel(uploaded_file)
        
        # Validate DataFrame
        if df.empty:
            st.error("❌ The uploaded Excel file is empty.")
            return None
        
        if len(df.columns) == 0:
            st.error("❌ The uploaded Excel file has no columns.")
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {str(e)}")
        st.info("💡 Please ensure the file is a valid Excel file (.xlsx) and is not corrupted.")
        return None


def main():
    """Main application function."""
    
    # Header
    st.title("📊 Interactive Excel Dashboard Generator")
    st.markdown("""
    Upload your Excel file and instantly generate an interactive dashboard with automatic visualizations,
    KPIs, and dynamic filtering capabilities.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx)",
        type=['xlsx'],
        help="Upload an Excel file to generate an interactive dashboard"
    )
    
    if uploaded_file is not None:
        # Load the Excel file
        with st.spinner("📥 Loading Excel file..."):
            df = load_excel_file(uploaded_file)
        
        if df is not None:
            # Analyze the data
            with st.spinner("🔍 Analyzing data structure..."):
                analyzer = ExcelAnalyzer(df)
            
            # Show dataset summary
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            with st.expander("ℹ️ Dataset Summary", expanded=False):
                stats = analyzer.get_summary_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{stats['total_records']:,}")
                with col2:
                    st.metric("Total Columns", stats['total_columns'])
                with col3:
                    st.metric("Column Types", 
                             f"{stats['date_columns']}D / {stats['numeric_columns']}N / {stats['categorical_columns']}C")
                
                st.write("**Detected Columns:**")
                st.write(f"- 📅 Date: {', '.join(analyzer.date_columns) if analyzer.date_columns else 'None'}")
                st.write(f"- 🔢 Numeric: {', '.join(analyzer.numeric_columns) if analyzer.numeric_columns else 'None'}")
                st.write(f"- 🏷️ Categorical: {', '.join(analyzer.categorical_columns) if analyzer.categorical_columns else 'None'}")
            
            # Apply filters
            filter_manager = FilterManager(df, analyzer)
            filtered_df = filter_manager.apply_filters()
            
            # Generate dashboard with filtered data
            dashboard = DashboardGenerator(filtered_df, analyzer)
            
            # KPI Section
            st.header("📈 Key Performance Indicators")
            dashboard.create_kpi_cards()
            
            st.markdown("---")
            
            # Charts Section
            st.header("📊 Visualizations")
            
            # Time series (if date columns exist)
            if analyzer.date_columns:
                st.subheader("📈 Time Series Analysis")
                dashboard.create_time_series_chart()
                st.markdown("---")
            
            # Categorical distributions
            if analyzer.categorical_columns:
                st.subheader("📊 Category Distributions")
                dashboard.create_categorical_charts()
                st.markdown("---")
            
            # Data preview
            dashboard.create_data_preview()
            
            # Download filtered data
            st.markdown("---")
            st.subheader("💾 Export Filtered Data")
            
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Upload an Excel file to get started")
        
        st.markdown("""
        ### ✨ Features:
        - 🎯 **Automatic column detection** (dates, numbers, categories)
        - 📊 **Dynamic KPI cards** with intelligent metric selection
        - 📈 **Interactive charts** with Plotly
        - 🔍 **Smart filtering** by date range, categories, and numeric ranges
        - 📱 **Responsive layout** that works on any screen size
        - 💾 **Export capability** for filtered data
        
        ### 📋 Supported Data Types:
        - Date columns (automatically detected and parsed)
        - Numeric columns (for metrics and trends)
        - Categorical columns (for distributions and filtering)
        
        ### 🚀 Example Use Cases:
        - Sales transaction analysis
        - Customer feedback dashboards
        - Product inventory tracking
        - Survey result visualization
        """)


if __name__ == "__main__":
    main()
