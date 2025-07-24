"""
Reusable UI components for the expense forecasting dashboard.
Contains common Streamlit widgets and utility functions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta


def create_sidebar_controls(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create standardized sidebar controls for the dashboard.
    
    Args:
        df: Main expense DataFrame
        
    Returns:
        Dictionary with control values
    """
    st.sidebar.header("Dashboard Controls")
    
    # Activity window control
    activity_window = st.sidebar.slider(
        "Recent activity window (days)", 
        min_value=7, 
        max_value=120, 
        value=70, 
        step=7,
        help="Only categories with expenses in this window will be forecasted"
    )
    
    # Forecast horizon control
    forecast_horizon = st.sidebar.slider(
        "Forecast horizon (days)", 
        min_value=1, 
        max_value=30, 
        value=7, 
        step=1,
        help="Number of days ahead to forecast"
    )
    
    # Spike threshold control
    spike_threshold = st.sidebar.slider(
        "Spike threshold (% error)", 
        min_value=10, 
        max_value=100, 
        value=30, 
        step=5,
        help="Points where |percent error| exceeds this will be annotated on the chart"
    )
    
    return {
        'activity_window': activity_window,
        'forecast_horizon': forecast_horizon,
        'spike_threshold': spike_threshold
    }


def create_category_filter(df: pd.DataFrame, default_selection: Optional[List[str]] = None) -> List[str]:
    """
    Create category filter multiselect widget.
    
    Args:
        df: DataFrame with category column
        default_selection: Default selected categories
        
    Returns:
        List of selected category names
    """
    st.sidebar.header("Category Filter")
    all_categories = sorted(df['category'].unique())
    
    if default_selection is None:
        default_selection = all_categories
    
    selected_categories = st.sidebar.multiselect(
        'Select Categories',
        options=all_categories,
        default=default_selection,
        help="Choose which expense categories to include in analysis"
    )
    
    return selected_categories


def create_date_range_selector(df: pd.DataFrame, months_back: int = 3) -> Dict[str, datetime]:
    """
    Create date range selector with reasonable defaults.
    
    Args:
        df: DataFrame with date column
        months_back: Default months to look back
        
    Returns:
        Dictionary with start_date and end_date
    """
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    default_end = max_date
    default_start = max(min_date, (pd.to_datetime(default_end) - pd.DateOffset(months=months_back)).date())
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            help="Start date for analysis period"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            help="End date for analysis period"
        )
    
    if start_date > end_date:
        st.error("Start date must be before end date")
        st.stop()
    
    return {'start_date': start_date, 'end_date': end_date}


def display_key_metrics(metrics: Dict[str, float], layout: str = "columns") -> None:
    """
    Display key metrics in a standardized format.
    
    Args:
        metrics: Dictionary of metric_name -> value
        layout: Layout style ("columns" or "rows")
    """
    if layout == "columns":
        cols = st.columns(len(metrics))
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, float):
                    if 'percentage' in name.lower() or '%' in name:
                        st.metric(name, f"{value:.1f}%")
                    elif 'error' in name.lower() or 'mae' in name.lower():
                        st.metric(name, f"${value:.2f}")
                    else:
                        st.metric(name, f"{value:.2f}")
                else:
                    st.metric(name, str(value))
    else:
        for name, value in metrics.items():
            st.write(f"**{name}:** {value}")


def create_info_box(title: str, content: str, box_type: str = "info") -> None:
    """
    Create standardized info boxes.
    
    Args:
        title: Box title
        content: Box content
        box_type: Type of box ("info", "success", "warning", "error")
    """
    if box_type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif box_type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif box_type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif box_type == "error":
        st.error(f"**{title}**\n\n{content}")


def create_expandable_help(title: str, content: str) -> None:
    """
    Create expandable help section.
    
    Args:
        title: Help section title
        content: Help content (supports markdown)
    """
    with st.expander(f"ℹ️ {title}"):
        st.markdown(content)


def display_dataframe_with_controls(df: pd.DataFrame, 
                                  title: str = "",
                                  show_download: bool = True,
                                  show_search: bool = True,
                                  height: Optional[int] = None,
                                  download_key: Optional[str] = None) -> None:
    """
    Display DataFrame with standard controls.
    
    Args:
        df: DataFrame to display
        title: Optional title
        show_download: Whether to show download button
        show_search: Whether to show search functionality
        height: Optional fixed height
        download_key: Unique key for download button (auto-generated if None)
    """
    if title:
        st.subheader(title)
    
    if show_download and not df.empty:
        csv = df.to_csv(index=False)
        # Generate unique key if not provided
        if download_key is None:
            download_key = f"download_{title.lower().replace(' ', '_').replace('-', '_')}_{hash(str(df.shape))}"
        
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=download_key
        )
    
    if df.empty:
        st.info("No data to display")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def create_status_indicator(status: str, message: str = "") -> None:
    """
    Create status indicator with appropriate styling.
    
    Args:
        status: Status level ("success", "warning", "error", "info")
        message: Status message
    """
    status_config = {
        "success": {"icon": "✅", "color": "green"},
        "warning": {"icon": "⚠️", "color": "orange"}, 
        "error": {"icon": "❌", "color": "red"},
        "info": {"icon": "ℹ️", "color": "blue"}
    }
    
    config = status_config.get(status, status_config["info"])
    st.markdown(f":{config['color']}[{config['icon']} {message}]")


def create_progress_indicator(current: int, total: int, label: str = "Progress") -> None:
    """
    Create progress indicator.
    
    Args:
        current: Current step
        total: Total steps
        label: Progress label
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")


def create_tabs_with_icons(tab_configs: List[Dict[str, str]]) -> List:
    """
    Create tabs with icons and labels.
    
    Args:
        tab_configs: List of dicts with 'icon' and 'label' keys
        
    Returns:
        List of tab objects
    """
    tab_labels = [f"{config['icon']} {config['label']}" for config in tab_configs]
    return st.tabs(tab_labels)


def format_currency(value: float, show_sign: bool = False) -> str:
    """
    Format currency values consistently.
    
    Args:
        value: Numeric value
        show_sign: Whether to show +/- sign
        
    Returns:
        Formatted currency string
    """
    if show_sign:
        return f"${value:+,.2f}"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format percentage values consistently.
    
    Args:
        value: Numeric value (0-100)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"


def create_metric_card(title: str, value: Union[str, float], 
                      delta: Optional[Union[str, float]] = None,
                      help_text: Optional[str] = None) -> None:
    """
    Create a metric card with optional delta and help.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
        help_text: Optional help text
    """
    if isinstance(value, float):
        if title.lower().endswith(('%', 'percentage', 'rate')):
            value_str = format_percentage(value)
        elif any(word in title.lower() for word in ['amount', 'total', 'cost', 'error', 'mae']):
            value_str = format_currency(value)
        else:
            value_str = f"{value:.2f}"
    else:
        value_str = str(value)
    
    delta_str = None
    if delta is not None:
        if isinstance(delta, float):
            delta_str = format_currency(delta, show_sign=True)
        else:
            delta_str = str(delta)
    
    st.metric(
        label=title,
        value=value_str,
        delta=delta_str,
        help=help_text
    )


def display_method_comparison_table(methods_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Display a comparison table for different methods.
    
    Args:
        methods_data: Dict of method_name -> metrics_dict
    """
    if not methods_data:
        st.info("No methods to compare")
        return
    
    # Convert to DataFrame for easy display
    comparison_data = []
    for method_name, metrics in methods_data.items():
        row = {'Method': method_name}
        row.update(metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format numeric columns
    for col in comparison_df.columns:
        if col != 'Method' and comparison_df[col].dtype in ['float64', 'int64']:
            if 'percentage' in col.lower() or 'accuracy' in col.lower():
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.1f}%")
            elif 'error' in col.lower() or 'mae' in col.lower():
                comparison_df[col] = comparison_df[col].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(comparison_df, use_container_width=True)


def create_section_header(title: str, description: str = "", divider: bool = True) -> None:
    """
    Create standardized section headers.
    
    Args:
        title: Section title
        description: Optional description
        divider: Whether to add a divider after
    """
    st.header(title)
    if description:
        st.caption(description)
    if divider:
        st.divider()


def create_loading_placeholder(message: str = "Loading...") -> Any:
    """
    Create loading placeholder with spinner.
    
    Args:
        message: Loading message
        
    Returns:
        Streamlit spinner context
    """
    return st.spinner(message)