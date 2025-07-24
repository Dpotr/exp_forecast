"""
Dashboard sections for the expense forecasting application.
Contains modular sections that can be composed into the main dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .components import (
    create_section_header, display_key_metrics, create_info_box,
    display_dataframe_with_controls, create_expandable_help,
    display_method_comparison_table, create_metric_card
)
from .charts import (
    create_forecast_chart, create_error_chart, create_category_comparison_chart,
    create_seasonality_heatmap, create_stacked_forecast_chart
)
import plotly.express as px
from anomaly_utils import (
    get_comprehensive_anomalies, create_anomaly_visualization, 
    create_anomaly_summary_table, detect_outliers, detect_anomaly_transactions, 
    recurring_payments
)
from forecast_metrics import calculate_forecast_metrics


def render_data_overview_section(df: pd.DataFrame) -> None:
    """
    Render the data overview section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📊 Data Overview", 
        "Summary of your expense data and basic statistics"
    )
    
    if df.empty:
        create_info_box("No Data", "No expense data found. Please upload your data.", "warning")
        return
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Records", len(df))
    with col2:
        create_metric_card("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    with col3:
        create_metric_card("Categories", df['category'].nunique())
    with col4:
        create_metric_card("Total Amount", df['amount'].sum())
    
    # Recent data preview
    st.subheader("Recent Transactions")
    recent_data = df.head(10)
    display_dataframe_with_controls(recent_data, show_download=False, download_key="recent_data")
    
    # Category breakdown
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    fig = create_category_comparison_chart(
        {cat: total for cat, total in category_totals.head(10).items()},
        chart_type="pie",
        title="Top 10 Categories by Total Amount"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_detection_section(df: pd.DataFrame) -> None:
    """
    Render the enhanced anomaly detection section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "🚨 Enhanced Anomaly & Outlier Detection", 
        "Comprehensive anomaly detection using multiple statistical methods"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for anomaly detection.", "info")
        return
    
    # Get comprehensive anomaly results
    anomalies_dict = get_comprehensive_anomalies(df)
    
    # Create tabs for different views
    tab_configs = [
        {"icon": "📊", "label": "Visual Dashboard"},
        {"icon": "📋", "label": "Summary Table"},
        {"icon": "🔍", "label": "Detailed Results"}
    ]
    tabs = st.tabs([f"{config['icon']} {config['label']}" for config in tab_configs])
    
    with tabs[0]:  # Visual Dashboard
        create_expandable_help(
            "How to interpret anomaly detection methods",
            """
            **Z-Score Method**: Uses standard deviation to find outliers. Points >3 standard deviations from mean are flagged.
            - *Best for*: Normal distributions, detecting extreme values
            - *Limitation*: Sensitive to other outliers
            
            **IQR Method**: Uses interquartile range (Q3-Q1). Points beyond Q1-1.5×IQR or Q3+1.5×IQR are outliers.
            - *Best for*: Skewed data, robust to extreme values
            - *Limitation*: May miss subtle anomalies
            
            **Modified Z-Score**: Uses median and MAD (median absolute deviation) instead of mean/std.
            - *Best for*: Data with outliers already present
            - *Limitation*: More conservative, may miss some anomalies
            
            **Seasonal Method**: Analyzes anomalies within each day-of-week pattern.
            - *Best for*: Regular weekly spending patterns
            - *Limitation*: Requires sufficient data per weekday
            
            **Category Enhanced**: Combines Z-score and IQR methods per spending category.
            - *Best for*: Transaction-level analysis
            - *Limitation*: Needs minimum transactions per category
            """
        )
        
        anomaly_fig = create_anomaly_visualization(df, anomalies_dict)
        st.plotly_chart(anomaly_fig, use_container_width=True)
    
    with tabs[1]:  # Summary Table
        st.subheader("Anomaly Detection Summary")
        summary_table = create_anomaly_summary_table(anomalies_dict)
        
        if not summary_table.empty:
            display_dataframe_with_controls(summary_table, show_download=True, download_key="anomaly_summary_table")
            
            # Quick stats
            total_anomalies = summary_table['Count'].sum()
            create_metric_card("Total Anomalies Detected", total_anomalies)
        else:
            create_info_box("No Anomalies", "No anomalies detected across all methods.", "success")
    
    with tabs[2]:  # Detailed Results
        st.subheader("Detailed Anomaly Results")
        
        # Legacy outlier detection (for backward compatibility)
        outlier_days = detect_outliers(df)
        if not outlier_days.empty:
            st.subheader("Outlier Days (Z-Score Method)")
            st.caption("Days with total spending far from the 7-day rolling mean")
            display_dataframe_with_controls(outlier_days, show_download=True, download_key="outlier_days")
        
        # Legacy transaction anomalies
        anomaly_tx = detect_anomaly_transactions(df)
        if not anomaly_tx.empty:
            st.subheader("Anomalous Transactions (Legacy Method)")
            st.caption("Transactions with unusually high/low amounts for their category")
            display_dataframe_with_controls(anomaly_tx, show_download=True, download_key="anomaly_transactions")
        
        # Enhanced results by method
        for method_name, anomaly_df in anomalies_dict.items():
            if not anomaly_df.empty:
                method_display = method_name.replace('_', ' ').title()
                st.subheader(f"{method_display} Results")
                display_dataframe_with_controls(anomaly_df, show_download=True, download_key=f"anomaly_{method_name}")


def render_recurring_payments_section(df: pd.DataFrame) -> None:
    """
    Render the recurring payments detection section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "🔄 Recurring Payments & Alerts", 
        "Auto-detected subscriptions and recurring expenses"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for recurring payment detection.", "info")
        return
    
    rec_df = recurring_payments(df)
    
    if not rec_df.empty:
        st.subheader("Detected Recurring Payments")
        st.caption("Auto-detected subscriptions/rent with interval and next due date")
        display_dataframe_with_controls(rec_df, show_download=True, download_key="recurring_payments")
        
        # Check for overdue payments
        today = pd.to_datetime(df['date'].max())
        overdue = rec_df[(today - rec_df['last_date']).dt.days > rec_df['interval_days']]
        
        if not overdue.empty:
            st.warning(f"⚠️ Some recurring payments may be overdue: {', '.join(overdue['category'].tolist())}")
        else:
            create_info_box("All Up to Date", "No overdue recurring payments detected.", "success")
    else:
        create_info_box("No Recurring Patterns", "No recurring payments detected in your data.", "info")


def render_seasonality_section(df: pd.DataFrame) -> None:
    """
    Render the seasonality analysis section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📅 Seasonality Analysis", 
        "Spending patterns by day of week and month"
    )
    
    if df.empty or len(df) < 30:
        create_info_box("Insufficient Data", "Need at least 30 days of data for seasonality analysis.", "info")
        return
    
    # Filter to last 60 days for seasonality
    recent_cutoff = df['date'].max() - timedelta(days=60)
    recent_df = df[df['date'] >= recent_cutoff].copy()
    
    if len(recent_df) < 10:
        create_info_box("Insufficient Recent Data", "Need more recent data for seasonality analysis.", "info")
        return
    
    # Create seasonality heatmap
    fig = create_seasonality_heatmap(
        recent_df, 
        value_col="amount",
        title="Spending Patterns (Last 60 Days)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average by Day of Week")
        dow_avg = recent_df.groupby(recent_df['date'].dt.day_name())['amount'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig_dow = go.Figure(data=[
            go.Bar(x=dow_avg.index, y=dow_avg.values, 
                   hovertemplate='<b>%{x}</b><br>Avg: $%{y:.2f}<extra></extra>')
        ])
        fig_dow.update_layout(title="Average Daily Spending by Day of Week", height=400)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    with col2:
        st.subheader("Average by Day of Month")
        dom_avg = recent_df.groupby(recent_df['date'].dt.day)['amount'].mean()
        
        fig_dom = go.Figure(data=[
            go.Scatter(x=dom_avg.index, y=dom_avg.values, mode='lines+markers',
                      hovertemplate='<b>Day %{x}</b><br>Avg: $%{y:.2f}<extra></extra>')
        ])
        fig_dom.update_layout(title="Average Daily Spending by Day of Month", height=400)
        st.plotly_chart(fig_dom, use_container_width=True)


def render_forecast_performance_section(df: pd.DataFrame, 
                                      backward_results: pd.DataFrame,
                                      controls: Dict[str, Any]) -> None:
    """
    Render the comprehensive forecast performance section.
    
    Args:
        df: Main expense DataFrame
        backward_results: Backward forecast results
        controls: Dashboard controls dictionary
    """
    create_section_header(
        "📈 Comprehensive Forecast Performance Analysis", 
        "Detailed accuracy metrics and performance assessment"
    )
    
    if backward_results.empty:
        create_info_box("No Results", "No forecast performance data available.", "info")
        return
    
    # Calculate comprehensive metrics
    comprehensive_metrics = calculate_forecast_metrics(
        actual=backward_results['actual'].values,
        forecast=backward_results['forecast'].values,
        dates=backward_results['date'].values
    )
    
    # Create tabs for different metric categories
    tab_configs = [
        {"icon": "📈", "label": "Basic Metrics"},
        {"icon": "📊", "label": "Advanced Metrics"},
        {"icon": "🎯", "label": "Quality Assessment"},
        {"icon": "📅", "label": "Time Analysis"}
    ]
    tabs = st.tabs([f"{config['icon']} {config['label']}" for config in tab_configs])
    
    with tabs[0]:  # Basic Metrics
        st.write("**Core Accuracy Metrics**")
        basic_metrics = comprehensive_metrics['basic_metrics']
        percentage_metrics = comprehensive_metrics['percentage_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_metric_card("MAE", basic_metrics['mae'])
            create_metric_card("RMSE", basic_metrics['rmse'])
        with col2:
            create_metric_card("MAPE", f"{percentage_metrics['mape']:.1f}%")
            create_metric_card("SMAPE", f"{percentage_metrics['smape']:.1f}%")
        with col3:
            create_metric_card("Mean Error (Bias)", basic_metrics['me'])
            create_metric_card("Max Error", basic_metrics['max_error'])
        with col4:
            create_metric_card("Correlation", f"{comprehensive_metrics['summary_stats']['correlation']:.3f}")
            create_metric_card("R²", f"{comprehensive_metrics['summary_stats']['r_squared']:.3f}")
    
    with tabs[1]:  # Advanced Metrics
        st.write("**Directional & Distribution Metrics**")
        directional_metrics = comprehensive_metrics['directional_metrics']
        distribution_metrics = comprehensive_metrics['distribution_metrics']
        scaled_metrics = comprehensive_metrics['scaled_metrics']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Directional Accuracy**")
            create_metric_card("Direction Correct", f"{directional_metrics['directional_accuracy']:.1f}%")
            create_metric_card("Hit Rate (±10%)", f"{directional_metrics['hit_rate_10pct']:.1f}%")
            create_metric_card("Hit Rate (±20%)", f"{directional_metrics['hit_rate_20pct']:.1f}%")
        
        with col2:
            st.write("**Forecast Bias**")
            create_metric_card("Overforecast %", f"{distribution_metrics['overforecast_percentage']:.1f}%")
            create_metric_card("Underforecast %", f"{distribution_metrics['underforecast_percentage']:.1f}%")
            create_metric_card("Theil's U", f"{distribution_metrics['theil_u']:.3f}")
        
        with col3:
            st.write("**Scale-Independent**")
            if not np.isinf(scaled_metrics['mase']):
                create_metric_card("MASE", f"{scaled_metrics['mase']:.3f}")
            else:
                create_metric_card("MASE", "N/A")
            cv_rmse_val = scaled_metrics['cv_rmse']
            create_metric_card("CV-RMSE", f"{cv_rmse_val:.3f}" if not np.isinf(cv_rmse_val) else "N/A")
    
    with tabs[2]:  # Quality Assessment
        st.write("**Overall Quality Assessment**")
        assessment = comprehensive_metrics['overall_assessment']
        
        # Quality rating with color coding
        rating_colors = {
            'excellent': 'green',
            'good': 'blue', 
            'fair': 'orange',
            'poor': 'red'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            rating = assessment['overall_rating']
            color = rating_colors.get(rating, 'gray')
            st.markdown(f"**Overall Rating:** :{color}[{rating.upper()}]")
            create_metric_card("Quality Score", f"{assessment['overall_score']:.2f}/4.0")
            
            st.write("**Individual Ratings:**")
            st.write(f"• MAPE: {assessment['mape_rating']}")
            st.write(f"• Direction: {assessment['directional_rating']}")  
            st.write(f"• Hit Rate: {assessment['hit_rate_rating']}")
        
        with col2:
            st.write("**Recommendations:**")
            for i, rec in enumerate(assessment['recommendations'], 1):
                st.write(f"{i}. {rec}")
    
    with tabs[3]:  # Time Analysis
        if 'time_metrics' in comprehensive_metrics:
            st.write("**Time-Based Performance Analysis**")
            time_metrics = comprehensive_metrics['time_metrics']
            
            col1, col2 = st.columns(2)
            with col1:
                create_metric_card("Recent Performance", f"${time_metrics['recent_mae']:.2f}")
                create_metric_card("Historical Performance", f"${time_metrics['historical_mae']:.2f}")
                
            with col2:
                improvement = time_metrics['improvement_over_time']
                create_metric_card("Improvement Over Time", f"{improvement:+.1f}%")
            
            # Monthly and day-of-week performance
            if 'monthly_performance' in time_metrics and time_metrics['monthly_performance']:
                st.write("**Monthly Performance (MAE)**")
                monthly_perf = time_metrics['monthly_performance']['mae']
                if monthly_perf:
                    monthly_df = pd.DataFrame(list(monthly_perf.items()), columns=['Month', 'MAE'])
                    monthly_df['Month'] = monthly_df['Month'].astype(str)
                    display_dataframe_with_controls(monthly_df, show_download=False, download_key="monthly_performance")
            
            if 'day_of_week_performance' in time_metrics and time_metrics['day_of_week_performance']:
                st.write("**Day of Week Performance (MAE)**")
                dow_perf = time_metrics['day_of_week_performance']['mae']
                if dow_perf:
                    dow_df = pd.DataFrame(list(dow_perf.items()), columns=['Day', 'MAE'])
                    display_dataframe_with_controls(dow_df, show_download=False, download_key="dow_performance")
        else:
            create_info_box("Time Analysis", "Time-based analysis not available - insufficient date information", "info")


def render_expense_breakdown_section(df: pd.DataFrame) -> None:
    """
    Render the expense breakdown and cumulative totals section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📊 Expense Breakdown & Cumulative Totals",
        "Category analysis and spending trends over time"
    )
    
    if df.empty:
        create_info_box("No Data", "No expense data available for breakdown analysis.", "info")
        return
    
    # Top categories pie chart
    cat_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    fig_pie = create_category_comparison_chart(
        {cat: total for cat, total in cat_sum.items()},
        chart_type="pie",
        title="Top Categories (Total)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Running total/cumulative chart
    df_sorted = df.sort_values('date')
    df_sorted['cumulative'] = df_sorted['amount'].cumsum()
    st.subheader("Cumulative Spending Over Time")
    st.line_chart(df_sorted.set_index('date')['cumulative'], use_container_width=True)


def render_comparisons_section(df: pd.DataFrame) -> None:
    """
    Render comparisons and benchmarks section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📈 Comparisons & Benchmarks",
        "Month-on-month and week-on-week spending comparisons"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for comparison analysis.", "info")
        return
    
    # Month-on-Month comparison
    st.subheader("Month-on-Month Comparison")
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['date'].dt.to_period('M').astype(str)
    monthly_totals = df_temp.groupby('year_month')['amount'].sum().reset_index()
    st.bar_chart(monthly_totals.set_index('year_month'), use_container_width=True)
    
    # Week-on-Week comparison
    st.subheader("Week-on-Week Comparison")
    df_temp['year_week'] = df_temp['date'].dt.strftime('%Y-%U')
    weekly_totals = df_temp.groupby('year_week')['amount'].sum().reset_index()
    st.line_chart(weekly_totals.set_index('year_week'), use_container_width=True)


def render_export_section(df: pd.DataFrame, config) -> None:
    """
    Render export and sharing section.
    
    Args:
        df: Main expense DataFrame
        config: Configuration object
    """
    create_section_header(
        "💾 Export & Sharing",
        "Download and export your expense data"
    )
    
    if st.button("Download All Data as CSV", key="export_all_data"):
        export_path = str(config.EXPENSES_EXPORT_FILE)
        df.to_csv(export_path, index=False)
        st.success(f"Exported to {export_path}")


def render_user_guidance_section() -> None:
    """
    Render user guidance and explanations section.
    """
    create_section_header(
        "📚 User Guidance & Explanations",
        "Tips and explanations for using the dashboard"
    )
    
    create_info_box(
        "Dashboard Usage Tips",
        "Hover over any chart or table for tooltips. Outliers/anomalies use z-score > 3. "
        "Recurring payments are detected if intervals are regular. Forecast methods are explained "
        "in the diagnostics table. For details, see the README.",
        "info"
    )


def render_monthly_expenses_section(df: pd.DataFrame) -> None:
    """
    Render the monthly expenses section with stacked charts and period analysis.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📅 Monthly Expenses",
        "Monthly spending analysis with category breakdown and period regimes"
    )
    
    if df.empty:
        create_info_box("No Data", "No expense data available for monthly analysis.", "info")
        return
    
    # Basic period aggregation
    st.subheader("Expenses by Category and Period")
    agg = df.groupby(['period', 'category'])['amount'].sum().reset_index()
    st.caption("Total expenses by category and period")
    display_dataframe_with_controls(agg, show_download=True, download_key="monthly_period_agg")
    
    # Complex stacked chart with period highlighting
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['date'].dt.to_period('M').astype(str)
    monthly = df_temp.groupby(['year_month', 'category', 'period'])['amount'].sum().reset_index()
    
    # Pivot for stacked bar
    pivot = monthly.pivot_table(index=['year_month', 'period'], columns='category', values='amount', fill_value=0)
    pivot = pivot.reset_index()
    
    # Prepare for plotting
    categories = [c for c in pivot.columns if c not in ['year_month','period']]
    
    fig = go.Figure()
    
    for cat in categories:
        fig.add_trace(go.Bar(
            x=pivot['year_month'],
            y=pivot[cat],
            name=cat,
            marker_line_width=0,
        ))
    
    # Highlight periods with colored backgrounds
    history_periods = [
        {"name": "Operations", "start": "2023-04", "end": "2023-12", "color": "rgba(0,200,0,0.1)"},
        {"name": "Overstock", "start": "2023-12", "end": "2024-05", "color": "rgba(0,0,200,0.1)"},
        {"name": "Regular", "start": "2024-05", "end": df_temp['year_month'].max(), "color": "rgba(200,0,0,0.1)"},
    ]
    
    for period in history_periods:
        fig.add_vrect(
            x0=period['start'], x1=period['end'],
            fillcolor=period['color'], opacity=0.3, layer="below", line_width=0,
            annotation_text=period['name'], annotation_position="top left"
        )
    
    fig.update_layout(
        title="Monthly Expenses by Category (Stacked)",
        xaxis_title="Month",
        yaxis_title="Amount",
        barmode='stack',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_trendline_section(df: pd.DataFrame) -> None:
    """
    Render the trendline analysis section.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "📈 Trendline Analysis",
        "Last 90 days expenses with moving average trend"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for trendline analysis.", "info")
        return
    
    # Last 90 Days with Trendline
    st.subheader("Expenses: Last 90 Days (with Trendline)")
    last_90 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=90))]
    
    if len(last_90) < 7:
        create_info_box("Insufficient Data", "Need at least 7 days of data for trendline analysis.", "info")
        return
    
    daily = last_90.groupby('date')['amount'].sum().reset_index()
    
    # Calculate moving average as trendline
    window = 7  # 7-day moving average
    if len(daily) >= window:
        daily['trend'] = daily['amount'].rolling(window=window, min_periods=1).mean()
    else:
        daily['trend'] = daily['amount']
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=daily['date'], y=daily['amount'], 
        name='Daily Expenses', 
        marker_color='lightblue'
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily['date'], y=daily['trend'], 
        name=f'{window}-Day Moving Avg', 
        line=dict(color='red', width=2)
    ))
    
    fig_trend.update_layout(
        title="Last 90 Days Expenses with Trendline",
        xaxis_title="Date",
        yaxis_title="Total Expenses",
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Add trend analysis
    if len(daily) >= 14:  # Need at least 2 weeks for meaningful trend
        recent_trend = daily['trend'].tail(7).mean()
        earlier_trend = daily['trend'].head(7).mean()
        trend_change = ((recent_trend - earlier_trend) / earlier_trend * 100) if earlier_trend > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recent Average (7d)", f"${recent_trend:.2f}")
        with col2:
            st.metric("Earlier Average (7d)", f"${earlier_trend:.2f}")
        with col3:
            st.metric("Trend Change", f"{trend_change:+.1f}%")


def render_per_category_forecasts_section(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """
    Render the per-category forecasts section with budget tracking.
    
    Args:
        df: Main expense DataFrame
        forecast_df: Forecast results DataFrame
    """
    create_section_header(
        "🎨 Per-Category Forecasts",
        "Individual forecasts for each category with budget tracking"
    )
    
    if forecast_df.empty:
        create_info_box("No Forecast Data", "No forecast data available for category analysis.", "info")
        return
    
    # Individual category forecasts
    st.subheader("Individual Category Forecasts")
    for cat in forecast_df['category'].unique():
        df_plot = forecast_df[forecast_df['category'] == cat]
        df_cat_hist = df[(df['category'] == cat) & (df['date'] >= (df['date'].max() - pd.Timedelta(days=30)))]
        
        if not df_cat_hist.empty:
            df_cat_hist = df_cat_hist.groupby('date')['amount'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_cat_hist['date'], 
                y=df_cat_hist['amount'], 
                name='Actual (last 30d)', 
                marker_color='gray', 
                opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=df_plot['date'], 
                y=df_plot['forecast'], 
                mode='lines+markers', 
                name='Forecast', 
                line=dict(color='royalblue')
            ))
            fig.update_layout(
                title=f"Forecast for {cat} (with last 30d history)", 
                xaxis_title="Date", 
                yaxis_title="Amount", 
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Moving average baseline
    st.subheader("Moving Average Baseline (last 30 days)")
    if not df.empty:
        hist_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
        if not hist_30.empty:
            ma_baseline = hist_30.groupby('date')['amount'].sum().rolling(window=7, min_periods=1).mean()
            st.line_chart(ma_baseline, use_container_width=True)
    
    # Monthly Budget Tracking
    st.subheader("Monthly Budget Tracking")
    
    # Budget configuration
    current_date = pd.Timestamp.now()
    current_month = current_date.replace(day=1)
    days_in_month = (current_month + pd.offsets.MonthEnd(1)).day
    current_day = min(current_date.day, days_in_month)
    
    DEFAULT_BUDGET = 4000
    with st.expander("Budget Settings"):
        monthly_budget = st.number_input(
            "Monthly Budget ($)",
            min_value=0.0,
            value=float(DEFAULT_BUDGET),
            step=100.0,
            format="%.2f",
            help="Set your monthly budget amount",
            key="monthly_budget_input"
        )
    
    # Calculate and display budget tracking
    daily_budget = monthly_budget / days_in_month
    
    # Get current month data
    current_month_mask = (
        (df['date'].dt.year == current_date.year) & 
        (df['date'].dt.month == current_date.month) &
        (df['date'].dt.date <= current_date.date())
    )
    current_month_data = df[current_month_mask]
    
    if not current_month_data.empty:
        total_spent = current_month_data['amount'].sum()
        budgeted_so_far = current_day * daily_budget
        remaining_budget = monthly_budget - total_spent
        variance = total_spent - budgeted_so_far
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spent", f"${total_spent:.2f}")
        with col2:
            st.metric("Budget to Date", f"${budgeted_so_far:.2f}")
        with col3:
            st.metric("Remaining Budget", f"${remaining_budget:.2f}")
        with col4:
            variance_color = "red" if variance > 0 else "green"
            st.metric("Variance", f"${variance:+.2f}")
    
    else:
        create_info_box("No Current Month Data", "No expense data available for the current month.", "info")


def render_stacked_chart_section(df: pd.DataFrame, forecast_df: pd.DataFrame, forecast_horizon: int = 7) -> None:
    """
    Render the stacked chart section showing history + forecast.
    
    Args:
        df: Main expense DataFrame
        forecast_df: Forecast results DataFrame
        forecast_horizon: Number of forecast days
    """
    create_section_header(
        "📉 Stacked Chart Analysis",
        f"Combined view: 30 days history + {forecast_horizon} days forecast"
    )
    
    if df.empty or forecast_df.empty:
        create_info_box("No Data", "Need both historical and forecast data for stacked chart.", "info")
        return
    
    st.subheader(f"Stacked Column Chart: 30 Days History + {forecast_horizon} Days Forecast")
    
    # Prepare historical data (last 30 days)
    last_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
    if last_30.empty:
        create_info_box("Insufficient History", "Need at least some data in the last 30 days.", "info")
        return
    
    hist_pivot = last_30.groupby(['date', 'category'])['amount'].sum().reset_index()
    hist_pivot = hist_pivot.pivot(index='date', columns='category', values='amount').fillna(0)
    
    # Combine history and forecast
    forecast_pivot = forecast_df.pivot(index='date', columns='category', values='forecast').fillna(0)
    combined = pd.concat([hist_pivot, forecast_pivot], axis=0)
    
    # Create stacked chart
    fig_stacked = go.Figure()
    for cat in combined.columns:
        fig_stacked.add_trace(go.Bar(
            x=combined.index,
            y=combined[cat],
            name=cat
        ))
    
    fig_stacked.update_layout(
        barmode='stack',
        title=f"Expenses: Last 30 Days (Actual) + Next {forecast_horizon} Days (Forecast)",
        xaxis_title="Date",
        yaxis_title="Expenses",
        legend_title="Category",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig_stacked, use_container_width=True)


def render_forecast_charts_section(backward_results: pd.DataFrame,
                                 spike_threshold: float = 30) -> None:
    """
    Render the forecast visualization charts section.
    
    Args:
        backward_results: Backward forecast results
        spike_threshold: Threshold for spike detection
    """
    create_section_header(
        "📊 Forecast Visualization", 
        "Interactive charts showing actual vs forecast performance"
    )
    
    if backward_results.empty:
        create_info_box("No Data", "No forecast data available for visualization.", "info")
        return
    
    # Main forecast chart
    fig = create_forecast_chart(
        actual_data=backward_results['actual'],
        forecast_data=backward_results['forecast'],
        dates=backward_results['date'],
        title="Actual vs Forecast Over Time"
    )
    
    # Add spike annotations if spike threshold is met
    spike_mask = backward_results['pct_error'].abs() > spike_threshold
    if spike_mask.any():
        spike_dates = backward_results.loc[spike_mask, 'date']
        spike_actuals = backward_results.loc[spike_mask, 'actual']
        
        fig.add_trace(go.Scatter(
            x=spike_dates,
            y=spike_actuals,
            mode='markers',
            name='Spike',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> $%{y:,.2f}<extra></extra>'
        ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error chart
    error_fig = create_error_chart(
        dates=backward_results['date'],
        errors=backward_results['pct_error'],
        title="Forecast Error Over Time",
        error_type="percentage"
    )
    st.plotly_chart(error_fig, use_container_width=True)