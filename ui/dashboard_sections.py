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
from forecast_utils import calculate_weekly_metrics


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
        st.write("**Overall Quality Assessment with Subjective Analysis**")
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
        
        # Comprehensive metrics table with subjective explanations
        st.subheader("📋 Общее резюме (General Summary)")
        
        # Get TAE (Time Absolute Error) from metrics
        if 'basic_metrics' in comprehensive_metrics and 'tae' in comprehensive_metrics['basic_metrics']:
            tae = comprehensive_metrics['basic_metrics']['tae']
        else:
            tae = 0
            
        # Create comprehensive metrics table with Russian subjective explanations
        def get_subjective_comment(metric_name, value):
            """Get subjective commentary in Russian for each metric"""
            comments = {
                'Direction Correct': {
                    'high': "Хорошо",
                    'medium': "Слабовато, можно улучшить", 
                    'low': "Очень низко"
                },
                'Hit Rate ±10%': {
                    'high': "Надёжный показатель",
                    'medium': "Плохо",
                    'low': "Очень низко"
                },
                'Hit Rate ±20%': {
                    'high': "Надёжный показатель", 
                    'medium': "Плохо",
                    'low': "Плохо"
                },
                'Over / Under Bias': {
                    'underforecast': "Не критично, но требует внимания",
                    'overforecast': "Не критично, но требует внимания",
                    'balanced': "Хорошо"
                },
                'Theil\'s U': {
                    'excellent': "Хорошо",
                    'good': "Хорошо", 
                    'fair': "Средне",
                    'poor': "Плохо"
                },
                'MASE': {
                    'excellent': "Надёжный показатель",
                    'good': "Надёжный показатель",
                    'fair': "Средне", 
                    'poor': "Ошибка довольно большая"
                },
                'CV-RMSE': {
                    'low': "Хорошо",
                    'medium': "Средне",
                    'high': "Ошибка довольно большая"
                },
                'TAE': {
                    'low': "Отлично",
                    'medium': "Хорошо",
                    'high': "Требует улучшения"
                }
            }
            
            if metric_name == 'Direction Correct':
                if value >= 60:
                    return comments[metric_name]['high']
                elif value >= 40:
                    return comments[metric_name]['medium']
                else:
                    return comments[metric_name]['low']
            elif metric_name in ['Hit Rate ±10%', 'Hit Rate ±20%']:
                if value >= 70:
                    return comments[metric_name]['high']
                elif value >= 50:
                    return comments[metric_name]['medium'] 
                else:
                    return comments[metric_name]['low']
            elif metric_name == 'Over / Under Bias':
                if 'underforecast' in str(value).lower():
                    return comments[metric_name]['underforecast']
                elif 'overforecast' in str(value).lower():
                    return comments[metric_name]['overforecast']
                else:
                    return comments[metric_name]['balanced']
            elif metric_name == 'Theil\'s U':
                if value <= 0.8:
                    return comments[metric_name]['excellent']
                elif value <= 1.0:
                    return comments[metric_name]['good']
                elif value <= 1.5:
                    return comments[metric_name]['fair']
                else:
                    return comments[metric_name]['poor']
            elif metric_name == 'MASE':
                if value <= 0.8:
                    return comments[metric_name]['excellent']
                elif value <= 1.0:
                    return comments[metric_name]['good']
                elif value <= 1.5:
                    return comments[metric_name]['fair']
                else:
                    return comments[metric_name]['poor']
            elif metric_name == 'CV-RMSE':
                if value <= 0.3:
                    return comments[metric_name]['low']
                elif value <= 0.7:
                    return comments[metric_name]['medium']
                else:
                    return comments[metric_name]['high']
            elif metric_name == 'TAE':
                if value <= 50:
                    return comments[metric_name]['low']
                elif value <= 100:
                    return comments[metric_name]['medium']
                else:
                    return comments[metric_name]['high']
            
            return "Нормально"
        
        # Build summary table
        summary_metrics = []
        
        # Direction Correct
        direction_pct = comprehensive_metrics['directional_metrics']['directional_accuracy']
        summary_metrics.append({
            'Метрика': 'Direction Correct',
            'Оценка': f"🟡 {direction_pct:.1f}%",
            'Комментарий': get_subjective_comment('Direction Correct', direction_pct)
        })
        
        # Hit Rates
        hit_10 = comprehensive_metrics['directional_metrics']['hit_rate_10pct']
        summary_metrics.append({
            'Метрика': 'Hit Rate ±10%', 
            'Оценка': f"🔴 {hit_10:.1f}%",
            'Комментарий': get_subjective_comment('Hit Rate ±10%', hit_10)
        })
        
        hit_20 = comprehensive_metrics['directional_metrics']['hit_rate_20pct']
        summary_metrics.append({
            'Метрика': 'Hit Rate ±20%',
            'Оценка': f"🔴 {hit_20:.1f}%", 
            'Комментарий': get_subjective_comment('Hit Rate ±20%', hit_20)
        })
        
        # Over/Under Bias
        over_pct = comprehensive_metrics['distribution_metrics']['overforecast_percentage']
        under_pct = comprehensive_metrics['distribution_metrics']['underforecast_percentage']
        if over_pct > under_pct:
            bias_text = "Смещение в overforecast"
        else:
            bias_text = "Смещение в underforecast" 
        summary_metrics.append({
            'Метрика': 'Over / Under Bias',
            'Оценка': f"🟡 {bias_text}",
            'Комментарий': get_subjective_comment('Over / Under Bias', bias_text)
        })
        
        # Theil's U
        theil_u = comprehensive_metrics['distribution_metrics']['theil_u']
        theil_display = f"{theil_u:.3f}" if not np.isinf(theil_u) else "N/A"
        color_icon = "🟢" if theil_u <= 0.8 else "🟡" if theil_u <= 1.0 else "🔴"
        summary_metrics.append({
            'Метрика': "Theil's U",
            'Оценка': f"{color_icon} {theil_display}",
            'Комментарий': get_subjective_comment("Theil's U", theil_u)
        })
        
        # MASE
        mase = comprehensive_metrics['scaled_metrics']['mase']
        mase_display = f"{mase:.3f}" if not np.isinf(mase) else "N/A"
        mase_icon = "🟢" if mase <= 0.8 else "🟡" if mase <= 1.0 else "🔴"
        summary_metrics.append({
            'Метрика': 'MASE',
            'Оценка': f"{mase_icon} {mase_display}",
            'Комментарий': get_subjective_comment('MASE', mase)
        })
        
        # CV-RMSE  
        cv_rmse = comprehensive_metrics['scaled_metrics']['cv_rmse']
        cv_rmse_display = f"{cv_rmse:.3f}" if not np.isinf(cv_rmse) else "N/A"
        cv_icon = "🟢" if cv_rmse <= 0.3 else "🟡" if cv_rmse <= 0.7 else "🔴"
        summary_metrics.append({
            'Метрика': 'CV-RMSE',
            'Оценка': f"{cv_icon} {cv_rmse_display}",
            'Комментарий': get_subjective_comment('CV-RMSE', cv_rmse)
        })
        
        # TAE (Time Absolute Error)
        tae_icon = "🟢" if tae <= 50 else "🟡" if tae <= 100 else "🔴"
        summary_metrics.append({
            'Метрика': 'TAE',
            'Оценка': f"{tae_icon} {tae:.2f}",
            'Комментарий': get_subjective_comment('TAE', tae)
        })
        
        # Display the summary table
        summary_df = pd.DataFrame(summary_metrics)
        display_dataframe_with_controls(summary_df, show_download=True, download_key="comprehensive_summary")
    
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


def calculate_backward_accuracy(df, forecast_horizon=7, min_data_points=30, timeframe='daily', activity_window=60):
    """
    Calculate forecast accuracy by looking back at past forecasts and comparing with actuals.
    
    Args:
        df: DataFrame with transaction data with columns: date, amount, category
        forecast_horizon: Number of days ahead to forecast
        activity_window: Number of recent days used as training window for generating each historical forecast
        min_data_points: Minimum number of data points required for daily analysis
        timeframe: 'daily' or 'weekly' - whether to return daily or weekly aggregated results
        
    Returns:
        DataFrame with date, actual, forecast, and error metrics.
    """
    from croston import croston
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error
    
    try:
        # Validate inputs
        if timeframe not in ['daily', 'weekly', 'monthly']:
            raise ValueError("timeframe must be either 'daily' or 'weekly' or 'monthly'")
            
        if df.empty:
            return pd.DataFrame()
            
        # Ensure we have a copy of the dataframe
        df = df.copy()
        
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Get unique dates with actual data
        actual_dates = df['date'].sort_values().unique()
        
        # Adjust minimum data requirements based on timeframe
        min_days_required = forecast_horizon + 1  # At least one full forecast cycle
        if timeframe == 'weekly':
            # For weekly, we can be more lenient - need at least 2 weeks of data
            min_days_required = max(forecast_horizon + 1, 14)  # At least 2 weeks
            # Adjust min_data_points to be at least min_days_required
            min_data_points = min(min_data_points, min_days_required)
        
        if len(actual_dates) < min_days_required:
            st.warning(f"Not enough data points for {timeframe} analysis. Need at least {min_days_required} days of data. Have {len(actual_dates)} days.")
            return pd.DataFrame()
            
        # For weekly view, we can proceed with fewer data points than the default min_data_points
        if timeframe == 'weekly' and len(actual_dates) < min_data_points + forecast_horizon:
            st.warning(f"Limited data available for weekly analysis. Using all available {len(actual_dates)} days.")
        
        results = []
        
        # Define forecasting methods
        def mean_forecast(ts, fh):
            return ts.mean() if len(ts) > 0 else 0
        
        def median_forecast(ts, fh):
            return ts.median() if len(ts) > 0 else 0
        
        def zero_forecast(ts, fh):
            return 0
        
        def periodic_spike_forecast(ts, fh):
            if len(ts) < 3:
                return ts.mean() if len(ts) > 0 else 0
            # Calculate average of non-zero values
            nonzero = ts[ts > 0]
            if len(nonzero) == 0:
                return 0
            return nonzero.mean()
        
        def croston_forecast(ts, fh):
            try:
                if len(ts) < 2:
                    return ts.mean() if len(ts) > 0 else 0
                result = croston(ts.values, h=fh, alpha=0.1, beta=0.1)
                return result['forecast'][0] if len(result['forecast']) > 0 else 0
            except:
                return ts.mean() if len(ts) > 0 else 0
        
        def prophet_forecast(ts, fh):
            try:
                if len(ts) < 10:
                    return ts.mean() if len(ts) > 0 else 0
                
                # Prepare data for Prophet
                prophet_df = pd.DataFrame({
                    'ds': ts.index,
                    'y': ts.values
                })
                
                # Create and fit model
                model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=True)
                model.fit(prophet_df)
                
                # Make forecast
                future = model.make_future_dataframe(periods=fh)
                forecast = model.predict(future)
                
                # Return the forecast for the horizon period
                return max(0, forecast['yhat'].iloc[-1])
            except:
                return ts.mean() if len(ts) > 0 else 0
        
        methods = {
            'mean': mean_forecast,
            'median': median_forecast,
            'zero': zero_forecast,
            'croston': croston_forecast,
            'prophet': prophet_forecast,
            'periodic_spike': periodic_spike_forecast
        }
        
        # Generate forecasts for each date
        for i in range(activity_window + forecast_horizon, len(actual_dates)):
            current_date = actual_dates[i]
            
            # Get actual values for current date (sum across all categories)
            actual_data = df[df['date'] == current_date]
            if len(actual_data) == 0:
                continue
                
            actual = actual_data['amount'].sum()
            
            # Initialize total_forecast at the start of each date's processing
            total_forecast = 0
            forecast_by_category = {}
            
            # Get unique categories in the data
            all_categories = df['category'].unique()
            
            # Generate forecast for each category independently
            for cat in all_categories:
                # Get training data for this category up to current date
                training_end_date = current_date - pd.Timedelta(days=forecast_horizon)
                cat_data = df[(df['category'] == cat) & (df['date'] <= training_end_date)].copy()
                
                if len(cat_data) < 1:
                    continue
                    
                # Create time series with proper datetime index
                ts = cat_data.groupby('date')['amount'].sum().reindex(
                    pd.date_range(start=training_end_date - pd.Timedelta(days=activity_window-1), 
                                end=training_end_date, freq='D'), fill_value=0
                )
                
                # Select best method for this category
                nonzero_days = (ts > 0).sum()
                spike_cats = ['school', 'rent + communal', 'car rent']
                if any(key in cat.lower() for key in spike_cats) or nonzero_days <= 2:
                    method_name = 'periodic_spike'
                else:
                    # Simple method selection - use mean for most cases
                    method_name = 'mean'
                
                # Generate forecast
                forecast = methods[method_name](ts, forecast_horizon)
                forecast_by_category[cat] = forecast
                total_forecast += forecast
            
            # Store results
            error = actual - total_forecast
            abs_error = abs(error)
            pct_error = (error / actual * 100) if actual != 0 else 0
            within_10pct = abs(pct_error) <= 10
            within_20pct = abs(pct_error) <= 20
            
            results.append({
                'date': current_date,
                'actual': actual,
                'forecast': total_forecast,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error,
                'within_10pct': within_10pct,
                'within_20pct': within_20pct
            })
        
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            return pd.DataFrame()
        
        # Convert to requested timeframe
        if timeframe == 'weekly':
            # Group by week and aggregate
            results_df['week'] = results_df['date'].dt.to_period('W')
            weekly_results = results_df.groupby('week').agg({
                'actual': 'sum',
                'forecast': 'sum',
                'error': 'sum',
                'abs_error': 'sum'
            }).reset_index()
            
            # Recalculate percentage metrics for weekly data
            weekly_results['pct_error'] = (weekly_results['error'] / weekly_results['actual'] * 100).fillna(0)
            weekly_results['within_10pct'] = weekly_results['pct_error'].abs() <= 10
            weekly_results['within_20pct'] = weekly_results['pct_error'].abs() <= 20
            weekly_results['date'] = weekly_results['week'].dt.end_time
            
            return weekly_results.drop('week', axis=1)
        
        return results_df
        
    except Exception as e:
        st.error(f"Error in backward forecast calculation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


def render_backward_forecast_section(df: pd.DataFrame, 
                                   controls: Dict[str, Any]) -> None:
    """
    Render the comprehensive backward forecast performance analysis section.
    
    Args:
        df: Main expense DataFrame
        controls: Dictionary with control values from sidebar
    """
    create_section_header(
        "📈 Backward Forecast Performance", 
        "Analyze forecast accuracy over time by comparing past forecasts with actuals"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for backward forecast analysis.", "info")
        return
    
    # Extract activity window from controls
    activity_window = controls.get('activity_window', 70)
    
    # Create columns for the controls
    col1, col2, col3 = st.columns(3)
    with col1:
        max_horizon = min(30, len(df['date'].unique()) - 15)  # Ensure enough data
        horizon = st.slider("Forecast Horizon (days)", 1, max(1, max_horizon), 7, 1,
                           help="Number of days ahead to evaluate forecast accuracy")
    
    # Add timeframe selector (daily/weekly)
    # View is fixed to Daily to simplify Backward Forecast Performance
    timeframe = 'daily'  # only daily view supported
    
    # Add date range selection
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    default_end = max_date
    default_start = max(min_date, (pd.to_datetime(default_end) - pd.DateOffset(months=3)).date())
    
    with st.expander("Date Range Settings", expanded=True):
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="Start date for the analysis period"
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="End date for the analysis period"
            )
    
        # Ensure start date is before end date
        if start_date > end_date:
            st.error("Error: Start date must be before end date")
            st.stop()
    
    # Add category selection
    all_categories = sorted(df['category'].unique())
    selected_categories = st.multiselect(
        'Select Categories (Leave empty for all)',
        options=all_categories,
        default=[],
        help="Select specific categories to analyze or leave empty to include all categories"
    )
    
    # Extend data window by forecast horizon
    window_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=horizon)).date()
    analysis_df = df[
        (df['date'].dt.date >= window_start_date) &
        (df['date'].dt.date <= end_date)
    ].copy()
    
    if selected_categories:
        analysis_df = analysis_df[analysis_df['category'].isin(selected_categories)]
        st.info(f"Showing results for categories: {', '.join(selected_categories)} from {start_date} to {end_date}")
    else:
        st.info(f"Showing results for all categories from {start_date} to {end_date}")
    
    # Calculate backward forecast performance
    with st.spinner(f"Calculating {timeframe} forecast accuracy..."):
        backward_results = calculate_backward_accuracy(
            analysis_df,
            forecast_horizon=horizon,
            activity_window=activity_window,
            timeframe=timeframe
        )
        # Keep only rows that fall inside the user-selected report window
        if not backward_results.empty and 'date' in backward_results.columns:
            backward_results = backward_results[
                (backward_results['date'] >= pd.to_datetime(start_date)) &
                (backward_results['date'] <= pd.to_datetime(end_date))
            ]
        else:
            # If DataFrame is empty or missing 'date', set to empty DataFrame with expected columns
            backward_results = pd.DataFrame(columns=[
                'date', 'actual', 'forecast', 'error', 'abs_error', 'pct_error', 'within_10pct', 'within_20pct'])
    
    if not backward_results.empty:
        # Calculate summary metrics based on the selected timeframe
        # --- Calculate Total Actual from original df, not from backward_results ---
        actual_period_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        if selected_categories:
            actual_period_df = actual_period_df[actual_period_df['category'].isin(selected_categories)]
        if timeframe == 'monthly':
            # Group by month and sum
            actual_period_df['year_month'] = actual_period_df['date'].dt.to_period('M')
            total_actual = actual_period_df.groupby('year_month')['amount'].sum().sum()
        elif timeframe == 'weekly':
            # Group by week and sum
            actual_period_df['year_week'] = actual_period_df['date'].dt.strftime('%Y-W%U')
            total_actual = actual_period_df.groupby('year_week')['amount'].sum().sum()
        else:
            total_actual = actual_period_df['amount'].sum()

        if timeframe == 'daily':
            # For daily view, calculate metrics from daily values
            total_forecast = backward_results['forecast'].sum()
            total_error = total_forecast - total_actual
            total_abs_error = abs(total_error)
            # Weighted Absolute Percentage Error (WAPE)
            wape = (backward_results['abs_error'].sum() / total_actual * 100) if total_actual > 0 else 0
            # Mean Absolute Percentage Error (daily average of |pct_error|)
            avg_mape = backward_results['pct_error'].abs().mean()
            # Calculate accuracy metrics
            accuracy_10pct = backward_results['within_10pct'].mean() * 100 if 'within_10pct' in backward_results.columns else 0
            accuracy_20pct = backward_results['within_20pct'].mean() * 100 if 'within_20pct' in backward_results.columns else 0
            # Calculate average daily metrics
            avg_mae = backward_results['abs_error'].mean()  # daily average MAE
        else:  # weekly or monthly
            metrics = calculate_weekly_metrics(backward_results)
            total_forecast = metrics['total_forecast']
            total_error = total_forecast - total_actual
            total_abs_error = abs(total_error)
            avg_mape = metrics['avg_mape']
            wape = metrics.get('wape', 0)
            accuracy_10pct = metrics['accuracy_10pct']
            accuracy_20pct = metrics['accuracy_20pct']
            avg_mae = metrics['avg_weekly_actual'] if timeframe == 'weekly' else metrics['avg_weekly_forecast']
        
        # Show category information if filtered
        if selected_categories:
            st.subheader(f'Analysis for {len(selected_categories)} Selected Categories')
        
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Actual", f"${total_actual:,.2f}")
        with col2:
            st.metric("Total Forecast", f"${total_forecast:,.2f}", delta=f"{total_error:+,.2f}")
        with col3:
            st.metric("Mean Absolute Error (MAE)", f"${avg_mae:,.2f}")
        with col4:
            st.metric("Mean Absolute % Error (MAPE)", f"{avg_mape:.1f}%")
        with col5:
            st.metric("Weighted APE (WAPE)", f"{wape:.1f}%")
        
        # Display accuracy metrics in a separate row
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy (±10%)", f"{accuracy_10pct:.1f}%")
        with col2:
            st.metric("Accuracy (±20%)", f"{accuracy_20pct:.1f}%")
        
        # Comprehensive forecast performance metrics using modular component
        render_forecast_performance_section(df, backward_results, controls)
        
        # Forecast charts using modular component
        render_forecast_charts_section(backward_results, spike_threshold=controls['spike_threshold'])
        
        # --- 7-Day Moving Average Comparison ---
        st.header("7-Day Moving Average Comparison")
        st.caption("Compare smoothed actual and forecasted values to identify trends")
        
        if not backward_results.empty and len(backward_results) >= 7:
            # Calculate 7-day moving averages
            ma_results = backward_results.copy()
            ma_results['actual_ma7'] = ma_results['actual'].rolling(window=7, min_periods=1).mean()
            ma_results['forecast_ma7'] = ma_results['forecast'].rolling(window=7, min_periods=1).mean()
            
            # Calculate MAE and MAPE for moving averages
            ma_results['ma7_abs_error'] = (ma_results['actual_ma7'] - ma_results['forecast_ma7']).abs()
            ma_results['ma7_pct_error'] = (ma_results['ma7_abs_error'] / ma_results['actual_ma7'] * 100).replace([np.inf, -np.inf], np.nan)
            
            # Calculate summary metrics
            avg_ma7_mae = ma_results['ma7_abs_error'].mean()
            avg_ma7_mape = ma_results['ma7_pct_error'].mean()
            
            # Create the plot
            fig_ma = go.Figure()
            
            # Add actual MA7 line
            fig_ma.add_trace(go.Scatter(
                x=ma_results['date'],
                y=ma_results['actual_ma7'],
                name='Actual (7-day MA)',
                line=dict(color='#1f77b4', width=3),
                mode='lines+markers',
                marker=dict(size=6),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                            '<b>Actual MA7</b>: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add forecast MA7 line
            fig_ma.add_trace(go.Scatter(
                x=ma_results['date'],
                y=ma_results['forecast_ma7'],
                name=f'Forecast (7-day MA, {horizon}-day horizon)',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                            f'<b>Forecast MA7 ({horizon}-day horizon)</b>: $%{{y:,.2f}}<extra></extra>'
            ))
            
            # Add error metrics as annotations
            ma_annotations = [
                dict(
                    x=0.02,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"MAE (MA7): ${avg_ma7_mae:,.2f}<br>MAPE (MA7): {avg_ma7_mape:.1f}%",
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=4
                )
            ]
            
            fig_ma.update_layout(
                title=f'7-Day Moving Average: Actual vs {horizon}-Day Forecast',
                xaxis_title='Date',
                yaxis_title='Amount (7-day MA)',
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                annotations=ma_annotations
            )
            
            # Add range slider
            fig_ma.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            st.plotly_chart(fig_ma, use_container_width=True)
    else:
        create_info_box("No Results", "No backward forecast results available for the selected criteria.", "warning")


def render_budget_tracking_section(df: pd.DataFrame) -> None:
    """
    Render the budget tracking section with monthly budget analysis.
    
    Args:
        df: Main expense DataFrame
    """
    create_section_header(
        "💰 Budget Tracking", 
        "Track your monthly budget vs actual spending with daily breakdown"
    )
    
    if df.empty:
        create_info_box("No Data", "No data available for budget tracking.", "info")
        return
    
    # Budget configuration
    current_date = pd.Timestamp.now()
    current_month = current_date.replace(day=1)
    days_in_month = (current_month + pd.offsets.MonthEnd(1)).day
    current_day = min(current_date.day, days_in_month)
    
    # Add budget input with default value
    DEFAULT_BUDGET = 4000
    with st.expander("Budget Settings"):
        monthly_budget = st.number_input(
            "Monthly Budget ($)",
            min_value=0.0,
            value=float(DEFAULT_BUDGET),
            step=100.0,
            format="%.2f",
            help="Set your monthly budget amount"
        )
    
    # Calculate daily budget
    daily_budget = monthly_budget / days_in_month
    
    # Create date range for the current month (up to current date)
    date_range = pd.date_range(
        start=current_month, 
        end=current_date,
        freq='D'
    )
    
    # Initialize budget dataframe with all required columns
    budget_data = []
    for i, date in enumerate(date_range, 1):
        budget_data.append({
            'date': date,
            'day_of_month': i,
            'cumulative_budget': round(i * daily_budget, 2),
            'daily_amount': 0.0,
            'cumulative_actual': 0.0,
            'variance': 0.0
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # Get current month data
    current_month_mask = (
        (df['date'].dt.year == current_date.year) & 
        (df['date'].dt.month == current_date.month) &
        (df['date'].dt.date <= current_date.date())
    )
    current_month_data = df[current_month_mask].copy()
    
    # Process daily totals if we have data
    if not current_month_data.empty:
        # Convert dates to match for comparison
        current_month_data['date_only'] = current_month_data['date'].dt.date
        
        # Group by date and sum amounts
        daily_totals = current_month_data.groupby('date_only')['amount'].sum().reset_index()
        
        # Update budget_df with actual amounts
        for _, row in daily_totals.iterrows():
            date = row['date_only']
            amount = row['amount']
            mask = budget_df['date'].dt.date == date
            if mask.any():
                budget_df.loc[mask, 'daily_amount'] = amount
        
        # Calculate running totals and variance
        budget_df['cumulative_actual'] = budget_df['daily_amount'].cumsum()
        budget_df['variance'] = budget_df['cumulative_actual'] - budget_df['cumulative_budget']
        
        # Current status
        current_actual = budget_df['cumulative_actual'].iloc[-1] if not budget_df.empty else 0
        current_budget = budget_df['cumulative_budget'].iloc[-1] if not budget_df.empty else 0
        current_variance = current_actual - current_budget
        current_day = budget_df['day_of_month'].iloc[-1] if not budget_df.empty else 0
        
        # Debug info
        with st.expander("Debug Info - Budget Data"):
            st.write(f"Current date: {current_date}")
            st.write(f"Current month data range: {current_month} to {current_date}")
            st.write(f"Found {len(current_month_data)} transactions in current month")
            if not current_month_data.empty:
                st.write("Sample transactions:", current_month_data[['date', 'category', 'amount']].head())
            st.write("Budget DataFrame:", budget_df)
        
        # Create the plot
        fig_budget = go.Figure()
        
        # Add budget line (full month)
        fig_budget.add_trace(go.Scatter(
            x=budget_df['date'].dt.day,
            y=budget_df['cumulative_budget'],
            name='Budget',
            line=dict(color='#00cc96', width=3, dash='dash'),
            hovertemplate='Day %{x}<br>Budget: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add actuals line (up to current day)
        fig_budget.add_trace(go.Scatter(
            x=budget_df[budget_df['date'] <= current_date]['date'].dt.day,
            y=budget_df[budget_df['date'] <= current_date]['cumulative_actual'],
            name='Actual',
            line=dict(color='#636efa', width=3),
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate='Day %{x}<br>Actual: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add today's line
        fig_budget.add_vline(
            x=current_day, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Today (Day {current_day})",
            annotation_position="top right"
        )
        
        # Add budget status annotation
        status_color = "red" if current_variance > 0 else "green"
        status_text = f"Over budget by ${abs(current_variance):,.2f}" if current_variance > 0 \
            else f"Under budget by ${abs(current_variance):,.2f}"
        
        # Update layout
        fig_budget.update_layout(
            title=f"Monthly Budget: ${monthly_budget:,.0f} (${daily_budget:,.2f}/day)",
            xaxis_title="Day of Month",
            yaxis_title="Cumulative Amount ($)",
            hovermode="x unified",
            height=500,
            showlegend=True,
            annotations=[
                dict(
                    x=0.02,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Current Status (Day {current_day}):</b><br>"
                         f"Spent: ${current_actual:,.2f}<br>"
                         f"Budget: ${current_budget:,.2f}<br>"
                         f"<span style='color:{status_color}'>{status_text}</span>",
                    showarrow=False,
                    align="left",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.9
                )
            ]
        )
        
        st.plotly_chart(fig_budget, use_container_width=True)
        
        # Add daily spending table for the current month
        st.write("#### Daily Spending This Month")
        display_df = budget_df[['date', 'day_of_month', 'daily_amount', 'cumulative_actual', 'cumulative_budget', 'variance']].copy()
        display_df = display_df.rename(columns={
            'date': 'Date',
            'day_of_month': 'Day',
            'daily_amount': 'Daily Amount',
            'cumulative_actual': 'Cumulative Actual',
            'cumulative_budget': 'Cumulative Budget',
            'variance': 'Variance'
        })
        
        # Format the display
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Daily Amount'] = display_df['Daily Amount'].apply(lambda x: f"${x:,.2f}")
        display_df['Cumulative Actual'] = display_df['Cumulative Actual'].apply(lambda x: f"${x:,.2f}")
        display_df['Cumulative Budget'] = display_df['Cumulative Budget'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        create_info_box("No Data", "No data available for the current month.", "info")