"""
Chart generation functions for the expense forecasting dashboard.
Contains Plotly chart creation and styling utilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta


def create_base_layout(title: str, 
                      xaxis_title: str = "Date", 
                      yaxis_title: str = "Amount",
                      height: int = 500) -> Dict[str, Any]:
    """
    Create base layout configuration for charts.
    
    Args:
        title: Chart title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        height: Chart height
        
    Returns:
        Layout dictionary
    """
    return {
        'title': title,
        'xaxis_title': xaxis_title,
        'yaxis_title': yaxis_title,
        'height': height,
        'hovermode': 'x unified',
        'legend': dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        'xaxis': dict(
            tickformat='%b %d, %Y'
        )
    }


def create_forecast_chart(actual_data: pd.Series, 
                         forecast_data: pd.Series,
                         dates: pd.Series,
                         title: str = "Actual vs Forecast",
                         show_confidence: bool = False,
                         confidence_upper: Optional[pd.Series] = None,
                         confidence_lower: Optional[pd.Series] = None) -> go.Figure:
    """
    Create actual vs forecast comparison chart.
    
    Args:
        actual_data: Actual values
        forecast_data: Forecast values
        dates: Date values
        title: Chart title
        show_confidence: Whether to show confidence intervals
        confidence_upper: Upper confidence bound
        confidence_lower: Lower confidence bound
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_data,
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        mode='lines+markers',
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Add forecast values
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast_data,
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Add confidence intervals if provided
    if show_confidence and confidence_upper is not None and confidence_lower is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([dates, dates[::-1]]),
            y=pd.concat([confidence_upper, confidence_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))
    
    fig.update_layout(**create_base_layout(title))
    return fig


def create_error_chart(dates: pd.Series, 
                      errors: pd.Series,
                      title: str = "Forecast Error Over Time",
                      error_type: str = "absolute") -> go.Figure:
    """
    Create forecast error chart.
    
    Args:
        dates: Date values
        errors: Error values
        title: Chart title
        error_type: Type of error ("absolute", "percentage")
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Determine color based on error values
    error_colors = ['red' if e > 0 else 'green' for e in errors]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=errors,
        name='Error',
        line=dict(color='#d62728', width=2),
        mode='lines+markers',
        marker=dict(size=6, color=error_colors),
        hovertemplate=f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Error:</b> %{{y:.2f}}{"%" if error_type == "percentage" else ""}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(
        y=0, 
        line_dash="dash",
        annotation_text="Perfect Forecast",
        annotation_position="bottom right",
        line_width=1
    )
    
    yaxis_title = "Error (%)" if error_type == "percentage" else "Error ($)"
    fig.update_layout(**create_base_layout(title, yaxis_title=yaxis_title))
    return fig


def create_category_comparison_chart(category_data: Dict[str, pd.Series],
                                   chart_type: str = "bar",
                                   title: str = "Category Comparison") -> go.Figure:
    """
    Create category comparison chart.
    
    Args:
        category_data: Dict of category_name -> values
        chart_type: Chart type ("bar", "pie", "line")
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if chart_type == "pie":
        # Create pie chart
        labels = list(category_data.keys())
        values = [series.sum() if isinstance(series, pd.Series) else series for series in category_data.values()]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
    elif chart_type == "line":
        # Create line chart
        fig = go.Figure()
        
        for category, data in category_data.items():
            if isinstance(data, pd.Series):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=category,
                    mode='lines+markers',
                    hovertemplate=f'<b>{category}</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: $%{{y:,.2f}}<extra></extra>'
                ))
    
    else:  # bar chart
        categories = list(category_data.keys())
        values = [series.sum() if isinstance(series, pd.Series) else series for series in category_data.values()]
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.2f}<extra></extra>'
        )])
    
    fig.update_layout(**create_base_layout(title))
    return fig


def create_seasonality_heatmap(data: pd.DataFrame,
                              value_col: str = "amount",
                              title: str = "Seasonality Heatmap") -> go.Figure:
    """
    Create seasonality heatmap showing patterns by day of week and day of month.
    
    Args:
        data: DataFrame with date and value columns
        value_col: Column name for values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Prepare data
    data = data.copy()
    data['day_of_week'] = data['date'].dt.day_name()
    data['day_of_month'] = data['date'].dt.day
    
    # Create pivot table
    heatmap_data = data.groupby(['day_of_week', 'day_of_month'])[value_col].mean().unstack(fill_value=0)
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        hovertemplate='Day of Month: %{x}<br>Day of Week: %{y}<br>Avg Amount: $%{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Day of Month",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig


def create_stacked_forecast_chart(historical_data: Dict[str, pd.Series],
                                 forecast_data: Dict[str, pd.Series],
                                 title: str = "Historical and Forecast by Category") -> go.Figure:
    """
    Create stacked area chart showing historical and forecast data.
    
    Args:
        historical_data: Dict of category -> historical series
        forecast_data: Dict of category -> forecast series
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add historical data
    for category, series in historical_data.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=f"{category} (Historical)",
            stackgroup="historical",
            mode='lines',
            line=dict(width=0.5),
            hovertemplate=f'<b>{category}</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: $%{{y:,.2f}}<extra></extra>'
        ))
    
    # Add forecast data
    for category, series in forecast_data.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=f"{category} (Forecast)",
            stackgroup="forecast",
            mode='lines',
            line=dict(width=0.5, dash='dash'),
            hovertemplate=f'<b>{category}</b><br>Date: %{{x|%Y-%m-%d}}<br>Forecast: $%{{y:,.2f}}<extra></extra>'
        ))
    
    fig.update_layout(**create_base_layout(title))
    return fig


def create_metrics_radar_chart(metrics: Dict[str, float],
                              title: str = "Performance Metrics") -> go.Figure:
    """
    Create radar chart for performance metrics.
    
    Args:
        metrics: Dict of metric_name -> value
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Normalize metrics to 0-100 scale for radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        hovertemplate='<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig


def create_distribution_chart(data: pd.Series,
                            title: str = "Data Distribution",
                            chart_type: str = "histogram") -> go.Figure:
    """
    Create distribution chart (histogram or box plot).
    
    Args:
        data: Data series
        title: Chart title
        chart_type: Chart type ("histogram", "box")
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if chart_type == "box":
        fig.add_trace(go.Box(
            y=data.values,
            name="Distribution",
            boxpoints='outliers',
            hovertemplate='Value: $%{y:,.2f}<extra></extra>'
        ))
        layout = create_base_layout(title, yaxis_title="Amount")
        layout['xaxis_title'] = ""
    else:  # histogram
        fig.add_trace(go.Histogram(
            x=data.values,
            nbinsx=30,
            name="Distribution",
            hovertemplate='Range: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
        ))
        layout = create_base_layout(title, xaxis_title="Amount", yaxis_title="Frequency")
    
    fig.update_layout(**layout)
    return fig


def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                              title: str = "Correlation Matrix") -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=max(400, len(correlation_matrix) * 30)
    )
    
    return fig


def add_annotations_to_chart(fig: go.Figure, 
                           annotations: List[Dict[str, Any]]) -> go.Figure:
    """
    Add annotations to existing chart.
    
    Args:
        fig: Plotly figure
        annotations: List of annotation configurations
        
    Returns:
        Updated figure
    """
    for annotation in annotations:
        fig.add_annotation(
            x=annotation.get('x'),
            y=annotation.get('y'),
            text=annotation.get('text', ''),
            showarrow=annotation.get('showarrow', True),
            arrowhead=annotation.get('arrowhead', 2),
            arrowsize=annotation.get('arrowsize', 1),
            arrowwidth=annotation.get('arrowwidth', 2),
            arrowcolor=annotation.get('arrowcolor', 'red'),
            bgcolor=annotation.get('bgcolor', 'white'),
            bordercolor=annotation.get('bordercolor', 'black'),
            borderwidth=annotation.get('borderwidth', 1)
        )
    
    return fig


def style_chart_for_theme(fig: go.Figure, theme: str = "default") -> go.Figure:
    """
    Apply theme styling to chart.
    
    Args:
        fig: Plotly figure
        theme: Theme name ("default", "dark", "minimal")
        
    Returns:
        Styled figure
    """
    if theme == "dark":
        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            xaxis=dict(gridcolor='#404040'),
            yaxis=dict(gridcolor='#404040')
        )
    elif theme == "minimal":
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
    
    return fig


def export_chart(fig: go.Figure, 
                filename: str, 
                format: str = "png",
                width: int = 1200,
                height: int = 600) -> bytes:
    """
    Export chart to various formats.
    
    Args:
        fig: Plotly figure
        filename: Output filename
        format: Export format ("png", "pdf", "svg", "html")
        width: Export width
        height: Export height
        
    Returns:
        Exported file bytes
    """
    if format.lower() == "html":
        return fig.to_html().encode()
    else:
        return fig.to_image(format=format, width=width, height=height)