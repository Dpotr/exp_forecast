import pandas as pd
import numpy as np
import streamlit as st
<<<<<<< HEAD
import plotly.graph_objects as go
=======
>>>>>>> 595a9bf1857c174548d23beaa8c4b7cf6644fd72
from datetime import datetime, timedelta
from forecast_utils import aggregate_daily_to_weekly, calculate_weekly_metrics, add_week_metadata

def generate_test_data():
    """Generate test data for weekly aggregation testing."""
    # Create a date range for the last 30 days
    dates = pd.date_range(end=datetime.today(), periods=30).date
    
    # Create test data with some weekly patterns
    np.random.seed(42)
    base_values = np.random.normal(100, 20, 30).clip(0)  # Base values with some noise
    weekly_pattern = np.array([1.2, 0.9, 1.1, 0.8, 1.3, 1.5, 0.5])  # Weekly pattern (Mon-Sun)
    
    # Apply weekly pattern
    day_of_week = np.array([d.weekday() for d in dates])
    values = base_values * weekly_pattern[day_of_week]
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'actual': values,
        'forecast': values * np.random.normal(1.0, 0.1, 30),  # Add some forecast error
    })
    
    # Calculate error metrics
    df['error'] = df['forecast'] - df['actual']
    df['abs_error'] = np.abs(df['error'])
    df['pct_error'] = np.where(df['actual'] > 0, (df['abs_error'] / df['actual']) * 100, np.nan)
    df['within_10pct'] = (df['pct_error'] <= 10).astype(int)
    df['within_20pct'] = (df['pct_error'] <= 20).astype(int)
    
    return df

def test_weekly_aggregation():
    """Test the weekly aggregation functions."""
    st.header("Weekly Aggregation Test")
    
    # Generate test data
    test_data = generate_test_data()
    
    st.subheader("Test Data (Daily)")
    st.dataframe(test_data)
    
    # Test weekly aggregation
    st.subheader("Weekly Aggregation")
    weekly_data = aggregate_daily_to_weekly(test_data)
    
    if not weekly_data.empty:
        st.write("Weekly aggregation successful!")
        st.dataframe(weekly_data)
        
        # Test adding week metadata
        weekly_with_meta = add_week_metadata(weekly_data)
        st.subheader("With Week Metadata")
        st.dataframe(weekly_with_meta)
        
        # Test weekly metrics
        metrics = calculate_weekly_metrics(weekly_data)
        st.subheader("Weekly Metrics")
        st.json(metrics)
        
        # Visualize the results
        st.subheader("Visualization")
        
        # Daily vs Weekly comparison
        fig = go.Figure()
        
        # Add daily data
        fig.add_trace(go.Scatter(
            x=test_data['date'],
            y=test_data['actual'],
            name='Daily Actual',
            mode='lines+markers',
            line=dict(color='lightblue', width=1)
        ))
        
        # Add weekly data
        fig.add_trace(go.Scatter(
            x=weekly_data['date'],
            y=weekly_data['actual'],
            name='Weekly Actual',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Daily vs Weekly Aggregated Data',
            xaxis_title='Date',
            yaxis_title='Amount',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error metrics comparison
        fig_metrics = go.Figure()
        
        # Daily metrics
        daily_mape = test_data['pct_error'].mean()
        daily_accuracy_10pct = test_data['within_10pct'].mean() * 100
        
        # Weekly metrics
        weekly_mape = metrics['avg_mape']
        weekly_accuracy_10pct = metrics['accuracy_10pct']
        
        fig_metrics.add_trace(go.Bar(
            x=['Daily', 'Weekly'],
            y=[daily_mape, weekly_mape],
            name='MAPE (%)',
            text=[f'{daily_mape:.1f}%', f'{weekly_mape:.1f}%'],
            textposition='auto'
        ))
        
        fig_metrics.add_trace(go.Bar(
            x=['Daily', 'Weekly'],
            y=[daily_accuracy_10pct, weekly_accuracy_10pct],
            name='Accuracy (±10%)',
            text=[f'{daily_accuracy_10pct:.1f}%', f'{weekly_accuracy_10pct:.1f}%'],
            textposition='auto'
        ))
        
        fig_metrics.update_layout(
            title='Forecast Metrics: Daily vs Weekly',
            barmode='group',
            yaxis_title='Value (%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        return True
    else:
        st.error("Weekly aggregation failed!")
        return False

if __name__ == "__main__":
    import streamlit as st
    import plotly.graph_objects as go
    
    st.set_page_config(page_title="Weekly Aggregation Test", layout="wide")
    test_weekly_aggregation()
