import pandas as pd
import numpy as np

def aggregate_daily_to_weekly(daily_results):
    """
    Aggregate daily forecast results to weekly level and calculate metrics on weekly aggregates.
    
    Args:
        daily_results: DataFrame with columns ['date', 'actual', 'forecast', 'error', 'abs_error', 'pct_error', 'within_10pct', 'within_20pct']
        
    Returns:
        DataFrame with weekly aggregated metrics
    """
    if daily_results.empty:
        return pd.DataFrame()
        
    # Create a copy to avoid modifying the original
    df = daily_results.copy()
    
    # Ensure date is datetime and set as index for resampling
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # First, aggregate actual and forecast values by week
    weekly = df.resample('W-SUN').agg({
        'actual': 'sum',
        'forecast': 'sum'
    })
    
    # Calculate error metrics on the weekly aggregates
    weekly['error'] = weekly['forecast'] - weekly['actual']
    weekly['abs_error'] = weekly['error'].abs()
    weekly['pct_error'] = np.where(
        weekly['actual'] > 0,
        (weekly['abs_error'] / weekly['actual']) * 100,
        np.nan
    )
    weekly['within_10pct'] = (weekly['pct_error'] <= 10).astype(int)
    weekly['within_20pct'] = (weekly['pct_error'] <= 20).astype(int)
    
    # Reset index to make date a column again
    weekly = weekly.reset_index()
    
    return weekly

def calculate_weekly_metrics(weekly_df):
    """
    Calculate summary metrics for weekly forecast results.
    
    Args:
        weekly_df: DataFrame with weekly aggregated forecast results
        
    Returns:
        Dictionary of summary metrics
    """
    if weekly_df.empty or len(weekly_df) == 0:
        return {
            'avg_mape': np.nan,
            'accuracy_10pct': 0,
            'accuracy_20pct': 0,
            'avg_weekly_actual': 0,
            'avg_weekly_forecast': 0,
            'total_actual': 0,
            'total_forecast': 0,
            'total_error': 0,
            'total_abs_error': 0
        }
    
    # Calculate metrics based on weekly aggregates
    total_actual = weekly_df['actual'].sum()
    total_forecast = weekly_df['forecast'].sum()
    total_error = total_forecast - total_actual
    total_abs_error = abs(total_error)
    
    # Calculate MAPE based on totals (more accurate for aggregated data)
    avg_mape = (total_abs_error / total_actual * 100) if total_actual > 0 else np.nan
    
    # Calculate accuracy within thresholds based on weekly aggregates
    accuracy_10pct = (weekly_df['within_10pct'].mean() * 100) if 'within_10pct' in weekly_df.columns else 0
    accuracy_20pct = (weekly_df['within_20pct'].mean() * 100) if 'within_20pct' in weekly_df.columns else 0
    
    # Calculate average weekly actual and forecast
    avg_weekly_actual = weekly_df['actual'].mean()
    avg_weekly_forecast = weekly_df['forecast'].mean()
    
    return {
        'avg_mape': avg_mape,
        'accuracy_10pct': accuracy_10pct,
        'accuracy_20pct': accuracy_20pct,
        'avg_weekly_actual': avg_weekly_actual,
        'avg_weekly_forecast': avg_weekly_forecast,
        'total_actual': total_actual,
        'total_forecast': total_forecast,
        'total_error': total_error,
        'total_abs_error': total_abs_error
    }

def add_week_metadata(weekly_df):
    """
    Add week number and year-week string to the weekly dataframe.
    
    Args:
        weekly_df: DataFrame with 'date' column
        
    Returns:
        DataFrame with additional week metadata columns
    """
    if weekly_df.empty:
        return weekly_df
        
    df = weekly_df.copy()
    df['week_number'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    df['year_week'] = df['date'].dt.strftime('%Y-W%U')
    
    return df
