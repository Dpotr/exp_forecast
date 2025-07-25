import pandas as pd
import numpy as np
from config import config

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
    
    
def aggregate_daily_to_monthly(daily_results):
    """
    Aggregate daily forecast results to monthly (calendar month) level and calculate metrics on monthly aggregates.
    Args:
        daily_results: DataFrame with columns ['date', 'actual', 'forecast', 'error', 'abs_error', 'pct_error', 'within_10pct', 'within_20pct']
    Returns:
        DataFrame with monthly aggregated metrics
    """
    if daily_results.empty:
        return pd.DataFrame()
    df = daily_results.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # 'M' = calendar month end
    monthly = df.resample('M').agg({
        'actual': 'sum',
        'forecast': 'sum'
    })
    monthly['error'] = monthly['forecast'] - monthly['actual']
    monthly['abs_error'] = monthly['error'].abs()
    monthly['pct_error'] = np.where(
        monthly['actual'] > 0,
        (monthly['abs_error'] / monthly['actual']) * 100,
        np.nan
    )
    monthly['within_10pct'] = (monthly['pct_error'] <= 10).astype(int)
    monthly['within_20pct'] = (monthly['pct_error'] <= 20).astype(int)
    monthly = monthly.reset_index()
    # Ensure the date column is named 'date' and is a timestamp (first day of month)
    monthly['date'] = monthly['date'].dt.to_period('M').dt.start_time
    return monthly


def add_rolling_features(ts, windows=[7, 14, 30]):
    """
    Add rolling average features to time series for improved forecasting.
    
    Args:
        ts: Time series (pandas Series with datetime index)
        windows: List of rolling window sizes in days
        
    Returns:
        DataFrame with original values and rolling features
    """
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    if len(ts) < max(windows):
        # Not enough data for rolling features
        return pd.DataFrame({'value': ts})
    
    df = pd.DataFrame({'value': ts})
    
    # Add rolling averages
    for window in windows:
        if len(ts) >= window:
            df[f'rolling_mean_{window}d'] = ts.rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}d'] = ts.rolling(window=window, min_periods=1).std()
    
    # Add trend indicators
    if len(ts) >= 7:
        df['trend_7d'] = ts.rolling(7).mean().diff()
    if len(ts) >= 14:
        df['trend_14d'] = ts.rolling(14).mean().diff()
    
    # Add volatility measures
    if len(ts) >= 7:
        df['volatility_7d'] = ts.rolling(7).std() / ts.rolling(7).mean()
    
    return df


def cap_outliers(ts, method='percentile', cap_percentile=None, floor_percentile=None):
    """
    Cap outliers instead of removing them to preserve data integrity.
    
    Args:
        ts: Time series (pandas Series)
        method: 'percentile' or 'iqr'
        cap_percentile: Upper percentile for capping (default from config)
        floor_percentile: Lower percentile for flooring (default from config)
        
    Returns:
        Time series with capped outliers
    """
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    if len(ts) < 10:  # Not enough data for meaningful capping
        return ts
    
    ts_capped = ts.copy()
    
    if method == 'percentile':
        cap_pct = cap_percentile or config.OUTLIER_CAP_PERCENTILE
        floor_pct = floor_percentile or config.OUTLIER_FLOOR_PERCENTILE
        
        upper_cap = np.percentile(ts.dropna(), cap_pct)
        lower_cap = np.percentile(ts.dropna(), floor_pct)
        
        ts_capped = ts_capped.clip(lower=lower_cap, upper=upper_cap)
        
    elif method == 'iqr':
        Q1 = ts.quantile(0.25)
        Q3 = ts.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_cap = Q1 - config.IQR_MULTIPLIER * IQR
        upper_cap = Q3 + config.IQR_MULTIPLIER * IQR
        
        ts_capped = ts_capped.clip(lower=lower_cap, upper=upper_cap)
    
    return ts_capped


def enhance_time_series(ts, add_features=True, cap_outliers_flag=True):
    """
    Enhance time series with rolling features and outlier capping.
    
    Args:
        ts: Time series (pandas Series)
        add_features: Whether to add rolling features
        cap_outliers_flag: Whether to cap outliers
        
    Returns:
        Enhanced time series or DataFrame with features
    """
    if cap_outliers_flag and config.OUTLIER_CAPPING_ENABLED:
        ts = cap_outliers(ts)
    
    if add_features:
        return add_rolling_features(ts)
    else:
        return ts
