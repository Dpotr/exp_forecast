import pandas as pd
import numpy as np
from scipy import stats
from config import config

def detect_outliers(df, window_days=None, z_thresh=None):
    # Detects days with total amount > z_thresh std from rolling mean
    if window_days is None:
        window_days = config.OUTLIER_WINDOW_DAYS
    if z_thresh is None:
        z_thresh = config.OUTLIER_Z_THRESHOLD
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    daily = last.groupby('date')['amount'].sum().reset_index()
    daily['mean'] = daily['amount'].rolling(window=config.ROLLING_WINDOW_DAYS, min_periods=1).mean()
    daily['std'] = daily['amount'].rolling(window=config.ROLLING_WINDOW_DAYS, min_periods=1).std(ddof=0)
    daily['z'] = (daily['amount'] - daily['mean']) / (daily['std'] + 1e-8)
    outlier_days = daily[daily['z'].abs() > z_thresh]
    return outlier_days[['date', 'amount', 'z']]

def detect_anomaly_transactions(df, z_thresh=None, window_days=None, min_transactions=None):
    # Detects individual transactions that are outliers within their category
    if z_thresh is None:
        z_thresh = config.ANOMALY_Z_THRESHOLD
    if window_days is None:
        window_days = config.ANOMALY_WINDOW_DAYS
    if min_transactions is None:
        min_transactions = config.ANOMALY_MIN_TRANSACTIONS
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    anomalies = []
    for cat, group in last.groupby('category'):
        if len(group) < min_transactions:
            continue
        mean = group['amount'].mean()
        std = group['amount'].std(ddof=0)
        z = (group['amount'] - mean) / (std + 1e-8)
        outliers = group[z.abs() > z_thresh]
        if not outliers.empty:
            outliers = outliers.assign(z=z[outliers.index])
            anomalies.append(outliers[['date','category','amount','z']])
    if anomalies:
        return pd.concat(anomalies)
    else:
        return pd.DataFrame(columns=['date','category','amount','z'])

def recurring_payments(df, min_periods=None, threshold=None, std_threshold=None):
    # Detects recurring payments (amounts and dates)
    if min_periods is None:
        min_periods = config.MIN_RECURRING_PERIODS
    if threshold is None:
        threshold = config.RECURRING_THRESHOLD
    if std_threshold is None:
        std_threshold = config.RECURRING_STD_THRESHOLD
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    rec = []
    for cat, group in df.groupby('category'):
        group = group.sort_values('date')
        if len(group) < min_periods:
            continue
        diffs = group['date'].diff().dt.days.dropna()
        if diffs.empty:
            continue
        median = diffs.median()
        std = diffs.std(ddof=0)
        # Recurring if std is low and median interval is 7/14/30/31
        if std < std_threshold and median in [7,14,28,30,31]:
            rec.append({'category': cat, 'interval_days': median, 'last_date': group['date'].max(), 'amount': group['amount'].iloc[-1]})
    return pd.DataFrame(rec)

def detect_outliers_iqr(df, window_days=None):
    """
    Enhanced outlier detection using IQR method with seasonal awareness
    """
    if window_days is None:
        window_days = config.OUTLIER_WINDOW_DAYS
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    daily = last.groupby('date')['amount'].sum().reset_index()
    
    if len(daily) < config.MIN_BASELINE_DAYS:
        return pd.DataFrame(columns=['date', 'amount', 'method', 'score'])
    
    # Calculate IQR-based outliers
    Q1 = daily['amount'].quantile(0.25)
    Q3 = daily['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
    upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
    
    outliers = daily[(daily['amount'] < lower_bound) | (daily['amount'] > upper_bound)].copy()
    outliers['method'] = 'IQR'
    outliers['score'] = np.where(
        outliers['amount'] > upper_bound,
        (outliers['amount'] - upper_bound) / IQR,
        (lower_bound - outliers['amount']) / IQR
    )
    
    return outliers[['date', 'amount', 'method', 'score']]

def detect_outliers_modified_zscore(df, window_days=None):
    """
    Modified Z-score using median and MAD (more robust to outliers)
    """
    if window_days is None:
        window_days = config.OUTLIER_WINDOW_DAYS
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    daily = last.groupby('date')['amount'].sum().reset_index()
    
    if len(daily) < config.MIN_BASELINE_DAYS:
        return pd.DataFrame(columns=['date', 'amount', 'method', 'score'])
    
    median = daily['amount'].median()
    mad = np.median(np.abs(daily['amount'] - median))
    
    if mad == 0:
        return pd.DataFrame(columns=['date', 'amount', 'method', 'score'])
    
    modified_z_scores = 0.6745 * (daily['amount'] - median) / mad
    threshold = stats.norm.ppf(config.ANOMALY_CONFIDENCE_LEVEL)
    
    outliers = daily[np.abs(modified_z_scores) > threshold].copy()
    outliers['method'] = 'Modified Z-Score'
    outliers['score'] = modified_z_scores[outliers.index]
    
    return outliers[['date', 'amount', 'method', 'score']]

def detect_seasonal_anomalies(df, window_days=None):
    """
    Detect anomalies considering day-of-week patterns
    """
    if window_days is None:
        window_days = config.OUTLIER_WINDOW_DAYS
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    daily = last.groupby('date')['amount'].sum().reset_index()
    
    if len(daily) < config.MIN_BASELINE_DAYS:
        return pd.DataFrame(columns=['date', 'amount', 'method', 'score'])
    
    daily['day_of_week'] = daily['date'].dt.day_name()
    
    outliers_list = []
    for day, group in daily.groupby('day_of_week'):
        if len(group) < 3:
            continue
        
        mean_amount = group['amount'].mean()
        std_amount = group['amount'].std(ddof=0)
        
        if std_amount == 0:
            continue
            
        z_scores = (group['amount'] - mean_amount) / std_amount
        day_outliers = group[np.abs(z_scores) > config.OUTLIER_Z_THRESHOLD].copy()
        
        if not day_outliers.empty:
            day_outliers['method'] = f'Seasonal ({day})'
            day_outliers['score'] = z_scores[day_outliers.index]
            outliers_list.append(day_outliers[['date', 'amount', 'method', 'score']])
    
    if outliers_list:
        return pd.concat(outliers_list)
    else:
        return pd.DataFrame(columns=['date', 'amount', 'method', 'score'])

def detect_category_anomalies_enhanced(df, z_thresh=None, window_days=None, min_transactions=None):
    """
    Enhanced category-wise anomaly detection with multiple methods
    """
    if z_thresh is None:
        z_thresh = config.ANOMALY_Z_THRESHOLD
    if window_days is None:
        window_days = config.ANOMALY_WINDOW_DAYS
    if min_transactions is None:
        min_transactions = config.ANOMALY_MIN_TRANSACTIONS
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    
    all_anomalies = []
    
    for cat, group in last.groupby('category'):
        if len(group) < min_transactions:
            continue
        
        # Method 1: Standard Z-score
        mean_amt = group['amount'].mean()
        std_amt = group['amount'].std(ddof=0)
        if std_amt > 0:
            z_scores = (group['amount'] - mean_amt) / std_amt
            z_outliers = group[np.abs(z_scores) > z_thresh].copy()
            if not z_outliers.empty:
                z_outliers['method'] = 'Z-Score'
                z_outliers['score'] = z_scores[z_outliers.index]
                all_anomalies.append(z_outliers[['date', 'category', 'amount', 'method', 'score']])
        
        # Method 2: IQR method
        Q1 = group['amount'].quantile(0.25)
        Q3 = group['amount'].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
            iqr_outliers = group[(group['amount'] < lower_bound) | (group['amount'] > upper_bound)].copy()
            if not iqr_outliers.empty:
                iqr_outliers['method'] = 'IQR'
                iqr_outliers['score'] = np.where(
                    iqr_outliers['amount'] > upper_bound,
                    (iqr_outliers['amount'] - upper_bound) / IQR,
                    (lower_bound - iqr_outliers['amount']) / IQR
                )
                all_anomalies.append(iqr_outliers[['date', 'category', 'amount', 'method', 'score']])
    
    if all_anomalies:
        result = pd.concat(all_anomalies).drop_duplicates(subset=['date', 'category', 'amount'])
        return result.sort_values(['date', 'category'])
    else:
        return pd.DataFrame(columns=['date', 'category', 'amount', 'method', 'score'])

def get_comprehensive_anomalies(df):
    """
    Combine all anomaly detection methods and return consolidated results
    """
    results = {
        'daily_zscore': detect_outliers(df),
        'daily_iqr': detect_outliers_iqr(df),
        'daily_modified_zscore': detect_outliers_modified_zscore(df),
        'daily_seasonal': detect_seasonal_anomalies(df),
        'category_enhanced': detect_category_anomalies_enhanced(df)
    }
    
    return results

def create_anomaly_visualization(df, anomalies_dict):
    """
    Create enhanced visualizations for anomaly detection results
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Prepare daily data
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    daily_amounts = df_copy.groupby('date')['amount'].sum().reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Spending with Anomalies', 'Anomaly Methods Comparison', 
                       'Category Anomalies Heatmap', 'Anomaly Score Distribution'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "histogram"}]]
    )
    
    # Plot 1: Daily spending with anomalies highlighted
    fig.add_trace(
        go.Scatter(x=daily_amounts['date'], y=daily_amounts['amount'],
                  mode='lines+markers', name='Daily Spending',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add anomalies from different methods with different colors
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    method_names = ['daily_zscore', 'daily_iqr', 'daily_modified_zscore', 'daily_seasonal']
    
    for i, (method, anomaly_df) in enumerate(anomalies_dict.items()):
        if method in method_names and not anomaly_df.empty:
            fig.add_trace(
                go.Scatter(x=anomaly_df['date'], y=anomaly_df['amount'],
                          mode='markers', name=f'{method.replace("_", " ").title()}',
                          marker=dict(color=colors[i % len(colors)], size=10, 
                                    symbol='diamond')),
                row=1, col=1
            )
    
    # Plot 2: Method comparison (count of anomalies by method)
    method_counts = {}
    for method, anomaly_df in anomalies_dict.items():
        if method != 'category_enhanced':
            method_counts[method.replace('_', ' ').title()] = len(anomaly_df)
    
    if method_counts:
        fig.add_trace(
            go.Bar(x=list(method_counts.keys()), y=list(method_counts.values()),
                  marker_color='lightblue', name='Anomaly Count'),
            row=1, col=2
        )
    
    # Plot 3: Category anomalies heatmap
    if 'category_enhanced' in anomalies_dict and not anomalies_dict['category_enhanced'].empty:
        cat_anomalies = anomalies_dict['category_enhanced']
        cat_anomalies['date_str'] = cat_anomalies['date'].dt.strftime('%Y-%m-%d')
        
        # Create pivot table for heatmap
        heatmap_data = cat_anomalies.pivot_table(
            index='category', columns='date_str', values='score', 
            aggfunc='max', fill_value=0
        )
        
        if not heatmap_data.empty:
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values,
                          x=heatmap_data.columns,
                          y=heatmap_data.index,
                          colorscale='Reds',
                          name='Anomaly Scores'),
                row=2, col=1
            )
    
    # Plot 4: Anomaly score distribution
    all_scores = []
    for method, anomaly_df in anomalies_dict.items():
        if not anomaly_df.empty and 'score' in anomaly_df.columns:
            all_scores.extend(abs(anomaly_df['score'].values))
    
    if all_scores:
        fig.add_trace(
            go.Histogram(x=all_scores, nbinsx=20, 
                        marker_color='lightgreen',
                        name='Score Distribution'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Comprehensive Anomaly Detection Dashboard"
    )
    
    return fig

def create_anomaly_summary_table(anomalies_dict):
    """
    Create a summary table of all detected anomalies
    """
    summary_data = []
    
    for method, anomaly_df in anomalies_dict.items():
        if not anomaly_df.empty:
            method_name = method.replace('_', ' ').title()
            count = len(anomaly_df)
            
            if 'score' in anomaly_df.columns:
                avg_score = anomaly_df['score'].abs().mean()
                max_score = anomaly_df['score'].abs().max()
                
                if 'amount' in anomaly_df.columns:
                    max_amount = anomaly_df['amount'].max()
                    recent_date = anomaly_df['date'].max() if 'date' in anomaly_df.columns else None
                    
                    summary_data.append({
                        'Method': method_name,
                        'Count': count,
                        'Avg Score': f"{avg_score:.2f}",
                        'Max Score': f"{max_score:.2f}",
                        'Max Amount': f"${max_amount:.2f}",
                        'Most Recent': recent_date.strftime('%Y-%m-%d') if recent_date else 'N/A'
                    })
    
    return pd.DataFrame(summary_data)

def highlight_anomalies_in_chart(fig, anomalies_df, date_col='date', amount_col='amount'):
    """
    Add anomaly highlights to existing plotly chart
    """
    if not anomalies_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies_df[date_col],
                y=anomalies_df[amount_col],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                name='Anomalies',
                hovertemplate='<b>Anomaly Detected</b><br>' +
                             'Date: %{x}<br>' +
                             'Amount: $%{y:.2f}<br>' +
                             '<extra></extra>'
            )
        )
    return fig
