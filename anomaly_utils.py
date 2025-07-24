import pandas as pd
import numpy as np

def detect_outliers(df, window_days=60, z_thresh=3):
    # Detects days with total amount > z_thresh std from rolling mean
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=window_days))]
    daily = last.groupby('date')['amount'].sum().reset_index()
    daily['mean'] = daily['amount'].rolling(window=7, min_periods=1).mean()
    daily['std'] = daily['amount'].rolling(window=7, min_periods=1).std(ddof=0)
    daily['z'] = (daily['amount'] - daily['mean']) / (daily['std'] + 1e-8)
    outlier_days = daily[daily['z'].abs() > z_thresh]
    return outlier_days[['date', 'amount', 'z']]

def detect_anomaly_transactions(df, z_thresh=3):
    # Detects individual transactions that are outliers within their category (last 60 days)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=60))]
    anomalies = []
    for cat, group in last.groupby('category'):
        if len(group) < 10:
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

def recurring_payments(df, min_periods=3, threshold=0.7):
    # Detects recurring payments (amounts and dates)
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
        if std < 3 and median in [7,14,28,30,31]:
            rec.append({'category': cat, 'interval_days': median, 'last_date': group['date'].max(), 'amount': group['amount'].iloc[-1]})
    return pd.DataFrame(rec)
