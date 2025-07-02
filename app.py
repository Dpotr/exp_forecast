import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from pandas import ExcelWriter
import subprocess
from croston import croston
from anomaly_utils import detect_outliers, detect_anomaly_transactions, recurring_payments

def add_country_column(df):
    # Define date ranges for each country
    vietnam_end = datetime(2023, 8, 31)
    kazakhstan_start = datetime(2023, 9, 1)
    kazakhstan_end = datetime(2024, 3, 31)
    montenegro_start = datetime(2024, 4, 1)
    
    def get_country(date):
        if date <= vietnam_end:
            return 'Vietnam'
        elif kazakhstan_start <= date <= kazakhstan_end:
            return 'Kazakhstan'
        elif date >= montenegro_start:
            return 'Montenegro'
        else:
            return 'Unknown'
    
    df['country'] = df['date'].apply(get_country)
    return df

# --- Git auto-update function ---
def git_auto_update():
    try:
        # Get changed files
        changed = subprocess.check_output(['git', 'status', '--porcelain'], encoding='utf-8')
        if not changed.strip():
            st.info("No changes to commit.")
            return
        # Short summary
        changed_files = [line[3:] for line in changed.strip().split('\n') if line]
        summary = ", ".join(os.path.basename(f) for f in changed_files)
        commit_msg = f"Auto-update: {summary}"
        subprocess.check_call(['git', 'add', '.'])
        subprocess.check_call(['git', 'commit', '-m', commit_msg])
        subprocess.check_call(['git', 'push'])
        st.success(f"Changes committed and pushed: {commit_msg}")
    except Exception as e:
        st.error(f"Git auto-update failed: {e}")

st.set_page_config(page_title="Expense Forecast Dashboard", layout="wide")
st.title("Expense Forecast Dashboard")

# --- Streamlit button for git auto-update (always visible) ---
with st.sidebar:
    if st.button("Commit & Push to GitHub"):
        git_auto_update()

uploaded_file = st.file_uploader("Upload your expenses.xlsx file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel("expenses.xlsx")

df['date'] = pd.to_datetime(df['date'])
df = add_country_column(df)

# --- Category Filtering ---
st.sidebar.header("Category Filter")
all_categories = sorted(df['category'].unique())
def_select = all_categories if 'selected_categories' not in st.session_state else st.session_state['selected_categories']
select_all = st.sidebar.button("Select All Categories")
deselect_all = st.sidebar.button("Deselect All Categories")
if select_all:
    selected_categories = all_categories
elif deselect_all:
    selected_categories = []
else:
    selected_categories = st.sidebar.multiselect("Choose categories to show:", all_categories, default=def_select)
st.session_state['selected_categories'] = selected_categories
if selected_categories:
    df = df[df['category'].isin(selected_categories)]
else:
    st.warning("No categories selected. Showing empty dashboard.")

# --- Outlier & Anomaly Detection ---
st.header("Anomaly & Outlier Detection")
outlier_days = detect_outliers(df)
if not outlier_days.empty:
    st.subheader("Outlier Days (Last 60 Days)")
    st.caption("Days with total spending far from the 7-day rolling mean (z-score > 3)")
    st.dataframe(outlier_days)
else:
    st.info("No daily outliers detected in the last 60 days.")

anomaly_tx = detect_anomaly_transactions(df)
if not anomaly_tx.empty:
    st.subheader("Anomalous Transactions (Last 60 Days)")
    st.caption("Transactions with unusually high/low amounts for their category (z-score > 3)")
    st.dataframe(anomaly_tx)
else:
    st.info("No anomalous transactions detected in the last 60 days.")

# --- Recurring Payments Detection ---
st.header("Recurring Payments & Alerts")
rec_df = recurring_payments(df)
if not rec_df.empty:
    st.subheader("Detected Recurring Payments")
    st.caption("Auto-detected subscriptions/rent with interval and next due date")
    st.dataframe(rec_df)
    today = pd.to_datetime(df['date'].max())
    overdue = rec_df[(today - rec_df['last_date']).dt.days > rec_df['interval_days']]
    if not overdue.empty:
        st.warning(f"Some recurring payments may be overdue: {overdue['category'].tolist()}")
else:
    st.info("No recurring payments detected.")

# --- Expense Breakdown ---
st.header("Expense Breakdown & Cumulative Totals")
# Top categories pie chart
cat_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False)
fig_pie = px.pie(cat_sum, values=cat_sum.values, names=cat_sum.index, title="Top Categories (Total)")
st.plotly_chart(fig_pie, use_container_width=True)
# Running total
df_sorted = df.sort_values('date')
df_sorted['cumulative'] = df_sorted['amount'].cumsum()
st.line_chart(df_sorted.set_index('date')['cumulative'], use_container_width=True)

# --- Comparisons ---
st.header("Comparisons & Benchmarks")
# Month-on-Month comparison
df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly_totals = df.groupby('year_month')['amount'].sum().reset_index()
st.bar_chart(monthly_totals.set_index('year_month'), use_container_width=True)
# Week-on-Week comparison
df['year_week'] = df['date'].dt.strftime('%Y-%U')
weekly_totals = df.groupby('year_week')['amount'].sum().reset_index()
st.line_chart(weekly_totals.set_index('year_week'), use_container_width=True)

# --- Export & Sharing ---
st.header("Export & Sharing")
if st.button("Download All Data as CSV"):
    df.to_csv("expenses_export.csv", index=False)
    st.success("Exported to expenses_export.csv")

# --- User Guidance & Explanations ---
st.header("User Guidance & Explanations")
st.info("Hover over any chart or table for tooltips. Outliers/anomalies use z-score > 3. Recurring payments are detected if intervals are regular. Forecast methods are explained in the diagnostics table. For details, see the README.")

# --- Dashboard Layout: Group Boards in Logical Blocks ---
# 1. Raw Data Preview
st.header("Raw Data")
st.subheader("Raw Data Preview")
st.caption("First 20 rows of the raw data")
st.dataframe(df.head(20))

# --- Seasonality Heatmap ---
st.subheader("Average Spending by Day of Week and Day of Month (Last 60 Days)")
last_60 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=60))]
last_60['weekday'] = last_60['date'].dt.day_name()
last_60['dom'] = last_60['date'].dt.day
pivot = last_60.pivot_table(index='weekday', columns='dom', values='amount', aggfunc='mean', fill_value=0)
# Reorder weekdays
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
pivot = pivot.reindex(weekday_order)
import plotly.express as px
fig_heat = px.imshow(pivot.values, labels=dict(x="Day of Month", y="Day of Week", color="Avg Amount"),
                     x=pivot.columns, y=pivot.index, aspect="auto", color_continuous_scale='Viridis')
fig_heat.update_layout(title="Expense Seasonality Heatmap (Last 60 Days)", height=350)
st.plotly_chart(fig_heat, use_container_width=True)

# 2. Monthly Summary & Country Regimes
st.header("Monthly Expenses")
st.subheader("Expenses by Category and Country")
agg = df.groupby(['country', 'category'])['amount'].sum().reset_index()
st.caption("Total expenses by category and country")
st.dataframe(agg)

df['year_month'] = df['date'].dt.to_period('M').astype(str)
monthly = df.groupby(['year_month', 'category', 'country'])['amount'].sum().reset_index()
# Pivot for stacked bar
pivot = monthly.pivot_table(index=['year_month', 'country'], columns='category', values='amount', fill_value=0)
pivot = pivot.reset_index()

# Prepare for plotting
categories = [c for c in pivot.columns if c not in ['year_month','country']]

fig = go.Figure()

for cat in categories:
    fig.add_trace(go.Bar(
        x=pivot['year_month'],
        y=pivot[cat],
        name=cat,
        marker_line_width=0,
    ))

# Highlight country regions
country_periods = [
    {"name": "Vietnam", "start": "2022-10", "end": "2023-08", "color": "rgba(0,200,0,0.1)"},
    {"name": "Kazakhstan", "start": "2023-09", "end": "2024-03", "color": "rgba(0,0,200,0.1)"},
    {"name": "Montenegro", "start": "2024-04", "end": df['year_month'].max(), "color": "rgba(200,0,0,0.1)"},
]

for period in country_periods:
    fig.add_vrect(
        x0=period['start'], x1=period['end'],
        fillcolor=period['color'], opacity=0.3, layer="below", line_width=0,
        annotation_text=period['name'], annotation_position="top left"
    )

fig.update_layout(
    barmode='stack',
    title="Monthly Expenses by Category (Stacked, with Country Periods)",
    xaxis_title="Month",
    yaxis_title="Total Expenses",
    legend_title="Category",
    xaxis=dict(type='category'),
    margin=dict(l=40, r=40, t=60, b=40),
    height=500,
)

st.subheader("Monthly Expenses by Category (Stacked)")
st.plotly_chart(fig, use_container_width=True)

# 3. Last 90 Days with Trendline
st.header("Trendline")
st.subheader("Expenses: Last 90 Days (with Trendline)")
last_90 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=90))]
daily = last_90.groupby('date')['amount'].sum().reset_index()

# Calculate moving average as trendline
window = 7  # 7-day moving average
if len(daily) >= window:
    daily['trend'] = daily['amount'].rolling(window=window, min_periods=1).mean()
else:
    daily['trend'] = daily['amount']

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=daily['date'], y=daily['amount'], name='Daily Expenses', marker_color='lightblue'
))
fig_trend.add_trace(go.Scatter(
    x=daily['date'], y=daily['trend'], name=f'{window}-Day Moving Avg', line=dict(color='red', width=2)
))
fig_trend.update_layout(
    title="Last 90 Days Expenses with Trendline",
    xaxis_title="Date",
    yaxis_title="Total Expenses",
    legend_title="Legend",
    height=400,
    margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig_trend, use_container_width=True)

# 4. Forecast Diagnostics Table
st.header("Diagnostics")
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rolling_backtest(ts, methods, window=60, forecast_horizon=7):
    # ts: pandas Series with date index
    # methods: dict of {name: function}
    # Returns: {method: error}
    errors = {name: [] for name in methods}
    for i in range(window, len(ts) - forecast_horizon + 1):
        train = ts.iloc[i-window:i]
        test = ts.iloc[i:i+forecast_horizon]
        for name, func in methods.items():
            try:
                pred = func(train.values, forecast_horizon)
                err = mean_absolute_error(test.values, pred)
                errors[name].append(err)
            except Exception:
                errors[name].append(np.nan)
    avg_errors = {name: np.nanmean(errs) for name, errs in errors.items()}
    return avg_errors

def tracking_signals(y_true, y_pred):
    # Advanced tracking signals
    # 1. MAE
    mae = mean_absolute_error(y_true, y_pred)
    # 2. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-8))) * 100
    # 3. Bias (mean forecast error)
    bias = np.mean(np.array(y_pred) - np.array(y_true))
    # 4. Tracking Signal (Cumulative Forecast Error / MAE)
    cfe = np.sum(np.array(y_pred) - np.array(y_true))
    tsignal = cfe / (mae + 1e-8)
    return {'MAE': mae, 'MAPE': mape, 'Bias': bias, 'Tracking Signal': tsignal}

def subjective_score(mape, tsignal):
    # Subjective scoring based on MAPE and tracking signal
    if mape < 15 and abs(tsignal) < 2:
        return 'Good', 'Forecast is accurate and unbiased.'
    elif mape < 30 and abs(tsignal) < 4:
        return 'Fair', 'Forecast is somewhat accurate, but may have some bias or volatility.'
    else:
        return 'Poor', 'Forecast is unreliable or highly biased.'

# --- Streamlit sidebar: adjustable activity window and forecast horizon ---
st.sidebar.header("Forecast Settings")
activity_window = st.sidebar.slider("Recent activity window (days)", min_value=7, max_value=90, value=70, step=1)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=7, step=1)

results = []
forecast_export_rows = []
category_forecasts = []
for cat in df['category'].unique():
    df_cat = df[df['category'] == cat].copy()
    last_90_start = df['date'].max() - pd.Timedelta(days=90)
    df_cat = df_cat[df_cat['date'] >= last_90_start]
    # --- Skip if no data for this category ---
    if df_cat.empty or pd.isna(df_cat['date'].min()) or pd.isna(df_cat['date'].max()):
        st.warning(f"No data for category '{cat}' in the last 90 days. Skipping forecast.")
        continue
    # --- Only forecast if category is active in user-selected window ---
    last_window_start = df['date'].max() - pd.Timedelta(days=activity_window)
    recent_nonzero = (df_cat[df_cat['date'] >= last_window_start]['amount'] > 0).sum()
    forecast_active = recent_nonzero > 0
    ts = df_cat.groupby('date')['amount'].sum().reindex(pd.date_range(df_cat['date'].min(), df['date'].max()), fill_value=0)
    # --- Define forecasting methods ---
    def mean_method(ts, fh):
        # Enhanced forecast with better weekly seasonality and smoothing
        import pandas as pd
        ts = pd.Series(ts)
        
        # Use last 90 days for more stable estimates
        lookback_days = 90
        if len(ts) < lookback_days:
            lookback_days = len(ts)
        
        # Get last N days of data
        recent_ts = ts[-lookback_days:]
        
        # Calculate 7-day moving average to smooth the data
        ma7 = recent_ts.rolling(window=7, min_periods=1).mean()
        
        # Calculate base level using moving average
        base_level = ma7.iloc[-7:].mean()  # Average of last week's moving average
        
        # Calculate day-of-week seasonality factors
        weekday_avg = recent_ts.groupby(recent_ts.index.dayofweek).mean()
        weekday_factor = weekday_avg / weekday_avg.mean()
        
        # Calculate day-of-month factors (less weight than weekly)
        dom_avg = recent_ts.groupby(recent_ts.index.day).mean()
        dom_factor = dom_avg / dom_avg.mean()
        
        # Calculate recent trend (slope of last 7 days)
        if len(ma7) >= 7:
            x = np.arange(7)
            y = ma7.iloc[-7:].values
            z = np.polyfit(x, y, 1)
            trend_slope = z[0]  # Daily change
        else:
            trend_slope = 0
        
        # Generate forecast dates
        forecast_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=fh)
        
        # Generate forecast with seasonality and trend
        forecast = []
        for i, d in enumerate(forecast_dates):
            # Get day of week factor (0=Monday, 6=Sunday)
            wfac = weekday_factor.get(d.dayofweek, 1.0)
            # Get day of month factor (reduced weight)
            dfac = 0.7 + 0.3 * dom_factor.get(d.day, 1.0)  # Dampen the effect
            # Apply trend (reduced impact)
            trend_effect = 1.0 + (i * trend_slope * 0.3 / base_level) if base_level > 0 else 1.0
            # Combine all factors
            forecast_value = base_level * wfac * dfac * trend_effect
            forecast.append(max(0, forecast_value))  # Ensure non-negative
            
        # Add some random variation (5% of base level)
        if base_level > 0:
            noise = np.random.normal(0, 0.05 * base_level, fh)
            forecast = np.array(forecast) + noise
            forecast = np.maximum(forecast, 0)  # Ensure no negative values
            
        return forecast

    def median_method(ts, fh): return np.repeat(np.median(ts[-30:]), fh)
    def zero_method(ts, fh): return np.zeros(fh)
    def croston_method(ts, fh): return croston(ts, fh)
    def prophet_method(ts, fh):
        import pandas as pd
        from prophet import Prophet
        dfp = pd.DataFrame({'ds': ts.index, 'y': ts.values})
        if len(dfp.dropna()) < 2:
            return np.zeros(fh)
        m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
        m.fit(dfp)
        future = pd.DataFrame({'ds': pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=fh)})
        forecast = m.predict(future)
        return forecast['yhat'].clip(lower=0).values
    def periodic_spike_method(ts, fh, category=None):
        # Robust spike forecast for periodic categories, tolerant of small delays
        import numpy as np
        import pandas as pd
        from pandas import Series
        ts = Series(ts)
        spike_thresholds = {
            'school': 300,
            'rent + communal': 500,
            'car rent': 300
        }
        cat_lower = (category or '').lower() if category else ''
        threshold = 0
        for key, val in spike_thresholds.items():
            if key in cat_lower:
                threshold = val
                break
        if threshold:
            nonzero = ts[ts > threshold]
        else:
            nonzero = ts[ts > 0]
        if len(nonzero) < 2:
            return np.zeros(fh)
        spike_idxs = nonzero.index
        if not isinstance(spike_idxs[0], (pd.Timestamp, pd.DatetimeIndex, pd.Period)):
            intervals = np.diff(spike_idxs)
            if len(intervals) == 0:
                return np.zeros(fh)
            median_interval = int(np.median(intervals))
            last_spike_idx = spike_idxs[-1]
            spike_value = nonzero.iloc[-1]
            forecast = np.zeros(fh)
            next_spike = last_spike_idx + median_interval
            # Place spike at first day if next_spike is just before window (tolerance 3 days)
            if next_spike - len(ts) < 0 and abs(next_spike - len(ts)) <= 3:
                forecast[0] = spike_value
            elif 0 <= next_spike - len(ts) < fh:
                forecast[next_spike - len(ts)] = spike_value
            return forecast
        # Datetime index logic
        spike_dates = pd.to_datetime(spike_idxs)
        intervals = (spike_dates[1:] - spike_dates[:-1]).days
        if len(intervals) == 0:
            return np.zeros(fh)
        median_interval = int(np.median(intervals))
        last_spike_date = spike_dates[-1]
        spike_value = nonzero.iloc[-1]
        forecast = np.zeros(fh)
        forecast_start_date = ts.index[-1] + pd.Timedelta(days=1)
        next_spike_date = last_spike_date + pd.Timedelta(days=median_interval)
        offset = (next_spike_date - forecast_start_date).days
        # Place spike at first day if next_spike_date is just before window (tolerance 3 days)
        if offset < 0 and abs(offset) <= 3:
            forecast[0] = spike_value
        elif 0 <= offset < fh:
            forecast[offset] = spike_value
        return forecast

    methods = {
        'mean': mean_method,
        'median': median_method,
        'zero': zero_method,
        'croston': croston_method,
        'prophet': prophet_method,
        'periodic_spike': lambda ts, fh: periodic_spike_method(ts, fh, category=cat)
    }
    # --- Backtest and select best method ---
    avg_errors = rolling_backtest(ts, methods, window=activity_window, forecast_horizon=forecast_horizon)
    nonzero_days = (ts > 0).sum()
    # --- Special rule: if category is school, rent + communal, or car rent, force periodic_spike ---
    spike_cats = ['school', 'rent + communal', 'car rent']
    if any(key in cat.lower() for key in spike_cats):
        best_method = 'periodic_spike'
    elif nonzero_days <= 2:
        best_method = 'periodic_spike'
    else:
        best_method = min(avg_errors, key=avg_errors.get)
    # --- Forecast with best method only if active ---
    if forecast_active:
        forecast_vals = methods[best_method](ts, forecast_horizon)
        # --- Tracking signals for best method ---
        # Use last 60 days for backtest tracking signals
        if len(ts) > (forecast_horizon + 60):
            y_true = ts[-forecast_horizon:]
            y_pred = methods[best_method](ts[:-forecast_horizon], forecast_horizon)
        else:
            y_true = ts[-forecast_horizon:]
            y_pred = forecast_vals
        # Ensure y_true and y_pred are the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]
        signals = tracking_signals(y_true, y_pred)
        score, explanation = subjective_score(signals['MAPE'], signals['Tracking Signal'])
        for i in range(forecast_horizon):
            forecast_date = ts.index[-1] + pd.Timedelta(days=i+1)
            results.append({
                'category': cat,
                'date': forecast_date,
                'forecast': forecast_vals[i]
            })
            forecast_export_rows.append({
                'category': cat,
                'date': forecast_date,
                'forecast': forecast_vals[i]
            })
    else:
        signals = {'MAE': None, 'MAPE': None, 'Bias': None, 'Tracking Signal': None}
        score, explanation = 'Inactive', 'No recent activity. Forecast skipped.'
    category_forecasts.append({
        'category': cat,
        'Best Method': best_method,
        'MAE': avg_errors[best_method],
        'All MAEs': avg_errors,
        'Active for Forecast': forecast_active,
        'Activity Window': activity_window,
        'MAPE': signals['MAPE'],
        'Bias': signals['Bias'],
        'Tracking Signal': signals['Tracking Signal'],
        'Score': score,
        'Explanation': explanation
    })

st.subheader("Forecast Diagnostics Table")
st.caption("Forecast performance metrics for each category")
metric_tooltips = {
    'Best Method': 'The forecasting method with the lowest average error for this category.',
    'MAE': 'Mean Absolute Error: Average absolute difference between forecast and actual values. Lower is better.',
    'All MAEs': 'MAE for each method. Used to select the best method.',
    'Active for Forecast': 'Whether this category is included in the forecast.',
    'Activity Window': 'Number of days of recent data used for metrics and forecasting.',
    'MAPE': 'Mean Absolute Percentage Error: Average percent error between forecast and actual. Sensitive to small actuals.',
    'Bias': 'Average signed error. Positive = overestimate, Negative = underestimate.',
    'Tracking Signal': 'Indicates if forecast errors are consistently biased. Large values mean persistent over/underforecast.',
    'Score': 'Subjective rating of forecast reliability (Good/Poor/Inactive).',
    'Explanation': 'Short explanation of the forecast score and reliability.'
}
# Show tooltips above the diagnostics table
with st.expander("ℹ️ Metric Explanations (click to expand)"):
    for k, v in metric_tooltips.items():
        st.markdown(f"**{k}:** {v}")
df_diag = pd.DataFrame(category_forecasts)
st.dataframe(df_diag)

# --- Forecast Metrics Heatmap ---
st.header("Forecast Metrics Heatmap")
st.caption("Explore how forecast error (MAE) varies with activity window and forecast horizon. Lower is better.")

# Debug: Show basic data info
st.write("### Data Overview")
st.write(f"Date range in data: {df['date'].min().date()} to {df['date'].max().date()}")
st.write(f"Number of categories: {len(df['category'].unique())}")
st.write(f"Total transactions: {len(df)}")

activity_windows = list(range(14, 91, 14))  # Reduced granularity for faster computation
forecast_horizons = list(range(7, 31, 7))
heatmap_data = []

# Get the most recent date in the data
latest_date = df['date'].max()

# Debug: Show latest date
st.write(f"Latest date in data: {latest_date.date()}")

# Pre-process category data to avoid redundant computations
category_data = {}
categories_with_data = []

for cat in df['category'].unique():
    df_cat = df[df['category'] == cat].copy()
    # Get last 90 days of data for each category
    last_90_start = latest_date - pd.Timedelta(days=90)
    df_cat = df_cat[df_cat['date'] >= last_90_start]
    
    if not df_cat.empty and not (pd.isna(df_cat['date'].min()) or pd.isna(df_cat['date'].max())):
        # Create time series with proper datetime index
        ts = df_cat.groupby('date')['amount'].sum()
        # Ensure we have a complete date range
        date_range = pd.date_range(ts.index.min(), ts.index.max())
        ts = ts.reindex(date_range, fill_value=0)
        
        # Debug: Show category data info
        if len(ts) > 0:
            categories_with_data.append(cat)
            st.write(f"\n**{cat}** - Data points: {len(ts)}, Date range: {ts.index.min().date()} to {ts.index.max().date()}, Non-zero days: {(ts > 0).sum()}")
        
        category_data[cat] = ts

st.write(f"\nTotal categories with valid data: {len(categories_with_data)}")

if not category_data:
    st.error("No valid category data found for heatmap generation.")
    st.stop()

# Generate heatmap data
with st.spinner('Generating heatmap data...'):
    debug_info = []
    
    # Use a single method for simplicity - we'll focus on mean method for now
    def simple_forecast(ts, fh):
        # Simple mean forecast with weekly seasonality
        ts = pd.Series(ts)
        if len(ts) < 7:
            return np.repeat(ts.mean(), fh)
        
        # Calculate weekly seasonality
        weekly_avg = ts.groupby(ts.index.dayofweek).mean()
        if len(weekly_avg) < 7:
            return np.repeat(ts.mean(), fh)
            
        # Forecast next fh days
        forecast = []
        for i in range(1, fh + 1):
            day_of_week = (ts.index[-1].dayofweek + i) % 7
            forecast.append(weekly_avg[day_of_week])
            
        return np.array(forecast)
    
    # Simplified backtest function
    def simple_backtest(ts, window=30, forecast_horizon=7):
        errors = []
        for i in range(window, len(ts) - forecast_horizon, forecast_horizon):
            train = ts.iloc[i-window:i]
            actual = ts.iloc[i:i+forecast_horizon].values
            
            try:
                pred = simple_forecast(train, forecast_horizon)
                if len(pred) == len(actual):
                    mae = np.mean(np.abs(pred - actual))
                    if np.isfinite(mae):
                        errors.append(mae)
            except Exception as e:
                continue
                
        return np.mean(errors) if errors else np.nan
    
    # Generate heatmap data
    for aw in activity_windows:
        row = []
        st.write(f"\nProcessing activity window: {aw} days")
        
        for fh in forecast_horizons:
            maes = []
            
            for cat, ts in category_data.items():
                if len(ts) < (aw + fh):
                    continue
                    
                try:
                    mae = simple_backtest(ts, window=aw, forecast_horizon=fh)
                    if mae is not None and np.isfinite(mae):
                        maes.append(mae)
                except Exception as e:
                    st.warning(f"Error processing {cat}: {str(e)}")
                    continue
            
            # Calculate average MAE for this cell
            avg_mae = np.nanmean(maes) if maes else np.nan
            row.append(avg_mae)
            
            debug_info.append({
                'Activity Window': aw,
                'Forecast Horizon': fh,
                'Categories Processed': len(maes),
                'Average MAE': f"{avg_mae:.2f}" if not np.isnan(avg_mae) else "N/A"
            })
        
        heatmap_data.append(row)
    
    # Show debug info in an expander
    with st.expander("Debug Information"):
        st.write("### Heatmap Generation Details")
        if debug_info:
            st.dataframe(pd.DataFrame(debug_info))
        else:
            st.warning("No debug information was generated. Check if there's enough data.")
import pandas as pd
heatmap_df = pd.DataFrame(heatmap_data, index=activity_windows, columns=forecast_horizons)
fig_hm = px.imshow(
    heatmap_df.values,
    x=forecast_horizons,
    y=activity_windows,
    labels=dict(x="Forecast Horizon (days)", y="Activity Window (days)", color="Avg MAE"),
    color_continuous_scale='Viridis',
    aspect="auto"
)
fig_hm.update_layout(title="Forecast MAE by Activity Window and Forecast Horizon", height=400)
st.plotly_chart(fig_hm, use_container_width=True)

# Recommend best parameter (lowest MAE)
min_mae = np.nanmin(heatmap_df.values)
if not np.isnan(min_mae):
    best_idx = np.unravel_index(np.nanargmin(heatmap_df.values), heatmap_df.shape)
    best_aw = activity_windows[best_idx[0]]
    best_fh = forecast_horizons[best_idx[1]]
    st.success(f"Recommended: Activity Window = {best_aw} days, Forecast Horizon = {best_fh} days (lowest MAE: {min_mae:.2f})")

# 5. Forecast Table & Summary
st.header("Forecast")
st.subheader(f"{forecast_horizon}-Day Forecast by Category")
forecast_df = pd.DataFrame(results)
st.caption(f"Forecasted expenses for the next {forecast_horizon} days, by category")
st.dataframe(forecast_df)
st.subheader(f"Total Forecast Summary (All Categories)")
total_forecast = forecast_df.groupby('date')['forecast'].sum().reset_index()
total_sum = total_forecast['forecast'].sum()
st.caption(f"Total forecasted expenses for the next {forecast_horizon} days")
st.dataframe(total_forecast.rename(columns={"forecast": "Total Forecast"}))
st.metric(label=f"Total Forecasted Expenses ({forecast_horizon} days)", value=f"{total_sum:.2f}")

# 6. Stacked Column Chart: History + Forecast
st.header("Stacked Chart")
st.subheader(f"Stacked Column Chart: 30 Days History + {forecast_horizon} Days Forecast")
# Prepare historical data (last 30 days)
last_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
hist_pivot = last_30.groupby(['date', 'category'])['amount'].sum().reset_index()
hist_pivot = hist_pivot.pivot(index='date', columns='category', values='amount').fillna(0)
# Combine history and forecast
combined = pd.concat([hist_pivot, forecast_df.pivot(index='date', columns='category', values='forecast').fillna(0)], axis=0)
# Plot
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

# 7. Forecast for Each Category
st.header("Per-Category Forecasts")
for cat in forecast_df['category'].unique():
    df_plot = forecast_df[forecast_df['category'] == cat]
    df_cat_hist = df[(df['category'] == cat) & (df['date'] >= (df['date'].max() - pd.Timedelta(days=30)))]
    df_cat_hist = df_cat_hist.groupby('date')['amount'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_cat_hist['date'], y=df_cat_hist['amount'], name='Actual (last 30d)', marker_color='gray', opacity=0.5))
    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['forecast'], mode='lines+markers', name='Forecast', line=dict(color='royalblue')))
    fig.update_layout(title=f"Forecast for {cat} (with last 30d history)", xaxis_title="Date", yaxis_title="Amount", height=350)
    st.plotly_chart(fig, use_container_width=True)

# --- Moving average baseline for comparison ---
st.write("#### Moving Average Baseline (last 30 days)")
hist_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
ma_baseline = hist_30.groupby('date')['amount'].sum().rolling(window=7, min_periods=1).mean()
st.line_chart(ma_baseline, use_container_width=True)
###############################################################################################
# --- Monthly Budget Tracking ---
st.write("#### Monthly Budget Tracking")

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
    
    # Color code variance
    def color_variance(val):
        if pd.isna(val) or val == '':
            return ''
        # Convert string with $ and commas to float if needed
        if isinstance(val, str):
            try:
                val = float(val.replace('$', '').replace(',', ''))
            except (ValueError, AttributeError):
                return ''
        if val > 0:
            return 'color: red'
        return 'color: green'
    
    variance_styled = display_df.style.applymap(
        lambda x: color_variance(float(x.replace('$', '').replace(',', '')) if isinstance(x, str) and x != '' else None),
        subset=['Variance']
    )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No data available for the current month.")
###############################################################################################################
# --- Warning if Prophet forecast is much higher than moving average ---
if not forecast_df.empty:
    forecast_total = forecast_df.groupby('date')['forecast'].sum().mean()
    ma_total = ma_baseline.mean()
    if forecast_total > 1.5 * ma_total:
        st.warning(f"Forecasted daily expenses are much higher than recent 7-day moving average. Please review model results.")

st.info("Forecasts use a hybrid robust approach: Prophet is used if enough data and its forecast is not extreme; otherwise, mean, median, or zero is used. 95% confidence intervals are not shown. Model auto-updates with new data.")

# --- Backward Forecast Performance ---
st.header("Backward Forecast Performance")
st.caption("Analyze forecast accuracy over time by comparing past forecasts with actuals")

# Calculate backward forecast performance
def calculate_backward_accuracy(df, forecast_horizon=7, min_data_points=30):
    """
    Calculate forecast accuracy by looking back at past forecasts and comparing with actuals.
    
    Args:
        df: DataFrame with transaction data
        forecast_horizon: Number of days ahead to forecast
        min_data_points: Minimum number of data points required for analysis
        
    Returns:
        DataFrame with date, actual, forecast, and error metrics
    """
    try:
        # Ensure we have a copy of the dataframe
        df = df.copy()
        
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Get unique dates with actual data
        actual_dates = df['date'].sort_values().unique()
        
        if len(actual_dates) < min_data_points + forecast_horizon:
            st.warning(f"Not enough data points for analysis. Need at least {min_data_points + forecast_horizon} days of data. Have {len(actual_dates)} days.")
            return pd.DataFrame()
        
        results = []
        
        # For each date with actual data, look back to find the forecast made 'forecast_horizon' days ago
        for i in range(forecast_horizon, len(actual_dates)):
            current_date = actual_dates[i]
            forecast_date = actual_dates[i - forecast_horizon]
            
            # Get actual values for current date (sum across all categories)
            actual_data = df[df['date'] == current_date]
            if len(actual_data) == 0:
                continue
                
            actual = actual_data['amount'].sum()
            
            # Get forecast made on forecast_date for current_date
            forecast_df = df[df['date'] <= forecast_date].copy()
            if len(forecast_df) == 0:
                continue
                
            forecast_values = []
            
            # Generate forecast for each category
            for cat in forecast_df['category'].unique():
                df_cat = forecast_df[forecast_df['category'] == cat].copy()
                if len(df_cat) < 14:  # Skip if not enough data
                    continue
                    
                # Create time series with proper datetime index
                ts = df_cat.groupby('date')['amount'].sum()
                
                # Skip if forecast date is not in the index
                if forecast_date not in ts.index:
                    continue
                    
                try:
                    # Get the position of the forecast date
                    forecast_date_idx = ts.index.get_loc(forecast_date)
                    
                    # Get last 30 days of data up to forecast date (or as much as available)
                    lookback = min(30, forecast_date_idx + 1)
                    forecast_ts = ts.iloc[forecast_date_idx - lookback + 1:forecast_date_idx + 1]
                    
                    # Simple forecast: average of same day of week
                    day_of_week = pd.Timestamp(current_date).dayofweek
                    same_day = forecast_ts[forecast_ts.index.dayofweek == day_of_week]
                    forecast = same_day.mean() if not same_day.empty else 0
                    forecast_values.append(forecast)
                    
                except (KeyError, ValueError) as e:
                    # Skip this category if there's an error with date handling
                    continue
            
            # Sum forecasts across all categories
            total_forecast = sum(forecast_values) if forecast_values else 0
            
            # Calculate error metrics
            error = total_forecast - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual * 100) if actual > 0 else 0
            
            results.append({
                'date': current_date,
                'forecast_date': forecast_date,
                'forecast_horizon': forecast_horizon,
                'actual': actual,
                'forecast': total_forecast,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error,
                'within_10pct': 1 if pct_error <= 10 else 0,
                'within_20pct': 1 if pct_error <= 20 else 0
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Error in backward forecast calculation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()

# Add UI controls for backward forecast analysis
st.write("### Backward Forecast Performance Analysis")

# Create columns for the controls
col1, col2, col3 = st.columns(3)
with col1:
    max_horizon = min(30, len(df['date'].unique()) - 15)  # Ensure enough data
    horizon = st.slider("Forecast Horizon (days)", 1, max(1, max_horizon), 7, 1,
                       help="Number of days ahead to evaluate forecast accuracy")

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

# Filter data based on date range and categories
analysis_df = df[
    (df['date'].dt.date >= start_date) & 
    (df['date'].dt.date <= end_date)
].copy()

if selected_categories:
    analysis_df = analysis_df[analysis_df['category'].isin(selected_categories)]
    st.info(f"Showing results for categories: {', '.join(selected_categories)} from {start_date} to {end_date}")
else:
    st.info(f"Showing results for all categories from {start_date} to {end_date}")

# Calculate backward forecast performance
with st.spinner("Calculating forecast accuracy..."):
    backward_results = calculate_backward_accuracy(analysis_df, forecast_horizon=horizon)

if not backward_results.empty:
    # Calculate summary metrics
    avg_mae = backward_results['abs_error'].mean()
    avg_mape = backward_results['pct_error'].mean()
    accuracy_10pct = backward_results['within_10pct'].mean() * 100
    accuracy_20pct = backward_results['within_20pct'].mean() * 100
    
    # Show category information if filtered
    if selected_categories:
        st.subheader(f'Analysis for {len(selected_categories)} Selected Categories')
    
    # Display summary metrics
    st.metric("Mean Absolute Error (MAE)", f"${avg_mae:,.2f}")
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{avg_mape:.1f}%")
    st.metric(f"Accuracy (±10%)", f"{accuracy_10pct:.1f}%")
    st.metric(f"Accuracy (±20%)", f"{accuracy_20pct:.1f}%")
    
    # Plot actual vs forecast over time
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=backward_results['date'],
        y=backward_results['actual'],
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # Add forecasted values
    fig.add_trace(go.Scatter(
        x=backward_results['date'],
        y=backward_results['forecast'],
        name=f'{horizon}-Day Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Add error bands
    fig.add_trace(go.Scatter(
        x=backward_results['date'].tolist() + backward_results['date'].tolist()[::-1],
        y=(backward_results['forecast'] * 1.2).tolist() + (backward_results['forecast'] * 0.8).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.1)',
        line=dict(width=0),
        showlegend=True,
        name='±20% Forecast Range',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Actual vs {horizon}-Day Forecast for {len(selected_categories) or "All"} Categories',
        xaxis_title='Date',
        yaxis_title='Amount',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot error over time with more details
    fig_error = go.Figure()
    
    # Add error line
    fig_error.add_trace(go.Scatter(
        x=backward_results['date'],
        y=backward_results['pct_error'],
        name='Error %',
        line=dict(color='#d62728', width=2),
        mode='lines+markers',
        marker=dict(size=6),
        customdata=backward_results[['forecast', 'actual']],
        hovertemplate=(
            '<b>Date</b>: %{x|%Y-%m-%d}<br>'
            '<b>Error</b>: %{y:.1f}%<br>'
            '<b>Forecast</b>: $%{customdata[0]:.2f}<br>'
            '<b>Actual</b>: $%{customdata[1]:.2f}<br>'
            '<extra></extra>'
        )
    ))
    
    # Add zero line
    fig_error.add_hline(
        y=0, line_dash="dash",
        annotation_text="Perfect Forecast",
        annotation_position="bottom right",
        line_width=1
    )
    
    # Add ±10% bands
    fig_error.add_hrect(
        y0=-10, y1=10,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="±10% Error Band",
        annotation_position="top left"
    )
    
    fig_error.update_layout(
        title=f'Forecast Error Over Time ({horizon}-Day Horizon)',
        xaxis_title='Date',
        yaxis_title='Error %',
        hovermode='x',
        height=400,
        showlegend=True
    )
    
    # Add metrics as annotations
    annotations = [
        dict(
            x=0.02,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f"MAE: ${avg_mae:,.2f}<br>MAPE: {avg_mape:.1f}%",
            showarrow=False,
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            borderpad=4
        )
    ]
    
    fig_error.update_layout(annotations=annotations)
    st.plotly_chart(fig_error, use_container_width=True)
    
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
        
        # Add explanation
        st.markdown("""
        **How to interpret this chart:**
        - The chart shows 7-day moving averages of both actual and forecasted values
        - This smooths out daily fluctuations to better show trends
        - The forecast line shows what was predicted (on average) for each day, looking back from the forecast date
        - The closer the two lines are, the better the forecast captures the underlying trend
        - The MAE and MAPE values show the average error in the moving averages
        """)

# --- Export forecast to Excel ---
if not forecast_df.empty:
    export_df = pd.DataFrame(forecast_export_rows)
    export_path = os.path.join(os.path.dirname(__file__), 'forecast_results.xlsx')
    with ExcelWriter(export_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False)
    st.success(f"Forecast results exported to {export_path}")
