import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from pandas import ExcelWriter
import subprocess
from croston import croston
from anomaly_utils import detect_outliers, detect_anomaly_transactions, recurring_payments, get_comprehensive_anomalies, create_anomaly_visualization, create_anomaly_summary_table
from forecast_metrics import ForecastMetrics, calculate_forecast_metrics
from forecast_utils import aggregate_daily_to_weekly, calculate_weekly_metrics, add_week_metadata, aggregate_daily_to_monthly
from config import config

# Import modular UI components
from ui.components import (
    create_sidebar_controls, create_category_filter, create_section_header,
    create_info_box, display_dataframe_with_controls, create_expandable_help
)
from ui.charts import (
    create_forecast_chart, create_seasonality_heatmap, create_category_comparison_chart
)
from ui.dashboard_sections import (
    render_data_overview_section, render_anomaly_detection_section,
    render_recurring_payments_section, render_seasonality_section,
    render_forecast_performance_section, render_forecast_charts_section,
    render_expense_breakdown_section, render_comparisons_section,
    render_export_section, render_user_guidance_section,
    render_monthly_expenses_section, render_trendline_section,
    render_per_category_forecasts_section, render_stacked_chart_section,
    render_backward_forecast_section
)



def select_method(ts, methods, cat, window, fh):
    """
    Select best forecasting method based on backtest errors and simple rules.
    Args:
        ts: time series (pd.Series)
        methods: dict of method_name->function
        cat: category name
        window: backtest window (days)
        fh: forecast horizon (days)
    Returns:
        method_name (str)
    """
    # Category-specific spike override or low activity
    nonzero_days = (ts > 0).sum()
    spike_cats = ['school', 'rent + communal', 'car rent']
    if any(key in cat.lower() for key in spike_cats) or nonzero_days <= 2:
        return 'periodic_spike'
    # Compute backtest MAEs
    avg_errors = rolling_backtest(ts, methods, window, fh)
    # Choose method with lowest MAE
    return min(avg_errors, key=avg_errors.get)

def add_period_column(df):
    # Define date ranges for each period
    operations_start = datetime(2023, 3, 31)
    overstock_start = datetime(2023, 12, 31)
    overstock_end = datetime(2024, 5, 1)
    regular_start = datetime(2024, 5, 1)
    
    def get_period(date):
        if date <= operations_start:
            return 'Operations'
        elif overstock_start <= date <= overstock_end:
            return 'Overstock'
        elif date >= regular_start:
            return 'Regular'
        else:
            return 'Unknown'
    
    df['period'] = df['date'].apply(get_period)
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
    df = pd.read_excel(config.get_expenses_path())

df['date'] = pd.to_datetime(df['date'])
df = add_period_column(df)

# --- Category Filtering ---
selected_categories = create_category_filter(df, 
    default_selection=st.session_state.get('selected_categories', sorted(df['category'].unique()))
)
st.session_state['selected_categories'] = selected_categories
if selected_categories:
    df = df[df['category'].isin(selected_categories)]
else:
    st.warning("No categories selected. Showing empty dashboard.")

# --- Enhanced Anomaly & Outlier Detection ---
render_anomaly_detection_section(df)

# --- Recurring Payments Detection ---
render_recurring_payments_section(df)

# --- Expense Breakdown ---
# --- Expense Breakdown & Cumulative Totals ---
render_expense_breakdown_section(df)

# --- Comparisons & Benchmarks ---
render_comparisons_section(df)

# --- Export & Sharing ---
render_export_section(df, config)

# --- User Guidance & Explanations ---
render_user_guidance_section()

# --- Dashboard Layout: Group Boards in Logical Blocks ---
# 1. Raw Data Preview
render_data_overview_section(df)

# --- Seasonality Analysis ---
render_seasonality_section(df)

# 2. Monthly Summary & Period Regimes
# --- Monthly Expenses ---
render_monthly_expenses_section(df)

# 3. Last 90 Days with Trendline
# --- Trendline Analysis ---
render_trendline_section(df)

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
                pred = func(train, forecast_horizon)
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

# --- Sidebar Controls ---
controls = create_sidebar_controls(df)
activity_window = controls['activity_window']
forecast_horizon = controls['forecast_horizon']
spike_threshold = controls['spike_threshold']
# Diagnostics date range selector
diag_dates = st.sidebar.date_input("Diagnostics date range", [df['date'].min().date(), df['date'].max().date()], min_value=df['date'].min().date(), max_value=df['date'].max().date())
if isinstance(diag_dates, (list, tuple)) and len(diag_dates) == 2:
    diag_start, diag_end = diag_dates
else:
    diag_start = diag_dates
    diag_end = diag_dates
# Note: spike_threshold is now handled by modular controls above

# Forecast Method Selection
method_names = ["mean", "median", "zero", "croston", "prophet", "periodic_spike"]
st.sidebar.header("Forecast Method Selection")
method_selection_mode = st.sidebar.radio("Forecast Method Selection Mode", ["Automatic", "Manual"], index=0)
if method_selection_mode == "Manual":
    manual_method = st.sidebar.selectbox("Choose Forecast Method", method_names)

# Add button to update expenses from daily payments
st.sidebar.header("Data Management")
if st.sidebar.button("🔄 Update Expenses from Daily Payments"):
    try:
        import subprocess
        result = subprocess.run(['python', 'update_expenses_from_daily.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("✅ Expenses updated successfully!")
            st.sidebar.info("Please refresh the page to see the updated data.")
        else:
            st.sidebar.error(f"❌ Error updating expenses:\n{result.stderr}")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to run update script: {str(e)}")

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
    avg_errors = rolling_backtest(ts, methods, activity_window, forecast_horizon)
    # Determine best method based on selection mode
    if method_selection_mode == "Automatic":
        best_method = select_method(ts, methods, cat, activity_window, forecast_horizon)
    else:
        best_method = manual_method
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
# Expand All MAEs into separate columns
mae_df = pd.json_normalize(df_diag['All MAEs'])
mae_df.columns = [f"MAE_{col}" for col in mae_df.columns]
df_diag = pd.concat([df_diag.drop(columns=['All MAEs']), mae_df], axis=1)
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

# --- Stacked Chart ---
render_stacked_chart_section(df, forecast_df, forecast_horizon)

# --- Per-Category Forecasts ---
render_per_category_forecasts_section(df, forecast_df)

# Original inline Per-Category Forecasts section has been replaced with modular component above
# The ~220 lines of inline code have been moved to render_per_category_forecasts_section()
df_plot = forecast_df[forecast_df['category'] == cat]
# Large inline section (~220 lines) removed - functionality moved to modular component

# ~~~ REMOVED: Large inline section (211+ lines) moved to render_per_category_forecasts_section() ~~~
# This included: Moving average baseline, Monthly budget tracking, Charts, and complex logic

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
if not forecast_df.empty and not forecast_df.empty:
    # Initialize variables with default values
    forecast_total = 0
    ma_total = 0
    
    try:
        # Calculate forecast total
        forecast_total = forecast_df.groupby('date')['forecast'].sum().mean()
        
        # Calculate 7-day moving average of historical data for comparison
        hist_30 = df[df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
        if not hist_30.empty:
            daily_totals = hist_30.groupby('date')['amount'].sum()
            if not daily_totals.empty:
                ma_baseline = daily_totals.rolling(window=7, min_periods=1).mean()
                ma_total = ma_baseline.mean()
    except Exception as e:
        st.warning(f"Could not calculate moving average baseline: {str(e)}")
    
    # Only show warning if we have valid values
    if forecast_total > 0 and ma_total > 0 and forecast_total > 1.5 * ma_total:
        st.warning(f"Forecasted daily expenses are much higher than recent 7-day moving average. Please review model results.")

    st.info("Forecasts use a hybrid robust approach: Prophet is used if enough data and its forecast is not extreme; otherwise, mean, median, or zero is used. 95% confidence intervals are not shown. Model auto-updates with new data.")
    # End of disabled inline section

# --- Backward Forecast Performance ---
st.header("Backward Forecast Performance")
st.caption("Analyze forecast accuracy over time by comparing past forecasts with actuals")

# Calculate backward forecast performance
def calculate_backward_accuracy(df, forecast_horizon=7, min_data_points=30, timeframe='daily', activity_window=60):
    """
    Calculate forecast accuracy by looking back at past forecasts and comparing with actuals.
    
    Args:
        df: DataFrame with transaction data with columns: date, amount, category
        forecast_horizon: Number of days ahead to forecast
        activity_window: Number of recent days used as training window for generating each historical forecast
        min_data_points: Minimum number of data points required for daily analysis
        df: DataFrame with transaction data with columns: date, amount, category
        forecast_horizon: Number of days ahead to forecast
        min_data_points: Minimum number of data points required for daily analysis
        timeframe: 'daily' or 'weekly' - whether to return daily or weekly aggregated results
        
    Returns:
        DataFrame with date, actual, forecast, and error metrics. For weekly timeframe,
        the date will be the end of the week (Sunday).
    """
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
        
        # (Initial-period logic removed) Forecasts will be computed uniformly for all dates with a full look-back horizon
        for i in range(0):  # disabled initial-period loop
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
                # Get all data for this category up to current date
                cat_data = df[(df['category'] == cat) & (df['date'] <= current_date)].copy()
                if len(cat_data) < 1:  # Need at least one data point
                    continue
                    
                # Create time series with proper datetime index
                ts = cat_data.groupby('date')['amount'].sum()
                
                try:
                    # Get the position of the current date
                    current_date_idx = ts.index.get_loc(current_date)
                    
                    # Use all available data up to current date
                    lookback = min(30, current_date_idx + 1)
                    if lookback < 1:
                        continue
                        
                    forecast_ts = ts.iloc[max(0, current_date_idx - lookback + 1):current_date_idx + 1]
                    
                    # For initial period, use simple average of available data
                    forecast = forecast_ts.mean() if not forecast_ts.empty else 0
                    
                    if forecast > 0:  # Only store non-zero forecasts
                        forecast_by_category[cat] = forecast
                        
                except (KeyError, ValueError) as e:
                    continue
            
            # Calculate error metrics
            error = total_forecast - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual * 100) if actual > 0 else 0
            
            results.append({
                'date': current_date,
                'forecast_date': current_date,  # Same as current_date for initial period
                'forecast_horizon': forecast_horizon,
                'actual': actual,
                'forecast': total_forecast,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error,
                'within_10pct': 1 if pct_error <= 10 else 0,
                'within_20pct': 1 if pct_error <= 20 else 0,
                'forecast_breakdown': forecast_by_category,
                'initial_period': True  # Mark as initial period forecast
            })
            
            # Calculate error metrics
            error = total_forecast - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual * 100) if actual > 0 else 0
            
            results.append({
                'date': current_date,
                'forecast_date': current_date,  # Same as current_date for initial period
                'forecast_horizon': forecast_horizon,
                'actual': actual,
                'forecast': total_forecast,
                'error': error,
                'abs_error': abs_error,
                'pct_error': pct_error,
                'within_10pct': 1 if pct_error <= 10 else 0,
                'within_20pct': 1 if pct_error <= 20 else 0,
                'forecast_breakdown': forecast_by_category,
                'initial_period': True  # Mark as initial period forecast
            })
        
        # Then process the remaining dates with full forecast horizon
        for i in range(forecast_horizon, len(actual_dates)):
            current_date = actual_dates[i]
            forecast_date = actual_dates[i - forecast_horizon]
            

                
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
                # Get all data for this category up to forecast date
                cat_data = df[(df['category'] == cat) & (df['date'] <= forecast_date)].copy()
                # Limit to recent activity_window days
                if activity_window is not None and activity_window > 0:
                    window_start = forecast_date - pd.Timedelta(days=activity_window - 1)
                    cat_data = cat_data[cat_data['date'] >= window_start]
                if len(cat_data) < 1:  # Need at least one data point
                    continue
                    
                # Create time series with proper datetime index
                ts = cat_data.groupby('date')['amount'].sum()
                
                try:
                    # Get the position of the forecast date
                    forecast_date_idx = ts.index.get_loc(forecast_date)
                    
                    # Determine lookback period based on timeframe
                    max_lookback = 30 if timeframe == 'daily' else 14
                    lookback = min(max_lookback, forecast_date_idx + 1)
                    
                    # Ensure we have at least one week of data
                    if lookback < 1:
                        continue
                        
                    forecast_ts = ts.iloc[max(0, forecast_date_idx - lookback + 1):forecast_date_idx + 1]
                    
                    # For weekly aggregation, use simple average of available data
                    if timeframe == 'weekly':
                        forecast = forecast_ts.mean() if not forecast_ts.empty else 0
                    else:
                        # For daily, use average of same day of week
                        day_of_week = pd.Timestamp(current_date).dayofweek
                        same_day = forecast_ts[forecast_ts.index.dayofweek == day_of_week]
                        forecast = same_day.mean() if not same_day.empty else 0
                        
                    if forecast > 0:  # Only store non-zero forecasts
                        forecast_by_category[cat] = forecast
                        
                except (KeyError, ValueError) as e:
                    continue
            
            # Initialize total_forecast
            total_forecast = 0
            
            # Only proceed if we have forecasts
            if forecast_by_category:
                total_forecast = sum(forecast_by_category.values())
            
            # If no forecasts were generated, skip this date
            if len(forecast_by_category) == 0:
                continue
            
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
                'within_20pct': 1 if pct_error <= 20 else 0,
                'forecast_breakdown': forecast_by_category
            })
        
        daily_results = pd.DataFrame(results)
        
        # If timeframe is daily, return as is
        if timeframe == 'daily':
            return daily_results
        
        # For weekly timeframe, aggregate the daily results
        if timeframe == 'weekly':
            try:
                weekly_results = aggregate_daily_to_weekly(daily_results)
                if not weekly_results.empty:
                    weekly_results = add_week_metadata(weekly_results)
                    return weekly_results
                return pd.DataFrame()
            except Exception as weekly_error:
                st.error(f"Error in weekly aggregation: {str(weekly_error)}")
                import traceback
                st.code(traceback.format_exc())
                return daily_results  # Fall back to daily results if weekly fails
        
        # For monthly timeframe, aggregate the daily results
        if timeframe == 'monthly':
            try:
                monthly_results = aggregate_daily_to_monthly(daily_results)
                if not monthly_results.empty:
                    return monthly_results
                return pd.DataFrame()
            except Exception as monthly_error:
                st.error(f"Error in monthly aggregation: {str(monthly_error)}")
                import traceback
                st.code(traceback.format_exc())
                return daily_results  # Fall back to daily results if monthly fails
        
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

# Add timeframe selector (daily/weekly)
# View is fixed to Daily to simplify Backward Forecast Performance (weekly/monthly buckets removed)
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

# --- Extend data window by forecast horizon so that forecasts for the first
# days in the reporting window are generated using data that would have been
# available at that time. We pull extra `horizon` days before the start date.
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
    # Guard against KeyError if 'date' is missing (e.g., empty DataFrame)
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

# --- Export forecast to Excel ---
if not forecast_df.empty:
    export_df = pd.DataFrame(forecast_export_rows)
    export_path = config.get_forecast_results_path()
    with ExcelWriter(export_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False)
    st.success(f"Forecast results exported to {export_path}")
