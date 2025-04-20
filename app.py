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

st.subheader("Raw Data Preview")
st.dataframe(df.head(20))

st.subheader("Expenses by Category and Country")
agg = df.groupby(['country', 'category'])['amount'].sum().reset_index()
st.dataframe(agg)

# --- Monthly stacked bar chart by category with country highlights ---
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

# --- Last 90 Days with Trendline ---
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

# --- Expert demand planning: backtest multiple methods and select best ---
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

# --- Streamlit sidebar: adjustable activity window ---
st.sidebar.header("Forecast Settings")
activity_window = st.sidebar.slider("Recent activity window (days)", min_value=7, max_value=90, value=30, step=1)

results = []
forecast_export_rows = []
category_forecasts = []
forecast_horizon = 7
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
    def mean_method(ts, fh): return np.repeat(np.mean(ts[-30:]), fh)
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
        future = pd.DataFrame({'ds': pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=fh)})
        forecast = m.predict(future)
        return forecast['yhat'].clip(lower=0).values
    methods = {
        'mean': mean_method,
        'median': median_method,
        'zero': zero_method,
        'croston': croston_method,
        'prophet': prophet_method
    }
    # --- Backtest and select best method ---
    avg_errors = rolling_backtest(ts, methods, window=60, forecast_horizon=7)
    best_method = min(avg_errors, key=avg_errors.get)
    # --- Forecast with best method only if active ---
    if forecast_active:
        forecast_vals = methods[best_method](ts.values, forecast_horizon)
        # --- Tracking signals for best method ---
        # Use last 60 days for backtest tracking signals
        if len(ts) > 67:
            y_true = ts[-7:]
            y_pred = methods[best_method](ts[:-7], 7)
        else:
            y_true = ts[-7:]
            y_pred = forecast_vals
        signals = tracking_signals(y_true, y_pred)
        score, explanation = subjective_score(signals['MAPE'], signals['Tracking Signal'])
        for i in range(forecast_horizon):
            forecast_date = (df['date'].max() + pd.Timedelta(days=i+1)).date()
            results.append({
                'category': cat,
                'date': forecast_date,
                'forecast': forecast_vals[i],
                'lower': None,
                'upper': None
            })
            forecast_export_rows.append({
                'category': cat,
                'date': forecast_date,
                'forecast': forecast_vals[i],
                'lower': None,
                'upper': None
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

# --- Show forecast diagnostics table with explanations ---
df_diag = pd.DataFrame(category_forecasts)
st.write('### Forecast Diagnostics Table (Expert Backtest Selection)')
st.dataframe(df_diag)

# --- Show subjective scoring and explanations ---
for idx, row in df_diag.iterrows():
    st.write(f"**{row['category']}**: {row['Score']} — {row['Explanation']}")

# Show forecast table
forecast_df = pd.DataFrame(results)
# --- Export forecast to Excel ---
if not forecast_df.empty:
    export_df = pd.DataFrame(forecast_export_rows)
    export_path = os.path.join(os.path.dirname(__file__), 'forecast_results.xlsx')
    with ExcelWriter(export_path, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False)
    st.success(f"Forecast results exported to {export_path}")

if not forecast_df.empty:
    st.write("### 7-Day Forecast by Category")
    st.dataframe(forecast_df)
    # --- Total Forecast Summary Board ---
    st.write("#### Total Forecast Summary (All Categories)")
    total_forecast = forecast_df.groupby('date')['forecast'].sum().reset_index()
    total_sum = total_forecast['forecast'].sum()
    st.metric(label="Total Forecasted Expenses (7 days)", value=f"{total_sum:.2f}")
    st.dataframe(total_forecast.rename(columns={"forecast": "Total Forecast"}))

    # --- Table: 7-day forecast by day/category ---
    st.write("#### Forecast Table: Next 7 Days by Category")
    forecast_pivot = forecast_df.pivot(index='date', columns='category', values='forecast').fillna(0)
    st.dataframe(forecast_pivot.style.format("{:.2f}"))

    # --- Stacked column chart: 30 days history + 7 days forecast ---
    st.write("#### Stacked Column Chart: 30 Days History + 7 Days Forecast")
    # Prepare historical data (last 30 days)
    last_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
    hist_pivot = last_30.groupby(['date', 'category'])['amount'].sum().reset_index()
    hist_pivot = hist_pivot.pivot(index='date', columns='category', values='amount').fillna(0)
    # Combine history and forecast
    combined = pd.concat([hist_pivot, forecast_pivot], axis=0)
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
        title="Expenses: Last 30 Days (Actual) + Next 7 Days (Forecast)",
        xaxis_title="Date",
        yaxis_title="Expenses",
        legend_title="Category",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Plot forecast for each category with last 30 days of history
    for cat in forecast_df['category'].unique():
        df_plot = forecast_df[forecast_df['category'] == cat]
        # Last 30 days of actuals
        df_cat_hist = df[(df['category'] == cat) & (df['date'] >= (df['date'].max() - pd.Timedelta(days=30)))]
        df_cat_hist = df_cat_hist.groupby('date')['amount'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_cat_hist['date'], y=df_cat_hist['amount'], name='Actual (last 30d)', marker_color='gray', opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['forecast'], mode='lines+markers', name='Forecast',
            line=dict(color='royalblue')
        ))
        fig.update_layout(title=f"Forecast for {cat} (with last 30d history)", xaxis_title="Date", yaxis_title="Amount", height=350)
        st.plotly_chart(fig, use_container_width=True)

# --- Moving average baseline for comparison ---
st.write("#### Moving Average Baseline (last 30 days)")
hist_30 = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=30))]
ma_baseline = hist_30.groupby('date')['amount'].sum().rolling(window=7, min_periods=1).mean()
st.line_chart(ma_baseline, use_container_width=True)

# --- Warning if Prophet forecast is much higher than moving average ---
if not forecast_df.empty:
    forecast_total = forecast_df.groupby('date')['forecast'].sum().mean()
    ma_total = ma_baseline.mean()
    if forecast_total > 1.5 * ma_total:
        st.warning(f"Forecasted daily expenses are much higher than recent 7-day moving average. Please review model results.")

st.info("Forecasts use a hybrid robust approach: Prophet is used if enough data and its forecast is not extreme; otherwise, mean, median, or zero is used. 95% confidence intervals are not shown. Model auto-updates with new data.")
