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

# --- Forecasting, Backtesting, KPI Tracking ---
st.subheader("Expense Forecasts by Category (7-day horizon)")

results = []
backtest_results = []
forecast_horizon = 7
backtest_window = 60  # days for backtesting

# For Excel export
forecast_export_rows = []

for cat in df['category'].unique():
    # --- Restrict to last 30 days for most adaptive forecast ---
    df_cat = df[df['category'] == cat].copy()
    last_30_start = df['date'].max() - pd.Timedelta(days=30)
    df_cat = df_cat[df_cat['date'] >= last_30_start]
    # --- Outlier removal: drop top 1% ---
    if len(df_cat) > 10:
        threshold = df_cat['amount'].quantile(0.99)
        df_cat = df_cat[df_cat['amount'] <= threshold]
    # --- Aggregate by date ---
    df_cat = df_cat.groupby(['date', 'country'], as_index=False)['amount'].sum()
    df_cat = df_cat.sort_values('date')
    # --- Log-transform for stability ---
    df_cat['log_amount'] = np.log1p(df_cat['amount'])
    # Prepare for Prophet
    prophet_df = pd.DataFrame({
        'ds': df_cat['date'],
        'y': df_cat['log_amount'],
        'country': df_cat['country']
    })
    prophet_df['country'] = prophet_df['country'].astype(str)
    # --- Skip if not enough data ---
    if len(prophet_df.dropna()) < 2:
        st.warning(f"Not enough data to forecast for category '{cat}' (need at least 2 days in recent period). Skipping.")
        continue
    m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
    for c in prophet_df['country'].unique():
        m.add_regressor(f"country_{c}")
    for c in prophet_df['country'].unique():
        prophet_df[f"country_{c}"] = (prophet_df['country'] == c).astype(int)
    m.fit(prophet_df[['ds', 'y'] + [f"country_{c}" for c in prophet_df['country'].unique()]])
    # Forecast future
    last_date = df_cat['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    last_country = df_cat.iloc[-1]['country']
    future = pd.DataFrame({'ds': future_dates})
    for c in prophet_df['country'].unique():
        future[f"country_{c}"] = int(last_country == c)
    forecast = m.predict(future)
    # Inverse log-transform, clamp to zero
    for i, row in forecast.iterrows():
        forecast_val = max(np.expm1(row['yhat']), 0)
        lower_val = max(np.expm1(row['yhat_lower']), 0)
        upper_val = max(np.expm1(row['yhat_upper']), 0)
        results.append({
            'category': cat,
            'date': row['ds'].date(),
            'forecast': forecast_val,
            'lower': lower_val,
            'upper': upper_val
        })
        forecast_export_rows.append({
            'category': cat,
            'date': row['ds'].date(),
            'forecast': forecast_val,
            'lower': lower_val,
            'upper': upper_val
        })
    # --- Backtesting ---
    # Use last N days for backtest
    if len(df_cat) > backtest_window + forecast_horizon:
        test_start = df_cat['date'].max() - pd.Timedelta(days=backtest_window)
        df_bt = df_cat[df_cat['date'] >= test_start]
        bt_dates = df_bt['date'].unique()
        actuals = []
        preds = []
        for d in bt_dates[:-forecast_horizon]:
            train_bt = df_cat[df_cat['date'] < d]
            test_bt = df_cat[df_cat['date'] == d]
            if len(train_bt) < 10: continue
            bt_prophet_df = pd.DataFrame({
                'ds': train_bt['date'],
                'y': train_bt['log_amount'],
                'country': train_bt['country'].astype(str)
            })
            for c in bt_prophet_df['country'].unique():
                bt_prophet_df[f"country_{c}"] = (bt_prophet_df['country'] == c).astype(int)
            m_bt = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
            for c in bt_prophet_df['country'].unique():
                m_bt.add_regressor(f"country_{c}")
            m_bt.fit(bt_prophet_df[['ds', 'y'] + [f"country_{c}" for c in bt_prophet_df['country'].unique()]])
            # Predict for d
            future_bt = pd.DataFrame({'ds': [d]})
            last_country_bt = train_bt.iloc[-1]['country']
            for c in bt_prophet_df['country'].unique():
                future_bt[f"country_{c}"] = int(last_country_bt == c)
            pred_bt = m_bt.predict(future_bt)
            actuals.append(np.expm1(test_bt['log_amount'].sum()))
            preds.append(np.expm1(pred_bt['yhat'].iloc[0]))
        # KPIs
        if len(actuals) > 0:
            mae = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / (np.array(actuals) + 1e-8))) * 100
            backtest_results.append({
                'category': cat,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })

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
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['upper'], mode='lines', name='Upper 95%',
            line=dict(width=0), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['lower'], mode='lines', name='Lower 95%',
            fill='tonexty', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), showlegend=True
        ))
        fig.update_layout(title=f"Forecast for {cat} (with last 30d history)", xaxis_title="Date", yaxis_title="Amount", height=350)
        st.plotly_chart(fig, use_container_width=True)

# Show backtest KPIs with subjective scoring
bt_df = pd.DataFrame(backtest_results)
def subjective_score(mape):
    if mape < 10:
        return "Excellent"
    elif mape < 20:
        return "Good"
    elif mape < 35:
        return "Acceptable"
    else:
        return "Poor"
if not bt_df.empty:
    bt_df['Subjective Score'] = bt_df['MAPE'].apply(subjective_score)
    st.write("### Backtest KPIs (last 60 days)")
    st.dataframe(bt_df)

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

st.info("Forecasts use Prophet with daily/yearly seasonality and country as a regressor. 95% confidence intervals are shown. KPIs are calculated by rolling backtest. Model auto-updates with new data.")
