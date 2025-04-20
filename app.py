import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

st.set_page_config(page_title="Expense Forecast Dashboard", layout="wide")
st.title("Expense Forecast Dashboard")

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

for cat in df['category'].unique():
    df_cat = df[df['category'] == cat].copy()
    df_cat = df_cat.sort_values('date')
    # Prepare for Prophet
    prophet_df = pd.DataFrame({
        'ds': df_cat['date'],
        'y': df_cat['amount'],
        'country': df_cat['country']
    })
    # Prophet expects string for extra regressors
    prophet_df['country'] = prophet_df['country'].astype(str)
    # Fit model
    m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
    for c in prophet_df['country'].unique():
        m.add_regressor(f"country_{c}")
    # Add country dummies
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
    # Collect forecast results (clamp to zero)
    for i, row in forecast.iterrows():
        results.append({
            'category': cat,
            'date': row['ds'].date(),
            'forecast': max(row['yhat'], 0),
            'lower': max(row['yhat_lower'], 0),
            'upper': max(row['yhat_upper'], 0)
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
                'y': train_bt['amount'],
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
            actuals.append(test_bt['amount'].sum())
            preds.append(pred_bt['yhat'].iloc[0])
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
if not forecast_df.empty:
    st.write("### 7-Day Forecast by Category")
    st.dataframe(forecast_df)
    # --- Total Forecast Summary Board ---
    st.write("#### Total Forecast Summary (All Categories)")
    total_forecast = forecast_df.groupby('date')['forecast'].sum().reset_index()
    total_sum = total_forecast['forecast'].sum()
    st.metric(label="Total Forecasted Expenses (7 days)", value=f"{total_sum:.2f}")
    st.dataframe(total_forecast.rename(columns={"forecast": "Total Forecast"}))
    # Plot forecast for each category
    for cat in forecast_df['category'].unique():
        df_plot = forecast_df[forecast_df['category'] == cat]
        fig = go.Figure()
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
        fig.update_layout(title=f"Forecast for {cat}", xaxis_title="Date", yaxis_title="Amount", height=350)
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

st.info("Forecasts use Prophet with daily/yearly seasonality and country as a regressor. 95% confidence intervals are shown. KPIs are calculated by rolling backtest. Model auto-updates with new data.")
