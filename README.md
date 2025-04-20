# Expense Forecast Dashboard

This project is an interactive web dashboard for forecasting categorized expenses, tracking accuracy KPIs, and ensuring reliable forecasts with backtesting and self-correction. It is built with Python, Streamlit, Prophet, and Plotly.

## Features
- **Expense Forecasting**: Forecasts daily expenses for the next 7 days by category, incorporating seasonality and country effects.
- **Backtesting & KPI Tracking**: Performs rolling backtests and calculates MAE, RMSE, and MAPE to ensure forecast quality before publishing.
- **Self-Correction**: Automatically updates and retrains models as new data arrives, ensuring adaptive and accurate predictions.
- **Interactive Visualizations**: Stacked bar charts, forecast plots with confidence intervals, and KPI tables.
- **Country Regimes**: Highlights periods spent in Vietnam, Kazakhstan, and Montenegro, and uses these as features in the model.

## Usage
1. Place your historical expenses in `expenses.xlsx` (columns: `date`, `category`, `amount`).
2. Run the dashboard locally: `streamlit run app.py`
3. Or, use the live online dashboard (if deployed via [streamlit.io](https://streamlit.io)):
   - The dashboard will automatically update online after each push to the GitHub repository.
   - Simply update your code, commit, and push to GitHub. The online dashboard will reflect the latest changes after a short delay.
4. Explore raw data, monthly summaries, forecasts, and backtest KPIs in the browser.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Model Details
- Uses Facebook Prophet for time series forecasting, with daily and yearly seasonality.
- Country is used as a categorical regressor.
- Forecasts are validated with rolling backtests, and confidence intervals are shown for each prediction.

## Repository
GitHub: [https://github.com/Dpotr/exp_forecast](https://github.com/Dpotr/exp_forecast)

---

If you have questions, suggestions, or want to contribute, please open an issue or pull request on GitHub.
