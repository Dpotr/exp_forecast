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

## How forecasting works (for dummies)

### 1. Data Preparation
- Expenses are grouped by category and day.
- Only the last 30 days are used for forecasting to adapt to your current lifestyle.
- Outliers (top 1% of daily expenses) are removed to avoid skewed predictions.

### 2. Forecasting Model
- For each category, a time series model (Prophet) is trained on the last 30 days.
- The model uses a log-transform to stabilize variance:
  - **Formula:** `log_amount = log(1 + amount)`
- The model forecasts the next 7 days in log scale, then results are transformed back:
  - **Formula:** `forecast = exp(log_amount_forecast) - 1`
- Forecasts are never negative (clamped to zero).

### 3. Moving Average Baseline
- For comparison, a simple 7-day moving average of daily total expenses is shown.
- If the model’s forecast is much higher than this average, a warning is displayed.

### 4. Output Files
- Forecast results are saved to `forecast_results.xlsx` after each run.

### 5. Example
Suppose your last 30 days of daily expenses are:

| Date       | Amount |
|------------|--------|
| 2025-03-21 | 120    |
| 2025-03-22 | 80     |
| ...        | ...    |
| 2025-04-19 | 150    |
| 2025-04-20 | 90     |

The model will:
- Remove any days above the 99th percentile (outliers)
- Fit a model to the log of these amounts
- Forecast the next 7 days
- Transform the forecast back to normal scale and show it in the dashboard and Excel export

## Forecasting Methodology and Tracking Signals

### 1. Activity Window
- You can set the "recent activity window" (number of days) in the Streamlit sidebar. Only categories with at least one nonzero expense in this window are forecasted.
- **Example:** If you set the window to 30 days, only categories with expenses in the last 30 days will be forecasted.

### 2. Model Selection and Backtesting
- For each category, the tool backtests several forecasting methods on the last 60 days:
  - Mean of last 30 days
  - Median of last 30 days
  - Croston’s method (for intermittent demand)
  - Prophet (for trend/seasonality)
  - Zero (no forecast)
- **The method with the lowest MAE (Mean Absolute Error) in backtesting is selected automatically for forecasting.**

### 3. Forecasting Formulas
- **Mean:**  
  `forecast = mean(amounts[-30:])`
- **Median:**  
  `forecast = median(amounts[-30:])`
- **Croston:**  
  Specialized for lumpy demand (see `croston.py` for code)
- **Prophet:**  
  Time series model with trend and seasonality, fitted to historical data.
- **Zero:**  
  `forecast = 0` (used for inactive/one-off categories)

### 4. Tracking Signals (Forecast Diagnostics)
- The dashboard shows a diagnostics table for each category:
  - Best method (chosen by backtest MAE)
  - MAE (Mean Absolute Error)
  - All MAEs for all methods
  - Whether the category is "active for forecast" (based on activity window)
  - The activity window size used
- Use this table to understand which method is used and how well it performed in backtesting.

### 5. Example (Step-by-Step)
Suppose you have the following expenses in the last 30 days:

| Date       | Category   | Amount |
|------------|------------|--------|
| 2025-04-01 | groceries  | 20     |
| 2025-04-02 | groceries  | 30     |
| ...        | ...        | ...    |
| 2025-04-28 | groceries  | 25     |
| 2025-04-15 | taxi car   | 100    |
| 2025-04-20 | hotel      | 400    |

- If you set the activity window to 14 days, only "groceries" (if it has expenses in that window) will be forecasted; "taxi car" and "hotel" will be skipped.
- The tool will backtest all methods for "groceries" and select the one with the lowest MAE.

### 6. How to Use the Tool (Step-by-Step Guide)
1. **Prepare your data:**
   - Place your expenses in `expenses.xlsx` (columns: `date`, `category`, `amount`).
2. **Launch the dashboard:**
   - Run: `streamlit run app.py`
3. **Adjust the activity window:**
   - Use the sidebar slider to set the "Recent activity window (days)" as desired (e.g., 14, 30, 60).
   - The dashboard and forecasts update automatically.
4. **Review diagnostics:**
   - Check the diagnostics table for method selection, MAE, and activity status.
5. **Export results:**
   - Forecasts are saved to `forecast_results.xlsx` after each run.
6. **Push updates to GitHub:**
   - Use the sidebar button to commit and push changes (if enabled).

### 7. Tracking Signals (for Dummies)
- **Tracking signal** is a way to monitor if your forecast is consistently too high or too low compared to actuals.
- In this tool, MAE (Mean Absolute Error) is used as a simple tracking signal: the lower the MAE, the better the forecast matches reality.
- If you see a high MAE, consider changing the activity window or reviewing your data for outliers.

## Robust Periodic Spike Forecasting

### What’s New
- The dashboard now robustly forecasts regular, lumpy, high-value expenses (like rent, school, car rent) even if the payment is slightly late or early.
- If the next expected spike is just before or at the start of the forecast window, the spike will appear at the first day of the window—no more missed rent or car payments due to real-world timing delays!
- Category-specific spike thresholds ensure only true spikes are forecasted (e.g., rent + communal >500, school >300, car rent >300).
- The dashboard info box will always show the next expected spike date and amount for these categories.

### How it works
- For categories like `rent + communal`, `school`, and `car rent`, the system:
  - Detects the last two true spikes (above threshold)
  - Calculates the typical interval between them
  - Predicts the next spike date
  - If the spike is due or slightly overdue (within 3 days of the forecast window), it is forecasted at the start of the window
- This makes your forecast robust against real-life payment delays and ensures you never miss a major recurring expense in your plan.

## Automated Expenses Updater

### What it does
- Reads new records from the "data" sheet in your `daily payments.xlsx` file (in `C:\Users\potre\OneDrive\Documents (excel files e t.c.)`)
- Only considers columns: `date`, `category`, and `price EUR` (used as `amount`)
- Cleans and normalizes all values:
  - Amount: removes currency symbols, converts to float, rounds to 2 decimals
  - Category: stripped and lowercased
  - Date: only the date part is used (no time)
- Filters out any rows where date/category is missing or amount is zero/NaN
- Appends only truly new records (no duplicates) to `expenses.xlsx`
- Sorts all records by date (oldest first)
- Removes all rows with missing/empty/zero values in key columns
- After each run, performs a test to guarantee no duplicates are ever inserted (idempotency)

### How to use
1. Place your daily payments in the Excel file mentioned above, in the `data` sheet.
2. Run: `python update_expenses_from_daily.py`
3. The script will print how many new records were added, or "No new records to add."
4. Your `expenses.xlsx` will always be clean, deduplicated, and sorted.
5. If you run the script again, it will not add duplicates.

### Troubleshooting
- If you get a diagnostic printout, it means some records were not matched due to subtle formatting issues. The script now filters out all empty/NaN/zero rows, so only real records are processed.

## Seasonality Diagnostics

You can now analyze both weekly and within-month seasonality in your expenses using the `seasonality_diagnostics.py` script.

### How to Use
- Run the script with `python seasonality_diagnostics.py` in your project directory.
- It will show two plots:
  1. **Average Spending by Weekday** (Monday–Sunday)
  2. **Average Spending by Day of Month** (1–31)
- You can also specify a category by editing the script to pass `category='your category'` to the plotting functions.

### Why This Matters
- These diagnostics help you see patterns like higher weekend spending or spikes around payday/rent.
- Use these insights to make your forecasts more life-like and confident!

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
