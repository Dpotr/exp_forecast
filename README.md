# Expense Forecast Dashboard

This project is an interactive web dashboard for forecasting categorized expenses, tracking accuracy KPIs, and ensuring reliable forecasts with backtesting and self-correction. It is built with Python, Streamlit, Prophet, and Plotly.

## Features
- **Expense Forecasting**: Forecasts daily expenses for the next 7 days by category, incorporating seasonality and country effects.
- **Backward Forecast Analysis**: Evaluates forecast accuracy by comparing past predictions with actuals for different time horizons. View results by day or aggregated by week for trend analysis.
- **Category-Specific Analysis**: Analyze forecast accuracy for individual categories or groups of categories.
- **Backtesting & KPI Tracking**: Performs rolling backtests and calculates MAE, RMSE, and MAPE to ensure forecast quality before publishing.
- **Self-Correction**: Automatically updates and retrains models as new data arrives, ensuring adaptive and accurate predictions.
- **Interactive Visualizations**: Stacked bar charts, forecast plots with confidence intervals, and KPI tables.
- **Forecast Diagnostics**: Enhanced diagnostics table displays MAE for each forecasting method with separate columns and features a date range selector to filter the analysis period.
- **Country Regimes**: Highlights periods spent in different countries and uses these as features in the model.

## Usage
1. Place your historical expenses in `expenses.xlsx` (columns: `date`, `category`, `amount`).
2. Run the dashboard locally: `streamlit run app.py`
3. Or, use the live online dashboard (if deployed via [streamlit.io](https://streamlit.io)):
   - The dashboard will automatically update online after each push to the GitHub repository.
   - Simply update your code, commit, and push to GitHub. The online dashboard will reflect the latest changes after a short delay.
4. Explore raw data, monthly summaries, forecasts, and backtest KPIs in the browser.
5. Use the Diagnostics date range selector in the sidebar to filter per-category metrics over a custom analysis period.

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

### 4. Backward Forecast Analysis
- **How It Works**:
  - For each day in your historical data, the system looks back to see what was forecasted N days ago
  - Compares these forecasts with what actually happened
  - Calculates accuracy metrics including MAE, MAPE, and accuracy within ±10% and ±20%
  - Visualizes forecast vs. actual over time with interactive plots

### 5. Category Selection
- Select specific categories to analyze their forecast accuracy
- Compare performance across different expense types
- Identify which categories are easiest/hardest to predict

## Monthly Bucket for Forecast Analysis

A new feature has been added to the forecast accuracy analysis: the **Monthly Bucket**. This allows you to aggregate and analyze forecast performance at the calendar month level, in addition to the previously available daily and weekly buckets.

### How It Works
- In the timeframe selector (in the Streamlit UI), you can now choose **Monthly** alongside Daily and Weekly.
- When Monthly is selected, all forecast and actual data are aggregated by calendar month.
- The app computes summary metrics (total/average actual, forecast, errors, WAPE, MAPE, accuracy bands, etc.) for each month, and displays the results in plots and tables.
- The monthly bucket is useful for high-level, long-term trend analysis and for identifying large-scale forecast deviations.

### Limitations
- **Forecast breakdown by category is NOT available for monthly bucket** (as with weekly), because category-level detail is lost during aggregation.
- When Monthly is selected, the UI will show an info message instead of category breakdown tables.
- Spike details and breakdowns are only available in Daily mode.

### Usage
- To use the monthly bucket, simply select "Monthly" in the timeframe selector in the app sidebar.
- All summary metrics and main plots will update to show results by month.
- X-axis and hover labels in plots will show the month for each data point.

### Supported Metrics and Visualizations
- All main forecast accuracy metrics are supported for monthly bucket: Total Actual, Total Forecast, Error, MAE, MAPE, WAPE, accuracy within ±10% and ±20%, etc.
- Main time series plots and summary tables support monthly aggregation.
- Category breakdown and spike details are only available for daily.

## Weekly Aggregation Feature

The dashboard now includes a weekly aggregation view for backward forecast analysis, allowing you to analyze forecast performance at a weekly level to identify trends that might be less visible in daily data.

### How to Use Weekly Aggregation

1. **Access the Feature**:
   - In the Backward Forecast Performance Analysis section, find the "View" radio buttons at the top.
   - Toggle between "Daily" and "Weekly" to switch between views.

2. **Understanding Weekly Aggregation**:
   - **Daily View**: Shows actual vs forecasted values for each day, with daily error metrics.
   - **Weekly View**: Aggregates daily values into weekly totals (Sunday-Saturday) and calculates metrics on the aggregated values.

3. **Key Differences in Weekly View**:
   - X-axis shows the start date of each week
   - Error metrics are calculated on weekly aggregates, not daily values
   - Error bands are hidden as they're less meaningful with weekly aggregation
   - Hover tooltips show weekly summaries instead of daily values

4. **When to Use Weekly Aggregation**:
   - To smooth out daily volatility and focus on weekly trends
   - When daily forecasts are too noisy to identify patterns
   - For longer-term planning where weekly aggregates are more relevant
   - To evaluate forecast performance over longer time horizons

5. **Technical Notes**:
   - Weekly aggregation is performed after calculating daily forecasts and metrics
   - The week is defined as Sunday-Saturday (following standard business conventions)
   - Weeks with partial data (at the beginning or end of your date range) are still included
   - All amounts are summed within each week before calculating percentage errors

### 6. Output Files
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

### Method Applicability and Overrides

- **Intermittent Demand:** Croston’s method is best suited for sparse or lumpy series with intermittent non-zero values.
- **Trend & Seasonality:** Prophet is preferred when the time series exhibits clear trends or seasonal patterns.
- **Regular Spikes:** Periodic Spike method handles categories with predictable periodic spikes (e.g., 'school', 'rent + communal', 'car rent').
- **Low Activity:** For series with fewer than 2 non-zero days in the activity window, the Periodic Spike method is used by default.
- **Manual Override:** Switch to Manual mode in the Streamlit sidebar to explicitly select any forecasting method for comparison or specific use cases.

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

### 7. How to Use the Tool (Step-by-Step Guide)
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

## Seasonality Heatmap & Dashboard Layout

- The dashboard now includes a **Seasonality Heatmap** showing average spending by day of week and day of month (last 60 days). This gives you instant insight into both weekly and within-month patterns in your expenses.
- All dashboard boards/sections are grouped into logical blocks with clear headers for easier navigation:
  1. Raw Data
  2. Seasonality Insights (Heatmap)
  3. Monthly Summary & Country Regimes
  4. Trendline (Last 90 Days)
  5. Forecast Diagnostics Table
  6. Forecast Table & Summary
  7. Stacked Chart (History + Forecast)
  8. Per-Category Forecasts
- Use the heatmap to spot recurring patterns and the new layout to quickly find the information you need.

## Advanced Insights and Features

- **Anomaly & Outlier Detection:** The dashboard highlights days and transactions that are statistical outliers, helping you spot suspicious or unusual activity quickly.
- **Forecast Confidence:** Forecast charts now include prediction intervals (upper/lower bounds) where available, and a visual traffic light/score for each forecast based on diagnostics.
- **Expense Breakdown:** See which categories dominate your spending, and visualize cumulative/running totals for any period.
- **Recurring Payment Detection:** Automatically detects subscriptions, rent, and other recurring payments, listing their next due date and amount. Alerts you if a recurring payment is overdue or missing.
- **Comparisons:** Instantly compare current week/month to previous periods, and see budget vs. actuals if budget data is provided.
- **Export & Sharing:** Download filtered tables/charts as Excel/CSV or images. One-click PDF/HTML summary report.
- **User Guidance:** Contextual tooltips and explanations are provided for every chart, metric, and anomaly, including why a certain forecast method was chosen.

## Version History

### v1.3.0 (2025-07-24)
- Added Diagnostics date range selector for filtering the Diagnostics table.
- Expanded All MAEs into separate columns in the Diagnostics table for clearer metrics.
- Rolling backtests now consider the selected Diagnostics date range.
- Improved UI tooltip placement for diagnostics metrics.

### v1.2.0 (2025-07-23)
- Forecast horizon alignment: for any date displayed, the value is the forecast generated exactly *N* days earlier (where *N* is the selected Forecast Horizon). This ensures that the earliest days in the report window use truly "look-ahead" predictions.
- The dashboard now automatically pulls an extra *horizon* days of data before the chosen start date so the underlying model has access to the information required to build those earlier forecasts.
- Removed the previous "initial period" shortcut; all dates are now handled with the same logic.
- UI unchanged – simply pick a date range and horizon, and the app does the rest.

### v1.1.0 (2025-04-21)
- Added Forecast Metrics Heatmap: Visualizes how MAE varies with activity window and forecast horizon, recommends best settings.
- Diagnostics table now recalculates metrics based on user-selected activity window and forecast horizon.
- Removed lower/upper forecast intervals from dashboard and exports for clarity.
- Added collapsible tooltips section for all metrics in the diagnostics table.
- Improved code and UI clarity based on user feedback.

### [2025-07-24] Bugfix: Weekly view KeyError for forecast_breakdown
- Исправлена ошибка KeyError при попытке построить breakdown по категориям для недельных (weekly) бакетов в разделе Spike Forecast Details.
- Теперь при выборе недельного режима breakdown не отображается, а пользователю выводится информативное сообщение: breakdown по категориям доступен только для дневного режима.
- Это связано с тем, что при агрегации по неделям детальная структура breakdown теряется и не может быть корректно отображена.

**Особенности:**
- В дневном (daily) режиме breakdown по категориям доступен для анализа аномалий.
- В недельном (weekly) режиме breakdown по категориям не формируется и не отображается.


- Python 3.8+
- See `requirements.txt` for dependencies

## Model Details
- Uses Facebook Prophet for time series forecasting, with daily and yearly seasonality.
- Country is used as a categorical regressor.
- Forecasts are validated with rolling backtests, and confidence intervals are shown for each prediction.

## Repository
GitHub: [https://github.com/Dpotr/exp_forecast](https://github.com/Dpotr/exp_forecast)

---

**Note (July 2025):**
- Backward Forecast Performance now only supports Daily view. Weekly and Monthly bucket aggregation has been removed to simplify analysis and avoid confusion.

**Historical Notes (April 2025):**
- The dashboard and forecast exports now show only the main forecast value for each day/category.
- Lower/upper confidence intervals are no longer included, as they were not consistently available or meaningful for all methods.
- This change simplifies the results and avoids confusion from empty columns.

---

If you have questions, suggestions, or want to contribute, please open an issue or pull request on GitHub.
