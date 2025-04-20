import numpy as np

def croston(ts, forecast_periods=7, alpha=0.1):
    '''
    Croston's method for intermittent demand forecasting
    ts: 1D array-like, historical demand (should be non-negative)
    forecast_periods: int, number of periods to forecast
    alpha: float, smoothing parameter (0 < alpha < 1)
    Returns: forecast array of length forecast_periods
    '''
    ts = np.array(ts)
    n = len(ts)
    demand = ts[ts > 0]
    if len(demand) == 0:
        return np.zeros(forecast_periods)
    intervals = np.diff(np.where(ts > 0)[0], prepend=-1)
    if len(intervals) < 1:
        intervals = np.ones_like(demand)
    z = demand[0]
    p = intervals[0]
    forecasts = []
    for i in range(1, len(demand)):
        z = alpha * demand[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p
    croston_forecast = z / p if p > 0 else 0
    return np.repeat(croston_forecast, forecast_periods)
