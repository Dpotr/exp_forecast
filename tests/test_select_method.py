import pandas as pd
import pytest
import app
from app import select_method


def test_select_method_spike_override():
    # Category-specific override for periodic spikes
    ts = pd.Series([0, 0, 0, 100, 0, 0])
    methods = {'mean': lambda ts, fh: None, 'median': lambda ts, fh: None, 'periodic_spike': lambda ts, fh: None}
    assert select_method(ts, methods, 'rent + communal', window=10, fh=7) == 'periodic_spike'


def test_select_method_low_activity_override():
    # Low activity override
    ts = pd.Series([0, 1, 0])
    methods = {'mean': lambda ts, fh: None, 'median': lambda ts, fh: None, 'periodic_spike': lambda ts, fh: None}
    assert select_method(ts, methods, 'other', window=10, fh=7) == 'periodic_spike'


def test_select_method_lowest_mae(monkeypatch):
    # Chooses method with lowest MAE from backtest
    ts = pd.Series([1, 2, 3, 4, 5])
    methods = {'mean': lambda ts, fh: None, 'median': lambda ts, fh: None}
    def fake_rb(ts_arg, methods_arg, window_arg, forecast_horizon_arg):
        return {'mean': 10, 'median': 5}
    # Monkeypatch rolling_backtest
    monkeypatch.setattr(app, 'rolling_backtest', fake_rb)
    assert select_method(ts, methods, 'other', window=10, fh=7) == 'median'
