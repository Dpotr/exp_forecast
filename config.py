"""
Configuration module for expense forecasting application.
Centralizes all file paths and application settings.
"""
import os
from pathlib import Path

class Config:
    """Configuration class for the expense forecasting application."""
    
    def __init__(self):
        # Base directory (where this config file is located)
        self.BASE_DIR = Path(__file__).parent.absolute()
        
        # Data files
        self.EXPENSES_FILE = self.BASE_DIR / 'expenses.xlsx'
        self.FORECAST_RESULTS_FILE = self.BASE_DIR / 'forecast_results.xlsx'
        self.EXPENSES_EXPORT_FILE = self.BASE_DIR / 'expenses_export.csv'
        
        # Daily payments file (configurable via environment variable)
        default_daily_path = r'C:\Users\potre\OneDrive\Documents (excel files e t.c.)\daily payments.xlsx'
        self.DAILY_PAYMENTS_FILE = os.getenv('DAILY_PAYMENTS_PATH', default_daily_path)
        
        # Forecast settings (defaults)
        self.DEFAULT_ACTIVITY_WINDOW = 70
        self.DEFAULT_FORECAST_HORIZON = 7
        self.DEFAULT_SPIKE_THRESHOLD = 30
        self.DEFAULT_MONTHLY_BUDGET = 4000
        
        # Analysis settings
        self.OUTLIER_WINDOW_DAYS = 60
        self.OUTLIER_Z_THRESHOLD = 3
        self.ANOMALY_Z_THRESHOLD = 3
        self.MIN_RECURRING_PERIODS = 3
        self.RECURRING_THRESHOLD = 0.7
        
        # Forecast method names
        self.FORECAST_METHODS = ["mean", "median", "zero", "croston", "prophet", "periodic_spike"]
        
        # Spike categories and thresholds
        self.SPIKE_CATEGORIES = {
            'school': 300,
            'rent + communal': 500,
            'car rent': 300
        }
        
    def get_expenses_path(self):
        """Get the path to the expenses file."""
        return str(self.EXPENSES_FILE)
    
    def get_daily_payments_path(self):
        """Get the path to the daily payments file."""
        return self.DAILY_PAYMENTS_FILE
    
    def get_forecast_results_path(self):
        """Get the path to the forecast results file."""
        return str(self.FORECAST_RESULTS_FILE)
    
    def file_exists(self, file_path):
        """Check if a file exists."""
        return Path(file_path).exists()

# Global config instance
config = Config()