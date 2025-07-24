import pandas as pd
import matplotlib.pyplot as plt
from config import config

def plot_weekly_seasonality(expenses_path=None, category=None):
    if expenses_path is None:
        expenses_path = config.get_expenses_path()
    df = pd.read_excel(expenses_path)
    df['date'] = pd.to_datetime(df['date'])
    if category:
        df = df[df['category'].str.lower().str.contains(category.lower())]
    df['weekday'] = df['date'].dt.day_name()
    weekly = df.groupby('weekday')['amount'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekly.plot(kind='bar', title=f'Average Spending by Weekday{f" ({category})" if category else ""}')
    plt.ylabel('Average Amount')
    plt.tight_layout()
    plt.show()

def plot_monthly_seasonality(expenses_path=None, category=None):
    if expenses_path is None:
        expenses_path = config.get_expenses_path()
    df = pd.read_excel(expenses_path)
    df['date'] = pd.to_datetime(df['date'])
    if category:
        df = df[df['category'].str.lower().str.contains(category.lower())]
    df['dom'] = df['date'].dt.day
    monthly = df.groupby('dom')['amount'].mean()
    monthly.plot(kind='bar', title=f'Average Spending by Day of Month{f" ({category})" if category else ""}', figsize=(12,4))
    plt.ylabel('Average Amount')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_weekly_seasonality()
    plot_monthly_seasonality()
