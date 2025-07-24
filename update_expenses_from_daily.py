import pandas as pd
import os
import re
from config import config

# Paths from configuration
expenses_path = config.get_expenses_path()
daily_path = config.get_daily_payments_path()

# Load daily payments from the correct sheet
if not config.file_exists(daily_path):
    print(f"Warning: Daily payments file not found at {daily_path}")
    print("Set the DAILY_PAYMENTS_PATH environment variable to the correct path")
    exit(1)
df_daily = pd.read_excel(daily_path, sheet_name='data')

# Try to robustly find columns
col_map = {}
for col in df_daily.columns:
    cname = str(col).strip().lower()
    if cname == 'date':
        col_map['date'] = col
    elif cname == 'category':
        col_map['category'] = col
    elif 'eur' in cname:
        col_map['amount'] = col

if set(col_map.keys()) != {'date','category','amount'}:
    raise ValueError(f"Could not auto-detect columns. Found mapping: {col_map}. Columns: {list(df_daily.columns)}")

# Rename and keep only needed columns
df_daily = df_daily.rename(columns={col_map['date']:'date', col_map['category']:'category', col_map['amount']:'amount'})
df_daily = df_daily[['date','category','amount']]

# Normalize fields
def clean_amount(val):
    if pd.isnull(val):
        return 0.0
    val = re.sub(r'[^\d\.-]', '', str(val))
    try:
        return round(float(val), 2)
    except:
        return 0.0

def normalize(df):
    df = df.copy()
    df['amount'] = df['amount'].apply(clean_amount)
    df['date'] = pd.to_datetime(df['date']).dt.date  # only date part
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    return df

df_daily = normalize(df_daily)

# Remove rows where date/category is NaN/empty or amount is zero/NaN
valid_mask = df_daily['date'].notna() & df_daily['category'].notna() & (df_daily['category'] != '') & df_daily['amount'].notna() & (df_daily['amount'] != 0)
df_daily = df_daily[valid_mask]

# Load expenses file
if os.path.exists(expenses_path):
    df_exp = pd.read_excel(expenses_path)
    if not df_exp.empty:
        df_exp = normalize(df_exp)
        valid_exp_mask = df_exp['date'].notna() & df_exp['category'].notna() & (df_exp['category'] != '') & df_exp['amount'].notna() & (df_exp['amount'] != 0)
        df_exp = df_exp[valid_exp_mask]
else:
    df_exp = pd.DataFrame(columns=['date','category','amount'])

# Find truly new records (date, category, amount as float, rounded, date only)
merge_cols = ['date','category','amount']
merged = df_daily.merge(df_exp[merge_cols], on=merge_cols, how='left', indicator=True)
missing = merged[merged['_merge'] == 'left_only'][merge_cols]

# Append missing records to expenses, drop duplicates just in case
df_updated = pd.concat([df_exp, missing], ignore_index=True)
df_updated = df_updated.drop_duplicates(subset=merge_cols, keep='first')

# Remove rows with any NaN in key columns
df_updated = df_updated.dropna(subset=merge_cols)
# Sort by date (old to new)
df_updated = df_updated.sort_values(by='date', ascending=True).reset_index(drop=True)

# Save updated expenses file
if not missing.empty:
    df_updated.to_excel(expenses_path, index=False)
    print(f"Added {len(missing)} new records to expenses.xlsx.")
else:
    print("No new records to add.")

# --- TEST ---
if __name__ == "__main__":
    # Test: run again, should add 0 records
    df_exp2 = pd.read_excel(expenses_path)
    if not df_exp2.empty:
        df_exp2 = normalize(df_exp2)
        merged2 = df_daily.merge(df_exp2[merge_cols], on=merge_cols, how='left', indicator=True)
        left_only = merged2[merged2['_merge'] == 'left_only']
        if not left_only.empty:
            print('DIAGNOSTIC: The following records from daily were NOT found in expenses after update:')
            print(left_only.head(10))
            print('Types:')
            print(left_only.dtypes)
            print('Sample from expenses:')
            print(df_exp2.head(10))
            raise AssertionError("Test failed: Duplicate records were added or normalization mismatch.")
        print("Test passed: No duplicates, update is idempotent.")
