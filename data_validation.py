"""
Data validation module for expense forecasting application.
Provides comprehensive validation functions for input data integrity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from config import config


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """Comprehensive data validation class for expense data."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.
        
        Args:
            strict_mode: If True, raises exceptions for validation failures.
                        If False, logs warnings and attempts to fix issues.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
        self.warnings = []
        self.errors = []
    
    def validate_expense_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of expense data.
        
        Args:
            df: DataFrame with expense data
            
        Returns:
            Dictionary with validation results and cleaned data
        """
        if df is None or df.empty:
            return self._handle_validation_issue(
                "empty_data", "Input DataFrame is empty or None", df
            )
        
        results = {
            'original_shape': df.shape,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'cleaned_data': df.copy(),
            'validation_summary': {}
        }
        
        # Run all validation checks
        validation_checks = [
            self._validate_required_columns,
            self._validate_date_column,
            self._validate_amount_column,
            self._validate_category_column,
            self._validate_data_types,
            self._validate_date_range,
            self._validate_amount_range,
            self._validate_duplicates,
            self._validate_missing_values,
            self._validate_data_consistency
        ]
        
        cleaned_df = df.copy()
        
        for check in validation_checks:
            try:
                check_result = check(cleaned_df)
                if check_result:
                    results['validation_summary'][check.__name__] = check_result
                    if 'cleaned_data' in check_result:
                        cleaned_df = check_result['cleaned_data']
                    if 'warnings' in check_result:
                        results['warnings'].extend(check_result['warnings'])
                    if 'errors' in check_result:
                        results['errors'].extend(check_result['errors'])
                        results['is_valid'] = False
            except Exception as e:
                error_msg = f"Validation check {check.__name__} failed: {str(e)}"
                results['errors'].append(error_msg)
                results['is_valid'] = False
                if self.strict_mode:
                    raise ValidationError(error_msg)
        
        results['cleaned_data'] = cleaned_df
        results['final_shape'] = cleaned_df.shape
        
        return results
    
    def _validate_required_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that required columns are present."""
        required_columns = ['date', 'amount', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            return {'errors': [error_msg], 'status': 'failed'}
        
        return {'status': 'passed', 'required_columns': required_columns}
    
    def _validate_date_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and clean date column."""
        if 'date' not in df.columns:
            return {'errors': ['Date column missing'], 'status': 'failed'}
        
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        cleaned_df = df.copy()
        
        # Count initial null dates
        initial_null_dates = cleaned_df['date'].isnull().sum()
        
        # Try to convert to datetime
        try:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
            # Count how many dates became NaT (Not a Time)
            final_null_dates = cleaned_df['date'].isnull().sum()
            invalid_dates = final_null_dates - initial_null_dates
            if invalid_dates > 0:
                warning_msg = f"Converted {invalid_dates} invalid date strings to NaT"
                result['warnings'].append(warning_msg)
        except Exception as e:
            error_msg = f"Cannot convert date column to datetime: {str(e)}"
            result['errors'].append(error_msg)
            return result
        
        # Check for null dates (update count after conversion)
        total_null_dates = cleaned_df['date'].isnull().sum()
        if total_null_dates > 0:
            warning_msg = f"Found {total_null_dates} null/invalid dates"
            result['warnings'].append(warning_msg)
            if not self.strict_mode:
                cleaned_df = cleaned_df.dropna(subset=['date'])
                result['removed_null_dates'] = total_null_dates
        
        # Check date range reasonableness
        if not cleaned_df.empty:
            min_date = cleaned_df['date'].min()
            max_date = cleaned_df['date'].max()
            today = datetime.now()
            
            # Check for future dates
            future_dates = (cleaned_df['date'] > today).sum()
            if future_dates > 0:
                warning_msg = f"Found {future_dates} future dates"
                result['warnings'].append(warning_msg)
            
            # Check for very old dates (more than 10 years ago)
            ten_years_ago = today - timedelta(days=10*365)
            old_dates = (cleaned_df['date'] < ten_years_ago).sum()
            if old_dates > 0:
                warning_msg = f"Found {old_dates} dates older than 10 years"
                result['warnings'].append(warning_msg)
            
            result['date_range'] = {'min': min_date, 'max': max_date}
        
        result['cleaned_data'] = cleaned_df
        return result
    
    def _validate_amount_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and clean amount column."""
        if 'amount' not in df.columns:
            return {'errors': ['Amount column missing'], 'status': 'failed'}
        
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        cleaned_df = df.copy()
        
        # Try to convert to numeric
        try:
            cleaned_df['amount'] = pd.to_numeric(cleaned_df['amount'], errors='coerce')
        except Exception as e:
            error_msg = f"Cannot convert amount column to numeric: {str(e)}"
            result['errors'].append(error_msg)
            return result
        
        # Check for null amounts
        null_amounts = cleaned_df['amount'].isnull().sum()
        if null_amounts > 0:
            warning_msg = f"Found {null_amounts} null/invalid amounts"
            result['warnings'].append(warning_msg)
            if not self.strict_mode:
                cleaned_df = cleaned_df.dropna(subset=['amount'])
                result['removed_null_amounts'] = null_amounts
        
        if not cleaned_df.empty:
            # Check for negative amounts
            negative_amounts = (cleaned_df['amount'] < 0).sum()
            if negative_amounts > 0:
                warning_msg = f"Found {negative_amounts} negative amounts"
                result['warnings'].append(warning_msg)
                if not self.strict_mode:
                    cleaned_df['amount'] = cleaned_df['amount'].abs()
                    result['converted_negative_amounts'] = negative_amounts
            
            # Check for zero amounts
            zero_amounts = (cleaned_df['amount'] == 0).sum()
            if zero_amounts > 0:
                warning_msg = f"Found {zero_amounts} zero amounts"
                result['warnings'].append(warning_msg)
            
            # Check for extremely large amounts (potential data entry errors)
            amount_stats = cleaned_df['amount'].describe()
            q99 = cleaned_df['amount'].quantile(0.99)
            extreme_amounts = (cleaned_df['amount'] > q99 * 10).sum()
            if extreme_amounts > 0:
                warning_msg = f"Found {extreme_amounts} extremely large amounts (>10x 99th percentile)"
                result['warnings'].append(warning_msg)
            
            result['amount_stats'] = {
                'min': amount_stats['min'],
                'max': amount_stats['max'],
                'mean': amount_stats['mean'],
                'median': amount_stats['50%'],
                'q99': q99
            }
        
        result['cleaned_data'] = cleaned_df
        return result
    
    def _validate_category_column(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and clean category column."""
        if 'category' not in df.columns:
            return {'errors': ['Category column missing'], 'status': 'failed'}
        
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        cleaned_df = df.copy()
        
        # Convert to string and clean
        cleaned_df['category'] = cleaned_df['category'].astype(str).str.strip().str.lower()
        
        # Check for null/empty categories
        null_categories = cleaned_df['category'].isin(['', 'nan', 'none', 'null']).sum()
        if null_categories > 0:
            warning_msg = f"Found {null_categories} null/empty categories"
            result['warnings'].append(warning_msg)
            if not self.strict_mode:
                cleaned_df.loc[cleaned_df['category'].isin(['', 'nan', 'none', 'null']), 'category'] = 'uncategorized'
                result['fixed_null_categories'] = null_categories
        
        # Check for very short category names (potential data entry errors)
        short_categories = (cleaned_df['category'].str.len() < 2).sum()
        if short_categories > 0:
            warning_msg = f"Found {short_categories} very short category names"
            result['warnings'].append(warning_msg)
        
        # Category statistics
        category_counts = cleaned_df['category'].value_counts()
        result['category_stats'] = {
            'unique_categories': len(category_counts),
            'most_common': category_counts.head(5).to_dict(),
            'single_occurrence': (category_counts == 1).sum()
        }
        
        result['cleaned_data'] = cleaned_df
        return result
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types are appropriate."""
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        expected_types = {
            'date': 'datetime64[ns]',
            'amount': ['float64', 'int64'],
            'category': 'object'
        }
        
        type_issues = []
        for col, expected in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if isinstance(expected, list):
                    if actual_type not in expected:
                        type_issues.append(f"{col}: expected {expected}, got {actual_type}")
                else:
                    if actual_type != expected:
                        type_issues.append(f"{col}: expected {expected}, got {actual_type}")
        
        if type_issues:
            result['warnings'].extend(type_issues)
        
        result['data_types'] = {col: str(df[col].dtype) for col in df.columns}
        return result
    
    def _validate_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate date range and chronological order."""
        if 'date' not in df.columns or df.empty:
            return {'status': 'skipped'}
        
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        # Check for chronological gaps
        df_sorted = df.sort_values('date')
        date_diffs = df_sorted['date'].diff().dt.days.dropna()
        
        if not date_diffs.empty:
            max_gap = date_diffs.max()
            large_gaps = (date_diffs > 30).sum()  # Gaps larger than 30 days
            
            if large_gaps > 0:
                warning_msg = f"Found {large_gaps} date gaps larger than 30 days (max gap: {max_gap} days)"
                result['warnings'].append(warning_msg)
            
            result['date_gaps'] = {
                'max_gap_days': max_gap,
                'large_gaps_count': large_gaps,
                'avg_gap_days': date_diffs.mean()
            }
        
        return result
    
    def _validate_amount_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate amount ranges for reasonableness."""
        if 'amount' not in df.columns or df.empty:
            return {'status': 'skipped'}
        
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        # Statistical analysis of amounts
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Potential outliers using IQR method
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        potential_outliers = ((df['amount'] < lower_bound) | (df['amount'] > upper_bound)).sum()
        if potential_outliers > 0:
            warning_msg = f"Found {potential_outliers} potential amount outliers"
            result['warnings'].append(warning_msg)
        
        result['amount_distribution'] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'potential_outliers': potential_outliers,
            'outlier_bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
        
        return result
    
    def _validate_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records."""
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        # Exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            warning_msg = f"Found {exact_duplicates} exact duplicate records"
            result['warnings'].append(warning_msg)
        
        # Near duplicates (same date, category, similar amount)
        if 'date' in df.columns and 'category' in df.columns and 'amount' in df.columns:
            df_grouped = df.groupby(['date', 'category'])['amount'].apply(list).reset_index()
            potential_duplicates = 0
            
            for _, row in df_grouped.iterrows():
                amounts = row['amount']
                if len(amounts) > 1:
                    # Check if amounts are very similar (within 1% or $1)
                    for i, amt1 in enumerate(amounts):
                        for amt2 in amounts[i+1:]:
                            if abs(amt1 - amt2) <= max(0.01 * max(amt1, amt2), 1.0):
                                potential_duplicates += 1
            
            if potential_duplicates > 0:
                warning_msg = f"Found {potential_duplicates} potential duplicate transactions"
                result['warnings'].append(warning_msg)
        
        result['duplicate_stats'] = {
            'exact_duplicates': exact_duplicates,
            'potential_duplicates': potential_duplicates if 'potential_duplicates' in locals() else 0
        }
        
        return result
    
    def _validate_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values across all columns."""
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            
            missing_stats[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_count > 0:
                warning_msg = f"Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)"
                result['warnings'].append(warning_msg)
        
        result['missing_values'] = missing_stats
        return result
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for data consistency issues."""
        result = {'warnings': [], 'errors': [], 'status': 'passed'}
        
        if df.empty:
            return result
        
        # Check for consistent category naming
        if 'category' in df.columns:
            category_variations = {}
            categories = df['category'].value_counts()
            
            for cat in categories.index:
                similar_cats = [c for c in categories.index 
                              if c != cat and (cat in c or c in cat)]
                if similar_cats:
                    category_variations[cat] = similar_cats
            
            if category_variations:
                warning_msg = f"Found potential category naming inconsistencies: {len(category_variations)} cases"
                result['warnings'].append(warning_msg)
                result['category_variations'] = category_variations
        
        # Check for unusual spending patterns
        if 'date' in df.columns and 'amount' in df.columns:
            daily_totals = df.groupby('date')['amount'].sum()
            if len(daily_totals) > 7:  # Need at least a week of data
                daily_mean = daily_totals.mean()
                daily_std = daily_totals.std()
                
                unusual_days = ((daily_totals - daily_mean).abs() > 3 * daily_std).sum()
                if unusual_days > 0:
                    warning_msg = f"Found {unusual_days} days with unusual spending patterns"
                    result['warnings'].append(warning_msg)
                
                result['spending_patterns'] = {
                    'daily_mean': daily_mean,
                    'daily_std': daily_std,
                    'unusual_days': unusual_days
                }
        
        return result
    
    def _handle_validation_issue(self, issue_type: str, message: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle validation issues based on strict mode."""
        if self.strict_mode:
            raise ValidationError(f"{issue_type}: {message}")
        else:
            return {
                'is_valid': False,
                'errors': [message],
                'warnings': [],
                'cleaned_data': df if df is not None else pd.DataFrame(),
                'validation_summary': {issue_type: {'status': 'failed', 'message': message}}
            }


def validate_expense_data(df: pd.DataFrame, strict_mode: bool = False) -> Dict[str, Any]:
    """
    Convenience function for validating expense data.
    
    Args:
        df: DataFrame with expense data
        strict_mode: If True, raises exceptions for validation failures
        
    Returns:
        Dictionary with validation results and cleaned data
    """
    validator = DataValidator(strict_mode=strict_mode)
    return validator.validate_expense_data(df)


def create_sample_valid_data() -> pd.DataFrame:
    """Create a sample of valid expense data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    categories = ['food', 'transport', 'entertainment', 'utilities', 'shopping']
    
    data = []
    for date in dates:
        # 1-3 transactions per day
        num_transactions = np.random.randint(1, 4)
        for _ in range(num_transactions):
            category = np.random.choice(categories)
            amount = np.random.lognormal(mean=3, sigma=0.5)  # Realistic expense distribution
            
            data.append({
                'date': date,
                'amount': round(amount, 2),
                'category': category
            })
    
    return pd.DataFrame(data)


def create_sample_invalid_data() -> pd.DataFrame:
    """Create a sample of invalid expense data for testing validation."""
    data = [
        {'date': '2024-01-01', 'amount': 50.0, 'category': 'food'},
        {'date': 'invalid-date', 'amount': 30.0, 'category': 'transport'},
        {'date': '2024-01-03', 'amount': -25.0, 'category': 'entertainment'},
        {'date': '2024-01-04', 'amount': 'invalid-amount', 'category': 'utilities'},
        {'date': '2024-01-05', 'amount': 0.0, 'category': ''},
        {'date': '2024-01-06', 'amount': 1000000.0, 'category': 'food'},  # Extreme amount
        {'date': None, 'amount': 40.0, 'category': 'transport'},
        {'date': '1900-01-01', 'amount': 20.0, 'category': 'old'},  # Very old date
        {'date': '2030-01-01', 'amount': 60.0, 'category': 'future'},  # Future date
    ]
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Example usage and testing
    print("Testing Data Validation Module")
    print("=" * 40)
    
    # Test with valid data
    print("\n1. Testing with valid data:")
    valid_data = create_sample_valid_data()
    print(f"Created {len(valid_data)} valid records")
    
    validation_result = validate_expense_data(valid_data)
    print(f"Validation result: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
    print(f"Warnings: {len(validation_result['warnings'])}")
    print(f"Errors: {len(validation_result['errors'])}")
    
    # Test with invalid data
    print("\n2. Testing with invalid data:")
    invalid_data = create_sample_invalid_data()
    print(f"Created {len(invalid_data)} potentially invalid records")
    
    validation_result = validate_expense_data(invalid_data, strict_mode=False)
    print(f"Validation result: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
    print(f"Warnings: {len(validation_result['warnings'])}")
    print(f"Errors: {len(validation_result['errors'])}")
    print(f"Original shape: {validation_result['original_shape']}")
    print(f"Final shape: {validation_result['final_shape']}")
    
    if validation_result['warnings']:
        print("\nWarnings:")
        for warning in validation_result['warnings'][:5]:  # Show first 5
            print(f"  - {warning}")
    
    if validation_result['errors']:
        print("\nErrors:")
        for error in validation_result['errors']:
            print(f"  - {error}")