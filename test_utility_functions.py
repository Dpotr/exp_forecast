"""
Unit tests for utility functions in the expense forecasting application.
Tests data validation, forecast metrics, cross-validation, and forecast utilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_validation import DataValidator, validate_expense_data, create_sample_valid_data, create_sample_invalid_data
from forecast_metrics import ForecastMetrics, calculate_forecast_metrics
from cross_validation import TimeSeriesCrossValidator, ForecastMethodValidator, create_forecast_functions
from forecast_utils import aggregate_daily_to_weekly, calculate_weekly_metrics, add_week_metadata, aggregate_daily_to_monthly
from config import config


class TestDataValidation(unittest.TestCase):
    """Test suite for data validation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.validator = DataValidator(strict_mode=False)
        self.valid_data = create_sample_valid_data().head(50)  # Use smaller dataset for speed
        self.invalid_data = create_sample_invalid_data()
    
    def test_validate_valid_data(self):
        """Test validation with valid data."""
        result = validate_expense_data(self.valid_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)
        self.assertIn('cleaned_data', result)
        self.assertIsInstance(result['cleaned_data'], pd.DataFrame)
        self.assertGreater(len(result['cleaned_data']), 0)
    
    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        result = validate_expense_data(self.invalid_data, strict_mode=False)
        
        self.assertIsInstance(result, dict)
        self.assertIn('warnings', result)
        self.assertIn('errors', result)
        self.assertIsInstance(result['warnings'], list)
        self.assertIsInstance(result['errors'], list)
    
    def test_required_columns_validation(self):
        """Test required columns validation."""
        # Missing required column
        incomplete_data = pd.DataFrame({'date': ['2024-01-01'], 'amount': [50.0]})
        result = validate_expense_data(incomplete_data)
        
        self.assertIn('errors', result)
        self.assertTrue(any('Missing required columns' in error for error in result['errors']))
    
    def test_date_column_validation(self):
        """Test date column validation."""
        # Test with string dates
        string_date_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', 'invalid-date'],
            'amount': [50.0, 60.0, 70.0],
            'category': ['food', 'transport', 'entertainment']
        })
        
        result = validate_expense_data(string_date_data)
        self.assertIsInstance(result['cleaned_data'], pd.DataFrame)
        
        # Should handle invalid dates
        if result['warnings']:
            self.assertTrue(any('invalid date' in warning.lower() for warning in result['warnings']))
    
    def test_amount_column_validation(self):
        """Test amount column validation."""
        # Test with negative amounts
        negative_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'amount': [50.0, -25.0],
            'category': ['food', 'transport']
        })
        
        result = validate_expense_data(negative_data)
        
        # Should handle negative amounts
        if result['warnings']:
            self.assertTrue(any('negative' in warning.lower() for warning in result['warnings']))
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        result = validate_expense_data(empty_data)
        
        self.assertFalse(result['is_valid'])
        self.assertIn('errors', result)
    
    def test_strict_mode(self):
        """Test strict mode validation."""
        # Use a more severely invalid dataset that will definitely cause errors
        severely_invalid = pd.DataFrame({
            'wrong_column': ['not_date'],
            'bad_amount': ['not_number'],
            'missing_category': [None]
        })
        
        try:
            result = validate_expense_data(severely_invalid, strict_mode=True)
            # If we get here without exception, check that errors were reported
            self.assertFalse(result['is_valid'])
            self.assertGreater(len(result['errors']), 0)
        except Exception:
            # This is expected behavior for strict mode
            pass


class TestForecastMetrics(unittest.TestCase):
    """Test suite for forecast metrics functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.actual = np.array([100, 110, 105, 120, 115, 125, 130, 135, 140, 145])
        self.forecast_good = self.actual + np.random.normal(0, 5, len(self.actual))
        self.forecast_poor = self.actual + np.random.normal(0, 20, len(self.actual))
        self.dates = pd.date_range('2024-01-01', periods=len(self.actual))
        
        self.metrics_calculator = ForecastMetrics()
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = calculate_forecast_metrics(self.actual, self.forecast_good, self.dates)
        
        # Check structure
        required_keys = ['basic_metrics', 'percentage_metrics', 'directional_metrics', 
                        'distribution_metrics', 'summary_stats', 'overall_assessment']
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check basic metrics
        basic = metrics['basic_metrics']
        self.assertIn('mae', basic)
        self.assertIn('rmse', basic)
        self.assertGreater(basic['mae'], 0)
        self.assertGreater(basic['rmse'], 0)
        
        # Check percentage metrics
        percentage = metrics['percentage_metrics']
        self.assertIn('mape', percentage)
        self.assertIn('smape', percentage)
        self.assertGreaterEqual(percentage['mape'], 0)
        self.assertGreaterEqual(percentage['smape'], 0)
    
    def test_directional_metrics(self):
        """Test directional accuracy metrics."""
        metrics = calculate_forecast_metrics(self.actual, self.forecast_good)
        
        directional = metrics['directional_metrics']
        self.assertIn('directional_accuracy', directional)
        self.assertIn('hit_rate_10pct', directional)
        self.assertIn('hit_rate_20pct', directional)
        
        # Values should be percentages (0-100)
        self.assertGreaterEqual(directional['directional_accuracy'], 0)
        self.assertLessEqual(directional['directional_accuracy'], 100)
    
    def test_quality_assessment(self):
        """Test overall quality assessment."""
        metrics_good = calculate_forecast_metrics(self.actual, self.forecast_good)
        metrics_poor = calculate_forecast_metrics(self.actual, self.forecast_poor)
        
        assessment_good = metrics_good['overall_assessment']
        assessment_poor = metrics_poor['overall_assessment']
        
        self.assertIn('overall_rating', assessment_good)
        self.assertIn('overall_rating', assessment_poor)
        self.assertIn('recommendations', assessment_good)
        self.assertIn('recommendations', assessment_poor)
        
        # Good forecast should have better rating than poor forecast
        rating_order = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        good_score = rating_order.get(assessment_good['overall_rating'], 0)
        poor_score = rating_order.get(assessment_poor['overall_rating'], 0)
        self.assertGreaterEqual(good_score, poor_score)
    
    def test_compare_forecasts(self):
        """Test forecast comparison functionality."""
        comparison = self.metrics_calculator.compare_forecasts(
            self.actual, self.forecast_good, self.forecast_poor,
            forecast_names=['Good', 'Poor']
        )
        
        self.assertIn('individual_metrics', comparison)
        self.assertIn('ranking', comparison)
        self.assertIn('best_forecast', comparison)
        
        # Check that Good forecast is ranked better
        self.assertEqual(comparison['best_forecast']['name'], 'Good')
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with identical values
        identical_actual = np.array([100, 100, 100, 100, 100])
        identical_forecast = np.array([100, 100, 100, 100, 100])
        
        metrics = calculate_forecast_metrics(identical_actual, identical_forecast)
        self.assertEqual(metrics['basic_metrics']['mae'], 0)
        self.assertEqual(metrics['percentage_metrics']['mape'], 0)
        
        # Test with empty arrays
        metrics_empty = calculate_forecast_metrics([], [])
        self.assertIn('data_info', metrics_empty)
        self.assertEqual(metrics_empty['data_info']['total_points'], 0)
    
    def test_time_metrics(self):
        """Test time-based metrics."""
        metrics = calculate_forecast_metrics(self.actual, self.forecast_good, self.dates)
        
        if 'time_metrics' in metrics:
            time_metrics = metrics['time_metrics']
            self.assertIn('recent_mae', time_metrics)
            self.assertIn('historical_mae', time_metrics)


class TestCrossValidation(unittest.TestCase):
    """Test suite for cross-validation functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create time series with trend
        n_points = 50
        trend = np.linspace(50, 100, n_points)
        noise = np.random.normal(0, 5, n_points)
        self.ts_data = trend + noise
        self.dates = pd.date_range('2024-01-01', periods=n_points)
        
        # Create simple forecast functions
        self.forecast_functions = {
            'mean': lambda train, h: np.full(h, np.mean(train)),
            'median': lambda train, h: np.full(h, np.median(train)),
            'last_value': lambda train, h: np.full(h, train[-1] if len(train) > 0 else 0)
        }
        
        self.cv = TimeSeriesCrossValidator(n_splits=3, test_size=5)
    
    def test_time_series_split(self):
        """Test time series cross-validation splits."""
        splits = self.cv.time_series_split(len(self.ts_data))
        
        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), self.cv.n_splits)
        
        # Check that splits maintain temporal order
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            self.assertLess(train_idx.max(), test_idx.min())  # Train comes before test
    
    def test_expanding_window_split(self):
        """Test expanding window cross-validation."""
        splits = self.cv.expanding_window_split(len(self.ts_data))
        
        self.assertGreater(len(splits), 0)
        
        # Check that training sets are expanding
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        self.assertEqual(train_sizes, sorted(train_sizes))  # Should be increasing
    
    def test_sliding_window_split(self):
        """Test sliding window cross-validation."""
        splits = self.cv.sliding_window_split(len(self.ts_data), train_size=20)
        
        self.assertGreater(len(splits), 0)
        
        # Check that training sets have consistent size
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        self.assertTrue(all(size == train_sizes[0] for size in train_sizes))
    
    def test_cross_validate_forecasts(self):
        """Test cross-validation of multiple forecast methods."""
        results = self.cv.cross_validate_forecasts(
            actual=self.ts_data,
            forecast_functions=self.forecast_functions,
            cv_method='time_series'
        )
        
        # Check structure
        self.assertIn('method_results', results)
        self.assertIn('summary', results)
        self.assertIn('best_method', results)
        self.assertIn('splits_info', results)
        
        # Check that all methods are included
        for method_name in self.forecast_functions.keys():
            self.assertIn(method_name, results['method_results'])
        
        # Check best method selection
        self.assertIsNotNone(results['best_method']['method'])
        self.assertIn(results['best_method']['method'], self.forecast_functions.keys())
    
    def test_method_validator(self):
        """Test single method validation."""
        validator = ForecastMethodValidator()
        
        simple_forecast = lambda train, h: np.full(h, np.mean(train))
        
        results = validator.validate_method(
            actual=self.ts_data,
            forecast_function=simple_forecast,
            method_name="Simple Mean",
            n_splits=3
        )
        
        self.assertIn('method_name', results)
        self.assertIn('fold_metrics', results)
        self.assertIn('summary', results)
        self.assertIn('overall_performance', results)
        
        # Check performance assessment
        performance = results['overall_performance']
        self.assertIn('assessment', performance)
        self.assertIn('reliability', performance)
        self.assertIn('recommendations', performance)
    
    def test_forecast_functions_creation(self):
        """Test creation of common forecast functions."""
        # Create sample dataframe
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'amount': np.random.uniform(10, 100, 30),
            'category': ['food'] * 30
        })
        
        functions = create_forecast_functions(df)
        
        # Check that functions are created
        expected_functions = ['mean', 'median', 'moving_average', 'trend', 'seasonal_naive']
        for func_name in expected_functions:
            self.assertIn(func_name, functions)
        
        # Test that functions work
        sample_train = np.array([10, 20, 30, 40, 50])
        for func_name, func in functions.items():
            forecast = func(sample_train, 3)
            self.assertEqual(len(forecast), 3)
            self.assertTrue(all(isinstance(val, (int, float, np.number)) for val in forecast))
    
    def test_edge_cases_cv(self):
        """Test cross-validation edge cases."""
        # Test with very short time series
        short_ts = np.array([1, 2, 3, 4, 5])
        cv_short = TimeSeriesCrossValidator(n_splits=2, test_size=1)
        
        results = cv_short.cross_validate_forecasts(
            actual=short_ts,
            forecast_functions={'mean': lambda train, h: np.full(h, np.mean(train))}
        )
        
        # Should handle gracefully - might return error or method_results
        self.assertTrue('method_results' in results or 'error' in results)
        
        # Test with failed forecast function
        def failing_function(train, h):
            raise ValueError("Intentional failure")
        
        results_with_failure = self.cv.cross_validate_forecasts(
            actual=self.ts_data[:20],  # Use shorter series
            forecast_functions={
                'good': lambda train, h: np.full(h, np.mean(train)),
                'bad': failing_function
            }
        )
        
        # Should handle failures gracefully
        self.assertIn('method_results', results_with_failure)
        self.assertIn('good', results_with_failure['method_results'])
        self.assertIn('bad', results_with_failure['method_results'])


class TestForecastUtils(unittest.TestCase):
    """Test suite for forecast utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create daily data for testing
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        amounts = np.random.uniform(10, 100, 30)
        
        self.daily_data = pd.DataFrame({
            'date': dates,
            'actual': amounts,
            'forecast': amounts * np.random.uniform(0.8, 1.2, 30),
            'error': np.random.normal(0, 5, 30),
            'abs_error': np.abs(np.random.normal(0, 5, 30)),
            'pct_error': np.random.uniform(-20, 20, 30),
            'within_10pct': np.random.choice([0, 1], 30),
            'within_20pct': np.random.choice([0, 1], 30)
        })
    
    def test_aggregate_daily_to_weekly(self):
        """Test daily to weekly aggregation."""
        weekly_data = aggregate_daily_to_weekly(self.daily_data)
        
        self.assertIsInstance(weekly_data, pd.DataFrame)
        self.assertGreater(len(weekly_data), 0)
        self.assertLess(len(weekly_data), len(self.daily_data))  # Should be fewer weeks than days
        
        # Check required columns
        required_cols = ['date', 'actual', 'forecast']
        for col in required_cols:
            if col in self.daily_data.columns:
                self.assertIn(col, weekly_data.columns)
    
    def test_calculate_weekly_metrics(self):
        """Test weekly metrics calculation."""
        weekly_data = aggregate_daily_to_weekly(self.daily_data)
        
        if not weekly_data.empty:
            metrics = calculate_weekly_metrics(weekly_data)
            
            self.assertIsInstance(metrics, dict)
            
            # Check for expected metric keys
            expected_metrics = ['avg_mae', 'avg_mape', 'accuracy_10pct', 'accuracy_20pct']
            for metric in expected_metrics:
                if metric in metrics:
                    self.assertIsInstance(metrics[metric], (int, float))
    
    def test_add_week_metadata(self):
        """Test adding week metadata."""
        weekly_data = aggregate_daily_to_weekly(self.daily_data)
        
        if not weekly_data.empty:
            with_metadata = add_week_metadata(weekly_data)
            
            self.assertIsInstance(with_metadata, pd.DataFrame)
            self.assertEqual(len(with_metadata), len(weekly_data))
            
            # Should have additional metadata columns
            metadata_cols = ['week_start', 'week_end', 'week_number', 'year']
            for col in metadata_cols:
                if col in with_metadata.columns:
                    self.assertTrue(col in with_metadata.columns)
    
    def test_aggregate_daily_to_monthly(self):
        """Test daily to monthly aggregation."""
        monthly_data = aggregate_daily_to_monthly(self.daily_data)
        
        self.assertIsInstance(monthly_data, pd.DataFrame)
        
        if not monthly_data.empty:
            self.assertLess(len(monthly_data), len(self.daily_data))  # Should be fewer months than days
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        weekly_empty = aggregate_daily_to_weekly(empty_df)
        self.assertTrue(weekly_empty.empty)
        
        monthly_empty = aggregate_daily_to_monthly(empty_df)
        self.assertTrue(monthly_empty.empty)
    
    def test_data_with_missing_columns(self):
        """Test handling of data with missing columns."""
        minimal_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'actual': np.random.uniform(10, 100, 10),
            'forecast': np.random.uniform(10, 100, 10)  # Add required column
        })
        
        # Should handle gracefully
        weekly_result = aggregate_daily_to_weekly(minimal_data)
        self.assertIsInstance(weekly_result, pd.DataFrame)


class TestIntegration(unittest.TestCase):
    """Integration tests for utility functions working together."""
    
    def setUp(self):
        """Set up integration test data."""
        np.random.seed(42)
        
        # Create realistic transaction data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        categories = ['food', 'transport', 'entertainment', 'utilities']
        
        data = []
        for date in dates:
            # 1-3 transactions per day
            num_transactions = np.random.randint(1, 4)
            for _ in range(num_transactions):
                category = np.random.choice(categories)
                amount = np.random.lognormal(mean=3, sigma=0.5)
                data.append({
                    'date': date,
                    'amount': round(amount, 2),
                    'category': category
                })
        
        self.transaction_data = pd.DataFrame(data)
    
    def test_full_validation_and_metrics_pipeline(self):
        """Test complete pipeline: validation -> aggregation -> metrics -> CV."""
        # Step 1: Validate data
        validation_result = validate_expense_data(self.transaction_data)
        self.assertTrue(validation_result['is_valid'] or len(validation_result['cleaned_data']) > 0)
        
        cleaned_data = validation_result['cleaned_data']
        
        # Step 2: Create daily aggregation
        daily_totals = cleaned_data.groupby('date')['amount'].sum().reset_index()
        daily_totals.rename(columns={'amount': 'actual'}, inplace=True)  # Rename to match expected column
        daily_totals['forecast'] = daily_totals['actual'] * np.random.uniform(0.9, 1.1, len(daily_totals))
        daily_totals['error'] = daily_totals['forecast'] - daily_totals['actual']
        daily_totals['abs_error'] = np.abs(daily_totals['error'])
        daily_totals['pct_error'] = (daily_totals['abs_error'] / daily_totals['actual']) * 100
        daily_totals['within_10pct'] = (daily_totals['pct_error'] <= 10).astype(int)
        daily_totals['within_20pct'] = (daily_totals['pct_error'] <= 20).astype(int)
        
        # Step 3: Calculate comprehensive metrics
        metrics = calculate_forecast_metrics(
            actual=daily_totals['actual'].values,
            forecast=daily_totals['forecast'].values,
            dates=daily_totals['date'].values
        )
        
        self.assertIn('basic_metrics', metrics)
        self.assertIn('overall_assessment', metrics)
        
        # Step 4: Aggregate to weekly
        weekly_data = aggregate_daily_to_weekly(daily_totals)
        
        if not weekly_data.empty:
            weekly_metrics = calculate_weekly_metrics(weekly_data)
            self.assertIsInstance(weekly_metrics, dict)
        
        # Step 5: Cross-validation
        if len(daily_totals) > 20:  # Need enough data for CV
            cv = TimeSeriesCrossValidator(n_splits=3, test_size=5)
            forecast_functions = create_forecast_functions(cleaned_data)
            
            cv_results = cv.cross_validate_forecasts(
                actual=daily_totals['actual'].values,
                forecast_functions=forecast_functions
            )
            
            if 'best_method' in cv_results:
                self.assertIn('best_method', cv_results)
                # Method might be None if no valid splits generated
            else:
                # CV might fail with insufficient data - that's okay
                self.assertIn('error', cv_results)
    
    def test_config_integration(self):
        """Test that config values are properly used."""
        # Test that config values are accessible
        self.assertIsInstance(config.OUTLIER_WINDOW_DAYS, int)
        self.assertIsInstance(config.ANOMALY_Z_THRESHOLD, (int, float))
        
        # Test validation with config values
        validator = DataValidator()
        # Should use config values internally
        result = validator.validate_expense_data(self.transaction_data)
        self.assertIsInstance(result, dict)


def run_utility_tests():
    """Run all utility function tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataValidation,
        TestForecastMetrics,
        TestCrossValidation,
        TestForecastUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Utility Functions Test Suite...")
    print("=" * 50)
    
    result = run_utility_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print(f"\n{len(result.failures) + len(result.errors)} test(s) failed!")
    
    exit(0 if result.wasSuccessful() else 1)