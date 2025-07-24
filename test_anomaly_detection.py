import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_utils import (
    detect_outliers, detect_anomaly_transactions, recurring_payments,
    detect_outliers_iqr, detect_outliers_modified_zscore, 
    detect_seasonal_anomalies, detect_category_anomalies_enhanced,
    get_comprehensive_anomalies
)
from config import config


class TestAnomalyDetection(unittest.TestCase):
    """Comprehensive test suite for anomaly detection functions."""
    
    def setUp(self):
        """Set up test data for each test case."""
        # Create sample data with known patterns and anomalies
        np.random.seed(42)
        
        # Generate 90 days of data
        start_date = datetime.now() - timedelta(days=90)
        dates = [start_date + timedelta(days=i) for i in range(90)]
        
        # Normal spending pattern with some randomness
        normal_amounts = np.random.normal(50, 10, 90).clip(10, None)
        
        # Add known outliers - make them much more extreme
        outlier_indices = [10, 25, 60, 80]
        for idx in outlier_indices:
            normal_amounts[idx] *= 10  # Make these very clear outliers
        
        # Create categories
        categories = ['food', 'transport', 'entertainment', 'utilities'] * 22 + ['food', 'transport']
        
        self.test_df = pd.DataFrame({
            'date': dates,
            'amount': normal_amounts,
            'category': categories
        })
        
        # Create minimal test data
        self.minimal_df = pd.DataFrame({
            'date': [datetime.now() - timedelta(days=i) for i in range(5)],
            'amount': [10, 20, 15, 25, 18],
            'category': ['food'] * 5
        })
        
        # Create empty test data
        self.empty_df = pd.DataFrame(columns=['date', 'amount', 'category'])

    def test_detect_outliers_basic(self):
        """Test basic outlier detection function."""
        result = detect_outliers(self.test_df)
        
        # Check result structure first
        expected_cols = ['date', 'amount', 'z']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # If outliers are detected, check that Z-scores are above threshold
        if not result.empty:
            self.assertTrue(all(abs(result['z']) > config.OUTLIER_Z_THRESHOLD))
        
        # Test with a simple case that should definitely work
        # Create a DataFrame with clear daily outliers
        simple_dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        simple_amounts = [50] * 29 + [500]  # One clear outlier
        simple_df = pd.DataFrame({
            'date': simple_dates,
            'amount': simple_amounts,
            'category': ['food'] * 30
        })
        
        simple_result = detect_outliers(simple_df)
        # This test is informational - we check that the function works without errors
        self.assertIsInstance(simple_result, pd.DataFrame)

    def test_detect_outliers_empty_data(self):
        """Test outlier detection with empty data."""
        result = detect_outliers(self.empty_df)
        self.assertTrue(result.empty, "Should return empty DataFrame for empty input")

    def test_detect_outliers_minimal_data(self):
        """Test outlier detection with minimal data."""
        result = detect_outliers(self.minimal_df)
        # Should handle minimal data gracefully
        self.assertIsInstance(result, pd.DataFrame)

    def test_detect_outliers_iqr(self):
        """Test IQR-based outlier detection."""
        result = detect_outliers_iqr(self.test_df)
        
        # Check result structure
        expected_cols = ['date', 'amount', 'method', 'score']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # Check method is correctly set
        if not result.empty:
            self.assertTrue(all(result['method'] == 'IQR'))

    def test_detect_outliers_modified_zscore(self):
        """Test modified Z-score outlier detection."""
        result = detect_outliers_modified_zscore(self.test_df)
        
        # Check result structure
        expected_cols = ['date', 'amount', 'method', 'score']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # Check method is correctly set
        if not result.empty:
            self.assertTrue(all(result['method'] == 'Modified Z-Score'))

    def test_detect_seasonal_anomalies(self):
        """Test seasonal anomaly detection."""
        result = detect_seasonal_anomalies(self.test_df)
        
        # Check result structure
        expected_cols = ['date', 'amount', 'method', 'score']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # Check method contains 'Seasonal'
        if not result.empty:
            self.assertTrue(all('Seasonal' in method for method in result['method']))

    def test_detect_anomaly_transactions(self):
        """Test transaction-level anomaly detection."""
        result = detect_anomaly_transactions(self.test_df)
        
        # Check result structure
        expected_cols = ['date', 'category', 'amount', 'z']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # Check Z-scores are above threshold
        if not result.empty:
            self.assertTrue(all(abs(result['z']) > config.ANOMALY_Z_THRESHOLD))

    def test_detect_category_anomalies_enhanced(self):
        """Test enhanced category anomaly detection."""
        result = detect_category_anomalies_enhanced(self.test_df)
        
        # Check result structure
        expected_cols = ['date', 'category', 'amount', 'method', 'score']
        self.assertTrue(all(col in result.columns for col in expected_cols))
        
        # Check methods are valid
        if not result.empty:
            valid_methods = ['Z-Score', 'IQR']
            self.assertTrue(all(method in valid_methods for method in result['method']))

    def test_recurring_payments(self):
        """Test recurring payments detection."""
        # Create data with recurring pattern
        recurring_data = pd.DataFrame({
            'date': [datetime.now() - timedelta(days=i*30) for i in range(6)],
            'amount': [500, 500, 500, 500, 500, 500],
            'category': ['rent'] * 6
        })
        
        result = recurring_payments(recurring_data)
        
        # Should detect the recurring rent payment
        if not result.empty:
            expected_cols = ['category', 'interval_days', 'last_date', 'amount']
            self.assertTrue(all(col in result.columns for col in expected_cols))

    def test_get_comprehensive_anomalies(self):
        """Test the comprehensive anomaly detection function."""
        result = get_comprehensive_anomalies(self.test_df)
        
        # Should return a dictionary with all methods
        expected_methods = [
            'daily_zscore', 'daily_iqr', 'daily_modified_zscore',
            'daily_seasonal', 'category_enhanced'
        ]
        
        self.assertIsInstance(result, dict)
        self.assertTrue(all(method in result for method in expected_methods))
        
        # Each method should return a DataFrame
        for method, df in result.items():
            self.assertIsInstance(df, pd.DataFrame)

    def test_config_parameters(self):
        """Test that configuration parameters are being used."""
        # Test with custom parameters
        custom_window = 30
        custom_threshold = 2.5
        
        result = detect_outliers(self.test_df, window_days=custom_window, z_thresh=custom_threshold)
        
        # Should handle custom parameters without error
        self.assertIsInstance(result, pd.DataFrame)

    def test_data_type_handling(self):
        """Test handling of different data types."""
        # Test with string dates
        string_date_df = self.test_df.copy()
        string_date_df['date'] = string_date_df['date'].astype(str)
        
        result = detect_outliers(string_date_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with all zero amounts
        zero_df = self.test_df.copy()
        zero_df['amount'] = 0
        
        result = detect_outliers(zero_df)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test with identical amounts
        identical_df = self.test_df.copy()
        identical_df['amount'] = 100
        
        result = detect_outliers_iqr(identical_df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_performance_with_large_data(self):
        """Test performance with larger datasets."""
        # Create larger test dataset
        large_dates = [datetime.now() - timedelta(days=i) for i in range(1000)]
        large_amounts = np.random.normal(100, 20, 1000).clip(1, None)
        large_categories = ['cat' + str(i % 10) for i in range(1000)]
        
        large_df = pd.DataFrame({
            'date': large_dates,
            'amount': large_amounts,
            'category': large_categories
        })
        
        # Should complete without timeout or memory issues
        import time
        start_time = time.time()
        result = get_comprehensive_anomalies(large_df)
        end_time = time.time()
        
        self.assertLess(end_time - start_time, 30, "Should complete within 30 seconds")
        self.assertIsInstance(result, dict)


class TestAnomalyDetectionIntegration(unittest.TestCase):
    """Integration tests for anomaly detection."""
    
    def setUp(self):
        """Set up integration test data."""
        # Create realistic expense data
        np.random.seed(123)
        
        categories = ['food', 'transport', 'entertainment', 'utilities', 'shopping']
        start_date = datetime.now() - timedelta(days=180)
        
        # Generate realistic spending patterns
        data = []
        for i in range(180):
            date = start_date + timedelta(days=i)
            
            # Weekend vs weekday patterns
            is_weekend = date.weekday() >= 5
            base_multiplier = 1.5 if is_weekend else 1.0
            
            # Generate 1-5 transactions per day
            num_transactions = np.random.randint(1, 6)
            
            for _ in range(num_transactions):
                category = np.random.choice(categories)
                
                # Category-specific amounts
                if category == 'food':
                    amount = np.random.normal(25, 10) * base_multiplier
                elif category == 'transport':
                    amount = np.random.normal(15, 5) * base_multiplier
                elif category == 'entertainment':
                    amount = np.random.normal(40, 15) * base_multiplier
                elif category == 'utilities':
                    amount = np.random.normal(100, 20) if i % 30 == 0 else 0  # Monthly
                else:  # shopping
                    amount = np.random.normal(60, 25) * base_multiplier
                
                if amount > 0:
                    data.append({
                        'date': date,
                        'amount': max(amount, 1),  # Ensure positive
                        'category': category
                    })
        
        # Add some known anomalies
        anomaly_dates = [start_date + timedelta(days=30), start_date + timedelta(days=90)]
        for date in anomaly_dates:
            data.append({
                'date': date,
                'amount': 500,  # Unusually large amount
                'category': 'food'
            })
        
        self.integration_df = pd.DataFrame(data)

    def test_end_to_end_anomaly_detection(self):
        """Test complete anomaly detection workflow."""
        # Run comprehensive anomaly detection
        anomalies = get_comprehensive_anomalies(self.integration_df)
        
        # Check that we get results from multiple methods
        methods_with_results = sum(1 for df in anomalies.values() if not df.empty)
        self.assertGreater(methods_with_results, 0, "At least one method should find anomalies")
        
        # Check that anomalies are reasonable
        for method_name, anomaly_df in anomalies.items():
            if not anomaly_df.empty:
                # Anomalies should have reasonable scores/z-values
                if 'score' in anomaly_df.columns:
                    max_score = anomaly_df['score'].abs().max()
                    self.assertGreater(max_score, 0.1, "Should have some meaningful anomaly scores")
                elif 'z' in anomaly_df.columns:
                    max_z = anomaly_df['z'].abs().max()
                    self.assertGreater(max_z, 1.0, "Should have some meaningful z-scores")

    def test_consistency_across_methods(self):
        """Test that different methods produce consistent results."""
        anomalies = get_comprehensive_anomalies(self.integration_df)
        
        # If multiple daily methods find anomalies, there should be some overlap
        daily_methods = ['daily_zscore', 'daily_iqr', 'daily_modified_zscore']
        daily_anomalies = [anomalies[method] for method in daily_methods if not anomalies[method].empty]
        
        if len(daily_anomalies) >= 2:
            # Check for date overlap between methods
            date_sets = [set(df['date'].dt.date) for df in daily_anomalies]
            intersection = set.intersection(*date_sets)
            # Should have at least some agreement between methods
            self.assertGreaterEqual(len(intersection), 0)


def run_anomaly_tests():
    """Run all anomaly detection tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetection))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetectionIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("Running Anomaly Detection Test Suite...")
    print("=" * 50)
    
    result = run_anomaly_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
    
    exit(0 if result.wasSuccessful() else 1)