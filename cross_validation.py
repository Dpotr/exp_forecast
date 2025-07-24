"""
Cross-validation module for expense forecasting application.
Provides time series cross-validation methods for model evaluation and selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from config import config
from forecast_metrics import ForecastMetrics, calculate_forecast_metrics


class TimeSeriesCrossValidator:
    """
    Time series cross-validation for forecast model evaluation.
    Implements various CV strategies suitable for time series data.
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV splits
            test_size: Size of test set (if None, uses 1/n_splits of data)
            gap: Gap between train and test sets (to avoid data leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.cv_results = {}
    
    def time_series_split(self, data_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series cross-validation splits.
        
        Args:
            data_length: Length of the time series
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.test_size is None:
            test_size = max(1, data_length // (self.n_splits + 1))
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate test start position
            test_start = data_length - test_size * (self.n_splits - i)
            test_end = test_start + test_size
            
            # Ensure we don't go beyond data bounds
            if test_start < 0 or test_end > data_length:
                continue
            
            # Calculate train end (considering gap)
            train_end = max(0, test_start - self.gap)
            
            # Ensure minimum training size
            min_train_size = max(10, test_size)  # At least 10 points or test_size
            if train_end < min_train_size:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits
    
    def expanding_window_split(self, data_length: int, min_train_size: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate expanding window cross-validation splits.
        Training set grows with each split while test set moves forward.
        
        Args:
            data_length: Length of the time series
            min_train_size: Minimum training set size
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.test_size is None:
            test_size = max(1, data_length // (self.n_splits + 2))
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # Test set position
            test_start = min_train_size + (i * test_size)
            test_end = test_start + test_size
            
            if test_end > data_length:
                break
            
            # Training set grows from start to test_start (minus gap)
            train_end = max(min_train_size, test_start - self.gap)
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) >= min_train_size and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits
    
    def sliding_window_split(self, data_length: int, train_size: int = 60) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate sliding window cross-validation splits.
        Both training and test sets slide forward maintaining fixed sizes.
        
        Args:
            data_length: Length of the time series
            train_size: Fixed training set size
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.test_size is None:
            test_size = max(1, train_size // 4)
        else:
            test_size = self.test_size
        
        splits = []
        
        # Calculate step size
        step_size = max(1, (data_length - train_size - test_size) // self.n_splits)
        
        for i in range(self.n_splits):
            train_start = i * step_size
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > data_length:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def cross_validate_forecasts(self, 
                                actual: Union[pd.Series, np.ndarray],
                                forecast_functions: Dict[str, Callable],
                                cv_method: str = 'time_series',
                                dates: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for multiple forecast methods.
        
        Args:
            actual: Actual time series values
            forecast_functions: Dict of {method_name: forecast_function}
            cv_method: CV method ('time_series', 'expanding', 'sliding')
            dates: Optional dates corresponding to actual values
            
        Returns:
            Dictionary with cross-validation results
        """
        # Convert to numpy array for consistent indexing
        actual_array = np.array(actual)
        dates_array = np.array(dates) if dates is not None else None
        
        # Generate splits based on method
        if cv_method == 'expanding':
            splits = self.expanding_window_split(len(actual_array))
        elif cv_method == 'sliding':
            splits = self.sliding_window_split(len(actual_array))
        else:  # time_series
            splits = self.time_series_split(len(actual_array))
        
        if not splits:
            return {'error': 'No valid splits generated', 'splits': []}
        
        cv_results = {
            'method_results': {},
            'splits_info': [],
            'summary': {},
            'best_method': None
        }
        
        # Initialize results for each method
        for method_name in forecast_functions.keys():
            cv_results['method_results'][method_name] = {
                'fold_metrics': [],
                'predictions': [],
                'actuals': [],
                'dates': [] if dates_array is not None else None
            }
        
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_actual = actual_array[train_idx]
            test_actual = actual_array[test_idx]
            
            # Store split info
            split_info = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start': train_idx[0] if len(train_idx) > 0 else None,
                'train_end': train_idx[-1] if len(train_idx) > 0 else None,
                'test_start': test_idx[0] if len(test_idx) > 0 else None,
                'test_end': test_idx[-1] if len(test_idx) > 0 else None
            }
            
            if dates_array is not None:
                split_info.update({
                    'train_start_date': dates_array[train_idx[0]] if len(train_idx) > 0 else None,
                    'train_end_date': dates_array[train_idx[-1]] if len(train_idx) > 0 else None,
                    'test_start_date': dates_array[test_idx[0]] if len(test_idx) > 0 else None,
                    'test_end_date': dates_array[test_idx[-1]] if len(test_idx) > 0 else None
                })
            
            cv_results['splits_info'].append(split_info)
            
            # Test each forecast method
            for method_name, forecast_func in forecast_functions.items():
                try:
                    # Generate forecast for test period
                    forecast = forecast_func(train_actual, len(test_idx))
                    
                    # Ensure forecast is same length as test
                    if isinstance(forecast, (list, np.ndarray)):
                        forecast = np.array(forecast)
                        if len(forecast) != len(test_actual):
                            # If forecast is shorter, pad with last value
                            if len(forecast) < len(test_actual):
                                last_val = forecast[-1] if len(forecast) > 0 else np.mean(train_actual)
                                forecast = np.pad(forecast, (0, len(test_actual) - len(forecast)), 
                                                'constant', constant_values=last_val)
                            else:
                                forecast = forecast[:len(test_actual)]
                    else:
                        # Single value forecast - repeat for all test points
                        forecast = np.full(len(test_actual), forecast)
                    
                    # Calculate metrics for this fold
                    fold_metrics = calculate_forecast_metrics(test_actual, forecast)
                    
                    # Store results
                    cv_results['method_results'][method_name]['fold_metrics'].append(fold_metrics)
                    cv_results['method_results'][method_name]['predictions'].extend(forecast)
                    cv_results['method_results'][method_name]['actuals'].extend(test_actual)
                    
                    if dates_array is not None:
                        cv_results['method_results'][method_name]['dates'].extend(dates_array[test_idx])
                
                except Exception as e:
                    # Record error for this fold
                    error_metrics = self._create_error_metrics(str(e))
                    cv_results['method_results'][method_name]['fold_metrics'].append(error_metrics)
                    
                    # Pad with NaN values
                    cv_results['method_results'][method_name]['predictions'].extend([np.nan] * len(test_idx))
                    cv_results['method_results'][method_name]['actuals'].extend(test_actual)
                    
                    if dates_array is not None:
                        cv_results['method_results'][method_name]['dates'].extend(dates_array[test_idx])
        
        # Calculate summary statistics
        cv_results['summary'] = self._calculate_cv_summary(cv_results['method_results'])
        
        # Determine best method
        cv_results['best_method'] = self._determine_best_method(cv_results['summary'])
        
        return cv_results
    
    def _create_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Create error metrics structure."""
        return {
            'basic_metrics': {'mae': np.inf, 'rmse': np.inf, 'me': np.inf},
            'percentage_metrics': {'mape': np.inf, 'smape': np.inf},
            'directional_metrics': {'directional_accuracy': 0, 'hit_rate_20pct': 0},
            'overall_assessment': {'overall_rating': 'error', 'overall_score': 0},
            'error': error_message
        }
    
    def _calculate_cv_summary(self, method_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all CV folds."""
        summary = {}
        
        for method_name, results in method_results.items():
            fold_metrics = results['fold_metrics']
            
            if not fold_metrics:
                continue
            
            # Extract metrics across folds (skip error folds)
            valid_folds = [fm for fm in fold_metrics if 'error' not in fm]
            
            if not valid_folds:
                summary[method_name] = {
                    'mean_mae': np.inf,
                    'std_mae': np.inf,
                    'mean_mape': np.inf,
                    'std_mape': np.inf,
                    'mean_directional_accuracy': 0,
                    'mean_hit_rate_20pct': 0,
                    'mean_overall_score': 0,
                    'valid_folds': 0,
                    'total_folds': len(fold_metrics)
                }
                continue
            
            # Calculate means and standard deviations
            maes = [fm['basic_metrics']['mae'] for fm in valid_folds]
            mapes = [fm['percentage_metrics']['mape'] for fm in valid_folds]
            das = [fm['directional_metrics']['directional_accuracy'] for fm in valid_folds]
            hrs = [fm['directional_metrics']['hit_rate_20pct'] for fm in valid_folds]
            scores = [fm['overall_assessment']['overall_score'] for fm in valid_folds]
            
            summary[method_name] = {
                'mean_mae': np.mean(maes),
                'std_mae': np.std(maes),
                'mean_mape': np.mean(mapes),
                'std_mape': np.std(mapes),
                'mean_directional_accuracy': np.mean(das),
                'std_directional_accuracy': np.std(das),
                'mean_hit_rate_20pct': np.mean(hrs),
                'std_hit_rate_20pct': np.std(hrs),
                'mean_overall_score': np.mean(scores),
                'std_overall_score': np.std(scores),
                'valid_folds': len(valid_folds),
                'total_folds': len(fold_metrics),
                'stability': 1 / (1 + np.std(maes) / (np.mean(maes) + 1e-6))  # Stability metric
            }
        
        return summary
    
    def _determine_best_method(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best forecasting method based on CV results."""
        if not summary:
            return {'method': None, 'reason': 'No valid methods'}
        
        # Filter out methods with no valid folds
        valid_methods = {name: metrics for name, metrics in summary.items() 
                        if metrics['valid_folds'] > 0}
        
        if not valid_methods:
            return {'method': None, 'reason': 'No methods with valid folds'}
        
        # Scoring system: combine accuracy, stability, and reliability
        method_scores = {}
        
        for method_name, metrics in valid_methods.items():
            # Accuracy score (lower MAE and MAPE is better)
            mae_score = 1 / (1 + metrics['mean_mae'])
            mape_score = 1 / (1 + metrics['mean_mape'] / 100)
            
            # Stability score (lower std is better)
            stability_score = metrics['stability']
            
            # Reliability score (more valid folds is better)
            reliability_score = metrics['valid_folds'] / metrics['total_folds']
            
            # Directional accuracy score
            directional_score = metrics['mean_directional_accuracy'] / 100
            
            # Combined score
            combined_score = (mae_score * 0.3 + mape_score * 0.3 + 
                             stability_score * 0.2 + reliability_score * 0.1 + 
                             directional_score * 0.1)
            
            method_scores[method_name] = {
                'combined_score': combined_score,
                'mae_score': mae_score,
                'mape_score': mape_score,
                'stability_score': stability_score,
                'reliability_score': reliability_score,
                'directional_score': directional_score
            }
        
        # Find best method
        best_method = max(method_scores.items(), key=lambda x: x[1]['combined_score'])
        
        return {
            'method': best_method[0],
            'score': best_method[1]['combined_score'],
            'details': best_method[1],
            'reason': f"Best combined score: {best_method[1]['combined_score']:.3f}"
        }


class ForecastMethodValidator:
    """Validator for individual forecasting methods using cross-validation."""
    
    def __init__(self):
        self.validator = TimeSeriesCrossValidator()
    
    def validate_method(self, 
                       actual: Union[pd.Series, np.ndarray],
                       forecast_function: Callable,
                       method_name: str = "Method",
                       cv_method: str = 'time_series',
                       n_splits: int = 5) -> Dict[str, Any]:
        """
        Validate a single forecasting method using cross-validation.
        
        Args:
            actual: Actual time series values
            forecast_function: Function that takes (train_data, forecast_horizon) and returns forecast
            method_name: Name of the method
            cv_method: CV method to use
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with validation results
        """
        self.validator.n_splits = n_splits
        
        results = self.validator.cross_validate_forecasts(
            actual=actual,
            forecast_functions={method_name: forecast_function},
            cv_method=cv_method
        )
        
        # Extract results for the single method
        method_results = results['method_results'][method_name]
        
        return {
            'method_name': method_name,
            'cv_method': cv_method,
            'n_splits': n_splits,
            'fold_metrics': method_results['fold_metrics'],
            'summary': results['summary'][method_name] if method_name in results['summary'] else {},
            'splits_info': results['splits_info'],
            'overall_performance': self._assess_method_performance(
                results['summary'][method_name] if method_name in results['summary'] else {}
            )
        }
    
    def _assess_method_performance(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance of a method."""
        if not summary or summary.get('valid_folds', 0) == 0:
            return {
                'assessment': 'insufficient_data',
                'reliability': 'low',
                'recommendations': ['Insufficient data for reliable assessment']
            }
        
        # Assess different aspects
        mean_mape = summary.get('mean_mape', np.inf)
        stability = summary.get('stability', 0)
        reliability_ratio = summary.get('valid_folds', 0) / summary.get('total_folds', 1)
        
        # Performance assessment
        if mean_mape <= 15 and stability >= 0.7 and reliability_ratio >= 0.8:
            assessment = 'excellent'
        elif mean_mape <= 25 and stability >= 0.5 and reliability_ratio >= 0.6:
            assessment = 'good'
        elif mean_mape <= 40 and stability >= 0.3 and reliability_ratio >= 0.4:
            assessment = 'fair'
        else:
            assessment = 'poor'
        
        # Reliability assessment
        if reliability_ratio >= 0.9:
            reliability = 'high'
        elif reliability_ratio >= 0.7:
            reliability = 'medium'
        else:
            reliability = 'low'
        
        # Generate recommendations
        recommendations = []
        if mean_mape > 30:
            recommendations.append("High MAPE suggests poor accuracy - consider alternative methods")
        if stability < 0.5:
            recommendations.append("Low stability - method performance varies significantly across time periods")
        if reliability_ratio < 0.8:
            recommendations.append("Some folds failed - method may not be robust to different data conditions")
        
        if not recommendations:
            recommendations.append("Method performance is satisfactory")
        
        return {
            'assessment': assessment,
            'reliability': reliability,
            'key_metrics': {
                'mean_mape': mean_mape,
                'stability': stability,
                'reliability_ratio': reliability_ratio
            },
            'recommendations': recommendations
        }


# Convenience functions for common forecasting methods
def create_forecast_functions(df: pd.DataFrame, category: str = None) -> Dict[str, Callable]:
    """
    Create common forecast functions for cross-validation.
    
    Args:
        df: DataFrame with transaction data
        category: Specific category to forecast (if None, forecasts total)
        
    Returns:
        Dictionary of forecast functions
    """
    def mean_forecast(train_data, horizon):
        """Simple mean forecast."""
        if len(train_data) == 0:
            return np.zeros(horizon)
        return np.full(horizon, np.mean(train_data))
    
    def median_forecast(train_data, horizon):
        """Simple median forecast."""
        if len(train_data) == 0:
            return np.zeros(horizon)
        return np.full(horizon, np.median(train_data))
    
    def moving_average_forecast(train_data, horizon, window=7):
        """Moving average forecast."""
        if len(train_data) == 0:
            return np.zeros(horizon)
        if len(train_data) < window:
            return np.full(horizon, np.mean(train_data))
        return np.full(horizon, np.mean(train_data[-window:]))
    
    def trend_forecast(train_data, horizon):
        """Simple linear trend forecast."""
        if len(train_data) < 2:
            return np.full(horizon, np.mean(train_data) if len(train_data) > 0 else 0)
        
        # Fit simple linear trend
        x = np.arange(len(train_data))
        coeffs = np.polyfit(x, train_data, 1)
        
        # Project forward
        future_x = np.arange(len(train_data), len(train_data) + horizon)
        return np.polyval(coeffs, future_x)
    
    def seasonal_naive_forecast(train_data, horizon, season_length=7):
        """Seasonal naive forecast (repeat seasonal pattern)."""
        if len(train_data) == 0:
            return np.zeros(horizon)
        if len(train_data) < season_length:
            return np.full(horizon, np.mean(train_data))
        
        # Use last season as forecast
        last_season = train_data[-season_length:]
        forecast = []
        for i in range(horizon):
            forecast.append(last_season[i % season_length])
        return np.array(forecast)
    
    return {
        'mean': mean_forecast,
        'median': median_forecast,
        'moving_average': lambda train, h: moving_average_forecast(train, h, 7),
        'trend': trend_forecast,
        'seasonal_naive': seasonal_naive_forecast
    }


if __name__ == '__main__':
    # Example usage and testing
    print("Testing Cross-Validation Module")
    print("=" * 40)
    
    # Create sample time series data
    np.random.seed(42)
    n_points = 100
    
    # Generate time series with trend and seasonality
    trend = np.linspace(50, 100, n_points)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, n_points)
    ts_data = trend + seasonality + noise
    
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    
    print(f"\n1. Testing time series cross-validation with {n_points} points:")
    
    # Create forecast functions
    forecast_functions = {
        'mean': lambda train, h: np.full(h, np.mean(train)),
        'median': lambda train, h: np.full(h, np.median(train)),
        'moving_avg': lambda train, h: np.full(h, np.mean(train[-7:]) if len(train) >= 7 else np.mean(train)),
        'trend': lambda train, h: np.polyval(np.polyfit(range(len(train)), train, 1), 
                                           range(len(train), len(train) + h)) if len(train) > 1 else np.full(h, np.mean(train))
    }
    
    # Initialize cross-validator
    cv = TimeSeriesCrossValidator(n_splits=5, test_size=10, gap=1)
    
    # Run cross-validation
    cv_results = cv.cross_validate_forecasts(
        actual=ts_data,
        forecast_functions=forecast_functions,
        cv_method='time_series',
        dates=dates
    )
    
    print(f"Generated {len(cv_results['splits_info'])} CV splits")
    print(f"Best method: {cv_results['best_method']['method']}")
    print(f"Best method score: {cv_results['best_method']['score']:.3f}")
    
    # Display summary results
    print("\n2. Cross-validation summary:")
    for method, summary in cv_results['summary'].items():
        print(f"\n{method.upper()}:")
        print(f"  Mean MAE: ${summary['mean_mae']:.2f} ± ${summary['std_mae']:.2f}")
        print(f"  Mean MAPE: {summary['mean_mape']:.1f}% ± {summary['std_mape']:.1f}%")
        print(f"  Directional Accuracy: {summary['mean_directional_accuracy']:.1f}%")
        print(f"  Stability: {summary['stability']:.3f}")
        print(f"  Valid Folds: {summary['valid_folds']}/{summary['total_folds']}")
    
    print("\n3. Testing single method validation:")
    
    # Test single method validation
    method_validator = ForecastMethodValidator()
    
    single_method_results = method_validator.validate_method(
        actual=ts_data,
        forecast_function=lambda train, h: np.full(h, np.mean(train[-7:]) if len(train) >= 7 else np.mean(train)),
        method_name="7-Day Moving Average",
        cv_method='expanding',
        n_splits=4
    )
    
    performance = single_method_results['overall_performance']
    print(f"Method: {single_method_results['method_name']}")
    print(f"Assessment: {performance['assessment']}")
    print(f"Reliability: {performance['reliability']}")
    print(f"Key Metrics: MAPE={performance['key_metrics']['mean_mape']:.1f}%, Stability={performance['key_metrics']['stability']:.3f}")
    print("Recommendations:")
    for i, rec in enumerate(performance['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n4. Testing different CV methods:")
    
    # Test different CV methods
    cv_methods = ['time_series', 'expanding', 'sliding']
    simple_forecast = {'simple_mean': lambda train, h: np.full(h, np.mean(train))}
    
    for cv_method in cv_methods:
        cv_test = TimeSeriesCrossValidator(n_splits=3, test_size=8)
        results = cv_test.cross_validate_forecasts(
            actual=ts_data,
            forecast_functions=simple_forecast,
            cv_method=cv_method
        )
        
        if results['summary']:
            method_summary = list(results['summary'].values())[0]
            print(f"{cv_method}: MAE={method_summary['mean_mae']:.2f}, Valid Folds={method_summary['valid_folds']}/{method_summary['total_folds']}")
        else:
            print(f"{cv_method}: No valid results")
    
    print("\nTesting completed successfully!")