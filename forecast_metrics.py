"""
Forecast accuracy metrics module for expense forecasting application.
Provides comprehensive metrics for evaluating forecast performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
from config import config


class ForecastMetrics:
    """Comprehensive forecast accuracy metrics calculator."""
    
    def __init__(self):
        """Initialize forecast metrics calculator."""
        self.metrics_cache = {}
    
    def calculate_all_metrics(self, actual: Union[pd.Series, np.ndarray, List], 
                            forecast: Union[pd.Series, np.ndarray, List],
                            dates: Optional[Union[pd.Series, List]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive forecast accuracy metrics.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            dates: Optional dates for time-based analysis
            
        Returns:
            Dictionary with all calculated metrics
        """
        # Convert inputs to numpy arrays for consistent handling
        actual = np.array(actual, dtype=float)
        forecast = np.array(forecast, dtype=float)
        
        if len(actual) != len(forecast):
            raise ValueError(f"Length mismatch: actual ({len(actual)}) vs forecast ({len(forecast)})")
        
        if len(actual) == 0:
            return self._empty_metrics()
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(actual) | np.isnan(forecast))
        if not np.any(valid_mask):
            return self._empty_metrics()
        
        actual_clean = actual[valid_mask]
        forecast_clean = forecast[valid_mask]
        
        # Calculate all metrics
        metrics = {
            'basic_metrics': self._calculate_basic_metrics(actual_clean, forecast_clean),
            'percentage_metrics': self._calculate_percentage_metrics(actual_clean, forecast_clean),
            'scaled_metrics': self._calculate_scaled_metrics(actual_clean, forecast_clean),
            'directional_metrics': self._calculate_directional_metrics(actual_clean, forecast_clean),
            'distribution_metrics': self._calculate_distribution_metrics(actual_clean, forecast_clean),
            'summary_stats': self._calculate_summary_stats(actual_clean, forecast_clean),
            'data_info': {
                'total_points': len(actual),
                'valid_points': len(actual_clean),
                'missing_points': len(actual) - len(actual_clean),
                'missing_percentage': ((len(actual) - len(actual_clean)) / len(actual)) * 100
            }
        }
        
        # Add time-based metrics if dates provided
        if dates is not None and len(dates) == len(actual):
            dates_clean = np.array(dates)[valid_mask] if hasattr(dates, '__len__') else dates
            metrics['time_metrics'] = self._calculate_time_metrics(
                actual_clean, forecast_clean, dates_clean
            )
        
        # Add overall assessment
        metrics['overall_assessment'] = self._assess_forecast_quality(metrics)
        
        return metrics
    
    def _calculate_basic_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate basic forecast accuracy metrics."""
        errors = forecast - actual
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        return {
            'mae': np.mean(abs_errors),  # Mean Absolute Error
            'mse': np.mean(squared_errors),  # Mean Squared Error
            'rmse': np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error
            'me': np.mean(errors),  # Mean Error (bias)
            'mad': np.median(abs_errors),  # Median Absolute Deviation
            'max_error': np.max(abs_errors),  # Maximum absolute error
            'min_error': np.min(abs_errors),  # Minimum absolute error
            'std_error': np.std(errors),  # Standard deviation of errors
            'tae': np.sum(abs_errors),  # Time Absolute Error (total absolute error)
        }
    
    def _calculate_percentage_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate percentage-based forecast accuracy metrics."""
        # Avoid division by zero
        non_zero_mask = actual != 0
        if not np.any(non_zero_mask):
            return {
                'mape': np.inf,
                'smape': np.inf,
                'mpe': np.inf,
                'wape': np.inf
            }
        
        actual_nz = actual[non_zero_mask]
        forecast_nz = forecast[non_zero_mask]
        errors_nz = forecast_nz - actual_nz
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs(errors_nz / actual_nz)) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(errors_nz) / (np.abs(actual_nz) + np.abs(forecast_nz))) * 100
        
        # Mean Percentage Error
        mpe = np.mean(errors_nz / actual_nz) * 100
        
        # Weighted Absolute Percentage Error
        wape = np.sum(np.abs(forecast - actual)) / np.sum(np.abs(actual)) * 100
        
        return {
            'mape': mape,
            'smape': smape,
            'mpe': mpe,
            'wape': wape
        }
    
    def _calculate_scaled_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate scale-independent forecast accuracy metrics."""
        errors = forecast - actual
        abs_errors = np.abs(errors)
        
        # Mean Absolute Scaled Error (MASE)
        # For non-seasonal data, use naive forecast (previous period)
        if len(actual) > 1:
            naive_errors = np.abs(np.diff(actual))
            mae_naive = np.mean(naive_errors)
            mase = np.mean(abs_errors) / mae_naive if mae_naive > 0 else np.inf
        else:
            mase = np.inf
        
        # Normalized metrics
        actual_range = np.max(actual) - np.min(actual)
        actual_std = np.std(actual)
        actual_mean = np.mean(actual)
        
        return {
            'mase': mase,
            'nmae_range': np.mean(abs_errors) / actual_range if actual_range > 0 else np.inf,
            'nmae_std': np.mean(abs_errors) / actual_std if actual_std > 0 else np.inf,
            'nmae_mean': np.mean(abs_errors) / actual_mean if actual_mean > 0 else np.inf,
            'cv_rmse': np.sqrt(np.mean((errors) ** 2)) / actual_mean if actual_mean > 0 else np.inf
        }
    
    def _calculate_directional_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy metrics."""
        if len(actual) < 2:
            return {
                'directional_accuracy': 0.0,
                'hit_rate_10pct': 0.0,
                'hit_rate_20pct': 0.0,
                'prediction_of_change': 0.0
            }
        
        # Directional accuracy (for changes)
        actual_changes = np.diff(actual)
        forecast_changes = np.diff(forecast)
        
        # Count correct direction predictions
        correct_direction = np.sign(actual_changes) == np.sign(forecast_changes)
        directional_accuracy = np.mean(correct_direction) * 100
        
        # Hit rates (percentage within X% of actual)
        percentage_errors = np.abs((forecast - actual) / actual) * 100
        hit_rate_10pct = np.mean(percentage_errors <= 10) * 100
        hit_rate_20pct = np.mean(percentage_errors <= 20) * 100
        
        # Prediction of Change (PoC)
        actual_change_mask = actual_changes != 0
        if np.any(actual_change_mask):
            poc = np.mean(correct_direction[actual_change_mask]) * 100
        else:
            poc = 0.0
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate_10pct': hit_rate_10pct,
            'hit_rate_20pct': hit_rate_20pct,
            'prediction_of_change': poc
        }
    
    def _calculate_distribution_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate distribution-based metrics."""
        errors = forecast - actual
        
        # Error distribution statistics
        error_percentiles = np.percentile(np.abs(errors), [25, 50, 75, 90, 95, 99])
        
        # Theil's U statistic
        mse_forecast = np.mean((forecast - actual) ** 2)
        mse_naive = np.mean((actual[1:] - actual[:-1]) ** 2) if len(actual) > 1 else np.inf
        theil_u = np.sqrt(mse_forecast) / np.sqrt(mse_naive) if mse_naive > 0 else np.inf
        
        # Forecast bias indicators
        overforecast_pct = np.mean(forecast > actual) * 100
        underforecast_pct = np.mean(forecast < actual) * 100
        
        return {
            'error_25th_percentile': error_percentiles[0],
            'error_50th_percentile': error_percentiles[1],
            'error_75th_percentile': error_percentiles[2],
            'error_90th_percentile': error_percentiles[3],
            'error_95th_percentile': error_percentiles[4],
            'error_99th_percentile': error_percentiles[5],
            'theil_u': theil_u,
            'overforecast_percentage': overforecast_pct,
            'underforecast_percentage': underforecast_pct,
            'forecast_bias': np.mean(forecast) - np.mean(actual)
        }
    
    def _calculate_summary_stats(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate summary statistics for actual and forecast values."""
        return {
            'actual_mean': np.mean(actual),
            'actual_std': np.std(actual),
            'actual_min': np.min(actual),
            'actual_max': np.max(actual),
            'forecast_mean': np.mean(forecast),
            'forecast_std': np.std(forecast),
            'forecast_min': np.min(forecast),
            'forecast_max': np.max(forecast),
            'correlation': np.corrcoef(actual, forecast)[0, 1] if len(actual) > 1 else 0.0,
            'r_squared': np.corrcoef(actual, forecast)[0, 1] ** 2 if len(actual) > 1 else 0.0
        }
    
    def _calculate_time_metrics(self, actual: np.ndarray, forecast: np.ndarray, 
                              dates: np.ndarray) -> Dict[str, Any]:
        """Calculate time-based forecast metrics."""
        if len(dates) == 0:
            return {}
        
        # Convert dates to pandas datetime if they aren't already
        try:
            dates_pd = pd.to_datetime(dates)
        except:
            return {'error': 'Could not parse dates'}
        
        df = pd.DataFrame({
            'date': dates_pd,
            'actual': actual,
            'forecast': forecast,
            'error': forecast - actual,
            'abs_error': np.abs(forecast - actual)
        })
        
        # Monthly performance
        df['month'] = df['date'].dt.to_period('M')
        monthly_metrics = df.groupby('month').agg({
            'abs_error': 'mean',
            'error': 'mean'
        }).rename(columns={'abs_error': 'mae', 'error': 'bias'})
        
        # Day of week performance
        df['day_of_week'] = df['date'].dt.day_name()
        dow_metrics = df.groupby('day_of_week').agg({
            'abs_error': 'mean',
            'error': 'mean'
        }).rename(columns={'abs_error': 'mae', 'error': 'bias'})
        
        # Recent vs. historical performance
        median_date = df['date'].median()
        recent_mask = df['date'] >= median_date
        
        recent_mae = df.loc[recent_mask, 'abs_error'].mean()
        historical_mae = df.loc[~recent_mask, 'abs_error'].mean()
        
        return {
            'monthly_performance': monthly_metrics.to_dict(),
            'day_of_week_performance': dow_metrics.to_dict(),
            'recent_mae': recent_mae,
            'historical_mae': historical_mae,
            'improvement_over_time': (historical_mae - recent_mae) / historical_mae * 100 if historical_mae > 0 else 0
        }
    
    def _assess_forecast_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall forecast quality based on calculated metrics."""
        basic = metrics['basic_metrics']
        percentage = metrics['percentage_metrics']
        directional = metrics['directional_metrics']
        
        # Define quality thresholds
        quality_assessment = {
            'mape_rating': self._rate_mape(percentage['mape']),
            'directional_rating': self._rate_directional_accuracy(directional['directional_accuracy']),
            'hit_rate_rating': self._rate_hit_rate(directional['hit_rate_20pct']),
            'overall_rating': 'pending'
        }
        
        # Calculate overall rating
        ratings = [quality_assessment['mape_rating'], 
                  quality_assessment['directional_rating'], 
                  quality_assessment['hit_rate_rating']]
        
        rating_scores = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        avg_score = np.mean([rating_scores[r] for r in ratings])
        
        if avg_score >= 3.5:
            overall = 'excellent'
        elif avg_score >= 2.5:
            overall = 'good'
        elif avg_score >= 1.5:
            overall = 'fair'
        else:
            overall = 'poor'
        
        quality_assessment['overall_rating'] = overall
        quality_assessment['overall_score'] = avg_score
        
        # Add recommendations
        quality_assessment['recommendations'] = self._generate_recommendations(metrics)
        
        return quality_assessment
    
    def _rate_mape(self, mape: float) -> str:
        """Rate forecast based on MAPE."""
        if mape <= 10:
            return 'excellent'
        elif mape <= 20:
            return 'good'
        elif mape <= 50:
            return 'fair'
        else:
            return 'poor'
    
    def _rate_directional_accuracy(self, da: float) -> str:
        """Rate forecast based on directional accuracy."""
        if da >= 70:
            return 'excellent'
        elif da >= 60:
            return 'good'
        elif da >= 50:
            return 'fair'
        else:
            return 'poor'
    
    def _rate_hit_rate(self, hr: float) -> str:
        """Rate forecast based on hit rate (20% threshold)."""
        if hr >= 80:
            return 'excellent'
        elif hr >= 70:
            return 'good'
        elif hr >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        basic = metrics['basic_metrics']
        percentage = metrics['percentage_metrics']
        directional = metrics['directional_metrics']
        distribution = metrics['distribution_metrics']
        
        # Bias recommendations
        if abs(distribution['forecast_bias']) > basic['mae']:
            if distribution['forecast_bias'] > 0:
                recommendations.append("Forecast shows consistent overestimation bias - consider adjusting model parameters")
            else:
                recommendations.append("Forecast shows consistent underestimation bias - consider adjusting model parameters")
        
        # Accuracy recommendations
        if percentage['mape'] > 30:
            recommendations.append("High MAPE indicates poor accuracy - consider alternative forecasting methods")
        
        # Directional accuracy recommendations
        if directional['directional_accuracy'] < 55:
            recommendations.append("Low directional accuracy - model struggles with trend prediction")
        
        # Hit rate recommendations
        if directional['hit_rate_20pct'] < 60:
            recommendations.append("Low hit rate suggests high volatility - consider ensemble methods")
        
        # Correlation recommendations
        summary = metrics['summary_stats']
        if summary['correlation'] < 0.7:
            recommendations.append("Low correlation between actual and forecast - model may not capture underlying patterns")
        
        if not recommendations:
            recommendations.append("Forecast performance is satisfactory - continue monitoring")
        
        return recommendations
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no valid data is available."""
        return {
            'basic_metrics': {},
            'percentage_metrics': {},
            'scaled_metrics': {},
            'directional_metrics': {},
            'distribution_metrics': {},
            'summary_stats': {},
            'data_info': {'total_points': 0, 'valid_points': 0, 'missing_points': 0},
            'overall_assessment': {'overall_rating': 'insufficient_data', 'recommendations': ['Insufficient data for analysis']}
        }
    
    def compare_forecasts(self, actual: np.ndarray, *forecasts: np.ndarray, 
                         forecast_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple forecasts against actual values.
        
        Args:
            actual: Actual values
            *forecasts: Multiple forecast arrays
            forecast_names: Optional names for forecasts
            
        Returns:
            Dictionary with comparison results
        """
        if not forecasts:
            raise ValueError("At least one forecast must be provided")
        
        if forecast_names is None:
            forecast_names = [f"Forecast_{i+1}" for i in range(len(forecasts))]
        
        comparison_results = {
            'individual_metrics': {},
            'ranking': {},
            'best_forecast': {}
        }
        
        # Calculate metrics for each forecast
        all_metrics = {}
        for i, forecast in enumerate(forecasts):
            name = forecast_names[i] if i < len(forecast_names) else f"Forecast_{i+1}"
            all_metrics[name] = self.calculate_all_metrics(actual, forecast)
        
        comparison_results['individual_metrics'] = all_metrics
        
        # Rank forecasts by different criteria
        ranking_criteria = ['mae', 'mape', 'rmse', 'directional_accuracy', 'hit_rate_20pct']
        rankings = {}
        
        for criterion in ranking_criteria:
            criterion_values = {}
            for name, metrics in all_metrics.items():
                if criterion in metrics['basic_metrics']:
                    criterion_values[name] = metrics['basic_metrics'][criterion]
                elif criterion in metrics['percentage_metrics']:
                    criterion_values[name] = metrics['percentage_metrics'][criterion]
                elif criterion in metrics['directional_metrics']:
                    criterion_values[name] = metrics['directional_metrics'][criterion]
            
            if criterion_values:
                # For accuracy metrics (lower is better)
                if criterion in ['mae', 'mape', 'rmse']:
                    rankings[criterion] = sorted(criterion_values.items(), key=lambda x: x[1])
                # For directional/hit rate metrics (higher is better)
                else:
                    rankings[criterion] = sorted(criterion_values.items(), key=lambda x: x[1], reverse=True)
        
        comparison_results['ranking'] = rankings
        
        # Determine overall best forecast
        forecast_scores = {}
        for name in all_metrics.keys():
            score = 0
            for criterion, ranking in rankings.items():
                for rank, (forecast_name, _) in enumerate(ranking):
                    if forecast_name == name:
                        score += len(forecasts) - rank  # Higher rank = higher score
                        break
            forecast_scores[name] = score
        
        best_forecast_name = max(forecast_scores.items(), key=lambda x: x[1])[0]
        comparison_results['best_forecast'] = {
            'name': best_forecast_name,
            'score': forecast_scores[best_forecast_name],
            'metrics': all_metrics[best_forecast_name]
        }
        
        return comparison_results


def calculate_forecast_metrics(actual: Union[pd.Series, np.ndarray, List], 
                             forecast: Union[pd.Series, np.ndarray, List],
                             dates: Optional[Union[pd.Series, List]] = None) -> Dict[str, Any]:
    """
    Convenience function for calculating forecast metrics.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
        dates: Optional dates for time-based analysis
        
    Returns:
        Dictionary with all calculated metrics
    """
    calculator = ForecastMetrics()
    return calculator.calculate_all_metrics(actual, forecast, dates)


if __name__ == '__main__':
    # Example usage and testing
    print("Testing Forecast Metrics Module")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_points = 100
    
    # Generate actual values with trend and seasonality
    trend = np.linspace(100, 150, n_points)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, n_points)
    actual_values = trend + seasonality + noise
    
    # Generate forecasts with different characteristics
    good_forecast = actual_values + np.random.normal(0, 3, n_points)  # Low error
    biased_forecast = actual_values * 1.1 + np.random.normal(0, 3, n_points)  # Systematic bias
    poor_forecast = actual_values + np.random.normal(0, 15, n_points)  # High error
    
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    
    print("\n1. Testing individual forecast metrics:")
    metrics = calculate_forecast_metrics(actual_values, good_forecast, dates)
    
    print(f"MAPE: {metrics['percentage_metrics']['mape']:.2f}%")
    print(f"MAE: {metrics['basic_metrics']['mae']:.2f}")
    print(f"Directional Accuracy: {metrics['directional_metrics']['directional_accuracy']:.1f}%")
    print(f"Hit Rate (20%): {metrics['directional_metrics']['hit_rate_20pct']:.1f}%")
    print(f"Overall Rating: {metrics['overall_assessment']['overall_rating']}")
    
    print("\n2. Testing forecast comparison:")
    calculator = ForecastMetrics()
    comparison = calculator.compare_forecasts(
        actual_values, good_forecast, biased_forecast, poor_forecast,
        forecast_names=['Good', 'Biased', 'Poor']
    )
    
    print(f"Best forecast: {comparison['best_forecast']['name']}")
    print("\nMAE Rankings:")
    for rank, (name, mae) in enumerate(comparison['ranking']['mae'], 1):
        print(f"  {rank}. {name}: {mae:.2f}")
    
    print("\n3. Testing with edge cases:")
    # Test with identical values
    identical_forecast = np.full(10, 100.0)
    identical_actual = np.full(10, 100.0)
    identical_metrics = calculate_forecast_metrics(identical_actual, identical_forecast)
    print(f"Identical values MAPE: {identical_metrics['percentage_metrics']['mape']}")
    
    # Test with empty data
    try:
        empty_metrics = calculate_forecast_metrics([], [])
        print("Empty data handled gracefully")
    except Exception as e:
        print(f"Empty data error: {e}")
    
    print("\nTesting completed successfully!")