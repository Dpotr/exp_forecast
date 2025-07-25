# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run app.py
```

### Testing
```bash
# Run all tests
python test_anomaly_detection.py
python test_utility_functions.py
python test_weekly_aggregation.py

# Run specific test file
python -m pytest tests/test_select_method.py -v

# Run individual test modules
python -c "from test_anomaly_detection import run_anomaly_tests; run_anomaly_tests()"
python -c "from test_utility_functions import run_utility_tests; run_utility_tests()"
```

### Data Processing
```bash
# Update expenses from daily payments file
python update_expenses_from_daily.py

# Run seasonality diagnostics
python seasonality_diagnostics.py
```

### Module Testing
```bash
# Test individual modules
python forecast_metrics.py      # Test forecast metrics
python cross_validation.py      # Test cross-validation
python data_validation.py       # Test data validation
```

## Architecture Overview

### Core Application Structure

**Main Application (`app.py`)**
- 694-line Streamlit dashboard (was 1,413 lines - **51% reduction achieved**)
- **Modularized architecture** with proper separation of concerns
- Orchestrates UI components from modular `ui/` modules
- Integrates all utility modules for comprehensive expense forecasting
- Includes git auto-commit functionality (use with caution)

**Modular UI Architecture (`ui/` directory)**
- `ui/dashboard_sections.py`: Complete dashboard sections (render_*_section functions)
- `ui/components.py`: Reusable Streamlit widgets and utility functions  
- `ui/charts.py`: Plotly chart generation and styling utilities
- `ui/__init__.py`: Module initialization and imports

**Configuration System (`config.py`)**
- Centralized configuration for all file paths and parameters
- Environment variable support for customization (e.g., `DAILY_PAYMENTS_PATH`)
- Global `config` instance used throughout the application
- Default settings for forecasting windows, thresholds, and file locations

### Forecasting Engine

**Method Selection (`select_method()` in app.py)**
- Automated forecasting method selection based on backtesting
- Methods: mean, median, Croston's, Prophet, periodic spike, zero
- Category-specific overrides for special cases (rent, school payments)
- Rolling backtest evaluation with MAE comparison

**Specialized Forecasting Methods**
- `croston.py`: Croston's method for intermittent demand forecasting
- `forecast_utils.py`: Daily-to-weekly/monthly aggregation utilities
- Prophet integration for trend and seasonality detection
- Periodic spike detection for recurring large expenses

### Data Processing Pipeline

**Data Validation (`data_validation.py`)**
- `DataValidator` class with strict/lenient modes
- Comprehensive input validation with auto-cleaning capabilities
- Handles missing values, data type conversion, and outlier detection
- Detailed validation reporting with warnings and errors

**Anomaly Detection (`anomaly_utils.py`)**
- Multiple detection methods: Z-score, IQR, Modified Z-score, Seasonal
- Daily outlier detection and transaction-level anomaly identification
- Recurring payment pattern detection
- Comprehensive visualization functions for Streamlit integration

**Data Import (`update_expenses_from_daily.py`)**
- Automated expense import from Excel file
- Data cleaning, deduplication, and normalization
- Idempotent operations to prevent duplicate entries

### Advanced Analytics

**Forecast Metrics (`forecast_metrics.py`)**
- `ForecastMetrics` class with 30+ accuracy metrics
- Comprehensive evaluation: MAPE, MAE, directional accuracy, hit rates
- Quality assessment with automatic rating system
- Multi-forecast comparison and ranking capabilities
- Time-based performance analysis (monthly/weekly patterns)

**Cross-Validation (`cross_validation.py`)**
- `TimeSeriesCrossValidator` with multiple CV strategies
- Time series split, expanding window, sliding window methods
- Method comparison and stability scoring
- Performance assessment with reliability metrics

### Testing Framework

**Comprehensive Test Suite**
- `test_anomaly_detection.py`: 16 tests for anomaly detection methods
- `test_utility_functions.py`: 28 tests covering all utility modules
- Integration tests for end-to-end pipeline validation
- Edge case handling and error condition testing
- 100% test pass rate across 44 total tests

## Key Integration Points

**Streamlit Dashboard Integration**
- Comprehensive anomaly detection dashboard with tabbed interface
- Enhanced forecast metrics display with quality assessments
- Category-level performance analysis
- Interactive visualizations with method explanations

**Configuration-Driven Design**
- All thresholds and parameters centralized in `config.py`
- Environment variable support for deployment flexibility
- Consistent parameter usage across all modules

**Modular Architecture**
- Clear separation between data processing, analysis, and visualization
- Utility modules can be used independently or in combination
- Comprehensive error handling and validation throughout

## Data Flow Architecture

1. **Data Input**: Excel files (`expenses.xlsx`, daily payments)
2. **Validation**: `data_validation.py` cleans and validates input
3. **Processing**: Category-wise time series preparation
4. **Method Selection**: Automated backtesting and method selection
5. **Forecasting**: Multiple forecasting methods with error handling
6. **Evaluation**: Comprehensive metrics and cross-validation
7. **Anomaly Detection**: Multi-method outlier and pattern detection
8. **Visualization**: Modular Streamlit dashboard with interactive charts
9. **Export**: Results saved to `forecast_results.xlsx`

## Configuration Customization

Key configuration parameters in `config.py`:
- `DEFAULT_ACTIVITY_WINDOW`: Days of recent data for forecasting (default: 70)
- `DEFAULT_FORECAST_HORIZON`: Days ahead to forecast (default: 7)
- `OUTLIER_WINDOW_DAYS`: Window for outlier detection (default: 60)
- `ANOMALY_Z_THRESHOLD`: Z-score threshold for anomaly detection (default: 3)
- File paths for data input/output with environment variable overrides

## Development Notes

**Current Technical Debt** *(Significantly Reduced)*
- ✅ **Modularization Complete**: `app.py` reduced from 1,413 to 694 lines (51% reduction)
- ✅ **UI Separation**: Major dashboard sections moved to modular `ui/` components
- ⚠️ Git auto-commit function needs safety improvements
- ⏳ Large functions (e.g., some forecasting methods) could benefit from decomposition

**Testing Strategy**
- Comprehensive test coverage for all utility modules
- Integration tests for end-to-end workflows
- Edge case testing for data validation and error handling
- Modular testing approach allows independent module validation

## Refactoring Progress

**Completed (Phase 1-4B)**
- ✅ Configuration centralization
- ✅ Enhanced anomaly detection with multiple methods
- ✅ Comprehensive forecast metrics implementation
- ✅ Cross-validation framework
- ✅ Complete test suite with 100% pass rate
- ✅ Data validation framework
- ✅ **Major UI modularization**: Created comprehensive `ui/` module architecture
- ✅ **Massive code reduction**: 719 lines eliminated from `app.py` (51% smaller)
- ✅ **Proper separation of concerns**: Dashboard sections, components, and charts modularized

**Optional Future Work (Phase 5+)**
- Function decomposition for remaining large methods
- Git auto-commit safety improvements
- Additional inline section modularization (smaller sections)