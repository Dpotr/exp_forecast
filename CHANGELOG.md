# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-07-24

### Added
- `select_method` function to automatically choose the best forecasting methodology based on rolling backtest MAE and category-specific rules.
- Sidebar UI controls for manual override of forecast method selection mode.
- Integration of automatic/manual method selection into the forecast loop in `app.py`.
- Unit tests for `select_method` under `tests/test_select_method.py`.
- Documentation of method applicability and override rules in `README.md`.

### Changed
- Refactored forecast method selection logic to use `select_method` helper.
- Updated README with new forecasting options and manual override instructions.

### Fixed
- Ensured no regressions in existing forecast functionality when switching between Automatic and Manual modes.

