# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Titanic competition project with a modular machine learning pipeline architecture. The codebase is structured around abstract base classes that enable easy experimentation with different datasets and classifiers.

## Core Architecture

The project uses an **inheritance-based plugin architecture** with three main abstractions:

1. **BinaryClassificationData** (`base_data.py`) - Abstract base class for data handling
   - Handles loading, feature engineering, and preparation
   - Subclass: `TitanicData` (titanic_data.py) - Implements Titanic-specific features

2. **BinaryClassifier** (`base_classifier.py`) - Abstract base class for models
   - Provides training, evaluation, and analysis methods
   - Handles NaN imputation automatically for classifiers that need it
   - Subclasses:
     - `RandomForestBinaryClassifier` (random_forest_classifier.py) - Can handle NaN natively
     - `LogisticRegressionBinaryClassifier` (logistic_regression_classifier.py) - Requires imputation

3. **Generic CLI** (`cli.py`) - Dynamic command-line interface
   - Automatically generates CLI arguments from classifier's `get_cli_arguments()` method
   - Plugs together any data class + classifier class combination

## Key Design Patterns

### NaN Handling Strategy
- Classifiers declare whether they can handle NaN via `handles_nan()` class method
- If `handles_nan()` returns `False`, the framework automatically wraps the model in a Pipeline with SimpleImputer
- Random Forest handles NaN natively; Logistic Regression requires imputation
- This is managed transparently in `BinaryClassifier._create_model_with_pipeline()`

### Dynamic CLI Arguments
- Each classifier defines its parameters via `get_cli_arguments()` class method
- Returns list of dicts with 'name', 'type', 'default', 'help'
- CLI automatically extracts and passes these to the classifier constructor
- See `cli.py:24-32` and `cli.py:87-96` for implementation

### Feature Engineering Flow
1. `load_data()` - Load raw CSV
2. `engineer_features()` - Create domain-specific features (modifies `processed_data` in place)
3. `get_feature_columns()` - Return list of feature column names
4. `prepare_features()` - Apply `pd.get_dummies()` for one-hot encoding, convert to float

## Common Commands

### Activate Environment
```bash
source venv/bin/activate
```

### Run with Random Forest (default)
```bash
python predict.py cv --cv-folds 5
python predict.py importance --n-repeats 10
python predict.py drop-importance --cv-folds 5 --num-repeats 3
python predict.py train
python predict.py pdp
python predict.py correlation
python predict.py inspect  # Shows categorical value spaces from train+test by default
python predict.py inspect --test-data ./test.csv  # Explicit test file path
```

### Run with Logistic Regression
```bash
python predict_lr.py cv --C 0.5 --max-iter 200
python predict_lr.py importance --C 1.0
```

### Adjust Classifier Hyperparameters
```bash
python predict.py cv --n-estimators 200 --max-depth 30
python predict_lr.py cv --C 0.1 --max-iter 500
```

## CLI Commands

- `cv` - Cross-validation with accuracy scoring
- `importance` - Permutation feature importance
- `drop-importance` - Feature importance by dropping each column (parallel execution)
- `train` - Train final model on full dataset
- `pdp` - Partial dependence plots for numerical features
- `correlation` - Feature correlation heatmap
- `inspect` - Data quality inspection (NaN analysis, shape, target distribution, categorical value spaces)
  - Use `--test-data` to show combined train+test categorical values (last names, ticket tokens, decks)

## Titanic-Specific Features

The `TitanicData` class engineers these features from raw data:
- **Gender**: Binary Male indicator
- **Name features**: Name length, last name, count of passengers with same last name
- **Cabin features**: Cabin count, boolean indicators for each deck (A-G, T, U)
- **Ticket features**: Numeric ticket number (log-scaled), ticket number length, boolean indicators for ticket prefix tokens (e.g., 'PC', 'STON', 'CA')

### Categorical Feature Consistency

When `prepare_for_submission()` is called, the system computes the **complete categorical value space** from train+test combined:
- **Last names**: All unique surnames (875 total) for accurate family size counting
- **Ticket tokens**: All unique ticket prefixes (27 total) like 'pc', 'ston', 'ca', 'aq', 'lp'
- **Deck letters**: All unique deck values (9 total: A, B, C, D, E, F, G, T, U)

This ensures train and test have **identical feature spaces**:
- If deck 'T' only exists in train, test will still have a `Deck_T` feature (all zeros)
- If token 'aq' only exists in test, train will still have a `ticket_token_aq` feature (all zeros)
- Models see consistent features regardless of which dataset has which categorical values

Features created: `Deck_A`, `Deck_B`, ..., `Deck_U`, `ticket_token_a`, `ticket_token_ah`, ..., `ticket_token_we`

## Python Environment

- Python 3.12.3
- Virtual environment in `./venv`
- Key dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

## Adding New Classifiers

1. Create new file (e.g., `xgboost_classifier.py`)
2. Inherit from `BinaryClassifier`
3. Implement:
   - `create_model()` - Return model instance
   - `get_cli_arguments()` - Define CLI args as list of dicts
   - `handles_nan()` - Return True if model handles NaN natively
4. Create entry point script (e.g., `predict_xgb.py`) that calls `create_cli()` with your classifier

## Adding New Datasets

1. Create new file (e.g., `my_data.py`)
2. Inherit from `BinaryClassificationData`
3. Implement:
   - `load_data()` - Load from CSV
   - `engineer_features()` - Create features (modify `self.processed_data`)
   - `get_feature_columns()` - Return list of feature names
   - `get_target_column()` - Return target column name
4. Create entry point that calls `create_cli()` with your data class

## Important Implementation Details

- All features are converted to float after `pd.get_dummies()` to avoid sklearn warnings
- Drop-column importance uses parallel processing (half of available CPU cores)
- Partial dependence plots auto-detect numerical features (non-binary, non-one-hot)
- Cross-validation uses StratifiedKFold for balanced class distribution
- Random state defaults to 42 for reproducibility
