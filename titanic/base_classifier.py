"""Base classes for binary classification models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
import multiprocessing

from base_data import BinaryClassificationData


class BinaryClassifier(ABC):
    """Base class for binary classifiers.

    Provides training, evaluation, and analysis methods for binary classification
    models. Subclasses implement specific model types.
    """

    def __init__(self, data: BinaryClassificationData, random_state=42,
                 impute_strategy='mean', impute_fill_value=None, **model_params):
        """Initialize classifier with data and parameters.

        Args:
            data: BinaryClassificationData instance
            random_state: Random seed for reproducibility
            impute_strategy: Strategy for imputing missing values
                           ('mean', 'median', 'most_frequent', 'constant')
            impute_fill_value: Fill value when impute_strategy='constant'
            **model_params: Additional parameters to pass to the model
        """
        self.data = data
        self.random_state = random_state
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value
        self.model_params = model_params
        self.model = None

    @classmethod
    def get_cli_arguments(cls):
        """Return list of argparse argument definitions for this classifier.

        Each argument should be a dict with keys: 'name', 'type', 'default', 'help'
        Subclasses should override to define their specific parameters.

        Returns:
            List of argument definition dicts

        Example:
            [
                {'name': '--n-estimators', 'type': int, 'default': 100,
                 'help': 'Number of trees'},
                {'name': '--max-depth', 'type': int, 'default': 20,
                 'help': 'Maximum depth of trees'}
            ]
        """
        return []

    @classmethod
    def handles_nan(cls):
        """Return whether this classifier can handle NaN values natively.

        Subclasses should override this if they can handle NaN values.
        If False, the framework will automatically add imputation via Pipeline.

        Returns:
            bool: True if classifier handles NaN, False if imputation needed

        Example:
            RandomForest handles NaN → True
            LogisticRegression needs imputation → False
        """
        return False

    @abstractmethod
    def create_model(self):
        """Create and return a new model instance.

        Returns:
            Untrained model instance
        """
        pass

    def create_pipeline(self, X=None):
        """Create sklearn Pipeline with preprocessing, imputation, and classification.

        The pipeline always includes:
        1. ColumnTransformer - one-hot encodes categorical features
        2. SimpleImputer - imputes missing values
        3. Classifier - the actual model

        Args:
            X: Optional DataFrame to determine columns from. If None, uses all features from data.

        Returns:
            Pipeline instance
        """
        # Ensure features are prepared
        self.data.prepare_features()

        # Get categorical and numerical feature columns
        all_categorical_cols = self.data.get_categorical_feature_columns()
        all_numerical_cols = self.data.get_numerical_feature_columns()

        # If X is provided, filter to only columns present in X
        if X is not None:
            available_cols = set(X.columns)
            categorical_cols = [c for c in all_categorical_cols if c in available_cols]
            numerical_cols = [c for c in all_numerical_cols if c in available_cols]
        else:
            categorical_cols = all_categorical_cols
            numerical_cols = all_numerical_cols

        # Create preprocessing transformer
        # OneHotEncoder for categorical, passthrough for numerical
        transformers = []
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
        if numerical_cols:
            transformers.append(('num', 'passthrough', numerical_cols))

        preprocessor = ColumnTransformer(transformers=transformers)

        # Create imputer with configurable strategy
        if self.impute_strategy == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=self.impute_fill_value)
        else:
            imputer = SimpleImputer(strategy=self.impute_strategy)

        # Create classifier model
        model = self.create_model()

        # Build pipeline
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('imputer', imputer),
            ('classifier', model)
        ])

        return pipeline

    def _get_feature_names_from_pipeline(self, pipeline):
        """Extract feature names after preprocessing step in pipeline.

        Args:
            pipeline: Fitted Pipeline instance

        Returns:
            List of feature names
        """
        preprocessor = pipeline.named_steps['preprocessing']
        return list(preprocessor.get_feature_names_out())

    def train(self):
        """Train model on all available data.

        Returns:
            Trained Pipeline instance
        """
        print(f"\n=== Training Final Model ===")
        print(f"Parameters: {self.model_params}\n")

        X, y = self.data.get_X_y()
        self.model = self.create_pipeline()
        self.model.fit(X, y)

        print("Model trained successfully on full dataset")
        return self.model

    def cross_validate(self, cv_folds=5):
        """Run cross-validation to evaluate model performance.

        Args:
            cv_folds: Number of cross-validation folds

        Returns:
            Array of cross-validation scores
        """
        print(f"\n=== Running Cross-Validation ===")
        print(f"Parameters: {self.model_params}, cv_folds={cv_folds}\n")

        X, y = self.data.get_X_y()
        pipeline = self.create_pipeline()

        scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')

        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}")
        print(f"Mean Score: {scores.mean() * 100:.2f}%")

        return scores

    def calculate_feature_importance(self, n_repeats=10):
        """Calculate feature importance using permutation importance.

        Args:
            n_repeats: Number of times to permute each feature

        Returns:
            Series of feature importances sorted by value
        """
        print(f"\n=== Calculating Feature Importance ===")
        print(f"Parameters: {self.model_params}, n_repeats={n_repeats}\n")

        X, y = self.data.get_X_y()

        # Split data for holdout validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Train pipeline
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)

        # Calculate permutation importance
        # NOTE: Permutation importance operates on the INPUT features (raw),
        # not the transformed features (after one-hot encoding)
        result = permutation_importance(
            pipeline, X_val, y_val,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='accuracy'
        )

        # Use raw feature names (before preprocessing)
        feature_names = X_val.columns

        # Create DataFrame with results
        importances = pd.Series(result.importances_mean, index=feature_names)
        importances_std = pd.Series(result.importances_std, index=feature_names)

        # Sort by importance
        importances = importances.sort_values(ascending=False)

        # Print results
        print("Feature Importance (sorted by mean importance):\n")
        for feature in importances.index:
            mean_imp = importances[feature]
            std_imp = importances_std[feature]
            print(f"{feature:30s}: {mean_imp:.4f} (+/- {std_imp:.4f})")

        return importances

    def calculate_drop_column_importance(self, cv_folds=5, num_repeats=1):
        """Calculate feature importance by dropping each column.

        Measures performance drop when each feature is removed.

        Args:
            cv_folds: Number of cross-validation folds
            num_repeats: Number of times to repeat with different random seeds

        Returns:
            Series of feature importances sorted by value
        """
        print(f"\n=== Calculating Drop-Column Importance ===")
        print(f"Parameters: {self.model_params}, cv_folds={cv_folds}, num_repeats={num_repeats}")

        # Determine number of parallel jobs
        n_cores = multiprocessing.cpu_count()
        n_jobs = max(1, n_cores // 2)
        print(f"Using {n_jobs} parallel jobs (out of {n_cores} cores)\n")

        X, y = self.data.get_X_y()

        # Run repeats in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_drop_repeat)(
                repeat, self.random_state + repeat, X, y, cv_folds, num_repeats
            )
            for repeat in range(num_repeats)
        )

        # Collect results
        all_importances = {col: [] for col in X.columns}
        all_base_scores = []

        for repeat_importances, base_score in results:
            all_base_scores.append(base_score)
            for col, importance in repeat_importances.items():
                all_importances[col].append(importance)

        # Calculate mean and std
        mean_importances = {col: np.mean(vals) for col, vals in all_importances.items()}
        std_importances = {col: np.std(vals) for col, vals in all_importances.items()}

        importance_series = pd.Series(mean_importances).sort_values(ascending=False)

        # Print results
        print(f"\n=== Drop-Column Importance (sorted) ===")
        mean_base = np.mean(all_base_scores)
        std_base = np.std(all_base_scores)
        if num_repeats > 1:
            print(f"Baseline with all features: {mean_base:.4f} (+/- {std_base:.4f})\n")
        else:
            print(f"Baseline with all features: {mean_base:.4f}\n")

        for feature in importance_series.index:
            imp = mean_importances[feature]
            if num_repeats > 1:
                std = std_importances[feature]
                print(f"{feature:40s}: {imp:+.4f} (+/- {std:.4f})")
            else:
                print(f"{feature:40s}: {imp:+.4f}")

        return importance_series

    def _run_single_drop_repeat(self, repeat, seed, X, y, cv_folds, num_repeats):
        """Helper for parallel drop-column importance calculation.

        Args:
            repeat: Repeat index
            seed: Random seed for this repeat
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            num_repeats: Total number of repeats (for printing)

        Returns:
            Tuple of (importances dict, baseline score)
        """
        if num_repeats > 1:
            print(f"\n--- Repeat {repeat + 1}/{num_repeats} (seed={seed}) ---")

        # Create CV splitter
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        # Get baseline score
        print(f"Computing baseline score with all {len(X.columns)} features...")
        # Temporarily override random_state for this repeat
        original_random_state = self.random_state
        self.random_state = seed
        pipeline = self.create_pipeline(X)
        self.random_state = original_random_state
        base_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        base_score = base_scores.mean()

        print(f"Baseline accuracy: {base_score:.4f} (+/- {base_scores.std():.4f})\n")
        print(f"Computing drop-column importance for each feature...")

        # Calculate importance by dropping each column
        repeat_importances = {}
        for i, col in enumerate(X.columns, 1):
            reduced_X = X.drop(columns=[col])
            # Temporarily override random_state for this repeat
            self.random_state = seed
            pipeline = self.create_pipeline(reduced_X)
            self.random_state = original_random_state
            cv_scores = cross_val_score(pipeline, reduced_X, y, cv=cv, scoring='accuracy')
            score = cv_scores.mean()
            importance = base_score - score

            repeat_importances[col] = importance
            print(f"  [{i}/{len(X.columns)}] {col}: {importance:+.4f} (score without: {score:.4f})")

        return repeat_importances, base_score

    def plot_partial_dependence(self, features=None, output_file='partial_dependence.png'):
        """Generate partial dependence plots.

        Args:
            features: List of features to plot, or None for auto-detection
            output_file: Path to save the plot

        Returns:
            PartialDependenceDisplay object
        """
        print(f"\n=== Generating Partial Dependence Plots ===")
        print(f"Parameters: {self.model_params}\n")

        X, y = self.data.get_X_y()

        # Train pipeline
        pipeline = self.create_pipeline()
        pipeline.fit(X, y)

        # Determine which features to plot
        if features:
            plot_features = features
        else:
            # Auto-detect numerical features (exclude categorical)
            categorical_cols = self.data.get_categorical_feature_columns()
            numerical_cols = self.data.get_numerical_feature_columns()

            # Only plot continuous numerical features (not binary indicators)
            plot_features = []
            for col in numerical_cols:
                if col not in categorical_cols:
                    unique_vals = X[col].dropna().unique()
                    # Include if more than 2 unique values (not binary)
                    if len(unique_vals) > 2:
                        plot_features.append(col)

            print(f"Auto-detected {len(plot_features)} numerical features")

        # Filter to valid features
        valid_features = [f for f in plot_features if f in X.columns]

        if not valid_features:
            print(f"Error: None of the specified features exist in the dataset")
            print(f"Available features: {list(X.columns)}")
            return

        print(f"Plotting partial dependence for: {valid_features}\n")

        # Calculate figure size
        n_features = len(valid_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig_width = n_cols * 6
        fig_height = n_rows * 5

        # Create plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        display = PartialDependenceDisplay.from_estimator(
            pipeline, X, valid_features,
            kind='average',
            random_state=self.random_state,
            ax=ax
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Partial dependence plot saved to: {output_file}")
        plt.show()

        return display
