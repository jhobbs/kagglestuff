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
from joblib import Parallel, delayed
import multiprocessing

from base_data import BinaryClassificationData


class BinaryClassifier(ABC):
    """Base class for binary classifiers.

    Provides training, evaluation, and analysis methods for binary classification
    models. Subclasses implement specific model types.
    """

    def __init__(self, data: BinaryClassificationData, random_state=42, **model_params):
        """Initialize classifier with data and parameters.

        Args:
            data: BinaryClassificationData instance
            random_state: Random seed for reproducibility
            **model_params: Additional parameters to pass to the model
        """
        self.data = data
        self.random_state = random_state
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

    def _create_model_with_pipeline(self):
        """Create model, optionally wrapped in Pipeline with imputation.

        If data.impute_nans is True, wraps the model in a Pipeline that:
        - Uses mean imputation for all columns (after get_dummies, all are numeric)

        Returns:
            Either a Pipeline or raw model, depending on imputation needs
        """
        model = self.create_model()

        if not self.data.impute_nans:
            return model

        # After get_dummies() and conversion to float, all columns are numeric
        # Use simple mean imputation for all columns
        imputer = SimpleImputer(strategy='mean')

        # Wrap in pipeline
        pipeline = Pipeline([
            ('imputer', imputer),
            ('classifier', model)
        ])

        return pipeline

    def train(self):
        """Train model on all available data.

        Returns:
            Trained model instance
        """
        print(f"\n=== Training Final Model ===")
        print(f"Parameters: {self.model_params}\n")

        X, y = self.data.get_X_y()
        self.model = self._create_model_with_pipeline()
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
        model = self._create_model_with_pipeline()

        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

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

        # Train model
        model = self._create_model_with_pipeline()
        model.fit(X_train, y_train)

        # Calculate permutation importance
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='accuracy'
        )

        # Create DataFrame with results
        importances = pd.Series(result.importances_mean, index=X_val.columns)
        importances_std = pd.Series(result.importances_std, index=X_val.columns)

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
        model = self._create_model_with_pipeline()
        self.random_state = original_random_state
        base_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        base_score = base_scores.mean()

        print(f"Baseline accuracy: {base_score:.4f} (+/- {base_scores.std():.4f})\n")
        print(f"Computing drop-column importance for each feature...")

        # Calculate importance by dropping each column
        repeat_importances = {}
        for i, col in enumerate(X.columns, 1):
            reduced_X = X.drop(columns=[col])
            # Temporarily override random_state for this repeat
            self.random_state = seed
            model = self._create_model_with_pipeline()
            self.random_state = original_random_state
            cv_scores = cross_val_score(model, reduced_X, y, cv=cv, scoring='accuracy')
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

        # Train model
        model = self._create_model_with_pipeline()
        model.fit(X, y)

        # Determine which features to plot
        if features:
            plot_features = features
        else:
            # Auto-detect numerical features (not one-hot encoded)
            plot_features = []
            for col in X.columns:
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) > 2 or not set(unique_vals).issubset({0.0, 1.0}):
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
            model, X, valid_features,
            kind='average',
            random_state=self.random_state,
            ax=ax
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Partial dependence plot saved to: {output_file}")
        plt.show()

        return display
