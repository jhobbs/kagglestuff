"""Base classes for binary classification data handling."""

from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BinaryClassificationData(ABC):
    """Base class for binary classification datasets.

    Handles data loading, cleaning, feature engineering, and preparation.
    Subclasses must implement dataset-specific logic.
    """

    def __init__(self, filepath, excluded_features=None):
        """Initialize with path to data file.

        Args:
            filepath: Path to CSV file containing the data
            excluded_features: List of feature names to exclude from the model (default: None)
        """
        self.filepath = filepath
        self.excluded_features = excluded_features if excluded_features is not None else []
        self.raw_data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self._prepared = False

    def prepare_for_submission(self, test_filepath):
        """Prepare data class for submission by computing stats from train+test combined.

        This is called before training when generating competition submissions.
        Subclasses can override to compute features that need the full dataset.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            None
        """
        # Default implementation does nothing
        pass

    @abstractmethod
    def load_data(self):
        """Load raw data, engineer features, and return prepared (X, y).

        This method should:
        1. Load CSV into self.raw_data
        2. Engineer features into self.processed_data
        3. Extract X (features) and y (target)
        4. Store in self.X and self.y
        5. Return (X, y) tuple

        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        pass

    @abstractmethod
    def engineer_features(self):
        """Engineer features from raw data.

        Should modify self.processed_data in place to add new features.
        """
        pass

    @abstractmethod
    def get_feature_columns(self):
        """Return list of feature column names to use in model.

        Returns:
            List of column names
        """
        pass

    @abstractmethod
    def get_target_column(self):
        """Return name of target column.

        Returns:
            String column name
        """
        pass

    @abstractmethod
    def get_preprocessor(self, feature_columns=None):
        """Return unfitted sklearn ColumnTransformer for preprocessing features.

        The ColumnTransformer should define all preprocessing steps including:
        - Imputation strategies for numeric and categorical features
        - Scaling/normalization for numeric features
        - One-hot encoding for categorical features

        Args:
            feature_columns: Optional list of feature column names to include.
                           If None, uses all available features. This allows
                           creating preprocessors for subsets of features
                           (e.g., for drop-column importance testing).

        Returns:
            Unfitted sklearn ColumnTransformer instance
        """
        pass

    def get_X_y(self):
        """Return prepared feature matrix and target vector.

        Calls load_data() if data hasn't been loaded yet.
        Filters out excluded features if any are specified.

        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        if not self._prepared:
            self.load_data()
            self._prepared = True

        # Filter out excluded features
        if self.excluded_features:
            features_to_drop = [f for f in self.excluded_features if f in self.X.columns]
            if features_to_drop:
                X_filtered = self.X.drop(columns=features_to_drop)
                return X_filtered, self.y

        return self.X, self.y

    def get_numeric_columns(self):
        """Return list of numeric column names from prepared features.

        Returns:
            List of column names that are numeric/continuous
        """
        X, _ = self.get_X_y()

        # Numeric columns are those with float dtype
        numeric_cols = X.select_dtypes(include=['float64', 'float32']).columns.tolist()
        return numeric_cols

    def get_categorical_columns(self):
        """Return list of categorical column names from prepared features.

        This includes one-hot encoded features and integer features.

        Returns:
            List of column names that are categorical/discrete
        """
        X, _ = self.get_X_y()

        # Categorical columns are those with int dtype or object dtype
        # (though after get_dummies, most will be float)
        # For our purposes, columns not in numeric are categorical
        all_cols = set(X.columns)
        numeric_cols = set(self.get_numeric_columns())
        categorical_cols = list(all_cols - numeric_cols)
        return categorical_cols

    def plot_correlation_matrix(self, output_file='correlation_matrix.png'):
        """Generate and save feature correlation heatmap.

        Args:
            output_file: Path to save the plot

        Returns:
            DataFrame containing the correlation matrix
        """
        print(f"\n=== Generating Feature Correlation Matrix ===\n")

        X, _ = self.get_X_y()

        # Apply one-hot encoding for correlation calculation
        X_encoded = pd.get_dummies(X)
        print(f"Computing correlations for {len(X_encoded.columns)} features...")

        # Calculate correlation matrix
        corr = X_encoded.corr()

        # Create figure
        fig_size = max(10, len(X_encoded.columns) * 0.3)
        plt.figure(figsize=(fig_size, fig_size))

        # Create heatmap
        sns.heatmap(
            corr,
            cmap='coolwarm',
            center=0,
            annot=len(X_encoded.columns) <= 20,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
        )

        plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Correlation matrix saved to: {output_file}")
        plt.show()

        # Print highly correlated feature pairs
        print(f"\nHighly correlated feature pairs (|correlation| > 0.7):\n")
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        if high_corr_pairs:
            for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1:30s} <-> {feat2:30s}: {corr_val:+.3f}")
        else:
            print("  No highly correlated pairs found.")

        return corr

    def inspect_data(self):
        """Inspect data quality and print detailed statistics.

        Shows information about NaN values, data shape, and feature distributions.

        Returns:
            Dict containing inspection results
        """
        print("="*80)
        print("DATA INSPECTION")
        print("="*80)

        X, y = self.get_X_y()

        # Check for NaN values
        nan_counts = X.isna().sum()
        nan_columns = nan_counts[nan_counts > 0].sort_values(ascending=False)

        print(f"\n{'='*80}")
        print("NaN VALUE ANALYSIS")
        print("="*80)

        if len(nan_columns) > 0:
            print(f"\nFound {len(nan_columns)} columns with NaN values:\n")
            for col, count in nan_columns.items():
                pct = (count / len(X)) * 100
                print(f"  {col:40s}: {count:4d} NaNs ({pct:5.2f}%)")

            print(f"\nFeatures with NaN: {len(nan_columns)}/{len(X.columns)}")
            print(f"Features without NaN: {len(X.columns) - len(nan_columns)}/{len(X.columns)}")
        else:
            print("\nNo NaN values found in any column!")

        # Show summary statistics
        print(f"\n{'='*80}")
        print("DATA SHAPE")
        print("="*80)
        print(f"Rows: {len(X)}")
        print(f"Columns: {len(X.columns)}")
        print(f"Total cells: {len(X) * len(X.columns)}")
        total_nans = X.isna().sum().sum()
        print(f"Total NaN cells: {total_nans}")
        if len(X) * len(X.columns) > 0:
            pct_nan = (total_nans / (len(X) * len(X.columns))) * 100
            print(f"Percentage NaN: {pct_nan:.2f}%")

        # Target variable info
        print(f"\n{'='*80}")
        print("TARGET VARIABLE")
        print("="*80)
        print(f"Name: {self.get_target_column()}")
        print(f"Class distribution:")
        for value, count in y.value_counts().sort_index().items():
            pct = (count / len(y)) * 100
            print(f"  {value}: {count} ({pct:.2f}%)")

        # Show sample of columns with NaNs
        if len(nan_columns) > 0:
            print(f"\n{'='*80}")
            print("SAMPLE OF COLUMNS WITH NaN VALUES (first 10 rows)")
            print("="*80)
            sample_cols = list(nan_columns.head(5).index)
            print(X[sample_cols].head(10))

        return {
            'shape': X.shape,
            'nan_columns': nan_columns.to_dict(),
            'total_nans': total_nans,
            'target_distribution': y.value_counts().to_dict()
        }
