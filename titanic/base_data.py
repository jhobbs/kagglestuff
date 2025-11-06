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

    def __init__(self, filepath):
        """Initialize with path to data file.

        Args:
            filepath: Path to CSV file containing the data
        """
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        self.X = None
        self.y = None
        self._prepared = False

    @abstractmethod
    def load_data(self):
        """Load raw data from file.

        Returns:
            DataFrame containing the raw data
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

    def prepare_features(self):
        """Prepare feature matrix X and target vector y.

        Loads data, engineers features, and creates model-ready arrays.
        """
        if self._prepared:
            return

        # Load data if not already loaded
        if self.raw_data is None:
            self.load_data()

        # Engineer features if not already done
        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()
            self.engineer_features()

        # Get target
        self.y = self.processed_data[self.get_target_column()]

        # Get feature columns
        feature_cols = self.get_feature_columns()

        # Create feature matrix with one-hot encoding
        self.X = pd.get_dummies(self.processed_data[feature_cols])

        # Convert all numeric columns to float to avoid sklearn warnings
        numeric_cols = self.X.select_dtypes(include=['int64', 'int32']).columns
        self.X[numeric_cols] = self.X[numeric_cols].astype(float)

        self._prepared = True

    def get_X_y(self):
        """Return prepared feature matrix and target vector.

        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        if not self._prepared:
            self.prepare_features()
        return self.X, self.y

    def plot_correlation_matrix(self, output_file='correlation_matrix.png'):
        """Generate and save feature correlation heatmap.

        Args:
            output_file: Path to save the plot

        Returns:
            DataFrame containing the correlation matrix
        """
        print(f"\n=== Generating Feature Correlation Matrix ===\n")

        X, _ = self.get_X_y()
        print(f"Computing correlations for {len(X.columns)} features...")

        # Calculate correlation matrix
        corr = X.corr()

        # Create figure
        fig_size = max(10, len(X.columns) * 0.3)
        plt.figure(figsize=(fig_size, fig_size))

        # Create heatmap
        sns.heatmap(
            corr,
            cmap='coolwarm',
            center=0,
            annot=len(X.columns) <= 20,
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
