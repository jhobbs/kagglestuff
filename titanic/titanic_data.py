"""Titanic-specific data handling and feature engineering."""

import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from base_data import BinaryClassificationData


class TitanicData(BinaryClassificationData):
    """Titanic dataset implementation with domain-specific feature engineering."""

    def __init__(self, filepath):
        """Initialize with path to data file.

        Args:
            filepath: Path to CSV file containing the data
        """
        super().__init__(filepath)
        self.ticket_tokens = None  # Store discovered ticket tokens from training data
        self.lastname_counts = None  # Store lastname counts from combined train+test data

    def load_data(self):
        """Load Titanic training data, engineer features, and return (X, y).

        Returns:
            Tuple of (X, y) where X is feature DataFrame and y is target Series
        """
        # Load CSV
        self.raw_data = pd.read_csv(self.filepath)

        # Engineer features
        self.processed_data = self.raw_data.copy()
        self.engineer_features()

        # Get target
        self.y = self.processed_data[self.get_target_column()]

        # Get feature columns
        feature_cols = self.get_feature_columns()

        # Create feature matrix (raw features, preprocessing will be done by ColumnTransformer)
        self.X = self.processed_data[feature_cols].copy()

        return self.X, self.y

    def get_target_column(self):
        """Return target column name."""
        return "Survived"

    def get_preprocessor(self, feature_columns=None):
        """Return unfitted ColumnTransformer for Titanic feature preprocessing.

        Args:
            feature_columns: Optional list of feature column names to include.
                           If None, uses all available features.

        Returns:
            Unfitted sklearn ColumnTransformer with pipelines for numeric and categorical features
        """
        # Categorical features (object dtype, need one-hot encoding)
        categorical_features = ["Embarked", "Deck"]

        # Numeric features (all others, including binary indicators and ticket tokens)
        # Note: Must be called after feature engineering to know all columns
        if self.processed_data is None:
            raise ValueError("Must engineer features before creating preprocessor")

        # Determine which features to use
        if feature_columns is None:
            all_features = self.get_feature_columns()
        else:
            all_features = feature_columns

        # Filter categorical and numeric features to only those present
        categorical_features = [f for f in categorical_features if f in all_features]
        numeric_features = [f for f in all_features if f not in categorical_features]

        # Numeric pipeline: impute with mean, then scale
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: impute with most frequent, then one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )

        return preprocessor

    def get_feature_columns(self):
        """Return base feature columns plus any dynamically created ones."""
        base_features = [
            "Pclass", "Male", "SibSp", "Parch", "Age", "Name_length",
            "Same_lastname_count", "Fare", "Embarked", "Cabin_count", "Deck",
            "Ticket_number", "Ticket_number_length"
        ]

        # Add dynamically created ticket token features
        ticket_token_cols = [col for col in self.processed_data.columns
                            if col.startswith('ticket_token_')]

        return base_features + ticket_token_cols

    def get_categorical_feature_columns(self):
        """Return list of categorical feature columns (object dtype).

        These columns will be one-hot encoded by the Pipeline.
        """
        return ["Embarked", "Deck"]

    def get_numerical_feature_columns(self):
        """Return list of numerical feature columns.

        These columns will pass through the Pipeline without encoding.
        """
        # Must be called after feature engineering
        if self.processed_data is None:
            raise ValueError("Must call engineer_features() before get_numerical_feature_columns()")

        # Get all features except categorical ones
        all_features = self.get_feature_columns()
        categorical = self.get_categorical_feature_columns()
        numerical = [f for f in all_features if f not in categorical]
        return numerical

    def engineer_features(self, use_stored_tokens=False):
        """Create Titanic-specific features.

        Args:
            use_stored_tokens: If True, use self.ticket_tokens instead of discovering new ones.
                              Used for test data to ensure same features as training data.
        """
        data = self.processed_data

        # Gender feature
        data["Male"] = (data["Sex"] == "male").astype(int)

        # Name features
        data["Name_length"] = data["Name"].str.len()
        data["Last_name"] = data["Name"].str.split(',').str[0]

        # Use stored lastname counts if available (from combined train+test),
        # otherwise compute from this dataset only
        if self.lastname_counts is not None:
            data["Same_lastname_count"] = data["Last_name"].map(self.lastname_counts).fillna(1)
        else:
            lastname_counts = data["Last_name"].value_counts()
            data["Same_lastname_count"] = data["Last_name"].map(lastname_counts)

        # Cabin features
        data["Cabin"] = data["Cabin"].fillna("Unknown")
        data["Cabin_count"] = data['Cabin'].str.split().str.len()
        data["Deck"] = data['Cabin'].str[0]

        # Ticket features
        data["Ticket_number"] = data["Ticket"].apply(self._parse_ticket)
        data["Ticket_number_length"] = data["Ticket_number"].apply(
            lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan
        )

        # Ticket token features (boolean indicators for ticket prefixes)
        if use_stored_tokens and self.ticket_tokens is not None:
            # Use stored tokens from training data
            all_tokens = self.ticket_tokens
        else:
            # Discover tokens from this data and store them
            all_tokens = set()
            for ticket in data["Ticket"]:
                all_tokens.update(self._tokenize_ticket(ticket))
            # Store tokens for later use with test data
            self.ticket_tokens = all_tokens

        for token in sorted(all_tokens):
            feature_name = f"ticket_token_{token}"
            data[feature_name] = data["Ticket"].apply(
                lambda x: 1 if token in self._tokenize_ticket(x) else 0
            )

    def prepare_for_submission(self, test_filepath):
        """Prepare for submission by computing combined lastname counts.

        Computes lastname counts from the combined train+test datasets to get
        accurate family size information across the entire passenger manifest.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            None
        """
        self.compute_combined_lastname_counts(test_filepath)

    def compute_combined_lastname_counts(self, test_filepath):
        """Compute lastname counts from combined train+test data.

        This gives the most accurate picture of family sizes across the entire
        passenger manifest. Should be called before training when doing submissions.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            Series mapping lastname to count
        """
        # Load both datasets
        train_data = pd.read_csv(self.filepath)
        test_data = pd.read_csv(test_filepath)

        # Extract last names from both
        train_lastnames = train_data["Name"].str.split(',').str[0]
        test_lastnames = test_data["Name"].str.split(',').str[0]

        # Combine and count
        all_lastnames = pd.concat([train_lastnames, test_lastnames])
        lastname_counts = all_lastnames.value_counts()

        # Store for use in feature engineering
        self.lastname_counts = lastname_counts

        print(f"Computed lastname counts from {len(train_data)} train + {len(test_data)} test passengers")
        print(f"Found {len(lastname_counts)} unique surnames")

        return lastname_counts

    def load_and_prepare_test(self, test_filepath):
        """Load and prepare test data using the same features as training data.

        IMPORTANT: Must be called AFTER training data has been prepared, so that
        ticket_tokens are already discovered.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            Tuple of (X_test, passenger_ids) where X_test is the feature DataFrame
            and passenger_ids is the PassengerId column
        """
        if self.ticket_tokens is None:
            raise ValueError("Must prepare training data first to discover ticket tokens")

        # Load test data
        test_data = pd.read_csv(test_filepath)

        # Store PassengerId before processing
        passenger_ids = test_data["PassengerId"].copy()

        # Apply same feature engineering as training data
        test_processed = test_data.copy()

        # Temporarily set processed_data to test data for feature engineering
        original_processed = self.processed_data
        self.processed_data = test_processed

        # Engineer features using stored tokens
        self.engineer_features(use_stored_tokens=True)

        # Get feature columns
        feature_cols = self.get_feature_columns()

        # Create feature matrix
        X_test = self.processed_data[feature_cols].copy()

        # Restore original processed_data
        self.processed_data = original_processed

        return X_test, passenger_ids

    @staticmethod
    def _tokenize_ticket(ticket):
        """Tokenize ticket string into set of alphabetic tokens.

        Args:
            ticket: Ticket string

        Returns:
            Set of lowercase tokens
        """
        if pd.isna(ticket):
            return set()
        tokens = re.findall(r'[a-zA-Z]+', str(ticket))
        return set(token.lower() for token in tokens)

    @staticmethod
    def _parse_ticket(ticket):
        """Parse ticket number from ticket string.

        Args:
            ticket: Ticket string

        Returns:
            Float ticket number or None if not parseable
        """
        if pd.isna(ticket):
            return None

        parts = ticket.split()
        if len(parts) == 2:
            number = parts[1]
        elif len(parts) == 3:
            number = parts[2]
        else:
            number = parts[0]

        try:
            return float(number)
        except:
            return None
