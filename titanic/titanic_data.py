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
        self.all_lastnames = None  # All unique lastnames from train+test
        self.all_ticket_tokens = None  # All unique ticket tokens from train+test
        self.all_decks = None  # All unique decks from train+test

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
        categorical_features = ["Embarked"]

        # Numeric features (all others, including binary indicators, ticket tokens, and deck indicators)
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
            "Same_lastname_count", "Fare", "Embarked", "Cabin_count",
            "Ticket_number", "Ticket_number_length"
        ]

        # Add dynamically created ticket token features
        ticket_token_cols = [col for col in self.processed_data.columns
                            if col.startswith('ticket_token_')]

        # Add dynamically created deck features (but not Deck_raw which is intermediate)
        deck_cols = [col for col in self.processed_data.columns
                    if col.startswith('Deck_') and col != 'Deck_raw']

        return base_features + ticket_token_cols + deck_cols

    def get_categorical_feature_columns(self):
        """Return list of categorical feature columns (object dtype).

        These columns will be one-hot encoded by the Pipeline.
        """
        return ["Embarked"]

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
        data["Deck_raw"] = data['Cabin'].str[0]

        # Ticket features
        data["Ticket_number"] = data["Ticket"].apply(self._parse_ticket)
        data["Ticket_number_length"] = data["Ticket_number"].apply(
            lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan
        )

        # Ticket token features (boolean indicators for ticket prefixes)
        # Use stored tokens if available (from train+test combined), otherwise discover
        if use_stored_tokens and self.ticket_tokens is not None:
            # Use stored tokens from training data (legacy behavior)
            all_tokens = self.ticket_tokens
        elif self.all_ticket_tokens is not None:
            # Use all tokens from train+test combined (preferred)
            all_tokens = self.all_ticket_tokens
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

        # Deck features (boolean indicators for each deck letter)
        # Use stored decks if available (from train+test combined), otherwise discover
        if self.all_decks is not None:
            # Use all decks from train+test combined
            all_decks = self.all_decks
        else:
            # Discover decks from this data
            all_decks = sorted(data["Deck_raw"].unique())

        for deck in all_decks:
            feature_name = f"Deck_{deck}"
            data[feature_name] = (data["Deck_raw"] == deck).astype(int)

    def prepare_for_submission(self, test_filepath):
        """Prepare for submission by computing combined statistics from train+test.

        Computes lastname counts, ticket tokens, and deck values from the combined
        train+test datasets to ensure consistent feature space across both datasets.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            None
        """
        self.compute_combined_lastname_counts(test_filepath)
        self.compute_all_categorical_values(test_filepath)

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

    def compute_all_categorical_values(self, test_filepath):
        """Compute all unique categorical values from combined train+test data.

        This identifies the complete space of categorical values for last names,
        ticket tokens, and decks across the entire passenger manifest.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            Dict with keys 'lastnames', 'ticket_tokens', 'decks' containing sets/lists
        """
        # Load both datasets
        train_data = pd.read_csv(self.filepath)
        test_data = pd.read_csv(test_filepath)

        # Extract last names from both
        train_lastnames = train_data["Name"].str.split(',').str[0]
        test_lastnames = test_data["Name"].str.split(',').str[0]
        all_lastnames = pd.concat([train_lastnames, test_lastnames])
        unique_lastnames = sorted(all_lastnames.unique())

        # Extract ticket tokens from both
        all_tokens = set()
        for ticket in pd.concat([train_data["Ticket"], test_data["Ticket"]]):
            all_tokens.update(self._tokenize_ticket(ticket))
        unique_tokens = sorted(all_tokens)

        # Extract decks from both
        train_cabins = train_data["Cabin"].fillna("Unknown")
        test_cabins = test_data["Cabin"].fillna("Unknown")
        train_decks = train_cabins.str[0]
        test_decks = test_cabins.str[0]
        all_decks = pd.concat([train_decks, test_decks])
        unique_decks = sorted(all_decks.unique())

        # Store for later use
        self.all_lastnames = unique_lastnames
        self.all_ticket_tokens = unique_tokens
        self.all_decks = unique_decks

        print(f"\nComputed categorical values from {len(train_data)} train + {len(test_data)} test passengers")
        print(f"  Last names: {len(unique_lastnames)} unique")
        print(f"  Ticket tokens: {len(unique_tokens)} unique")
        print(f"  Decks: {len(unique_decks)} unique")

        return {
            'lastnames': unique_lastnames,
            'ticket_tokens': unique_tokens,
            'decks': unique_decks
        }

    def load_and_prepare_test(self, test_filepath):
        """Load and prepare test data using the same features as training data.

        IMPORTANT: Must be called AFTER training data has been prepared, so that
        categorical values are already discovered.

        Args:
            test_filepath: Path to test CSV file

        Returns:
            Tuple of (X_test, passenger_ids) where X_test is the feature DataFrame
            and passenger_ids is the PassengerId column
        """
        # Check if categorical values have been computed
        # (either from prepare_for_submission or from training data engineering)
        if self.ticket_tokens is None and self.all_ticket_tokens is None:
            raise ValueError("Must prepare training data first to discover categorical values")

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

    def inspect_data(self, test_filepath='./test.csv'):
        """Inspect Titanic data quality and show categorical value spaces.

        Extends base inspection with Titanic-specific categorical values from
        train+test combined datasets.

        Args:
            test_filepath: Path to test CSV file (default: ./test.csv)

        Returns:
            Dict containing inspection results
        """
        # Call parent inspection first
        result = super().inspect_data()

        # Try to load test data and show categorical values
        import os
        if os.path.exists(test_filepath):
            print(f"\n{'='*80}")
            print("CATEGORICAL VALUE SPACE (from train + test combined)")
            print("="*80)

            categorical_values = self.compute_all_categorical_values(test_filepath)

            # Display last names
            print(f"\n--- ALL LAST NAMES ({len(categorical_values['lastnames'])}) ---")
            # Show in columns for readability
            lastnames = categorical_values['lastnames']
            num_cols = 4
            for i in range(0, len(lastnames), num_cols):
                row = lastnames[i:i+num_cols]
                print("  " + "  |  ".join(f"{name:20s}" for name in row))

            # Display ticket tokens
            print(f"\n--- ALL TICKET TOKENS ({len(categorical_values['ticket_tokens'])}) ---")
            print("  " + ", ".join(categorical_values['ticket_tokens']))

            # Display decks
            print(f"\n--- ALL DECKS ({len(categorical_values['decks'])}) ---")
            print("  " + ", ".join(categorical_values['decks']))

            result['categorical_values'] = categorical_values
        else:
            print(f"\n{'='*80}")
            print("CATEGORICAL VALUE SPACE")
            print("="*80)
            print(f"\nTest file not found at: {test_filepath}")
            print("Run with --test-data flag to see combined train+test categorical values")

        return result

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
