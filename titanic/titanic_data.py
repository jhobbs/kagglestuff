"""Titanic-specific data handling and feature engineering."""

import re
import numpy as np
import pandas as pd
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

    def load_data(self):
        """Load Titanic training data from CSV."""
        self.raw_data = pd.read_csv(self.filepath)
        return self.raw_data

    def get_target_column(self):
        """Return target column name."""
        return "Survived"

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
