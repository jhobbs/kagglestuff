"""Titanic-specific data handling and feature engineering."""

import re
import numpy as np
import pandas as pd
from base_data import BinaryClassificationData


class TitanicData(BinaryClassificationData):
    """Titanic dataset implementation with domain-specific feature engineering."""

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

    def engineer_features(self):
        """Create Titanic-specific features."""
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
        all_tokens = set()
        for ticket in data["Ticket"]:
            all_tokens.update(self._tokenize_ticket(ticket))

        for token in sorted(all_tokens):
            feature_name = f"ticket_token_{token}"
            data[feature_name] = data["Ticket"].apply(
                lambda x: 1 if token in self._tokenize_ticket(x) else 0
            )

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
