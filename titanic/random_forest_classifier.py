"""Random Forest classifier implementation."""

from sklearn.ensemble import RandomForestClassifier
from base_classifier import BinaryClassifier
from base_data import BinaryClassificationData


class RandomForestBinaryClassifier(BinaryClassifier):
    """Random Forest implementation of binary classifier."""

    def __init__(self, data: BinaryClassificationData, n_estimators=100, max_depth=20,
                 random_state=42, **kwargs):
        """Initialize Random Forest classifier.

        Args:
            data: BinaryClassificationData instance
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__(data, random_state=random_state,
                        n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    def create_model(self):
        """Create a new RandomForestClassifier instance.

        Returns:
            RandomForestClassifier with configured parameters
        """
        return RandomForestClassifier(
            n_estimators=self.model_params.get('n_estimators', 100),
            max_depth=self.model_params.get('max_depth', 20),
            random_state=self.random_state
        )
