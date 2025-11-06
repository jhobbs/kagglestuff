"""Logistic Regression classifier implementation (example)."""

from sklearn.linear_model import LogisticRegression
from base_classifier import BinaryClassifier
from base_data import BinaryClassificationData


class LogisticRegressionBinaryClassifier(BinaryClassifier):
    """Logistic Regression implementation of binary classifier.

    This is an example to demonstrate how the dynamic CLI argument
    system works with different classifier types.
    """

    def __init__(self, data: BinaryClassificationData, C=1.0, max_iter=100,
                 random_state=42, **kwargs):
        """Initialize Logistic Regression classifier.

        Args:
            data: BinaryClassificationData instance
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed
            **kwargs: Additional parameters for LogisticRegression
        """
        super().__init__(data, random_state=random_state,
                        C=C, max_iter=max_iter, **kwargs)

    @classmethod
    def get_cli_arguments(cls):
        """Return Logistic Regression specific CLI arguments.

        Returns:
            List of argument definition dicts for LR parameters
        """
        return [
            {
                'name': '--C',
                'type': float,
                'default': 1.0,
                'help': 'Inverse of regularization strength (default: 1.0)'
            },
            {
                'name': '--max-iter',
                'type': int,
                'default': 100,
                'help': 'Maximum number of iterations (default: 100)'
            }
        ]

    def create_model(self):
        """Create a new LogisticRegression instance.

        Returns:
            LogisticRegression with configured parameters
        """
        return LogisticRegression(
            C=self.model_params.get('C', 1.0),
            max_iter=self.model_params.get('max_iter', 100),
            random_state=self.random_state
        )
