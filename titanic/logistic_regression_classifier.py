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
                 random_state=None, impute_strategy='mean', impute_fill_value=None, **kwargs):
        """Initialize Logistic Regression classifier.

        Args:
            data: BinaryClassificationData instance
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed (None for random)
            impute_strategy: Strategy for imputing missing values
            impute_fill_value: Fill value when impute_strategy='constant'
            **kwargs: Additional parameters for LogisticRegression
        """
        super().__init__(data, random_state=random_state,
                        impute_strategy=impute_strategy, impute_fill_value=impute_fill_value,
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
            },
            {
                'name': '--impute-strategy',
                'type': str,
                'default': 'mean',
                'help': 'Strategy for imputing missing values: mean, median, most_frequent, constant (default: mean)'
            },
            {
                'name': '--impute-fill-value',
                'type': float,
                'default': None,
                'help': 'Fill value when impute-strategy=constant (default: None)'
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
