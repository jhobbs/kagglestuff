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
                 random_state=None, **kwargs):
        """Initialize Logistic Regression classifier.

        Args:
            data: BinaryClassificationData instance (provides both data and preprocessor)
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed (None for random)
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

    @classmethod
    def get_param_grid(cls, search_type='default'):
        """Return Logistic Regression parameter grid for hyperparameter search.

        Args:
            search_type: Type of search space:
                - 'narrow': Small search space (fast)
                - 'default': Moderate search space (recommended)
                - 'wide': Large search space (comprehensive)

        Returns:
            Dict mapping parameter names to lists of values
        """
        if search_type == 'narrow':
            # Quick search around common values
            return {
                'classifier__C': [0.5, 1.0, 2.0],
                'classifier__max_iter': [100, 200, 500]
            }
        elif search_type == 'default':
            # Moderate search space
            return {
                'classifier__C': [0.01, 0.1, 0.5, 1.0, 2.0, 10.0],
                'classifier__max_iter': [100, 200, 500, 1000]
            }
        elif search_type == 'wide':
            # Comprehensive search space
            return {
                'classifier__C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
                'classifier__max_iter': [100, 200, 500, 1000, 2000],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        else:
            raise ValueError(f"search_type must be 'narrow', 'default', or 'wide', got '{search_type}'")

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
