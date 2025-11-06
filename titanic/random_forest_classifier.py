"""Random Forest classifier implementation."""

from sklearn.ensemble import RandomForestClassifier
from base_classifier import BinaryClassifier
from base_data import BinaryClassificationData


class RandomForestBinaryClassifier(BinaryClassifier):
    """Random Forest implementation of binary classifier."""

    def __init__(self, data: BinaryClassificationData, n_estimators=100, max_depth=20,
                 random_state=None, impute_strategy='mean', impute_fill_value=None, **kwargs):
        """Initialize Random Forest classifier.

        Args:
            data: BinaryClassificationData instance
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed (None for random)
            impute_strategy: Strategy for imputing missing values
            impute_fill_value: Fill value when impute_strategy='constant'
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__(data, random_state=random_state,
                        impute_strategy=impute_strategy, impute_fill_value=impute_fill_value,
                        n_estimators=n_estimators, max_depth=max_depth, **kwargs)

    @classmethod
    def get_cli_arguments(cls):
        """Return Random Forest specific CLI arguments.

        Returns:
            List of argument definition dicts for RF parameters
        """
        return [
            {
                'name': '--n-estimators',
                'type': int,
                'default': 100,
                'help': 'Number of trees in random forest (default: 100)'
            },
            {
                'name': '--max-depth',
                'type': int,
                'default': 20,
                'help': 'Maximum depth of trees (default: 20)'
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

    @classmethod
    def handles_nan(cls):
        """Random Forest can handle NaN values natively.

        Returns:
            bool: True (RandomForest handles NaN)
        """
        return True

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
