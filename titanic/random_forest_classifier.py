"""Random Forest classifier implementation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from base_classifier import BinaryClassifier
from base_data import BinaryClassificationData


class RandomForestBinaryClassifier(BinaryClassifier):
    """Random Forest implementation of binary classifier."""

    def __init__(self, data: BinaryClassificationData, n_estimators=300, max_depth=10,
                 min_samples_leaf=2, min_samples_split=4, max_features='sqrt',
                 oob_score=True, class_weight=None, random_state=None, **kwargs):
        """Initialize Random Forest classifier.

        Args:
            data: BinaryClassificationData instance (provides both data and preprocessor)
            n_estimators: Number of trees in the forest (default: 300)
            max_depth: Maximum depth of trees (default: 10)
            min_samples_leaf: Minimum samples required at leaf node (default: 2)
            min_samples_split: Minimum samples required to split (default: 4)
            max_features: Number of features to consider for splits (default: 'sqrt')
            oob_score: Whether to use out-of-bag samples to estimate accuracy (default: True)
            class_weight: Weights for classes - None or 'balanced' (default: None)
            random_state: Random seed (None for random)
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__(data, random_state=random_state,
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                        max_features=max_features, oob_score=oob_score,
                        class_weight=class_weight, **kwargs)

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
                'default': 300,
                'help': 'Number of trees in random forest (default: 300)'
            },
            {
                'name': '--max-depth',
                'type': int,
                'default': 10,
                'help': 'Maximum depth of trees (default: 10)'
            },
            {
                'name': '--min-samples-leaf',
                'type': int,
                'default': 2,
                'help': 'Minimum samples required at leaf node (default: 2)'
            },
            {
                'name': '--min-samples-split',
                'type': int,
                'default': 4,
                'help': 'Minimum samples required to split node (default: 4)'
            },
            {
                'name': '--max-features',
                'type': str,
                'default': 'sqrt',
                'help': 'Number of features for splits: int, float, "sqrt", "log2", or None (default: sqrt)'
            },
            {
                'name': '--oob-score',
                'type': lambda x: x.lower() == 'true',
                'default': True,
                'help': 'Use out-of-bag samples to estimate accuracy (default: True)'
            },
            {
                'name': '--class-weight',
                'type': str,
                'default': None,
                'help': 'Class weights: None or "balanced" (default: None)'
            }
        ]

    def create_model(self):
        """Create a new RandomForestClassifier instance.

        Returns:
            RandomForestClassifier with configured parameters
        """
        return RandomForestClassifier(
            n_estimators=self.model_params.get('n_estimators', 300),
            max_depth=self.model_params.get('max_depth', 10),
            min_samples_leaf=self.model_params.get('min_samples_leaf', 2),
            min_samples_split=self.model_params.get('min_samples_split', 4),
            max_features=self.model_params.get('max_features', 'sqrt'),
            bootstrap=True,
            oob_score=self.model_params.get('oob_score', True),
            class_weight=self.model_params.get('class_weight', None),
            random_state=self.random_state
        )

    def display_post_train_stats(self):
        """Display Random Forest specific training statistics.

        Shows training accuracy, OOB score, and the gap between them.
        """
        X, y = self.data.get_X_y()
        train_score = self.model.score(X, y)
        print(f"Training accuracy: {train_score:.4f}")

        # Display OOB score if available
        classifier = self.model.named_steps['classifier']
        if hasattr(classifier, 'oob_score_'):
            oob_score = classifier.oob_score_
            gap = train_score - oob_score
            print(f"OOB accuracy:      {oob_score:.4f}")
            print(f"Train-OOB gap:     {gap:+.4f} (positive = overfit)")

    def get_per_fold_metrics(self, cv_results):
        """Extract OOB scores from each fold's fitted Random Forest.

        Args:
            cv_results: Results dict from cross_validate (includes 'estimator' key)

        Returns:
            Dict with 'oob' key containing array of OOB scores per fold
        """
        if not self.model_params.get('oob_score', True):
            return {}

        oob_scores = []
        for estimator in cv_results['estimator']:
            # Extract the RF classifier from the pipeline
            classifier = estimator.named_steps['classifier']
            # Get OOB score if available, otherwise use NaN
            oob = getattr(classifier, 'oob_score_', np.nan)
            oob_scores.append(oob)

        return {'oob': np.array(oob_scores)}
