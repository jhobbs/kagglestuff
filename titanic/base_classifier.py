"""Base classes for binary classification models."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import precision_recall_curve, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV
from joblib import Parallel, delayed
import multiprocessing

from base_data import BinaryClassificationData


class BinaryClassifier(ABC):
    """Base class for binary classifiers.

    Provides training, evaluation, and analysis methods for binary classification
    models. Subclasses implement specific model types.
    """

    def __init__(self, data: BinaryClassificationData, random_state=None, **model_params):
        """Initialize classifier with data and parameters.

        Args:
            data: BinaryClassificationData instance (provides both data and preprocessor)
            random_state: Random seed for reproducibility (None for random)
            **model_params: Additional parameters to pass to the model
        """
        self.data = data
        self.random_state = random_state
        self.model_params = model_params
        self.model = None

    @classmethod
    def get_cli_arguments(cls):
        """Return list of argparse argument definitions for this classifier.

        Each argument should be a dict with keys: 'name', 'type', 'default', 'help'
        Subclasses should override to define their specific parameters.

        Returns:
            List of argument definition dicts

        Example:
            [
                {'name': '--n-estimators', 'type': int, 'default': 100,
                 'help': 'Number of trees'},
                {'name': '--max-depth', 'type': int, 'default': 20,
                 'help': 'Maximum depth of trees'}
            ]
        """
        return []

    @classmethod
    def get_param_grid(cls, search_type='default'):
        """Return parameter grid for hyperparameter search.

        Subclasses should override to define their search space.

        Args:
            search_type: Type of search space - 'default', 'narrow', or 'wide'

        Returns:
            Dict mapping parameter names to lists of values to try

        Example:
            {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30, None]
            }
        """
        return {}

    @abstractmethod
    def create_model(self):
        """Create and return a new model instance.

        Returns:
            Untrained model instance
        """
        pass

    def create_pipeline(self, X=None):
        """Create sklearn Pipeline with preprocessing and classification.

        The pipeline includes:
        1. ColumnTransformer - handles imputation, scaling, and encoding (from data class)
        2. Classifier - the actual model

        Args:
            X: Optional DataFrame. If provided, only create preprocessor for columns in X.
               This is useful for drop-column importance testing.

        Returns:
            Pipeline instance
        """
        # Get preprocessor from data class
        # If X is provided, only use columns present in X
        if X is not None:
            feature_columns = list(X.columns)
            preprocessor = self.data.get_preprocessor(feature_columns=feature_columns)
        else:
            preprocessor = self.data.get_preprocessor()

        # Create classifier model
        model = self.create_model()

        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        return pipeline

    def _get_feature_names_from_pipeline(self, pipeline):
        """Extract feature names after preprocessing step in pipeline.

        Args:
            pipeline: Fitted Pipeline instance

        Returns:
            List of feature names
        """
        preprocessor = pipeline.named_steps['preprocessor']
        return list(preprocessor.get_feature_names_out())

    def train(self):
        """Train model on all available data.

        Returns:
            Trained Pipeline instance
        """
        print(f"\n=== Training Final Model ===")
        print(f"Parameters: {self.model_params}\n")

        X, y = self.data.get_X_y()
        self.model = self.create_pipeline(X)
        self.model.fit(X, y)

        print("Model trained successfully on full dataset")

        # Hook for subclasses to display additional stats
        self.display_post_train_stats()

        return self.model

    def display_post_train_stats(self):
        """Display classifier-specific statistics after training.

        Subclasses can override this to display model-specific metrics
        (e.g., OOB score for Random Forest).
        """
        pass

    def cross_validate(self, cv_folds=5, scoring='accuracy', find_threshold=False, threshold=None):
        """Run cross-validation to evaluate model performance.

        Uses StratifiedKFold and returns both train and validation scores
        to help detect overfitting.

        Args:
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric(s) - string or tuple/list of strings
                     Examples: 'accuracy', ('accuracy', 'f1', 'roc_auc')
            find_threshold: If True, find optimal classification threshold by maximizing F1
            threshold: If provided, evaluate using this classification threshold instead of 0.5

        Returns:
            Dict of cross-validation scores (train and validation)
        """
        print(f"\n=== Running Cross-Validation (with train scores) ===")
        print(f"Parameters: {self.model_params}, cv_folds={cv_folds}")
        if threshold is not None:
            print(f"Classification threshold: {threshold}")
        print()

        X, y = self.data.get_X_y()
        pipeline = self.create_pipeline(X)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        results = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )

        # Evaluate with custom threshold if provided
        if threshold is not None:
            self._evaluate_with_threshold(X, y, cv, results['estimator'], threshold)

        # Find optimal threshold if requested
        if find_threshold:
            self._find_optimal_threshold(X, y, cv, results['estimator'])

        # Determine which metrics to display
        if isinstance(scoring, str):
            metrics_to_display = [scoring]
        else:
            metrics_to_display = list(scoring)

        # Get additional per-fold metrics from subclass (e.g., OOB for RF)
        extra_metrics = self.get_per_fold_metrics(results)

        # Print results for each metric
        for metric in metrics_to_display:
            # sklearn uses 'train_score'/'test_score' for single metric, 'train_{metric}' for multiple
            if len(metrics_to_display) == 1:
                train_key = 'train_score'
                test_key = 'test_score'
            else:
                train_key = f'train_{metric}'
                test_key = f'test_{metric}'

            train_scores = results[train_key]
            val_scores = results[test_key]
            gap_scores = train_scores - val_scores

            print(f"{metric}:")
            print(f"  train mean={train_scores.mean():.4f} ± {train_scores.std():.4f}")
            print(f"  valid mean={val_scores.mean():.4f} ± {val_scores.std():.4f}")
            print(f"  gap  mean={gap_scores.mean():.4f} (positive = overfit)")

        # Print extra metric means if available
        for metric_name, metric_values in extra_metrics.items():
            print(f"{metric_name}:")
            print(f"  mean={np.nanmean(metric_values):.4f} ± {np.nanstd(metric_values):.4f}")

        # Print fold-by-fold scores for first metric
        first_metric = metrics_to_display[0]
        if len(metrics_to_display) == 1:
            train_scores = results['train_score']
            val_scores = results['test_score']
        else:
            train_scores = results[f'train_{first_metric}']
            val_scores = results[f'test_{first_metric}']
        gap_scores = train_scores - val_scores

        print()
        print(f"Fold-by-fold scores ({first_metric}):")
        for i, (train, val, gap) in enumerate(zip(train_scores, val_scores, gap_scores), 1):
            fold_str = f"  Fold {i}: train={train:.4f}, valid={val:.4f}"
            # Add extra metrics to fold display
            for metric_name, metric_values in extra_metrics.items():
                fold_str += f", {metric_name}={metric_values[i-1]:.4f}"
            fold_str += f", gap={gap:+.4f}"
            print(fold_str)

        return results

    def hyperparameter_search(self, param_grid=None, search_type='grid',
                             cv_folds=5, n_iter=50, scoring='accuracy',
                             n_jobs=-1, verbose=2):
        """Perform hyperparameter search using GridSearchCV or RandomizedSearchCV.

        Args:
            param_grid: Dict of parameter names to lists/distributions of values.
                       If None, uses get_param_grid() from the classifier class.
            search_type: 'grid' for exhaustive GridSearchCV or 'random' for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search (ignored for grid search)
            scoring: Scoring metric - e.g., 'accuracy', 'f1', 'roc_auc'
            n_jobs: Number of parallel jobs (-1 uses all cores)
            verbose: Verbosity level (0=silent, 1=minimal, 2=detailed, 3=very detailed)

        Returns:
            GridSearchCV or RandomizedSearchCV fitted object with results
        """
        print(f"\n=== Hyperparameter Search ===")
        print(f"Search type: {search_type}")
        print(f"CV folds: {cv_folds}")
        print(f"Scoring: {scoring}")

        X, y = self.data.get_X_y()
        pipeline = self.create_pipeline(X)

        # Get parameter grid
        if param_grid is None:
            param_grid = self.get_param_grid()
            if not param_grid:
                raise ValueError(
                    f"{type(self).__name__} does not define a default param_grid. "
                    "Please provide param_grid argument or implement get_param_grid() class method."
                )

        print(f"\nSearching over {len(param_grid)} parameters:")
        total_combinations = 1
        for param_name, param_values in param_grid.items():
            n_values = len(param_values) if isinstance(param_values, (list, tuple)) else 'continuous'
            print(f"  {param_name}: {param_values}")
            if isinstance(n_values, int):
                total_combinations *= n_values

        if search_type == 'grid':
            print(f"\nTotal combinations to test: {total_combinations}")
            print(f"Total fits: {total_combinations * cv_folds}")

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True,
                refit=True
            )
        elif search_type == 'random':
            print(f"\nRandom search iterations: {n_iter}")
            print(f"Total fits: {n_iter * cv_folds}")

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True,
                refit=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"search_type must be 'grid' or 'random', got '{search_type}'")

        # Perform search
        print(f"\nStarting search...\n")
        search.fit(X, y)

        # Display results
        print("\n" + "="*80)
        print("=== Search Results ===")
        print("="*80)

        print(f"\nBest {scoring} score: {search.best_score_:.4f}")
        print(f"\nBest parameters:")
        for param_name, param_value in sorted(search.best_params_.items()):
            # Remove 'classifier__' prefix for cleaner display
            display_name = param_name.replace('classifier__', '')
            print(f"  {display_name}: {param_value}")

        # Show top 10 parameter combinations
        results_df = pd.DataFrame(search.cv_results_)

        # Sort by mean test score
        results_df = results_df.sort_values('mean_test_score', ascending=False)

        print(f"\n=== Top 10 Parameter Combinations ===\n")
        print(f"{'Rank':<6} {'Mean Score':<12} {'Std Score':<12} {'Parameters'}")
        print("-" * 80)

        for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
            params_str = ', '.join([
                f"{k.replace('classifier__', '')}={v}"
                for k, v in row['params'].items()
            ])
            print(f"{idx:<6} {row['mean_test_score']:<12.4f} {row['std_test_score']:<12.4f} {params_str}")

        # Show train vs validation scores for best model
        best_idx = search.best_index_
        best_train_score = search.cv_results_['mean_train_score'][best_idx]
        best_test_score = search.best_score_
        gap = best_train_score - best_test_score

        print(f"\n=== Best Model Performance ===")
        print(f"Mean train score: {best_train_score:.4f}")
        print(f"Mean test score:  {best_test_score:.4f}")
        print(f"Train-test gap:   {gap:+.4f} (positive = overfit)")

        # Save detailed results
        output_file = f'hyperparameter_search_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

        return search

    def _evaluate_with_threshold(self, X, y, cv, estimators, threshold):
        """Evaluate model using a custom classification threshold.

        Args:
            X: Feature matrix
            y: Target vector
            cv: Cross-validation splitter
            estimators: List of fitted estimators from each fold
            threshold: Classification threshold to use
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        print()
        print(f"=== Scores with Threshold = {threshold:.3f} ===")
        print()

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

            # Get predictions with custom threshold
            estimator = estimators[fold_idx - 1]
            probas = estimator.predict_proba(X_valid)[:, 1]
            preds = (probas >= threshold).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_valid, preds)
            prec = precision_score(y_valid, preds, zero_division=0)
            rec = recall_score(y_valid, preds, zero_division=0)
            f1 = f1_score(y_valid, preds, zero_division=0)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            print(f"Fold {fold_idx}: accuracy={acc:.4f}, precision={prec:.4f}, "
                  f"recall={rec:.4f}, F1={f1:.4f}")

        # Print summary
        print()
        print(f"Mean scores (threshold={threshold:.3f}):")
        print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"  F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    def _find_optimal_threshold(self, X, y, cv, estimators):
        """Find optimal classification threshold by maximizing F1 score.

        Args:
            X: Feature matrix
            y: Target vector
            cv: Cross-validation splitter
            estimators: List of fitted estimators from each fold
        """
        print()
        print("=== Finding Optimal Classification Threshold ===")
        print()

        best_thresholds = []
        fold_metrics = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

            # Get predicted probabilities
            estimator = estimators[fold_idx - 1]
            probs = estimator.predict_proba(X_valid)[:, 1]

            # Find best threshold using precision-recall curve
            prec, rec, thresh = precision_recall_curve(y_valid, probs)
            f1s = 2 * prec * rec / (prec + rec)
            best_idx = np.nanargmax(f1s)
            best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.5

            best_thresholds.append(best_thresh)
            fold_metrics.append({
                'precision': prec[best_idx],
                'recall': rec[best_idx],
                'f1': f1s[best_idx]
            })

            print(f"Fold {fold_idx}: threshold={best_thresh:.3f}, "
                  f"precision={prec[best_idx]:.3f}, recall={rec[best_idx]:.3f}, "
                  f"F1={f1s[best_idx]:.3f}")

        # Print summary
        avg_thresh = np.mean(best_thresholds)
        avg_prec = np.mean([m['precision'] for m in fold_metrics])
        avg_rec = np.mean([m['recall'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])

        print()
        print(f"Average optimal threshold: {avg_thresh:.3f} ± {np.std(best_thresholds):.3f}")
        print(f"Average precision: {avg_prec:.3f}")
        print(f"Average recall:    {avg_rec:.3f}")
        print(f"Average F1:        {avg_f1:.3f}")

    def get_per_fold_metrics(self, cv_results):
        """Get additional per-fold metrics from cross-validation results.

        Subclasses can override this to extract model-specific metrics
        from the fitted estimators (e.g., OOB score for Random Forest).

        Args:
            cv_results: Results dict from cross_validate (includes 'estimator' key)

        Returns:
            Dict of metric_name -> array of per-fold values
            Example: {'oob': np.array([0.82, 0.83, 0.81, 0.84, 0.82])}
        """
        return {}

    def calculate_feature_importance(self, n_repeats=10):
        """Calculate feature importance using permutation importance.

        Args:
            n_repeats: Number of times to permute each feature

        Returns:
            Series of feature importances sorted by value
        """
        print(f"\n=== Calculating Feature Importance ===")
        print(f"Parameters: {self.model_params}, n_repeats={n_repeats}\n")

        X, y = self.data.get_X_y()

        # Split data for holdout validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Train pipeline
        pipeline = self.create_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        # Calculate permutation importance
        # NOTE: Permutation importance operates on the INPUT features (raw),
        # not the transformed features (after one-hot encoding)
        result = permutation_importance(
            pipeline, X_val, y_val,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='accuracy'
        )

        # Use raw feature names (before preprocessing)
        feature_names = X_val.columns

        # Create DataFrame with results
        importances = pd.Series(result.importances_mean, index=feature_names)
        importances_std = pd.Series(result.importances_std, index=feature_names)

        # Sort by importance
        importances = importances.sort_values(ascending=False)

        # Print results
        print("Feature Importance (sorted by mean importance):\n")
        for feature in importances.index:
            mean_imp = importances[feature]
            std_imp = importances_std[feature]
            print(f"{feature:30s}: {mean_imp:.4f} (+/- {std_imp:.4f})")

        return importances

    def calculate_drop_column_importance(self, cv_folds=5, num_repeats=1):
        """Calculate feature importance by dropping each column.

        Measures performance drop when each feature is removed.

        Args:
            cv_folds: Number of cross-validation folds
            num_repeats: Number of times to repeat with different random seeds

        Returns:
            Series of feature importances sorted by value
        """
        print(f"\n=== Calculating Drop-Column Importance ===")
        print(f"Parameters: {self.model_params}, cv_folds={cv_folds}, num_repeats={num_repeats}")

        # Determine number of parallel jobs
        n_cores = multiprocessing.cpu_count()
        n_jobs = max(1, n_cores // 2)
        print(f"Using {n_jobs} parallel jobs (out of {n_cores} cores)\n")

        X, y = self.data.get_X_y()

        # Generate seeds for each repeat
        # If random_state is None, use repeat index; otherwise add to random_state
        if self.random_state is None:
            seeds = list(range(num_repeats))
        else:
            seeds = [self.random_state + repeat for repeat in range(num_repeats)]

        # Run repeats in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_drop_repeat)(
                repeat, seeds[repeat], X, y, cv_folds, num_repeats
            )
            for repeat in range(num_repeats)
        )

        # Collect results
        all_importances = {col: [] for col in X.columns}
        all_base_scores = []

        for repeat_importances, base_score in results:
            all_base_scores.append(base_score)
            for col, importance in repeat_importances.items():
                all_importances[col].append(importance)

        # Calculate mean and std
        mean_importances = {col: np.mean(vals) for col, vals in all_importances.items()}
        std_importances = {col: np.std(vals) for col, vals in all_importances.items()}

        importance_series = pd.Series(mean_importances).sort_values(ascending=False)

        # Print results
        print(f"\n=== Drop-Column Importance (sorted) ===")
        mean_base = np.mean(all_base_scores)
        std_base = np.std(all_base_scores)
        if num_repeats > 1:
            print(f"Baseline with all features: {mean_base:.4f} (+/- {std_base:.4f})\n")
        else:
            print(f"Baseline with all features: {mean_base:.4f}\n")

        for feature in importance_series.index:
            imp = mean_importances[feature]
            if num_repeats > 1:
                std = std_importances[feature]
                print(f"{feature:40s}: {imp:+.4f} (+/- {std:.4f})")
            else:
                print(f"{feature:40s}: {imp:+.4f}")

        return importance_series

    def _run_single_drop_repeat(self, repeat, seed, X, y, cv_folds, num_repeats):
        """Helper for parallel drop-column importance calculation.

        Args:
            repeat: Repeat index
            seed: Random seed for this repeat
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            num_repeats: Total number of repeats (for printing)

        Returns:
            Tuple of (importances dict, baseline score)
        """
        if num_repeats > 1:
            print(f"\n--- Repeat {repeat + 1}/{num_repeats} (seed={seed}) ---")

        # Create CV splitter
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        # Get baseline score
        print(f"Computing baseline score with all {len(X.columns)} features...")
        # Temporarily override random_state for this repeat
        original_random_state = self.random_state
        self.random_state = seed
        pipeline = self.create_pipeline(X)
        self.random_state = original_random_state
        base_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        base_score = base_scores.mean()

        print(f"Baseline accuracy: {base_score:.4f} (+/- {base_scores.std():.4f})\n")
        print(f"Computing drop-column importance for each feature...")

        # Calculate importance by dropping each column
        repeat_importances = {}
        for i, col in enumerate(X.columns, 1):
            reduced_X = X.drop(columns=[col])
            # Temporarily override random_state for this repeat
            self.random_state = seed
            pipeline = self.create_pipeline(reduced_X)
            self.random_state = original_random_state
            cv_scores = cross_val_score(pipeline, reduced_X, y, cv=cv, scoring='accuracy')
            score = cv_scores.mean()
            importance = base_score - score

            repeat_importances[col] = importance
            print(f"  [{i}/{len(X.columns)}] {col}: {importance:+.4f} (score without: {score:.4f})")

        return repeat_importances, base_score

    def suggest_features(self, correlation_threshold=0.9, variance_threshold=0.01,
                        missing_threshold=0.5, univariate_threshold=0.05):
        """Analyze features and suggest improvements.

        Provides recommendations for:
        - Highly correlated features that could be dropped
        - Features with low univariate scores (weak predictive power)
        - Low variance features that might not be informative
        - Features with high missing rates
        - Other feature engineering suggestions

        Args:
            correlation_threshold: Threshold for identifying highly correlated features (default: 0.9)
            variance_threshold: Threshold for identifying low variance features (default: 0.01)
            missing_threshold: Threshold for flagging high missing rate features (default: 0.5)
            univariate_threshold: P-value threshold for univariate feature selection (default: 0.05)

        Returns:
            Dict with feature suggestions organized by category
        """
        print(f"\n=== Feature Analysis and Suggestions ===")
        print(f"Correlation threshold: {correlation_threshold}")
        print(f"Variance threshold: {variance_threshold}")
        print(f"Missing threshold: {missing_threshold}")
        print(f"Univariate p-value threshold: {univariate_threshold}\n")

        X, y = self.data.get_X_y()
        suggestions = {
            'highly_correlated': [],
            'weak_univariate': [],
            'low_variance': [],
            'high_missing': [],
            'constant_features': [],
            'recommendations': []
        }

        # 1. Analyze feature correlations
        print("1. Analyzing feature correlations...")
        # Only compute correlation for numeric features
        numeric_X = X.select_dtypes(include=[np.number])
        if len(numeric_X.columns) > 1:
            corr = numeric_X.corr().abs()
            # Get upper triangle of correlation matrix
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

            # Find pairs of highly correlated features
            corr_pairs = []
            for column in upper.columns:
                for idx in upper.index:
                    if upper.loc[idx, column] >= correlation_threshold:
                        corr_pairs.append({
                            'feature1': idx,
                            'feature2': column,
                            'correlation': upper.loc[idx, column]
                        })

            if corr_pairs:
                print(f"  Found {len(corr_pairs)} pairs of highly correlated features (>{correlation_threshold}):")
                for pair in corr_pairs:
                    print(f"    - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
                    suggestions['highly_correlated'].append(pair)

                # Determine which features to suggest dropping
                to_drop = set()
                for pair in corr_pairs:
                    # Prefer to keep feature with fewer missing values
                    missing1 = X[pair['feature1']].isnull().sum()
                    missing2 = X[pair['feature2']].isnull().sum()
                    if missing1 > missing2:
                        to_drop.add(pair['feature1'])
                    else:
                        to_drop.add(pair['feature2'])

                if to_drop:
                    suggestions['recommendations'].append(
                        f"Consider dropping {len(to_drop)} highly correlated features: {sorted(to_drop)}"
                    )
                print()
            else:
                print(f"  No highly correlated features found (threshold: {correlation_threshold})\n")

        # 2. Univariate feature screening
        print("2. Univariate feature screening (statistical tests)...")

        # Handle missing values for univariate screening
        X_imputed = X.copy()
        for col in X_imputed.columns:
            if X_imputed[col].dtype in [np.number, 'float64', 'int64']:
                X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
            else:
                mode_val = X_imputed[col].mode()[0] if not X_imputed[col].mode().empty else 'missing'
                X_imputed[col] = X_imputed[col].fillna(mode_val)

        # Separate numeric and categorical features
        numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = X_imputed.select_dtypes(exclude=[np.number]).columns

        univariate_scores = {}

        # Score numeric features using f_classif (ANOVA F-statistic)
        if len(numeric_cols) > 0:
            selector_numeric = SelectKBest(score_func=f_classif, k='all')
            selector_numeric.fit(X_imputed[numeric_cols], y)
            scores_numeric = pd.Series(
                selector_numeric.scores_,
                index=numeric_cols
            )
            pvalues_numeric = pd.Series(
                selector_numeric.pvalues_,
                index=numeric_cols
            )

            for col in numeric_cols:
                univariate_scores[col] = {
                    'score': scores_numeric[col],
                    'pvalue': pvalues_numeric[col],
                    'type': 'numeric'
                }

        # For categorical features (if we had any non-numeric), we would use chi2
        # But since pd.get_dummies is used in prepare_features, all features are numeric

        # Sort features by score
        sorted_features = sorted(
            univariate_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Identify weak features
        weak_features = []
        print(f"  Top 10 features by univariate score:")
        for i, (feat, info) in enumerate(sorted_features[:10]):
            print(f"    {i+1:2d}. {feat}: F={info['score']:8.2f}, p-value={info['pvalue']:.4f}")

        print(f"\n  Features with p-value > {univariate_threshold} (weak predictive power):")
        for feat, info in sorted_features:
            if info['pvalue'] > univariate_threshold:
                weak_features.append({
                    'feature': feat,
                    'score': info['score'],
                    'pvalue': info['pvalue']
                })
                print(f"    - {feat}: F={info['score']:8.2f}, p-value={info['pvalue']:.4f}")

        if weak_features:
            suggestions['weak_univariate'] = weak_features
            suggestions['recommendations'].append(
                f"Consider removing {len(weak_features)} features with p-value > {univariate_threshold} (weak univariate predictive power)"
            )

        print()

        # 3. Analyze feature variance
        print("3. Analyzing feature variance...")
        low_var_features = []
        constant_features = []

        for col in X.columns:
            if X[col].dtype in [np.number, 'float64', 'int64']:
                # Check for constant features (single unique value)
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) <= 1:
                    constant_features.append(col)
                    print(f"  - {col}: CONSTANT (only {len(unique_vals)} unique value)")
                # Check for low variance
                elif X[col].var() < variance_threshold:
                    low_var_features.append({
                        'feature': col,
                        'variance': X[col].var(),
                        'unique_values': len(unique_vals)
                    })
                    print(f"  - {col}: Low variance ({X[col].var():.6f}), {len(unique_vals)} unique values")

        if constant_features:
            suggestions['constant_features'] = constant_features
            suggestions['recommendations'].append(
                f"Remove {len(constant_features)} constant features: {constant_features}"
            )

        if low_var_features:
            suggestions['low_variance'] = low_var_features
            suggestions['recommendations'].append(
                f"Review {len(low_var_features)} low-variance features for potential removal"
            )

        if not constant_features and not low_var_features:
            print(f"  No low variance features found (threshold: {variance_threshold})")
        print()

        # 4. Analyze missing data patterns
        print("4. Analyzing missing data...")
        missing_info = []
        for col in X.columns:
            missing_rate = X[col].isnull().mean()
            if missing_rate > 0:
                missing_info.append({
                    'feature': col,
                    'missing_rate': missing_rate,
                    'missing_count': X[col].isnull().sum()
                })

        # Sort by missing rate
        missing_info.sort(key=lambda x: x['missing_rate'], reverse=True)

        if missing_info:
            print(f"  Features with missing values:")
            high_missing = []
            for info in missing_info:
                status = "HIGH" if info['missing_rate'] >= missing_threshold else ""
                print(f"    - {info['feature']}: {info['missing_rate']:.1%} ({info['missing_count']} rows) {status}")
                if info['missing_rate'] >= missing_threshold:
                    high_missing.append(info)

            if high_missing:
                suggestions['high_missing'] = high_missing
                suggestions['recommendations'].append(
                    f"Review {len(high_missing)} features with >={missing_threshold:.0%} missing values"
                )
            print()
        else:
            print("  No missing values found in any features\n")

        # 5. Additional feature engineering suggestions
        print("5. Additional feature engineering suggestions:")

        # Check for potential interaction features
        if len(numeric_X.columns) > 1:
            # Find features with moderate correlation that might benefit from interaction
            moderate_corr_pairs = []
            for column in upper.columns:
                for idx in upper.index:
                    corr_val = upper.loc[idx, column]
                    if 0.3 <= corr_val < correlation_threshold:
                        moderate_corr_pairs.append((idx, column, corr_val))

            if len(moderate_corr_pairs) > 0:
                top_pairs = sorted(moderate_corr_pairs, key=lambda x: x[2], reverse=True)[:3]
                print(f"  - Consider creating interaction features for moderately correlated pairs:")
                for f1, f2, corr in top_pairs:
                    print(f"    * {f1} × {f2} (correlation: {corr:.3f})")
                suggestions['recommendations'].append(
                    "Consider creating interaction features for moderately correlated feature pairs"
                )

        # Check for potential polynomial features
        continuous_features = []
        for col in numeric_X.columns:
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) > 10:  # Likely continuous
                continuous_features.append(col)

        if continuous_features:
            print(f"  - Consider polynomial features for continuous variables:")
            for feat in continuous_features[:5]:  # Show max 5
                print(f"    * {feat}")
            if len(continuous_features) > 5:
                print(f"    * ... and {len(continuous_features) - 5} more")
            suggestions['recommendations'].append(
                f"Consider polynomial features for {len(continuous_features)} continuous variables"
            )

        print()

        # 6. Summary and actionable recommendations
        print("=== Summary of Recommendations ===\n")
        if suggestions['recommendations']:
            for i, rec in enumerate(suggestions['recommendations'], 1):
                print(f"{i}. {rec}")
        else:
            print("No specific feature improvements recommended. Features look good!")

        print()

        # 7. Optional: Create cleaned feature set
        if suggestions['highly_correlated'] or suggestions['constant_features'] or suggestions['weak_univariate']:
            print("=== Suggested Feature Set ===")
            features_to_drop = set()

            # Add constant features
            features_to_drop.update(suggestions['constant_features'])

            # Add one feature from each highly correlated pair
            for pair in suggestions['highly_correlated']:
                # Keep the feature that's not already marked for dropping
                if pair['feature1'] not in features_to_drop:
                    features_to_drop.add(pair['feature2'])
                elif pair['feature2'] not in features_to_drop:
                    features_to_drop.add(pair['feature1'])

            # Optionally add weak univariate features (with high p-values)
            # We'll be conservative and only suggest dropping if p-value is very high
            very_weak_threshold = 0.2  # More conservative than univariate_threshold
            for feat_info in suggestions['weak_univariate']:
                if feat_info['pvalue'] > very_weak_threshold:
                    features_to_drop.add(feat_info['feature'])

            remaining_features = [f for f in X.columns if f not in features_to_drop]
            print(f"Original features: {len(X.columns)}")
            print(f"Suggested features: {len(remaining_features)} (drop {len(features_to_drop)})")

            # Categorize dropped features
            dropped_categories = {
                'constant': [],
                'highly_correlated': [],
                'weak_univariate': []
            }

            for feat in features_to_drop:
                if feat in suggestions['constant_features']:
                    dropped_categories['constant'].append(feat)
                elif any(feat in [p['feature1'], p['feature2']] for p in suggestions['highly_correlated']):
                    dropped_categories['highly_correlated'].append(feat)
                elif any(feat == w['feature'] for w in suggestions['weak_univariate'] if w['pvalue'] > very_weak_threshold):
                    dropped_categories['weak_univariate'].append(feat)

            print(f"\nFeatures to drop by reason:")
            if dropped_categories['constant']:
                print(f"  Constant: {sorted(dropped_categories['constant'])}")
            if dropped_categories['highly_correlated']:
                print(f"  Highly correlated: {sorted(dropped_categories['highly_correlated'])}")
            if dropped_categories['weak_univariate']:
                print(f"  Weak univariate (p>{very_weak_threshold:.1f}): {sorted(dropped_categories['weak_univariate'])}")

            suggestions['features_to_drop'] = sorted(features_to_drop)
            suggestions['remaining_features'] = remaining_features

            # Show how to exclude these features
            print(f"\n=== How to Exclude These Features ===")
            print(f"\nMethod 1: Using --exclude-features CLI parameter")
            features_str = ','.join(sorted(features_to_drop))
            print(f"  python predict.py cv --exclude-features \"{features_str}\"")

            print(f"\nMethod 2: Programmatically in Python")
            print(f"  from titanic_data import TitanicData")
            print(f"  data = TitanicData('./train.csv')")
            print(f"  data.excluded_features = {sorted(features_to_drop)}")
            print(f"  X, y = data.get_X_y()  # Returns features with exclusions applied")

        return suggestions

    def rfecv_analysis(self, cv_folds=5, step=1, scoring='accuracy', n_jobs=-1):
        """Perform Recursive Feature Elimination with Cross-Validation.

        Uses RFECV to determine the optimal number of features and identify
        which features are most important for model performance.

        Note: RFECV operates on preprocessed features (after one-hot encoding
        and scaling), not raw input columns.

        Args:
            cv_folds: Number of cross-validation folds (default: 5)
            step: Number (or fraction) of features to remove at each iteration (default: 1)
            scoring: Scoring metric - e.g., 'accuracy', 'f1', 'roc_auc' (default: 'accuracy')
            n_jobs: Number of parallel jobs (-1 uses all cores, default: -1)

        Returns:
            RFECV fitted object with results
        """
        print(f"\n=== Recursive Feature Elimination with Cross-Validation ===")
        print(f"Parameters: {self.model_params}")
        print(f"CV folds: {cv_folds}")
        print(f"Step size: {step}")
        print(f"Scoring: {scoring}")
        print(f"Parallel jobs: {n_jobs}\n")

        X, y = self.data.get_X_y()

        # Create pipeline and fit preprocessor to get feature names
        pipeline = self.create_pipeline(X)

        # Fit the preprocessor to get transformed feature names
        print("Preprocessing data...")
        preprocessor = pipeline.named_steps['preprocessor']
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        feature_names = list(preprocessor.get_feature_names_out())
        print(f"After preprocessing: {len(feature_names)} features\n")

        # Create CV splitter
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Create model for RFECV (just the classifier, not the full pipeline)
        model = self.create_model()

        # Create RFECV selector with the classifier
        print("Starting RFECV (this may take a while)...")
        print("Note: RFECV will evaluate feature subsets by retraining the model.\n")
        rfecv = RFECV(
            estimator=model,
            step=step,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        # Fit RFECV on preprocessed data
        rfecv.fit(X_transformed, y)

        # Display results
        print("\n" + "="*80)
        print("=== RFECV Results ===")
        print("="*80)

        print(f"\nOptimal number of features: {rfecv.n_features_}")
        print(f"Total preprocessed features: {len(feature_names)}")
        print(f"Features to remove: {len(feature_names) - rfecv.n_features_}")

        # Get selected and rejected features (using preprocessed feature names)
        selected_features = [feature_names[i] for i, selected in enumerate(rfecv.support_) if selected]
        rejected_features = [feature_names[i] for i, selected in enumerate(rfecv.support_) if not selected]

        print(f"\n=== Selected Features ({len(selected_features)}) ===")
        for feat in selected_features:
            print(f"  {feat}")

        print(f"\n=== Rejected Features ({len(rejected_features)}) ===")
        for feat in rejected_features:
            idx = feature_names.index(feat)
            rank = rfecv.ranking_[idx]
            print(f"  {feat} (rank: {rank})")

        # Display CV scores vs number of features
        print(f"\n=== Cross-Validation Scores by Number of Features ===")
        cv_scores = rfecv.cv_results_['mean_test_score']
        cv_stds = rfecv.cv_results_['std_test_score']

        # Show scores for key feature counts
        max_score = np.max(cv_scores)
        max_idx = np.argmax(cv_scores)
        optimal_n_features = max_idx + 1

        print(f"\nBest CV score: {max_score:.4f} (±{cv_stds[max_idx]:.4f}) with {optimal_n_features} features")

        # Show scores around the optimal point
        print(f"\nCV scores around optimal:")
        start_idx = max(0, max_idx - 2)
        end_idx = min(len(cv_scores), max_idx + 3)
        for i in range(start_idx, end_idx):
            n_features = i + 1
            score = cv_scores[i]
            std = cv_stds[i]
            marker = " <-- optimal" if i == max_idx else ""
            print(f"  {n_features:3d} features: {score:.4f} (±{std:.4f}){marker}")

        # Plot CV scores vs number of features
        print(f"\n=== Plotting CV Scores vs Number of Features ===")
        plt.figure(figsize=(10, 6))
        n_features_range = range(1, len(cv_scores) + 1)
        plt.plot(n_features_range, cv_scores, 'o-', linewidth=2, markersize=6)
        plt.fill_between(n_features_range,
                        cv_scores - cv_stds,
                        cv_scores + cv_stds,
                        alpha=0.2)
        plt.axvline(x=optimal_n_features, color='r', linestyle='--',
                   label=f'Optimal: {optimal_n_features} features')
        plt.xlabel('Number of Features')
        plt.ylabel(f'Cross-Validation Score ({scoring})')
        plt.title('RFECV: Feature Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = 'rfecv_scores.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        plt.close()

        # Show how to use the selected features
        print(f"\n=== How to Use Selected Features ===")
        print(f"\nNote: RFECV operates on preprocessed features (after one-hot encoding).")
        print(f"The feature names shown above are the transformed feature names.")
        print(f"\nTo use these results:")
        print(f"1. Identify which raw input features correspond to rejected preprocessed features")
        print(f"2. Use the suggest-features command for raw feature analysis")
        print(f"3. Consider the preprocessed feature rankings when doing feature engineering")

        if rejected_features:
            print(f"\nRejected preprocessed features ({len(rejected_features)}):")
            for feat in rejected_features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(rejected_features) > 10:
                print(f"  ... and {len(rejected_features) - 10} more")

        return rfecv

    def plot_partial_dependence(self, features=None, output_file='partial_dependence.png'):
        """Generate partial dependence plots.

        Args:
            features: List of features to plot, or None for auto-detection
            output_file: Path to save the plot

        Returns:
            PartialDependenceDisplay object
        """
        print(f"\n=== Generating Partial Dependence Plots ===")
        print(f"Parameters: {self.model_params}\n")

        X, y = self.data.get_X_y()

        # Train pipeline
        pipeline = self.create_pipeline(X)
        pipeline.fit(X, y)

        # Determine which features to plot
        if features:
            plot_features = features
        else:
            # Auto-detect numerical features (exclude categorical)
            categorical_cols = self.data.get_categorical_feature_columns()
            numerical_cols = self.data.get_numerical_feature_columns()

            # Only plot continuous numerical features (not binary indicators)
            plot_features = []
            for col in numerical_cols:
                if col not in categorical_cols:
                    unique_vals = X[col].dropna().unique()
                    # Include if more than 2 unique values (not binary)
                    if len(unique_vals) > 2:
                        plot_features.append(col)

            print(f"Auto-detected {len(plot_features)} numerical features")

        # Filter to valid features
        valid_features = [f for f in plot_features if f in X.columns]

        if not valid_features:
            print(f"Error: None of the specified features exist in the dataset")
            print(f"Available features: {list(X.columns)}")
            return

        print(f"Plotting partial dependence for: {valid_features}\n")

        # Calculate figure size
        n_features = len(valid_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig_width = n_cols * 6
        fig_height = n_rows * 5

        # Create plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        display = PartialDependenceDisplay.from_estimator(
            pipeline, X, valid_features,
            kind='average',
            random_state=self.random_state,
            ax=ax
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Partial dependence plot saved to: {output_file}")
        plt.show()

        return display

    def predict_submission(self, test_filepath, output_file='submission.csv', threshold=0.5):
        """Train on full dataset and predict on test data for competition submission.

        Requires that the data class has a load_and_prepare_test() method.

        Args:
            test_filepath: Path to test CSV file
            output_file: Path to save submission CSV
            threshold: Classification threshold for positive class (default: 0.5)

        Returns:
            DataFrame with predictions (PassengerId, target column)
        """
        print(f"\n=== Generating Competition Submission ===")
        print(f"Parameters: {self.model_params}")
        print(f"Test data: {test_filepath}")
        print(f"Output file: {output_file}")
        print(f"Classification threshold: {threshold}\n")

        # Check if data class supports test data loading
        if not hasattr(self.data, 'load_and_prepare_test'):
            raise NotImplementedError(
                f"{type(self.data).__name__} does not implement load_and_prepare_test()"
            )

        # Prepare data for submission (compute any combined train+test statistics)
        self.data.prepare_for_submission(test_filepath)

        # Train on full training dataset
        print("Training model on full training dataset...")
        X, y = self.data.get_X_y()
        pipeline = self.create_pipeline(X)
        pipeline.fit(X, y)
        print(f"Model trained on {len(X)} samples\n")

        # Load and prepare test data
        print("Loading and preparing test data...")
        X_test, passenger_ids = self.data.load_and_prepare_test(test_filepath)
        print(f"Test data prepared: {len(X_test)} samples\n")

        # Make predictions
        print("Making predictions...")
        if threshold != 0.5:
            # Use custom threshold
            probas = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probas >= threshold).astype(int)
        else:
            # Use default predict method
            predictions = pipeline.predict(X_test)
        print(f"Predictions complete\n")

        # Create submission DataFrame
        target_col = self.data.get_target_column()
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            target_col: predictions
        })

        # Save to CSV
        submission.to_csv(output_file, index=False)
        print(f"Submission saved to: {output_file}")
        print(f"\nSubmission preview:")
        print(submission.head(10))
        print(f"...")
        print(f"\nPrediction distribution:")
        print(submission[target_col].value_counts().sort_index())

        return submission
