"""Generic CLI interface for binary classification pipelines."""

import argparse


def create_cli(data_class, classifier_class, default_data_path='./train.csv',
               description='Binary Classification ML Pipeline'):
    """Create and run a generic CLI for binary classification.

    Args:
        data_class: Class to instantiate for data handling (must inherit from BinaryClassificationData)
        classifier_class: Class to instantiate for classification (must inherit from BinaryClassifier)
        default_data_path: Default path to data file
        description: Description for the CLI

    Returns:
        None (runs the CLI and executes the requested command)
    """
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data', type=str, default=default_data_path,
                              help=f'Path to training data (default: {default_data_path})')
    parent_parser.add_argument('--random-state', type=int, default=None,
                              help='Random seed for reproducibility (default: None, uses random seed)')

    # Dynamically add classifier-specific arguments
    classifier_args = classifier_class.get_cli_arguments()
    for arg_def in classifier_args:
        parent_parser.add_argument(
            arg_def['name'],
            type=arg_def['type'],
            default=arg_def['default'],
            help=arg_def['help']
        )

    # Main parser
    parser = argparse.ArgumentParser(description=description)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Cross-validation command
    cv_parser = subparsers.add_parser('cv', parents=[parent_parser],
                                      help='Run cross-validation')
    cv_parser.add_argument('--cv-folds', type=int, default=5,
                          help='Number of cross-validation folds (default: 5)')
    cv_parser.add_argument('--scoring', type=str, default='accuracy',
                          help='Scoring metric(s) - comma-separated: accuracy,f1,roc_auc,precision,recall (default: accuracy)')
    cv_parser.add_argument('--find-threshold', action='store_true',
                          help='Find optimal classification threshold by maximizing F1 score')
    cv_parser.add_argument('--threshold', type=float, default=None,
                          help='Evaluate CV with specific classification threshold (default: None, uses 0.5)')

    # Feature importance command
    importance_parser = subparsers.add_parser('importance', parents=[parent_parser],
                                             help='Calculate feature importance')
    importance_parser.add_argument('--n-repeats', type=int, default=10,
                                  help='Number of times to permute a feature (default: 10)')

    # Drop-column importance command
    drop_importance_parser = subparsers.add_parser('drop-importance', parents=[parent_parser],
                                                   help='Calculate drop-column importance')
    drop_importance_parser.add_argument('--cv-folds', type=int, default=5,
                                       help='Number of cross-validation folds (default: 5)')
    drop_importance_parser.add_argument('--num-repeats', type=int, default=1,
                                       help='Number of times to repeat calculation (default: 1)')

    # Train command
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Train final model on all data')

    # Partial dependence plot command
    pdp_parser = subparsers.add_parser('pdp', parents=[parent_parser],
                                       help='Generate partial dependence plots')
    pdp_parser.add_argument('--features', nargs='+', type=str,
                           help='Features to plot (default: all numerical features)')

    # Correlation matrix command
    corr_parser = subparsers.add_parser('correlation', parents=[parent_parser],
                                        help='Generate feature correlation matrix heatmap')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', parents=[parent_parser],
                                          help='Inspect data quality and show NaN statistics')
    inspect_parser.add_argument('--test-data', type=str, default='./test.csv',
                               help='Path to test data for categorical value inspection (default: ./test.csv)')

    # Suggest features command
    suggest_parser = subparsers.add_parser('suggest-features', parents=[parent_parser],
                                          help='Analyze features and suggest improvements')
    suggest_parser.add_argument('--correlation-threshold', type=float, default=0.9,
                               help='Threshold for identifying highly correlated features (default: 0.9)')
    suggest_parser.add_argument('--variance-threshold', type=float, default=0.01,
                               help='Threshold for identifying low variance features (default: 0.01)')
    suggest_parser.add_argument('--missing-threshold', type=float, default=0.5,
                               help='Threshold for flagging high missing rate features (default: 0.5)')
    suggest_parser.add_argument('--univariate-threshold', type=float, default=0.05,
                               help='P-value threshold for univariate feature selection (default: 0.05)')

    # Submit command (for Kaggle competitions)
    submit_parser = subparsers.add_parser('submit', parents=[parent_parser],
                                         help='Train model and generate predictions for test data')
    submit_parser.add_argument('--test-data', type=str, default='./test.csv',
                              help='Path to test data (default: ./test.csv)')
    submit_parser.add_argument('--output', type=str, default='submission.csv',
                              help='Path to save submission file (default: submission.csv)')
    submit_parser.add_argument('--threshold', type=float, default=0.45,
                              help='Classification threshold for positive class (default: 0.45)')

    args = parser.parse_args()

    # Check if no command was provided
    if args.command is None:
        parser.print_help()
        return

    # Initialize data class
    data = data_class(args.data)

    # Extract classifier-specific parameters from args
    classifier_kwargs = {}
    for arg_def in classifier_args:
        # Convert '--arg-name' to 'arg_name' for attribute access
        arg_name = arg_def['name'].lstrip('-').replace('-', '_')
        if hasattr(args, arg_name):
            classifier_kwargs[arg_name] = getattr(args, arg_name)

    # Initialize classifier with data and dynamic parameters
    # The classifier will call data.load_data() and data.get_preprocessor() as needed
    classifier = classifier_class(data, random_state=args.random_state, **classifier_kwargs)

    # Execute the appropriate command
    if args.command == 'cv':
        # Parse comma-separated scoring metrics
        scoring_list = [s.strip() for s in args.scoring.split(',')]
        scoring = scoring_list[0] if len(scoring_list) == 1 else tuple(scoring_list)
        classifier.cross_validate(cv_folds=args.cv_folds, scoring=scoring,
                                 find_threshold=args.find_threshold, threshold=args.threshold)
    elif args.command == 'importance':
        classifier.calculate_feature_importance(n_repeats=args.n_repeats)
    elif args.command == 'drop-importance':
        classifier.calculate_drop_column_importance(
            cv_folds=args.cv_folds,
            num_repeats=args.num_repeats
        )
    elif args.command == 'train':
        classifier.train()
    elif args.command == 'pdp':
        classifier.plot_partial_dependence(features=args.features)
    elif args.command == 'correlation':
        data.plot_correlation_matrix()
    elif args.command == 'inspect':
        # Check if inspect_data accepts test_filepath parameter (Titanic-specific)
        import inspect
        sig = inspect.signature(data.inspect_data)
        if 'test_filepath' in sig.parameters:
            data.inspect_data(test_filepath=args.test_data)
        else:
            data.inspect_data()
    elif args.command == 'suggest-features':
        classifier.suggest_features(
            correlation_threshold=args.correlation_threshold,
            variance_threshold=args.variance_threshold,
            missing_threshold=args.missing_threshold,
            univariate_threshold=args.univariate_threshold
        )
    elif args.command == 'submit':
        classifier.predict_submission(
            test_filepath=args.test_data,
            output_file=args.output,
            threshold=args.threshold
        )
