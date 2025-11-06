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
    parent_parser.add_argument('--n-estimators', type=int, default=100,
                              help='Number of trees in random forest (default: 100)')
    parent_parser.add_argument('--max-depth', type=int, default=20,
                              help='Maximum depth of trees (default: 20)')

    # Main parser
    parser = argparse.ArgumentParser(description=description)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Cross-validation command
    cv_parser = subparsers.add_parser('cv', parents=[parent_parser],
                                      help='Run cross-validation')
    cv_parser.add_argument('--cv-folds', type=int, default=5,
                          help='Number of cross-validation folds (default: 5)')

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

    args = parser.parse_args()

    # Initialize data and classifier
    data = data_class(args.data)
    classifier = classifier_class(
        data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    # Execute the appropriate command
    if args.command == 'cv':
        classifier.cross_validate(cv_folds=args.cv_folds)
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
    else:
        parser.print_help()
