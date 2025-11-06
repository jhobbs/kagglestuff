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

    args = parser.parse_args()

    # Check if classifier can handle NaN values
    # If not, data class will impute missing values
    impute_nans = not classifier_class.handles_nan()

    # Initialize data with imputation flag
    data = data_class(args.data, impute_nans=impute_nans)

    # Extract classifier-specific parameters from args
    classifier_kwargs = {}
    for arg_def in classifier_args:
        # Convert '--arg-name' to 'arg_name' for attribute access
        arg_name = arg_def['name'].lstrip('-').replace('-', '_')
        if hasattr(args, arg_name):
            classifier_kwargs[arg_name] = getattr(args, arg_name)

    # Initialize classifier with dynamic parameters
    classifier = classifier_class(data, random_state=42, **classifier_kwargs)

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
    elif args.command == 'inspect':
        data.inspect_data()
    else:
        parser.print_help()
