import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def load_data(filepath):
    """Load the training data from CSV file."""
    data = pd.read_csv(filepath)
    data.head()
    return data


def tokenize_ticket(ticket):
    """
    Tokenize a ticket string by splitting on punctuation and whitespace.
    Returns a set of lowercase tokens.
    """
    if pd.isna(ticket):
        return set()

    # Split on punctuation and whitespace, keep only alphabetic tokens
    tokens = re.findall(r'[a-zA-Z]+', str(ticket))

    # Lowercase and return as set
    return set(token.lower() for token in tokens)


def parse_ticket(t):
    """Parse ticket information into number only."""
    if pd.isna(t):
        return None
    parts = t.split()
    if len(parts) == 2:
        number = parts[1]
    elif len(parts) == 3:
        number = parts[2]
    else:
        number = parts[0]

    # Convert to numeric, return None if not a valid number
    try:
        return float(number)
    except:
        return None


def engineer_features(data):
    """Create new features from the raw data."""
    # Convert Sex to boolean Male feature
    data["Male"] = (data["Sex"] == "male").astype(int)

    data["Name_length"] = data["Name"].str.len()

    # Extract last name (everything before the comma)
    data["Last_name"] = data["Name"].str.split(',').str[0]

    # Count passengers with the same last name
    lastname_counts = data["Last_name"].value_counts()
    data["Same_lastname_count"] = data["Last_name"].map(lastname_counts)

    data["Cabin"] = data["Cabin"].fillna("Unknown")
    data["Cabin_count"] = data['Cabin'].str.split().str.len()
    data["Deck"] = data['Cabin'].str[0]

    # Parse ticket number
    data["Ticket_number"] = data["Ticket"].apply(parse_ticket)

    # Calculate ticket number length (log10 gives approximate number of digits)
    data["Ticket_number_length"] = data["Ticket_number"].apply(
        lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan
    )

    # Tokenize ticket codes and create boolean features
    # First, get all unique tokens across all tickets
    all_tokens = set()
    for ticket in data["Ticket"]:
        all_tokens.update(tokenize_ticket(ticket))

    # Create boolean feature for each token
    for token in sorted(all_tokens):
        feature_name = f"ticket_token_{token}"
        data[feature_name] = data["Ticket"].apply(
            lambda x: 1 if token in tokenize_ticket(x) else 0
        )

    return data


def get_feature_list():
    """Return the list of base features to use in the model."""
    return ["Pclass", "Male", "SibSp", "Parch", "Age", "Name_length",
            "Same_lastname_count", "Fare", "Embarked", "Cabin_count", "Deck",
            "Ticket_number", "Ticket_number_length"]


def prepare_features(data, feature_columns):
    """Prepare feature matrix and target variable."""
    y = data["Survived"]

    # Add dynamically created ticket token features
    ticket_token_cols = [col for col in data.columns if col.startswith('ticket_token_')]
    all_features = feature_columns + ticket_token_cols

    X = pd.get_dummies(data[all_features])

    # Convert all numeric columns to float to avoid sklearn warnings
    numeric_cols = X.select_dtypes(include=['int64', 'int32']).columns
    X[numeric_cols] = X[numeric_cols].astype(float)

    return X, y


def run_cross_validation(args):
    """Run cross-validation on the model."""
    print(f"\n=== Running Cross-Validation ===")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, cv_folds={args.cv_folds}\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    # Create model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=args.cv_folds, scoring='accuracy')

    # Print results
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    print(f"Mean Score: {scores.mean() * 100:.2f}%")

    return scores


def calculate_feature_importance(args):
    """Calculate and display feature importance using permutation importance."""
    print(f"\n=== Calculating Feature Importance ===")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, n_repeats={args.n_repeats}\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    # Split data for holdout validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Calculate permutation importance
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=args.n_repeats,
        random_state=42,
        scoring='accuracy'
    )

    # Create DataFrame with results
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    importances_std = pd.Series(result.importances_std, index=X_val.columns)

    # Sort by importance
    importances = importances.sort_values(ascending=False)

    # Print results
    print("Feature Importance (sorted by mean importance):\n")
    for feature in importances.index:
        mean_imp = importances[feature]
        std_imp = importances_std[feature]
        print(f"{feature:30s}: {mean_imp:.4f} (+/- {std_imp:.4f})")

    return importances


def calculate_drop_column_importance(args):
    """Calculate feature importance by dropping each column and measuring performance drop."""
    print(f"\n=== Calculating Drop-Column Importance ===")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, cv_folds={args.cv_folds}, num_repeats={args.num_repeats}\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    from sklearn.model_selection import StratifiedKFold

    # Store importance values across all repeats
    all_importances = {col: [] for col in X.columns}
    all_base_scores = []

    # Repeat calculation with different random seeds
    for repeat in range(args.num_repeats):
        seed = 42 + repeat
        if args.num_repeats > 1:
            print(f"\n--- Repeat {repeat + 1}/{args.num_repeats} (seed={seed}) ---")

        # Use a fixed CV splitter for consistency within this repeat
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)

        # Create model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=seed
        )

        # Get baseline score with all features
        print(f"Computing baseline score with all {len(X.columns)} features...")
        base_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        base_score = base_scores.mean()
        all_base_scores.append(base_score)

        print(f"Baseline accuracy: {base_score:.4f} (+/- {base_scores.std():.4f})\n")
        print(f"Computing drop-column importance for each feature...")

        # Calculate importance by dropping each column
        for i, col in enumerate(X.columns, 1):
            # Drop one column and evaluate
            reduced_X = X.drop(columns=[col])
            cv_scores = cross_val_score(model, reduced_X, y, cv=cv, scoring='accuracy')
            score = cv_scores.mean()
            importance = base_score - score  # positive = important (dropping it hurts)

            all_importances[col].append(importance)
            print(f"  [{i}/{len(X.columns)}] {col}: {importance:+.4f} (score without: {score:.4f})")

    # Calculate mean and std of importances
    mean_importances = {col: np.mean(vals) for col, vals in all_importances.items()}
    std_importances = {col: np.std(vals) for col, vals in all_importances.items()}

    # Convert to sorted series
    importance_series = pd.Series(mean_importances).sort_values(ascending=False)

    # Print final results
    print(f"\n=== Drop-Column Importance (sorted) ===")
    mean_base = np.mean(all_base_scores)
    std_base = np.std(all_base_scores)
    if args.num_repeats > 1:
        print(f"Baseline with all features: {mean_base:.4f} (+/- {std_base:.4f})\n")
    else:
        print(f"Baseline with all features: {mean_base:.4f}\n")

    for feature in importance_series.index:
        imp = mean_importances[feature]
        if args.num_repeats > 1:
            std = std_importances[feature]
            print(f"{feature:40s}: {imp:+.4f} (+/- {std:.4f})")
        else:
            print(f"{feature:40s}: {imp:+.4f}")

    return importance_series


def train_final_model(args):
    """Train a final model on all data."""
    print(f"\n=== Training Final Model ===")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X, y)

    print("Model trained successfully on full dataset")
    return model


def plot_partial_dependence(args):
    """Generate partial dependence plots for specified features."""
    print(f"\n=== Generating Partial Dependence Plots ===")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X, y)

    # Determine which features to plot
    if args.features:
        plot_features = args.features
    else:
        # Auto-detect numerical features (not one-hot encoded)
        # One-hot encoded features typically only have values 0 and 1
        # Numerical features have more diverse values
        plot_features = []
        for col in X.columns:
            unique_vals = X[col].dropna().unique()
            # Include if: more than 2 unique values, OR has non-binary values
            if len(unique_vals) > 2 or not set(unique_vals).issubset({0.0, 1.0}):
                plot_features.append(col)

        print(f"Auto-detected {len(plot_features)} numerical features")

    # Filter to only features that exist in X
    valid_features = [f for f in plot_features if f in X.columns]

    if not valid_features:
        print(f"Error: None of the specified features {plot_features} exist in the dataset")
        print(f"Available features: {list(X.columns)}")
        return

    print(f"Plotting partial dependence for: {valid_features}\n")

    # Calculate figure size based on number of features
    # Use 3 columns max, calculate rows needed
    n_features = len(valid_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    # Make figure bigger: 6 inches per column, 5 inches per row
    fig_width = n_cols * 6
    fig_height = n_rows * 5

    # Create the partial dependence plot with custom figure size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    display = PartialDependenceDisplay.from_estimator(
        model, X, valid_features,
        kind='average',
        random_state=42,
        ax=ax
    )

    plt.tight_layout()
    plt.savefig('partial_dependence.png', dpi=150, bbox_inches='tight')
    print(f"Partial dependence plot saved to: partial_dependence.png")
    plt.show()

    return display


def plot_correlation_matrix(args):
    """Generate a correlation heatmap for all features."""
    print(f"\n=== Generating Feature Correlation Matrix ===\n")

    # Load and prepare data
    train_data = load_data(args.data)
    train_data = engineer_features(train_data)
    features = get_feature_list()
    X, y = prepare_features(train_data, features)

    print(f"Computing correlations for {len(X.columns)} features...")

    # Calculate correlation matrix
    corr = X.corr()

    # Create figure
    fig_size = max(10, len(X.columns) * 0.3)  # Scale with number of features
    plt.figure(figsize=(fig_size, fig_size))

    # Create heatmap
    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        annot=len(X.columns) <= 20,  # Only show annotations if not too many features
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )

    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Correlation matrix saved to: correlation_matrix.png")
    plt.show()

    # Print highly correlated feature pairs
    print(f"\nHighly correlated feature pairs (|correlation| > 0.7):\n")
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr_pairs:
        for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {feat1:30s} <-> {feat2:30s}: {corr_val:+.3f}")
    else:
        print("  No highly correlated pairs found.")

    return corr


def main():
    """Main function to handle command-line arguments."""
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data', type=str, default='./train.csv',
                              help='Path to training data (default: ./train.csv)')
    parent_parser.add_argument('--n-estimators', type=int, default=100,
                              help='Number of trees in random forest (default: 100)')
    parent_parser.add_argument('--max-depth', type=int, default=20,
                              help='Maximum depth of trees (default: 20)')

    # Main parser
    parser = argparse.ArgumentParser(
        description='Titanic ML Pipeline - Modular training and evaluation'
    )

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
                                       help='Number of times to repeat calculation with different seeds (default: 1)')

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

    # Execute the appropriate command
    if args.command == 'cv':
        run_cross_validation(args)
    elif args.command == 'importance':
        calculate_feature_importance(args)
    elif args.command == 'drop-importance':
        calculate_drop_column_importance(args)
    elif args.command == 'train':
        train_final_model(args)
    elif args.command == 'pdp':
        plot_partial_dependence(args)
    elif args.command == 'correlation':
        plot_correlation_matrix(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
