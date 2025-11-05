import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance


def load_data(filepath):
    """Load the training data from CSV file."""
    data = pd.read_csv(filepath)
    data.head()
    return data


def parse_ticket(t):
    """Parse ticket information into code and number."""
    if pd.isna(t):
        return pd.Series([None, None])
    parts = t.split()
    if len(parts) == 2:
        code, number = parts[0], parts[1]
    elif len(parts) == 3:
        code, number = parts[0] + parts[1], parts[2]
    else:
        code, number = None, parts[0]
    return pd.Series([code, number])


def engineer_features(data):
    """Create new features from the raw data."""
    data["Name_length"] = data["Name"].str.len()
    data["Name_words"] = data["Name"].str.split().str.len()

    # Extract last name (everything before the comma)
    data["Last_name"] = data["Name"].str.split(',').str[0]

    # Count passengers with the same last name
    lastname_counts = data["Last_name"].value_counts()
    data["Same_lastname_count"] = data["Last_name"].map(lastname_counts)

    data["Cabin"] = data["Cabin"].fillna("Unknown")
    data["Cabin_count"] = data['Cabin'].str.split().str.len()
    data["Deck"] = data['Cabin'].str[0]
    data[["Ticket_code", "Ticket_number"]] = data["Ticket"].apply(parse_ticket)
    data["Ticket_number"] = pd.to_numeric(data["Ticket_number"], errors="coerce")
    return data


def get_feature_list():
    """Return the list of features to use in the model."""
    return ["Pclass", "Sex", "SibSp", "Parch", "Age", "Name_length", "Name_words",
            "Same_lastname_count", "Fare", "Embarked", "Cabin_count", "Deck",
            "Ticket_code", "Ticket_number"]


def prepare_features(data, feature_columns):
    """Prepare feature matrix and target variable."""
    y = data["Survived"]
    X = pd.get_dummies(data[feature_columns])
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

    # Train command
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                        help='Train final model on all data')

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == 'cv':
        run_cross_validation(args)
    elif args.command == 'importance':
        calculate_feature_importance(args)
    elif args.command == 'train':
        train_final_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
