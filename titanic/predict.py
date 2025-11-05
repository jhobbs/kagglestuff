import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def load_data(filepath):
    """Load the training data from CSV file."""
    data = pd.read_csv(filepath)
    data.head()
    return data


def parse_ticket(t):
    if pd.isna(t):
        return pd.Series([None, None])
    parts = t.split()
    if len(parts) == 2:
        code, number = parts[0], parts[1]
    elif len(parts) == 3:
        print(parts)
        code, number = parts[0] + parts[1], parts[2]
    else:
        code, number = None, parts[0]
    return pd.Series([code, number])


def engineer_features(data):
    """Create new features from the raw data."""
    data["Name_length"] = data["Name"].str.len()
    data["Cabin"] = data["Cabin"].fillna("Unknown")
    data["Cabin_count"] = data['Cabin'].str.split().str.len()
    data["Deck"] = data['Cabin'].str[0]
    data[["Ticket_code", "Ticket_number"]] = data["Ticket"].apply(parse_ticket)
    data["Ticket_number"] = pd.to_numeric(data["Ticket_number"], errors="coerce")

    print(data["Ticket_code"].unique())

    return data


def prepare_features(data, feature_columns):
    """Prepare feature matrix and target variable."""
    y = data["Survived"]
    X = pd.get_dummies(data[feature_columns])
    print(X)
    return X, y


def train_and_evaluate_model(X, y, n_estimators=100, max_depth=20, cv=5):
    """Train a Random Forest classifier and evaluate using cross-validation."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Print results
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    print(f"Mean Score: {scores.mean() * 100:.2f}%")

    # Train on full dataset for final model
    model.fit(X, y)

    return model, scores


def main():
    """Main function to orchestrate the training pipeline."""
    # Load data
    train_data = load_data("./train.csv")

    # Engineer features
    train_data = engineer_features(train_data)

    # Prepare features
    features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Name_length",
                "Fare", "Embarked", "Cabin_count", "Deck", "Ticket_code", "Ticket_number"]
    X, y = prepare_features(train_data, features)

    # Train and evaluate model using cross-validation
    model, cv_scores = train_and_evaluate_model(X, y, n_estimators=100, max_depth=20, cv=5)

    return model, cv_scores


if __name__ == "__main__":
    model, cv_scores = main()
