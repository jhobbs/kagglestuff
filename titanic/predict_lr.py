"""Titanic survival prediction CLI using Logistic Regression.

This is an example demonstrating how the dynamic CLI system
works with different classifier types and their parameters.
"""

from cli import create_cli
from titanic_data import TitanicData
from logistic_regression_classifier import LogisticRegressionBinaryClassifier


if __name__ == "__main__":
    create_cli(
        data_class=TitanicData,
        classifier_class=LogisticRegressionBinaryClassifier,
        default_data_path='./train.csv',
        description='Titanic ML Pipeline - Logistic Regression'
    )
