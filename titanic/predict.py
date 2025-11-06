"""Titanic survival prediction CLI."""

from cli import create_cli
from titanic_data import TitanicData
from random_forest_classifier import RandomForestBinaryClassifier


if __name__ == "__main__":
    create_cli(
        data_class=TitanicData,
        classifier_class=RandomForestBinaryClassifier,
        default_data_path='./train.csv',
        description='Titanic ML Pipeline - Modular training and evaluation'
    )
