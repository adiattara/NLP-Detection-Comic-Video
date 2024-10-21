import importlib

import click
import joblib
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from data import make_dataset
from feature import make_features

import pandas as pd

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_function", default="make_logistic_regression", help="look models.py for model functions")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename, model_function):

    model_module = importlib.import_module('models')
    model_func = getattr(model_module, model_function)
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = model_func()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    model = joblib.load(model_dump_filename)

    # Load the input data
    df = pd.read_csv(input_filename)

    # Transform the input data
    X, _ = make_features(df,fit=False)

    # Make predictions
    predictions = model.predict(X)

    # Save the predictions
    df['predictions'] = predictions
    df.to_csv(output_filename, index=False)
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_function", default="make_logistic_regression", help="look models.py for model functions")
def evaluate(input_filename,model_function):
    model_module = importlib.import_module('models')
    model_func = getattr(model_module, model_function)
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, fit=False)

    # Object with .fit, .predict methods
    model = model_func()
    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):

    # Define the number of folds for cross-validation
    n_folds = 5

    # Perform k-fold cross-validation
    scores = cross_val_score(model, X, y, cv=n_folds)

    # Print the cross-validation results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean()}")
    print(f"Standard deviation: {scores.std()}")



@click.command()
@click.option("--input_filename", default="data/raw/test.csv", help="File test data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to load model")
def validate(input_filename, model_dump_filename):
    model = joblib.load(model_dump_filename)
    df = pd.read_csv(input_filename)
    X, y = make_features(df, fit=False)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    recall = recall_score(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{conf_matrix}")




cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(validate)


if __name__ == "__main__":
    cli()
