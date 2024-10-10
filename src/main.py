import click
import joblib
from sklearn.model_selection import cross_val_score

from data import make_dataset
from feature import make_features
from models import make_model
import pandas as pd

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
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
    X, _ = make_features(df)

    # Make predictions
    predictions = model.predict(X)

    # Save the predictions
    df['predictions'] = predictions
    df.to_csv(output_filename, index=False)
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

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



cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
