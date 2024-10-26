import importlib
import click
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, precision_score, \
    f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

import mlflow
from data import make_dataset
from feature import make_features
import os
from dotenv import load_dotenv
import pandas as pd
from mlflow.models import infer_signature

# Load the environment variables from the .env file
load_dotenv()

# Récupère les variables nécessaires
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME')

# Vérifie que les variables sont bien définies
if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI is not set in the environment variables")
if not MLFLOW_S3_ENDPOINT_URL:
    raise ValueError("MLFLOW_S3_ENDPOINT_URL is not set in the environment variables")
if not EXPERIMENT_NAME:
    raise ValueError("EXPERIMENT_NAME is not set in the environment variables")

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

    _,model = model_func()

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
    X, y = make_features(df)

    # Object with .fit, .predict methods
    param,model = model_func()
    # Run k-fold cross validation. Print results

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    # Initialize an empty list to store f1-score for each fold
    f1 = []
    run_name = f"{EXPERIMENT_NAME} / {model_function}"
    signature = infer_signature(X)

    # Start a new MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Set the tags for the run

        # Log the model parameters to the run

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(x_train_fold, y_train_fold)

            # Predict the labels on the test data
            y_pred_fold = model.predict(x_test_fold)

            mlflow.log_param('param', param)
            mlflow.log_param('model_name', model_function)
            # Compute and log the evaluation metrics
            cr = classification_report(y_test_fold, y_pred_fold, output_dict=True)
            recall_0 = cr['0']['recall']
            f1_score_0 = cr['0']['f1-score']
            recall_1 = cr['1']['recall']
            f1_score_1 = cr['1']['f1-score']
            acc = accuracy_score(y_test_fold, y_pred_fold)
            precision = precision_score(y_test_fold, y_pred_fold, average='micro')
            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall_0", recall_0)
            mlflow.log_metric("f1_score_0", f1_score_0)
            mlflow.log_metric("recall_1", recall_1)
            mlflow.log_metric("f1_score_1", f1_score_1)
            f1.append(f1_score(y_test_fold, y_pred_fold))
            mlflow.log_metric("val_f1_fold", f1_score(y_test_fold, y_pred_fold))

        mlflow.log_metric("val_f1_score", np.mean(f1))
        model.fit(X, y)

        mlflow.sklearn.log_model(model, model_function, signature=signature)
        mlflow.log_artifact( 'models/vectorized.pkl')
        model_uri = f"runs:/{run.info.run_id}/model"
        print(model_uri)
        mlflow.register_model(model_uri, model_function)




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
