from fastapi import FastAPI
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import joblib
import spacy
from data_model import DataInput
import pandas as pd
import uvicorn
import os

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_tracking_uri:
    mlflow_tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(mlflow_tracking_uri)


def get_latest_production_model():
    # Récupérer tous les modèles enregistrés dans le Model Registry
    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_registered_models()

    # Parcourir les modèles pour trouver celui en Production
    for model in registered_models:
        latest_versions = client.get_latest_versions(model.name, stages=["Production"])
        if latest_versions:
            return latest_versions[0]  # Retourner la version en production
    return None


# Récupérer le modèle et le vectorizer associés au dernier modèle en Production
def load_model_and_vectorizer():
    # Récupérer le dernier modèle en Production
    latest_model = get_latest_production_model()
    if latest_model is None:
        raise ValueError("Aucun modèle en production trouvé")

    # Charger le modèle
    model_uri = f"runs:/{latest_model.run_id}/{latest_model.name}"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Charger le vectorizer associé à ce run
    print("here")
    vectorizer_uri = mlflow.artifacts.download_artifacts(
        f"runs:/{latest_model.run_id}/vectorized.pkl"
    )
    loaded_vectorizer = joblib.load(vectorizer_uri)

    return loaded_model, loaded_vectorizer


def make_features(df, vectorizer=None, fit=True):
    # reduction de bruit
    df["video_name"] = df["video_name"].str.replace(
        f"[{string.punctuation}]", " ", regex=True
    )

    french_stopwords = set(stopwords.words("french"))

    df["video_name"] = df["video_name"].apply(
        lambda x: " ".join([word for word in x.split() if word not in french_stopwords])
    )

    nlp = spacy.load("fr_core_news_sm")

    # lemmatisation
    df["video_name"] = df["video_name"].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x)])
    )

    if fit:
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
        X = vectorizer.fit_transform(df["video_name"])
        joblib.dump(vectorizer, "models/vectorized.pkl")

    else:
        if vectorizer is None:
            vectorizer = joblib.load("models/vectorized.pkl")

        X = vectorizer.transform(df["video_name"])

    y = df["is_comic"].values

    return X, y


model, vectorizer = load_model_and_vectorizer()
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}


@app.post("/api/v1/get_prediction")
def predict(data: DataInput):
    # Créer un DataFrame avec la donnée d'entrée
    data = {"video_name": data.text, "is_comic": data.target}

    df = pd.DataFrame(data)
    df.head()

    X, _ = make_features(df, vectorizer=vectorizer, fit=False)

    prediction = model.predict(X)

    return {"is_comic": bool(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, reload=True)
