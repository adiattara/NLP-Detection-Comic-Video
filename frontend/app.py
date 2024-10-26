import streamlit as st
import pandas as pd
import sys
import os
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import mlflow
import joblib
import spacy

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
    vectorizer_uri = mlflow.artifacts.download_artifacts(f"runs:/{latest_model.run_id}/vectorized.pkl")
    loaded_vectorizer = joblib.load(vectorizer_uri)

    return loaded_model, loaded_vectorizer

def make_features(df,vectorizer=None, fit = True):
    # reduction de bruit
    df['video_name'] = df['video_name'].str.replace(f'[{string.punctuation}]', ' ', regex=True)

    french_stopwords = set(stopwords.words('french'))

    df['video_name'] = df['video_name'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in french_stopwords])
    )

    nlp = spacy.load('fr_core_news_sm')

    #lemmatisation
    df['video_name'] = df['video_name'].apply(

        lambda x: ' '.join([token.lemma_ for token in nlp(x)])
    )

    if fit:

        vectorizer = CountVectorizer(ngram_range=(1, 2),max_features=5000)
        X = vectorizer.fit_transform(df['video_name'])
        joblib.dump(vectorizer, 'models/vectorized.pkl')

    else:

        if vectorizer is None:
            vectorizer = joblib.load('models/vectorized.pkl')

        X = vectorizer.transform(df['video_name'])

    y = df['is_comic'].values

    return X, y
model, vectorizer = load_model_and_vectorizer()

st.title('Application de Classification de Texte')
# Charger une image ou un logo (optionnel)
st.image('me-2.webp')
# Personnalisation avec CSS (ajout de style)
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Interface utilisateur


st.markdown("""
    Bienvenue dans l'outil de classification automatique de vidéos. Cet outil vous permet 
    de classer un nom de vidéo pour savoir s'il est comique ou non, grâce à un modèle de machine learning. 
    Veuillez entrer le titre d'une vidéo dans le champ ci-dessous pour obtenir une prédiction.
""")

# Entrée utilisateur
user_input = st.text_area('Entrez le nom de la vidéo à classifier')

# Préparation des données d'entrée
format_input = {
    "video_name": [user_input],
    "is_comic": [0]  # Valeur fictive pour la classification
}

df = pd.DataFrame(format_input)

# Ajout d'une touche élégante avec des colonnes (pour centrer le bouton)
col1, col2, col3 = st.columns(3)
with col2:
    if st.button('Classer la Vidéo'):
        if user_input:
            # Prétraitement de l'entrée
            X_input, _ = make_features(df,vectorizer, fit=False)

            # Faire une prédiction
            prediction = model.predict(X_input)

            # Afficher le résultat avec un style personnalisé
            if prediction[0] == 1:
                st.success('Cette vidéo est classée comme **comique**.')
            else:
                st.warning('Cette vidéo n\'est pas classée comme **comique**.')
        else:
            st.error('Veuillez entrer un texte à classifier.')

# Ajout d'une section de pied de page ou un remerciement
st.markdown("""
    ---
    **Projet réalisé par [Ton Nom]** - Présentation pour RFI.
""")
