import streamlit as st

import os
from dotenv import load_dotenv
import requests

# Load the environment variables from the .env file
load_dotenv()
API_URL = os.getenv("API_URL")


st.title("Application de Classification de Texte")
# Charger une image ou un logo (optionnel)
st.image("me-2.webp")
# Personnalisation avec CSS (ajout de style)
st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Interface utilisateur


st.markdown("""
    Bienvenue dans l'outil de classification automatique de vidéos. Cet outil vous permet 
    de classer un nom de vidéo pour savoir s'il est comique ou non, grâce à un modèle de machine learning. 
    Veuillez entrer le titre d'une vidéo dans le champ ci-dessous pour obtenir une prédiction.
""")

# Entrée utilisateur
user_input = st.text_area("Entrez le nom de la vidéo à classifier")

# Préparation des données d'entrée
format_input = {
    "text": [user_input],
    "target": [0],  # Valeur fictive pour la classification
}

# Ajout d'une touche élégante avec des colonnes (pour centrer le bouton)
col1, col2, col3 = st.columns(3)
with col2:
    if st.button("Classer la Vidéo"):
        if user_input:
            request = requests.post(API_URL, json=format_input)
            target = request.json()
            # Afficher le résultat avec un style personnalisé
            if target["is_comic"] == True:

                st.success("Cette vidéo est classée comme **comique**.")

            else:
                st.warning("Cette vidéo n'est pas classée comme **comique**.")
        else:
            st.error("Veuillez entrer un texte à classifier.")

# Ajout d'une section de pied de page ou un remerciement
st.markdown("""
    ---
    Projet réalisé par [DIATTARA Amadou]""")
