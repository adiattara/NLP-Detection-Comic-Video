#!/bin/bash

# Quitter immédiatement si une commande échoue
set -e

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les stopwords de NLTK
python -c "import nltk; nltk.download('stopwords')"

# Télécharger le modèle français de spaCy
python -m spacy download fr_core_news_sm

echo "Setup complet. Pour activer l'environnement virtuel, exécutez 'source venv/bin/activate'."

source venv/bin/activate