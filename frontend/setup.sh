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


source venv/bin/activate