# NLP TD 1: classification

L'objectif de ce TD est de créer un modèle "nom de vidéo" -> "is_comic" (is_comic vaut 1 si c'est une chronique humouristique, 0 sinon).

Dans ce TD, on s'intéresse surtout à la démarche. Pour chaque tâche:
- Bien poser le problème
- Avoir une baseline
- Experimenter diverses features et modèles
- Garder une trace écrite des expérimentations dans un rapport. Dans le rapport, on s'intéresse plus au sens du travail effectué (quelles expérimentations ont été faites, pourquoi, quelles conclusions) qu'à la liste de chiffres.
- Avoir une codebase clean, permettant de reproduire les expérimentations.

On se contentera de méthodes pré-réseaux de neurones. Nos features sont explicables et calculables "à la main".
## setup
- executer la commande suivante pour installer les dépendances
```
make setup
```

La codebase doit fournir les entry points suivant:
- Un entry point pour train prenant en entrée le nom de la fonction de création du modèle et le nom du modèle
- Un Exemple: pour le random forest  la fonction de création du modèle est make_random_forest, le nom du modèle est random_forest 
```
make train MODEL_FUNCTION=make_random_forest MODEL_NAME=random_forest
```
- Un entry point pour prédire, prenant en entrée le nom du modèle et sortant un fichier CSV avec les prédictions
```
make predict  MODEL_NAME=random_forest 
```
- Un entry point pour évaluer, prenant en entrée le nom du modèle et sortant les performances du modèle
```
make train MODEL_FUNCTION=make_random_forest MODEL_NAME=random_forest
```
- un entry point pour valider le modèle, prenant en entrée le nom du modèle et sortant les performances du modèle
``` 
make validate MODEL_NAME=random_forest
```

## Dataset

Dans [ce lien](https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/edit?usp=sharing), on a un CSV avec 2 colonnes:
- video_name: le nom de la video
- is_comic: est-ce une chronique humoristique

## Partie 1: Text classification: prédire si la vidéo est une chronique comique

### Tasks

- Créer une pipeline train, qui:
  - load le CSV
  - transforme les titres de videos en one-hot-encoded words (avec sklearn: CountVectorizer)
  - train un modèle (linéaire ou random forest)
  - dump le model
- Créer la pipeline predict, qui:
  - prend le modèle dumpé
  - prédit sur de nouveaux noms de video
  <br\>(comment cette partie one-hot encode les mots ? ERREUR à éviter: l'encoding en "predict" ne pointe pas les mots vers les mêmes index. Par exemple, en train, un nom de video avec le mot chronique aurait 1 dans la colonne \#10, mais en predict, il aurait 1 dans la colonne \#23)
- (optionel mais recommandé: créer une pipeline "evaluate" qui fait la cross-validation du modèle pour connaître ses performances)
- Transformer les noms de video avec différentes opérations de NLTK (Stemming, remove stop words) ou de CountVectorizer (min / max document frequency)
- Envoyer ce code (à la fin du cour)
- Itérer avec les différentes features / différents modèles pour trouver le plus performant
- Faire un rapport avec les différentes itérations faites, et les conclusions
- Envoyer le rapport et le code entraînant le meilleur modèle
