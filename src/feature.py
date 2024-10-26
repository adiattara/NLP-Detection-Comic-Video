import string

import joblib
import spacy
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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