import joblib

from sklearn.feature_extraction.text import CountVectorizer



def make_features(df,fit = True):



    if fit:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['video_name'])
        joblib.dump(vectorizer, 'models/vectorized.pkl')

    else:
        vectorizer = joblib.load('models/vectorized.pkl')
        X = vectorizer.transform(df['video_name'])

    y = df['is_comic'].values

    return X, y