
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

def baseline():
    params = {
        'strategy': 'most_frequent'
    }
    return params, DummyClassifier(strategy='most_frequent')

def random_forest():
    params = {
        'max_depth': 5,
        'class_weight': 'balanced',
        'n_estimators': 100
    }
    return params,RandomForestClassifier(**params)

def logistic_regression():
    params = {
        'max_iter': 1000,
        'class_weight': 'balanced',
    }
    return None, LogisticRegression(**params)

def naive_bayes():

    return None, MultinomialNB()

def decision_tree():
    params = {
        'max_depth': 5,
        'class_weight': 'balanced',
    }
    return params,DecisionTreeClassifier(**params)

