from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


def make_baseline():
    return DummyClassifier(strategy='most_frequent')

def make_random_forest():
    return RandomForestClassifier(max_depth=5,class_weight='balanced',n_estimators=100)

def make_logistic_regression():
    return LogisticRegression(class_weight='balanced',max_iter=1000)

def make_naive_bayes():
    return MultinomialNB()

def make_decision_tree():
    return DecisionTreeClassifier(max_depth=5,class_weight='balanced')