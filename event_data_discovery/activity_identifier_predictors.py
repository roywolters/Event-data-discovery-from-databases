import random
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


def make_sklearn_pipeline(clf):
    pipeline = Pipeline([
        ('dictvectorizer', DictVectorizer(sparse=False)),
        ('normalizer', StandardScaler()),
        ('clf', clf)
    ])
    return pipeline


class OnePerTS(BaseEstimator, ClassifierMixin):

    def __init__(self, random_state):
        self.random_state = random_state

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        return self

    def predict(self, X):
        # y = [0] * len(X)
        y = np.zeros(len(X))
        ts_to_cand = dict()
        for i, x in enumerate(X):
            if x['timestamp_attribute_id'] not in ts_to_cand:
                ts_to_cand[x['timestamp_attribute_id']] = list()
            ts_to_cand[x['timestamp_attribute_id']].append(i)
        for i in ts_to_cand:
            y[random.choice(ts_to_cand[i])] = 1
        return y
