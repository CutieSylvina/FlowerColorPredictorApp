import numpy as np
from collections import Counter

class KNNSimple:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X).astype(float)
        self.y = np.asarray(y)
        if len(self.X) != len(self.y):
            raise ValueError("X and y lengths differ in KNNSimple.fit")
        return self

    def _distances(self, x):
        return np.sqrt(((self.X - x) ** 2).sum(axis=1))

    def predict(self, x):
        if self.X is None:
            raise RuntimeError("KNNSimple not fitted")
        d = self._distances(np.asarray(x))
        idx = np.argsort(d)[: self.k]
        votes = self.y[idx]
        counts = Counter(votes)
        return counts.most_common(1)[0][0]

    def predict_batch(self, X):
        return [self.predict(x) for x in X]