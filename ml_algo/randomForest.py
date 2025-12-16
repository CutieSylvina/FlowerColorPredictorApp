import numpy as np
from collections import Counter

class DecisionTreeRegressorSimple:
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if self.y.ndim == 1:
            # treat leaf as scalar
            pass
        self.n_features = 1 if self.X.ndim == 1 else self.X.shape[1]
        self.tree = self._build_tree(self.X, self.y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return np.mean(y, axis=0)
        # Choose feature with max variance
        if X.ndim == 1:
            feature_idx = 0
            threshold = np.median(X)
            left_idx = X <= threshold
            right_idx = X > threshold
        else:
            feature_idx = np.argmax(np.var(X, axis=0))
            threshold = np.median(X[:, feature_idx])
            left_idx = X[:, feature_idx] <= threshold
            right_idx = X[:, feature_idx] > threshold

        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return np.mean(y, axis=0)
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return {"feature": feature_idx, "threshold": float(threshold), "left": left_tree, "right": right_tree}

    def _predict_tree(self, x, tree):
        if isinstance(tree, (np.ndarray, float, int)):
            return tree
        # x might be 1D or 2D (single row)
        val = x if x.ndim == 1 else x.ravel()
        if val[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(val, tree["left"])
        else:
            return self._predict_tree(val, tree["right"])

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_tree(x, self.tree) for x in X])

class DecisionTreeClassifierSimple:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        # Stopping conditions
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            # Return majority class
            counts = Counter(y)
            return counts.most_common(1)[0][0]

        # Split on feature with highest variance
        feature_idx = np.argmax(np.var(X, axis=0))
        threshold = np.median(X[:, feature_idx])

        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold

        if left_idx.sum() == 0 or right_idx.sum() == 0:
            counts = Counter(y)
            return counts.most_common(1)[0][0]

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {"feature": feature_idx, "threshold": threshold, "left": left_tree, "right": right_tree}

    def _predict_tree(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(x, tree["left"])
        else:
            return self._predict_tree(x, tree["right"])

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_tree(x, self.tree) for x in X])


class RandomForestClassifierSimple:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=2,
                 max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Default: sqrt of features
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(n_features)))

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            idxs = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[idxs]
            y_boot = y[idxs]

            # Random feature subset
            feat_idx = rng.choice(n_features, self.max_features, replace=False)

            # Train tree
            tree = DecisionTreeClassifierSimple(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append((tree, feat_idx))

        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = []

        for x in X:
            votes = []
            for tree, feat_idx in self.trees:
                pred = tree.predict(x[feat_idx].reshape(1, -1))[0]
                votes.append(pred)
            # Majority vote
            counts = Counter(votes)
            pred_class = counts.most_common(1)[0][0]
            preds.append(pred_class)

        return np.array(preds)