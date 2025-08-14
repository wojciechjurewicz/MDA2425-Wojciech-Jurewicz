from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class OutlierRemoveTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
        self.means = X[self.numerical_cols].mean()
        self.stds = X[self.numerical_cols].std()
        return self

    def transform(self, X):
        X_num = X[self.numerical_cols]
        z_scores = ((X_num - self.means) / self.stds).abs()
        outliers = (z_scores > self.threshold).any(axis=1)
        return X[~outliers]