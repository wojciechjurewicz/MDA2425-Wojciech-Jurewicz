import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from sklearn.feature_extraction.text import CountVectorizer


# Custom transformer class to detect and remove outliers
class CustomOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], threshold: float = 3):
        self.columns = columns
        self.threshold = threshold
        self._outliers = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # Calculate z-scores for specified columns
        z_scores = pd.DataFrame(index=X.index)
        for col in self.columns:
            if col in X.columns:
                # Fill NaN with mean for z-score calculation
                col_filled = X[col].fillna(X[col].mean())
                z_scores[col] = (col_filled - col_filled.mean()) / col_filled.std()

        # Identify outliers
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        
        # Remove outliers
        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, patterns_to_remove: List[str]):
        self.column = column
        self.patterns_to_remove = patterns_to_remove
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        # Remove all specified patterns
        for pattern in self.patterns_to_remove:
            X_transformed[self.column] = X_transformed[self.column].str.replace(pattern, '', case=False, regex=True)
        return X_transformed
    
class TfidfEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, max_features: int = 500):
        self.column = column
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.feature_names = []

    def fit(self, X, y=None):
        texts = X[self.column].fillna('').astype(str)
        self.vectorizer.fit(texts)
        self.feature_names = [f"{self.column}_tfidf_{feat}" for feat in self.vectorizer.get_feature_names_out()]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        texts = X_transformed[self.column].fillna('').astype(str)
        tfidf_matrix = self.vectorizer.transform(texts)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.feature_names, index=X_transformed.index)
        
        # Drop original text column and add TF-IDF features
        X_transformed = X_transformed.drop(columns=[self.column])
        X_transformed = pd.concat([X_transformed, tfidf_df], axis=1)
        return X_transformed
    
class CustomTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, vocab_size: int = 10000):
        self.column = column
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece()
        self.trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

    def fit(self, X, y=None):
        texts = X[self.column].fillna('').astype(str).tolist()
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.column] = X_[self.column].fillna('').astype(str).apply(lambda x: self.tokenizer.encode(x).tokens)
        return X_
    

class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

class CustomTokenizerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, vocab_size: int = 1000):
        self.column = column
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece()
        self.trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

    def fit(self, X, y=None):
        texts = X[self.column].fillna('').astype(str).tolist()
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)

        # Zapisz słownik tokenów (pomijając specjalne tokeny)
        self.vocab = [token for token in self.tokenizer.get_vocab().keys() if token not in {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}]
        return self

    def transform(self, X):
        texts = X[self.column].fillna('').astype(str).tolist()

        X_tokenized = []
        for text in texts:
            tokens = set(self.tokenizer.encode(text).tokens)
            X_tokenized.append([1 if token in tokens else 0 for token in self.vocab])

        return np.array(X_tokenized)

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column}_{token}" for token in self.vocab]
    
class ToDataFrameFromColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_names = self.column_transformer.get_feature_names_out()
        return pd.DataFrame(X, columns=feature_names)