from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class MeasurementCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _convert_numeric(self, value):
        if pd.isna(value) or value == 'None or Unspecified':
            return np.nan
        try:
            # Convert to string in case it's a float
            value = str(value)
            # Remove quotes, inch, and other text
            value = value.replace('"', '').replace("'", '').replace(' inch', '').replace(' Inch', '')
            # Extract first number found
            import re
            numbers = re.findall(r'[\d.]+', value)
            return float(numbers[0]) if numbers else np.nan
        except (ValueError, IndexError):
            return np.nan

    def _convert_stick_length(self, value):
        if pd.isna(value) or value == 'None or Unspecified':
            return np.nan
        try:
            # Split into feet and inches
            parts = str(value).replace("'", '').split('"')[0].split()
            feet = float(parts[0])
            inches = float(parts[1]) if len(parts) > 1 else 0
            # Convert to inches
            return feet * 12 + inches
        except (ValueError, IndexError):
            return np.nan

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # Convert measurements
        measurement_cols = [
            'Tire_Size', 'Undercarriage_Pad_Width'
        ]
        
        for col in measurement_cols:
            if col in X.columns:
                X_transformed[col] = X[col].apply(self._convert_numeric)
        
        if 'Stick_Length' in X.columns:
            X_transformed['Stick_Length'] = X['Stick_Length'].apply(self._convert_stick_length)
            
        return X_transformed


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns: Optional[List[str]] = None, fill_value=None):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=fill_value)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], ordering: Optional[Dict[str, Dict[str, int]]] = None):
        self.columns = columns
        self.ordering = ordering or {}
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            if column in self.ordering:
                # Create a custom LabelEncoder for ordered categories
                unique_values = list(self.ordering[column].keys())
                encoder = LabelEncoder()
                encoder.fit(unique_values)
                self.encoders[column] = encoder
            else:
                # Use standard LabelEncoder for unordered categories
                self.encoders[column] = LabelEncoder()
                self.encoders[column].fit(X[column].fillna('None or Unspecified'))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            if column in self.ordering:
                # Map values using predefined ordering
                X_transformed[column] = X[column].map(self.ordering[column])
                # Handle unknown values
                X_transformed[column] = X_transformed[column].fillna(-1)
            else:
                try:
                    X_transformed[column] = self.encoders[column].transform(
                        X[column].fillna('None or Unspecified')
                    )
                except ValueError:
                    # Handle unknown values
                    X_transformed[column] = -1
        return X_transformed


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], handle_unknown: str = 'ignore'):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown
            )
            # Fill NA with 'None or Unspecified' before fitting
            X_filled = X[column].fillna('None or Unspecified')
            self.encoders[column].fit(X_filled.values.reshape(-1, 1))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            # Fill NA with 'None or Unspecified' before transforming
            X_filled = X[column].fillna('None or Unspecified')
            encoded = pd.DataFrame(
                self.encoders[column].transform(X_filled.values.reshape(-1, 1)),
                columns=self.encoders[column].get_feature_names_out([column]),
                index=X.index
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=column), encoded],
                axis=1
            )
        return X_transformed


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fill NaN with mean before fitting
        X_filled = X[self.columns].fillna(X[self.columns].mean())
        self.scaler.fit(X_filled)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        # Fill NaN with mean before transforming
        X_filled = X[self.columns].fillna(X[self.columns].mean())
        X_transformed[self.columns] = self.scaler.transform(X_filled)
        return X_transformed


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


# Custom transformer for Normalization
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


class ExtendedLabelEncoder(CustomLabelEncoder):
    """Extended Label Encoder that supports predefined ordering for categories"""
    def __init__(self, columns: List[str], ordering: Optional[Dict[str, Dict[str, int]]] = None):
        super().__init__(columns)
        self.ordering = ordering or {}
        
    def fit(self, X, y=None):
        for column in self.columns:
            if column in self.ordering:
                # Create a custom LabelEncoder for ordered categories
                unique_values = list(self.ordering[column].keys())
                encoder = LabelEncoder()
                encoder.fit(unique_values)
                self.encoders[column] = encoder
            else:
                # Use standard LabelEncoder for unordered categories
                self.encoders[column] = LabelEncoder()
                self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            if column in self.ordering:
                # Map values using predefined ordering
                X_transformed[column] = X[column].map(self.ordering[column])
                # Handle unknown values
                X_transformed[column] = X_transformed[column].fillna(-1)
            else:
                try:
                    X_transformed[column] = self.encoders[column].transform(X[column])
                except ValueError:
                    # Handle unknown values for non-ordered categories
                    X_transformed[column] = -1
        return X_transformed

class ExtendedOneHotEncoder(CustomOneHotEncoder):
    """Extended One Hot Encoder that handles unknown values and frequency-based filtering"""
    def __init__(self, columns: List[str], handle_unknown: str = 'ignore', 
                 max_categories: int = None, min_frequency: float = None):
        super().__init__(columns)
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.category_maps = {}
        
    def fit(self, X, y=None):
        for column in self.columns:
            # Calculate value frequencies
            value_counts = X[column].value_counts(normalize=True)
            
            # Filter categories based on frequency threshold
            if self.min_frequency is not None:
                value_counts = value_counts[value_counts >= self.min_frequency]
                
            # Take top N categories if max_categories is specified
            if self.max_categories is not None:
                value_counts = value_counts.nlargest(self.max_categories)
                
            # Store selected categories
            self.category_maps[column] = value_counts.index.tolist()
            
            # Create and fit encoder
            self.encoders[column] = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown,
                categories=[self.category_maps[column]]
            )
            # Transform column to only include selected categories
            column_data = X[column].apply(lambda x: x if x in self.category_maps[column] else None)
            self.encoders[column].fit(column_data.to_frame())
            
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            if column in X.columns:
                # Map values not in selected categories to None
                column_data = X[column].apply(
                    lambda x: x if x in self.category_maps[column] else None
                )
                encoded = pd.DataFrame(
                    self.encoders[column].transform(column_data.to_frame()),
                    columns=self.encoders[column].get_feature_names_out([column]),
                    index=X.index
                )
                X_transformed = pd.concat(
                    [X_transformed.drop(columns=column), encoded],
                    axis=1
                )
        return X_transformed


class ModelDescriptionTransformer(BaseEstimator, TransformerMixin):
    """Transformer for handling model description related columns"""
    def __init__(self, drop_original: bool = True):
        self.drop_original = drop_original
        self.model_components = None
        
    def fit(self, X, y=None):
        # Store unique values for each component during fit
        self.model_components = {
            'base_model': X['fiBaseModel'].unique(),
            'secondary_desc': X['fiSecondaryDesc'].dropna().unique(),
            'model_series': X['fiModelSeries'].dropna().unique(),
            'model_descriptor': X['fiModelDescriptor'].dropna().unique()
        }
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        
        # Create combined features
        X_transformed['model_full'] = X_transformed['fiBaseModel'].fillna('') + '_' + \
                                    X_transformed['fiSecondaryDesc'].fillna('') + '_' + \
                                    X_transformed['fiModelSeries'].fillna('') + '_' + \
                                    X_transformed['fiModelDescriptor'].fillna('')
        
        # Drop original columns if specified
        if self.drop_original:
            cols_to_drop = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 
                           'fiModelDescriptor', 'fiModelDesc']
            X_transformed = X_transformed.drop(columns=cols_to_drop)
            
        return X_transformed 

class ProductClassTransformer(BaseEstimator, TransformerMixin):
    """Transformer to extract meaningful components from fiProductClassDesc."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        
        # Extract equipment type (part before the hyphen)
        X_transformed['equipment_type'] = X_transformed['fiProductClassDesc'].str.extract(r'^([^-]+)')
        
        # Extract capacity/power range
        # Handle both Horsepower and Operating Capacity cases
        X_transformed['capacity_min'] = X_transformed['fiProductClassDesc'].str.extract(r'(\d+\.?\d*)\s*to').astype(float)
        X_transformed['capacity_max'] = X_transformed['fiProductClassDesc'].str.extract(r'to\s*(\d+\.?\d*)').astype(float)
        
        # Create capacity/power unit feature
        X_transformed['capacity_unit'] = X_transformed['fiProductClassDesc'].apply(lambda x: 
            'Horsepower' if 'Horsepower' in str(x) 
            else 'Lb Operating Capacity' if 'Operating Capacity' in str(x)
            else 'Other'
        )
        
        # Clean up equipment type
        X_transformed['equipment_type'] = X_transformed['equipment_type'].str.strip()
        
        # Drop original column since we've extracted its components
        X_transformed = X_transformed.drop('fiProductClassDesc', axis=1)
        
        return X_transformed
    

class DateProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.year_mode = None
        
    def fit(self, X, y=None):
        # Calculate mode of YearMade for values after 1900
        self.year_mode = X[X['YearMade'] > 1900]['YearMade'].mode().iloc[0]
        return self
        
    def transform(self, X):
        X_transformed = X.copy()
        
        # Extract date components
        X_transformed['sale_year'] = X_transformed['saledate'].dt.year
        X_transformed['sale_month'] = X_transformed['saledate'].dt.month
        X_transformed['sale_quarter'] = X_transformed['saledate'].dt.quarter
        X_transformed['sale_day_of_week'] = X_transformed['saledate'].dt.dayofweek
        X_transformed['sale_is_weekend'] = X_transformed['saledate'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Calculate machine age
        X_transformed['machine_age'] = X_transformed['sale_year'] - X_transformed['YearMade']
        
        # Fix YearMade anomaly (year 1000)
        X_transformed.loc[X_transformed['YearMade'] < 1900, 'YearMade'] = self.year_mode
        
        # Drop original date column
        X_transformed = X_transformed.drop('saledate', axis=1)
        
        return X_transformed
    