# Main Problems in Multivariate Analysis

Multivariate analysis involves dealing with multiple variables simultaneously. Here are four main types of machine learning problems in multivariate analysis:

## 1. Classification

- **Main Task:** Assign instances to predefined categories or classes.
- **Example Data:**

| Feature 1 | Feature 2 | Class Label |
|-----------|-----------|-------------|
| 5.1       | 3.5       | Setosa      |
| 4.9       | 3.0       | Versicolor  |
| 6.7       | 3.1       | Virginica   |

## 2. Regression/Prediction

- **Main Task:** Predict a continuous target variable based on input features.
- **Example Data:**

| Feature 1 | Feature 2 | Target Variable |
|-----------|-----------|-----------------|
| 5.1       | 3.5       | 1.4             |
| 4.9       | 3.0       | 4.0             |
| 6.7       | 3.1       | 5.8             |

## 3. Cluster Analysis

- **Main Task:** Group similar instances into clusters without predefined labels.
- **Example Data:**

| Feature 1 | Feature 2 |
|-----------|-----------|
| 5.1       | 3.5       |
| 4.9       | 3.0       |
| 6.7       | 3.1       |
| 1.4       | 0.2       |
| 4.0       | 1.3       |
| 5.8       | 2.7       |

## 4. Dimensionality Reduction

- **Main Task:** Reduce the number of features while preserving relevant information.
- **Example Data (Before Reduction):**

| Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|-----------|-----------|-----------|-----------|-----------|
| 2.3       | 1.5       | 3.1       | 0.7       | 5.2       |
| 4.1       | 2.3       | 1.9       | 3.5       | 2.8       |
| 3.2       | 1.8       | 2.5       | 4.0       | 1.7       |

- **Example Data (After Reduction):**

| Principal Component 1 | Principal Component 2 |
|-----------------------|-----------------------|
| -1.7                  | 0.5                   |
| 1.3                   | -1.2                  |
| 0.4                   | 0.7                   |

In this example, dimensionality reduction techniques (e.g., PCA) were applied to transform the original data with five features into a reduced set of two principal components, effectively capturing the most important information while reducing dimensionality.

# Common Sources of Data on the Internet

The internet is a vast repository of data from various sources, providing a rich landscape for data analysis. Here are common sources where valuable data can be found:

## 1. **Government Websites:**

- Many government agencies provide datasets on topics such as demographics, economics, health, and more. Examples include:
  - [Data.gov](https://www.data.gov/) (United States)
  - [Data.gov.uk](https://data.gov.uk/) (United Kingdom)
  - [Eurostat](https://ec.europa.eu/eurostat/data/database) (European Union)
  - [U.S. Census Bureau](https://www.census.gov/data.html) (United States)

## 2. **Open Data Platforms:**

- Dedicated platforms host a wide range of open datasets contributed by individuals, organizations, and governments. Examples include:
  - [Kaggle Datasets](https://www.kaggle.com/datasets)
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
  - [AWS Public Datasets](https://registry.opendata.aws/)

## 3. **Social Media APIs:**

- Social media platforms often provide APIs that allow access to user-generated data. Examples include:
  - [Twitter API](https://developer.twitter.com/en/docs/twitter-api)
  - [Facebook Graph API](https://developers.facebook.com/docs/graph-api)

## 4. **Other Platforms:**

- Explore specific platforms for diverse datasets:
  - [Gapminder](https://www.gapminder.org/)

## Where to Find Open Data:

- Open data initiatives aim to make datasets freely available for public use. Explore the following platforms:
  - [Open Data Portal](https://opendataportal.org/)
  - [DataHub](https://datahub.io/)
  - [Google Dataset Search](https://datasetsearch.research.google.com/)

Always ensure that you adhere to the licensing and usage terms associated with the datasets you find, especially when using them for analysis or projects.

# Accessing Classical Machine Learning Datasets in Python

Python, particularly with the scikit-learn library, provides easy access to several classical machine learning datasets. These datasets are commonly used for practice, experimentation, and testing machine learning algorithms.

## Using scikit-learn:

1. **Import scikit-learn:**
   ```python
   from sklearn import datasets
   ```

## Explore Available Datasets:

Scikit-learn has a variety of datasets, including the famous Iris dataset, breast cancer dataset, and more. To see the available datasets, use the following code:

```python
from sklearn import datasets

# Get information about the Iris dataset
iris = datasets.load_iris()
print(iris)
```
Replace load_iris with the name of the desired dataset. The datasets module provides various functions for loading different datasets, such as load_digits, load_wine, etc.

# Pass - Fail Task: Find and Document Open Datasets

Dear student, find 3 open and available datasets - each of one type classification, regression and cluster analysis. Then create three separate markdown files with description on how to attain each of them and place them in paths 
```
dataset
|
 - classification 
 |
   - dataset_name.md
 - regression
 |
  - dataset_name.md
 - clusters 
 |
  - dataset_name.md
```

Dataset cannot be duplicated with other students. The earlier MR wins in case those students claim the same dataset

Create also a note in md document `datasets/dataset list.md` on which dataset you have found.

