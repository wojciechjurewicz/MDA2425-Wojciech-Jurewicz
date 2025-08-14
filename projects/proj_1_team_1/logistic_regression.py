import os
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


while any(marker in os.getcwd() for marker in ('exercises', 'notebooks', 'students', 'research', 'projects')):
    os.chdir("..")
os.chdir("projects/proj_1_team_1")

df = pd.read_csv("mushrooms_preprocessed.csv", index_col=0)

X = df.drop(columns='poisonous')
y = df['poisonous']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

logreg_pipeline = make_pipeline(LogisticRegression(max_iter=1000))

logreg_pipeline.fit(X_train, y_train)

joblib.dump(logreg_pipeline, "logreg_pipeline.joblib")