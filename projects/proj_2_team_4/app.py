import streamlit as st
import mlflow.pyfunc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from dotenv import load_dotenv
import numpy as np

# Load environment variables from the correct .env file location
env_path = '/Users/alanmakowski1/Desktop/project2/.env3.template'
load_dotenv(env_path)

# === Load raw data ===
train_path = os.getenv('TRAIN_RAW_MERGED_PATH')
df_train = pd.read_csv(train_path)
df = df_train.copy()
feature_names = [col for col in df.columns if col not in ["SalePrice"]]
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
target_name = "SalePrice"


# === Load model from MLflow ===
mlflow_path = os.getenv('MLFLOW_PATH')
mlflow.set_tracking_uri(mlflow_path)  # or another URL
MODEL_URI = 'runs:/f619af59195a40cc9c53100a39fcd71f/model'  # Replace with your run ID
model = mlflow.pyfunc.load_model(MODEL_URI)

# === Streamlit UI ===
st.title("🚜 Bluebook Bulldozers - Sale Price Predictor")
st.markdown("Predict the sale price of bulldozers using a trained CatBoost model.")

# === Feature input ===
st.sidebar.header("🔢 Input features")

st.write(df.head())

user_input = {}

for feature in feature_names:
    if feature in categorical_features:
        options = df[feature].dropna().astype(str).unique().tolist()
        options.sort()
        options = ["None"] + options  # Add "None" as the first option

        user_input[feature] = st.sidebar.selectbox(f"{feature}", options)
    else:
        try:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            
            if feature == "YearMade":
                col_min = max(1901, int(df[feature].min())) 
                col_max = int(df[feature].max())
            #elif feature == "MachineHoursCurrentMeter":
                #col_min = int(df[feature].min())
                #col_max = min(10000, int(df[feature].max()))  
            else:
                col_min = df[feature].min()
                col_max = df[feature].max()

            col_mean = df[feature].mean()

            # Check if it's an integer-like column
            if pd.api.types.is_integer_dtype(df[feature].dropna()):
                # Adjust mean within bounds if needed, handling missing values
                default_val = int(col_mean)
                if default_val < col_min:
                    default_val = col_min
                elif default_val > col_max:
                    default_val = col_max
                
                user_input[feature] = st.sidebar.number_input(
                    label=feature,
                    min_value=int(col_min),
                    max_value=int(col_max),
                    value=default_val,
                    step=1
                )
                #user_input[feature] = st.sidebar.slider(
                    #label=feature,
                    #min_value=int(col_min),
                    #max_value=int(col_max),
                    #value=default_val,
                    #step=1
                #)
            else:
                default_val = float(col_mean)
                if default_val < col_min:
                    default_val = col_min
                elif default_val > col_max:
                    default_val = col_max
                
                user_input[feature] = st.sidebar.number_input(
                    label=feature,
                    min_value=float(col_min),
                    max_value=float(col_max),
                    value=default_val,
                    step=(col_max - col_min) / 100 if col_max > col_min else 1.0        
                )
                #user_input[feature] = st.sidebar.slider(
                    #label=feature,
                    #min_value=float(col_min),
                    #max_value=float(col_max),
                    #value=default_val,
                    #step=(col_max - col_min) / 100 if col_max > col_min else 1.0
                #)
        except Exception as e:
            st.sidebar.warning(f"Skipping {feature}: not numeric or invalid ({e})")

input_df = pd.DataFrame([user_input])

st.write("Input to model:", input_df)

# === Predict ===
if st.button("Predict Sale Price"):
    try:
        prediction = np.expm1(model.predict(input_df)[0])
        st.success(f" Predicted Sale Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

