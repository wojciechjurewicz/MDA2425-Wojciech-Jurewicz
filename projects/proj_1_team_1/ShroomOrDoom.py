import os
import sys
import tkinter as tk
from PIL import Image, ImageTk
import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline

all_column_names = ['bruises', 'gill-size', 'stalk-shape', 'ring-number', 'population', 'cap-shape_b', 'cap-shape_c', 'cap-shape_f', 'cap-shape_k', 'cap-shape_s', 'cap-shape_x', 'cap-surface_f', 'cap-surface_g', 'cap-surface_s', 'cap-surface_y', 'cap-color_b', 'cap-color_c', 'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w', 'cap-color_y', 'odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', 'odor_s', 'odor_y', 'gill-attachment_a', 'gill-attachment_f', 'gill-spacing_c', 'gill-spacing_w', 'gill-color_b', 'gill-color_e', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_o', 'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y', 'stalk-surface-above-ring_f', 'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s', 'stalk-surface-above-ring_y', 'stalk-surface-below-ring_f', 'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s', 'stalk-surface-below-ring_y', 'stalk-color-above-ring_b', 'stalk-color-above-ring_c', 'stalk-color-above-ring_e', 'stalk-color-above-ring_g', 'stalk-color-above-ring_n', 'stalk-color-above-ring_o', 'stalk-color-above-ring_p', 'stalk-color-above-ring_w', 'stalk-color-above-ring_y', 'stalk-color-below-ring_b', 'stalk-color-below-ring_c', 'stalk-color-below-ring_e', 'stalk-color-below-ring_g', 'stalk-color-below-ring_n', 'stalk-color-below-ring_o', 'stalk-color-below-ring_p', 'stalk-color-below-ring_w', 'stalk-color-below-ring_y', 'veil-color_n', 'veil-color_o', 'veil-color_w', 'veil-color_y', 'ring-type_e', 'ring-type_f', 'ring-type_l', 'ring-type_n', 'ring-type_p', 'spore-print-color_b', 'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n', 'spore-print-color_o', 'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w', 'spore-print-color_y', 'habitat_d', 'habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_u', 'habitat_w']

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "src"))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from custom_transformers import (
    DropColumnTransformer,
    CustomLabelEncoder,
    CustomOneHotEncoder,
)

# Load the preprocessing pipeline and model
model = joblib.load(os.path.join(current_dir, "logreg_pipeline.joblib"))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns, order=None):
        self.columns = columns
        self.order = order if order is not None else {}
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            if column in self.order:
                self.encoders[column] = OrdinalEncoder(categories=[self.order[column]])
            else:
                self.encoders[column] = OrdinalEncoder()
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[[column]])
        return X_transformed

preprocessing = make_pipeline(
    DropColumnTransformer(columns=["stalk-root"]), 
    CustomLabelEncoder(columns=["bruises"]),
    
    CustomOrdinalEncoder(
        columns=["gill-size", "stalk-shape", "ring-number", "population"],
        order={
            "gill-size": ["n", "b"],  # narrow < broad
            "stalk-shape": ["t", "e"],  # tapering < enlarging
            "ring-number": ["n", "o", "t"],  # none < one < two
            "population": ["y", "v", "s", "n", "c", "a"]  # solitary < several < scattered < numerous < clustered < abundant
        }
    ),
    
    CustomOneHotEncoder(columns=[
        "cap-shape", "cap-surface", "cap-color", "odor", "gill-attachment",
        "gill-spacing", "gill-color", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
        "veil-color", "ring-type", "spore-print-color", "habitat"
    ])
)

# Descriptive mushroom feature options
features = {
    "cap-shape": {'b': "bell", 'c': "conical", 'x': "convex", 'f': "flat", 'k': "knobbed", 's': "sunken"},
    "cap-surface": {'f': "fibrous", 'g': "grooves", 'y': "scaly", 's': "smooth"},
    "cap-color": {'n': "brown", 'b': "buff", 'c': "cinnamon", 'g': "gray", 'r': "green", 'p': "pink",
                  'u': "purple", 'e': "red", 'w': "white", 'y': "yellow"},
    "bruises": {'t': "bruises", 'f': "no bruises"},
    "odor": {'a': "almond", 'l': "anise", 'c': "creosote", 'y': "fishy", 'f': "foul", 'm': "musty",
             'n': "none", 'p': "pungent", 's': "spicy"},
    "gill-attachment": {'a': "attached", 'd': "descending", 'f': "free", 'n': "notched"},
    "gill-spacing": {'c': "close", 'w': "crowded", 'd': "distant"},
    "gill-size": {'b': "broad", 'n': "narrow"},
    "gill-color": {'k': "black", 'n': "brown", 'b': "buff", 'h': "chocolate", 'g': "gray", 'r': "green",
                   'o': "orange", 'p': "pink", 'u': "purple", 'e': "red", 'w': "white", 'y': "yellow"},
    "stalk-shape": {'e': "enlarging", 't': "tapering"},
    "stalk-root": {'b': "bulbous", 'c': "club", 'u': "cup", 'e': "equal", 'z': "rhizomorphs",
                   'r': "rooted", '?': "missing"},
    "stalk-surface-above-ring": {'f': "fibrous", 'y': "scaly", 'k': "silky", 's': "smooth"},
    "stalk-surface-below-ring": {'f': "fibrous", 'y': "scaly", 'k': "silky", 's': "smooth"},
    "stalk-color-above-ring": {'n': "brown", 'b': "buff", 'c': "cinnamon", 'g': "gray", 'o': "orange", 'p': "pink",
                               'e': "red", 'w': "white", 'y': "yellow"},
    "stalk-color-below-ring": {'n': "brown", 'b': "buff", 'c': "cinnamon", 'g': "gray", 'o': "orange", 'p': "pink",
                               'e': "red", 'w': "white", 'y': "yellow"},
    "veil-color": {'n': "brown", 'o': "orange", 'w': "white", 'y': "yellow"},
    "ring-number": {'n': "none", 'o': "one", 't': "two"},
    "ring-type": {'e': "evanescent", 'f': "flaring", 'l': "large", 'n': "none",
                  'p': "pendant"},
    "spore-print-color": {'k': "black", 'n': "brown", 'b': "buff", 'h': "chocolate", 'r': "green", 'o': "orange",
                          'u': "purple", 'w': "white", 'y': "yellow"},
    "population": {'a': "abundant", 'c': "clustered", 'n': "numerous", 's': "scattered", 'v': "several", 'y': "solitary"},
    "habitat": {'g': "grasses", 'l': "leaves", 'm': "meadows", 'p': "paths", 'u': "urban", 'w': "waste", 'd': "woods"}
}

# Track classification state
classified = False

# Initialize main window
root = tk.Tk()
root.title("Shroom or Doom")
root.geometry("1160x800")
root.resizable(True, True)

# Logo image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "logo.jpg")
img = Image.open(img_path)
img = img.resize((100, 100))
photo = ImageTk.PhotoImage(img)

# Frame for logo and title
header_frame = tk.Frame(root)
header_frame.grid(row=0, column=0, columnspan=12, pady=10)
tk.Label(header_frame, image=photo).pack(side="left")
tk.Label(header_frame, text="Shroom or Doom", font=("Helvetica", 24, "bold")).pack(side="left", padx=20)

# Feature input
variables = {}
row_offset = 1

def handle_change(*_):
    global classified
    if classified:
        result_label.config(text="Classification:", bg="darkgreen", fg="white")
        classified = False

for i, (feature, options) in enumerate(features.items()):
    tk.Label(root, text=feature.replace('-', ' ').capitalize()).grid(row=row_offset + i, column=0, sticky="w", padx=10)
    var = tk.StringVar()
    var.trace_add("write", handle_change)
    variables[feature] = var
    for j, (code, desc) in enumerate(options.items()):
        tk.Radiobutton(root, text=desc, variable=var, value=code).grid(row=row_offset + i, column=j + 1, sticky="w")

# Classify function
def classify():
    global classified

    for feature, var in variables.items():
        if var.get() == "":
            result_label.config(
                text=f"Please select a value for '{feature.replace('-', ' ').capitalize()}'",
                bg="orange", fg="black"
            )
            return
    

    # Prepare one-row DataFrame
    input_data = {feature: [var.get()] for feature, var in variables.items()}
    input_df = pd.DataFrame(input_data)

    X_input = preprocessing.fit_transform(input_df)

    X_input = X_input.reindex(columns=all_column_names, fill_value=0)

    
    prediction = model.predict(X_input)[0]
    print("Prediction:", prediction)
    if prediction == 'p':
        result_label.config(text="Classification: poisonous", bg="darkred", fg="white")
    else:
        result_label.config(text="Classification: edible", bg="darkgreen", fg="white")
    classified = True

# Reset choices
def reset():
    global classified
    for var in variables.values():
        var.set("")
    result_label.config(text="Classification:", bg="darkgreen", fg="white")
    classified = False

# Buttons for classification and reset
btn_frame = tk.Frame(root)
btn_frame.grid(row=row_offset + len(features), column=0, columnspan=12, pady=20)
tk.Button(btn_frame, text="Classify", width=15, command=classify).pack(side="left", padx=20)
tk.Button(btn_frame, text="Reset", width=15, command=reset).pack(side="left", padx=20)

# Result
result_label = tk.Label(root, text="Classification:", font=("Helvetica", 18, "bold"), fg="white", bg="darkgreen", padx=20, pady=10)
result_label.grid(row=row_offset + len(features) + 1, column=0, columnspan=12, pady=20)

root.mainloop()