import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

PROJECT_DIR = "predictive_maintenance_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "engine_data.csv")
TARGET_COL = "Engine_Condition"
FEATURE_COLS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
]
SPLIT_FILES = {
    "Xtrain": "Xtrain.csv",
    "Xtest": "Xtest.csv",
    "ytrain": "ytrain.csv",
    "ytest": "ytest.csv",
}

os.makedirs(DATA_DIR, exist_ok=True)

# Load dataset from the local project data folder
df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset loaded from {RAW_DATA_PATH}")
print(f"Original shape: {df.shape}")

# Standardize column names and clean data
df.columns = FEATURE_COLS + [TARGET_COL]
df.drop_duplicates(inplace=True)

# All columns are sensor readings or the target label; no unnecessary columns to remove
X = df[FEATURE_COLS]
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

split_paths = {
    "Xtrain": os.path.join(DATA_DIR, SPLIT_FILES["Xtrain"]),
    "Xtest": os.path.join(DATA_DIR, SPLIT_FILES["Xtest"]),
    "ytrain": os.path.join(DATA_DIR, SPLIT_FILES["ytrain"]),
    "ytest": os.path.join(DATA_DIR, SPLIT_FILES["ytest"]),
}

Xtrain.to_csv(split_paths["Xtrain"], index=False)
Xtest.to_csv(split_paths["Xtest"], index=False)
ytrain.to_csv(split_paths["ytrain"], index=False)
ytest.to_csv(split_paths["ytest"], index=False)

print("Train and test sets saved locally in predictive_maintenance_project/data/")
print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")
