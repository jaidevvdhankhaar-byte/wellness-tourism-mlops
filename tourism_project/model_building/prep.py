
import pandas as pd
import os
import json

from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "tourism_project/model_building/artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Remove unnecessary columns
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
df = df.drop_duplicates()

# Remove records without target
df = df.dropna(subset=["ProdTaken"])

# Convert target to integer
df["ProdTaken"] = df["ProdTaken"].astype(int)

# Features available before customer contact
FEATURES = [
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

TARGET = "ProdTaken"

X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save train and test datasets
train_data = X_train.copy()
train_data[TARGET] = y_train

test_data = X_test.copy()
test_data[TARGET] = y_test

train_data.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

test_data.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)

with open(f"{OUTPUT_DIR}/feature_columns.json", "w") as f:
    json.dump(FEATURES, f, indent=4)

print("DATA PREPARATION SUCCESSFUL")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
print("Features used:", len(FEATURES))
print("Training target distribution:")
print(y_train.value_counts())
print("Testing target distribution:")
print(y_test.value_counts())
print("Train and test files saved successfully!")
