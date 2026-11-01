
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

assert os.path.exists(DATA_PATH), "Dataset not found!"

df = pd.read_csv(DATA_PATH)

# Remove unnecessary index column
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

assert not missing_columns, f"Missing columns: {missing_columns}"

print("DATA REGISTRATION SUCCESSFUL")
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())

print("\nDataset validation completed!")
