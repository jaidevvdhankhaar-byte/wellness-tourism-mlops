
import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Paths
ARTIFACT_DIR = "tourism_project/model_building/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load prepared datasets
train = pd.read_csv(f"{ARTIFACT_DIR}/train.csv")
test = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")

TARGET = "ProdTaken"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

# Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Models and hyperparameter distributions
models = {
    "Random Forest": RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        random_state=42,
        class_weight="balanced"
    )
}

parameter_grid = {
    "classifier__n_estimators": [100, 150, 200],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"]
}

results = []
best_score = -1
best_model = None
best_name = None
best_params = None

# Model experimentation
for model_name, estimator in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_grid,
        n_iter=4,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_

    predictions = tuned_model.predict(X_test)
    probabilities = tuned_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "best_cv_roc_auc": round(search.best_score_, 4),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(
            y_test, predictions, zero_division=0
        ), 4),
        "recall": round(recall_score(
            y_test, predictions, zero_division=0
        ), 4),
        "f1_score": round(f1_score(
            y_test, predictions, zero_division=0
        ), 4),
        "roc_auc": round(roc_auc_score(
            y_test, probabilities
        ), 4),
        "best_parameters": str(search.best_params_)
    }

    results.append(metrics)

    print("Best parameters:", search.best_params_)
    print("Evaluation:", metrics)

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = tuned_model
        best_name = model_name
        best_params = search.best_params_

# Save experiment results
results_df = pd.DataFrame(results)
results_df.to_csv(
    f"{ARTIFACT_DIR}/experiment_results.csv",
    index=False
)

# Save the best model
joblib.dump(
    best_model,
    f"{ARTIFACT_DIR}/best_model.pkl"
)

# Save model metadata
metadata = {
    "best_model": best_name,
    "best_cv_roc_auc": float(best_score),
    "best_parameters": best_params,
    "features": X_train.columns.tolist()
}

with open(f"{ARTIFACT_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4, default=str)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("Best model:", best_name)
print("Best cross-validation ROC-AUC:", round(best_score, 4))
print("\nExperiment comparison:")
print(results_df.to_string(index=False))
print("\nBest model saved successfully!")
print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    zero_division=0
))
