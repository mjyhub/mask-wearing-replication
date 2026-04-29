#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final cross-validation for Model 1 XGBoost.

This script reads the best parameters selected from the small grid search,
then evaluates the XGBoost model using 5-fold Stratified Shuffle Split
cross-validation.

Model 1: predicting face mask usage
Model type: XGBoost
"""

# %% Packages

import json
import pickle
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit


# %% Parameter setup

model_number = "model_1"
model_type = "xgboost"


# %% Read best parameters from small grid search

with open(f"../results/{model_number}_{model_type}_smallgrid_best.json", "r") as f:
    smallgrid_result = json.load(f)

best_params = smallgrid_result["best_params"]
centre_params = smallgrid_result["centre_params"]

params = {
    "learning_rate": float(best_params["learning_rate"]),
    "subsample": float(best_params["subsample"]),
    "max_depth": int(centre_params["max_depth"]),
    "colsample_bytree": float(centre_params["colsample_bytree"]),
    "n_estimators": 250
}


# %% Load training data

x = pd.read_csv(
    f"../data/X_train_{model_number}.csv",
    keep_default_na=False
)

y = pd.read_csv(
    f"../data/y_train_{model_number}.csv",
    keep_default_na=False
).values.ravel()


# %% Cross-validation setting

n_splits = 5
seed = 20240627

cv = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1 / n_splits,
    random_state=seed
)


# %% Define model and metrics

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=seed,
    n_jobs=-1,
    **params
)

metric_list = [
    "precision",
    "recall",
    "roc_auc",
    "accuracy",
    "f1"
]


# %% Run cross-validation

cv_scores = cross_validate(
    model,
    x,
    y,
    cv=cv,
    scoring=metric_list,
    return_train_score=False
)


# %% Print results

print(f"{model_type}-{model_number} final CV results")
print("Parameters used:")
print(params)
print()

print("Mean precision:", round(cv_scores["test_precision"].mean(), 3))
print("Mean recall:", round(cv_scores["test_recall"].mean(), 3))
print("Mean roc_auc:", round(cv_scores["test_roc_auc"].mean(), 3))
print("Mean accuracy:", round(cv_scores["test_accuracy"].mean(), 3))
print("Mean f1:", round(cv_scores["test_f1"].mean(), 3))


# %% Save results

with open(f"../results/{model_number}_{model_type}.pkl", "wb") as f:
    pickle.dump(cv_scores, f)

print()
print(f"Saved CV results to ../results/{model_number}_{model_type}.pkl")