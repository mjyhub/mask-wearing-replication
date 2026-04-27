#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small grid search for Model 1 XGBoost.

This script uses the best-within-one-standard-error parameters from the
previous Optuna tuning stage as the centre of a small grid search.

Model 1: predicting face mask usage
Model type: XGBoost
"""

# %% Packages

import json
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


# %% Parameter setup

model_number = "model_1"
model_type = "xgboost"


# %% Read best-within-one parameters

with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    best_params = json.load(f)


# %% Extract centre parameters

center_learning_rate = float(best_params["learning_rate"])
center_max_depth = int(best_params["max_depth"])
center_subsample = float(best_params["subsample"])
center_colsample_bytree = float(best_params["colsample_bytree"])


# %% Define small grid search space

param_grid = {
    "learning_rate": [
        center_learning_rate * 0.75,
        center_learning_rate,
        center_learning_rate * 1.25
    ],
    "max_depth": [
        max(1, center_max_depth - 1),
        center_max_depth,
        center_max_depth + 1
    ],
    "subsample": [
        max(0.1, center_subsample - 0.05),
        center_subsample,
        min(1.0, center_subsample + 0.05)
    ],
    "colsample_bytree": [
        max(0.1, center_colsample_bytree - 0.05),
        center_colsample_bytree,
        min(1.0, center_colsample_bytree + 0.05)
    ]
}


# %% Cross-validation setting

n_splits = 5
seed = 20240627

cv = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1 / n_splits,
    random_state=seed
)


# %% Load training data

x = pd.read_csv(
    f"../data/X_train_{model_number}.csv",
    keep_default_na=False
)

y = pd.read_csv(
    f"../data/y_train_{model_number}.csv",
    keep_default_na=False
).values.ravel()


# %% Define XGBoost model

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=seed,
    n_jobs=-1
)


# %% Run small grid search

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(x, y)


# %% Print results

print(f"{model_type}-{model_number} small grid search results")
print("Best ROC-AUC:", round(grid_search.best_score_, 4))
print("Best parameters:")
print(grid_search.best_params_)


# %% Save best small grid search result

smallgrid_result = {
    "best_score": grid_search.best_score_,
    "best_params": grid_search.best_params_,
    "centre_params": {
        "learning_rate": center_learning_rate,
        "max_depth": center_max_depth,
        "subsample": center_subsample,
        "colsample_bytree": center_colsample_bytree
    },
    "param_grid": param_grid
}

with open(f"../results/{model_number}_{model_type}_smallgrid_best.json", "w") as f:
    json.dump(smallgrid_result, f, indent=4)

print()
print(f"Saved small grid search result to ../results/{model_number}_{model_type}_smallgrid_best.json")