#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small grid search for model_1a xgboost
around the original best_within_one values

Model 1a: predicting face mask usage before mandates
Model type: XGBoost
"""

# %% Packages
import json
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

# %% Model info
model_number = "model_1a"
model_type = "xgboost"

# %% Read original best_within_one
with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    original_best = json.load(f)

# %% Centre values
# Some best_within_one files only contain the tuned parameters.
# Missing parameters are fixed at standard XGBoost centre values.
centre_params = {
    "learning_rate": float(original_best.get("learning_rate", 0.03)),
    "max_depth": int(float(original_best.get("max_depth", 6))),
    "min_child_weight": float(original_best.get("min_child_weight", 1)),
    "subsample": float(original_best.get("subsample", 0.8)),
    "colsample_bytree": float(original_best.get("colsample_bytree", 1.0)),
    "gamma": float(original_best.get("gamma", 0))
}

# %% Small grid around best_within_one
# Model 1a uses 9 parameter combinations:
# 3 learning_rate values x 3 subsample values.
param_grid = {
    "learning_rate": [
        0.02,
        centre_params["learning_rate"],
        0.03
    ],
    "subsample": [
        0.80,
        centre_params["subsample"],
        0.90
    ]
}

# %% Load data
x = pd.read_csv(
    f"../data/X_train_{model_number}.csv",
    keep_default_na=False
)

y = pd.read_csv(
    f"../data/y_train_{model_number}.csv",
    keep_default_na=False
).values.ravel()

# %% CV settings
seed = 20240627
n_splits = 5

cv = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1 / n_splits,
    random_state=seed
)

scale_pos_weight = sum(1 - y) / sum(y)

results = []

# %% All combinations
all_combos = list(itertools.product(
    param_grid["learning_rate"],
    param_grid["subsample"]
))

# %% Run small grid
for i, (learning_rate, subsample) in enumerate(all_combos, start=1):

    clf = xgb.XGBClassifier(
        learning_rate=float(learning_rate),
        max_depth=int(centre_params["max_depth"]),
        min_child_weight=centre_params["min_child_weight"],
        subsample=float(subsample),
        colsample_bytree=centre_params["colsample_bytree"],
        gamma=centre_params["gamma"],
        n_estimators=250,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )

    score = cross_validate(
        clf,
        x,
        y,
        cv=cv,
        scoring=["roc_auc", "accuracy", "recall"]
    )

    mean_roc = score["test_roc_auc"].mean()
    std_err = np.std(score["test_roc_auc"]) / np.sqrt(n_splits)

    result = {
        "number": i,
        "value": float(mean_roc),
        "params": {
            "learning_rate": float(learning_rate),
            "max_depth": int(centre_params["max_depth"]),
            "min_child_weight": float(centre_params["min_child_weight"]),
            "subsample": float(subsample),
            "colsample_bytree": float(centre_params["colsample_bytree"]),
            "gamma": float(centre_params["gamma"])
        },
        "user_attrs": {
            "std_err": float(std_err),
            "mean_accuracy": float(score["test_accuracy"].mean()),
            "mean_recall": float(score["test_recall"].mean())
        }
    }

    results.append(result)
    print(f"Trial {i}: ROC-AUC={mean_roc:.4f}, params={result['params']}")

# %% Save all trials
with open(f"../results/{model_number}_{model_type}_smallgrid_trials.json", "w") as f:
    json.dump(results, f, indent=2)

# %% Save best trial
best_trial = max(results, key=lambda item: item["value"])

with open(f"../results/{model_number}_{model_type}_smallgrid_trial_best.json", "w") as f:
    json.dump(best_trial, f, indent=2)

print("\nBest trial:")
print(best_trial)
