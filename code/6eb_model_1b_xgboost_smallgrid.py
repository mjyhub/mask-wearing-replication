'''
Small grid search for model_1b xgboost
around the original best_within_one values
'''

# %% Packages
import json
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

# %% Model info
model_number = "model_1b"
model_type = "xgboost"

# %% Read original best_within_one
with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    original_best = json.load(f)

# %% Small grid around best_within_one
param_grid = {
    "learning_rate": [0.01, float(original_best["learning_rate"])],
    "max_depth": [8, int(float(original_best["max_depth"]))],
    "min_child_weight": [1, int(float(original_best["min_child_weight"]))],
    "subsample": [0.75, float(original_best["subsample"])],
    "colsample_bytree": [0.50, float(original_best["colsample_bytree"])]
}

# %% Load data
x = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()

# %% CV settings
seed = 20240627
n_splits = 5
cv = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1/n_splits,
    random_state=seed
)

scale_pos_weight = sum(1 - y) / sum(y)

results = []

# %% All combinations
all_combos = list(itertools.product(
    param_grid["learning_rate"],
    param_grid["max_depth"],
    param_grid["min_child_weight"],
    param_grid["subsample"],
    param_grid["colsample_bytree"]
))

# %% Run small grid
for i, (learning_rate, max_depth, min_child_weight, subsample, colsample_bytree) in enumerate(all_combos, start=1):
    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=250,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed
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
            "max_depth": int(max_depth),
            "min_child_weight": int(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree)
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
best_trial = max(results, key=lambda x: x["value"])

with open(f"../results/{model_number}_{model_type}_smallgrid_trial_best.json", "w") as f:
    json.dump(best_trial, f, indent=2)

print("\nBest trial:")
print(best_trial)