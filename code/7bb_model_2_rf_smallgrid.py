'''
Small grid search for model_2 random forest
around the original best_within_one values
'''

# %% Packages
import json
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.ensemble import RandomForestClassifier

# %% Model info
model_number = "model_2"
model_type = "rf"

# %% Read original best_within_one
with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    original_best = json.load(f)

# %% Small grid around best_within_one
param_grid = {
    "max_depth": [12, int(float(original_best["max_depth"]))],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
    "max_features": [None, "sqrt"]
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

results = []

# %% All combinations
all_combos = list(itertools.product(
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["min_samples_leaf"],
    param_grid["max_features"]
))

# %% Run small grid
for i, (max_depth, min_samples_split, min_samples_leaf, max_features) in enumerate(all_combos, start=1):
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
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
            "max_depth": int(max_depth),
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features
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