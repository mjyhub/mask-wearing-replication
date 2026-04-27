'''
Small grid search for model_2b binary tree
around the original best_within_one values
'''

# %% Packages
import json
import itertools
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

# %% Model info
model_number = "model_2b"
model_type = "binary_tree"

# %% Read original best_within_one
with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    original_best = json.load(f)

# %% Small grid around best_within_one
param_grid = {
    "min_weight_fraction_leaf": [0.0010, float(original_best["min_weight_fraction_leaf"]), 0.0020],
    "min_impurity_decrease": [0.00020, float(original_best["min_impurity_decrease"]), 0.00040]
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
    param_grid["min_weight_fraction_leaf"],
    param_grid["min_impurity_decrease"]
))

# %% Run small grid
for i, (min_weight_fraction_leaf, min_impurity_decrease) in enumerate(all_combos, start=1):
    clf = DecisionTreeClassifier(
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        min_impurity_decrease=min_impurity_decrease,
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
            "min_weight_fraction_leaf": float(min_weight_fraction_leaf),
            "min_impurity_decrease": float(min_impurity_decrease)
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