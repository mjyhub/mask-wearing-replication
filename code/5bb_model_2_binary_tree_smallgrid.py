'''
Small grid search for model_2 binary tree
around the original best_within_one values
'''

import json
import itertools
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

model_number = "model_2"
model_type = "binary_tree"

with open(f"../results/{model_number}_{model_type}_best_within_one.json", "r") as f:
    original_best = json.load(f)

param_grid = {
    "min_samples_leaf": [2, int(float(original_best["min_samples_leaf"]))],
    "min_weight_fraction_leaf": [0.0015, float(original_best["min_weight_fraction_leaf"])],
    "min_impurity_decrease": [0.00008, float(original_best["min_impurity_decrease"])]
}

x = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()

seed = 20240627
n_splits = 5
cv = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1/n_splits,
    random_state=seed
)

results = []

all_combos = list(itertools.product(
    param_grid["min_samples_leaf"],
    param_grid["min_weight_fraction_leaf"],
    param_grid["min_impurity_decrease"]
))

for i, (min_samples_leaf, min_weight_fraction_leaf, min_impurity_decrease) in enumerate(all_combos, start=1):
    clf = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf,
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
            "min_samples_leaf": int(min_samples_leaf),
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

with open(f"../results/{model_number}_{model_type}_smallgrid_trials.json", "w") as f:
    json.dump(results, f, indent=2)

best_trial = max(results, key=lambda x: x["value"])

with open(f"../results/{model_number}_{model_type}_smallgrid_trial_best.json", "w") as f:
    json.dump(best_trial, f, indent=2)

print("\nBest trial:")
print(best_trial)