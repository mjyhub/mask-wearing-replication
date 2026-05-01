'''
Final cross validation for model_1a binary tree
using small grid best parameter

Author:
    jiayi ma
Date created:
    22/04/2026
'''

# %% Packages
import json
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

# %% Model info
model_number = "model_1a"
model_type = "binary_tree"

# %% Read best parameter from small grid
with open(f"../results/{model_number}_{model_type}_smallgrid_trial_best.json", "r") as f:
    best_result = json.load(f)

min_impurity_decrease = float(best_result["min_impurity_decrease"])

# %% CV settings
seed = 20240627
n_splits = 5
kf = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1/n_splits,
    random_state=seed
)

metric_list = ['precision', 'recall', 'roc_auc', 'accuracy', 'f1']

model = DecisionTreeClassifier(
    min_impurity_decrease=min_impurity_decrease,
    random_state=seed
)

# %%
def cross_validate_model(model_number):
    # Load data
    x = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()

    # Cross validate model
    cv_scores = cross_validate(model, x, y, cv=kf, scoring=metric_list)

    # Print results
    print(f"{model_type}-{model_number}")
    print("Mean recall: ", round(cv_scores["test_recall"].mean(), 3))
    print("Mean roc: ", round(cv_scores["test_roc_auc"].mean(), 3))
    print("Mean accuracy: ", round(cv_scores["test_accuracy"].mean(), 3))

    # Save PKL
    with open(f"../results/{model_number}_{model_type}_final_cv.pkl", "wb") as f:
        pickle.dump(cv_scores, f)

    # Save JSON summary
    result_summary = {
        "model_number": model_number,
        "model_type": model_type,
        "source": "smallgrid_trial_best",
        "selected_trial_number": best_result["number"],
        "selected_trial_value": best_result["value"],
        "selected_trial_std_err": best_result["std_err"],
        "min_impurity_decrease": min_impurity_decrease,
        "mean_precision": float(cv_scores["test_precision"].mean()),
        "mean_recall": float(cv_scores["test_recall"].mean()),
        "mean_roc_auc": float(cv_scores["test_roc_auc"].mean()),
        "mean_accuracy": float(cv_scores["test_accuracy"].mean()),
        "mean_f1": float(cv_scores["test_f1"].mean())
    }

    with open(f"../results/{model_number}_{model_type}_final_cv.json", "w") as f:
        json.dump(result_summary, f, indent=2)

# %%
cross_validate_model(model_number)