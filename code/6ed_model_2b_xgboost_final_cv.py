'''
Final cross validation for model_2b xgboost
using small grid best parameters
'''

# %% Packages
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
import pickle

# %% Model info
model_number = "model_2b"
model_type = "xgboost"

# %% Read best parameters from small grid
with open(f"../results/{model_number}_{model_type}_smallgrid_trial_best.json", "r") as f:
    best_result = json.load(f)

params = best_result["params"]

# %% Load y first for scale_pos_weight
y_all = pd.read_csv(
    f"../data/y_train_{model_number}.csv",
    keep_default_na=False
).values.ravel()

scale_pos_weight = sum(1 - y_all) / sum(y_all)

# Add fixed parameters
params["n_estimators"] = 250
params["scale_pos_weight"] = scale_pos_weight
params["objective"] = "binary:logistic"
params["eval_metric"] = "logloss"
params["random_state"] = 20240627

# %% CV settings
n_splits = 5
seed = 20240627
kf = StratifiedShuffleSplit(
    n_splits=n_splits,
    test_size=1/n_splits,
    random_state=seed
)

metric_list = ['precision', 'recall', 'roc_auc', 'accuracy', 'f1']

model = xgb.XGBClassifier(**params)

# %%
def cross_validate_model(model_number):
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv",
        keep_default_na=False
    )
    y = pd.read_csv(
        f"../data/y_train_{model_number}.csv",
        keep_default_na=False
    ).values.ravel()

    cv_scores = cross_validate(
        model,
        x,
        y,
        cv=kf,
        scoring=metric_list
    )

    print(f"{model_type}-{model_number}")
    print("Mean recall: ", round(cv_scores["test_recall"].mean(), 3))
    print("Mean roc: ", round(cv_scores["test_roc_auc"].mean(), 3))
    print("Mean accuracy: ", round(cv_scores["test_accuracy"].mean(), 3))

    with open(f"../results/{model_number}_{model_type}_final_cv.pkl", "wb") as f:
        pickle.dump(cv_scores, f)

    result_summary = {
        "model_number": model_number,
        "model_type": model_type,
        "source": "smallgrid_trial_best",
        "selected_trial_number": best_result["number"],
        "selected_trial_value": best_result["value"],
        "selected_trial_std_err": best_result["user_attrs"]["std_err"],
        "params": params,
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