#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate final fitted models on the held-out test sets
Adapted for current project naming
"""

# %%
import pickle
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score
)

# %%
model_numbers = ["model_1", "model_2", "model_1a",
                 "model_2a", "model_1b", "model_2b"]

model_types = ["xgboost", "rf"]

# %%
res_list = []

for model_number in model_numbers:
    x = pd.read_csv(
        f"../data/X_test_{model_number}.csv",
        keep_default_na=False
    )
    y = pd.read_csv(
        f"../data/y_test_{model_number}.csv",
        keep_default_na=False
    ).values.ravel()

    for model_type in model_types:
        model_path = f"../models/{model_number}_{model_type}.pkl"

        with open(model_path, "rb") as f:
            M = pickle.load(f)

        preds = M.predict(x)
        prob_preds = M.predict_proba(x)

        test_metrics = {
            "model_number": model_number,
            "model_type": model_type,
            "test_precision": precision_score(y_true=y, y_pred=preds),
            "test_recall": recall_score(y_true=y, y_pred=preds),
            "test_roc_auc": roc_auc_score(y_true=y, y_score=prob_preds[:, 1]),
            "test_accuracy": accuracy_score(y_true=y, y_pred=preds),
            "test_f1": f1_score(y_true=y, y_pred=preds)
        }

        res_list.append(test_metrics)

final_df = pd.DataFrame(res_list)

final_df.to_csv("../results/final_model_test_metrics.csv", index=False)

print(final_df)
print("\nSaved to ../results/final_model_test_metrics.csv")