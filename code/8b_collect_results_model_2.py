#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect the cross validation results for model_2
Adapted for files saved as *_final_cv.pkl
"""

# %%
import pickle
import numpy as np
import pandas as pd

# %%
file_count = "02"
model_number = "model_2"

# 手动指定每个模型对应的 pkl 文件
model_files = {
    "logistic_reg": f"../results/{model_number}_logistic_reg.pkl",
    "binary_tree": f"../results/{model_number}_binary_tree_final_cv.pkl",
    "xgboost": f"../results/{model_number}_xgboost_final_cv.pkl",
    "rf": f"../results/{model_number}_rf_final_cv.pkl"
}

# %%
final_df = pd.DataFrame()

for idx, (model_type, file_path) in enumerate(model_files.items()):

    with open(file_path, "rb") as f:
        M = pickle.load(f)

    model_dict = {
        "model_number": model_number,
        "model_type": model_type,
        "precision": M["test_precision"].mean(),
        "precision_std": M["test_precision"].std() / np.sqrt(M["test_precision"].size),
        "recall": M["test_recall"].mean(),
        "recall_std": M["test_recall"].std() / np.sqrt(M["test_recall"].size),
        "roc_auc": M["test_roc_auc"].mean(),
        "roc_auc_std": M["test_roc_auc"].std() / np.sqrt(M["test_roc_auc"].size),
        "accuracy": M["test_accuracy"].mean(),
        "accuracy_std": M["test_accuracy"].std() / np.sqrt(M["test_accuracy"].size),
        "f1": M["test_f1"].mean(),
        "f1_std": M["test_f1"].std() / np.sqrt(M["test_f1"].size)
    }

    model_df = pd.DataFrame(model_dict, index=[idx])
    final_df = pd.concat((final_df, model_df), ignore_index=True)

# %%
final_df.to_csv(f"../results/{file_count}_{model_number}_final_results.csv", index=False)

print(final_df)
print(f"\nSaved to ../results/{file_count}_{model_number}_final_results.csv")