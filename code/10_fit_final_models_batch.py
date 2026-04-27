#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit final XGBoost and Random Forest models on full training data
Adapted for current project naming:
- Prefer reading *_smallgrid_trial_best.json
- Fallback to *_best_within_one.json
- Save final fitted models to ../models/

Author:
    Adapted for current project
"""

# %% libraries
import os
import json
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

# %% configuration
model_numbers = ["model_1", "model_2", "model_1a", "model_2a", "model_1b", "model_2b"]
model_types = ["xgboost", "rf"]

# Ryan 原始逻辑：
# model_1, model_2 不 upsample
# model_1a, model_2a, model_1b, model_2b upsample
upsample_models = ["model_1a", "model_2a", "model_1b", "model_2b"]

# 确保 models 文件夹存在
os.makedirs("../models", exist_ok=True)

# %% helper functions
def load_train_data(model_number):
    X = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()
    return X, y


def read_best_params_file(model_number, model_type):
    """
    优先读取 smallgrid_trial_best.json
    如果没有，再读取 best_within_one.json
    """
    smallgrid_path = f"../results/{model_number}_{model_type}_smallgrid_trial_best.json"
    bestwithin_path = f"../results/{model_number}_{model_type}_best_within_one.json"

    if os.path.exists(smallgrid_path):
        with open(smallgrid_path, "r") as f:
            data = json.load(f)
        source = "smallgrid_trial_best"
        return data, source

    if os.path.exists(bestwithin_path):
        with open(bestwithin_path, "r") as f:
            data = json.load(f)
        source = "best_within_one"
        return data, source

    raise FileNotFoundError(
        f"No parameter file found for {model_number}-{model_type}.\n"
        f"Tried:\n{smallgrid_path}\n{bestwithin_path}"
    )


def normalize_params(model_number, model_type, raw_data):
    """
    把不同格式的 JSON 参数统一整理成建模参数
    """
    # smallgrid_trial_best.json 一般是 {"params": {...}, "user_attrs": {...}}
    # best_within_one.json 可能是直接平铺参数
    if "params" in raw_data:
        params = raw_data["params"].copy()
    else:
        params = raw_data.copy()

    # 删除非模型参数字段
    remove_keys = [
        "number", "value", "std_err",
        "mean_accuracy", "mean_recall",
        "user_attrs"
    ]
    for k in remove_keys:
        if k in params:
            del params[k]

    # 基本固定参数
    params["n_estimators"] = 250
    params["random_state"] = 20240627

    # 读取 y，用于 xgboost 的 scale_pos_weight
    X, y = load_train_data(model_number)

    if model_type == "xgboost":
        scale_pos_weight = sum(1 - y) / sum(y)
        params["scale_pos_weight"] = scale_pos_weight
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

    if model_type == "rf":
        params["bootstrap"] = True
        params["n_jobs"] = -1

    # 将应为整数的参数转成 int
    int_params = [
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "min_child_weight"
    ]
    for p in int_params:
        if p in params and params[p] is not None:
            try:
                params[p] = int(float(params[p]))
            except Exception:
                pass

    return params


def build_model(model_type, params):
    if model_type == "xgboost":
        return XGBClassifier(**params)
    elif model_type == "rf":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def fit_model_full_data(model_number, model_type, model):
    X, y = load_train_data(model_number)
    model_fitted = model.fit(X, y)

    out_path = f"../models/{model_number}_{model_type}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_fitted, f)

    return out_path


def fit_model_with_upsample(model_number, model_type, model):
    X, y = load_train_data(model_number)

    upsampler = RandomOverSampler(random_state=2024)
    X_up, y_up = upsampler.fit_resample(X, y)

    model_fitted = model.fit(X_up, y_up)

    out_path = f"../models/{model_number}_{model_type}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_fitted, f)

    return out_path


def fit_one_model(model_number, model_type):
    raw_data, source = read_best_params_file(model_number, model_type)
    params = normalize_params(model_number, model_type, raw_data)
    model = build_model(model_type, params)

    if model_number in upsample_models:
        out_path = fit_model_with_upsample(model_number, model_type, model)
        fit_mode = "upsample"
    else:
        out_path = fit_model_full_data(model_number, model_type, model)
        fit_mode = "full_data"

    print(f"[DONE] {model_number}-{model_type} | source={source} | mode={fit_mode}")
    print(f"Saved to: {out_path}")
    print(f"Params: {params}")
    print("-" * 70)


# %% main
if __name__ == "__main__":
    for model_number in model_numbers:
        for model_type in model_types:
            try:
                fit_one_model(model_number, model_type)
            except Exception as e:
                print(f"[SKIP] {model_number}-{model_type}")
                print(f"Reason: {e}")
                print("-" * 70)