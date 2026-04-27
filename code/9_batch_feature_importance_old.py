#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch feature importance for all model combinations
Adapted to current project naming:
- Prefer reading *_smallgrid_trial_best.json
- Fallback to *_best_within_one.json
- Save one CSV per model combination

Author:
    Adapted for current project
"""

# %% libraries
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime

start_time = datetime.now()

# %% script parameters
model_numbers = ["model_1", "model_2", "model_1a", "model_2a", "model_1b", "model_2b"]
model_types = ["xgboost", "rf"]

# 如果你想先只测试几个，可以改这里
run_all = True

# %% helper functions
def load_train_data(model_number):
    X = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()
    return X, y


def upsample_data(X, y):
    upsampler = RandomOverSampler(random_state=20240627)
    X_up, y_up = upsampler.fit_resample(X, y)
    return X_up, y_up


def read_best_params(model_number, model_type):
    """
    优先读取 smallgrid_trial_best.json
    如果没有，再读取 best_within_one.json
    """
    smallgrid_path = f"../results/{model_number}_{model_type}_smallgrid_trial_best.json"
    bestwithin_path = f"../results/{model_number}_{model_type}_best_within_one.json"

    source = None
    data = None

    if os.path.exists(smallgrid_path):
        with open(smallgrid_path, "r") as f:
            data = json.load(f)
        source = "smallgrid_trial_best"
    elif os.path.exists(bestwithin_path):
        with open(bestwithin_path, "r") as f:
            data = json.load(f)
        source = "best_within_one"
    else:
        raise FileNotFoundError(
            f"No parameter file found for {model_number}-{model_type}. "
            f"Tried:\n{smallgrid_path}\n{bestwithin_path}"
        )

    return data, source


def normalize_params(model_number, model_type, raw_data):
    """
    把不同格式的 json 参数统一整理成可用于建模的 params
    """
    # 有些文件是 {"params": {...}} 结构
    if "params" in raw_data:
        params = raw_data["params"].copy()
    else:
        params = raw_data.copy()

    # 删除非模型参数字段
    for k in ["number", "value", "std_err", "mean_accuracy", "mean_recall"]:
        if k in params:
            del params[k]

    # 有些文件 user_attrs 在外层，不需要
    if "user_attrs" in params:
        del params["user_attrs"]

    # 固定参数
    params["n_estimators"] = 250
    params["random_state"] = 20240627

    # 训练数据，用于 scale_pos_weight
    X, y = load_train_data(model_number)

    if model_type == "xgboost":
        scale_pos_weight = sum(1 - y) / sum(y)
        params["scale_pos_weight"] = scale_pos_weight
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

    if model_type == "rf":
        params["bootstrap"] = True
        params["n_jobs"] = -1

    # 把应该转成 int 的参数转一下
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
        model = XGBClassifier(**params)
    elif model_type == "rf":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


def fit_model_with_upsample(model_number, model_type):
    raw_data, source = read_best_params(model_number, model_type)
    params = normalize_params(model_number, model_type, raw_data)

    X, y = load_train_data(model_number)
    X_up, y_up = upsample_data(X, y)

    model = build_model(model_type, params)
    model.fit(X_up, y_up)

    return model, X.columns, source, params


def save_feature_importance_csv(model_number, model_type):
    model, feature_names, source, params = fit_model_with_upsample(model_number, model_type)

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    out_path = f"../results/{model_number}_{model_type}_feature_importance.csv"
    feature_importance.to_csv(out_path, index=False)

    print(f"[DONE] {model_number}-{model_type} | source={source}")
    print(f"Saved: {out_path}")
    print(feature_importance.head(10))
    print("-" * 60)

    return {
        "model_number": model_number,
        "model_type": model_type,
        "source": source,
        "output_file": out_path
    }


# %% main
if __name__ == "__main__":
    summary = []

    combos = [(mn, mt) for mn in model_numbers for mt in model_types]

    for model_number, model_type in combos:
        try:
            result = save_feature_importance_csv(model_number, model_type)
            summary.append(result)
        except Exception as e:
            print(f"[SKIP] {model_number}-{model_type}")
            print(f"Reason: {e}")
            print("-" * 60)

    summary_df = pd.DataFrame(summary)
    summary_path = "../results/feature_importance_batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nAll finished.")
    print(f"Summary saved to: {summary_path}")
    print(f"Time taken: {datetime.now() - start_time}")