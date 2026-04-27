#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch repeated feature importance for all model combinations
Adapted to current project naming:
- Prefer reading *_smallgrid_trial_best.json
- Fallback to *_best_within_one.json
- Refit each model multiple times with random upsampling
- Save repeated feature importance and summary statistics

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

# 重复次数：建议先 50 或 100，不要一开始就 10000
num_perm = 50

# 如果你只想先跑几个模型，可以改这里
selected_combos = [
    ("model_1a", "rf"),
    ("model_1b", "rf"),
    ("model_2a", "rf"),
    ("model_2b", "rf"),
    ("model_1a", "xgboost"),
    ("model_1b", "xgboost"),
    ("model_2a", "xgboost"),
    ("model_2b", "xgboost"),
]
# 如果你想跑全部，就改成：
# selected_combos = [(mn, mt) for mn in model_numbers for mt in model_types]

# %% helper functions
def load_train_data(model_number):
    X = pd.read_csv(f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()
    return X, y


def upsample_data(X, y, seed):
    upsampler = RandomOverSampler(random_state=seed)
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
    if "params" in raw_data:
        params = raw_data["params"].copy()
    else:
        params = raw_data.copy()

    for k in ["number", "value", "std_err", "mean_accuracy", "mean_recall"]:
        if k in params:
            del params[k]

    if "user_attrs" in params:
        del params["user_attrs"]

    params["n_estimators"] = 250

    X, y = load_train_data(model_number)

    if model_type == "xgboost":
        scale_pos_weight = sum(1 - y) / sum(y)
        params["scale_pos_weight"] = scale_pos_weight
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

    if model_type == "rf":
        params["bootstrap"] = True
        params["n_jobs"] = -1

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


def build_model(model_type, params, seed):
    params = params.copy()
    params["random_state"] = seed

    if model_type == "xgboost":
        model = XGBClassifier(**params)
    elif model_type == "rf":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


def fit_model_with_upsample(model_number, model_type, seed):
    raw_data, source = read_best_params(model_number, model_type)
    params = normalize_params(model_number, model_type, raw_data)

    X, y = load_train_data(model_number)
    X_up, y_up = upsample_data(X, y, seed=seed)

    model = build_model(model_type, params, seed=seed)
    model.fit(X_up, y_up)

    return model, X.columns, source, params


def run_repeated_feature_importance(model_number, model_type, num_perm=100):
    all_runs = []

    for i in range(num_perm):
        seed = 20240627 + i
        model, feature_names, source, params = fit_model_with_upsample(
            model_number=model_number,
            model_type=model_type,
            seed=seed
        )

        importance_dict = {
            "run_id": i + 1,
            "seed": seed
        }

        for fname, imp in zip(feature_names, model.feature_importances_):
            importance_dict[fname] = imp

        all_runs.append(importance_dict)

    repeated_df = pd.DataFrame(all_runs)

    repeated_out = f"../results/{model_number}_{model_type}_feature_importance_repeated.csv"
    repeated_df.to_csv(repeated_out, index=False)

    # 汇总统计：median, q1, q3, iqr
    feature_cols = [c for c in repeated_df.columns if c not in ["run_id", "seed"]]

    summary_df = pd.DataFrame({
        "feature": feature_cols,
        "median_importance": repeated_df[feature_cols].median().values,
        "q1": repeated_df[feature_cols].quantile(0.25).values,
        "q3": repeated_df[feature_cols].quantile(0.75).values,
    })
    summary_df["iqr"] = summary_df["q3"] - summary_df["q1"]
    summary_df = summary_df.sort_values(by="median_importance", ascending=False)

    summary_out = f"../results/{model_number}_{model_type}_feature_importance_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    # 为了兼容你后面的 PDP 脚本，顺便保留一个旧格式 top importance 文件
    single_out = f"../results/{model_number}_{model_type}_feature_importance.csv"
    summary_df[["feature", "median_importance"]].rename(
        columns={"median_importance": "importance"}
    ).to_csv(single_out, index=False)

    print(f"[DONE] {model_number}-{model_type} | source={source}")
    print(f"Repeated saved: {repeated_out}")
    print(f"Summary saved:  {summary_out}")
    print(summary_df.head(10))
    print("-" * 60)

    return {
        "model_number": model_number,
        "model_type": model_type,
        "source": source,
        "num_perm": num_perm,
        "repeated_file": repeated_out,
        "summary_file": summary_out
    }


# %% main
if __name__ == "__main__":
    summary = []

    for model_number, model_type in selected_combos:
        try:
            result = run_repeated_feature_importance(
                model_number=model_number,
                model_type=model_type,
                num_perm=num_perm
            )
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