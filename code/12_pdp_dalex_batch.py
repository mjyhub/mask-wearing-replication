#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch PDP results using DALEX
Adapted for current project naming:
- Read final models from ../models/
- Read feature importance from ../results/
- Automatically select top N features
- Save pdp results to ../results/
"""

# %%
import pickle
import pandas as pd
import dalex as dx

# %%
model_numbers = ["model_1a", "model_2a", "model_1b", "model_2b"]
model_types = ["rf", "xgboost"]

top_n = 5  # 取前5个最重要变量

# %%
def get_pdp_results(model_number, model_type, top_n=5):
    # 1. 读取最终模型
    with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. 读取训练数据
    X_train = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y_train = pd.read_csv(
        f"../data/y_train_{model_number}.csv", keep_default_na=False).values.ravel()

    # 3. 读取 feature importance，并取前 top_n 个变量
    fi = pd.read_csv(
        f"../results/{model_number}_{model_type}_feature_importance.csv"
    )
    top_features = fi["feature"].head(top_n).tolist()

    # 4. 建立 explainer
    explainer = dx.Explainer(
        model,
        X_train,
        y_train,
        label=f"{model_number}_{model_type}_explainer",
        verbose=False
    )

    # 5. 计算 PDP
    pd_features = explainer.model_profile(
        variables=top_features
    )

    # 6. 保存结果
    out_path = f"../results/{model_number}_{model_type}_pdp_results.csv"
    pd_features.result.to_csv(out_path, index=False)

    print(f"[DONE] {model_number}-{model_type}")
    print(f"Top features: {top_features}")
    print(f"Saved to: {out_path}")
    print("-" * 60)


# %%
if __name__ == "__main__":
    for model_number in model_numbers:
        for model_type in model_types:
            try:
                get_pdp_results(model_number, model_type, top_n=top_n)
            except Exception as e:
                print(f"[SKIP] {model_number}-{model_type}")
                print(f"Reason: {e}")
                print("-" * 60)