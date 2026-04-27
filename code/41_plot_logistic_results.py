import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 结果文件夹
results_dir = "../results"

# 6个模型文件
model_files = [
    "model_1_logistic_reg.pkl",
    "model_1a_logistic_reg.pkl",
    "model_1b_logistic_reg.pkl",
    "model_2_logistic_reg.pkl",
    "model_2a_logistic_reg.pkl",
    "model_2b_logistic_reg.pkl",
]

rows = []

for file in model_files:
    path = os.path.join(results_dir, file)

    with open(path, "rb") as f:
        result = pickle.load(f)

    model_name = file.replace("_logistic_reg.pkl", "")

    rows.append({
        "model": model_name,
        "recall_mean": np.mean(result["test_recall"]),
        "recall_sd": np.std(result["test_recall"], ddof=1),
        "roc_auc_mean": np.mean(result["test_roc_auc"]),
        "roc_auc_sd": np.std(result["test_roc_auc"], ddof=1),
        "accuracy_mean": np.mean(result["test_accuracy"]),
        "accuracy_sd": np.std(result["test_accuracy"], ddof=1),
        "precision_mean": np.mean(result["test_precision"]),
        "precision_sd": np.std(result["test_precision"], ddof=1),
        "f1_mean": np.mean(result["test_f1"]),
        "f1_sd": np.std(result["test_f1"], ddof=1),
    })

df = pd.DataFrame(rows)
print(df)

# 保存汇总表
df.to_csv("../results/logistic_cv_summary.csv", index=False)

# -------- 画 Recall 图 --------
plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["recall_mean"], yerr=df["recall_sd"], capsize=4)
plt.title("Recall Comparison Across Logistic Models")
plt.ylabel("Recall")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_recall_comparison.png", dpi=300)
plt.show()

# -------- 画 ROC-AUC 图 --------
plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["roc_auc_mean"], yerr=df["roc_auc_sd"], capsize=4)
plt.title("ROC-AUC Comparison Across Logistic Models")
plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_roc_comparison.png", dpi=300)
plt.show()

# -------- 画 Accuracy 图 --------
plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["accuracy_mean"], yerr=df["accuracy_sd"], capsize=4)
plt.title("Accuracy Comparison Across Logistic Models")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_accuracy_comparison.png", dpi=300)
plt.show()